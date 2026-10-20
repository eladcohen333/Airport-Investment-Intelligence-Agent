"""Deterministic question routing and templated answers (Demo / deterministic mode).

Used when no LLM key is configured, and as the fallback renderer when an LLM call fails after tools ran.
Everything here is rule-based: parse -> route -> execute allow-listed tools -> templated conclusion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from src import analytics as an
from src import scoring as sc
from src.airport_resolver import describe_states, parse_geo, parse_size_words
from src.models import (
    AgentTurn, AirportProfile, CapacityPressureResult, ComparisonResult, Condition, LiveContext, Methodology, PeriodRequest, RankingResult,
    ResolveResult, RouteMixResult, RunDefaults, SessionContext, ToolCall, ToolResult,
)
from src.tools import ToolContext, execute_tool

SUGGESTED_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LAX and Santa Ana Airport, SNA, congestion levels.",
    "What percentage of departures from Anchorage, ANC, are long-haul?",
    "What is the capacity-pressure signal at SFO and why?",
]

HISTORICAL_HEADING = "Historical Investment Signal"
LIVE_HEADING = "Current Operational Context"


@dataclass
class ParsedQuestion:
    text: str
    intent: str = "unknown"            # methodology | route_mix | capacity_pressure | compare | rank | profile | live | unknown
    airports: list[ResolveResult] = field(default_factory=list)
    region: Optional[str] = None
    states: Optional[list[str]] = None
    size_bands: Optional[list[str]] = None
    period: Optional[PeriodRequest] = None
    period_notes: list[str] = field(default_factory=list)
    long_haul_threshold_miles: Optional[int] = None
    metric_hint: Optional[str] = None
    conditions: list[Condition] = field(default_factory=list)
    wants_live: bool = False
    is_follow_up: bool = False
    methodology_topic: str = "overview"

    @property
    def airport_codes(self) -> list[str]:
        return [r.airport.airport_code for r in self.airports if r.status == "resolved" and r.airport]

    @property
    def ambiguous(self) -> list[ResolveResult]:
        return [r for r in self.airports if r.status != "resolved"]


@dataclass
class RoutedCall:
    tool: str
    input: dict[str, Any]


@dataclass
class Clarification:
    question: str
    candidates: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

METRIC_HINTS: list[tuple[str, str]] = [
    (r"\b(demand growth|passenger growth|fastest[- ]growing|growing|growth)\b", "demand_growth"),
    (r"\b(utili[sz]ation|load factor|seat occupancy)\b", "capacity_utilization"),
    (r"\b(cancel\w*)\b", "cancellation_rate"),
    (r"\b(delay\w*|congest\w*|on[- ]time|operational pressure)\b", "operational_pressure"),
    (r"\b(expansion|candidates?|invest\w*|opportunit\w*|terminal)\b", "expansion_opportunity_score"),
]
METHODOLOGY_TOPICS: list[tuple[str, str]] = [
    (r"capacity[- ]pressure|unmet\s+(?:\w+\s+){0,2}demand", "capacity_pressure_proxy"),
    (r"expansion|opportunity score", "expansion_opportunity_score"),
    (r"demand growth", "demand_growth"),
    (r"utili[sz]ation", "capacity_utilization"),
    (r"operational pressure|delay|cancel", "operational_pressure"),
    (r"long[- ]haul", "long_haul"),
    (r"size|hub", "size_band"),
    (r"peer|normali[sz]", "peer_group_and_normalization"),
    (r"coverage|confiden", "coverage_and_confidence"),
    (r"source|data", "data_sources"),
]
FOLLOW_UP_RE = re.compile(r"^\s*(what about|how about|and\b|same for|now\b|also\b|instead|only\b|what if|again for)", re.I)
LIVE_RE = re.compile(r"\b(current(ly)?|right now|live|today|at the moment|real[- ]time|status|happening now)\b", re.I)
RANK_RE = re.compile(r"\b(which|what)\s+(airports?|ones?|hubs?)\b|\b(top|best|strongest|highest|most|rank\w*|candidates|list|screen)\b", re.I)
COMPARE_RE = re.compile(r"\b(compare|comparison|versus|vs\.?|against|between|differ\w*)\b", re.I)
PROFILE_RE = re.compile(r"\b(what changed|what has changed|how has .* changed|trend\w*|profile|overview|tell me about|how is .* doing|summary of|changes? at)\b", re.I)
ROUTE_RE = re.compile(r"\b(long[- ]haul|route mix|share of departures|percentage of departures|percent of departures|international share)\b", re.I)
# "unmet demand", "unmet flight demand", "unmet passenger demand" (the brief's wording) all mean the capacity-pressure proxy
CPP_RE = re.compile(r"\b(capacity[- ]pressure|unmet\s+(?:\w+\s+){0,2}demand|latent\s+(?:\w+\s+){0,2}demand|capacity signal|constrained|pressure signal)\b", re.I)
METHOD_RE = re.compile(r"\b(how (is|are|do you|does it|did you) .*(calculat|comput|defin|scor|measur|work)|methodology|formula|definition|what counts as|define\b|how do you define|what is the (score|proxy|definition))", re.I)
THRESHOLD_RE = re.compile(r"(\d{1,2}[,.]?\d{3}|\d{3,5})\s*(statute\s*)?(miles|mi\b|nm\b|nautical)", re.I)
WORSENING_CANCEL_RE = re.compile(r"\b(worsening|rising|increasing|more|growing|higher)\s+cancellation", re.I)
WORSENING_DELAY_RE = re.compile(r"\b(worsening|rising|increasing|more|growing|higher)\s+delays?", re.I)
GROWING_PAX_RE = re.compile(r"\b(growing|rising|increasing)\s+(passengers?|traffic|demand)", re.I)
DECLINING_RE = re.compile(r"\b(declin\w*|shrink\w*|falling|losing)\s+(passengers?|traffic|demand)", re.I)


def parse_question(text: str, tool_ctx: ToolContext, today: Optional[date] = None) -> ParsedQuestion:
    q = ParsedQuestion(text=text)
    spans: list[tuple[int, int]] = []
    today = today or tool_ctx.today

    req, p_spans, notes = an.parse_period_phrase(text, today)
    q.period, q.period_notes = req, notes
    spans.extend(p_spans)

    m = THRESHOLD_RE.search(text)
    if m:
        raw = float(m.group(1).replace(",", ""))
        unit = m.group(3).lower()
        if unit.startswith("n"):
            raw = raw * 1.1508
            q.period_notes.append("threshold converted from nautical to statute miles")
        q.long_haul_threshold_miles = int(round(raw))
        spans.append(m.span())

    bands, s_spans = parse_size_words(text)
    q.size_bands = bands
    spans.extend(s_spans)

    geo = parse_geo(text)
    q.region, q.states = geo.region, geo.states
    spans.extend(geo.spans)

    for pattern, metric in METRIC_HINTS:
        if re.search(pattern, text, re.I):
            q.metric_hint = metric
            break
    if GROWING_PAX_RE.search(text):
        q.conditions.append(Condition(metric="demand_growth", op=">", value=0))
        q.metric_hint = "demand_growth"
    if DECLINING_RE.search(text):
        q.conditions.append(Condition(metric="demand_growth", op="<", value=0))
    if WORSENING_CANCEL_RE.search(text):
        q.conditions.append(Condition(metric="cancellation_rate_change", op=">", value=0))
    if WORSENING_DELAY_RE.search(text):
        q.conditions.append(Condition(metric="delay_rate_change", op=">", value=0))

    q.airports = tool_ctx.resolver.extract_airports(text, spans)
    q.wants_live = bool(LIVE_RE.search(text))
    q.is_follow_up = bool(FOLLOW_UP_RE.search(text))

    if METHOD_RE.search(text):
        q.intent = "methodology"
        for pattern, topic in METHODOLOGY_TOPICS:
            if re.search(pattern, text, re.I):
                q.methodology_topic = topic
                break
    elif ROUTE_RE.search(text):
        q.intent = "route_mix"
    elif CPP_RE.search(text):
        q.intent = "capacity_pressure"
    elif COMPARE_RE.search(text) or (len(q.airport_codes) + len(q.ambiguous) >= 2 and not RANK_RE.search(text)):
        q.intent = "compare"
    elif RANK_RE.search(text) or ((q.region or q.states or q.size_bands) and not q.airports):
        q.intent = "rank"
    elif q.wants_live and q.airports and not PROFILE_RE.search(text):
        q.intent = "live"
    elif q.airports or PROFILE_RE.search(text):
        q.intent = "profile"
    elif q.is_follow_up or (not q.airports and (q.period or q.size_bands)):
        q.intent = "follow_up"
    else:
        q.intent = "unknown"
    return q


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------

def _period_dict(req: Optional[PeriodRequest]) -> Optional[dict[str, Any]]:
    return None if req is None or req.is_empty() else req.model_dump(exclude_none=True)


class DemoAgent:
    mode = "demo"

    def __init__(self, today: Optional[date] = None):
        self.today = today

    # ---- routing ----------------------------------------------------------------------
    def route(self, q: ParsedQuestion, ctx: SessionContext, defaults: RunDefaults) -> list[RoutedCall] | Clarification:
        if q.ambiguous:
            amb = q.ambiguous[0]
            cands = [c.display for c in amb.candidates]
            if amb.status == "ambiguous":
                return Clarification(f"'{amb.query}' matches several airports: {'; '.join(cands)}. Which one do you mean?", cands)
            return Clarification(amb.note or f"I could not find an airport matching '{amb.query}'.", cands)

        if q.intent == "methodology":
            return [RoutedCall("get_methodology", {"topic": q.methodology_topic})]

        codes = q.airport_codes
        period = _period_dict(q.period)
        if q.period_notes and "non_adjacent_years" in q.period_notes:
            return Clarification("Only adjacent 12-month windows are supported (for example 2024 vs 2023). Which year should be the latest window?")

        follow_up = q.intent == "follow_up" or (q.is_follow_up and ctx.last_tool and ctx.last_tool_input is not None)
        if follow_up:
            if not ctx.last_tool or ctx.last_tool_input is None:
                return Clarification("Which airport, state or region should I analyze, and over what period?")
            base = dict(ctx.last_tool_input)
            tool = ctx.last_tool
            if period:
                base["period"] = period
            if q.long_haul_threshold_miles and "long_haul_threshold_miles" in base:
                base["long_haul_threshold_miles"] = q.long_haul_threshold_miles
            if tool == "rank_airports":
                if q.region or q.states:
                    base["region"], base["states"] = q.region, q.states
                if q.size_bands:
                    base["size_bands"] = q.size_bands
                if q.metric_hint and q.metric_hint in ("demand_growth", "capacity_utilization", "operational_pressure", "cancellation_rate", "expansion_opportunity_score"):
                    base["sort_by"] = q.metric_hint
                if q.conditions:
                    base["conditions"] = [c.model_dump() for c in q.conditions]
                if len(codes) == 1:
                    return [RoutedCall("get_airport_profile", {"airport": codes[0], "period": base.get("period")})]
            elif tool == "compare_airports":
                if len(codes) >= 2:
                    base["airports"] = codes
                elif len(codes) == 1:
                    current = ", ".join(base.get("airports", []))
                    return Clarification(f"Add {codes[0]} to the comparison with {current}, or compare {codes[0]} against one of them only?")
            elif codes:
                base["airport"] = codes[0]
            if not period and not codes and not q.region and not q.states and not q.size_bands and not q.long_haul_threshold_miles and not q.metric_hint:
                return Clarification("What should change compared with the previous question: the airport, the period, or the filters?")
            calls = [RoutedCall(tool, base)]
            if q.wants_live and codes:
                calls.append(RoutedCall("get_live_operational_context", {"airport": codes[0]}))
            return calls

        if q.intent == "route_mix":
            if len(codes) != 1:
                return Clarification("Which single airport should I analyze for long-haul share?")
            inp: dict[str, Any] = {"airport": codes[0], "period": period, "long_haul_threshold_miles": q.long_haul_threshold_miles}
            return [RoutedCall("analyze_route_mix", inp)]

        if q.intent == "capacity_pressure":
            if len(codes) != 1:
                return Clarification("Which single airport should I assess for capacity pressure?")
            calls = [RoutedCall("analyze_capacity_pressure", {"airport": codes[0], "period": period, "same_size_band_peers": True})]
            if q.wants_live:
                calls.append(RoutedCall("get_live_operational_context", {"airport": codes[0]}))
            return calls

        if q.intent == "compare":
            if len(codes) < 2:
                if len(codes) == 1:
                    return Clarification(f"Compare {codes[0]} with which other airport?")
                return Clarification("Which airports should I compare? Name at least two.")
            metrics = None
            if q.metric_hint in ("operational_pressure", "cancellation_rate"):
                metrics = ["total_flights", "delayed_flights", "delay_rate", "avg_departure_delay_minutes", "cancellation_rate", "passengers", "capacity_utilization", "scores"]
            return [RoutedCall("compare_airports", {"airports": codes, "metrics": metrics, "period": period})]

        if q.intent == "rank":
            sort_by = q.metric_hint if q.metric_hint in ("demand_growth", "capacity_utilization", "operational_pressure", "cancellation_rate", "expansion_opportunity_score") else "expansion_opportunity_score"
            if q.metric_hint == "delay_rate":
                sort_by = "delay_rate"
            inp = {"region": q.region, "states": q.states, "size_bands": q.size_bands, "sort_by": sort_by,
                   "conditions": [c.model_dump() for c in q.conditions] or None, "top_n": 10, "ascending": False, "period": period}
            return [RoutedCall("rank_airports", inp)]

        if q.intent in ("profile", "live"):
            if not codes:
                return Clarification("Which single airport should I look at? I could not find an airport name or code in the question.")
            if len(codes) != 1:
                return Clarification("Which single airport should I look at?")
            calls: list[RoutedCall] = []
            if q.intent == "profile" or (q.wants_live and PROFILE_RE.search(q.text)):
                calls.append(RoutedCall("get_airport_profile", {"airport": codes[0], "period": period}))
            if q.wants_live or defaults.live_enabled and q.intent == "live":
                calls.append(RoutedCall("get_live_operational_context", {"airport": codes[0]}))
            if not calls:
                calls.append(RoutedCall("get_airport_profile", {"airport": codes[0], "period": period}))
            return calls

        return Clarification(
            "I can rank airports by region, state or size; compare airports; analyze long-haul share or capacity pressure; profile what changed at an "
            "airport; show live status; or explain the methodology. Try one of the suggested questions."
        )

    # ---- run --------------------------------------------------------------------------
    def run(self, question: str, ctx: SessionContext, defaults: RunDefaults, tool_ctx: ToolContext) -> AgentTurn:
        parsed = parse_question(question, tool_ctx, self.today)
        routed = self.route(parsed, ctx, defaults)
        if isinstance(routed, Clarification):
            kind = "unsupported" if parsed.intent == "unknown" and not parsed.ambiguous and not routed.candidates else "clarification"
            return AgentTurn(mode="demo", answer_kind=kind, conclusion=routed.question, answer_markdown=routed.question, tool_calls=[])
        calls: list[ToolCall] = []
        for rc in routed:
            result = execute_tool(rc.tool, rc.input, tool_ctx)
            calls.append(ToolCall(name=rc.tool, input=rc.input, result=result))
        return build_turn_from_results(calls, mode="demo")


# --------------------------------------------------------------------------------------
# Templated rendering (shared with the LLM fallback)
# --------------------------------------------------------------------------------------

def _pct(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def _num(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:,.{d}f}"


def _int(v: Optional[int]) -> str:
    return "n/a" if v is None else f"{int(v):,}"


def render_result(result: ToolResult) -> tuple[str, str]:
    """Return (conclusion, markdown body) for one tool result."""
    status, p = result.status, result.payload
    period = result.period.label if result.period else "the available period"
    if status == "ambiguous":
        cands = "; ".join(c.display for c in (p.candidates if isinstance(p, ResolveResult) else [])) or "see candidates"
        return result.message or "Ambiguous airport.", f"{result.message}\n\nCandidates: {cands}"
    if status == "error":
        return result.message or "The request could not be completed.", result.message or ""
    if status == "unavailable":
        msg = result.message or "Unavailable."
        return msg, f"{msg}\n\nThe historical analysis is unaffected."
    if status == "insufficient_data":
        msg = result.message or "Not enough historical coverage; no score is reported."
        body = msg
        if isinstance(p, RankingResult) and p.exclusions:
            body += "\n\nExcluded for coverage: " + ", ".join(f"{e.airport.airport_code} ({', '.join(e.reasons)})" for e in p.exclusions[:10])
        return msg, body

    if isinstance(p, RankingResult):
        top = p.rows[:3]
        by_score = p.sort_by == "expansion_opportunity_score"

        def _value(r) -> str:
            if by_score:
                return _num(r.scores.expansion_opportunity_score) if r.scores.expansion_opportunity_score is not None else "raw only"
            v = getattr(r.raw, p.sort_by, None)
            return _pct(v) if p.sort_by in ("demand_growth", "capacity_utilization", "delay_rate", "cancellation_rate") else _num(v, 3)

        parts = ", ".join(f"{r.airport.airport_code} ({_value(r)})" for r in top)
        metric = "Expansion Opportunity Score" if by_score else p.sort_by.replace("_", " ")
        conclusion = (f"Within {p.peer_group.describe()} over {period}, {p.peer_group_size} airports met the coverage rule; "
                      f"the strongest {metric} signals are {parts}.")
        if p.exclusions:
            conclusion += f" {len(p.exclusions)} airports were excluded for insufficient coverage."
        bullets = [f"- #{r.rank} {r.airport.airport_code} {r.airport.airport_name}: score {_num(r.scores.expansion_opportunity_score)}; growth {_pct(r.raw.demand_growth)}, "
                   f"utilization {_pct(r.raw.capacity_utilization)}, delay rate {_pct(r.raw.delay_rate)}, cancellation rate {_pct(r.raw.cancellation_rate, 2)}"
                   for r in p.rows[:5]]
        return conclusion, "\n".join(bullets)

    if isinstance(p, ComparisonResult):
        rows = {r["airport_code"]: r for r in p.rows}
        hi, most = p.higher_operational_pressure, p.most_flights
        if hi and len(rows) >= 2:
            others = [c for c in rows if c != hi]
            h = rows[hi]
            o = rows[others[0]]
            conclusion = (f"Over {period}, {hi} shows the higher operational pressure (delay rate {_pct(h['delay_rate'])} vs {_pct(o['delay_rate'])}, "
                          f"cancellation rate {_pct(h['cancellation_rate'], 2)} vs {_pct(o['cancellation_rate'], 2)}); ")
            conclusion += (f"{most} handled more reported flights ({_int(rows[most]['total_flights'])} vs {_int(rows[[c for c in rows if c != most][0]]['total_flights'])}), "
                           "which by itself does not indicate congestion.")
        else:
            conclusion = f"Comparison over {period}: see the table."
        bullets = [f"- {c}: {_int(r['total_flights'])} reported flights, delay rate {_pct(r['delay_rate'])}, avg delay {_num(r['avg_departure_delay_minutes'])} min, "
                   f"cancellation rate {_pct(r['cancellation_rate'], 2)}, passengers {_int(r['passengers'])}, utilization {_pct(r['capacity_utilization'])}"
                   + (f", expansion score {_num(r['expansion_opportunity_score'])}" if r.get("expansion_opportunity_score") is not None else "")
                   for c, r in rows.items()]
        return conclusion, "\n".join(bullets)

    if isinstance(p, RouteMixResult):
        conclusion = (f"{_pct(p.long_haul_share)} of departures from {p.airport.airport_code} ({_int(p.long_haul_departures)} of {_int(p.total_departures)}) "
                      f"were long-haul (at least {p.threshold_miles:,} statute miles) over the 12 months ending {result.period.end_month if result.period else ''}.")
        bullets = [f"- {r.destination_airport}: {r.distance_miles:,.0f} mi, {r.departures:,} departures" for r in p.top_long_haul_routes[:5]]
        return conclusion, ("Top long-haul routes:\n" + "\n".join(bullets)) if bullets else "No long-haul routes at this threshold."

    if isinstance(p, CapacityPressureResult):
        comps = {c.name: c for c in p.components}
        conclusion = (f"{p.airport.airport_code}'s Capacity Pressure Proxy is {p.proxy_score:.1f}/100 among {p.peer_group.describe()} (n={p.peer_group_size}) over {period}: "
                      f"utilization {_pct(p.raw.capacity_utilization)} (p{comps['capacity_utilization'].percentile:.0f}), demand growth {_pct(p.raw.demand_growth)} "
                      f"(p{comps['demand_growth'].percentile:.0f}), operational pressure {_num(p.raw.operational_pressure, 3)} (p{comps['operational_pressure'].percentile:.0f}). "
                      f"{sc.UNMET_DEMAND_DISCLAIMER}")
        bullets = [f"- {c.label}: raw {_num(c.raw_value, 4)}, percentile {c.percentile:.1f}, weight {c.weight:.0%}, contribution {c.contribution:.1f}" for c in p.components]
        if p.expansion and p.expansion.expansion_opportunity_score is not None:
            bullets.append(f"- Expansion Opportunity Score in the same peer group: {p.expansion.expansion_opportunity_score:.1f}")
        return conclusion, "\n".join(bullets)

    if isinstance(p, AirportProfile):
        ch = {c.metric: c for c in p.changes}
        def fmt(metric: str, pct: bool = True, d: int = 1) -> str:
            c = ch.get(metric)
            if c is None or c.latest is None or c.prior is None:
                return f"{metric.replace('_', ' ')} n/a"
            if pct:
                return f"{c.label.lower()} {_pct(c.prior, d)} to {_pct(c.latest, d)}"
            return f"{c.label.lower()} {_int(c.prior)} to {_int(c.latest)} ({'+' if c.pct_change and c.pct_change > 0 else ''}{_pct(c.pct_change)})"
        conclusion = (f"At {p.airport.airport_name} ({p.airport.airport_code}), {period}: {fmt('passengers', pct=False)}, {fmt('capacity_utilization')}, "
                      f"{fmt('delay_rate')}, {fmt('cancellation_rate', d=2)}.")
        bullets = [f"- {c.label}: {_num(c.prior, 3) if c.unit in ('ratio', 'index') else _int(c.prior) if c.unit in ('passengers', 'seats', 'departures', 'flights') else _num(c.prior)}"
                   f" to {_num(c.latest, 3) if c.unit in ('ratio', 'index') else _int(c.latest) if c.unit in ('passengers', 'seats', 'departures', 'flights') else _num(c.latest)}"
                   f" ({c.direction}{'' if c.pct_change is None else ', ' + ('+' if c.pct_change > 0 else '') + _pct(c.pct_change)})"
                   for c in p.changes if c.latest is not None and c.prior is not None]
        return conclusion, "\n".join(bullets)

    if isinstance(p, LiveContext):
        conclusion = (f"As of {p.retrieved_at}, {p.airport_code}: {p.flights_sampled} departures in the live sample, {p.delayed_15m} delayed 15+ minutes, "
                      f"{p.cancelled} cancelled, {p.active} active, average departure delay {_num(p.avg_departure_delay_minutes)} minutes. "
                      "Live context does not change the historical signal.")
        return conclusion, p.window_description

    if isinstance(p, Methodology):
        lines = [f"- {k}: {v}" for k, v in p.formulas.items()] + [f"- {r}" for r in p.rules]
        return f"{p.title}: " + (next(iter(p.formulas.values())) if p.formulas else (p.rules[0] if p.rules else "")), "\n".join(lines)

    return result.message or "Done.", ""


def build_turn_from_results(calls: list[ToolCall], mode: str, usage: Optional[dict] = None, model: Optional[str] = None) -> AgentTurn:
    historical = [i for i, c in enumerate(calls) if not c.result.is_live]
    live = [i for i, c in enumerate(calls) if c.result.is_live]
    sections: list[str] = []
    conclusions: list[str] = []
    for idx_list, heading in ((historical, HISTORICAL_HEADING), (live, LIVE_HEADING)):
        for i in idx_list:
            conclusion, body = render_result(calls[i].result)
            conclusions.append(conclusion)
            block = f"**Conclusion:** {conclusion}" + (f"\n\n{body}" if body else "")
            if live and historical:
                block = f"### {heading}\n\n" + block
            sections.append(block)
    if not calls:
        return AgentTurn(mode=mode, answer_kind="error", conclusion="No tool was executed.", answer_markdown="No tool was executed.")
    statuses = {c.result.status for c in calls}
    kind = "answer"
    if statuses <= {"error"}:
        kind = "error"
    elif statuses <= {"ambiguous"}:
        kind = "clarification"
    return AgentTurn(mode=mode, answer_kind=kind, conclusion=conclusions[0] if conclusions else "", answer_markdown="\n\n".join(sections),
                     tool_calls=calls, historical_calls=historical, live_calls=live, usage=usage, model=model)


__all__ = ["DemoAgent", "ParsedQuestion", "RoutedCall", "Clarification", "parse_question", "render_result", "build_turn_from_results",
           "SUGGESTED_QUESTIONS", "HISTORICAL_HEADING", "LIVE_HEADING"]
