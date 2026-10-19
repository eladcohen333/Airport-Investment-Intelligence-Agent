"""The single tool-using analytics agent.

Responsibilities (brief section G): classify the question, resolve entities and dates, choose an approved tool, call it
with validated parameters, explain only the tool result, and preserve short-term context. The agent loop is backend-
neutral; numbers are checked against tool output before the answer is shown.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from src import scoring as sc
from src.analytics import CoverageInfo
from src.config import Settings, get_logger
from src.demo_router import HISTORICAL_HEADING, LIVE_HEADING, DemoAgent, build_turn_from_results
from src.llm_backends import BackendError, LLMBackend, build_backend
from src.models import AgentTurn, RunDefaults, SessionContext, ToolCall, ToolResult
from src.tools import ToolContext, anthropic_tool_definitions, execute_tool, gemini_tool_definitions

log = get_logger("agent")

MAX_ITERATIONS = 4
MAX_TOOL_CALLS = 6
RECENT_MESSAGES = 8
ANSWER_NOW = "Answer now using only the tool results above. Do not call more tools."
NO_GROUNDING_MESSAGE = ("I could not ground that answer in tool output, so I will not quote figures. "
                        "Which airport, state or region should I analyze, and over what period?")

SYSTEM_PROMPT = """You are the Airport Investment Intelligence Agent, a screening analyst for a U.S. airport-modernization investment firm.
You help analysts identify airports that merit further review because they show sustained demand growth, high scheduled-capacity
utilization, or operational pressure. You are a screening tool: you never calculate investment ROI, never forecast profitability,
and never claim that terminal renovation causes an operational problem.

HARD RULES
1. Every number you state must come from a tool result in this conversation (its `metrics` or `payload`), quoted as given (rounding is fine).
   Never compute new numbers (no differences, ratios, averages, or extrapolations) and never invent dates, sources or values.
2. If a tool returns status insufficient_data, ambiguous, unavailable or error, explain that status; never state a score for it.
3. Historical data (BTS) is the only basis for rankings and scores. Live data is current context only; it never changes a score.
4. "Congestion" means operational pressure from delays and cancellations, not proof of terminal congestion. More flights alone is not congestion.
5. Capacity pressure is a proxy; say that direct unmet demand would require search, booking, fare, leakage, or alternative-airport data.
6. Airport size is only a peer-comparison filter and never earns points.

HOW TO WORK
- Classify the question as historical, live, combined or methodology, then call exactly the approved tool(s) you need (usually one).
- Pass airport names through as written; tools resolve them. If a tool returns `ambiguous`, present the candidates and ask one short question. Do not guess.
- Pass `period: null` unless the user names a period. Use {{"year": N}} for a calendar year ("in 2024" means 2024 vs 2023) and
  {{"end_month": "YYYY-MM"}} for a 12-month window ending in that month.
- For follow-ups, reuse the Session context block below and change only what the user mentioned (e.g. "What about 2024?" keeps the same
  airports and tool and changes only the period). If a follow-up cannot be interpreted, ask one concise clarification question.
- Leave filters null to use the sidebar defaults unless the user names a region, state or size.

OUTPUT FORMAT (plain markdown, no tables - the application renders tables and charts from the tool results)
- First line: `**Conclusion:** ...` with the direct answer and the key figures.
- Then 2-6 short bullets of evidence drawn from the tool result (include raw values and component scores when present).
- When a live result is present, use exactly these two headings: `### {hist}` and `### {live}` and keep them separate.
- End with one line of caveats taken from the tool's `limitations`.

DEFINITIONS
{definitions}
"""


def build_system_prompt(defaults: RunDefaults, ctx: SessionContext, coverage: CoverageInfo) -> str:
    definitions = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in sc.DEFINITIONS.items()
                            if k in ("delayed_departure", "long_haul_flight", "new_england", "airport_congestion", "capacity_pressure", "airport_size"))
    prompt = SYSTEM_PROMPT.format(hist=HISTORICAL_HEADING, live=LIVE_HEADING, definitions=definitions)
    sidebar = {
        "data_scope": coverage.data_scope,
        "default_period_end_month": defaults.period_end_month or coverage.common_end_month,
        "coverage": {"traffic_through": coverage.traffic_last, "operations_through": coverage.ops_last, "earliest_valid_end_month": coverage.earliest_valid_end_month},
        "state_filter": defaults.states, "region_filter": defaults.region, "size_band_filter": defaults.size_bands,
        "long_haul_threshold_miles": defaults.long_haul_threshold_miles, "min_reported_flights": defaults.min_reported_flights,
        "live_context_enabled": defaults.live_enabled,
    }
    prompt += "\nSIDEBAR DEFAULTS (JSON)\n" + json.dumps(sidebar, default=str)
    prompt += "\n\nSESSION CONTEXT (JSON; reuse for follow-ups)\n" + json.dumps(ctx.compact(), default=str) + "\n"
    return prompt


# --------------------------------------------------------------------------------------
# Grounding guard
# --------------------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(million|thousand|k\b|%|percent)?", re.I)


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER_RE.finditer(text):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if "," not in raw and "." not in raw and 1990 <= value <= 2100:
            continue  # years
        if abs(value) <= 12 and "." not in raw and "," not in raw and suffix in ("",):
            continue  # ordinals, months, small structural counts
        if suffix == "million":
            value *= 1e6
        elif suffix in ("thousand", "k"):
            value *= 1e3
        out.append(value)
    return out


def _numbers_in(obj: Any, acc: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc.add(float(obj))
    elif isinstance(obj, str):
        for v in extract_numbers(obj):
            acc.add(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, acc)


def grounded_numbers(results: list[ToolResult]) -> set[float]:
    acc: set[float] = set()
    for r in results:
        _numbers_in(r.to_llm_dict(max_rows=50, include_series=False), acc)
        if isinstance(r.payload, object) and hasattr(r.payload, "model_dump"):
            _numbers_in(r.payload.model_dump(mode="json"), acc)
    expanded = set(acc)
    for v in acc:
        expanded.add(v * 100.0)
        expanded.add(v / 100.0)
    return expanded


def _round_sig(value: float, digits: int) -> float:
    if value == 0:
        return 0.0
    import math
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)


def _matches(x: float, candidates: set[float]) -> bool:
    """A quoted figure is grounded when it equals a tool value up to ordinary rounding (decimals or significant figures)."""
    for y in candidates:
        if abs(x - y) <= 0.051:
            return True
        for nd in (0, 1, 2):                      # 24.39 -> 24.4 or 24
            if abs(x - round(y, nd)) <= 1e-9:
                return True
        for sig in (2, 3, 4):                     # 21,437,809 -> 21.4 million; 190,106 -> 190,000
            if abs(x - _round_sig(y, sig)) <= 1e-9 * max(1.0, abs(y)):
                return True
    return False


def grounding_warnings(answer: str, results: list[ToolResult]) -> list[str]:
    nums = extract_numbers(answer)
    if not nums:
        return []
    pool = grounded_numbers(results)
    return [f"Figure {n:g} does not appear in any tool result." for n in nums if not _matches(n, pool)]


def extract_conclusion(text: str) -> str:
    m = re.search(r"\*\*Conclusion:?\*\*:?\s*(.+)", text)
    if m:
        return m.group(1).strip()
    for para in re.split(r"\n\s*\n", text.strip()):
        clean = para.strip().lstrip("#").strip()
        if clean:
            return clean.splitlines()[0][:400]
    return text.strip()[:400]


# --------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------

class AnalyticsAgent:
    def __init__(self, backend: LLMBackend, mode: str):
        self.backend = backend
        self.mode = mode  # "anthropic" | "gemini"

    def _tools(self, live_enabled: bool) -> list[dict[str, Any]]:
        return anthropic_tool_definitions(live_enabled) if self.mode == "anthropic" else gemini_tool_definitions(live_enabled)

    def run(self, question: str, ctx: SessionContext, defaults: RunDefaults, tool_ctx: ToolContext) -> AgentTurn:
        system = build_system_prompt(defaults, ctx, tool_ctx.coverage)
        live_ok = defaults.live_enabled and getattr(tool_ctx.live_provider, "is_configured", lambda: False)()
        tools = self._tools(live_ok or defaults.live_enabled)
        history: list[dict[str, Any]] = [{"role": m["role"], "text": m["content"]} for m in ctx.recent_messages[-RECENT_MESSAGES:] if m.get("content")]
        history.append({"role": "user", "text": question})
        calls: list[ToolCall] = []
        usage_total = {"input_tokens": 0, "output_tokens": 0}
        text = ""
        try:
            for _ in range(MAX_ITERATIONS):
                resp = self.backend.complete(system, history, tools)
                self._accumulate(usage_total, resp.usage)
                if resp.stop_reason == "refusal":
                    return self._error_turn("The model declined to answer this request.", calls, usage_total)
                if not resp.tool_calls:
                    text = resp.text
                    break
                history.append({"role": "assistant", "text": resp.text, "tool_calls": resp.tool_calls, "raw": resp.raw, "raw_backend": resp.raw_backend})
                results = []
                for tc in resp.tool_calls:
                    if len(calls) >= MAX_TOOL_CALLS:
                        result = ToolResult(status="error", tool_name=tc.name, data_scope=tool_ctx.data_scope, message="Tool budget for this turn is exhausted.")
                    else:
                        t0 = time.time()
                        result = execute_tool(tc.name, tc.input, tool_ctx)
                        calls.append(ToolCall(name=tc.name, input=dict(tc.input or {}), result=result, duration_ms=int((time.time() - t0) * 1000)))
                    results.append({"id": tc.id, "name": tc.name, "content": result.to_llm_json(), "is_error": result.status == "error"})
                history.append({"role": "tool_results", "results": results})
                if len(calls) >= MAX_TOOL_CALLS:
                    history.append({"role": "user", "text": "Tool budget reached. " + ANSWER_NOW})
                    tools = []
            else:
                history.append({"role": "user", "text": ANSWER_NOW})
                resp = self.backend.complete(system, history, [])
                self._accumulate(usage_total, resp.usage)
                text = resp.text
        except BackendError as exc:
            log.warning("backend error (%s): %s", exc.kind, exc)
            if calls:
                turn = build_turn_from_results(calls, mode=self.mode, usage=usage_total, model=self.backend.model)
                turn.grounding_warnings.append(f"The language model failed after the tools ran ({exc}); showing the deterministic summary instead.")
                return turn
            # the model never answered: route the question deterministically so the analyst still gets a grounded answer
            fallback = DemoAgent(today=tool_ctx.today).run(question, ctx, defaults, tool_ctx)
            fallback.usage = usage_total
            fallback.model = self.backend.model
            fallback.grounding_warnings.insert(0, f"The language model was unavailable ({exc}); this answer was produced by the deterministic router.")
            return fallback
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the UI
            log.exception("agent loop failed")
            if calls:
                turn = build_turn_from_results(calls, mode=self.mode, usage=usage_total, model=self.backend.model)
                turn.grounding_warnings.append("The language model step failed; showing the deterministic summary instead.")
                return turn
            return self._error_turn(f"The agent failed unexpectedly ({type(exc).__name__}); details were logged.", calls, usage_total)
        return self.finalize(text, calls, usage_total)

    @staticmethod
    def _accumulate(total: dict[str, int], usage: Optional[dict[str, Any]]) -> None:
        if not usage:
            return
        for k in ("input_tokens", "output_tokens"):
            v = usage.get(k)
            if isinstance(v, int):
                total[k] += v

    def _error_turn(self, message: str, calls: list[ToolCall], usage: dict[str, int]) -> AgentTurn:
        return AgentTurn(mode=self.mode, answer_kind="error", conclusion=message, answer_markdown=message, tool_calls=calls,
                         historical_calls=[i for i, c in enumerate(calls) if not c.result.is_live], live_calls=[i for i, c in enumerate(calls) if c.result.is_live],
                         usage=usage, model=self.backend.model)

    def finalize(self, text: str, calls: list[ToolCall], usage: dict[str, int]) -> AgentTurn:
        text = (text or "").strip()
        results = [c.result for c in calls]
        warnings: list[str] = []
        if not calls:
            if extract_numbers(text):
                log.warning("answer without tool results contained numbers; replaced with clarification. original=%r", text[:300])
                return AgentTurn(mode=self.mode, answer_kind="clarification", conclusion=NO_GROUNDING_MESSAGE, answer_markdown=NO_GROUNDING_MESSAGE,
                                 usage=usage, model=self.backend.model)
            kind = "clarification" if "?" in text or not text else "answer"
            if not text:
                text = NO_GROUNDING_MESSAGE
                kind = "clarification"
            return AgentTurn(mode=self.mode, answer_kind=kind, conclusion=extract_conclusion(text), answer_markdown=text, usage=usage, model=self.backend.model)
        if not text:
            turn = build_turn_from_results(calls, mode=self.mode, usage=usage, model=self.backend.model)
            turn.grounding_warnings.append("The model returned no text; showing the deterministic summary.")
            return turn
        warnings = grounding_warnings(text, results)
        statuses = {r.status for r in results}
        kind = "answer"
        if statuses <= {"ambiguous"}:
            kind = "clarification"
        elif statuses <= {"error"}:
            kind = "error"
        return AgentTurn(mode=self.mode, answer_kind=kind, conclusion=extract_conclusion(text), answer_markdown=text, tool_calls=calls,
                         historical_calls=[i for i, c in enumerate(calls) if not c.result.is_live], live_calls=[i for i, c in enumerate(calls) if c.result.is_live],
                         grounding_warnings=warnings, usage=usage, model=self.backend.model)


# --------------------------------------------------------------------------------------
# Session context update (shared by all modes)
# --------------------------------------------------------------------------------------

def update_context(ctx: SessionContext, question: str, turn: AgentTurn) -> SessionContext:
    new = ctx.model_copy(deep=True)
    new.turn_count += 1
    for call in turn.tool_calls:
        inp, res = call.input, call.result
        codes: list[str] = []
        payload = res.payload
        if res.status == "ok" and payload is not None:
            for attr in ("airport",):
                ref = getattr(payload, attr, None)
                if ref is not None and getattr(ref, "airport_code", None):
                    codes.append(ref.airport_code)
            rows = getattr(payload, "rows", None)
            if rows and call.name == "compare_airports":
                codes.extend([r["airport_code"] for r in rows if isinstance(r, dict)])
            if call.name == "get_live_operational_context" and getattr(payload, "airport_code", None):
                codes.append(payload.airport_code)
        for c in codes:
            if c in new.airport_codes:
                new.airport_codes.remove(c)
            new.airport_codes.insert(0, c)
        new.airport_codes = new.airport_codes[:6]
        if call.name == "compare_airports" and codes:
            new.comparison_set = list(dict.fromkeys(codes))
        if call.name == "rank_airports":
            new.region = inp.get("region") or new.region
            new.states = inp.get("states") or new.states
            new.size_bands = inp.get("size_bands") or new.size_bands
            new.selected_metric = inp.get("sort_by") or new.selected_metric
        if call.name == "analyze_route_mix" and inp.get("long_haul_threshold_miles"):
            new.long_haul_threshold_miles = int(inp["long_haul_threshold_miles"])
        if res.period is not None:
            new.period = res.period
        if not res.is_live:
            new.last_tool = call.name
            new.last_tool_input = dict(inp)
            summary = {"status": res.status, "period_label": res.period.label if res.period else None, "confidence": res.confidence}
            headline = {k: v for k, v in list(res.metrics.items())[:6] if not isinstance(v, (list, dict))}
            summary["metrics"] = headline
            if codes:
                summary["airports"] = codes
            new.last_result_summary = summary
    new.recent_messages.append({"role": "user", "content": question})
    new.recent_messages.append({"role": "assistant", "content": turn.answer_markdown})
    new.recent_messages = new.recent_messages[-10:]
    return new


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

@dataclass
class AgentBundle:
    agent: Any                 # AnalyticsAgent or DemoAgent
    mode: str                  # anthropic | gemini | demo
    model: Optional[str]


def build_agent(settings: Settings) -> AgentBundle:
    backend, provider = build_backend(settings)
    if backend is None:
        return AgentBundle(agent=DemoAgent(), mode="demo", model=None)
    return AgentBundle(agent=AnalyticsAgent(backend, provider), mode=provider, model=backend.model)


__all__ = ["AnalyticsAgent", "AgentBundle", "build_agent", "build_system_prompt", "update_context", "extract_numbers", "grounding_warnings",
           "extract_conclusion", "MAX_ITERATIONS", "MAX_TOOL_CALLS", "NO_GROUNDING_MESSAGE"]
