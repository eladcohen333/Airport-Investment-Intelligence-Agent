"""Streamlit presentation layer: CSS, sidebar, answer rendering (tables, chart, methodology, limitations, evidence)."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from src.analytics import CoverageInfo
from src.charts import chart_for_result
from src.config import Settings
from src.demo_router import HISTORICAL_HEADING, LIVE_HEADING, SUGGESTED_QUESTIONS
from src.models import (
    NEW_ENGLAND_STATES, AgentTurn, AirportProfile, CapacityPressureResult, ComparisonResult, LiveContext, Methodology, RankingResult,
    ResolveResult, RouteMixResult, RunDefaults, ToolResult,
)
from src.airport_resolver import STATE_NAMES
from src.scoring import EXCLUSION_LABELS

CSS = """
<style>
:root { --ink: #1b2a41; --muted: #52514e; --accent: #1f5fbf; --card: #f4f6f9; --line: #e1e0d9; }
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px; }
h1 { font-size: 1.55rem; font-weight: 650; letter-spacing: -0.01em; margin-bottom: 0.2rem; }
h2, h3 { font-weight: 620; color: var(--ink); }
.scope-note { color: var(--muted); font-size: 0.92rem; margin-bottom: 0.9rem; }
.conclusion { font-size: 1.02rem; font-weight: 600; color: var(--ink); border-left: 3px solid var(--accent); padding: 0.35rem 0.75rem; background: var(--card); border-radius: 4px; margin: 0.3rem 0 0.8rem 0; }
.evidence { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 0.6rem 0.9rem; font-size: 0.84rem; color: var(--muted); line-height: 1.45; }
.evidence b { color: var(--ink); }
.section-head { font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0.9rem 0 0.2rem 0; }
.badge { display: inline-block; padding: 0.12rem 0.5rem; border-radius: 10px; font-size: 0.75rem; font-weight: 600; margin-left: 0.3rem; }
.badge-ok { background: #e3f1e3; color: #1d5c1d; } .badge-warn { background: #fdf1d6; color: #7a5200; } .badge-off { background: #e9eaee; color: #4a4f5a; }
.stButton > button { border-radius: 6px; border: 1px solid var(--line); background: #fff; color: var(--ink); font-size: 0.86rem; text-align: left; white-space: normal; height: auto; padding: 0.5rem 0.7rem; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
[data-testid="stSidebar"] { background: #fafbfc; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stMarkdown p { font-size: 0.86rem; }
footer, #MainMenu { visibility: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------

def pct(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def num(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:,.{d}f}"


def intf(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{int(round(v)):,}"


def _df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _show_df(df: pd.DataFrame, height: Optional[int] = None, key: Optional[str] = None) -> None:
    # Streamlit derives the auto element ID from the data itself, so two identical tables in one run
    # (e.g. the same question asked twice) raise StreamlitDuplicateElementId unless each has its own key.
    if df.empty:
        return
    try:
        st.dataframe(df, hide_index=True, width="stretch", height=height, key=key)
    except TypeError:  # older Streamlit
        st.dataframe(df, hide_index=True, use_container_width=True, height=height, key=key)


def _show_chart(fig, key: Optional[str] = None) -> None:
    # Same as _show_df: the auto ID hashes the figure spec, so identical charts need an explicit key.
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


# --------------------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------------------

def render_sidebar(settings: Settings, coverage: Optional[CoverageInfo], live_configured: bool) -> tuple[RunDefaults, bool]:
    """Return (RunDefaults, reset_requested)."""
    st.sidebar.markdown("### Analysis settings")
    scope = coverage.data_scope if coverage else "no data"
    refreshed = coverage.refresh_timestamp.replace("T", " ") if coverage and coverage.refresh_timestamp else "never"
    st.sidebar.markdown(f"**Data scope:** {scope}  \n**Latest historical refresh:** {refreshed}")
    if coverage and coverage.traffic_last:
        st.sidebar.caption(f"Traffic through {coverage.traffic_last}; operations through {coverage.ops_last}.")
    if coverage and coverage.source_status:
        bad = [s for s, status in coverage.source_status.items() if status != "ok"]
        if bad:
            st.sidebar.warning("Some sources failed in the last refresh: " + "; ".join(bad))

    options: list[tuple[str, Optional[str]]] = []
    if coverage and coverage.ready:
        options.append((f"Latest available (12 months ending {coverage.common_end_month})", None))
        for year in sorted(coverage.available_calendar_years(), reverse=True):
            options.append((f"Calendar {year} vs {year - 1}", f"{year}-12"))
        for m in sorted(coverage.available_end_months(), reverse=True):
            if not m.endswith("-12") or f"{m[:4]}-12" != m:
                options.append((f"12 months ending {m}", m))
    labels = [o[0] for o in options] or ["No historical data loaded"]
    choice = st.sidebar.selectbox("Analysis period", labels, index=0, help="Two adjacent 12-month windows are compared.")
    period_end = dict(options).get(choice) if options else None

    st.sidebar.markdown("**Peer-group filters**")
    region_preset = st.sidebar.checkbox("New England only (CT, ME, MA, NH, RI, VT)", value=False)
    state_choices = sorted(STATE_NAMES, key=lambda c: STATE_NAMES[c])
    states = st.sidebar.multiselect("States (optional)", state_choices, format_func=lambda c: f"{STATE_NAMES[c]} ({c})", disabled=region_preset)
    size_bands = st.sidebar.multiselect("Airport size band (peer filter only)", ["large", "medium", "small", "non_hub"],
                                        format_func=lambda b: {"large": "Large hub (>=10M pax/yr)", "medium": "Medium hub (2.5-10M)", "small": "Small hub (0.5-2.5M)", "non_hub": "Non-hub (<0.5M)"}[b])
    threshold = st.sidebar.number_input("Long-haul threshold (statute miles)", min_value=100, max_value=10000, value=settings.long_haul_threshold_miles, step=100)
    min_flights = st.sidebar.number_input("Minimum reported flights per 12-month window", min_value=0, max_value=100000, value=settings.min_reported_flights, step=100)

    st.sidebar.markdown("**Live operational context**")
    live_toggle = st.sidebar.toggle("Include live context", value=False, help="Current-status questions only; never affects historical scores.")
    if live_configured:
        st.sidebar.markdown('Aviationstack key: <span class="badge badge-ok">configured</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('Aviationstack key: <span class="badge badge-off">not set - live mode unavailable</span>', unsafe_allow_html=True)

    st.sidebar.markdown("**LLM mode**")
    label = settings.active_model_label
    badge = "badge-ok" if settings.resolved_provider != "demo" else "badge-warn"
    st.sidebar.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)
    if settings.resolved_provider == "demo":
        st.sidebar.caption("No LLM key found. Questions are routed deterministically; suggested questions and the supported patterns work fully.")
    reset = st.sidebar.button("Reset conversation")
    defaults = RunDefaults(
        data_scope=scope if scope in ("demo", "nationwide") else "demo",
        period_end_month=period_end,
        states=None if region_preset else (list(states) or None),
        region="new_england" if region_preset else None,
        size_bands=list(size_bands) or None,
        long_haul_threshold_miles=int(threshold),
        min_reported_flights=int(min_flights),
        live_enabled=bool(live_toggle),
    )
    return defaults, reset


# --------------------------------------------------------------------------------------
# header and suggested questions
# --------------------------------------------------------------------------------------

def render_header(coverage: Optional[CoverageInfo]) -> None:
    st.title("Airport Investment Intelligence Agent")
    scope_txt = ""
    if coverage and coverage.ready:
        scope_txt = (f" Historical data: BTS TranStats, {coverage.data_scope} scope, {coverage.traffic_first} to {coverage.common_end_month}"
                     f" (operations through {coverage.ops_last}).")
    st.markdown(
        '<div class="scope-note">Screening tool for U.S. commercial airports showing sustained demand growth, high scheduled-capacity utilization '
        'or operational pressure. It does not estimate ROI, forecast profitability, or attribute causes.' + scope_txt + "</div>",
        unsafe_allow_html=True,
    )


def render_suggested_questions() -> Optional[str]:
    st.markdown('<div class="section-head">Suggested questions</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    clicked = None
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i].button(q, key=f"suggested_{i}"):
            clicked = q
    return clicked


# --------------------------------------------------------------------------------------
# tables per tool
# --------------------------------------------------------------------------------------

def table_for_result(result: ToolResult) -> Optional[pd.DataFrame]:
    p = result.payload
    if isinstance(p, RankingResult):
        return _df([{
            "Rank": r.rank, "Airport": r.airport.airport_code, "Name": r.airport.airport_name, "State": r.airport.state, "Size band": r.airport.size_band or "",
            "Score": num(r.scores.expansion_opportunity_score), "Demand growth": pct(r.raw.demand_growth), "Utilization": pct(r.raw.capacity_utilization),
            "Delay rate": pct(r.raw.delay_rate), "Cancellation rate": pct(r.raw.cancellation_rate, 2), "Passengers (12m)": intf(r.raw.pax_latest),
            "Growth score": num(r.scores.demand_growth_score), "Utilization score": num(r.scores.capacity_utilization_score),
            "Pressure score": num(r.scores.operational_pressure_score), "Confidence": r.confidence,
        } for r in p.rows])
    if isinstance(p, ComparisonResult):
        cols = [c for c in p.columns]
        data = {"Measure": [c.label for c in cols]}
        for r in p.rows:
            vals = []
            for c in cols:
                v = r.get(c.key)
                if c.unit == "%":
                    vals.append(pct(v, 2 if c.key == "cancellation_rate" else 1))
                elif c.unit == "minutes":
                    vals.append(num(v))
                elif c.kind == "score":
                    vals.append(num(v))
                elif c.unit == "index":
                    vals.append(num(v, 4))
                else:
                    vals.append(intf(v))
            data[r["airport_code"]] = vals
        return pd.DataFrame(data)
    if isinstance(p, RouteMixResult):
        return _df([{"Destination": r.destination_airport, "Distance (mi)": f"{r.distance_miles:,.0f}", "Departures (12m)": f"{r.departures:,}",
                     "Passengers (12m)": f"{r.passengers:,}"} for r in p.top_long_haul_routes])
    if isinstance(p, CapacityPressureResult):
        return _df([{"Component": c.label, "Raw value": pct(c.raw_value) if c.name != "operational_pressure" else num(c.raw_value, 4),
                     "Percentile": f"{c.percentile:.1f}", "Weight": f"{c.weight:.0%}", "Contribution": f"{c.contribution:.1f}"} for c in p.components])
    if isinstance(p, AirportProfile):
        rows = []
        for c in p.changes:
            if c.latest is None and c.prior is None:
                continue
            fmt = (lambda v: pct(v, 2 if c.metric == "cancellation_rate" else 1)) if c.unit == "ratio" else (lambda v: num(v, 3)) if c.unit == "index" else (lambda v: num(v)) if c.unit == "minutes" else intf
            rows.append({"Metric": c.label, "Prior 12m": fmt(c.prior), "Latest 12m": fmt(c.latest), "Direction": c.direction,
                         "Change": ("n/a" if c.pct_change is None else f"{c.pct_change * 100:+.1f}%") if c.unit not in ("ratio",) else ("n/a" if c.abs_delta is None else f"{c.abs_delta * 100:+.2f} pp")})
        return _df(rows)
    if isinstance(p, LiveContext):
        return _df([{"Status": k, "Flights": v} for k, v in p.status_counts.items()] or [{"Status": "n/a", "Flights": 0}])
    if isinstance(p, Methodology):
        rows = [{"Item": k, "Definition / formula": v} for k, v in p.formulas.items()]
        for score, weights in p.weights.items():
            rows.append({"Item": f"{score} weights", "Definition / formula": ", ".join(f"{k} {v:.0%}" for k, v in weights.items())})
        return _df(rows)
    if isinstance(p, ResolveResult):
        return _df([{"Code": c.airport_code, "Airport": c.airport_name, "City": c.city, "State": c.state, "Size band": c.size_band or "",
                     "Passengers (latest 12m)": f"{c.latest_12m_passengers:,}"} for c in p.candidates])
    return None


def exclusions_table(p: RankingResult) -> pd.DataFrame:
    return _df([{"Airport": e.airport.airport_code, "Name": e.airport.airport_name, "Reason": "; ".join(EXCLUSION_LABELS.get(r, r) for r in e.reasons), "Detail": e.detail}
                for e in p.exclusions])


# --------------------------------------------------------------------------------------
# answer rendering
# --------------------------------------------------------------------------------------

def _evidence_block(result: ToolResult) -> None:
    defs = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in result.definitions.items()) or "n/a"
    lims = " | ".join(result.limitations) or "none stated"
    period = result.period.label if result.period else ("N/A (live)" if result.is_live else "N/A")
    live_at = result.refresh_timestamp if result.is_live else "N/A"
    hist_refresh = f" (refreshed {result.refresh_timestamp})" if (not result.is_live and result.refresh_timestamp) else ""
    st.markdown(
        '<div class="evidence">'
        f"<b>Sources:</b> {', '.join(result.sources) or 'n/a'}{hist_refresh}<br>"
        f"<b>Data scope:</b> {result.data_scope}<br>"
        f"<b>Historical period:</b> {period}<br>"
        f"<b>Live retrieved at:</b> {live_at}<br>"
        f"<b>Definitions:</b> {defs}<br>"
        f"<b>Confidence:</b> {result.confidence}<br>"
        f"<b>Limitations:</b> {lims}"
        "</div>",
        unsafe_allow_html=True,
    )


def _methodology_panel(result: ToolResult, key_prefix: str) -> None:
    p = result.payload
    with st.expander("How this was calculated"):
        st.markdown(f"**Tool:** `{result.tool_name}`  |  **Status:** {result.status}  |  **Period:** {result.period.label if result.period else 'n/a'}")
        if isinstance(p, RankingResult):
            st.markdown(f"**Peer group:** {p.peer_group.describe()} - {p.peer_group_size} airports passed the coverage rule "
                        f"(minimum {p.peer_group.min_reported_flights} reported flights in both 12-month windows).")
            st.markdown("**Formula:** Expansion Opportunity Score = 40% demand growth + 35% capacity utilization + 25% operational pressure, "
                        "each winsorized at p5/p95 within the peer group and min-max normalized to 0-100. Operational pressure = 70% delay rate + 30% cancellation rate.")
            bounds = p.normalization.get("bounds", {})
            if bounds:
                st.markdown("**Winsorization bounds (p5, p95):** " + "; ".join(f"{k}: {v}" for k, v in bounds.items()))
            if p.conditions:
                st.markdown("**Screening conditions:** " + ", ".join(f"{c.metric} {c.op} {c.value}" for c in p.conditions))
            if p.exclusions:
                st.markdown(f"**Excluded airports ({len(p.exclusions)}):**")
                _show_df(exclusions_table(p), height=min(300, 40 + 35 * len(p.exclusions)), key=f"{key_prefix}_exclusions")
        elif isinstance(p, CapacityPressureResult):
            st.markdown(f"**Peer group:** {p.peer_group.describe()} (n={p.peer_group_size}). Capacity Pressure Proxy = 50% utilization percentile + "
                        "30% demand-growth percentile + 20% operational-pressure percentile (Hazen percentile ranks within the peer group).")
            st.markdown(f"**Coverage:** latest/prior On-Time flights {p.raw.flights_latest:,}/{p.raw.flights_prior:,}; T-100 departures {p.raw.departures_latest:,}/{p.raw.departures_prior:,}.")
        elif isinstance(p, ComparisonResult):
            st.markdown(p.note)
            if p.scores_peer_group:
                st.markdown(f"Score columns are normalized within {p.scores_peer_group.describe()}.")
            for code, cov in p.coverage.items():
                if not cov.passed:
                    st.markdown(f"- {code}: below coverage rule ({cov.detail})")
        elif isinstance(p, RouteMixResult):
            st.markdown(f"long-haul share = long-haul departures / total departures = {p.long_haul_departures:,} / {p.total_departures:,} "
                        f"over {p.months_covered} months; long-haul = route distance >= {p.threshold_miles:,} statute miles. Source: {p.source}.")
        elif isinstance(p, AirportProfile):
            st.markdown("Latest vs prior 12-month windows summed from monthly aggregates; rates are ratios of the window sums "
                        f"(delay rate = delayed 15+ min / operated; cancellation rate = cancelled / scheduled). Long-haul threshold {p.long_haul_threshold_miles:,} mi.")
            st.markdown(f"**Coverage:** {'passes' if p.coverage.passed else 'fails'} the minimum-flights rule" + (f" ({p.coverage.detail})" if p.coverage.detail else "") + ".")
        elif isinstance(p, LiveContext):
            st.markdown(f"Provider {p.provider}; {p.window_description}. Delayed = departure delay >= 15 minutes in the live feed. Not comparable to historical rates.")
        elif isinstance(p, Methodology):
            for r in p.rules:
                st.markdown(f"- {r}")
        if result.metrics:
            flat = {k: v for k, v in result.metrics.items() if not isinstance(v, (list, dict))}
            st.markdown("**Headline metrics returned to the agent:**")
            st.json(flat, expanded=False)


def _render_result(result: ToolResult, key_prefix: str) -> None:
    p = result.payload
    if isinstance(p, RouteMixResult):
        c1, c2, c3 = st.columns(3)
        c1.metric("Long-haul share", pct(p.long_haul_share))
        c2.metric("Long-haul departures", f"{p.long_haul_departures:,}")
        c3.metric("Total departures", f"{p.total_departures:,}")
    elif isinstance(p, CapacityPressureResult):
        c1, c2, c3 = st.columns(3)
        c1.metric("Capacity Pressure Proxy", f"{p.proxy_score:.1f} / 100")
        c2.metric("Utilization", pct(p.raw.capacity_utilization))
        c3.metric("Demand growth", pct(p.raw.demand_growth))
    elif isinstance(p, LiveContext):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sampled departures", p.flights_sampled)
        c2.metric("Delayed 15+ min", p.delayed_15m)
        c3.metric("Cancelled", p.cancelled)
        c4.metric("Avg departure delay", f"{p.avg_departure_delay_minutes:.1f} min" if p.avg_departure_delay_minutes is not None else "n/a")
    table = table_for_result(result)
    if table is not None and not table.empty:
        _show_df(table, height=min(420, 40 + 35 * (len(table) + 1)), key=f"{key_prefix}_table")
    if result.status != "ok" and result.message:
        (st.warning if result.status in ("insufficient_data", "ambiguous", "unavailable") else st.error)(result.message)
    fig = chart_for_result(result)
    if fig is not None:
        _show_chart(fig, key=f"{key_prefix}_chart")
    _methodology_panel(result, key_prefix)
    if result.limitations:
        with st.container():
            st.markdown('<div class="section-head">Assumptions and limitations</div>', unsafe_allow_html=True)
            st.info("\n".join(f"- {l}" for l in result.limitations))
    _evidence_block(result)


def render_turn(turn: AgentTurn, key_prefix: str = "turn") -> None:
    """Render one assistant turn. `key_prefix` must be unique per turn within a script run; it seeds the
    explicit keys of the tables and charts so identical results in different turns do not collide."""
    conclusion = turn.conclusion or "(no conclusion)"
    st.markdown(f'<div class="conclusion">{conclusion}</div>', unsafe_allow_html=True)
    body = (turn.answer_markdown or "").strip()
    # the conclusion is already shown above; drop a leading "**Conclusion:** ..." paragraph from the body
    if body.startswith("**Conclusion:**"):
        parts = body.split("\n\n", 1)
        body = parts[1].strip() if len(parts) > 1 else ""
    if body and body != conclusion:
        st.markdown(body)
    if turn.grounding_warnings:
        st.caption("Note: " + " ".join(turn.grounding_warnings))
    if not turn.tool_calls:
        return
    both = bool(turn.historical_calls) and bool(turn.live_calls)
    if turn.historical_calls:
        if both:
            st.markdown(f"#### {HISTORICAL_HEADING}")
        for i in turn.historical_calls:
            _render_result(turn.tool_calls[i].result, f"{key_prefix}_call{i}")
    if turn.live_calls:
        if both:
            st.divider()
            st.markdown(f"#### {LIVE_HEADING}")
        for i in turn.live_calls:
            _render_result(turn.tool_calls[i].result, f"{key_prefix}_call{i}")


def render_missing_database(db_path: str) -> None:
    st.error("No historical database found. The application needs the aggregated DuckDB store before it can answer questions.")
    st.code(f"python scripts/refresh_data.py --scope demo        # ~25 minutes, writes {db_path}", language="bash")
    st.caption("The refresh downloads public BTS TranStats data (T-100 Segment and On-Time Performance) and aggregates it locally. No raw files are used at runtime.")


__all__ = ["inject_css", "render_sidebar", "render_header", "render_suggested_questions", "render_turn", "render_missing_database", "table_for_result"]
