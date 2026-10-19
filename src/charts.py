"""One restrained Plotly chart per tool result.

Palette: validated categorical slots (dataviz validator, light surface): blue, orange, aqua, yellow, magenta, violet.
Rules applied: one axis per plot (small multiples instead of dual axes), fixed hue per entity/component, thin marks,
2 px surface gaps between stacked segments, hairline grid, legend for >= 2 series, selective direct labels.
Every chart is accompanied by a table in the UI, which satisfies the contrast relief rule.
"""
from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import AirportProfile, CapacityPressureResult, ComparisonResult, LiveContext, Methodology, RankingResult, RouteMixResult, ToolResult
from src.scoring import COMPONENT_LABELS, EXPANSION_WEIGHTS

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
COMPONENT_COLORS = {"demand_growth": SLOTS[0], "capacity_utilization": SLOTS[1], "operational_pressure": SLOTS[2]}
SURFACE = "#ffffff"
INK = "#1b2a41"
MUTED = "#52514e"
GRID = "#e1e0d9"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _style(fig: go.Figure, title: str, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="simple_white", title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        font=dict(family=FONT, size=12, color=MUTED), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=20, t=56, b=40), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11), title_text=""),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(family=FONT, color=INK)), bargap=0.35,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID, tickfont=dict(size=11), title_font=dict(size=11))
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=GRID, tickfont=dict(size=11), title_font=dict(size=11))
    return fig


def chart_for_result(result: ToolResult) -> Optional[go.Figure]:
    p = result.payload
    if result.status != "ok" or p is None:
        if isinstance(p, RankingResult) and p.exclusions:
            return None
        return None
    if isinstance(p, RankingResult):
        return ranking_chart(p)
    if isinstance(p, ComparisonResult):
        return comparison_chart(p)
    if isinstance(p, AirportProfile):
        return profile_chart(p, result.period.latest_start_month if result.period else None)
    if isinstance(p, CapacityPressureResult):
        return capacity_pressure_chart(p)
    if isinstance(p, RouteMixResult):
        return route_mix_chart(p)
    if isinstance(p, LiveContext):
        return live_chart(p)
    if isinstance(p, Methodology):
        return weights_chart(p)
    return None


def ranking_chart(p: RankingResult) -> Optional[go.Figure]:
    rows = [r for r in p.rows[:10] if r.scores.expansion_opportunity_score is not None]
    if not rows:
        return None
    codes = [r.airport.airport_code for r in rows][::-1]
    fig = go.Figure()
    for comp in ("demand_growth", "capacity_utilization", "operational_pressure"):
        vals = [r.contributions.get(comp, 0.0) for r in rows][::-1]
        fig.add_trace(go.Bar(
            y=codes, x=vals, name=f"{COMPONENT_LABELS[comp]} ({EXPANSION_WEIGHTS[comp]:.0%})", orientation="h",
            marker=dict(color=COMPONENT_COLORS[comp], line=dict(color=SURFACE, width=2)),
            hovertemplate="%{y} - " + COMPONENT_LABELS[comp] + ": %{x:.1f} points<extra></extra>",
        ))
    totals = [r.scores.expansion_opportunity_score for r in rows][::-1]
    fig.add_trace(go.Scatter(y=codes, x=[t + 1.5 for t in totals], mode="text", text=[f"{t:.0f}" for t in totals], textposition="middle right",
                             textfont=dict(color=INK, size=11), showlegend=False, hoverinfo="skip"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 112], title_text="Expansion Opportunity Score (0-100, weighted component contributions)")
    return _style(fig, "Score decomposition - ranked airports", height=max(300, 60 + 34 * len(rows)))


def comparison_chart(p: ComparisonResult) -> Optional[go.Figure]:
    rows = p.rows
    if len(rows) < 2:
        return None
    panels = [("delay_rate", "Delay rate (%)", 100.0), ("cancellation_rate", "Cancellation rate (%)", 100.0),
              ("avg_departure_delay_minutes", "Avg departure delay (min)", 1.0), ("capacity_utilization", "Utilization (%)", 100.0)]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[t for _, t, _ in panels], horizontal_spacing=0.06)
    codes = [r["airport_code"] for r in rows]
    for col, (key, _, scale) in enumerate(panels, start=1):
        for i, r in enumerate(rows):
            v = r.get(key)
            fig.add_trace(go.Bar(
                x=[None if v is None else v * scale], y=[r["airport_code"]], orientation="h", name=r["airport_code"], legendgroup=r["airport_code"],
                showlegend=(col == 1), marker=dict(color=SLOTS[i % len(SLOTS)], line=dict(color=SURFACE, width=2)),
                hovertemplate=f"{r['airport_code']}: %{{x:,.1f}}<extra></extra>",
            ), row=1, col=col)
    fig.update_yaxes(categoryorder="array", categoryarray=codes[::-1])
    fig.update_layout(barmode="group")
    fig = _style(fig, "Rate-based comparison (volume is shown in the table)", height=320)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    return fig


def profile_chart(p: AirportProfile, latest_start_month: Optional[str]) -> Optional[go.Figure]:
    series = [s for s in p.series if s.passengers is not None or s.delay_rate is not None]
    if not series:
        return None
    x = [s.year_month for s in series]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("Passengers per month", "Delayed-departure rate (%)"))
    fig.add_trace(go.Scatter(x=x, y=[s.passengers for s in series], mode="lines+markers", name="Passengers", line=dict(color=SLOTS[0], width=2),
                             marker=dict(size=6), hovertemplate="%{x}: %{y:,.0f} passengers<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[None if s.delay_rate is None else s.delay_rate * 100 for s in series], mode="lines+markers", name="Delay rate",
                             line=dict(color=SLOTS[1], width=2), marker=dict(size=6), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"), row=2, col=1)
    if latest_start_month and latest_start_month in x:
        for r in (1, 2):
            fig.add_vrect(x0=latest_start_month, x1=x[-1], fillcolor="#f4f6f9", opacity=0.9, layer="below", line_width=0, row=r, col=1)
        fig.add_annotation(x=latest_start_month, y=1.0, yref="paper", text="latest 12-month window", showarrow=False, xanchor="left",
                           font=dict(size=10, color=MUTED), yshift=14)
    fig = _style(fig, f"{p.airport.airport_code} - 24-month trend", height=420)
    fig.update_annotations(font=dict(size=11, color=MUTED))
    fig.update_xaxes(showgrid=False, tickangle=-45, nticks=12)
    fig.update_yaxes(showgrid=True, gridcolor=GRID)
    return fig


def capacity_pressure_chart(p: CapacityPressureResult) -> go.Figure:
    comps = p.components[::-1]
    fig = go.Figure(go.Bar(
        y=[f"{c.label} ({c.weight:.0%})" for c in comps], x=[c.percentile for c in comps], orientation="h",
        marker=dict(color=[COMPONENT_COLORS.get(c.name, SLOTS[0]) for c in comps], line=dict(color=SURFACE, width=2)),
        text=[f"p{c.percentile:.0f} -> {c.contribution:.1f} pts" for c in comps], textposition="outside", textfont=dict(color=INK, size=11),
        hovertemplate="%{y}: percentile %{x:.1f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 125], title_text=f"Percentile within peer group (n={p.peer_group_size}); proxy = {p.proxy_score:.1f}/100")
    return _style(fig, f"{p.airport.airport_code} - capacity pressure proxy components", height=300, legend=False)


def route_mix_chart(p: RouteMixResult) -> go.Figure:
    if p.top_long_haul_routes:
        routes = p.top_long_haul_routes[:8][::-1]
        fig = go.Figure(go.Bar(
            y=[f"{r.destination_airport} ({r.distance_miles:,.0f} mi)" for r in routes], x=[r.departures for r in routes], orientation="h",
            marker=dict(color=SLOTS[0], line=dict(color=SURFACE, width=2)), hovertemplate="%{y}: %{x:,} departures<extra></extra>",
        ))
        fig.update_xaxes(title_text="Departures in the latest 12 months")
        return _style(fig, f"{p.airport.airport_code} - top long-haul routes (>= {p.threshold_miles:,} mi)", height=max(280, 60 + 34 * len(routes)), legend=False)
    other = p.total_departures - p.long_haul_departures
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Departures"], x=[p.long_haul_departures], name="Long-haul", orientation="h", marker=dict(color=SLOTS[0], line=dict(color=SURFACE, width=2))))
    fig.add_trace(go.Bar(y=["Departures"], x=[other], name="Other", orientation="h", marker=dict(color="#c3c2b7", line=dict(color=SURFACE, width=2))))
    fig.update_layout(barmode="stack")
    return _style(fig, f"{p.airport.airport_code} - long-haul vs other departures", height=220)


def live_chart(p: LiveContext) -> Optional[go.Figure]:
    counts = {k: v for k, v in p.status_counts.items() if v}
    if not counts:
        return None
    order = ["scheduled", "active", "landed", "cancelled", "incident", "diverted"]
    keys = [k for k in order if k in counts] + [k for k in counts if k not in order]
    fig = go.Figure(go.Bar(x=keys, y=[counts[k] for k in keys], marker=dict(color=SLOTS[0], line=dict(color=SURFACE, width=2)),
                           text=[counts[k] for k in keys], textposition="outside", textfont=dict(color=INK), hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, title_text="flights in live sample")
    return _style(fig, f"{p.airport_code} - live departure status sample ({p.flights_sampled} flights)", height=280, legend=False)


def weights_chart(p: Methodology) -> Optional[go.Figure]:
    weights = p.weights.get("expansion_opportunity_score") or p.weights.get("capacity_pressure_proxy")
    if not weights:
        return None
    fig = go.Figure()
    for comp, w in weights.items():
        fig.add_trace(go.Bar(y=["weights"], x=[w * 100], name=f"{COMPONENT_LABELS.get(comp, comp)} ({w:.0%})", orientation="h",
                             marker=dict(color=COMPONENT_COLORS.get(comp, SLOTS[0]), line=dict(color=SURFACE, width=2))))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], title_text="share of the composite score (%)")
    fig.update_yaxes(showticklabels=False)
    return _style(fig, p.title + " - component weights", height=200)


__all__ = ["chart_for_result", "SLOTS"]
