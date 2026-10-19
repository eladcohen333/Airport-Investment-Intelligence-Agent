"""Allow-listed typed analytical tools. This is the only way the agent (or the demo router) reaches data.

Every tool returns a `ToolResult` with the required envelope keys. Tools never raise: validation problems and
internal errors become `status="error"` results with safe messages (details go to the developer log).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

import duckdb
from pydantic import BaseModel, ValidationError

from src import analytics as an
from src import scoring as sc
from src.airport_resolver import AirportResolver, region_states
from src.config import get_logger
from src.live_provider import LiveFlightProvider, NullLiveProvider
from src.models import (
    REGIONS, AirportProfile, AirportProfileInput, AirportRef, AnalysisPeriod, CapacityPressureInput, CapacityPressureResult,
    CompareAirportsInput, CompareColumn, ComparisonResult, ComponentScores, CoverageCheck, LiveContextInput, MethodologyInput,
    PeerGroupSpec, PeriodRequest, RankAirportsInput, RankingResult, ResolveAirportInput, ResolveResult, RouteMixInput,
    RouteMixResult, RunDefaults, ToolResult,
)

log = get_logger("tools")

SCREENING_NOTE = "Screening signal only: not an ROI estimate, a profitability forecast, or proof that terminal capacity causes operational problems."
OTP_NOTE = "Delay and cancellation statistics cover BTS reporting carriers only."


# --------------------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------------------

@dataclass
class ToolContext:
    conn: duckdb.DuckDBPyConnection
    resolver: AirportResolver
    refs: dict[str, AirportRef]
    defaults: RunDefaults
    coverage: an.CoverageInfo
    live_provider: LiveFlightProvider = field(default_factory=NullLiveProvider)
    today: date = field(default_factory=date.today)

    @property
    def data_scope(self) -> str:
        return self.coverage.data_scope or self.defaults.data_scope


def build_tool_context(conn: duckdb.DuckDBPyConnection, defaults: RunDefaults, live_provider: Optional[LiveFlightProvider] = None,
                       today: Optional[date] = None) -> ToolContext:
    rows = an.airport_rows(conn)
    resolver = AirportResolver.from_rows(rows)
    refs = an.airport_refs(conn)
    coverage = an.get_coverage(conn)
    return ToolContext(conn=conn, resolver=resolver, refs=refs, defaults=defaults, coverage=coverage,
                       live_provider=live_provider or NullLiveProvider(), today=today or date.today())


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

def _pct(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value) * 100.0, digits)


def _num(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _sources(ctx: ToolContext, *kinds: str) -> list[str]:
    """Pick source names from refresh metadata by kind: 'traffic', 'operations', 'reference'."""
    keys = {"traffic": ("T-100", "traffic", "route"), "operations": ("On-Time", "operations"), "reference": ("OurAirports", "reference")}
    out: list[str] = []
    for src in ctx.coverage.sources:
        if any(k.lower() in src.lower() for kind in kinds for k in keys[kind]):
            if src not in out:
                out.append(src)
    if not out:
        out = list(ctx.coverage.sources)
    return out


def _defs(*keys: str) -> dict[str, str]:
    return {k: sc.DEFINITIONS[k] for k in keys if k in sc.DEFINITIONS}


def _result(tool_name: str, ctx: ToolContext, status: str, *, period: Optional[AnalysisPeriod] = None, sources: Optional[list[str]] = None,
            definitions: Optional[dict[str, str]] = None, metrics: Optional[dict[str, Any]] = None, limitations: Optional[list[str]] = None,
            confidence: str = "low", message: Optional[str] = None, payload: Any = None, is_live: bool = False) -> ToolResult:
    return ToolResult(
        status=status, tool_name=tool_name, data_scope=ctx.data_scope,
        data_period=period.data_period() if period else None,
        sources=sources if sources is not None else list(ctx.coverage.sources),
        refresh_timestamp=ctx.coverage.refresh_timestamp,
        definitions=definitions or {}, metrics=metrics or {}, limitations=limitations or [], confidence=confidence,
        message=message, period=period, payload=payload, is_live=is_live,
    )


def _resolve(ctx: ToolContext, name: str, tool_name: str) -> tuple[Optional[AirportRef], Optional[ToolResult]]:
    res = ctx.resolver.resolve(name)
    if res.status == "resolved" and res.airport is not None:
        ref = ctx.refs.get(res.airport.airport_code, res.airport)
        return ref, None
    if res.status == "ambiguous":
        names = "; ".join(c.display for c in res.candidates)
        return None, _result(tool_name, ctx, "ambiguous", message=f"'{name}' is ambiguous. Candidates: {names}. Please specify one.",
                             payload=res, limitations=["No calculation was run because the airport could not be identified unambiguously."])
    msg = res.note or f"No airport matching '{name}' in the loaded data."
    status = "ambiguous" if res.candidates else "error"
    return None, _result(tool_name, ctx, status, message=msg, payload=res,
                         limitations=["No calculation was run because the airport could not be identified."])


def _period(ctx: ToolContext, req: Optional[PeriodRequest], tool_name: str) -> tuple[Optional[AnalysisPeriod], list[str], Optional[ToolResult]]:
    period, notes = an.resolve_period(req, ctx.defaults.period_end_month, ctx.coverage)
    if period is None:
        return None, notes, _result(tool_name, ctx, "insufficient_data", message=" ".join(notes) or "No usable analysis period.",
                                    limitations=notes + ["No score is reported without two complete 12-month windows."])
    return period, notes, None


def _spec(ctx: ToolContext, *, states: Optional[list[str]] = None, region: Optional[str] = None, size_bands: Optional[list[str]] = None,
          use_defaults: bool = True) -> PeerGroupSpec:
    if use_defaults and not states and not region:
        states, region = ctx.defaults.states, ctx.defaults.region
    if use_defaults and not size_bands:
        size_bands = ctx.defaults.size_bands
    if region and not states:
        states = region_states(region)
    return PeerGroupSpec(data_scope=ctx.data_scope, states=states or None, region=region, size_bands=size_bands or None,
                         min_reported_flights=ctx.defaults.min_reported_flights)


def _peer_frame(ctx: ToolContext, spec: PeerGroupSpec, period: AnalysisPeriod):
    codes = an.peer_codes(ctx.conn, spec)
    return codes, an.compute_raw_kpis(ctx.conn, period, codes)


# --------------------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------------------

def resolve_airport(inp: ResolveAirportInput, ctx: ToolContext) -> ToolResult:
    res: ResolveResult = ctx.resolver.resolve(inp.name)
    status = {"resolved": "ok", "ambiguous": "ambiguous", "unknown": "error"}[res.status]
    if res.status == "unknown" and res.candidates:
        status = "ambiguous"
    metrics = {"airport_code": res.airport.airport_code} if res.airport else {}
    return _result("resolve_airport", ctx, status, sources=_sources(ctx, "reference", "traffic"), metrics=metrics, payload=res,
                   message=res.note if res.status != "resolved" else None, confidence="high" if res.match_type in ("code", "alias", "city") else "medium")


def get_airport_profile(inp: AirportProfileInput, ctx: ToolContext) -> ToolResult:
    tool = "get_airport_profile"
    ref, err = _resolve(ctx, inp.airport, tool)
    if err:
        return err
    period, notes, perr = _period(ctx, inp.period, tool)
    if perr:
        return perr
    raw = an.raw_for_airport(ctx.conn, ref.airport_code, period)
    if raw is None or (raw.traffic_months_latest + raw.traffic_months_prior + raw.ops_months_latest + raw.ops_months_prior) == 0:
        return _result(tool, ctx, "insufficient_data", period=period,
                       message=f"No historical traffic or operations rows for {ref.airport_code} in {period.label}.",
                       limitations=notes + [f"{ref.airport_code} has no coverage in the loaded {ctx.data_scope} data for this period."])
    thr = ctx.defaults.long_haul_threshold_miles
    mix_latest = an.route_mix(ctx.conn, ref.airport_code, period.latest_start, period.latest_end, thr)
    mix_prior = an.route_mix(ctx.conn, ref.airport_code, period.prior_start, period.prior_end, thr)
    changes = an.profile_changes(raw, mix_latest.share, mix_prior.share)
    series = an.monthly_series(ctx.conn, ref.airport_code, period)
    coverage = sc.check_coverage(raw, ctx.defaults.min_reported_flights)
    limitations = list(notes) + [SCREENING_NOTE, OTP_NOTE]
    if not coverage.passed:
        limitations.append(f"Below the minimum coverage rule ({coverage.detail}); values are shown but no scores are computed.")
    m, all_months, _ = sc.months_complete(raw)
    if not all_months:
        limitations.append(f"Only {m} of 12 months are present in at least one window; comparisons are partial.")
    limitations.extend(raw.quality_flags)
    payload = AirportProfile(airport=ref, changes=changes, series=series, raw=raw, coverage=coverage, long_haul_threshold_miles=thr,
                             long_haul_share_latest=mix_latest.share)
    metrics = {
        "passengers_latest_12m": raw.pax_latest, "passengers_prior_12m": raw.pax_prior, "demand_growth_pct": _pct(raw.demand_growth),
        "capacity_utilization_pct": _pct(raw.capacity_utilization), "delay_rate_pct": _pct(raw.delay_rate),
        "delay_rate_prior_pct": _pct(raw.delay_rate_prior), "cancellation_rate_pct": _pct(raw.cancellation_rate, 2),
        "cancellation_rate_prior_pct": _pct(raw.cancellation_rate_prior, 2), "avg_departure_delay_minutes": _num(raw.avg_departure_delay_minutes),
        "long_haul_share_pct": _pct(mix_latest.share), "reported_flights_latest_12m": raw.flights_latest,
    }
    conf = sc.derive_confidence(coverage, raw, peer_n=sc.HIGH_CONFIDENCE_PEER_GROUP, clamped=period.clamped)
    return _result(tool, ctx, "ok", period=period, sources=_sources(ctx, "traffic", "operations"),
                   definitions=_defs("demand_growth", "capacity_utilization", "delayed_departure", "long_haul_flight"),
                   metrics=metrics, limitations=limitations, confidence=conf, payload=payload)


def rank_airports(inp: RankAirportsInput, ctx: ToolContext) -> ToolResult:
    tool = "rank_airports"
    period, notes, perr = _period(ctx, inp.period, tool)
    if perr:
        return perr
    spec = _spec(ctx, states=inp.states, region=inp.region, size_bands=list(inp.size_bands) if inp.size_bands else None)
    codes, kpis = _peer_frame(ctx, spec, period)
    if not codes:
        return _result(tool, ctx, "insufficient_data", period=period, message=f"No airports with traffic and operations data match: {spec.describe()}.",
                       limitations=notes + ["Adjust the state/region or size filters, or refresh a wider data scope."])
    comp = sc.rank_peer_group(kpis, ctx.refs, spec, sort_by=inp.sort_by, conditions=inp.conditions or [], ascending=inp.ascending,
                              top_n=inp.top_n, clamped=period.clamped)
    payload = RankingResult(peer_group=spec, peer_group_size=comp.peer_n, sort_by=inp.sort_by, conditions=inp.conditions or [], rows=comp.rows,
                            total_ranked=comp.total_ranked, exclusions=comp.exclusions,
                            normalization={"method": "winsorize p5/p95 within peer group, then min-max 0-100", "bounds": {k: [None if v != v else round(v, 6) for v in b] for k, b in comp.bounds.items()},
                                           "weights": sc.EXPANSION_WEIGHTS})
    limitations = list(notes) + comp.notes + [SCREENING_NOTE, OTP_NOTE, "Airport size is a peer filter only and never adds points."]
    if comp.exclusions:
        limitations.append(f"{len(comp.exclusions)} airports excluded for insufficient coverage (minimum {spec.min_reported_flights} reported flights in both windows); see exclusions.")
    if not comp.rows:
        status = "insufficient_data"
        message = "No airport in this peer group met the coverage rule" + (" and screening conditions." if inp.conditions else ".")
        return _result(tool, ctx, status, period=period, message=message, payload=payload, limitations=limitations,
                       sources=_sources(ctx, "traffic", "operations"), definitions=_defs("expansion_opportunity_score", "airport_size"))
    top = comp.rows[0]
    metrics = {
        "peer_group": spec.describe(), "peer_group_size": comp.peer_n, "airports_ranked": comp.total_ranked, "airports_excluded": len(comp.exclusions),
        "sort_by": inp.sort_by, "top_airport": top.airport.airport_code,
        "top_expansion_opportunity_score": _num(top.scores.expansion_opportunity_score),
        "top_demand_growth_pct": _pct(top.raw.demand_growth), "top_capacity_utilization_pct": _pct(top.raw.capacity_utilization),
        "top_delay_rate_pct": _pct(top.raw.delay_rate), "top_cancellation_rate_pct": _pct(top.raw.cancellation_rate, 2),
        "ranking": [{"rank": r.rank, "airport_code": r.airport.airport_code, "score": _num(r.scores.expansion_opportunity_score),
                     "demand_growth_pct": _pct(r.raw.demand_growth), "capacity_utilization_pct": _pct(r.raw.capacity_utilization),
                     "operational_pressure": _num(r.raw.operational_pressure, 4)} for r in comp.rows[:10]],
    }
    return _result(tool, ctx, "ok", period=period, sources=_sources(ctx, "traffic", "operations"),
                   definitions=_defs("expansion_opportunity_score", "demand_growth", "capacity_utilization", "operational_pressure", "airport_size", "new_england" if spec.region == "new_england" else "airport_congestion"),
                   metrics=metrics, limitations=limitations, confidence=comp.confidence, payload=payload)


COMPARE_COLUMNS: list[CompareColumn] = [
    CompareColumn(key="total_flights", label="Reported flights (12m)", kind="absolute", unit="flights"),
    CompareColumn(key="delayed_flights", label="Delayed departures 15+ min (12m)", kind="absolute", unit="flights"),
    CompareColumn(key="delay_rate", label="Delay rate", kind="rate", unit="%"),
    CompareColumn(key="avg_departure_delay_minutes", label="Average departure delay", kind="rate", unit="minutes"),
    CompareColumn(key="cancellation_rate", label="Cancellation rate", kind="rate", unit="%"),
    CompareColumn(key="passengers", label="Passengers (12m)", kind="absolute", unit="passengers"),
    CompareColumn(key="capacity_utilization", label="Capacity utilization", kind="rate", unit="%"),
    CompareColumn(key="demand_growth", label="Demand growth", kind="rate", unit="%"),
    CompareColumn(key="operational_pressure", label="Operational pressure index", kind="rate", unit="index"),
    CompareColumn(key="demand_growth_score", label="Demand growth score", kind="score", unit="0-100"),
    CompareColumn(key="capacity_utilization_score", label="Utilization score", kind="score", unit="0-100"),
    CompareColumn(key="operational_pressure_score", label="Operational pressure score", kind="score", unit="0-100"),
    CompareColumn(key="expansion_opportunity_score", label="Expansion Opportunity Score", kind="score", unit="0-100"),
]


def compare_airports(inp: CompareAirportsInput, ctx: ToolContext) -> ToolResult:
    tool = "compare_airports"
    refs: list[AirportRef] = []
    for name in inp.airports:
        ref, err = _resolve(ctx, name, tool)
        if err:
            return err
        if ref.airport_code not in [r.airport_code for r in refs]:
            refs.append(ref)
    if len(refs) < 2:
        return _result(tool, ctx, "error", message="At least two distinct airports are required for a comparison.")
    period, notes, perr = _period(ctx, inp.period, tool)
    if perr:
        return perr
    codes = [r.airport_code for r in refs]
    kpis = an.compute_raw_kpis(ctx.conn, period, codes).set_index("airport_code", drop=False)
    # scores within the scope-wide, all-sizes peer group (where coverage allows)
    spec = _spec(ctx, use_defaults=False)
    all_codes, all_kpis = _peer_frame(ctx, spec, period)
    comp = sc.rank_peer_group(all_kpis, ctx.refs, spec, top_n=None, clamped=period.clamped) if all_codes else None
    score_by: dict[str, ComponentScores] = {r.airport.airport_code: r.scores for r in comp.rows} if comp else {}
    peer_n = comp.peer_n if comp else 0

    wanted = set(inp.metrics or [])
    include_scores = not inp.metrics or "scores" in wanted
    base_keys = [c.key for c in COMPARE_COLUMNS if c.kind != "score"]
    if inp.metrics:
        keep = {m for m in wanted if m != "scores"}
        base_keys = [k for k in base_keys if k in keep or k in ("operational_pressure",) and ("delay_rate" in keep or "cancellation_rate" in keep)]
        if not base_keys:
            base_keys = [c.key for c in COMPARE_COLUMNS if c.kind != "score"]
    columns = [c for c in COMPARE_COLUMNS if c.key in base_keys] + ([c for c in COMPARE_COLUMNS if c.kind == "score"] if include_scores else [])

    rows: list[dict[str, Any]] = []
    coverage: dict[str, CoverageCheck] = {}
    confs = []
    missing: list[str] = []
    for ref in refs:
        if ref.airport_code not in kpis.index:
            missing.append(ref.airport_code)
            continue
        raw = sc.row_to_raw(kpis.loc[ref.airport_code])
        cov = sc.check_coverage(raw, ctx.defaults.min_reported_flights)
        coverage[ref.airport_code] = cov
        scores = score_by.get(ref.airport_code, ComponentScores())
        row = {
            "airport_code": ref.airport_code, "airport_name": ref.airport_name, "size_band": ref.size_band,
            "total_flights": raw.flights_latest, "delayed_flights": raw.delayed_latest, "delay_rate": raw.delay_rate,
            "avg_departure_delay_minutes": raw.avg_departure_delay_minutes, "cancellation_rate": raw.cancellation_rate,
            "passengers": raw.pax_latest, "capacity_utilization": raw.capacity_utilization, "demand_growth": raw.demand_growth,
            "operational_pressure": raw.operational_pressure,
            "demand_growth_score": scores.demand_growth_score, "capacity_utilization_score": scores.capacity_utilization_score,
            "operational_pressure_score": scores.operational_pressure_score, "expansion_opportunity_score": scores.expansion_opportunity_score,
            "coverage_ok": cov.passed, "months_present": sc.months_complete(raw)[0],
        }
        rows.append(row)
        confs.append(sc.derive_confidence(cov, raw, peer_n, period.clamped))
    if missing:
        notes.append(f"No historical rows for {', '.join(missing)} in this period.")
    if len(rows) < 2:
        return _result(tool, ctx, "insufficient_data", period=period, message="Fewer than two compared airports have historical data in this period.",
                       limitations=notes)
    with_pressure = [r for r in rows if r["operational_pressure"] is not None]
    higher = max(with_pressure, key=lambda r: r["operational_pressure"])["airport_code"] if with_pressure else None
    most = max(rows, key=lambda r: r["total_flights"])["airport_code"]
    payload = ComparisonResult(columns=columns, rows=rows, scores_peer_group=spec if include_scores else None, higher_operational_pressure=higher,
                               most_flights=most, note=sc.CONGESTION_NOTE, coverage=coverage)
    limitations = list(notes) + [sc.CONGESTION_NOTE, SCREENING_NOTE, OTP_NOTE]
    for code, cov in coverage.items():
        if not cov.passed:
            limitations.append(f"{code} is below the minimum coverage rule ({cov.detail}); its scores are omitted.")
    if include_scores:
        limitations.append(f"Score components are normalized within {spec.describe()} (n={peer_n}).")
    metrics: dict[str, Any] = {"higher_operational_pressure": higher, "most_flights": most, "period": period.short_label}
    for r in rows:
        c = r["airport_code"]
        metrics[f"{c}_total_flights"] = r["total_flights"]
        metrics[f"{c}_delay_rate_pct"] = _pct(r["delay_rate"])
        metrics[f"{c}_cancellation_rate_pct"] = _pct(r["cancellation_rate"], 2)
        metrics[f"{c}_avg_departure_delay_minutes"] = _num(r["avg_departure_delay_minutes"])
        metrics[f"{c}_passengers"] = r["passengers"]
        metrics[f"{c}_capacity_utilization_pct"] = _pct(r["capacity_utilization"])
        metrics[f"{c}_demand_growth_pct"] = _pct(r["demand_growth"])
        if include_scores and r["expansion_opportunity_score"] is not None:
            metrics[f"{c}_expansion_opportunity_score"] = _num(r["expansion_opportunity_score"])
    return _result(tool, ctx, "ok", period=period, sources=_sources(ctx, "traffic", "operations"),
                   definitions=_defs("airport_congestion", "delayed_departure", "operational_pressure", "capacity_utilization"),
                   metrics=metrics, limitations=limitations, confidence=sc.min_confidence(confs), payload=payload)


def analyze_route_mix(inp: RouteMixInput, ctx: ToolContext) -> ToolResult:
    tool = "analyze_route_mix"
    ref, err = _resolve(ctx, inp.airport, tool)
    if err:
        return err
    period, notes, perr = _period(ctx, inp.period, tool)
    if perr:
        return perr
    thr = inp.long_haul_threshold_miles or ctx.defaults.long_haul_threshold_miles
    mix = an.route_mix(ctx.conn, ref.airport_code, period.latest_start, period.latest_end, thr)
    definitions = _defs("long_haul_flight")
    definitions["long_haul_flight"] = f"A route segment of at least {thr:,} statute miles (configurable; default 2,000)."
    sources = _sources(ctx, "traffic")
    if mix.total_departures == 0:
        return _result(tool, ctx, "insufficient_data", period=period, sources=sources, definitions=definitions,
                       message=f"No route-level departures for {ref.airport_code} in the 12 months ending {period.end_month}.",
                       limitations=notes + ["Route mix needs T-100 segment rows for the origin airport."])
    payload = RouteMixResult(airport=ref, long_haul_departures=mix.long_haul_departures, total_departures=mix.total_departures, long_haul_share=mix.share,
                             threshold_miles=thr, months_covered=mix.months_covered, top_long_haul_routes=mix.top_routes, source=sources[0] if sources else "")
    limitations = list(notes) + ["Departures are scheduled passenger/combination service (T-100 class F) from the origin airport; all-cargo and charter operations are excluded.",
                                 "Distance is the segment's great-circle statute miles as reported in T-100."]
    if mix.total_departures < ctx.defaults.min_reported_flights:
        limitations.append(f"Only {mix.total_departures:,} departures in the window (below the {ctx.defaults.min_reported_flights} coverage minimum); treat the share as indicative.")
        conf = "low"
    elif mix.months_covered >= 12:
        conf = "high" if not period.clamped else "medium"
    else:
        conf = "medium"
        limitations.append(f"Only {mix.months_covered} of 12 months have route rows.")
    metrics = {"long_haul_departures": mix.long_haul_departures, "total_departures": mix.total_departures, "long_haul_share_pct": _pct(mix.share),
               "threshold_miles": thr, "months_covered": mix.months_covered, "window": f"{period.latest_start_month} to {period.end_month}",
               "top_long_haul_routes": [f"{r.destination_airport} ({r.distance_miles:,.0f} mi, {r.departures:,} dep)" for r in mix.top_routes[:5]]}
    return _result(tool, ctx, "ok", period=period, sources=sources, definitions=definitions, metrics=metrics, limitations=limitations,
                   confidence=conf, payload=payload)


def analyze_capacity_pressure(inp: CapacityPressureInput, ctx: ToolContext) -> ToolResult:
    tool = "analyze_capacity_pressure"
    ref, err = _resolve(ctx, inp.airport, tool)
    if err:
        return err
    period, notes, perr = _period(ctx, inp.period, tool)
    if perr:
        return perr
    spec = _spec(ctx, use_defaults=False)
    band_note = None
    if inp.same_size_band_peers and ref.size_band:
        band_spec = PeerGroupSpec(data_scope=ctx.data_scope, size_bands=[ref.size_band], min_reported_flights=ctx.defaults.min_reported_flights)
        band_codes, band_kpis = _peer_frame(ctx, band_spec, period)
        covered = [c for c in band_codes if sc.check_coverage(sc.row_to_raw(band_kpis.set_index("airport_code", drop=False).loc[c]), spec.min_reported_flights).passed] if band_codes else []
        if len(covered) >= sc.SMALL_PEER_GROUP:
            spec, codes, kpis = band_spec, band_codes, band_kpis
        else:
            band_note = f"Fewer than {sc.SMALL_PEER_GROUP} covered airports share the {ref.size_band} size band; percentiles use all size bands instead."
            codes, kpis = _peer_frame(ctx, spec, period)
    else:
        codes, kpis = _peer_frame(ctx, spec, period)
    if ref.airport_code not in set(kpis["airport_code"]) if not kpis.empty else True:
        return _result(tool, ctx, "insufficient_data", period=period, message=f"{ref.airport_code} has no traffic and operations coverage in {period.label}.",
                       limitations=notes + [sc.UNMET_DEMAND_DISCLAIMER])
    single = sc.score_single_airport(kpis, ref.airport_code, spec, clamped=period.clamped)
    definitions = _defs("capacity_pressure", "capacity_pressure_proxy", "capacity_utilization", "demand_growth", "operational_pressure")
    sources = _sources(ctx, "traffic", "operations")
    if single is None or not single.coverage.passed:
        detail = single.coverage.detail if single else "no data"
        return _result(tool, ctx, "insufficient_data", period=period, sources=sources, definitions=definitions,
                       message=f"{ref.airport_code} does not meet the minimum coverage rule ({detail}); no capacity-pressure proxy is reported.",
                       limitations=notes + [sc.UNMET_DEMAND_DISCLAIMER, "Insufficient coverage: raw values may exist but a proxy score would be unreliable."])
    payload = CapacityPressureResult(airport=ref, proxy_score=round(single.capacity_pressure_proxy, 2), components=single.components, peer_group=spec,
                                     peer_group_size=single.peer_n, raw=single.raw, coverage=single.coverage, expansion=single.scores)
    limitations = list(notes) + [sc.UNMET_DEMAND_DISCLAIMER, SCREENING_NOTE, OTP_NOTE] + single.notes
    if band_note:
        limitations.append(band_note)
    limitations.extend(single.raw.quality_flags)
    comps = {c.name: c for c in single.components}
    metrics = {
        "capacity_pressure_proxy": round(single.capacity_pressure_proxy, 1), "peer_group": spec.describe(), "peer_group_size": single.peer_n,
        "capacity_utilization_pct": _pct(single.raw.capacity_utilization), "capacity_utilization_percentile": round(comps["capacity_utilization"].percentile, 1),
        "demand_growth_pct": _pct(single.raw.demand_growth), "demand_growth_percentile": round(comps["demand_growth"].percentile, 1),
        "operational_pressure_index": _num(single.raw.operational_pressure, 4), "operational_pressure_percentile": round(comps["operational_pressure"].percentile, 1),
        "delay_rate_pct": _pct(single.raw.delay_rate), "cancellation_rate_pct": _pct(single.raw.cancellation_rate, 2),
        "expansion_opportunity_score": _num(single.scores.expansion_opportunity_score),
        "contributions": {c.name: round(c.contribution, 1) for c in single.components},
    }
    return _result(tool, ctx, "ok", period=period, sources=sources, definitions=definitions, metrics=metrics, limitations=limitations,
                   confidence=single.confidence, payload=payload)


def get_live_operational_context(inp: LiveContextInput, ctx: ToolContext) -> ToolResult:
    tool = "get_live_operational_context"
    ref, err = _resolve(ctx, inp.airport, tool)
    if err:
        err.is_live = True
        return err
    base_limits = ["Live context is shown separately and never modifies historical scores or rankings.",
                   "Live figures are a point-in-time sample of the provider's feed, not comparable to 12-month historical rates."]
    sources = ["Aviationstack /v1/flights (live)"]
    if not ctx.defaults.live_enabled:
        return _result(tool, ctx, "unavailable", sources=sources, is_live=True, message="Live context is disabled in the sidebar.",
                       limitations=base_limits, metrics={"airport_code": ref.airport_code})
    snap = ctx.live_provider.get_departure_snapshot(ref.airport_code)
    if not snap.ok:
        lim = list(base_limits)
        if snap.retry_after_seconds:
            lim.append(f"Provider asked to retry after {snap.retry_after_seconds} seconds.")
        return _result(tool, ctx, snap.status if snap.status in ("unavailable", "error") else "error", sources=sources, is_live=True,
                       message=snap.message or "Live context unavailable.", limitations=lim, metrics={"airport_code": ref.airport_code})
    live = snap.context
    metrics = {"airport_code": ref.airport_code, "retrieved_at": live.retrieved_at, "flights_sampled": live.flights_sampled, "scheduled": live.scheduled,
               "active": live.active, "landed": live.landed, "cancelled": live.cancelled, "delayed_15m": live.delayed_15m,
               "avg_departure_delay_minutes": _num(live.avg_departure_delay_minutes)}
    conf = "high" if live.flights_sampled >= 20 else ("medium" if live.flights_sampled >= 5 else "low")
    result = _result(tool, ctx, "ok", sources=sources, is_live=True, definitions=_defs("delayed_departure"), metrics=metrics,
                     limitations=base_limits + [live.window_description], confidence=conf, payload=live)
    result.refresh_timestamp = live.retrieved_at
    return result


def get_methodology(inp: MethodologyInput, ctx: ToolContext) -> ToolResult:
    m = sc.methodology(inp.topic)
    return _result("get_methodology", ctx, "ok", sources=_sources(ctx, "traffic", "operations", "reference"), definitions=dict(sc.DEFINITIONS),
                   metrics={"topic": m.topic, "weights": m.weights, "thresholds": {k: v for k, v in m.thresholds.items() if not isinstance(v, dict)}},
                   limitations=[SCREENING_NOTE], confidence="high", payload=m)


# --------------------------------------------------------------------------------------
# Registry, execution, schema export
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], ToolResult]
    live: bool = False


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "resolve_airport": ToolSpec(
        "resolve_airport",
        "Resolve a free-text airport name, city or code to a canonical IATA code; returns candidates when ambiguous. "
        "Other tools resolve names themselves, so call this only to disambiguate before asking the user.",
        ResolveAirportInput, resolve_airport),
    "get_airport_profile": ToolSpec(
        "get_airport_profile",
        "Historical profile of one airport: latest vs prior 12-month values and changes for passengers, seats, utilization, flights, "
        "delay rate, cancellation rate, average delay and long-haul share, plus a 24-month series. Use for 'what changed', trends, overviews.",
        AirportProfileInput, get_airport_profile),
    "rank_airports": ToolSpec(
        "rank_airports",
        "Rank airports in a peer group (region or states, optional size bands, period) by Expansion Opportunity Score or one KPI, with optional "
        "screening conditions (e.g. demand_growth > 0 and cancellation_rate_change > 0). Returns ranked rows with raw KPIs, component scores, "
        "exclusions and limitations. Use for 'which airports ...' questions. Omit filters to use the sidebar defaults.",
        RankAirportsInput, rank_airports),
    "compare_airports": ToolSpec(
        "compare_airports",
        "Compare 2-6 airports on absolute and rate-based measures (reported flights, delayed flights, delay rate, average departure delay, "
        "cancellation rate, passengers, utilization, demand growth) plus score components where coverage allows. Congestion ordering is rate-based.",
        CompareAirportsInput, compare_airports),
    "analyze_route_mix": ToolSpec(
        "analyze_route_mix",
        "Share of departures from one airport that are long-haul (route distance >= threshold statute miles) over the latest 12 months: "
        "numerator, denominator, percentage, threshold and top long-haul routes.",
        RouteMixInput, analyze_route_mix),
    "analyze_capacity_pressure": ToolSpec(
        "analyze_capacity_pressure",
        "Capacity Pressure Proxy for one airport (50% utilization percentile + 30% demand-growth percentile + 20% operational-pressure percentile "
        "within its peer group) with components, raw values and limitations. A proxy, never measured unmet demand.",
        CapacityPressureInput, analyze_capacity_pressure),
    "get_live_operational_context": ToolSpec(
        "get_live_operational_context",
        "Current operational context for one airport from the live flight provider (sampled departures now: scheduled, active, delayed 15+ min, "
        "cancelled). Never affects historical scores. Returns status 'unavailable' without a key or when the provider fails.",
        LiveContextInput, get_live_operational_context, live=True),
    "get_methodology": ToolSpec(
        "get_methodology",
        "Definitions, formulas, weights, thresholds, peer-group, normalization, coverage and confidence rules. Use for 'how is X calculated'.",
        MethodologyInput, get_methodology),
}


def execute_tool(name: str, raw_input: Optional[dict[str, Any]], ctx: ToolContext) -> ToolResult:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return _result(name or "unknown_tool", ctx, "error", message=f"Tool '{name}' is not in the allow-list.")
    try:
        inp = spec.input_model.model_validate(raw_input or {})
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()[:4])
        return _result(name, ctx, "error", message=f"Invalid parameters for {name}: {problems}")
    t0 = time.time()
    try:
        result = spec.handler(inp, ctx)
    except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
        log.exception("tool %s failed", name)
        return _result(name, ctx, "error", message=f"{name} failed internally ({type(exc).__name__}); details were logged.")
    log.info("tool %s status=%s in %.0f ms", name, result.status, (time.time() - t0) * 1000)
    return result


def tool_names(live_enabled: bool = True) -> list[str]:
    return [n for n, s in TOOL_REGISTRY.items() if live_enabled or not s.live]


# ---- JSON schema export ----------------------------------------------------------------

def _inline(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"].split("/")[-1]
            return _inline(copy.deepcopy(defs[ref]), defs)
        return {k: _inline(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def _strip(node: Any, drop: tuple[str, ...]) -> Any:
    if isinstance(node, dict):
        return {k: _strip(v, drop) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_strip(v, drop) for v in node]
    return node


def _split_nullable(node: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if "anyOf" in node:
        branches = [b for b in node["anyOf"] if b.get("type") != "null"]
        nullable = len(branches) < len(node["anyOf"])
        if len(branches) == 1:
            merged = dict(node)
            merged.pop("anyOf")
            merged.update(branches[0])
            return merged, nullable
    return node, False


def _to_anthropic(node: Any) -> Any:
    if isinstance(node, list):
        return [_to_anthropic(v) for v in node]
    if not isinstance(node, dict):
        return node
    node, nullable = _split_nullable(node)
    out: dict[str, Any] = {}
    for k, v in node.items():
        out[k] = _to_anthropic(v)
    if out.get("type") == "object" and "properties" in out:
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    if nullable:
        t = out.get("type")
        if isinstance(t, str) and t != "object" and "enum" not in out:
            out["type"] = [t, "null"]
        elif "enum" in out and isinstance(t, str):
            out["type"] = [t, "null"]
            if None not in out["enum"]:
                out["enum"] = list(out["enum"]) + [None]
        else:
            return {"anyOf": [out, {"type": "null"}]}
    return out


def _to_gemini(node: Any) -> Any:
    if isinstance(node, list):
        return [_to_gemini(v) for v in node]
    if not isinstance(node, dict):
        return node
    node, nullable = _split_nullable(node)
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k in ("additionalProperties",):
            continue
        out[k] = _to_gemini(v)
    if nullable:
        out["nullable"] = True
    return out


def to_anthropic_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    inlined = _inline(schema, defs)
    return _to_anthropic(_strip(inlined, ("title", "default")))


def to_gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})
    inlined = _inline(schema, defs)
    return _to_gemini(_strip(inlined, ("title", "default")))


def anthropic_tool_definitions(live_enabled: bool = True) -> list[dict[str, Any]]:
    return [{"name": s.name, "description": s.description, "input_schema": to_anthropic_schema(s.input_model), "strict": True}
            for s in TOOL_REGISTRY.values() if live_enabled or not s.live]


def gemini_tool_definitions(live_enabled: bool = True) -> list[dict[str, Any]]:
    return [{"name": s.name, "description": s.description, "parameters": to_gemini_schema(s.input_model)}
            for s in TOOL_REGISTRY.values() if live_enabled or not s.live]


__all__ = ["ToolContext", "build_tool_context", "TOOL_REGISTRY", "ToolSpec", "execute_tool", "tool_names", "anthropic_tool_definitions",
           "gemini_tool_definitions", "to_anthropic_schema", "to_gemini_schema", "COMPARE_COLUMNS", "SCREENING_NOTE"]
