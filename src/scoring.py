"""Deterministic KPI, normalization and scoring logic. Pure functions; no database or LLM access.

Formulas (brief section E)
  Demand Growth            = (pax latest 12m - pax prior 12m) / pax prior 12m
  Capacity Utilization     = passengers / available seats (latest 12m)
  Operational Pressure     = 0.70 x delayed-departure rate + 0.30 x cancellation rate
  Expansion Opportunity    = 0.40 x growth score + 0.35 x utilization score + 0.25 x pressure score   (0-100 components)
  Capacity Pressure Proxy  = 0.50 x utilization pct + 0.30 x growth pct + 0.20 x pressure pct          (percentile ranks)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from src.models import (
    AirportRef, ComponentScores, Condition, Confidence, CoverageCheck, CPPComponent, Exclusion, Methodology,
    PeerGroupSpec, RankedAirport, RawKPIs,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

EXPANSION_WEIGHTS: dict[str, float] = {"demand_growth": 0.40, "capacity_utilization": 0.35, "operational_pressure": 0.25}
CPP_WEIGHTS: dict[str, float] = {"capacity_utilization": 0.50, "demand_growth": 0.30, "operational_pressure": 0.20}
OPERATIONAL_PRESSURE_WEIGHTS: dict[str, float] = {"delay_rate": 0.70, "cancellation_rate": 0.30}

WINSOR_LOW_PCT = 5.0
WINSOR_HIGH_PCT = 95.0
DEFAULT_MIN_REPORTED_FLIGHTS = 500
MIN_PEERS_FOR_SCORES = 2
SMALL_PEER_GROUP = 5
HIGH_CONFIDENCE_PEER_GROUP = 10
UTILIZATION_FLAG_THRESHOLD = 1.05
COMPLETE_MONTHS = 12
MEDIUM_CONFIDENCE_MIN_MONTHS = 10

COMPONENTS: tuple[str, ...] = ("demand_growth", "capacity_utilization", "operational_pressure")
COMPONENT_LABELS = {
    "demand_growth": "Demand growth",
    "capacity_utilization": "Capacity utilization",
    "operational_pressure": "Operational pressure",
}

DEFINITIONS: dict[str, str] = {
    "delayed_departure": "Departure delayed by 15 minutes or more.",
    "long_haul_flight": "A route segment of at least 2,000 statute miles (threshold configurable).",
    "new_england": "Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, and Vermont.",
    "airport_congestion": "Operational pressure from delays and cancellations; not proof of terminal congestion.",
    "capacity_pressure": "A proxy based on historical demand, utilization, delays, and cancellations; not direct unmet demand.",
    "airport_size": "A peer-comparison filter (FAA-style hub band from annual origin passengers), not a score component.",
    "demand_growth": "(passengers in latest rolling 12 months - passengers in prior rolling 12 months) / passengers in prior rolling 12 months.",
    "capacity_utilization": "Total passengers / total available seats over the latest 12 months.",
    "operational_pressure": "70% x delayed departure rate + 30% x cancellation rate over the latest 12 months.",
    "expansion_opportunity_score": "40% demand growth + 35% capacity utilization + 25% operational pressure, each normalized 0-100 within the peer group.",
    "capacity_pressure_proxy": "50% capacity-utilization percentile + 30% demand-growth percentile + 20% operational-pressure percentile within the peer group.",
}
UNMET_DEMAND_DISCLAIMER = (
    "This is a capacity-pressure proxy, not measured unmet demand. Direct unmet demand would require search, booking, "
    "fare, leakage, or alternative-airport data."
)
CONGESTION_NOTE = "Congestion ordering uses rate-based measures (delay rate, cancellation rate, average delay); total flights are context, not evidence of congestion."
SIZE_BAND_THRESHOLDS = {"large": 10_000_000, "medium": 2_500_000, "small": 500_000}

EXCLUSION_LABELS = {
    "insufficient_operations_coverage": "fewer reported On-Time flights than the minimum in one or both 12-month windows",
    "insufficient_traffic_coverage": "fewer reported T-100 departures than the minimum in one or both 12-month windows",
    "zero_prior_passengers": "no passengers reported in the prior 12-month window",
    "zero_seats": "no available seats reported in the latest 12-month window",
    "no_operated_flights": "no operated flights in the latest 12-month window",
    "missing_component": "a score component could not be computed",
}


# --------------------------------------------------------------------------------------
# Elementary helpers
# --------------------------------------------------------------------------------------

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def safe_ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if _is_missing(numerator) or _is_missing(denominator) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def operational_pressure(delay_rate: Optional[float], cancellation_rate: Optional[float]) -> Optional[float]:
    if _is_missing(delay_rate) or _is_missing(cancellation_rate):
        return None
    w = OPERATIONAL_PRESSURE_WEIGHTS
    return w["delay_rate"] * float(delay_rate) + w["cancellation_rate"] * float(cancellation_rate)


def demand_growth(pax_latest: Any, pax_prior: Any) -> Optional[float]:
    if _is_missing(pax_latest) or _is_missing(pax_prior) or float(pax_prior) <= 0:
        return None
    return (float(pax_latest) - float(pax_prior)) / float(pax_prior)


def winsorize(values: pd.Series, low_pct: float = WINSOR_LOW_PCT, high_pct: float = WINSOR_HIGH_PCT) -> tuple[pd.Series, tuple[float, float]]:
    """Clip to [p_low, p_high] (linear interpolation). Returns clipped series and the bounds."""
    clean = values.astype(float)
    if clean.empty:
        return clean, (float("nan"), float("nan"))
    lo = float(np.percentile(clean.values, low_pct))
    hi = float(np.percentile(clean.values, high_pct))
    return clean.clip(lower=lo, upper=hi), (lo, hi)


def minmax_0_100(values: pd.Series) -> pd.Series:
    """Min-max to 0-100. Constant input (or a single value) maps to 50 for every row."""
    clean = values.astype(float)
    if clean.empty:
        return clean
    lo, hi = float(clean.min()), float(clean.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(hi, lo, rel_tol=0.0, abs_tol=1e-12):
        return pd.Series(50.0, index=clean.index)
    return (clean - lo) / (hi - lo) * 100.0


def percentile_ranks(values: pd.Series) -> pd.Series:
    """Hazen percentile ranks in 0-100: 100 * (average rank - 0.5) / n. Ties share the average rank."""
    clean = values.astype(float)
    n = len(clean)
    if n == 0:
        return clean
    if n == 1:
        return pd.Series(50.0, index=clean.index)
    ranks = clean.rank(method="average")
    return (ranks - 0.5) / n * 100.0


def weighted_sum(components: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    total = 0.0
    for key, weight in weights.items():
        value = components.get(key)
        if _is_missing(value):
            return None
        total += weight * float(value)
    return total


# --------------------------------------------------------------------------------------
# Coverage and confidence
# --------------------------------------------------------------------------------------

def check_coverage(raw: RawKPIs, min_reported_flights: int = DEFAULT_MIN_REPORTED_FLIGHTS) -> CoverageCheck:
    reasons: list[str] = []
    details: list[str] = []
    if raw.flights_latest < min_reported_flights or raw.flights_prior < min_reported_flights:
        reasons.append("insufficient_operations_coverage")
        details.append(f"On-Time flights latest/prior = {raw.flights_latest:,}/{raw.flights_prior:,} (minimum {min_reported_flights:,} each)")
    if raw.departures_latest < min_reported_flights or raw.departures_prior < min_reported_flights:
        reasons.append("insufficient_traffic_coverage")
        details.append(f"T-100 departures latest/prior = {raw.departures_latest:,}/{raw.departures_prior:,} (minimum {min_reported_flights:,} each)")
    if raw.pax_prior <= 0:
        reasons.append("zero_prior_passengers")
        details.append("no passengers in the prior window")
    if raw.seats_latest <= 0:
        reasons.append("zero_seats")
        details.append("no available seats in the latest window")
    if raw.operated_latest <= 0:
        reasons.append("no_operated_flights")
        details.append("no operated flights in the latest window")
    if not reasons and any(_is_missing(getattr(raw, c)) for c in COMPONENTS):
        reasons.append("missing_component")
        details.append("one or more score components could not be computed")
    return CoverageCheck(passed=not reasons, reasons=reasons, detail="; ".join(details), min_reported_flights=min_reported_flights)


def months_complete(raw: RawKPIs) -> tuple[int, bool, bool]:
    """(minimum months present across the four windows, all 12 present, at least 10 present)."""
    m = min(raw.traffic_months_latest, raw.traffic_months_prior, raw.ops_months_latest, raw.ops_months_prior)
    return m, m >= COMPLETE_MONTHS, m >= MEDIUM_CONFIDENCE_MIN_MONTHS


def derive_confidence(coverage: CoverageCheck, raw: RawKPIs, peer_n: int, clamped: bool = False) -> Confidence:
    if not coverage.passed:
        return "low"
    _, all_months, enough_months = months_complete(raw)
    if not enough_months or peer_n < SMALL_PEER_GROUP:
        return "low"
    if all_months and peer_n >= HIGH_CONFIDENCE_PEER_GROUP and not raw.quality_flags and not clamped:
        return "high"
    return "medium"


def downgrade(conf: Confidence) -> Confidence:
    return {"high": "medium", "medium": "low", "low": "low"}[conf]


def min_confidence(values: Iterable[Confidence]) -> Confidence:
    order = {"high": 2, "medium": 1, "low": 0}
    vals = list(values)
    if not vals:
        return "low"
    return min(vals, key=lambda v: order[v])


def aggregate_confidence(values: Iterable[Confidence]) -> Confidence:
    """Ranking-level confidence: the median row confidence, capped at medium when any ranked row is low."""
    order = {"high": 2, "medium": 1, "low": 0}
    vals = sorted(values, key=lambda v: order[v])
    if not vals:
        return "low"
    median = vals[len(vals) // 2]
    if "low" in vals and median == "high":
        return "medium"
    return median


# --------------------------------------------------------------------------------------
# Component normalization and scoring over a peer group
# --------------------------------------------------------------------------------------

def normalize_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Add `<component>_score` columns (0-100) using winsorized min-max within the frame."""
    out = frame.copy()
    bounds: dict[str, tuple[float, float]] = {}
    for comp in COMPONENTS:
        series = out[comp].astype(float)
        if len(series) < MIN_PEERS_FOR_SCORES:
            out[f"{comp}_score"] = np.nan
            bounds[comp] = (float("nan"), float("nan"))
            continue
        clipped, bnd = winsorize(series)
        out[f"{comp}_score"] = minmax_0_100(clipped)
        bounds[comp] = bnd
    if len(out) >= MIN_PEERS_FOR_SCORES:
        out["expansion_opportunity_score"] = sum(EXPANSION_WEIGHTS[c] * out[f"{c}_score"] for c in COMPONENTS)
    else:
        out["expansion_opportunity_score"] = np.nan
    return out, bounds


def component_scores_from_row(row: pd.Series) -> ComponentScores:
    def val(key: str) -> Optional[float]:
        v = row.get(key)
        return None if _is_missing(v) else float(v)
    return ComponentScores(
        demand_growth_score=val("demand_growth_score"),
        capacity_utilization_score=val("capacity_utilization_score"),
        operational_pressure_score=val("operational_pressure_score"),
        expansion_opportunity_score=val("expansion_opportunity_score"),
    )


def contributions_from_row(row: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for comp, weight in EXPANSION_WEIGHTS.items():
        v = row.get(f"{comp}_score")
        if not _is_missing(v):
            out[comp] = round(weight * float(v), 4)
    return out


def row_to_raw(row: pd.Series) -> RawKPIs:
    data: dict[str, Any] = {}
    for name, info in RawKPIs.model_fields.items():
        if name not in row.index or name == "airport_code":
            continue
        v = row[name]
        if name == "quality_flags":
            data[name] = list(v) if isinstance(v, (list, tuple)) else []
            continue
        if _is_missing(v):
            continue
        ann = info.annotation
        if ann is int or "int" in str(ann):
            data[name] = int(round(float(v)))
        else:
            data[name] = float(v)
    data["airport_code"] = str(row["airport_code"])
    return RawKPIs(**data)


def evaluate_condition(raw: RawKPIs, cond: Condition) -> Optional[bool]:
    value = getattr(raw, cond.metric, None)
    if _is_missing(value):
        return None
    v = float(value)
    return {">": v > cond.value, ">=": v >= cond.value, "<": v < cond.value, "<=": v <= cond.value}[cond.op]


@dataclass
class RankingComputation:
    rows: list[RankedAirport]
    exclusions: list[Exclusion]
    peer_n: int
    bounds: dict[str, tuple[float, float]]
    total_ranked: int
    confidence: Confidence
    notes: list[str] = field(default_factory=list)
    scored_frame: Optional[pd.DataFrame] = None


def rank_peer_group(
    kpis: pd.DataFrame,
    refs: dict[str, AirportRef],
    spec: PeerGroupSpec,
    *,
    sort_by: str = "expansion_opportunity_score",
    conditions: Optional[list[Condition]] = None,
    ascending: bool = False,
    top_n: Optional[int] = 10,
    clamped: bool = False,
) -> RankingComputation:
    """Apply coverage rule, normalize within the covered peer group, score, filter, sort."""
    notes: list[str] = []
    raws: dict[str, RawKPIs] = {}
    exclusions: list[Exclusion] = []
    covered_codes: list[str] = []
    for _, row in kpis.iterrows():
        raw = row_to_raw(row)
        raws[raw.airport_code] = raw
        cov = check_coverage(raw, spec.min_reported_flights)
        ref = refs.get(raw.airport_code) or AirportRef(airport_code=raw.airport_code, airport_name=raw.airport_code)
        if cov.passed:
            covered_codes.append(raw.airport_code)
        else:
            exclusions.append(Exclusion(airport=ref, reasons=cov.reasons, detail=cov.detail, raw=raw))
    exclusions.sort(key=lambda e: e.airport.airport_code)
    peer_n = len(covered_codes)
    if peer_n == 0:
        return RankingComputation(rows=[], exclusions=exclusions, peer_n=0, bounds={}, total_ranked=0, confidence="low",
                                  notes=["No airport in the peer group meets the minimum coverage rule."])

    frame = kpis[kpis["airport_code"].isin(covered_codes)].copy().set_index("airport_code", drop=False)
    scored, bounds = normalize_components(frame)
    if peer_n < MIN_PEERS_FOR_SCORES:
        notes.append("Only one airport passes the coverage rule in this peer group; raw KPIs are shown without normalized scores.")
    elif peer_n < SMALL_PEER_GROUP:
        notes.append(f"Small peer group ({peer_n} airports): normalized scores are sensitive to each member; confidence is low.")

    ranked: list[RankedAirport] = []
    dropped_by_condition = 0
    for code, row in scored.iterrows():
        raw = raws[code]
        if conditions:
            verdicts = [evaluate_condition(raw, c) for c in conditions]
            if any(v is None for v in verdicts) or not all(verdicts):
                dropped_by_condition += 1
                continue
        cov = CoverageCheck(passed=True, min_reported_flights=spec.min_reported_flights)
        ranked.append(RankedAirport(
            rank=0,
            airport=refs.get(code) or AirportRef(airport_code=code, airport_name=code),
            raw=raw,
            scores=component_scores_from_row(row),
            contributions=contributions_from_row(row),
            confidence=derive_confidence(cov, raw, peer_n, clamped),
        ))
    if conditions and dropped_by_condition:
        notes.append(f"{dropped_by_condition} covered airports did not meet the screening conditions.")

    def sort_key(item: RankedAirport):
        if sort_by == "expansion_opportunity_score":
            value = item.scores.expansion_opportunity_score
        else:
            value = getattr(item.raw, sort_by, None)
        missing = value is None
        v = float(value) if value is not None else 0.0
        return (missing, -v if not ascending else v, item.airport.airport_code)

    ranked.sort(key=sort_key)
    for i, item in enumerate(ranked, start=1):
        item.rank = i
    total_ranked = len(ranked)
    if top_n is not None:
        ranked = ranked[:top_n]
    conf = aggregate_confidence(r.confidence for r in ranked) if ranked else "low"
    partial = [r.airport.airport_code for r in ranked if months_complete(r.raw)[0] < COMPLETE_MONTHS]
    if partial:
        notes.append(f"{', '.join(partial)} report fewer than 12 months in at least one window (seasonal or partial reporting); their rows carry lower confidence.")
    if exclusions and len(exclusions) > peer_n:
        conf = downgrade(conf)
        notes.append("More airports were excluded for coverage than ranked; confidence downgraded.")
    return RankingComputation(rows=ranked, exclusions=exclusions, peer_n=peer_n, bounds=bounds, total_ranked=total_ranked,
                              confidence=conf, notes=notes, scored_frame=scored)


@dataclass
class SingleAirportScores:
    raw: RawKPIs
    coverage: CoverageCheck
    scores: ComponentScores
    percentiles: dict[str, float]
    capacity_pressure_proxy: Optional[float]
    components: list[CPPComponent]
    peer_n: int
    bounds: dict[str, tuple[float, float]]
    confidence: Confidence
    notes: list[str] = field(default_factory=list)


def score_single_airport(kpis: pd.DataFrame, target: str, spec: PeerGroupSpec, clamped: bool = False) -> Optional[SingleAirportScores]:
    """Component scores, percentile ranks and Capacity Pressure Proxy for `target` within its peer group."""
    if target not in set(kpis["airport_code"]):
        return None
    target_row = kpis[kpis["airport_code"] == target].iloc[0]
    raw_target = row_to_raw(target_row)
    cov = check_coverage(raw_target, spec.min_reported_flights)
    notes: list[str] = []
    if not cov.passed:
        return SingleAirportScores(raw=raw_target, coverage=cov, scores=ComponentScores(), percentiles={}, capacity_pressure_proxy=None,
                                   components=[], peer_n=0, bounds={}, confidence="low", notes=[cov.detail])
    covered = [row_to_raw(r) for _, r in kpis.iterrows()]
    covered_codes = [r.airport_code for r in covered if check_coverage(r, spec.min_reported_flights).passed]
    frame = kpis[kpis["airport_code"].isin(covered_codes)].copy().set_index("airport_code", drop=False)
    peer_n = len(frame)
    scored, bounds = normalize_components(frame)
    percentiles: dict[str, float] = {}
    for comp in COMPONENTS:
        pct = percentile_ranks(frame[comp].astype(float))
        percentiles[comp] = float(pct.loc[target])
    cpp = weighted_sum(percentiles, CPP_WEIGHTS)
    components = [
        CPPComponent(name=comp, label=COMPONENT_LABELS[comp], raw_value=getattr(raw_target, comp), percentile=round(percentiles[comp], 2),
                     weight=CPP_WEIGHTS[comp], contribution=round(CPP_WEIGHTS[comp] * percentiles[comp], 2))
        for comp in CPP_WEIGHTS
    ]
    if peer_n < SMALL_PEER_GROUP:
        notes.append(f"Small peer group ({peer_n} airports); percentiles and scores are coarse.")
    conf = derive_confidence(cov, raw_target, peer_n, clamped)
    return SingleAirportScores(raw=raw_target, coverage=cov, scores=component_scores_from_row(scored.loc[target]), percentiles=percentiles,
                               capacity_pressure_proxy=cpp, components=components, peer_n=peer_n, bounds=bounds, confidence=conf, notes=notes)


# --------------------------------------------------------------------------------------
# Methodology descriptions
# --------------------------------------------------------------------------------------

def methodology(topic: str = "overview") -> Methodology:
    base_thresholds = {
        "long_haul_miles_default": 2000,
        "min_reported_flights_default": DEFAULT_MIN_REPORTED_FLIGHTS,
        "winsorize_percentiles": [WINSOR_LOW_PCT, WINSOR_HIGH_PCT],
        "size_band_annual_passengers": SIZE_BAND_THRESHOLDS,
        "small_peer_group": SMALL_PEER_GROUP,
        "high_confidence_peer_group": HIGH_CONFIDENCE_PEER_GROUP,
    }
    common_rules = [
        "All calculations are deterministic Python/DuckDB; the language model never computes or alters a score.",
        "Historical data (BTS) is the only input to rankings and scores; live data is shown separately and never changes a score.",
        "Airports with fewer than the minimum reported flights in either 12-month window are excluded and listed with the reason.",
        "Components are winsorized at the 5th/95th percentile within the peer group, then min-max normalized to 0-100.",
        "Peer group = airports in the active data scope that match the optional state/region and size-band filters and pass the coverage rule.",
        "Airport size is only a peer-comparison filter; no points are awarded for being large or small.",
        "Scores are screening signals, not ROI, profitability or causal findings.",
    ]
    topics: dict[str, Methodology] = {
        "overview": Methodology(topic="overview", title="Scoring methodology overview",
                                formulas={k: DEFINITIONS[k] for k in ("demand_growth", "capacity_utilization", "operational_pressure", "expansion_opportunity_score", "capacity_pressure_proxy")},
                                weights={"expansion_opportunity_score": EXPANSION_WEIGHTS, "capacity_pressure_proxy": CPP_WEIGHTS, "operational_pressure": OPERATIONAL_PRESSURE_WEIGHTS},
                                thresholds=base_thresholds, rules=common_rules, definitions=DEFINITIONS),
        "expansion_opportunity_score": Methodology(topic="expansion_opportunity_score", title="Expansion Opportunity Score",
                                                   formulas={"expansion_opportunity_score": DEFINITIONS["expansion_opportunity_score"]},
                                                   weights={"expansion_opportunity_score": EXPANSION_WEIGHTS}, thresholds=base_thresholds,
                                                   rules=common_rules[:5], definitions={k: DEFINITIONS[k] for k in ("demand_growth", "capacity_utilization", "operational_pressure")}),
        "capacity_pressure_proxy": Methodology(topic="capacity_pressure_proxy", title="Capacity Pressure Proxy",
                                               formulas={"capacity_pressure_proxy": DEFINITIONS["capacity_pressure_proxy"]},
                                               weights={"capacity_pressure_proxy": CPP_WEIGHTS}, thresholds=base_thresholds,
                                               rules=[UNMET_DEMAND_DISCLAIMER, "Percentiles are Hazen ranks (ties averaged) within the peer group."] + common_rules[2:6],
                                               definitions={k: DEFINITIONS[k] for k in ("capacity_pressure", "capacity_utilization", "demand_growth", "operational_pressure")}),
        "demand_growth": Methodology(topic="demand_growth", title="Demand Growth", formulas={"demand_growth": DEFINITIONS["demand_growth"]},
                                     thresholds=base_thresholds, rules=["Passengers are T-100 origin passengers for scheduled service (class F) at U.S. airports."],
                                     definitions={"demand_growth": DEFINITIONS["demand_growth"]}),
        "capacity_utilization": Methodology(topic="capacity_utilization", title="Capacity Utilization", formulas={"capacity_utilization": DEFINITIONS["capacity_utilization"]},
                                            thresholds=base_thresholds, rules=["Seats are T-100 available seats on departing segments; values above 1.05 are flagged as data-quality issues."],
                                            definitions={"capacity_utilization": DEFINITIONS["capacity_utilization"]}),
        "operational_pressure": Methodology(topic="operational_pressure", title="Operational Pressure", formulas={"operational_pressure": DEFINITIONS["operational_pressure"]},
                                            weights={"operational_pressure": OPERATIONAL_PRESSURE_WEIGHTS}, thresholds=base_thresholds,
                                            rules=["Delay rate = departures delayed 15+ minutes / operated departures; cancellation rate = cancelled / scheduled flights (BTS On-Time, reporting carriers)."],
                                            definitions={k: DEFINITIONS[k] for k in ("delayed_departure", "airport_congestion", "operational_pressure")}),
        "long_haul": Methodology(topic="long_haul", title="Long-haul route share", formulas={"long_haul_share": "long-haul departures / total departures over the latest 12 months"},
                                 thresholds=base_thresholds, rules=["Route distance comes from T-100 segment DISTANCE (statute miles); the threshold is configurable (default 2,000)."],
                                 definitions={"long_haul_flight": DEFINITIONS["long_haul_flight"]}),
        "size_band": Methodology(topic="size_band", title="Airport size band", formulas={"size_band": "FAA-style band from origin passengers in the latest 12 months"},
                                 thresholds={"size_band_annual_passengers": SIZE_BAND_THRESHOLDS}, rules=["Size is a peer filter only; it never adds or removes points."],
                                 definitions={"airport_size": DEFINITIONS["airport_size"]}),
        "peer_group_and_normalization": Methodology(topic="peer_group_and_normalization", title="Peer groups and normalization", thresholds=base_thresholds,
                                                    rules=common_rules[2:6] + ["All-equal components map to 50; a single-airport peer group yields raw KPIs without scores."]),
        "coverage_and_confidence": Methodology(topic="coverage_and_confidence", title="Coverage rule and confidence", thresholds=base_thresholds,
                                               rules=["Coverage: at least the minimum reported flights (default 500) in BOTH 12-month windows for On-Time flights and T-100 departures.",
                                                      "Confidence high: coverage ok, 12/12 months in all windows, peer group >= 10, no data-quality flags, period not clamped.",
                                                      "Confidence medium: 10-11 months in a window, peer group 5-9, a quality flag, or a clamped period.",
                                                      "Confidence low: fewer than 10 months, peer group < 5, or any non-ok status."]),
        "data_sources": Methodology(topic="data_sources", title="Data sources",
                                    rules=["BTS TranStats T-100 Segment (All Carriers): passengers, seats, departures, distance (monthly, U.S. origins, scheduled service).",
                                           "BTS TranStats Reporting Carrier On-Time Performance: departures, 15-minute delays, cancellations (reporting carriers only).",
                                           "OurAirports airports.csv: airport names, locations and types (public domain).",
                                           "Aviationstack (optional, live): current departures status; never used in scores."]),
    }
    return topics.get(topic, topics["overview"])


__all__ = [
    "EXPANSION_WEIGHTS", "CPP_WEIGHTS", "OPERATIONAL_PRESSURE_WEIGHTS", "DEFINITIONS", "UNMET_DEMAND_DISCLAIMER", "CONGESTION_NOTE",
    "EXCLUSION_LABELS", "COMPONENTS", "COMPONENT_LABELS", "safe_ratio", "operational_pressure", "demand_growth", "winsorize",
    "minmax_0_100", "percentile_ranks", "weighted_sum", "check_coverage", "derive_confidence", "downgrade", "min_confidence",
    "normalize_components", "row_to_raw", "rank_peer_group", "RankingComputation", "score_single_airport", "SingleAirportScores",
    "evaluate_condition", "methodology",
]
