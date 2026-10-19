"""Phase 3: deterministic scoring math on hand-built inputs (no database)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import scoring as sc
from src.models import AirportRef, Condition, CoverageCheck, PeerGroupSpec, RawKPIs


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def kpi_row(code, growth, util, delay, cancel, *, flights=12_000, departures=12_000, months=12, prior_delay=None, prior_cancel=None,
            pax_prior=1_000_000, flags=None):
    prior_delay = delay if prior_delay is None else prior_delay
    prior_cancel = cancel if prior_cancel is None else prior_cancel
    pax_latest = pax_prior * (1 + growth)
    cancelled = round(flights * cancel)
    operated = flights - cancelled
    cancelled_p = round(flights * prior_cancel)
    operated_p = flights - cancelled_p
    row = {
        "airport_code": code,
        "pax_latest": pax_latest, "pax_prior": pax_prior,
        "seats_latest": pax_latest / util if util else 0, "seats_prior": pax_prior / util if util else 0,
        "departures_latest": departures, "departures_prior": departures,
        "flights_latest": flights, "flights_prior": flights,
        "operated_latest": operated, "operated_prior": operated_p,
        "delayed_latest": operated * delay, "delayed_prior": operated_p * prior_delay,
        "cancelled_latest": cancelled, "cancelled_prior": cancelled_p,
        "delay_minutes_latest": operated * 10.0, "delay_minutes_prior": operated_p * 10.0,
        "traffic_months_latest": months, "traffic_months_prior": months, "ops_months_latest": months, "ops_months_prior": months,
        "demand_growth": growth, "capacity_utilization": util, "capacity_utilization_prior": util,
        "delay_rate": delay, "delay_rate_prior": prior_delay, "cancellation_rate": cancel, "cancellation_rate_prior": prior_cancel,
        "avg_departure_delay_minutes": 10.0, "avg_departure_delay_minutes_prior": 10.0,
        "operational_pressure": sc.operational_pressure(delay, cancel), "operational_pressure_prior": sc.operational_pressure(prior_delay, prior_cancel),
        "delay_rate_change": delay - prior_delay, "cancellation_rate_change": cancel - prior_cancel,
        "quality_flags": flags or [],
    }
    return row


def frame(*rows):
    return pd.DataFrame(list(rows))


def refs_for(codes, bands=None):
    bands = bands or {}
    return {c: AirportRef(airport_code=c, airport_name=f"{c} Airport", size_band=bands.get(c, "large")) for c in codes}


SPEC = PeerGroupSpec(data_scope="demo", min_reported_flights=500)


# --------------------------------------------------------------------------------------
# weights and elementary functions
# --------------------------------------------------------------------------------------

def test_weights_sum_to_one():
    assert sum(sc.EXPANSION_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(sc.CPP_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(sc.OPERATIONAL_PRESSURE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sc.EXPANSION_WEIGHTS == {"demand_growth": 0.40, "capacity_utilization": 0.35, "operational_pressure": 0.25}
    assert sc.CPP_WEIGHTS == {"capacity_utilization": 0.50, "demand_growth": 0.30, "operational_pressure": 0.20}


def test_elementary_formulas_and_zero_denominators():
    assert sc.safe_ratio(1, 0) is None and sc.safe_ratio(None, 5) is None and sc.safe_ratio(3, 4) == 0.75
    assert sc.demand_growth(110, 100) == pytest.approx(0.10)
    assert sc.demand_growth(5, 0) is None
    assert sc.operational_pressure(0.20, 0.02) == pytest.approx(0.7 * 0.20 + 0.3 * 0.02)
    assert sc.operational_pressure(None, 0.02) is None
    assert sc.weighted_sum({"a": 1.0, "b": None}, {"a": 0.5, "b": 0.5}) is None


def test_winsorize_clips_known_vector():
    values = pd.Series(np.arange(1, 21, dtype=float))
    clipped, (lo, hi) = sc.winsorize(values)
    assert lo == pytest.approx(1.95) and hi == pytest.approx(19.05)
    assert clipped.min() == pytest.approx(1.95) and clipped.max() == pytest.approx(19.05)
    assert clipped.iloc[9] == 10.0


def test_minmax_and_percentiles():
    assert sc.minmax_0_100(pd.Series([3.0, 3.0, 3.0])).tolist() == [50.0, 50.0, 50.0]
    assert sc.minmax_0_100(pd.Series([0.0, 5.0, 10.0])).tolist() == [0.0, 50.0, 100.0]
    assert sc.percentile_ranks(pd.Series([10.0, 20.0, 30.0, 40.0])).tolist() == [12.5, 37.5, 62.5, 87.5]
    ties = sc.percentile_ranks(pd.Series([1.0, 1.0, 2.0])).tolist()
    assert ties[0] == pytest.approx(ties[1]) == pytest.approx(100 * (1.5 - 0.5) / 3)
    assert ties[2] == pytest.approx(100 * (3 - 0.5) / 3)
    assert sc.percentile_ranks(pd.Series([7.0])).tolist() == [50.0]


# --------------------------------------------------------------------------------------
# coverage and confidence
# --------------------------------------------------------------------------------------

def _raw(**kw) -> RawKPIs:
    base = dict(airport_code="AAA", pax_latest=1000, pax_prior=900, seats_latest=1200, seats_prior=1100, departures_latest=600,
                departures_prior=600, flights_latest=600, flights_prior=600, operated_latest=590, operated_prior=590,
                delayed_latest=100, delayed_prior=90, cancelled_latest=10, cancelled_prior=10,
                traffic_months_latest=12, traffic_months_prior=12, ops_months_latest=12, ops_months_prior=12,
                demand_growth=0.1, capacity_utilization=0.83, delay_rate=0.17, cancellation_rate=0.017, operational_pressure=0.124)
    base.update(kw)
    return RawKPIs(**base)


@pytest.mark.parametrize("latest,prior,passes", [(499, 600, False), (500, 500, True), (501, 499, False), (600, 600, True)])
def test_coverage_boundaries_operations(latest, prior, passes):
    cov = sc.check_coverage(_raw(flights_latest=latest, flights_prior=prior), 500)
    assert cov.passed is passes
    if not passes:
        assert "insufficient_operations_coverage" in cov.reasons and "500" in cov.detail


def test_coverage_other_reasons():
    assert "insufficient_traffic_coverage" in sc.check_coverage(_raw(departures_prior=499)).reasons
    assert "zero_prior_passengers" in sc.check_coverage(_raw(pax_prior=0, demand_growth=None)).reasons
    assert "zero_seats" in sc.check_coverage(_raw(seats_latest=0, capacity_utilization=None)).reasons
    assert "no_operated_flights" in sc.check_coverage(_raw(operated_latest=0, delay_rate=None, operational_pressure=None)).reasons
    assert sc.check_coverage(_raw(operational_pressure=None)).reasons == ["missing_component"]
    assert sc.check_coverage(_raw(), 300).passed


def test_confidence_truth_table():
    ok = CoverageCheck(passed=True)
    assert sc.derive_confidence(ok, _raw(), peer_n=10) == "high"
    assert sc.derive_confidence(ok, _raw(traffic_months_prior=11), peer_n=10) == "medium"
    assert sc.derive_confidence(ok, _raw(), peer_n=5) == "medium"
    assert sc.derive_confidence(ok, _raw(quality_flags=["utilization 1.1"]), peer_n=10) == "medium"
    assert sc.derive_confidence(ok, _raw(), peer_n=10, clamped=True) == "medium"
    assert sc.derive_confidence(ok, _raw(ops_months_latest=9), peer_n=10) == "low"
    assert sc.derive_confidence(ok, _raw(), peer_n=4) == "low"
    assert sc.derive_confidence(CoverageCheck(passed=False, reasons=["x"]), _raw(), peer_n=10) == "low"
    assert sc.min_confidence(["high", "medium", "low"]) == "low" and sc.downgrade("high") == "medium"


# --------------------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------------------

def test_rank_two_airports_exact_scores():
    kpis = frame(kpi_row("AAA", 0.10, 0.90, 0.10, 0.01), kpi_row("BBB", 0.00, 0.70, 0.30, 0.03))
    comp = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB"]), SPEC)
    assert comp.peer_n == 2 and comp.exclusions == []
    a, b = comp.rows
    assert (a.airport.airport_code, a.rank) == ("AAA", 1) and (b.airport.airport_code, b.rank) == ("BBB", 2)
    assert a.scores.demand_growth_score == 100 and a.scores.capacity_utilization_score == 100 and a.scores.operational_pressure_score == 0
    assert a.scores.expansion_opportunity_score == pytest.approx(75.0)
    assert b.scores.expansion_opportunity_score == pytest.approx(25.0)
    assert a.contributions == {"demand_growth": 40.0, "capacity_utilization": 35.0, "operational_pressure": 0.0}
    assert set(comp.bounds) == set(sc.COMPONENTS)
    assert a.raw.pax_prior == 1_000_000 and a.raw.delay_rate == pytest.approx(0.10)
    assert comp.confidence == "low"              # peer group of 2


def test_rank_excludes_insufficient_coverage_with_reason():
    kpis = frame(kpi_row("AAA", 0.1, 0.9, 0.1, 0.01), kpi_row("BBB", 0.0, 0.7, 0.3, 0.03), kpi_row("TNY", 0.2, 0.95, 0.05, 0.0, flights=400, departures=240))
    comp = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "TNY"]), SPEC)
    assert [r.airport.airport_code for r in comp.rows] == ["AAA", "BBB"]
    assert len(comp.exclusions) == 1
    ex = comp.exclusions[0]
    assert ex.airport.airport_code == "TNY"
    assert set(ex.reasons) == {"insufficient_operations_coverage", "insufficient_traffic_coverage"}
    assert "400/400" in ex.detail and ex.raw.flights_latest == 400


def test_rank_sort_by_raw_metric_and_tiebreak():
    kpis = frame(kpi_row("CCC", 0.05, 0.8, 0.2, 0.02), kpi_row("AAA", 0.05, 0.85, 0.1, 0.01), kpi_row("BBB", 0.05, 0.9, 0.3, 0.03))
    comp = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "CCC"]), SPEC, sort_by="demand_growth")
    assert [r.airport.airport_code for r in comp.rows] == ["AAA", "BBB", "CCC"]
    comp2 = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "CCC"]), SPEC, sort_by="delay_rate", ascending=True)
    assert [r.airport.airport_code for r in comp2.rows] == ["AAA", "CCC", "BBB"]
    assert all(r.scores.demand_growth_score == 50.0 for r in comp.rows)      # all-equal component -> 50


def test_conditions_filter_after_normalization():
    kpis = frame(
        kpi_row("AAA", 0.10, 0.9, 0.20, 0.020, prior_cancel=0.010),    # growing, worsening cancellations
        kpi_row("BBB", 0.05, 0.8, 0.20, 0.010, prior_cancel=0.020),    # growing, improving cancellations
        kpi_row("CCC", -0.02, 0.7, 0.20, 0.030, prior_cancel=0.010),   # shrinking
    )
    conds = [Condition(metric="demand_growth", op=">", value=0), Condition(metric="cancellation_rate_change", op=">", value=0)]
    comp = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "CCC"]), SPEC, sort_by="demand_growth", conditions=conds)
    assert [r.airport.airport_code for r in comp.rows] == ["AAA"]
    assert comp.peer_n == 3 and comp.total_ranked == 1
    assert any("did not meet the screening conditions" in n for n in comp.notes)
    # normalization happened over the whole peer group, not the filtered subset
    assert comp.rows[0].scores.demand_growth_score == 100.0


def test_size_band_never_changes_scores():
    kpis = frame(kpi_row("AAA", 0.1, 0.9, 0.1, 0.01), kpi_row("BBB", 0.0, 0.7, 0.3, 0.03), kpi_row("CCC", 0.05, 0.8, 0.2, 0.02))
    c1 = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "CCC"], {"AAA": "large", "BBB": "small", "CCC": "medium"}), SPEC)
    c2 = sc.rank_peer_group(kpis, refs_for(["AAA", "BBB", "CCC"], {"AAA": "non_hub", "BBB": "large", "CCC": "small"}), SPEC)
    assert [(r.airport.airport_code, r.scores.expansion_opportunity_score) for r in c1.rows] == \
           [(r.airport.airport_code, r.scores.expansion_opportunity_score) for r in c2.rows]


def test_determinism_under_row_shuffle():
    rows = [kpi_row(c, g, u, d, k) for c, g, u, d, k in [("AAA", 0.1, 0.9, 0.1, 0.01), ("BBB", 0.0, 0.7, 0.3, 0.03), ("CCC", 0.05, 0.8, 0.2, 0.02), ("DDD", 0.07, 0.75, 0.25, 0.015)]]
    r1 = sc.rank_peer_group(frame(*rows), refs_for(["AAA", "BBB", "CCC", "DDD"]), SPEC)
    r2 = sc.rank_peer_group(frame(*reversed(rows)), refs_for(["AAA", "BBB", "CCC", "DDD"]), SPEC)
    assert [(r.airport.airport_code, round(r.scores.expansion_opportunity_score, 9)) for r in r1.rows] == \
           [(r.airport.airport_code, round(r.scores.expansion_opportunity_score, 9)) for r in r2.rows]


def test_single_airport_peer_group_has_raw_only():
    kpis = frame(kpi_row("AAA", 0.1, 0.9, 0.1, 0.01), kpi_row("TNY", 0.2, 0.95, 0.05, 0.0, flights=100, departures=100))
    comp = sc.rank_peer_group(kpis, refs_for(["AAA", "TNY"]), SPEC)
    assert comp.peer_n == 1 and len(comp.rows) == 1
    assert comp.rows[0].scores.expansion_opportunity_score is None
    assert comp.rows[0].raw.demand_growth == pytest.approx(0.1)
    assert any("without normalized scores" in n for n in comp.notes)


def test_no_covered_airports():
    comp = sc.rank_peer_group(frame(kpi_row("TNY", 0.2, 0.95, 0.05, 0.0, flights=100, departures=100)), refs_for(["TNY"]), SPEC)
    assert comp.rows == [] and comp.peer_n == 0 and comp.exclusions[0].airport.airport_code == "TNY"


# --------------------------------------------------------------------------------------
# single airport / capacity pressure proxy
# --------------------------------------------------------------------------------------

def test_score_single_airport_capacity_pressure_proxy():
    kpis = frame(kpi_row("AAA", 0.10, 0.90, 0.10, 0.01), kpi_row("BBB", 0.00, 0.70, 0.30, 0.03),
                 kpi_row("CCC", 0.05, 0.80, 0.20, 0.02), kpi_row("DDD", 0.07, 0.75, 0.25, 0.015))
    res = sc.score_single_airport(kpis, "AAA", SPEC)
    assert res is not None and res.coverage.passed and res.peer_n == 4
    # AAA: highest utilization (87.5), highest growth (87.5), lowest pressure (12.5)
    assert res.percentiles["capacity_utilization"] == 87.5 and res.percentiles["demand_growth"] == 87.5 and res.percentiles["operational_pressure"] == 12.5
    assert res.capacity_pressure_proxy == pytest.approx(0.5 * 87.5 + 0.3 * 87.5 + 0.2 * 12.5)
    assert [c.name for c in res.components] == ["capacity_utilization", "demand_growth", "operational_pressure"]
    assert sum(c.weight for c in res.components) == pytest.approx(1.0)
    assert sum(c.contribution for c in res.components) == pytest.approx(res.capacity_pressure_proxy, abs=0.02)
    assert res.scores.expansion_opportunity_score == pytest.approx(75.0)
    assert res.confidence == "low"          # peer group < 5

    failing = sc.score_single_airport(frame(kpi_row("AAA", 0.1, 0.9, 0.1, 0.01, flights=100)), "AAA", SPEC)
    assert failing is not None and not failing.coverage.passed and failing.capacity_pressure_proxy is None
    assert sc.score_single_airport(kpis, "ZZZ", SPEC) is None


def test_methodology_topics():
    overview = sc.methodology("overview")
    assert overview.weights["expansion_opportunity_score"] == sc.EXPANSION_WEIGHTS
    assert "capacity_pressure_proxy" in overview.formulas
    cpp = sc.methodology("capacity_pressure_proxy")
    assert any("unmet demand" in r for r in cpp.rules)
    assert sc.methodology("unknown-topic").topic == "overview"
    for key in ("delayed_departure", "long_haul_flight", "new_england", "airport_congestion", "capacity_pressure", "airport_size"):
        assert key in sc.DEFINITIONS
