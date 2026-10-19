"""Phase 3: analytics over the synthetic fixture database (periods, coverage, KPIs, routes, series)."""
from __future__ import annotations

from datetime import date

import pytest

from src import analytics as an
from src import scoring as sc
from src.models import AnalysisPeriod, PeerGroupSpec, PeriodRequest


def test_analysis_period_windows():
    p = AnalysisPeriod(end_month="2024-12")
    assert (p.latest_start, p.latest_end) == (date(2024, 1, 1), date(2024, 12, 31))
    assert (p.prior_start, p.prior_end) == (date(2023, 1, 1), date(2023, 12, 31))
    assert p.calendar_year == 2024 and p.label == "calendar 2024 vs calendar 2023"
    q = AnalysisPeriod(end_month="2025-06")
    assert (q.latest_start, q.latest_end) == (date(2024, 7, 1), date(2025, 6, 30))
    assert (q.prior_start, q.prior_end) == (date(2023, 7, 1), date(2024, 6, 30))
    assert q.calendar_year is None and "2024-07 to 2025-06" in q.label
    assert q.data_period().start == date(2023, 7, 1) and q.data_period().end == date(2025, 6, 30)


def test_coverage_from_fixture(fixture_conn):
    cov = an.get_coverage(fixture_conn)
    assert (cov.traffic_first, cov.traffic_last) == ("2023-01", "2025-12")
    assert (cov.ops_first, cov.ops_last) == ("2023-01", "2026-02")
    assert cov.common_end_month == "2025-12" and cov.earliest_valid_end_month == "2024-12"
    assert cov.ready and len(cov.available_end_months()) == 13
    assert cov.available_calendar_years() == [2024, 2025]
    assert cov.data_scope == "demo" and cov.refresh_timestamp.startswith("2026-10-01")
    assert any("operations through 2026-02" in n for n in cov.notes)
    assert len(cov.sources) == 3


def test_latest_complete_month_guard():
    totals = {f"2025-{m:02d}": 1000.0 for m in range(1, 13)}
    totals["2026-01"] = 90.0            # partial load
    assert an.latest_complete_month(totals) == "2025-12"
    totals["2026-01"] = 600.0           # >= 50 % of median -> complete
    assert an.latest_complete_month(totals) == "2026-01"
    assert an.latest_complete_month({"2025-01": 5.0, "2025-02": 1.0}) == "2025-02"
    assert an.latest_complete_month({}) is None


@pytest.mark.parametrize(
    "text,expected,notes",
    [
        ("What changed at Boston Logan in 2024?", PeriodRequest(year=2024), []),
        ("compare 2024 vs 2023", PeriodRequest(year=2024), []),
        ("2024 compared to 2022", PeriodRequest(year=2024), ["non_adjacent_years"]),
        ("last year", PeriodRequest(year=2025), []),
        ("March 2024 delays", PeriodRequest(end_month="2024-03"), []),
        ("Q1 2024", PeriodRequest(end_month="2024-03"), []),
        ("as of 2025-06", PeriodRequest(end_month="2025-06"), []),
        ("year to date", None, ["year_to_date"]),
        ("Compare LAX and SNA congestion", None, []),
    ],
)
def test_parse_period_phrase(text, expected, notes, today):
    req, spans, got_notes = an.parse_period_phrase(text, today)
    assert req == expected and got_notes == notes
    if expected is not None:
        assert spans


def test_resolve_period_default_clamp_and_too_early(fixture_conn):
    cov = an.get_coverage(fixture_conn)
    p, notes = an.resolve_period(None, None, cov)
    assert p.end_month == "2025-12" and not p.clamped and notes == []
    p, notes = an.resolve_period(PeriodRequest(year=2024), None, cov)
    assert p.end_month == "2024-12" and p.calendar_year == 2024 and p.requested == "calendar 2024"
    p, notes = an.resolve_period(PeriodRequest(year=2026), None, cov)
    assert p.end_month == "2025-12" and p.clamped and "available through 2025-12" in notes[0]
    p, notes = an.resolve_period(PeriodRequest(year=2023), None, cov)
    assert p is None and "coverage starts" in notes[0]
    p, notes = an.resolve_period(PeriodRequest(end_month="2025-06"), None, cov)
    assert p.end_month == "2025-06" and p.requested == "12 months ending 2025-06"
    p, notes = an.resolve_period(None, "2025-03", cov)       # sidebar default
    assert p.end_month == "2025-03"


def test_raw_kpis_bos_calendar_2024(fixture_conn):
    df = an.compute_raw_kpis(fixture_conn, AnalysisPeriod(end_month="2024-12"), ["BOS"])
    raw = sc.row_to_raw(df.iloc[0])
    assert raw.pax_prior == 1_500_000 * 12 and raw.pax_latest == 1_575_000 * 12
    assert raw.demand_growth == pytest.approx(0.05)
    assert raw.capacity_utilization == pytest.approx(0.84, abs=1e-4)
    assert raw.delay_rate == pytest.approx(0.20) and raw.delay_rate_prior == pytest.approx(0.1875)
    assert raw.cancellation_rate == pytest.approx(0.0125) and raw.cancellation_rate_prior == pytest.approx(0.016)
    assert raw.delay_rate_change == pytest.approx(0.0125) and raw.cancellation_rate_change == pytest.approx(-0.0035)
    assert raw.operational_pressure == pytest.approx(0.7 * 0.20 + 0.3 * 0.0125)
    assert raw.flights_latest == 12_000 * 12 and raw.operated_latest == 11_850 * 12
    assert (raw.traffic_months_latest, raw.traffic_months_prior, raw.ops_months_latest, raw.ops_months_prior) == (12, 12, 12, 12)
    assert sc.check_coverage(raw).passed


def test_raw_kpis_lax_and_missing_month(fixture_conn):
    period = AnalysisPeriod(end_month="2025-12")
    df = an.compute_raw_kpis(fixture_conn, period, ["LAX", "AUS", "TNY"]).set_index("airport_code", drop=False)
    lax = sc.row_to_raw(df.loc["LAX"])
    assert lax.delay_rate == pytest.approx(0.20) and lax.cancellation_rate == pytest.approx(0.02)
    assert lax.avg_departure_delay_minutes == pytest.approx(7.6)
    assert lax.demand_growth == pytest.approx(0.02)
    aus = sc.row_to_raw(df.loc["AUS"])
    assert aus.traffic_months_latest == 11 and aus.traffic_months_prior == 12
    assert sc.derive_confidence(sc.check_coverage(aus), aus, peer_n=20) == "medium"
    tny = sc.row_to_raw(df.loc["TNY"])
    assert tny.flights_latest == 360 and tny.departures_latest == 240
    cov = sc.check_coverage(tny)
    assert not cov.passed and set(cov.reasons) == {"insufficient_operations_coverage", "insufficient_traffic_coverage"}


def test_raw_kpis_all_airports_and_empty_filter(fixture_conn):
    period = AnalysisPeriod(end_month="2025-12")
    df = an.compute_raw_kpis(fixture_conn, period)
    assert len(df) == 21 and df["airport_code"].is_monotonic_increasing
    assert an.compute_raw_kpis(fixture_conn, period, []).empty
    assert an.raw_for_airport(fixture_conn, "ZZZ", period) is None


def test_route_mix_anc_thresholds(fixture_conn):
    period = AnalysisPeriod(end_month="2025-12")
    mix = an.route_mix(fixture_conn, "ANC", period.latest_start, period.latest_end, 2000)
    assert (mix.long_haul_departures, mix.total_departures) == (1200, 4800)
    assert mix.share == pytest.approx(0.25) and mix.months_covered == 12
    assert [r.destination_airport for r in mix.top_routes] == ["SLC", "ORD", "HNL", "DFW"]
    assert an.route_mix(fixture_conn, "ANC", period.latest_start, period.latest_end, 1400).share == pytest.approx(0.60)
    assert an.route_mix(fixture_conn, "ANC", period.latest_start, period.latest_end, 2001).share == pytest.approx(0.15)
    empty = an.route_mix(fixture_conn, "SNA", period.latest_start, period.latest_end, 2000)
    assert empty.total_departures == 0 and empty.share is None


def test_monthly_series_and_profile_changes(fixture_conn):
    period = AnalysisPeriod(end_month="2025-12")
    series = an.monthly_series(fixture_conn, "BOS", period)
    assert len(series) == 24 and series[0].year_month == "2024-01" and series[-1].year_month == "2025-12"
    assert series[-1].passengers == 1_606_500 and series[-1].delay_rate == pytest.approx(0.20)
    raw = an.raw_for_airport(fixture_conn, "BOS", AnalysisPeriod(end_month="2024-12"))
    changes = {c.metric: c for c in an.profile_changes(raw, long_haul_latest=0.2, long_haul_prior=0.2)}
    assert changes["passengers"].direction == "up" and changes["passengers"].pct_change == pytest.approx(0.05)
    assert changes["cancellation_rate"].direction == "down" and changes["delay_rate"].direction == "up"
    assert changes["long_haul_share"].direction == "flat"


def test_peer_codes_filters(fixture_conn):
    assert an.peer_codes(fixture_conn, PeerGroupSpec(data_scope="demo", region="new_england")) == ["BDL", "BOS", "BTV", "MHT", "PVD", "PWM", "TNY"]
    assert an.peer_codes(fixture_conn, PeerGroupSpec(data_scope="demo", states=["TX"], size_bands=["medium"])) == ["AUS", "DAL", "HOU"]
    assert len(an.peer_codes(fixture_conn, PeerGroupSpec(data_scope="demo"))) == 21


def test_new_england_ranking_on_fixture(fixture_conn):
    spec = PeerGroupSpec(data_scope="demo", region="new_england")
    codes = an.peer_codes(fixture_conn, spec)
    kpis = an.compute_raw_kpis(fixture_conn, AnalysisPeriod(end_month="2025-12"), codes)
    comp = sc.rank_peer_group(kpis, an.airport_refs(fixture_conn), spec)
    assert comp.peer_n == 6 and [e.airport.airport_code for e in comp.exclusions] == ["TNY"]
    assert [r.airport.airport_code for r in comp.rows][:1] == ["BDL"]      # highest growth + high utilization + high pressure
    assert all(0 <= r.scores.expansion_opportunity_score <= 100 for r in comp.rows)
    assert comp.confidence == "medium"


def test_write_default_scores_snapshot(fresh_conn):
    n = an.write_default_scores_snapshot(fresh_conn, "demo")
    assert n == 21
    rows = fresh_conn.execute("SELECT airport_code, data_quality_status, expansion_opportunity_score FROM airport_scores ORDER BY airport_code").fetchall()
    status = {r[0]: r[1] for r in rows}
    assert status["TNY"].startswith("insufficient_data") and status["LAX"] == "ok"
    assert all(r[2] is None for r in rows if r[0] == "TNY")
    assert an.write_default_scores_snapshot(fresh_conn, "demo") == 21      # idempotent
    assert fresh_conn.execute("SELECT COUNT(*) FROM airport_scores").fetchone()[0] == 21
