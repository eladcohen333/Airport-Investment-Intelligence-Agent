"""Phase 2: ingestion adapters and aggregators tested on BTS-shaped fixture slices (no network)."""
from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from src import data_ingestion as ing

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------------------
# On-Time Performance aggregation
# --------------------------------------------------------------------------------------

def test_aggregate_otp_sample_counts_and_rates():
    df = ing.aggregate_otp_csv(FIXTURES / "fixture_bts_otp_sample.csv", "2026-07")
    by = df.set_index("airport_code")
    lax, sna = by.loc["LAX"], by.loc["SNA"]
    assert lax["total_flights"] == 5 and lax["cancelled_flights"] == 1 and lax["operated_flights"] == 4
    assert lax["delayed_departures_15m"] == 2
    assert lax["delay_rate"] == pytest.approx(0.5)
    assert lax["cancellation_rate"] == pytest.approx(0.2)
    assert lax["dep_delay_minutes_sum"] == pytest.approx(5 + 30 + 50 + 0)
    assert lax["avg_departure_delay_minutes"] == pytest.approx(85 / 4)
    assert sna["total_flights"] == 3 and sna["cancelled_flights"] == 0 and sna["delayed_departures_15m"] == 1
    assert sna["delay_rate"] == pytest.approx(1 / 3)
    assert sna["diverted_flights"] == 1
    assert df["year_month"].iloc[0] == date(2026, 7, 1)
    assert set(df["source"]) == {ing.SOURCE_OTP}


def test_aggregate_otp_rejects_wrong_month():
    with pytest.raises(ing.DataValidationError, match="different Year/Month"):
        ing.aggregate_otp_csv(FIXTURES / "fixture_bts_otp_sample.csv", "2026-06")


def test_otp_adapter_rejects_missing_columns(tmp_path):
    df = pd.read_csv(FIXTURES / "fixture_bts_otp_sample.csv")
    df = df.drop(columns=["DepDel15"])
    csv_path = tmp_path / "bad.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, "bad.csv")
    adapter = ing.BtsOnTimeAdapter(ing.HttpClient(session=requests.Session()), tmp_path)
    with pytest.raises(ing.DataValidationError, match="missing columns"):
        adapter.aggregate_month(zip_path, "2026-07", work_dir=tmp_path / "work")


def test_otp_adapter_aggregates_from_zip(tmp_path):
    zip_path = tmp_path / "otp_2026_07.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(FIXTURES / "fixture_bts_otp_sample.csv", "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_7.csv")
    adapter = ing.BtsOnTimeAdapter(ing.HttpClient(session=requests.Session()), tmp_path)
    df = adapter.aggregate_month(zip_path, "2026-07")
    assert len(df) == 2
    assert adapter.month_url("2026-07").endswith("_2026_7.zip")
    assert adapter.cache_path("2026-07").name == "otp_2026_07.zip"


def test_national_volume_validation():
    df = pd.DataFrame({"total_flights": [100, 200]})
    with pytest.raises(ing.DataValidationError, match="national flight count"):
        ing.validate_national_volume(df, "2026-07")
    ing.validate_national_volume(pd.DataFrame({"total_flights": [400_000]}), "2026-07")


# --------------------------------------------------------------------------------------
# T-100 aggregation
# --------------------------------------------------------------------------------------

def test_aggregate_t100_sample_filters_and_aggregates():
    res = ing.aggregate_t100_csv(FIXTURES / "fixture_bts_t100_sample.csv", 2025)
    assert res.raw_rows == 8
    assert res.months == ["2025-01", "2025-02"]
    traffic = res.traffic.set_index(["airport_code", "year_month"])
    anc_jan = traffic.loc[("ANC", date(2025, 1, 1))]
    # class F US-origin rows only: AS+DL SEA and UA ORD; cargo (G) and the Tokyo-origin row are excluded
    assert anc_jan["passengers"] == 3800 + 2400 + 300
    assert anc_jan["available_seats"] == 4500 + 3000 + 600
    assert anc_jan["departures_performed"] == 30 + 20 + 4
    assert anc_jan["departures_scheduled"] == 31 + 20 + 5
    assert anc_jan["load_factor"] == pytest.approx(6500 / 8100)
    lax_jan = traffic.loc[("LAX", date(2025, 1, 1))]
    assert lax_jan["passengers"] == 17000           # class L charter row excluded
    assert "NRT" not in res.traffic["airport_code"].values

    routes = res.routes.set_index(["origin_airport", "destination_airport", "year_month"])
    sea = routes.loc[("ANC", "SEA", date(2025, 1, 1))]
    assert sea["departures_performed"] == 50 and sea["distance_miles"] == 1448.0 and not sea["is_long_haul"]
    ord_ = routes.loc[("ANC", "ORD", date(2025, 1, 1))]
    assert bool(ord_["is_long_haul"]) is True
    assert ("ANC", "HKG", date(2025, 1, 1)) not in routes.index

    refs = res.airport_refs.set_index("airport_code")
    assert refs.loc["ANC", "city"] == "Anchorage" and refs.loc["ANC", "state"] == "AK"
    assert "NRT" not in refs.index


def test_aggregate_t100_rejects_wrong_year():
    with pytest.raises(ing.DataValidationError, match="different YEAR"):
        ing.aggregate_t100_csv(FIXTURES / "fixture_bts_t100_sample.csv", 2024)


def test_t100_form_build_from_fixture_page():
    html = (FIXTURES / "fixture_transtats_form_sample.html").read_text(encoding="utf-8")
    adapter = ing.BtsT100Adapter(ing.HttpClient(session=requests.Session()), Path("."))
    form = adapter.build_form(html, 2025)
    assert form["__VIEWSTATE"] == "FIXTUREVIEWSTATE=="
    assert form["__EVENTVALIDATION"] == "FIXTUREEVENTVALIDATION=="
    assert form["cboYear"] == "2025" and form["cboPeriod"] == "All" and form["cboGeography"] == "All"
    assert form["chkDownloadZip"] == "on" and form["btnDownload"] == "Download"
    for var in ing.T100_VARIABLES:
        assert form[var] == "on"
    with pytest.raises(ing.SourceError, match="__VIEWSTATE"):
        adapter.build_form(html.replace('name="__VIEWSTATE"', 'name="__GONE"'), 2025)
    with pytest.raises(ing.SourceError, match="missing expected variables"):
        adapter.build_form(html.replace('name="SEATS"', 'name="SEATZ"'), 2025)


# --------------------------------------------------------------------------------------
# Airport reference assembly
# --------------------------------------------------------------------------------------

def test_load_ourairports_sample():
    df = ing.load_ourairports_csv(FIXTURES / "fixture_ourairports_sample.csv")
    assert set(df["airport_code"]) == {"LAX", "SNA", "ANC", "BOS"}      # LHR dropped (not US), duplicate BOS strip dropped
    bos = df.set_index("airport_code").loc["BOS"]
    assert bos["airport_name"] == "Logan International Airport" and bos["state"] == "MA" and bos["airport_type"] == "large_airport"
    assert df.set_index("airport_code").loc["ANC", "icao"] == "PANC"


@pytest.mark.parametrize(
    "pax,band",
    [(10_000_000, "large"), (9_999_999, "medium"), (2_500_000, "medium"), (2_499_999, "small"), (500_000, "small"), (499_999, "non_hub"), (0, "non_hub")],
)
def test_assign_size_band_thresholds(pax, band):
    assert ing.assign_size_band(pax) == band


def test_build_airports_table_merges_sources():
    refs = pd.DataFrame({"airport_code": ["LAX", "BOS", "TNY"], "city": ["Los Angeles", "Boston", "Tinytown"], "state": ["CA", "MA", "MA"]})
    months = [date(2025, m, 1) for m in range(1, 13)]
    traffic = pd.DataFrame(
        [{"year_month": m, "airport_code": "LAX", "passengers": 3_000_000} for m in months]
        + [{"year_month": m, "airport_code": "BOS", "passengers": 1_600_000} for m in months]
        + [{"year_month": m, "airport_code": "TNY", "passengers": 1_000} for m in months]
    )
    reference = ing.load_ourairports_csv(FIXTURES / "fixture_ourairports_sample.csv")
    table = ing.build_airports_table(refs, otp_codes={"LAX", "BOS", "SNA"}, traffic=traffic, reference=reference, end_month="2025-12")
    by = table.set_index("airport_code")
    assert set(by.index) == {"LAX", "BOS", "TNY", "SNA"}
    assert by.loc["LAX", "airport_name"] == "Los Angeles International Airport"
    assert by.loc["LAX", "size_band"] == "large" and by.loc["LAX", "latest_12m_passengers"] == 36_000_000
    assert by.loc["BOS", "region"] == "new_england" and by.loc["BOS", "size_band"] == "large"
    assert by.loc["TNY", "size_band"] == "non_hub"
    assert by.loc["TNY", "airport_name"] == "Tinytown, MA (TNY)"            # not in the reference -> derived name
    assert bool(by.loc["SNA", "has_traffic_data"]) is False and bool(by.loc["SNA", "has_operations_data"]) is True
    assert by.loc["SNA", "size_band"] is None or pd.isna(by.loc["SNA", "size_band"])
    assert "los angeles" in by.loc["LAX", "aliases"] and "lax airport" in by.loc["LAX", "aliases"]
    assert by.loc["BOS", "icao_code"] == "KBOS"


# --------------------------------------------------------------------------------------
# HTTP retry behaviour and helpers
# --------------------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text

    def iter_content(self, chunk_size=1):
        yield self.text.encode()


class _FakeSession:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls += 1
        status = self.statuses.pop(0)
        if isinstance(status, Exception):
            raise status
        return _FakeResponse(status, "payload")


def test_http_client_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(ing.time, "sleep", lambda s: None)
    session = _FakeSession([500, requests.ConnectionError("boom"), 200])
    client = ing.HttpClient(session=session, retries=3, backoff=1.0)  # type: ignore[arg-type]
    resp = client.request("GET", "https://example.invalid/file.zip")
    assert resp.status_code == 200 and session.calls == 3


def test_http_client_gives_up(monkeypatch):
    monkeypatch.setattr(ing.time, "sleep", lambda s: None)
    session = _FakeSession([503, 503, 503])
    client = ing.HttpClient(session=session, retries=3, backoff=1.0)  # type: ignore[arg-type]
    with pytest.raises(ing.SourceError, match="Giving up"):
        client.request("GET", "https://example.invalid/file.zip")


def test_months_between():
    assert ing.months_between("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]
    with pytest.raises(ValueError):
        ing.months_between("2026-02", "2025-11")


# --------------------------------------------------------------------------------------
# End-to-end runner with fake downloads (fixture zips), no network
# --------------------------------------------------------------------------------------

def test_refresh_runner_end_to_end(tmp_path, monkeypatch):
    # fixture zips shaped like the real downloads
    otp_zip = tmp_path / "otp_2026_07.zip"
    with zipfile.ZipFile(otp_zip, "w") as zf:
        zf.write(FIXTURES / "fixture_bts_otp_sample.csv", "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2026_7.csv")
    t100_df = pd.read_csv(FIXTURES / "fixture_bts_t100_sample.csv", dtype=str, keep_default_na=False)
    t100_df["YEAR"] = "2026"
    t100_df["MONTH"] = "7"
    t100_csv = tmp_path / "T_T100_SEGMENT_ALL_CARRIER.csv"
    t100_df.to_csv(t100_csv, index=False)
    t100_zip = tmp_path / "t100_segment_all_2026.zip"
    with zipfile.ZipFile(t100_zip, "w") as zf:
        zf.write(t100_csv, "Documentation.csv")          # decoy member, like the real download
        zf.write(t100_csv, "T_T100_SEGMENT_ALL_CARRIER.csv")

    monkeypatch.setattr(ing, "NATIONAL_MONTHLY_FLIGHTS_RANGE", (1, 10**9))
    monkeypatch.setattr(ing.BtsOnTimeAdapter, "fetch_month", lambda self, ym: otp_zip)
    monkeypatch.setattr(ing.BtsOnTimeAdapter, "probe_latest_month", lambda self, not_after=None, lookback=18: "2026-07")
    monkeypatch.setattr(ing.BtsT100Adapter, "fetch_year", lambda self, year: t100_zip)
    monkeypatch.setattr(ing.OurAirportsAdapter, "fetch", lambda self: ing.load_ourairports_csv(FIXTURES / "fixture_ourairports_sample.csv"))

    db_path = tmp_path / "db" / "airport_intelligence.duckdb"
    runner = ing.RefreshRunner(db_path, tmp_path / "raw", http=ing.HttpClient(session=requests.Session()), keep_raw=True, workers=2, progress=lambda m: None)
    summary = runner.run("demo", "2026-07", None)

    assert summary.ok, summary
    assert summary.otp_months_loaded == ["2026-07"] and summary.t100_years_loaded == [2026]
    assert summary.traffic_end_month == "2026-07" and summary.ops_end_month == "2026-07"
    assert db_path.exists() and not db_path.with_name(db_path.name + ".tmp").exists()

    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM airport_monthly_operations").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM airport_monthly_traffic").fetchone()[0] == 2        # ANC, LAX for 2026-07
        # all sample rows were moved to 2026-07, so the February AS row (3,500 pax) joins the January rows
        assert conn.execute("SELECT passengers FROM airport_monthly_traffic WHERE airport_code='ANC'").fetchone()[0] == 3800 + 2400 + 300 + 3500
        airports = conn.execute("SELECT airport_code, airport_name, has_traffic_data, has_operations_data FROM airports ORDER BY 1").fetchall()
        assert [a[0] for a in airports] == ["ANC", "LAX", "SNA"]
        assert airports[0][1].startswith("Ted Stevens") and airports[0][2] is True and airports[0][3] is False
        assert airports[2][2] is False and airports[2][3] is True
        meta = conn.execute("SELECT source, status, record_count FROM data_refresh_metadata ORDER BY id").fetchall()
        assert {m[0] for m in meta} >= {ing.SOURCE_T100, ing.SOURCE_OTP, ing.SOURCE_OURAIRPORTS, ing.SOURCE_AIRPORTS_DERIVED}
        assert all(m[1] == "ok" for m in meta)
    finally:
        conn.close()


def test_refresh_runner_records_failures_honestly(tmp_path, monkeypatch):
    otp_zip = tmp_path / "otp_2026_07.zip"
    with zipfile.ZipFile(otp_zip, "w") as zf:
        zf.write(FIXTURES / "fixture_bts_otp_sample.csv", "otp.csv")
    monkeypatch.setattr(ing, "NATIONAL_MONTHLY_FLIGHTS_RANGE", (1, 10**9))
    monkeypatch.setattr(ing.BtsOnTimeAdapter, "fetch_month", lambda self, ym: otp_zip)
    monkeypatch.setattr(ing.BtsOnTimeAdapter, "probe_latest_month", lambda self, not_after=None, lookback=18: "2026-07")

    def failing_fetch(self, year):
        raise ing.SourceError("simulated TranStats outage")

    monkeypatch.setattr(ing.BtsT100Adapter, "fetch_year", failing_fetch)
    monkeypatch.setattr(ing.OurAirportsAdapter, "fetch", lambda self: (_ for _ in ()).throw(ing.SourceError("reference offline")))
    db_path = tmp_path / "airport_intelligence.duckdb"
    runner = ing.RefreshRunner(db_path, tmp_path / "raw", http=ing.HttpClient(session=requests.Session()), workers=1, progress=lambda m: None)
    summary = runner.run("demo", "2026-07", None)
    assert not summary.ok and summary.t100_years_failed == {2026: "simulated TranStats outage"}
    assert summary.otp_months_loaded == ["2026-07"]
    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        status = dict(conn.execute("SELECT source, status FROM data_refresh_metadata").fetchall())
        assert status[ing.SOURCE_T100] == "failed" and status[ing.SOURCE_OTP] == "ok" and status[ing.SOURCE_OURAIRPORTS] == "failed"
        assert conn.execute("SELECT COUNT(*) FROM airport_monthly_traffic").fetchone()[0] == 0
        names = conn.execute("SELECT airport_name FROM airports ORDER BY airport_code").fetchall()
        assert names[0][0] == "LAX"          # derived name when no reference and no T-100 city/state
    finally:
        conn.close()
