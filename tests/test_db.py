"""Phase 1 gate: schema creation, fixture loading, metadata recording."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from src import db as dbmod
from src.db import REQUIRED_TABLES


def test_schema_creates_all_required_tables():
    conn = dbmod.connect(":memory:")
    dbmod.init_schema(conn)
    assert dbmod.missing_tables(conn) == []
    assert set(REQUIRED_TABLES) <= set(dbmod.table_names(conn))
    # idempotent
    dbmod.init_schema(conn)
    assert dbmod.missing_tables(conn) == []


def test_fixtures_load_and_query(fixture_conn):
    counts = {t: fixture_conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in REQUIRED_TABLES}
    assert counts["airports"] == 21
    assert counts["airport_monthly_traffic"] == 21 * 36 - 1          # AUS is missing 2025-11
    assert counts["airport_monthly_operations"] == 21 * 38
    assert counts["route_monthly_metrics"] == 36 * (7 + 2 + 2 + 2)
    assert counts["data_refresh_metadata"] == 3
    assert counts["airport_scores"] == 0

    row = fixture_conn.execute(
        "SELECT passengers, available_seats FROM airport_monthly_traffic WHERE airport_code='BOS' AND year_month=DATE '2024-03-01'"
    ).fetchone()
    assert row[0] == 1_575_000                      # 1,500,000 * 1.05
    ops = fixture_conn.execute(
        "SELECT total_flights, operated_flights, delayed_departures_15m, cancelled_flights, delay_rate, cancellation_rate "
        "FROM airport_monthly_operations WHERE airport_code='LAX' AND year_month=DATE '2025-06-01'"
    ).fetchone()
    assert ops[:4] == (20_000, 19_600, 3_920, 400)
    assert abs(ops[4] - 0.20) < 1e-9 and abs(ops[5] - 0.02) < 1e-9

    ym_type = fixture_conn.execute(
        "SELECT data_type FROM information_schema.columns WHERE table_name='airport_monthly_traffic' AND column_name='year_month'"
    ).fetchone()[0]
    assert ym_type == "DATE"


def test_metadata_roundtrip(fresh_conn):
    before = dbmod.read_refresh_metadata(fresh_conn)
    dbmod.record_refresh_metadata(
        fresh_conn,
        source="BTS test source",
        data_scope="demo",
        coverage_start=date(2023, 1, 1),
        coverage_end=date(2026, 7, 31),
        record_count=1234,
        status="ok",
        notes="unit test",
        refreshed_at=datetime(2026, 10, 19, 12, 0, 0),
    )
    after = dbmod.read_refresh_metadata(fresh_conn)
    assert len(after) == len(before) + 1
    latest = dbmod.latest_refresh(fresh_conn)
    assert latest is not None
    assert latest["data_scope"] == "demo"
    assert latest["refreshed_at"].startswith("2026-10-19T12:00:00")
    sources = {s["source"]: s for s in latest["sources"]}
    assert sources["BTS test source"]["status"] == "ok"
    assert sources["BTS test source"]["coverage_end"] == date(2026, 7, 31)
    assert sources["BTS test source"]["record_count"] == 1234


def test_latest_refresh_none_when_empty():
    conn = dbmod.connect(":memory:")
    dbmod.init_schema(conn)
    assert dbmod.latest_refresh(conn) is None


def test_insert_dataframe_and_atomic_replace(tmp_path):
    tmp_db = tmp_path / "build.duckdb.tmp"
    final_db = tmp_path / "final.duckdb"
    conn = dbmod.connect(tmp_db)
    dbmod.init_schema(conn)
    df = pd.DataFrame(
        {
            "year_month": [date(2024, 1, 1), date(2024, 2, 1)],
            "airport_code": ["AAA", "AAA"],
            "passengers": [100, 120],
            "available_seats": [200, 200],
            "departures_performed": [3, 3],
            "departures_scheduled": [3, 3],
            "load_factor": [0.5, 0.6],
            "source": ["unit", "unit"],
        }
    )
    assert dbmod.insert_dataframe(conn, "airport_monthly_traffic", df) == 2
    conn.close()
    dbmod.atomic_replace(tmp_db, final_db)
    assert final_db.exists() and not tmp_db.exists()
    assert dbmod.database_ready(final_db)
    ro = dbmod.connect(final_db, read_only=True)
    assert ro.execute("SELECT SUM(passengers) FROM airport_monthly_traffic").fetchone()[0] == 220
    ro.close()
    assert dbmod.database_ready(tmp_path / "missing.duckdb") is False
