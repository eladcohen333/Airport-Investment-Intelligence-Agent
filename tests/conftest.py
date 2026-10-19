"""Shared pytest fixtures: an in-memory DuckDB loaded from tests/fixtures/fixture_*.csv."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src import db as dbmod

FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

FIXTURE_TABLES = {
    "airports": "fixture_airports.csv",
    "airport_monthly_traffic": "fixture_airport_monthly_traffic.csv",
    "airport_monthly_operations": "fixture_airport_monthly_operations.csv",
    "route_monthly_metrics": "fixture_route_monthly_metrics.csv",
    "data_refresh_metadata": "fixture_data_refresh_metadata.csv",
}


def load_fixture_db(conn: duckdb.DuckDBPyConnection) -> None:
    dbmod.init_schema(conn)
    for table, filename in FIXTURE_TABLES.items():
        path = (FIXTURES / filename).as_posix()
        cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
        csv_cols = pd.read_csv(FIXTURES / filename, nrows=0).columns.tolist()
        use_cols = [c for c in cols if c in csv_cols]
        col_list = ", ".join(use_cols)
        conn.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM read_csv('{path}', header=true, auto_detect=true)"
        )


@pytest.fixture(scope="session")
def fixture_conn() -> duckdb.DuckDBPyConnection:
    conn = dbmod.connect(":memory:")
    load_fixture_db(conn)
    yield conn
    conn.close()


@pytest.fixture()
def fresh_conn() -> duckdb.DuckDBPyConnection:
    conn = dbmod.connect(":memory:")
    load_fixture_db(conn)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def today() -> date:
    return date(2026, 10, 19)
