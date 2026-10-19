"""DuckDB storage: schema initialization, connections, refresh metadata.

The application only ever reads aggregated monthly tables from this database.
Raw BTS files are never queried at runtime.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

import duckdb
import pandas as pd

PathLike = Union[str, Path]

REQUIRED_TABLES = [
    "airports",
    "airport_monthly_traffic",
    "airport_monthly_operations",
    "route_monthly_metrics",
    "airport_scores",
    "data_refresh_metadata",
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS airports (
    airport_code            VARCHAR PRIMARY KEY,
    airport_name            VARCHAR,
    city                    VARCHAR,
    state                   VARCHAR,
    country                 VARCHAR DEFAULT 'US',
    iata_code               VARCHAR,
    icao_code               VARCHAR,
    region                  VARCHAR,
    size_band               VARCHAR,
    aliases                 VARCHAR,
    airport_type            VARCHAR,
    has_traffic_data        BOOLEAN DEFAULT FALSE,
    has_operations_data     BOOLEAN DEFAULT FALSE,
    latest_12m_passengers   BIGINT,
    latitude                DOUBLE,
    longitude               DOUBLE
);

CREATE TABLE IF NOT EXISTS airport_monthly_traffic (
    year_month              DATE,
    airport_code            VARCHAR,
    passengers              BIGINT,
    available_seats         BIGINT,
    departures_performed    BIGINT,
    departures_scheduled    BIGINT,
    load_factor             DOUBLE,
    source                  VARCHAR,
    PRIMARY KEY (year_month, airport_code)
);

CREATE TABLE IF NOT EXISTS airport_monthly_operations (
    year_month                  DATE,
    airport_code                VARCHAR,
    total_flights               BIGINT,
    operated_flights            BIGINT,
    delayed_departures_15m      BIGINT,
    delay_rate                  DOUBLE,
    cancelled_flights           BIGINT,
    cancellation_rate           DOUBLE,
    diverted_flights            BIGINT,
    dep_delay_minutes_sum       DOUBLE,
    avg_departure_delay_minutes DOUBLE,
    source                      VARCHAR,
    PRIMARY KEY (year_month, airport_code)
);

CREATE TABLE IF NOT EXISTS route_monthly_metrics (
    year_month              DATE,
    origin_airport          VARCHAR,
    destination_airport     VARCHAR,
    distance_miles          DOUBLE,
    passengers              BIGINT,
    available_seats         BIGINT,
    departures_performed    BIGINT,
    is_long_haul            BOOLEAN,
    source                  VARCHAR,
    PRIMARY KEY (year_month, origin_airport, destination_airport)
);

CREATE TABLE IF NOT EXISTS airport_scores (
    airport_code                VARCHAR,
    analysis_period_start       DATE,
    analysis_period_end         DATE,
    peer_group                  VARCHAR,
    demand_growth               DOUBLE,
    capacity_utilization        DOUBLE,
    operational_pressure        DOUBLE,
    demand_growth_score         DOUBLE,
    capacity_utilization_score  DOUBLE,
    operational_pressure_score  DOUBLE,
    expansion_opportunity_score DOUBLE,
    data_quality_status         VARCHAR,
    computed_at                 TIMESTAMP,
    PRIMARY KEY (airport_code, analysis_period_end, peer_group)
);

CREATE SEQUENCE IF NOT EXISTS seq_data_refresh_metadata;

CREATE TABLE IF NOT EXISTS data_refresh_metadata (
    id              INTEGER DEFAULT nextval('seq_data_refresh_metadata'),
    source          VARCHAR,
    data_scope      VARCHAR,
    coverage_start  DATE,
    coverage_end    DATE,
    refreshed_at    TIMESTAMP,
    record_count    BIGINT,
    status          VARCHAR,
    notes           VARCHAR
);
"""

# Column semantics (kept here so the schema and the docs agree):
#   airport_monthly_operations.total_flights        scheduled flights reported, including cancellations
#   airport_monthly_operations.operated_flights     total_flights - cancelled_flights
#   airport_monthly_operations.delayed_departures_15m operated flights with departure delay >= 15 minutes
#   airport_monthly_operations.delay_rate           delayed_departures_15m / operated_flights
#   airport_monthly_operations.cancellation_rate    cancelled_flights / total_flights
#   airport_monthly_operations.dep_delay_minutes_sum sum of max(departure delay, 0) over operated flights
#   route_monthly_metrics.is_long_haul              informational, computed at the default 2,000-mile threshold
#   airports.aliases                                JSON list of lower-case aliases
#   data_refresh_metadata.status                    ok | failed | partial


def connect(path: PathLike = ":memory:", read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection. Read-only connections are used by the UI."""
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(target, read_only=read_only)


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for statement in SCHEMA_SQL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)


def table_names(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def missing_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    present = set(table_names(conn))
    return [t for t in REQUIRED_TABLES if t not in present]


def database_ready(path: PathLike) -> bool:
    """True when the file exists and contains every required table."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        conn = connect(p, read_only=True)
    except duckdb.Error:
        return False
    try:
        return not missing_tables(conn)
    finally:
        conn.close()


def record_refresh_metadata(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    data_scope: str,
    coverage_start: Optional[date],
    coverage_end: Optional[date],
    record_count: int,
    status: str,
    notes: Optional[str] = None,
    refreshed_at: Optional[datetime] = None,
) -> None:
    ts = refreshed_at or datetime.now(timezone.utc).replace(tzinfo=None)
    conn.execute(
        """
        INSERT INTO data_refresh_metadata
            (source, data_scope, coverage_start, coverage_end, refreshed_at, record_count, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [source, data_scope, coverage_start, coverage_end, ts, int(record_count), status, notes],
    )


def read_refresh_metadata(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return conn.execute(
        "SELECT id, source, data_scope, coverage_start, coverage_end, refreshed_at, record_count, status, notes "
        "FROM data_refresh_metadata ORDER BY refreshed_at, id"
    ).df()


def _to_date(value) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def latest_refresh(conn: duckdb.DuckDBPyConnection) -> Optional[dict]:
    """Summarize the most recent refresh: scope, timestamp, per-source status and coverage."""
    df = read_refresh_metadata(conn)
    if df.empty:
        return None
    ordered = df.sort_values(["refreshed_at", "id"])
    latest_scope = ordered.iloc[-1]["data_scope"]
    rows = ordered[ordered["data_scope"] == latest_scope]
    last_ts = rows["refreshed_at"].max()
    sources = []
    for _, r in rows.drop_duplicates(subset=["source"], keep="last").iterrows():
        sources.append(
            {
                "source": r["source"],
                "status": r["status"],
                "coverage_start": _to_date(r["coverage_start"]),
                "coverage_end": _to_date(r["coverage_end"]),
                "record_count": 0 if pd.isna(r["record_count"]) else int(r["record_count"]),
                "notes": None if pd.isna(r["notes"]) else r["notes"],
            }
        )
    return {
        "data_scope": latest_scope,
        "refreshed_at": pd.Timestamp(last_ts).to_pydatetime().replace(microsecond=0).isoformat(),
        "sources": sources,
    }


def atomic_replace(tmp_path: PathLike, final_path: PathLike) -> None:
    """Replace the live database with a freshly built one. Removes stale WAL files."""
    tmp, final = Path(tmp_path), Path(final_path)
    stale = final.with_name(final.name + ".wal")
    if stale.exists():
        stale.unlink()
    os.replace(tmp, final)


def insert_dataframe(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
) -> int:
    """Append a DataFrame to a table using DuckDB's zero-copy registration."""
    if df is None or df.empty:
        return 0
    cols = list(columns) if columns else list(df.columns)
    conn.register("_df_to_insert", df[cols])
    col_list = ", ".join(cols)
    conn.execute(f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _df_to_insert")
    conn.unregister("_df_to_insert")
    return int(len(df))
