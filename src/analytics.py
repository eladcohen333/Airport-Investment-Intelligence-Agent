"""Historical analytics over the local DuckDB store: coverage, periods, raw KPIs, route mix, series, snapshots.

Only aggregated monthly tables are read here. All arithmetic is deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional

import duckdb
import numpy as np
import pandas as pd

from src import db as dbmod
from src import scoring
from src.models import (
    AirportRef, AnalysisPeriod, MetricChange, MonthlyPoint, PeerGroupSpec, PeriodRequest, RawKPIs, RouteRow,
    date_to_year_month, month_add, month_diff, month_end, month_start, parse_year_month,
)

PARTIAL_MONTH_RATIO = 0.5
WINDOW = 12


# --------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------

@dataclass
class CoverageInfo:
    traffic_first: Optional[str] = None
    traffic_last: Optional[str] = None
    ops_first: Optional[str] = None
    ops_last: Optional[str] = None
    routes_last: Optional[str] = None
    common_end_month: Optional[str] = None
    earliest_valid_end_month: Optional[str] = None
    data_scope: str = "demo"
    refresh_timestamp: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.common_end_month is not None and self.earliest_valid_end_month is not None and \
            month_diff(self.common_end_month, self.earliest_valid_end_month) >= 0

    def available_end_months(self) -> list[str]:
        if not self.ready:
            return []
        n = month_diff(self.common_end_month, self.earliest_valid_end_month)
        return [month_add(self.earliest_valid_end_month, i) for i in range(n + 1)]

    def available_calendar_years(self) -> list[int]:
        return sorted({parse_year_month(m)[0] for m in self.available_end_months() if m.endswith("-12")})


def latest_complete_month(monthly_totals: dict[str, float], ratio: float = PARTIAL_MONTH_RATIO) -> Optional[str]:
    """Latest month whose national total is >= ratio x median of the preceding 12 months (guards partial loads)."""
    if not monthly_totals:
        return None
    months = sorted(monthly_totals)
    for idx in range(len(months) - 1, -1, -1):
        ym = months[idx]
        prev = [monthly_totals[m] for m in months[max(0, idx - 12): idx]]
        if len(prev) < 3:
            return ym
        median = float(np.median(prev))
        if median <= 0 or monthly_totals[ym] >= ratio * median:
            return ym
    return None


def _monthly_totals(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> dict[str, float]:
    rows = conn.execute(f"SELECT year_month, SUM({column}) FROM {table} GROUP BY 1 ORDER BY 1").fetchall()
    return {date_to_year_month(pd.Timestamp(r[0]).date()): float(r[1] or 0) for r in rows}


def get_coverage(conn: duckdb.DuckDBPyConnection) -> CoverageInfo:
    info = CoverageInfo()
    traffic = _monthly_totals(conn, "airport_monthly_traffic", "departures_performed")
    ops = _monthly_totals(conn, "airport_monthly_operations", "total_flights")
    if traffic:
        info.traffic_first, info.traffic_last = min(traffic), latest_complete_month(traffic)
        if info.traffic_last != max(traffic):
            info.notes.append(f"Traffic month {max(traffic)} looks partial and is ignored for period defaults.")
    if ops:
        info.ops_first, info.ops_last = min(ops), latest_complete_month(ops)
        if info.ops_last != max(ops):
            info.notes.append(f"Operations month {max(ops)} looks partial and is ignored for period defaults.")
    routes = conn.execute("SELECT MAX(year_month) FROM route_monthly_metrics").fetchone()[0]
    info.routes_last = date_to_year_month(pd.Timestamp(routes).date()) if routes else None
    if info.traffic_last and info.ops_last:
        info.common_end_month = min(info.traffic_last, info.ops_last)
        info.earliest_valid_end_month = month_add(max(info.traffic_first, info.ops_first), 2 * WINDOW - 1)
        if info.traffic_last != info.ops_last:
            info.notes.append(f"Traffic data available through {info.traffic_last}, operations through {info.ops_last}; defaults use {info.common_end_month}.")
    latest = dbmod.latest_refresh(conn)
    if latest:
        info.data_scope = latest["data_scope"]
        info.refresh_timestamp = latest["refreshed_at"]
        info.sources = [s["source"] for s in latest["sources"]]
        info.source_status = {s["source"]: s["status"] for s in latest["sources"]}
    return info


# --------------------------------------------------------------------------------------
# Period handling
# --------------------------------------------------------------------------------------

MONTH_NAMES = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], start=1)}
MONTH_NAMES.update({k[:3]: v for k, v in list(MONTH_NAMES.items())})
_YEAR = r"(?P<y>(?:19|20)\d{2})"


def parse_period_phrase(text: str, today: Optional[date] = None) -> tuple[Optional[PeriodRequest], list[tuple[int, int]], list[str]]:
    """Deterministic extraction of a period request from free text."""
    today = today or date.today()
    spans: list[tuple[int, int]] = []
    notes: list[str] = []
    low = text.lower()

    m = re.search(rf"\b{_YEAR}\s*(?:vs\.?|versus|compared\s+(?:to|with)|against|over)\s*(?P<y2>(?:19|20)\d{{2}})\b", low)
    if m:
        y1, y2 = int(m.group("y")), int(m.group("y2"))
        spans.append(m.span())
        if y1 - y2 != 1:
            notes.append("non_adjacent_years")
        return PeriodRequest(year=max(y1, y2)), spans, notes

    m = re.search(r"\bq(?P<q>[1-4])\s*(?P<y>(?:19|20)\d{2})\b", low) or re.search(r"\b(?P<y>(?:19|20)\d{2})\s*q(?P<q>[1-4])\b", low)
    if m:
        q, y = int(m.group("q")), int(m.group("y"))
        spans.append(m.span())
        return PeriodRequest(end_month=f"{y:04d}-{q * 3:02d}"), spans, notes

    m = re.search(r"\b(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(?P<y>(?:19|20)\d{2})\b", low)
    if m:
        mon = MONTH_NAMES[m.group("mon")[:3]]
        spans.append(m.span())
        return PeriodRequest(end_month=f"{int(m.group('y')):04d}-{mon:02d}"), spans, notes

    m = re.search(r"\b(?P<y>(?:19|20)\d{2})-(?P<m>0[1-9]|1[0-2])\b", low)
    if m:
        spans.append(m.span())
        return PeriodRequest(end_month=f"{m.group('y')}-{m.group('m')}"), spans, notes

    if re.search(r"\blast\s+year\b", low):
        mm = re.search(r"\blast\s+year\b", low)
        spans.append(mm.span())
        return PeriodRequest(year=today.year - 1), spans, notes

    if re.search(r"\b(this\s+year|year[\s-]to[\s-]date|ytd)\b", low):
        mm = re.search(r"\b(this\s+year|year[\s-]to[\s-]date|ytd)\b", low)
        spans.append(mm.span())
        notes.append("year_to_date")
        return None, spans, notes

    m = re.search(rf"\b{_YEAR}\b", low)
    if m:
        spans.append(m.span())
        return PeriodRequest(year=int(m.group("y"))), spans, notes
    return None, spans, notes


def resolve_period(req: Optional[PeriodRequest], default_end_month: Optional[str], cov: CoverageInfo) -> tuple[Optional[AnalysisPeriod], list[str]]:
    """Turn a request into a concrete AnalysisPeriod respecting coverage. Returns (period or None, notes)."""
    notes: list[str] = []
    if not cov.ready:
        return None, ["Historical coverage is too short for two 12-month windows; refresh more history."]
    requested_label = None
    if req and req.end_month:
        end = req.end_month
        requested_label = f"12 months ending {end}"
    elif req and req.year:
        end = f"{req.year:04d}-12"
        requested_label = f"calendar {req.year}"
    else:
        end = default_end_month or cov.common_end_month
    clamped = False
    if month_diff(end, cov.common_end_month) > 0:
        notes.append(f"Requested {requested_label or end}, but data is available through {cov.common_end_month}; using the 12 months ending {cov.common_end_month}.")
        end = cov.common_end_month
        clamped = True
    if month_diff(end, cov.earliest_valid_end_month) < 0:
        notes.append(f"Requested {requested_label or end} needs data from {month_add(end, -(2 * WINDOW - 1))}, but coverage starts {max(cov.traffic_first or '', cov.ops_first or '')}.")
        return None, notes
    return AnalysisPeriod(end_month=end, requested=requested_label, clamped=clamped), notes


# --------------------------------------------------------------------------------------
# Airports and peer groups
# --------------------------------------------------------------------------------------

def airport_rows(conn: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    df = conn.execute(
        "SELECT airport_code, airport_name, city, state, region, size_band, aliases, airport_type, has_traffic_data, has_operations_data, latest_12m_passengers "
        "FROM airports ORDER BY airport_code"
    ).df()
    return df.where(pd.notna(df), None).to_dict("records")


def airport_refs(conn: duckdb.DuckDBPyConnection) -> dict[str, AirportRef]:
    refs: dict[str, AirportRef] = {}
    for r in airport_rows(conn):
        refs[r["airport_code"]] = AirportRef(
            airport_code=r["airport_code"], airport_name=r.get("airport_name") or r["airport_code"], city=r.get("city") or "",
            state=r.get("state") or "", size_band=r.get("size_band"), has_traffic_data=bool(r.get("has_traffic_data")),
            has_operations_data=bool(r.get("has_operations_data")), latest_12m_passengers=int(r.get("latest_12m_passengers") or 0),
        )
    return refs


def peer_codes(conn: duckdb.DuckDBPyConnection, spec: PeerGroupSpec) -> list[str]:
    clauses = ["has_traffic_data", "has_operations_data"]
    params: list[Any] = []
    states = spec.states
    if spec.region and not states:
        from src.models import REGIONS
        states = list(REGIONS.get(spec.region, ()))
    if states:
        clauses.append("state IN (" + ", ".join("?" for _ in states) + ")")
        params.extend(states)
    if spec.size_bands:
        clauses.append("size_band IN (" + ", ".join("?" for _ in spec.size_bands) + ")")
        params.extend(spec.size_bands)
    rows = conn.execute(f"SELECT airport_code FROM airports WHERE {' AND '.join(clauses)} ORDER BY airport_code", params).fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------------------
# Raw KPIs
# --------------------------------------------------------------------------------------

def compute_raw_kpis(conn: duckdb.DuckDBPyConnection, period: AnalysisPeriod, airport_codes: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One row per airport with window sums and derived KPIs (NaN where undefined)."""
    ls, le, ps, pe = period.latest_start, period.latest_end, period.prior_start, period.prior_end
    codes = list(airport_codes) if airport_codes is not None else None
    code_filter = ""
    params_base: list[Any] = [ls, le, ps, pe, ls, le, ps, pe, ls, le, ps, pe, ls, le, ps, pe, ps, le]
    if codes is not None:
        if not codes:
            return pd.DataFrame(columns=["airport_code"])
        code_filter = " AND airport_code IN (" + ", ".join("?" for _ in codes) + ")"
    traffic_sql = f"""
        SELECT airport_code,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN passengers END), 0) AS pax_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN passengers END), 0) AS pax_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN available_seats END), 0) AS seats_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN available_seats END), 0) AS seats_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN departures_performed END), 0) AS departures_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN departures_performed END), 0) AS departures_prior,
               COUNT(DISTINCT CASE WHEN year_month BETWEEN ? AND ? THEN year_month END) AS traffic_months_latest,
               COUNT(DISTINCT CASE WHEN year_month BETWEEN ? AND ? THEN year_month END) AS traffic_months_prior
        FROM airport_monthly_traffic
        WHERE year_month BETWEEN ? AND ?{code_filter}
        GROUP BY airport_code
    """
    ops_sql = f"""
        SELECT airport_code,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN total_flights END), 0) AS flights_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN total_flights END), 0) AS flights_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN operated_flights END), 0) AS operated_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN operated_flights END), 0) AS operated_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN delayed_departures_15m END), 0) AS delayed_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN delayed_departures_15m END), 0) AS delayed_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN cancelled_flights END), 0) AS cancelled_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN cancelled_flights END), 0) AS cancelled_prior,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN dep_delay_minutes_sum END), 0) AS delay_minutes_latest,
               COALESCE(SUM(CASE WHEN year_month BETWEEN ? AND ? THEN dep_delay_minutes_sum END), 0) AS delay_minutes_prior,
               COUNT(DISTINCT CASE WHEN year_month BETWEEN ? AND ? THEN year_month END) AS ops_months_latest,
               COUNT(DISTINCT CASE WHEN year_month BETWEEN ? AND ? THEN year_month END) AS ops_months_prior
        FROM airport_monthly_operations
        WHERE year_month BETWEEN ? AND ?{code_filter}
        GROUP BY airport_code
    """
    t_params = params_base[:16] + [ps, le] + (codes or [])
    # operations: 5 window sums + 1 month count = 6 latest/prior pairs (24 params) + range (2)
    o_params = [ls, le, ps, pe] * 6 + [ps, le] + (codes or [])
    traffic = conn.execute(traffic_sql, t_params).df()
    ops = conn.execute(ops_sql, o_params).df()
    merged = traffic.merge(ops, on="airport_code", how="outer")
    if codes is not None:
        merged = merged[merged["airport_code"].isin(codes)]
    count_cols = ["pax_latest", "pax_prior", "seats_latest", "seats_prior", "departures_latest", "departures_prior",
                  "traffic_months_latest", "traffic_months_prior", "flights_latest", "flights_prior", "operated_latest", "operated_prior",
                  "delayed_latest", "delayed_prior", "cancelled_latest", "cancelled_prior", "ops_months_latest", "ops_months_prior"]
    for c in count_cols:
        merged[c] = merged[c].fillna(0).astype("int64")
    for c in ("delay_minutes_latest", "delay_minutes_prior"):
        merged[c] = merged[c].fillna(0.0).astype(float)
    return _derive_kpis(merged).sort_values("airport_code").reset_index(drop=True)


def _derive_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    nan = np.nan
    out["demand_growth"] = np.where(out["pax_prior"] > 0, (out["pax_latest"] - out["pax_prior"]) / out["pax_prior"].replace(0, np.nan), nan)
    out["capacity_utilization"] = np.where(out["seats_latest"] > 0, out["pax_latest"] / out["seats_latest"].replace(0, np.nan), nan)
    out["capacity_utilization_prior"] = np.where(out["seats_prior"] > 0, out["pax_prior"] / out["seats_prior"].replace(0, np.nan), nan)
    out["delay_rate"] = np.where(out["operated_latest"] > 0, out["delayed_latest"] / out["operated_latest"].replace(0, np.nan), nan)
    out["delay_rate_prior"] = np.where(out["operated_prior"] > 0, out["delayed_prior"] / out["operated_prior"].replace(0, np.nan), nan)
    out["cancellation_rate"] = np.where(out["flights_latest"] > 0, out["cancelled_latest"] / out["flights_latest"].replace(0, np.nan), nan)
    out["cancellation_rate_prior"] = np.where(out["flights_prior"] > 0, out["cancelled_prior"] / out["flights_prior"].replace(0, np.nan), nan)
    out["avg_departure_delay_minutes"] = np.where(out["operated_latest"] > 0, out["delay_minutes_latest"] / out["operated_latest"].replace(0, np.nan), nan)
    out["avg_departure_delay_minutes_prior"] = np.where(out["operated_prior"] > 0, out["delay_minutes_prior"] / out["operated_prior"].replace(0, np.nan), nan)
    w = scoring.OPERATIONAL_PRESSURE_WEIGHTS
    out["operational_pressure"] = w["delay_rate"] * out["delay_rate"] + w["cancellation_rate"] * out["cancellation_rate"]
    out["operational_pressure_prior"] = w["delay_rate"] * out["delay_rate_prior"] + w["cancellation_rate"] * out["cancellation_rate_prior"]
    out["delay_rate_change"] = out["delay_rate"] - out["delay_rate_prior"]
    out["cancellation_rate_change"] = out["cancellation_rate"] - out["cancellation_rate_prior"]
    flags: list[list[str]] = []
    for _, r in out.iterrows():
        f: list[str] = []
        if pd.notna(r["capacity_utilization"]) and r["capacity_utilization"] > scoring.UTILIZATION_FLAG_THRESHOLD:
            f.append(f"utilization {r['capacity_utilization']:.2f} exceeds {scoring.UTILIZATION_FLAG_THRESHOLD} (possible seat under-reporting)")
        flags.append(f)
    out["quality_flags"] = flags
    return out


def raw_for_airport(conn: duckdb.DuckDBPyConnection, code: str, period: AnalysisPeriod) -> Optional[RawKPIs]:
    df = compute_raw_kpis(conn, period, [code])
    if df.empty:
        return None
    return scoring.row_to_raw(df.iloc[0])


# --------------------------------------------------------------------------------------
# Routes, series, profile changes
# --------------------------------------------------------------------------------------

@dataclass
class RouteMix:
    long_haul_departures: int
    total_departures: int
    share: Optional[float]
    threshold_miles: int
    months_covered: int
    top_routes: list[RouteRow]


def route_mix(conn: duckdb.DuckDBPyConnection, code: str, start: date, end: date, threshold_miles: int) -> RouteMix:
    df = conn.execute(
        """
        SELECT destination_airport, MAX(distance_miles) AS distance_miles, SUM(departures_performed) AS departures, SUM(passengers) AS passengers
        FROM route_monthly_metrics WHERE origin_airport = ? AND year_month BETWEEN ? AND ?
        GROUP BY 1 ORDER BY departures DESC, destination_airport
        """,
        [code, start, end],
    ).df()
    months = conn.execute(
        "SELECT COUNT(DISTINCT year_month) FROM route_monthly_metrics WHERE origin_airport = ? AND year_month BETWEEN ? AND ?", [code, start, end]
    ).fetchone()[0]
    if df.empty:
        return RouteMix(0, 0, None, threshold_miles, int(months or 0), [])
    long_mask = df["distance_miles"] >= threshold_miles
    total = int(df["departures"].sum())
    long_haul = int(df.loc[long_mask, "departures"].sum())
    top = [RouteRow(destination_airport=r.destination_airport, distance_miles=float(r.distance_miles), departures=int(r.departures), passengers=int(r.passengers or 0))
           for r in df[long_mask].head(10).itertuples()]
    return RouteMix(long_haul, total, (long_haul / total) if total else None, threshold_miles, int(months or 0), top)


def monthly_series(conn: duckdb.DuckDBPyConnection, code: str, period: AnalysisPeriod) -> list[MonthlyPoint]:
    df = conn.execute(
        """
        WITH months AS (SELECT DISTINCT year_month FROM (
            SELECT year_month FROM airport_monthly_traffic WHERE airport_code = ? AND year_month BETWEEN ? AND ?
            UNION SELECT year_month FROM airport_monthly_operations WHERE airport_code = ? AND year_month BETWEEN ? AND ?))
        SELECT m.year_month, t.passengers, t.available_seats, t.load_factor, o.total_flights, o.delay_rate, o.cancellation_rate
        FROM months m
        LEFT JOIN airport_monthly_traffic t ON t.year_month = m.year_month AND t.airport_code = ?
        LEFT JOIN airport_monthly_operations o ON o.year_month = m.year_month AND o.airport_code = ?
        ORDER BY m.year_month
        """,
        [code, period.prior_start, period.latest_end, code, period.prior_start, period.latest_end, code, code],
    ).df()
    points: list[MonthlyPoint] = []
    for r in df.itertuples():
        points.append(MonthlyPoint(
            year_month=date_to_year_month(pd.Timestamp(r.year_month).date()),
            passengers=None if pd.isna(r.passengers) else int(r.passengers),
            available_seats=None if pd.isna(r.available_seats) else int(r.available_seats),
            load_factor=None if pd.isna(r.load_factor) else float(r.load_factor),
            total_flights=None if pd.isna(r.total_flights) else int(r.total_flights),
            delay_rate=None if pd.isna(r.delay_rate) else float(r.delay_rate),
            cancellation_rate=None if pd.isna(r.cancellation_rate) else float(r.cancellation_rate),
        ))
    return points


def _change(metric: str, label: str, prior: Optional[float], latest: Optional[float], unit: str) -> MetricChange:
    if prior is None or latest is None:
        return MetricChange(metric=metric, label=label, prior=prior, latest=latest, abs_delta=None, pct_change=None, direction="n/a", unit=unit)
    delta = latest - prior
    pct = (delta / prior) if prior not in (0, 0.0) else None
    if abs(delta) < 1e-12:
        direction = "flat"
    else:
        direction = "up" if delta > 0 else "down"
    return MetricChange(metric=metric, label=label, prior=prior, latest=latest, abs_delta=delta, pct_change=pct, direction=direction, unit=unit)


def profile_changes(raw: RawKPIs, long_haul_latest: Optional[float] = None, long_haul_prior: Optional[float] = None) -> list[MetricChange]:
    return [
        _change("passengers", "Passengers (12m)", float(raw.pax_prior), float(raw.pax_latest), "passengers"),
        _change("available_seats", "Available seats (12m)", float(raw.seats_prior), float(raw.seats_latest), "seats"),
        _change("departures_performed", "Departures performed (12m)", float(raw.departures_prior), float(raw.departures_latest), "departures"),
        _change("capacity_utilization", "Capacity utilization", raw.capacity_utilization_prior, raw.capacity_utilization, "ratio"),
        _change("total_flights", "Reported flights (12m)", float(raw.flights_prior), float(raw.flights_latest), "flights"),
        _change("delay_rate", "Delayed-departure rate", raw.delay_rate_prior, raw.delay_rate, "ratio"),
        _change("cancellation_rate", "Cancellation rate", raw.cancellation_rate_prior, raw.cancellation_rate, "ratio"),
        _change("avg_departure_delay_minutes", "Average departure delay", raw.avg_departure_delay_minutes_prior, raw.avg_departure_delay_minutes, "minutes"),
        _change("operational_pressure", "Operational pressure", raw.operational_pressure_prior, raw.operational_pressure, "index"),
        _change("long_haul_share", "Long-haul departure share", long_haul_prior, long_haul_latest, "ratio"),
    ]


# --------------------------------------------------------------------------------------
# Default snapshot written at refresh time
# --------------------------------------------------------------------------------------

def write_default_scores_snapshot(conn: duckdb.DuckDBPyConnection, scope: str) -> int:
    cov = get_coverage(conn)
    if not cov.ready:
        return 0
    period = AnalysisPeriod(end_month=cov.common_end_month)
    spec = PeerGroupSpec(data_scope=scope, min_reported_flights=scoring.DEFAULT_MIN_REPORTED_FLIGHTS)
    codes = peer_codes(conn, spec)
    kpis = compute_raw_kpis(conn, period, codes)
    refs = airport_refs(conn)
    comp = scoring.rank_peer_group(kpis, refs, spec, top_n=None)
    label = f"{scope}:all_states:all_sizes"
    conn.execute("DELETE FROM airport_scores WHERE analysis_period_end = ? AND peer_group = ?", [period.latest_end, label])
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for r in comp.rows:
        rows.append([r.airport.airport_code, period.latest_start, period.latest_end, label, r.raw.demand_growth, r.raw.capacity_utilization,
                     r.raw.operational_pressure, r.scores.demand_growth_score, r.scores.capacity_utilization_score,
                     r.scores.operational_pressure_score, r.scores.expansion_opportunity_score, "ok", now])
    for e in comp.exclusions:
        raw = e.raw or RawKPIs(airport_code=e.airport.airport_code)
        rows.append([e.airport.airport_code, period.latest_start, period.latest_end, label, raw.demand_growth, raw.capacity_utilization,
                     raw.operational_pressure, None, None, None, None, "insufficient_data:" + ",".join(e.reasons), now])
    if rows:
        conn.executemany(
            "INSERT INTO airport_scores (airport_code, analysis_period_start, analysis_period_end, peer_group, demand_growth, capacity_utilization, "
            "operational_pressure, demand_growth_score, capacity_utilization_score, operational_pressure_score, expansion_opportunity_score, "
            "data_quality_status, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


__all__ = [
    "CoverageInfo", "get_coverage", "latest_complete_month", "parse_period_phrase", "resolve_period", "airport_rows", "airport_refs",
    "peer_codes", "compute_raw_kpis", "raw_for_airport", "route_mix", "RouteMix", "monthly_series", "profile_changes",
    "write_default_scores_snapshot",
]
