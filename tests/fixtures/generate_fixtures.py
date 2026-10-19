"""Regenerate the synthetic fixture CSVs from fixture_spec.py.

Run:  .venv\\Scripts\\python.exe tests/fixtures/generate_fixtures.py
The CSVs are committed; this script only exists to keep them reproducible.
FIXTURE DATA ONLY - not real BTS or live data.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fixture_spec import (  # noqa: E402
    AIRPORTS, MISSING_TRAFFIC, OPS_MONTHS_END, OPS_SOURCE, REFRESHED_AT, ROUTES, ROUTE_SOURCE,
    TRAFFIC_MONTHS_END, TRAFFIC_SOURCE, ops_for_year, pax_for_year,
)

NEW_ENGLAND = {"CT", "ME", "MA", "NH", "RI", "VT"}


def months(start: str, end: str) -> list[str]:
    sy, sm = int(start[:4]), int(start[5:])
    ey, em = int(end[:4]), int(end[5:])
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}-01")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} rows")


def main() -> None:
    traffic_months = months("2023-01", TRAFFIC_MONTHS_END)
    ops_months = months("2023-01", OPS_MONTHS_END)

    # airports -----------------------------------------------------------------------
    airport_rows = []
    for s in AIRPORTS:
        latest_12m = pax_for_year(s, 2025) * 12 if not any(k[0] == s.code for k in MISSING_TRAFFIC) else pax_for_year(s, 2025) * 11
        airport_rows.append([
            s.code, s.name, s.city, s.state, "US", s.code, f"K{s.code}" if s.state != "AK" else f"P{s.code}",
            "new_england" if s.state in NEW_ENGLAND else "",
            s.size_band, json.dumps(sorted(set(s.aliases))), s.airport_type, True, True, latest_12m, "", "",
        ])
    write_csv(HERE / "fixture_airports.csv",
              ["airport_code", "airport_name", "city", "state", "country", "iata_code", "icao_code", "region", "size_band",
               "aliases", "airport_type", "has_traffic_data", "has_operations_data", "latest_12m_passengers", "latitude", "longitude"],
              airport_rows)

    # traffic -------------------------------------------------------------------------
    traffic_rows = []
    for s in AIRPORTS:
        for ym in traffic_months:
            if (s.code, ym[:7]) in MISSING_TRAFFIC:
                continue
            year = int(ym[:4])
            pax = pax_for_year(s, year)
            seats = int(round(pax / s.lf))
            departures = max(1, pax // 100)
            traffic_rows.append([ym, s.code, pax, seats, departures, departures, round(pax / seats, 6), TRAFFIC_SOURCE])
    write_csv(HERE / "fixture_airport_monthly_traffic.csv",
              ["year_month", "airport_code", "passengers", "available_seats", "departures_performed", "departures_scheduled", "load_factor", "source"],
              traffic_rows)

    # operations ----------------------------------------------------------------------
    ops_rows = []
    for s in AIRPORTS:
        for ym in ops_months:
            year = int(ym[:4])
            total, cancelled, operated, delayed, minutes = ops_for_year(s, year)
            ops_rows.append([
                ym, s.code, total, operated, delayed,
                round(delayed / operated, 6) if operated else "",
                cancelled, round(cancelled / total, 6) if total else "",
                0, minutes, round(minutes / operated, 6) if operated else "", OPS_SOURCE,
            ])
    write_csv(HERE / "fixture_airport_monthly_operations.csv",
              ["year_month", "airport_code", "total_flights", "operated_flights", "delayed_departures_15m", "delay_rate",
               "cancelled_flights", "cancellation_rate", "diverted_flights", "dep_delay_minutes_sum", "avg_departure_delay_minutes", "source"],
              ops_rows)

    # routes --------------------------------------------------------------------------
    route_rows = []
    for origin, routes in ROUTES.items():
        for ym in traffic_months:
            for dest, dist, deps, pax in routes:
                route_rows.append([ym, origin, dest, dist, pax, int(round(pax / 0.8)), deps, dist >= 2000, ROUTE_SOURCE])
    write_csv(HERE / "fixture_route_monthly_metrics.csv",
              ["year_month", "origin_airport", "destination_airport", "distance_miles", "passengers", "available_seats",
               "departures_performed", "is_long_haul", "source"],
              route_rows)

    # metadata ------------------------------------------------------------------------
    write_csv(HERE / "fixture_data_refresh_metadata.csv",
              ["source", "data_scope", "coverage_start", "coverage_end", "refreshed_at", "record_count", "status", "notes"],
              [
                  [TRAFFIC_SOURCE, "demo", "2023-01-01", f"{TRAFFIC_MONTHS_END}-31", REFRESHED_AT, len(traffic_rows), "ok", "fixture"],
                  [OPS_SOURCE, "demo", "2023-01-01", f"{OPS_MONTHS_END}-28", REFRESHED_AT, len(ops_rows), "ok", "fixture"],
                  [ROUTE_SOURCE, "demo", "2023-01-01", f"{TRAFFIC_MONTHS_END}-31", REFRESHED_AT, len(route_rows), "ok", "fixture"],
              ])


if __name__ == "__main__":
    main()
