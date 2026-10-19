"""Refresh the local DuckDB store from public BTS TranStats data.

Examples
  python scripts/refresh_data.py --scope demo
  python scripts/refresh_data.py --scope nationwide --start 2019-01
  python scripts/refresh_data.py --scope demo --start 2023-01 --end 2026-07 --keep-raw --workers 4
  python scripts/refresh_data.py --show-metadata
  python scripts/refresh_data.py --scores-only

Scopes
  demo        quick real-data scope: all U.S. airports, months from DEMO_SCOPE_START (default 2023-01) to the latest
  nationwide  complete history from NATIONWIDE_SCOPE_START (default 2019-01) to the latest

Exit codes: 0 success, 2 one or more sources failed (database untouched if nothing loaded), 3 bad arguments.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db as dbmod  # noqa: E402
from src.config import configure_logging, load_settings  # noqa: E402
from src.data_ingestion import RefreshRunner  # noqa: E402
from src.models import parse_year_month  # noqa: E402

KEY_AIRPORTS = ["LAX", "SNA", "ANC", "SFO", "BOS", "BDL", "PVD", "MHT", "PWM", "BTV"]


def _console_logging(level: str) -> None:
    root = logging.getLogger("airport_intel")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level, logging.INFO))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def show_metadata(db_path: Path) -> int:
    if not db_path.exists():
        print(f"No database at {db_path}")
        return 1
    conn = dbmod.connect(db_path, read_only=True)
    try:
        print("=== data_refresh_metadata ===")
        print(dbmod.read_refresh_metadata(conn).to_string(index=False))
        print("\n=== table counts ===")
        for t in dbmod.REQUIRED_TABLES:
            print(f"{t:32s} {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>10,}")
        print("\n=== coverage for key airports ===")
        rows = conn.execute(
            """
            SELECT a.airport_code, a.airport_name, a.state, a.size_band,
                   (SELECT COUNT(*) FROM airport_monthly_traffic t WHERE t.airport_code = a.airport_code) AS traffic_months,
                   (SELECT MIN(year_month) FROM airport_monthly_traffic t WHERE t.airport_code = a.airport_code) AS traffic_from,
                   (SELECT MAX(year_month) FROM airport_monthly_traffic t WHERE t.airport_code = a.airport_code) AS traffic_to,
                   (SELECT COUNT(*) FROM airport_monthly_operations o WHERE o.airport_code = a.airport_code) AS ops_months,
                   (SELECT MAX(year_month) FROM airport_monthly_operations o WHERE o.airport_code = a.airport_code) AS ops_to,
                   a.latest_12m_passengers
            FROM airports a WHERE a.airport_code IN ({codes}) ORDER BY a.airport_code
            """.format(codes=", ".join(f"'{c}'" for c in KEY_AIRPORTS))
        ).df()
        print(rows.to_string(index=False))
        ne = conn.execute(
            "SELECT COUNT(*) FROM airports WHERE region = 'new_england' AND has_traffic_data AND has_operations_data"
        ).fetchone()[0]
        print(f"\nNew England airports with traffic + operations data: {ne}")
    finally:
        conn.close()
    return 0


def scores_only(db_path: Path) -> int:
    from src.analytics import write_default_scores_snapshot

    conn = dbmod.connect(db_path)
    try:
        latest = dbmod.latest_refresh(conn)
        scope = latest["data_scope"] if latest else "demo"
        n = write_default_scores_snapshot(conn, scope)
        print(f"airport_scores snapshot written: {n} rows (scope {scope})")
    finally:
        conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["demo", "nationwide"], default=None)
    ap.add_argument("--start", help="first month YYYY-MM (default from scope setting)")
    ap.add_argument("--end", help="last month YYYY-MM (default: latest available On-Time Performance month)")
    ap.add_argument("--keep-raw", action="store_true", help="cache downloaded zips under --raw-dir for re-runs")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    ap.add_argument("--db", help="DuckDB path (default DUCKDB_PATH)")
    ap.add_argument("--raw-dir", default=None, help="raw download cache directory (default data/raw)")
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first source failure")
    ap.add_argument("--show-metadata", action="store_true", help="print refresh metadata and coverage, then exit")
    ap.add_argument("--scores-only", action="store_true", help="recompute the default airport_scores snapshot only")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    settings = load_settings()
    configure_logging(settings)
    _console_logging(args.log_level.upper())
    db_path = Path(args.db) if args.db else settings.duckdb_path
    raw_dir = Path(args.raw_dir) if args.raw_dir else (settings.project_root / "data" / "raw")

    if args.show_metadata:
        return show_metadata(db_path)
    if args.scores_only:
        return scores_only(db_path)
    if not args.scope:
        ap.error("--scope is required (demo | nationwide)")

    start = args.start or (settings.demo_scope_start if args.scope == "demo" else settings.nationwide_scope_start)
    try:
        parse_year_month(start)
        if args.end:
            parse_year_month(args.end)
    except ValueError as exc:
        print(f"Bad month argument: {exc}")
        return 3

    print(f"Refreshing scope={args.scope} start={start} end={args.end or 'latest available'} db={db_path} raw={raw_dir} keep_raw={args.keep_raw}")
    runner = RefreshRunner(db_path, raw_dir, keep_raw=args.keep_raw, workers=args.workers, progress=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
    try:
        summary = runner.run(args.scope, start, args.end, fail_fast=args.fail_fast)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logging.getLogger("airport_intel").exception("refresh failed")
        print(f"\nREFRESH FAILED: {exc}")
        print(f"Retry with: python scripts/refresh_data.py --scope {args.scope} --start {start}" + (f" --end {args.end}" if args.end else "") + " --keep-raw")
        return 2

    print("\n=== refresh summary ===")
    print(f"scope: {summary.scope}   months: {summary.start_month}..{summary.ops_end_month or '?'}   elapsed: {summary.elapsed_seconds}s")
    print(f"T-100 years loaded: {summary.t100_years_loaded}   failed: {summary.t100_years_failed or 'none'}")
    print(f"OTP months loaded: {len(summary.otp_months_loaded)}   failed: {summary.otp_months_failed or 'none'}")
    print(f"traffic through {summary.traffic_end_month}, operations through {summary.ops_end_month}")
    print(f"airports table rows: {summary.airports_rows}   reference: {summary.reference_status}")
    for note in summary.notes:
        print(f"note: {note}")
    if summary.db_path:
        print(f"database written: {summary.db_path}")
        show_metadata(Path(summary.db_path))
    if not summary.ok:
        print("\nOne or more sources failed. The metadata table records the failure honestly.")
        print(f"Retry with: python scripts/refresh_data.py --scope {args.scope} --start {start}" + (f" --end {args.end}" if args.end else "") + " --keep-raw")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
