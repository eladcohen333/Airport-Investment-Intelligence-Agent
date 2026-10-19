"""Package the prepared historical DuckDB store as a GitHub Release asset.

Examples
  python scripts/package_dataset_release.py
  python scripts/package_dataset_release.py --db data/airport_intelligence.duckdb --out-dir dist
  python scripts/package_dataset_release.py --overwrite

What it does
  1. Validates the database with the project's own helpers: all six tables present and non-empty, refresh metadata
     present, both BTS historical sources recorded, every source status `ok`, and at least two adjacent 12-month
     windows of traffic and operations coverage. Anything else is rejected; nothing is fabricated.
  2. Reads scope, coverage dates, refresh timestamp, source metadata and table row counts from the database.
  3. Writes dist/airport_intelligence_<scope>_<start>_to_<end>.zip containing exactly
         airport_intelligence.duckdb
         dataset_manifest.json
     (no enclosing directory) plus a sibling .sha256 file, then prints the manual GitHub Release steps.

Nothing is uploaded, committed or pushed. Raw BTS files, .env files, keys and logs are never read or packaged.

Exit codes: 0 success, 1 validation or packaging failure, 2 an artifact already exists (use --overwrite), 3 bad arguments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset_release as rel  # noqa: E402
from src.config import load_settings  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "dist"


def _fmt_size(n: int) -> str:
    mib = n / (1024 * 1024)
    return f"{mib:,.1f} MiB" if mib < 1024 else f"{mib / 1024:,.2f} GiB"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def format_summary(artifact: rel.ReleaseArtifact) -> str:
    info, m = artifact.info, artifact.manifest
    lines = [
        "=== dataset release artifact ===",
        f"archive:         {_rel(artifact.archive_path)}",
        f"archive size:    {_fmt_size(artifact.archive_size_bytes)} ({artifact.archive_size_bytes:,} bytes)",
        f"archive sha256:  {artifact.archive_sha256}",
        f"checksum file:   {_rel(artifact.checksum_path)}",
        f"duckdb sha256:   {m['duckdb_sha256']}",
        f"duckdb size:     {_fmt_size(info.size_bytes)} ({info.size_bytes:,} bytes)",
        f"scope:           {info.data_scope}",
        f"coverage:        {info.coverage_start.isoformat()} to {info.coverage_end.isoformat()}",
        f"refreshed at:    {info.refreshed_at} UTC",
        f"duckdb library:  {m['duckdb_library_version']}",
        "sources:",
    ]
    for s in info.sources:
        cov = f"{s['coverage_start']} to {s['coverage_end']}" if s["coverage_start"] else "reference"
        lines.append(f"  - {s['status']:<7} {s['source']}  [{cov}; {s['record_count']:,} records]")
    lines.append("table row counts:")
    for table, n in info.table_row_counts.items():
        lines.append(f"  {table:<28} {n:>12,}")
    lines += [
        "",
        "Contents: aggregated historical DuckDB tables only. No raw BTS files, live API data, .env, keys or logs.",
        "",
        "Next manual steps (nothing was uploaded):",
        "  1. On GitHub open Releases -> Draft a new release; create a tag such as "
        f"dataset-{info.data_scope}-{info.coverage_end:%Y-%m} and a title naming the scope and coverage.",
        f"  2. Attach exactly two files: {artifact.archive_path.name} and {artifact.checksum_path.name}.",
        "  3. Publish, then copy the asset URL (https://github.com/<owner>/<repo>/releases/download/<tag>/"
        f"{artifact.archive_path.name}) and the archive sha256 above into the README quick-start placeholders.",
        "  4. Verify in a clean clone:",
        f"     python scripts/download_dataset_release.py --url <asset URL> --sha256 {artifact.archive_sha256}",
        "  See docs/dataset-release.md for the full checklist.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="DuckDB file to package (default: DUCKDB_PATH, i.e. data/airport_intelligence.duckdb)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="directory for the ZIP and .sha256 (default: dist/)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing archive/checksum with the same name")
    args = ap.parse_args(argv)

    db_path = Path(args.db) if args.db else load_settings().duckdb_path
    out_dir = Path(args.out_dir)
    print(f"Packaging {db_path} -> {out_dir}")
    try:
        artifact = rel.package_release(db_path, out_dir, overwrite=args.overwrite)
    except rel.ArtifactExistsError as exc:
        print(f"REFUSED: {exc}")
        return 2
    except rel.DatasetValidationError as exc:
        print("REJECTED: the database is not release-ready:")
        for problem in exc.problems:
            print(f"  - {problem}")
        print("Rebuild it with: python scripts/refresh_data.py --scope nationwide   (or --scope demo)")
        return 1
    except rel.DatasetReleaseError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(format_summary(artifact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
