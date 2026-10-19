"""Download, verify and install a prepared historical dataset from a GitHub Release asset.

Examples
  python scripts/download_dataset_release.py --url https://github.com/<owner>/<repo>/releases/download/<tag>/airport_intelligence_nationwide_2019-01_to_2026-07.zip --sha256 <hex from the .sha256 file>
  python scripts/download_dataset_release.py --url <asset URL> --sha256 <hex> --force      # replace an existing local database

What it does
  1. Refuses to continue when the destination database already exists, unless --force is given.
  2. Streams the asset to a temporary file and verifies its SHA-256 before anything is opened.
  3. Inspects the ZIP: exactly airport_intelligence.duckdb and dataset_manifest.json, plain file names only
     (no directories, traversal paths or links), a well-formed manifest whose declared size matches the member.
  4. Extracts only the DuckDB file next to the destination, checks it against the manifest hash, then validates it
     with the project's own database helpers (tables, refresh metadata, source statuses, 24-month coverage).
  5. Atomically replaces the destination and prints the loaded scope, coverage and refresh timestamp.
  Temporary files are removed on success and on failure.

Exit codes: 0 installed, 1 download/verification/installation failed, 2 destination exists (use --force), 3 bad arguments.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import dataset_release as rel  # noqa: E402
from src.config import load_settings  # noqa: E402

PROGRESS_STEP = 64 * 1024 * 1024


def _fmt_size(n: int) -> str:
    mib = n / (1024 * 1024)
    return f"{mib:,.1f} MiB" if mib < 1024 else f"{mib / 1024:,.2f} GiB"


def _progress_printer():
    last = {"mark": 0}

    def _report(received: int, declared):
        if received - last["mark"] >= PROGRESS_STEP:
            last["mark"] = received
            total = f" of {_fmt_size(declared)}" if declared else ""
            print(f"  downloaded {_fmt_size(received)}{total}", flush=True)

    return _report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="GitHub Release asset URL of the dataset ZIP (https)")
    ap.add_argument("--sha256", required=True, help="expected SHA-256 of the ZIP (from the published .sha256 file)")
    ap.add_argument("--dest", help="destination database path (default: DUCKDB_PATH, i.e. data/airport_intelligence.duckdb)")
    ap.add_argument("--force", action="store_true", help="replace an existing local database")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-read network timeout in seconds (default 60)")
    ap.add_argument("--max-bytes", type=int, default=rel.DEFAULT_MAX_DOWNLOAD_BYTES,
                    help="refuse downloads larger than this many bytes (default 16 GiB)")
    args = ap.parse_args(argv)

    try:
        expected = rel.normalize_sha256(args.sha256)
        rel.validate_asset_url(args.url)
    except ValueError as exc:
        print(f"Bad argument: {exc}")
        return 3
    if args.max_bytes <= 0:
        print("Bad argument: --max-bytes must be positive")
        return 3

    dest = Path(args.dest) if args.dest else load_settings().duckdb_path
    if dest.exists() and not args.force:
        print(f"REFUSED: {dest} already exists. Re-run with --force to replace it, or run with --dest to install elsewhere.")
        return 2

    asset_name = Path(urlparse(args.url).path).name or "dataset.zip"
    workdir = Path(tempfile.mkdtemp(prefix="airport_dataset_"))
    try:
        download_path = workdir / (asset_name + ".part")
        print(f"Downloading {args.url}")
        rel.download_file(args.url, download_path, expected_sha256=expected, timeout=args.timeout,
                          max_bytes=args.max_bytes, progress=_progress_printer())
        print(f"Checksum verified: sha256 {expected} ({_fmt_size(download_path.stat().st_size)})")
        print(f"Installing {rel.DATASET_FILENAME} -> {dest}")
        info = rel.install_archive(download_path, dest, force=args.force)
    except rel.ArtifactExistsError as exc:
        print(f"REFUSED: {exc}")
        return 2
    except rel.ChecksumMismatchError as exc:
        print(f"REJECTED: {exc}. The downloaded file was discarded; check the URL and the published .sha256 value.")
        return 1
    except rel.DatasetValidationError as exc:
        print("REJECTED: the downloaded database failed validation and was not installed:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 1
    except rel.DatasetReleaseError as exc:
        print(f"REJECTED: {exc}. Nothing was installed.")
        return 1
    except OSError as exc:
        print(f"FAILED: {exc}. If the Streamlit app is running, close it and retry.")
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("=== installed historical dataset ===")
    print(f"database:     {dest}")
    print(f"scope:        {info.data_scope}")
    print(f"coverage:     {info.coverage_start.isoformat()} to {info.coverage_end.isoformat()}")
    print(f"refreshed at: {info.refreshed_at} UTC")
    print(f"default analysis period ends {info.analysis_coverage['default_period_end_month']} "
          f"(earliest valid end month {info.analysis_coverage['earliest_valid_period_end_month']})")
    for table, n in info.table_row_counts.items():
        print(f"  {table:<28} {n:>12,}")
    print("Start the app with: python -m streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
