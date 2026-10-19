"""Prepared historical dataset distribution: validate, package, verify and install the aggregated DuckDB store.

`scripts/package_dataset_release.py` and `scripts/download_dataset_release.py` are thin wrappers around this module.
Everything here works on the already-aggregated DuckDB file produced by `scripts/refresh_data.py`; it never reads raw
BTS files, never calls a live provider and never changes scoring, ingestion or the UI.

Release asset layout (ZIP, no enclosing directory):
    airport_intelligence.duckdb
    dataset_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.parse import urlparse

import duckdb
import requests

from src import db as dbmod
from src.analytics import get_coverage
from src.data_ingestion import SOURCE_OTP, SOURCE_T100, use_system_trust_store

PathLike = Union[str, Path]

DATASET_FILENAME = "airport_intelligence.duckdb"
MANIFEST_FILENAME = "dataset_manifest.json"
ARCHIVE_MEMBERS = frozenset({DATASET_FILENAME, MANIFEST_FILENAME})
MANIFEST_FORMAT_VERSION = 1
OK_STATUS = "ok"

# The two BTS sources define the historical coverage window and must be present with status ok.
# Every other source recorded for the latest refresh must also be ok.
HISTORICAL_SOURCES: tuple[str, ...] = (SOURCE_T100, SOURCE_OTP)

CONTENT_STATEMENT = (
    "Aggregated historical data only: monthly airport, route and score tables derived from BTS TranStats T-100 Segment "
    "and Reporting Carrier On-Time Performance data plus the OurAirports airport reference. The archive contains no raw "
    "BTS ZIP/CSV files, no live API data, no API keys and no credentials."
)
EXCLUDED_CONTENT = [
    "raw BTS ZIP/CSV downloads (data/raw/)",
    "live Aviationstack API responses",
    ".env files, API keys, credentials",
    "logs",
]

CHUNK_SIZE = 1024 * 1024
MAX_MANIFEST_BYTES = 1_000_000
DEFAULT_MAX_DOWNLOAD_BYTES = 16 * 1024 ** 3
USER_AGENT = "airport-investment-intelligence-agent dataset installer"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_MEMBER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------

class DatasetReleaseError(Exception):
    """Base class: every failure in this module is reported through one of these."""


class DatasetValidationError(DatasetReleaseError):
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class ArchiveError(DatasetReleaseError):
    """The ZIP archive is malformed, unsafe or does not match its manifest."""


class ChecksumMismatchError(DatasetReleaseError):
    def __init__(self, expected: str, actual: str):
        self.expected, self.actual = expected, actual
        super().__init__(f"SHA-256 mismatch: expected {expected}, got {actual}")


class DownloadError(DatasetReleaseError):
    """Network or HTTP failure while fetching the asset."""


class ArtifactExistsError(DatasetReleaseError):
    """A file that the operation would overwrite already exists and no override flag was given."""


# --------------------------------------------------------------------------------------
# Database inspection (uses the project's own helpers)
# --------------------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    path: Path
    size_bytes: int
    data_scope: str
    refreshed_at: str
    coverage_start: date
    coverage_end: date
    sources: list[dict]
    table_row_counts: dict[str, int]
    analysis_coverage: dict

    @property
    def coverage_label(self) -> str:
        return f"{self.coverage_start:%Y-%m} to {self.coverage_end:%Y-%m}"


def _iso(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def inspect_database(path: PathLike) -> DatasetInfo:
    """Validate a DuckDB store with the project's helpers and summarize its metadata.

    Raises DatasetValidationError listing every problem found. A database passes only when it has all required tables
    with rows, metadata from `scripts/refresh_data.py`, both BTS historical sources, every source status `ok`, and
    enough history for the application's two adjacent 12-month windows.
    """
    p = Path(path)
    if not p.is_file():
        raise DatasetValidationError([f"database file not found: {p}"])
    size = p.stat().st_size
    if size == 0:
        raise DatasetValidationError([f"database file is empty: {p}"])
    try:
        conn = dbmod.connect(p, read_only=True)
    except duckdb.Error as exc:
        raise DatasetValidationError([f"not a readable DuckDB database ({type(exc).__name__}: {exc})"]) from exc
    try:
        missing = dbmod.missing_tables(conn)
        if missing:
            raise DatasetValidationError([f"missing required tables: {', '.join(missing)}"])
        counts = {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in dbmod.REQUIRED_TABLES}
        problems: list[str] = []
        for table, n in counts.items():
            if n == 0:
                hint = " (run: python scripts/refresh_data.py --scores-only)" if table == "airport_scores" else ""
                problems.append(f"table {table} is empty{hint}")
        latest = dbmod.latest_refresh(conn)
        if latest is None:
            problems.append("data_refresh_metadata is empty; the database was not produced by scripts/refresh_data.py")
            raise DatasetValidationError(problems)
        sources = latest["sources"]
        by_name = {s["source"]: s for s in sources}
        for name in HISTORICAL_SOURCES:
            if name not in by_name:
                problems.append(f"source missing from the latest refresh ({latest['data_scope']}): {name}")
        not_ok = [f"{s['source']} -> {s['status']}" for s in sources if s["status"] != OK_STATUS]
        if not_ok:
            problems.append("source status is not ok: " + "; ".join(not_ok))
        starts = [s["coverage_start"] for s in sources if s["source"] in HISTORICAL_SOURCES and s["coverage_start"]]
        ends = [s["coverage_end"] for s in sources if s["source"] in HISTORICAL_SOURCES and s["coverage_end"]]
        if not starts or not ends:
            problems.append("the BTS historical sources carry no coverage dates")
        cov = get_coverage(conn)
        analysis = {
            "traffic_first": cov.traffic_first,
            "traffic_last": cov.traffic_last,
            "operations_first": cov.ops_first,
            "operations_last": cov.ops_last,
            "routes_last": cov.routes_last,
            "default_period_end_month": cov.common_end_month,
            "earliest_valid_period_end_month": cov.earliest_valid_end_month,
            "two_window_ready": bool(cov.ready),
        }
        if not cov.ready:
            problems.append("historical coverage is too short for two adjacent 12-month windows (24 months of traffic and operations)")
        if problems:
            raise DatasetValidationError(problems)
        return DatasetInfo(
            path=p,
            size_bytes=size,
            data_scope=str(latest["data_scope"]),
            refreshed_at=str(latest["refreshed_at"]),
            coverage_start=min(starts),
            coverage_end=max(ends),
            sources=[
                {
                    "source": s["source"],
                    "status": s["status"],
                    "coverage_start": _iso(s["coverage_start"]),
                    "coverage_end": _iso(s["coverage_end"]),
                    "record_count": int(s["record_count"]),
                    "notes": s["notes"],
                }
                for s in sources
            ],
            table_row_counts=counts,
            analysis_coverage=analysis,
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------------------
# Hashing and naming
# --------------------------------------------------------------------------------------

def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_sha256(value: str) -> str:
    """Accept a bare hex digest or the first token of a `<hex>  <name>` checksum line."""
    token = (value or "").strip().split()
    digest = token[0].lower() if token else ""
    if not _SHA256_RE.match(digest):
        raise ValueError("expected a 64-character hexadecimal SHA-256 digest")
    return digest


def archive_stem(info: DatasetInfo) -> str:
    scope = re.sub(r"[^a-z0-9]+", "-", info.data_scope.lower()).strip("-") or "dataset"
    return f"airport_intelligence_{scope}_{info.coverage_start:%Y-%m}_to_{info.coverage_end:%Y-%m}"


def build_manifest(info: DatasetInfo, duckdb_sha256: str, archive_name: str, generated_at: Optional[datetime] = None) -> dict:
    now = generated_at or datetime.now(timezone.utc)
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "dataset_filename": DATASET_FILENAME,
        "archive_name": archive_name,
        "data_scope": info.data_scope,
        "coverage_start": info.coverage_start.isoformat(),
        "coverage_end": info.coverage_end.isoformat(),
        "refreshed_at": info.refreshed_at,
        "refreshed_at_timezone": "UTC",
        "sources": info.sources,
        "table_row_counts": info.table_row_counts,
        "analysis_coverage": info.analysis_coverage,
        "duckdb_sha256": duckdb_sha256,
        "duckdb_size_bytes": info.size_bytes,
        "duckdb_library_version": duckdb.__version__,
        "generated_at": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/package_dataset_release.py",
        "content_statement": CONTENT_STATEMENT,
        "excludes": list(EXCLUDED_CONTENT),
    }


def validate_manifest(manifest) -> dict:
    if not isinstance(manifest, dict):
        raise ArchiveError("dataset_manifest.json is not a JSON object")
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ArchiveError(f"unsupported manifest format_version {manifest.get('format_version')!r} (expected {MANIFEST_FORMAT_VERSION})")
    if manifest.get("dataset_filename") != DATASET_FILENAME:
        raise ArchiveError(f"manifest dataset_filename {manifest.get('dataset_filename')!r} is not {DATASET_FILENAME!r}")
    sha = manifest.get("duckdb_sha256")
    if not isinstance(sha, str) or not _SHA256_RE.match(sha.lower()):
        raise ArchiveError("manifest duckdb_sha256 is missing or malformed")
    size = manifest.get("duckdb_size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ArchiveError("manifest duckdb_size_bytes is missing or malformed")
    for key in ("data_scope", "coverage_start", "coverage_end", "refreshed_at", "sources", "table_row_counts"):
        if key not in manifest:
            raise ArchiveError(f"manifest is missing {key!r}")
    return manifest


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------

@dataclass
class ReleaseArtifact:
    archive_path: Path
    checksum_path: Path
    archive_sha256: str
    archive_size_bytes: int
    manifest: dict
    info: DatasetInfo


def _member_sha256(zf: zipfile.ZipFile, member: zipfile.ZipInfo) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with zf.open(member) as src:
        for chunk in iter(lambda: src.read(CHUNK_SIZE), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def package_release(db_path: PathLike, out_dir: PathLike, *, overwrite: bool = False,
                    generated_at: Optional[datetime] = None) -> ReleaseArtifact:
    """Validate the database, then write `<out_dir>/<stem>.zip` and `<stem>.zip.sha256`.

    The ZIP is built under a temporary name and moved into place only after its members were re-read and the DuckDB
    member's SHA-256 matched the hash taken from the source file. Existing artifacts are never overwritten silently.
    """
    db = Path(db_path)
    wal = db.with_name(db.name + ".wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise DatasetValidationError([f"a write-ahead log exists next to the database ({wal.name}); let the writer finish and "
                                      "check-point the file (open and close it once with DuckDB) before packaging"])
    info = inspect_database(db)
    stem = archive_stem(info)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"{stem}.zip"
    checksum = archive.with_name(archive.name + ".sha256")
    existing = [p for p in (archive, checksum) if p.exists()]
    if existing and not overwrite:
        raise ArtifactExistsError("refusing to overwrite existing artifact(s): " + ", ".join(str(p) for p in existing)
                                  + " (pass --overwrite to replace them)")

    duckdb_sha = sha256_file(db)
    manifest = build_manifest(info, duckdb_sha, archive.name, generated_at)
    partial = archive.with_name(f".{archive.name}.{os.getpid()}.partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(db, arcname=DATASET_FILENAME)
            zf.writestr(MANIFEST_FILENAME, json.dumps(manifest, indent=2) + "\n")
        # self-check: exactly the two members, and the archived database is byte-identical to the hashed source
        dinfo, archived_manifest = inspect_archive(partial)
        with zipfile.ZipFile(partial) as zf:
            member_sha, member_size = _member_sha256(zf, dinfo)
        if member_sha != duckdb_sha or member_size != info.size_bytes:
            raise ArchiveError("the archived database does not match the source file (was it modified during packaging?)")
        if archived_manifest["duckdb_sha256"] != duckdb_sha:
            raise ArchiveError("the archived manifest does not carry the expected DuckDB hash")
        for p in existing:
            p.unlink()
        os.replace(partial, archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    archive_sha = sha256_file(archive)
    checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    return ReleaseArtifact(
        archive_path=archive,
        checksum_path=checksum,
        archive_sha256=archive_sha,
        archive_size_bytes=archive.stat().st_size,
        manifest=manifest,
        info=info,
    )


# --------------------------------------------------------------------------------------
# Archive inspection and installation
# --------------------------------------------------------------------------------------

def is_safe_member_name(name: str) -> bool:
    """Plain file names only: no directories, separators, traversal, absolute paths or drive letters."""
    return bool(name) and name not in (".", "..") and bool(_SAFE_MEMBER_RE.match(name))


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(member.external_attr >> 16)


def inspect_archive(zip_path: PathLike) -> tuple[zipfile.ZipInfo, dict]:
    """Check the ZIP structure and manifest without extracting anything. Returns (DuckDB member, manifest)."""
    path = Path(zip_path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ArchiveError(f"not a ZIP archive: {path.name}")
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"malformed ZIP archive: {exc}") from exc
    with zf:
        members = zf.infolist()
        for m in members:
            if not is_safe_member_name(m.filename):
                raise ArchiveError(f"unsafe member path in archive: {m.filename!r}")
            if m.is_dir():
                raise ArchiveError(f"directory entries are not allowed: {m.filename!r}")
            if _is_symlink(m):
                raise ArchiveError(f"symbolic links are not allowed: {m.filename!r}")
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            raise ArchiveError("duplicate member names in archive")
        missing = ARCHIVE_MEMBERS - set(names)
        extra = set(names) - ARCHIVE_MEMBERS
        if missing:
            raise ArchiveError("archive is missing expected file(s): " + ", ".join(sorted(missing)))
        if extra:
            raise ArchiveError("archive contains unexpected file(s): " + ", ".join(sorted(extra)))
        minfo = zf.getinfo(MANIFEST_FILENAME)
        if minfo.file_size > MAX_MANIFEST_BYTES:
            raise ArchiveError("dataset_manifest.json is unreasonably large")
        try:
            manifest = json.loads(zf.read(minfo).decode("utf-8"))
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArchiveError(f"dataset_manifest.json is unreadable: {exc}") from exc
        validate_manifest(manifest)
        dinfo = zf.getinfo(DATASET_FILENAME)
        if dinfo.file_size != manifest["duckdb_size_bytes"]:
            raise ArchiveError(f"archive declares a {dinfo.file_size:,}-byte database but the manifest says {manifest['duckdb_size_bytes']:,}")
        return dinfo, manifest


def install_archive(zip_path: PathLike, dest: PathLike, *, force: bool = False) -> DatasetInfo:
    """Extract only the DuckDB file next to `dest`, verify it, then atomically replace `dest`.

    Verification order: manifest hash and size of the extracted bytes, then the project's database validation
    (required tables, metadata, source statuses, two-window coverage). The destination is never touched until the new
    file has passed every check.
    """
    target = Path(dest)
    if target.exists() and not force:
        raise ArtifactExistsError(f"refusing to overwrite the existing database {target} (pass --force to replace it)")
    dinfo, manifest = inspect_archive(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".download.tmp")
    try:
        h = hashlib.sha256()
        written = 0
        with zipfile.ZipFile(zip_path) as zf, zf.open(dinfo) as src, open(tmp, "wb") as out:
            for chunk in iter(lambda: src.read(CHUNK_SIZE), b""):
                written += len(chunk)
                if written > dinfo.file_size:
                    raise ArchiveError("the database member is larger than its declared size")
                h.update(chunk)
                out.write(chunk)
        if written != manifest["duckdb_size_bytes"]:
            raise ArchiveError(f"extracted {written:,} bytes but the manifest declares {manifest['duckdb_size_bytes']:,}")
        if h.hexdigest() != manifest["duckdb_sha256"].lower():
            raise ArchiveError("the extracted database does not match the SHA-256 recorded in the manifest")
        info = inspect_database(tmp)
        if info.data_scope != manifest["data_scope"]:
            raise ArchiveError(f"manifest scope {manifest['data_scope']!r} differs from database metadata {info.data_scope!r}")
        if target.exists() and not force:
            raise ArtifactExistsError(f"refusing to overwrite the existing database {target} (pass --force to replace it)")
        dbmod.atomic_replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    info.path = target
    return info


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_asset_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return url
    if parsed.scheme == "http" and (parsed.hostname or "") in _LOOPBACK_HOSTS:
        return url  # local test servers only
    raise ValueError("the asset URL must use https:// (plain http is accepted only for localhost)")


def download_file(url: str, target: PathLike, *, expected_sha256: str, timeout: float = 60.0,
                  max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES, session: Optional[requests.Session] = None,
                  progress: Optional[Callable[[int, Optional[int]], None]] = None) -> str:
    """Stream `url` to `target`, hashing as bytes arrive. Raises ChecksumMismatchError when the digest differs."""
    validate_asset_url(url)
    expected = normalize_sha256(expected_sha256)
    if session is None:
        use_system_trust_store()
        session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream"}
    h = hashlib.sha256()
    received = 0
    try:
        with session.get(url, stream=True, timeout=timeout, headers=headers, allow_redirects=True) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"HTTP {resp.status_code} for {url}")
            declared: Optional[int] = None
            raw_len = resp.headers.get("Content-Length")
            if raw_len and raw_len.isdigit():
                declared = int(raw_len)
                if declared > max_bytes:
                    raise DownloadError(f"asset is {declared:,} bytes, above the {max_bytes:,}-byte limit")
            with open(target, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > max_bytes:
                        raise DownloadError(f"download exceeded the {max_bytes:,}-byte limit")
                    h.update(chunk)
                    fh.write(chunk)
                    if progress:
                        progress(received, declared)
            if declared is not None and received != declared:
                raise DownloadError(f"incomplete download: received {received:,} of {declared:,} bytes")
    except requests.RequestException as exc:
        raise DownloadError(f"download failed: {exc}") from exc
    actual = h.hexdigest()
    if actual != expected:
        raise ChecksumMismatchError(expected, actual)
    return actual


__all__ = [
    "DATASET_FILENAME", "MANIFEST_FILENAME", "ARCHIVE_MEMBERS", "MANIFEST_FORMAT_VERSION", "HISTORICAL_SOURCES",
    "CONTENT_STATEMENT", "DatasetReleaseError", "DatasetValidationError", "ArchiveError", "ChecksumMismatchError",
    "DownloadError", "ArtifactExistsError", "DatasetInfo", "ReleaseArtifact", "inspect_database", "sha256_file",
    "normalize_sha256", "archive_stem", "build_manifest", "validate_manifest", "package_release", "is_safe_member_name",
    "inspect_archive", "install_archive", "validate_asset_url", "download_file",
]
