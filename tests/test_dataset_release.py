"""Dataset release packaging and verified installation.

Covers src/dataset_release.py and the two CLIs scripts/package_dataset_release.py and
scripts/download_dataset_release.py. Databases are built from the synthetic fixtures with real refresh-metadata
source names; the download path is exercised against a loopback HTTP server.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import zipfile
from datetime import date, datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src import db as dbmod
from src import dataset_release as rel
from src.analytics import write_default_scores_snapshot
from src.data_ingestion import SOURCE_OTP, SOURCE_OURAIRPORTS, SOURCE_T100
from tests.conftest import load_fixture_db

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
REFRESHED = datetime(2026, 10, 1, 0, 0, 0)
EXPECTED_ARCHIVE = "airport_intelligence_demo_2023-01_to_2026-02.zip"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_script = _load_script("package_dataset_release")
download_script = _load_script("download_dataset_release")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _write_metadata(conn, statuses: dict[str, str] | None = None, *, real_sources: bool = True) -> None:
    statuses = statuses or {}
    conn.execute("DELETE FROM data_refresh_metadata")
    if not real_sources:
        dbmod.record_refresh_metadata(conn, source="FIXTURE only", data_scope="demo", coverage_start=date(2023, 1, 1),
                                      coverage_end=date(2025, 12, 31), record_count=1, status="ok", refreshed_at=REFRESHED)
        return
    rows = [
        (SOURCE_T100, date(2023, 1, 1), date(2025, 12, 31), 755),
        (SOURCE_OTP, date(2023, 1, 1), date(2026, 2, 28), 798),
        (SOURCE_OURAIRPORTS, None, None, 21),
    ]
    for source, start, end, n in rows:
        dbmod.record_refresh_metadata(conn, source=source, data_scope="demo", coverage_start=start, coverage_end=end,
                                      record_count=n, status=statuses.get(source, "ok"), notes="fixture", refreshed_at=REFRESHED)


def make_database(path: Path, *, statuses: dict[str, str] | None = None, drop_table: str | None = None,
                  truncate_history: bool = False, with_scores: bool = True, real_sources: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = dbmod.connect(path)
    try:
        load_fixture_db(conn)
        _write_metadata(conn, statuses, real_sources=real_sources)
        if with_scores:
            assert write_default_scores_snapshot(conn, "demo") > 0
        if truncate_history:
            conn.execute("DELETE FROM airport_monthly_traffic WHERE year_month < DATE '2025-01-01'")
            conn.execute("DELETE FROM airport_monthly_operations WHERE year_month < DATE '2025-01-01'")
        if drop_table:
            conn.execute(f"DROP TABLE {drop_table}")
    finally:
        conn.close()
    return path


def _members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


def _manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(rel.MANIFEST_FILENAME))


def _manifest_for(payload: bytes, **overrides) -> bytes:
    m = {
        "format_version": rel.MANIFEST_FORMAT_VERSION,
        "dataset_filename": rel.DATASET_FILENAME,
        "data_scope": "demo",
        "coverage_start": "2023-01-01",
        "coverage_end": "2026-02-28",
        "refreshed_at": "2026-10-01T00:00:00",
        "sources": [],
        "table_row_counts": {},
        "duckdb_sha256": hashlib.sha256(payload).hexdigest(),
        "duckdb_size_bytes": len(payload),
    }
    m.update(overrides)
    return json.dumps(m).encode("utf-8")


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _no_leftovers(root: Path) -> bool:
    return not list(root.rglob("*.tmp")) and not list(root.rglob("*.partial")) and not list(root.rglob("*.part"))


@pytest.fixture()
def valid_db(tmp_path: Path) -> Path:
    return make_database(tmp_path / "source" / "airport_intelligence.duckdb")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # keep pytest output clean
        pass


@pytest.fixture()
def http_server(tmp_path: Path, monkeypatch):
    root = tmp_path / "served"
    root.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    try:
        yield root, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _publish(db: Path, root: Path) -> tuple[str, str]:
    artifact = rel.package_release(db, root)
    return artifact.archive_path.name, artifact.archive_sha256


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------

def test_package_valid_database_creates_archive_manifest_and_checksum(valid_db, tmp_path, capsys):
    out = tmp_path / "dist"
    assert package_script.main(["--db", str(valid_db), "--out-dir", str(out)]) == 0
    archive = out / EXPECTED_ARCHIVE
    checksum = out / (EXPECTED_ARCHIVE + ".sha256")
    assert sorted(p.name for p in out.iterdir()) == [archive.name, checksum.name]

    assert _members(archive) == sorted(rel.ARCHIVE_MEMBERS)
    m = _manifest(archive)
    assert m["format_version"] == 1
    assert m["dataset_filename"] == "airport_intelligence.duckdb"
    assert m["archive_name"] == EXPECTED_ARCHIVE
    assert m["data_scope"] == "demo"
    assert (m["coverage_start"], m["coverage_end"]) == ("2023-01-01", "2026-02-28")
    assert m["refreshed_at"].startswith("2026-10-01T00:00:00")
    assert {s["source"]: s["status"] for s in m["sources"]} == {SOURCE_T100: "ok", SOURCE_OTP: "ok", SOURCE_OURAIRPORTS: "ok"}
    assert m["table_row_counts"]["airports"] == 21
    assert m["table_row_counts"]["airport_scores"] > 0
    assert set(m["table_row_counts"]) == set(dbmod.REQUIRED_TABLES)
    assert m["duckdb_sha256"] == rel.sha256_file(valid_db)
    assert m["duckdb_size_bytes"] == valid_db.stat().st_size
    assert m["analysis_coverage"]["two_window_ready"] is True
    assert m["generated_at"].endswith("Z")
    statement = m["content_statement"].lower()
    assert "aggregated historical" in statement and "no raw" in statement and "no live api" in statement

    with zipfile.ZipFile(archive) as zf:
        assert hashlib.sha256(zf.read(rel.DATASET_FILENAME)).hexdigest() == m["duckdb_sha256"]
    digest, sep, name = checksum.read_text(encoding="ascii").strip().partition("  ")
    assert sep and digest == rel.sha256_file(archive) and name == archive.name

    text = capsys.readouterr().out
    assert archive.name in text and digest in text and "Next manual steps" in text and "scope:           demo" in text


def test_package_refuses_to_overwrite_without_flag(valid_db, tmp_path, capsys):
    out = tmp_path / "dist"
    assert package_script.main(["--db", str(valid_db), "--out-dir", str(out)]) == 0
    archive = out / EXPECTED_ARCHIVE
    before = rel.sha256_file(archive)
    assert package_script.main(["--db", str(valid_db), "--out-dir", str(out)]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert rel.sha256_file(archive) == before
    assert package_script.main(["--db", str(valid_db), "--out-dir", str(out), "--overwrite"]) == 0
    assert sorted(p.name for p in out.iterdir()) == [archive.name, archive.name + ".sha256"]
    assert _no_leftovers(out)


@pytest.mark.parametrize("bad_status", ["failed", "partial"])
def test_package_rejects_non_ok_source_status(tmp_path, bad_status, capsys):
    db = make_database(tmp_path / "db.duckdb", statuses={SOURCE_OTP: bad_status})
    out = tmp_path / "dist"
    assert package_script.main(["--db", str(db), "--out-dir", str(out)]) == 1
    text = capsys.readouterr().out
    assert "REJECTED" in text and f"-> {bad_status}" in text
    assert not out.exists() or not list(out.iterdir())


def test_package_rejects_missing_historical_source(tmp_path):
    db = make_database(tmp_path / "db.duckdb", real_sources=False)
    with pytest.raises(rel.DatasetValidationError, match="source missing from the latest refresh"):
        rel.inspect_database(db)


def test_package_rejects_incomplete_or_invalid_databases(tmp_path, capsys):
    cases = {
        make_database(tmp_path / "a.duckdb", drop_table="airport_scores"): "missing required tables: airport_scores",
        make_database(tmp_path / "b.duckdb", truncate_history=True): "two adjacent 12-month windows",
        make_database(tmp_path / "c.duckdb", with_scores=False): "table airport_scores is empty",
        tmp_path / "missing.duckdb": "not found",
    }
    garbage = tmp_path / "garbage.duckdb"
    garbage.write_bytes(b"definitely not a duckdb file\n" * 200)
    cases[garbage] = "not a readable DuckDB"
    empty = tmp_path / "empty.duckdb"
    empty.write_bytes(b"")
    cases[empty] = "is empty"
    for idx, (db, message) in enumerate(cases.items()):
        out = tmp_path / f"dist{idx}"
        assert package_script.main(["--db", str(db), "--out-dir", str(out)]) == 1, db
        assert message in capsys.readouterr().out, db
        assert not out.exists() or not list(out.iterdir())


def test_package_refuses_when_wal_present(valid_db, tmp_path):
    wal = valid_db.with_name(valid_db.name + ".wal")
    wal.write_bytes(b"pending")
    with pytest.raises(rel.DatasetValidationError, match="write-ahead log"):
        rel.package_release(valid_db, tmp_path / "dist")


# --------------------------------------------------------------------------------------
# Download and install
# --------------------------------------------------------------------------------------

def test_install_from_local_http_download_then_refuse_then_force(valid_db, http_server, tmp_path, capsys):
    root, base = http_server
    name, digest = _publish(valid_db, root)
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    argv = ["--url", f"{base}/{name}", "--sha256", digest, "--dest", str(dest)]

    assert download_script.main(argv) == 0
    text = capsys.readouterr().out
    assert dest.exists() and dbmod.database_ready(dest)
    assert rel.sha256_file(dest) == rel.sha256_file(valid_db)
    assert "Checksum verified" in text
    assert "scope:        demo" in text and "2023-01-01 to 2026-02-28" in text and "2026-10-01T00:00:00" in text
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]
    assert _no_leftovers(tmp_path)

    assert download_script.main(argv) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert rel.sha256_file(dest) == rel.sha256_file(valid_db)

    checksum_line = f"{digest.upper()}  {name}"      # the .sha256 file content is accepted as-is
    assert download_script.main(["--url", f"{base}/{name}", "--sha256", checksum_line, "--dest", str(dest), "--force"]) == 0
    assert dbmod.database_ready(dest) and [p.name for p in dest.parent.iterdir()] == [dest.name]


def test_install_rejects_checksum_mismatch(valid_db, http_server, tmp_path, capsys):
    root, base = http_server
    name, digest = _publish(valid_db, root)
    wrong = ("1" if digest[0] == "0" else "0") + digest[1:]
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    assert download_script.main(["--url", f"{base}/{name}", "--sha256", wrong, "--dest", str(dest)]) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().out
    assert not dest.exists() and _no_leftovers(tmp_path)


def test_install_rejects_http_errors_and_bad_arguments(http_server, tmp_path, capsys):
    root, base = http_server
    dest = tmp_path / "d.duckdb"
    assert download_script.main(["--url", f"{base}/missing.zip", "--sha256", "0" * 64, "--dest", str(dest)]) == 1
    assert "HTTP 404" in capsys.readouterr().out
    assert download_script.main(["--url", "http://example.com/a.zip", "--sha256", "0" * 64, "--dest", str(dest)]) == 3
    assert download_script.main(["--url", "https://example.com/a.zip", "--sha256", "not-hex", "--dest", str(dest)]) == 3
    assert not dest.exists()


def test_install_rejects_zip_slip_paths(valid_db, tmp_path):
    payload = valid_db.read_bytes()
    manifest = _manifest_for(payload)
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    for evil in ("../airport_intelligence.duckdb", "data/airport_intelligence.duckdb", "/tmp/airport_intelligence.duckdb",
                 "..\\airport_intelligence.duckdb", "C:\\airport_intelligence.duckdb"):
        archive = _write_zip(tmp_path / "slip.zip", {evil: payload, rel.MANIFEST_FILENAME: manifest})
        with pytest.raises(rel.ArchiveError, match="unsafe member path"):
            rel.install_archive(archive, dest)
        archive.unlink()
    assert not dest.exists() and not (tmp_path / "airport_intelligence.duckdb").exists() and _no_leftovers(tmp_path)


def test_install_rejects_extra_missing_or_malformed_archives(valid_db, tmp_path):
    payload = valid_db.read_bytes()
    manifest = _manifest_for(payload)
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    cases = [
        ({rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: manifest, "README.txt": b"extra"}, "unexpected file"),
        ({rel.DATASET_FILENAME: payload}, "missing expected file"),
        ({rel.MANIFEST_FILENAME: manifest}, "missing expected file"),
        ({rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: b"{not json"}, "unreadable"),
        ({rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: _manifest_for(payload, format_version=2)}, "unsupported manifest format_version"),
        ({rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: _manifest_for(payload, duckdb_size_bytes=len(payload) + 1)}, "manifest says"),
        ({rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: _manifest_for(payload, dataset_filename="other.duckdb")}, "dataset_filename"),
    ]
    for members, message in cases:
        archive = _write_zip(tmp_path / "case.zip", members)
        with pytest.raises(rel.ArchiveError, match=message):
            rel.install_archive(archive, dest)
        archive.unlink()
    not_zip = tmp_path / "notzip.zip"
    not_zip.write_bytes(b"PK\x03\x04 this is not a complete archive")
    with pytest.raises(rel.ArchiveError, match="not a ZIP archive"):
        rel.install_archive(not_zip, dest)
    assert not dest.exists() and _no_leftovers(tmp_path)


def test_install_rejects_invalid_duckdb_payload(tmp_path):
    payload = b"this is not a duckdb database\n" * 1000
    archive = _write_zip(tmp_path / "garbage.zip", {rel.DATASET_FILENAME: payload, rel.MANIFEST_FILENAME: _manifest_for(payload)})
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    with pytest.raises(rel.DatasetValidationError, match="not a readable DuckDB"):
        rel.install_archive(archive, dest)
    assert not dest.exists() and list(dest.parent.iterdir()) == []


def test_install_rejects_incomplete_failed_or_tampered_databases(tmp_path):
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    incomplete = make_database(tmp_path / "incomplete.duckdb", drop_table="route_monthly_metrics").read_bytes()
    archive = _write_zip(tmp_path / "incomplete.zip", {rel.DATASET_FILENAME: incomplete, rel.MANIFEST_FILENAME: _manifest_for(incomplete)})
    with pytest.raises(rel.DatasetValidationError, match="missing required tables: route_monthly_metrics"):
        rel.install_archive(archive, dest)

    failed = make_database(tmp_path / "failed.duckdb", statuses={SOURCE_T100: "failed"}).read_bytes()
    archive = _write_zip(tmp_path / "failed.zip", {rel.DATASET_FILENAME: failed, rel.MANIFEST_FILENAME: _manifest_for(failed)})
    with pytest.raises(rel.DatasetValidationError, match="status is not ok"):
        rel.install_archive(archive, dest)

    valid = make_database(tmp_path / "valid.duckdb").read_bytes()
    archive = _write_zip(tmp_path / "tampered.zip", {rel.DATASET_FILENAME: valid, rel.MANIFEST_FILENAME: _manifest_for(valid, duckdb_sha256="0" * 64)})
    with pytest.raises(rel.ArchiveError, match="does not match the SHA-256 recorded in the manifest"):
        rel.install_archive(archive, dest)

    archive = _write_zip(tmp_path / "scope.zip", {rel.DATASET_FILENAME: valid, rel.MANIFEST_FILENAME: _manifest_for(valid, data_scope="nationwide")})
    with pytest.raises(rel.ArchiveError, match="differs from database metadata"):
        rel.install_archive(archive, dest)
    assert not dest.exists() and list(dest.parent.iterdir()) == []


def test_install_archive_refuses_existing_destination_without_force(valid_db, tmp_path):
    archive = rel.package_release(valid_db, tmp_path / "dist").archive_path
    dest = tmp_path / "data" / "airport_intelligence.duckdb"
    dest.parent.mkdir()
    dest.write_bytes(b"existing")
    with pytest.raises(rel.ArtifactExistsError):
        rel.install_archive(archive, dest)
    assert dest.read_bytes() == b"existing"
    info = rel.install_archive(archive, dest, force=True)
    assert info.data_scope == "demo" and info.path == dest and dbmod.database_ready(dest)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, ok", [
    ("airport_intelligence.duckdb", True), ("dataset_manifest.json", True),
    ("../x.duckdb", False), ("a/b.duckdb", False), ("/abs.duckdb", False), ("C:\\x.duckdb", False), ("..", False), ("", False), (".hidden", False),
])
def test_safe_member_names(name, ok):
    assert rel.is_safe_member_name(name) is ok


def test_normalize_sha256_and_url_rules():
    digest = "ab" * 32
    assert rel.normalize_sha256(digest.upper()) == digest
    assert rel.normalize_sha256(f"{digest}  airport.zip\n") == digest
    with pytest.raises(ValueError):
        rel.normalize_sha256("zz" * 32)
    assert rel.validate_asset_url("https://github.com/o/r/releases/download/t/a.zip")
    assert rel.validate_asset_url("http://127.0.0.1:8000/a.zip")
    with pytest.raises(ValueError):
        rel.validate_asset_url("http://example.com/a.zip")
    with pytest.raises(ValueError):
        rel.validate_asset_url("ftp://example.com/a.zip")
