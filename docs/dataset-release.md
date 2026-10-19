# Publishing the prepared historical dataset

This is the manual process for distributing the aggregated DuckDB store as a GitHub Release asset so that a
reviewer can clone the repository, download one file and run the Streamlit app without the 1-2.5 GB BTS download.

The asset contains **real, already-aggregated historical data only**: the six DuckDB tables written by
`scripts/refresh_data.py`. It never contains raw BTS ZIP/CSV files, live API responses, `.env` files, API keys or logs.
Nothing in this process is automated against GitHub: the scripts create files locally; a maintainer uploads them.

## Asset layout

```text
airport_intelligence_<scope>_<YYYY-MM>_to_<YYYY-MM>.zip          e.g. airport_intelligence_nationwide_2019-01_to_2026-07.zip
  airport_intelligence.duckdb       the aggregated store (no enclosing data/ directory)
  dataset_manifest.json             scope, coverage, refresh timestamp, sources, row counts, DuckDB SHA-256, content statement
airport_intelligence_<scope>_<YYYY-MM>_to_<YYYY-MM>.zip.sha256    "<sha256>  <archive name>" (sha256sum format)
```

The name is derived from `data_refresh_metadata` (scope, earliest coverage start and latest coverage end of the two
BTS sources); it is never typed by hand.

## Step by step

1. **Refresh nationwide data locally.**

   ```bash
   python scripts/refresh_data.py --scope nationwide
   ```

   The command exits with code 2 and records `failed`/`partial` in the metadata table if any source failed. Do not
   package such a database; fix the cause and re-run (`--keep-raw` reuses cached downloads).

2. **Validate the metadata.**

   ```bash
   python scripts/refresh_data.py --show-metadata
   ```

   Every row of `data_refresh_metadata` for the latest scope must show `status = ok`, and both BTS sources
   (T-100 Segment and Reporting Carrier On-Time Performance) must be present with coverage dates. Close the Streamlit
   app before packaging if it is running (read-only cursors do not block packaging, but a stale `.wal` file does).

3. **Run the package script.**

   ```bash
   python scripts/package_dataset_release.py
   ```

   It re-validates the database with the project's own helpers (required tables with rows, metadata present, all
   source statuses `ok`, at least 24 months of traffic and operations for the two adjacent 12-month windows), hashes
   the DuckDB file, writes `dist/<name>.zip` and `dist/<name>.zip.sha256`, re-reads the archive to confirm it holds
   exactly the two allowed members with the expected hash, and prints a summary. It refuses to overwrite an existing
   artifact unless `--overwrite` is passed. Options: `--db`, `--out-dir`, `--overwrite`.

   Keep the printed `archive sha256` value: it goes into the README and is what the download script verifies.

4. **Create a GitHub Release manually.** On the repository page open *Releases -> Draft a new release*. Create a
   tag that names the dataset, for example `dataset-nationwide-2026-07`, and a title such as
   "Historical dataset: nationwide, 2019-01 to 2026-07 (refreshed 2026-10-19)". Paste the summary printed by the
   package script (scope, coverage, refresh timestamp, table counts, both SHA-256 values) into the release notes.

5. **Upload only the generated ZIP and `.sha256` file** from `dist/`. Nothing else. Publish the release.

6. **Copy the asset URL into the README.** After publishing, right-click the ZIP asset and copy its link
   (`https://github.com/<owner>/<repo>/releases/download/<tag>/<archive name>`). Replace the two placeholders in the
   README section "Quick start with prepared historical data" with that URL and the archive SHA-256 from step 3,
   and remove the "no asset published yet" note. Commit that README change only; `dist/` stays untracked.

7. **Verify the installation in a clean environment.** In a fresh clone with a fresh virtual environment:

   ```bash
   python -m pip install -r requirements.txt
   ```

   ```bash
   python scripts/download_dataset_release.py --url <asset URL> --sha256 <archive sha256>
   ```

   ```bash
   python -m streamlit run app.py
   ```

   The download script must print `Checksum verified`, the installed scope, coverage and refresh timestamp, and the
   app sidebar must show the same scope and refresh time. A reviewer with a pre-existing database must pass `--force`.

8. **Never upload `data/raw/`, `.env`, API keys, logs, or any file other than the two artifacts.** The repository's
   `.gitignore` excludes `dist/`, `*.zip`, `*.sha256`, `*.duckdb`, `data/raw/`, `.env` and `logs/`; the package
   script reads only the DuckDB file it is pointed at.

## What the download script enforces

- `https://` URLs only (plain `http://` is accepted only for `localhost`, used by the tests).
- Streams to a temporary directory; the SHA-256 is compared **before** the archive is opened.
- Archive must contain exactly `airport_intelligence.duckdb` and `dataset_manifest.json` as plain file names. Nested
  paths, `..` segments, absolute paths, directories, symbolic links, duplicates and extra files are rejected.
- Manifest must be `format_version` 1 with a matching dataset name, a well-formed DuckDB SHA-256 and a declared size
  that matches the ZIP member.
- Only the DuckDB member is extracted, next to the destination; its bytes are hashed while extracting and must match
  the manifest. The file is then validated with `src/db.py` and `src/analytics.py` helpers (tables, metadata, source
  statuses, two-window coverage) and installed with the same atomic replace used by the refresh.
- An existing local database is never replaced without `--force`. Temporary files are removed on success and failure.

## Exit codes

| Script | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `package_dataset_release.py` | archive written | database rejected or packaging failed | artifact exists (use `--overwrite`) | bad arguments |
| `download_dataset_release.py` | installed | download, verification or installation failed | destination exists (use `--force`) | bad arguments |

## Compatibility note

The DuckDB storage format is tied to the library version that wrote the file; the manifest records it under
`duckdb_library_version`. `requirements.txt` pins `duckdb>=1.5,<2`; publish a new asset after a refresh with a newer
major DuckDB release rather than asking reviewers to downgrade.
