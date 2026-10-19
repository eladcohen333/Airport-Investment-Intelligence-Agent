"""Historical data ingestion from public BTS TranStats sources into DuckDB monthly aggregates.

Sources
  * On-Time Performance (operations): static monthly zips under https://transtats.bts.gov/PREZIP/
  * T-100 Segment (All Carriers) (traffic + routes): TranStats download form (ASP.NET postback), one file per year
  * OurAirports airports.csv: public-domain airport reference (names, coordinates, type)

Raw files are streamed to a temporary (or opt-in cache) directory, aggregated with DuckDB and discarded.
The application never reads raw files.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import duckdb
import pandas as pd
import requests

from src import db as dbmod
from src.airport_resolver import generate_aliases
from src.config import get_logger
from src.models import NEW_ENGLAND_STATES, date_to_year_month, month_add, month_diff, month_end, month_start

log = get_logger("ingestion")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

OTP_PREZIP_URL = (
    "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
T100_FORM_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4+Pn44vr45"
OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

SOURCE_OTP = "BTS TranStats: Reporting Carrier On-Time Performance (1987-present)"
SOURCE_T100 = "BTS TranStats: T-100 Segment (All Carriers)"
SOURCE_OURAIRPORTS = "OurAirports airports.csv (public-domain airport reference)"
SOURCE_AIRPORTS_DERIVED = "Airport reference derived from BTS city/state fields"

TRAFFIC_SERVICE_CLASSES: tuple[str, ...] = ("F",)   # scheduled passenger / combination service
DEFAULT_LONG_HAUL_MILES = 2000

T100_VARIABLES = [
    "DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS", "DISTANCE", "UNIQUE_CARRIER",
    "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_COUNTRY",
    "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "DEST_COUNTRY", "YEAR", "MONTH", "CLASS",
]
T100_REQUIRED = set(T100_VARIABLES)

OTP_REQUIRED = ["Year", "Month", "Origin", "OriginState", "Dest", "DepDelayMinutes", "DepDel15", "Cancelled", "Diverted", "Distance"]
OTP_TYPES = {
    "Year": "INTEGER", "Month": "INTEGER", "Origin": "VARCHAR", "OriginState": "VARCHAR", "Dest": "VARCHAR",
    "DepDelay": "DOUBLE", "DepDelayMinutes": "DOUBLE", "DepDel15": "DOUBLE", "Cancelled": "DOUBLE", "Diverted": "DOUBLE",
    "Flights": "DOUBLE", "Distance": "DOUBLE",
}

SIZE_BAND_THRESHOLDS = {"large": 10_000_000, "medium": 2_500_000, "small": 500_000}   # annual origin passengers

# Lower bound admits the April-June 2020 COVID trough (reporting-carrier flights dropped to roughly 180k-220k per month).
NATIONAL_MONTHLY_FLIGHTS_RANGE = (150_000, 900_000)
AIRPORT_CODE_RE = re.compile(r"^[A-Z0-9]{3}$")
USER_AGENT = "AirportInvestmentIntelligencePOC/0.1 (+local research tool)"


class SourceError(RuntimeError):
    """A public data source could not be retrieved or returned an unexpected payload."""


class DataValidationError(RuntimeError):
    """Downloaded data failed structural or range validation."""


# --------------------------------------------------------------------------------------
# HTTP helper with retries
# --------------------------------------------------------------------------------------

def use_system_trust_store() -> bool:
    """Make TLS verification use the operating-system certificate store (corporate TLS-inspection roots).

    `requests` ships its own CA bundle (certifi); on managed Windows machines the inspection root is only in the
    Windows store, so downloads fail with CERTIFICATE_VERIFY_FAILED. `truststore` fixes that without disabling
    verification. Alternatively set REQUESTS_CA_BUNDLE / SSL_CERT_FILE to a bundle that includes the root.
    """
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        return True
    except Exception:  # pragma: no cover - optional dependency
        return False


class HttpClient:
    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 120.0, retries: int = 3, backoff: float = 2.0):
        if session is None:
            use_system_trust_store()
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def request(self, method: str, url: str, *, stream: bool = False, **kwargs) -> requests.Response:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, stream=stream, **kwargs)
                if resp.status_code >= 500:
                    raise SourceError(f"HTTP {resp.status_code} from {url}")
                return resp
            except (requests.RequestException, SourceError) as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = self.backoff ** attempt
                    log.warning("Attempt %s/%s failed for %s: %s; retrying in %.0fs", attempt, self.retries, url, exc, delay)
                    time.sleep(delay)
        raise SourceError(f"Giving up on {url} after {self.retries} attempts: {last_error}")

    def head_ok(self, url: str) -> bool:
        try:
            resp = self.session.head(url, timeout=min(self.timeout, 30), allow_redirects=True)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def download(self, url: str, dest: Path, *, method: str = "GET", data: Optional[dict] = None, headers: Optional[dict] = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        resp = self.request(method, url, stream=True, data=data, headers=headers)
        if resp.status_code != 200:
            raise SourceError(f"HTTP {resp.status_code} for {url}")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)
        return dest


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

def _is_valid_zip(path: Path, member_predicate: Callable[[str], bool]) -> bool:
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return any(member_predicate(n) for n in zf.namelist())
    except zipfile.BadZipFile:
        return False


def _extract_member(zip_path: Path, member_predicate: Callable[[str], bool], work_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if member_predicate(n)]
        if not members:
            raise DataValidationError(f"No expected CSV member in {zip_path.name}: {zf.namelist()}")
        member = members[0]
        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / Path(member).name
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        return target


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as fh:
        line = fh.readline()
    return [c.strip().strip('"') for c in line.rstrip("\r\n").split(",")]


def months_between(start: str, end: str) -> list[str]:
    n = month_diff(end, start)
    if n < 0:
        raise ValueError(f"start {start} is after end {end}")
    return [month_add(start, i) for i in range(n + 1)]


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


# --------------------------------------------------------------------------------------
# On-Time Performance adapter (operations)
# --------------------------------------------------------------------------------------

class BtsOnTimeAdapter:
    source = SOURCE_OTP

    def __init__(self, http: HttpClient, raw_dir: Path, keep_raw: bool = False):
        self.http = http
        self.raw_dir = raw_dir
        self.keep_raw = keep_raw

    @staticmethod
    def month_url(year_month: str) -> str:
        year, month = int(year_month[:4]), int(year_month[5:])
        return OTP_PREZIP_URL.format(year=year, month=month)

    def cache_path(self, year_month: str) -> Path:
        return self.raw_dir / f"otp_{year_month.replace('-', '_')}.zip"

    def probe_latest_month(self, not_after: Optional[str] = None, lookback: int = 18) -> Optional[str]:
        """Latest month whose zip exists, probing backwards from `not_after` (default: current month)."""
        ym = not_after or date_to_year_month(date.today())
        for _ in range(lookback):
            if self.http.head_ok(self.month_url(ym)):
                return ym
            ym = month_add(ym, -1)
        return None

    def fetch_month(self, year_month: str) -> Path:
        dest = self.cache_path(year_month)
        if _is_valid_zip(dest, lambda n: n.lower().endswith(".csv")):
            log.info("OTP %s: using cached %s", year_month, dest.name)
            return dest
        url = self.month_url(year_month)
        t0 = time.time()
        self.http.download(url, dest)
        if not _is_valid_zip(dest, lambda n: n.lower().endswith(".csv")):
            dest.unlink(missing_ok=True)
            raise SourceError(f"{url} did not return a zip with a CSV member")
        log.info("OTP %s: downloaded %.1f MB in %.0fs", year_month, dest.stat().st_size / 1e6, time.time() - t0)
        return dest

    def aggregate_month(self, zip_path: Path, year_month: str, work_dir: Optional[Path] = None) -> pd.DataFrame:
        """Aggregate one monthly file to airport_monthly_operations rows (validated)."""
        work = work_dir or Path(tempfile.mkdtemp(prefix="otp_"))
        csv_path = _extract_member(zip_path, lambda n: n.lower().endswith(".csv"), work)
        try:
            header = _csv_header(csv_path)
            missing = [c for c in OTP_REQUIRED if c not in header]
            if missing:
                raise DataValidationError(f"OTP {year_month}: missing columns {missing}")
            return aggregate_otp_csv(csv_path, year_month)
        finally:
            csv_path.unlink(missing_ok=True)
            if work_dir is None:
                shutil.rmtree(work, ignore_errors=True)


def aggregate_otp_csv(csv_path: Path, year_month: str, source: str = SOURCE_OTP) -> pd.DataFrame:
    """Pure aggregation + validation of an OTP-shaped CSV (used by tests with fixture slices)."""
    year, month = int(year_month[:4]), int(year_month[5:])
    header = _csv_header(csv_path)
    types = {k: v for k, v in OTP_TYPES.items() if k in header}
    types_sql = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    conn = duckdb.connect(":memory:")
    try:
        df = conn.execute(
            f"""
            SELECT
                Year AS year, Month AS month, Origin AS airport_code,
                COUNT(*) AS total_flights,
                SUM(CASE WHEN COALESCE(Cancelled, 0) >= 1 THEN 1 ELSE 0 END) AS cancelled_flights,
                SUM(CASE WHEN COALESCE(Cancelled, 0) < 1 AND COALESCE(DepDel15, 0) >= 1 THEN 1 ELSE 0 END) AS delayed_departures_15m,
                SUM(CASE WHEN COALESCE(Cancelled, 0) < 1 THEN COALESCE(DepDelayMinutes, 0) ELSE 0 END) AS dep_delay_minutes_sum,
                SUM(CASE WHEN COALESCE(Diverted, 0) >= 1 THEN 1 ELSE 0 END) AS diverted_flights
            FROM read_csv('{_sql_path(csv_path)}', header=true, auto_detect=true, types={{{types_sql}}}, null_padding=true, ignore_errors=false)
            GROUP BY 1, 2, 3
            """
        ).df()
    finally:
        conn.close()
    if df.empty:
        raise DataValidationError(f"OTP {year_month}: file produced zero rows")
    bad_period = df[(df["year"] != year) | (df["month"] != month)]
    if not bad_period.empty:
        raise DataValidationError(f"OTP {year_month}: {len(bad_period)} aggregated rows have a different Year/Month")
    df["airport_code"] = df["airport_code"].astype(str).str.strip().str.upper()
    bad_codes = df[~df["airport_code"].str.match(AIRPORT_CODE_RE.pattern)]
    if not bad_codes.empty:
        raise DataValidationError(f"OTP {year_month}: invalid airport codes {bad_codes['airport_code'].tolist()[:5]}")
    df["operated_flights"] = df["total_flights"] - df["cancelled_flights"]
    df["delay_rate"] = (df["delayed_departures_15m"] / df["operated_flights"].where(df["operated_flights"] > 0)).astype(float)
    df["cancellation_rate"] = (df["cancelled_flights"] / df["total_flights"]).astype(float)
    df["avg_departure_delay_minutes"] = (df["dep_delay_minutes_sum"] / df["operated_flights"].where(df["operated_flights"] > 0)).astype(float)
    for col in ("delay_rate", "cancellation_rate"):
        vals = df[col].dropna()
        if ((vals < 0) | (vals > 1)).any():
            raise DataValidationError(f"OTP {year_month}: {col} outside [0, 1]")
    df["year_month"] = month_start(year_month)
    df["source"] = source
    cols = [
        "year_month", "airport_code", "total_flights", "operated_flights", "delayed_departures_15m", "delay_rate",
        "cancelled_flights", "cancellation_rate", "diverted_flights", "dep_delay_minutes_sum", "avg_departure_delay_minutes", "source",
    ]
    out = df[cols].copy()
    for c in ("total_flights", "operated_flights", "delayed_departures_15m", "cancelled_flights", "diverted_flights"):
        out[c] = out[c].astype("int64")
    return out.sort_values("airport_code").reset_index(drop=True)


def validate_national_volume(ops_df: pd.DataFrame, year_month: str, bounds: Optional[tuple[int, int]] = None) -> None:
    total = int(ops_df["total_flights"].sum())
    lo, hi = bounds or NATIONAL_MONTHLY_FLIGHTS_RANGE
    if not lo <= total <= hi:
        raise DataValidationError(f"OTP {year_month}: national flight count {total:,} outside expected range {lo:,}-{hi:,}")


# --------------------------------------------------------------------------------------
# T-100 Segment adapter (traffic + routes)
# --------------------------------------------------------------------------------------

_HIDDEN_RE = re.compile(r'<input type="hidden" name="([^"]+)" id="[^"]+" value="([^"]*)"')


class BtsT100Adapter:
    source = SOURCE_T100

    def __init__(self, http: HttpClient, raw_dir: Path, keep_raw: bool = False, form_url: str = T100_FORM_URL):
        self.http = http
        self.raw_dir = raw_dir
        self.keep_raw = keep_raw
        self.form_url = form_url

    def cache_path(self, year: int) -> Path:
        return self.raw_dir / f"t100_segment_all_{year}.zip"

    @staticmethod
    def _member_ok(name: str) -> bool:
        return name.lower().endswith(".csv") and "documentation" not in name.lower()

    def build_form(self, page_html: str, year: int) -> dict[str, str]:
        hidden = dict(_HIDDEN_RE.findall(page_html))
        for key in ("__VIEWSTATE", "__EVENTVALIDATION"):
            if key not in hidden:
                raise SourceError(f"T-100 form page did not contain {key}; the TranStats page layout may have changed")
        checkbox_names = set(re.findall(r'<input[^>]*type="?checkbox"?[^>]*name="([^"]+)"', page_html, re.I))
        missing = [v for v in T100_VARIABLES if checkbox_names and v not in checkbox_names]
        if missing:
            raise SourceError(f"T-100 form is missing expected variables {missing}")
        form = dict(hidden)
        form.update({
            "__EVENTTARGET": "", "__EVENTARGUMENT": "", "__LASTFOCUS": "",
            "cboGeography": "All", "cboYear": str(year), "cboPeriod": "All",
            "chkDownloadZip": "on", "btnDownload": "Download",
        })
        for var in T100_VARIABLES:
            form[var] = "on"
        return form

    def fetch_year(self, year: int) -> Path:
        dest = self.cache_path(year)
        if _is_valid_zip(dest, self._member_ok):
            log.info("T-100 %s: using cached %s", year, dest.name)
            return dest
        t0 = time.time()
        page = self.http.request("GET", self.form_url)
        if page.status_code != 200:
            raise SourceError(f"T-100 form page returned HTTP {page.status_code}")
        form = self.build_form(page.text, year)
        self.http.download(
            self.form_url, dest, method="POST", data=form,
            headers={"Referer": self.form_url, "Content-Type": "application/x-www-form-urlencoded"},
        )
        if not _is_valid_zip(dest, self._member_ok):
            snippet = ""
            try:
                snippet = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", dest.read_text(errors="ignore")))[:300]
            except OSError:
                pass
            dest.unlink(missing_ok=True)
            raise SourceError(f"T-100 download for {year} did not return a zip with the data CSV. Response started with: {snippet!r}")
        log.info("T-100 %s: downloaded %.1f MB in %.0fs", year, dest.stat().st_size / 1e6, time.time() - t0)
        return dest

    def aggregate_year(self, zip_path: Path, year: int, work_dir: Optional[Path] = None) -> "T100YearResult":
        work = work_dir or Path(tempfile.mkdtemp(prefix="t100_"))
        csv_path = _extract_member(zip_path, self._member_ok, work)
        try:
            return aggregate_t100_csv(csv_path, year)
        finally:
            csv_path.unlink(missing_ok=True)
            if work_dir is None:
                shutil.rmtree(work, ignore_errors=True)


@dataclass
class T100YearResult:
    traffic: pd.DataFrame
    routes: pd.DataFrame
    airport_refs: pd.DataFrame      # airport_code, city, state (US origins seen)
    months: list[str]
    raw_rows: int


def aggregate_t100_csv(csv_path: Path, year: int, service_classes: Iterable[str] = TRAFFIC_SERVICE_CLASSES,
                       long_haul_miles: int = DEFAULT_LONG_HAUL_MILES, source: str = SOURCE_T100) -> T100YearResult:
    header = _csv_header(csv_path)
    missing = sorted(T100_REQUIRED - set(header))
    if missing:
        raise DataValidationError(f"T-100 {year}: missing columns {missing}")
    classes_sql = ", ".join(f"'{c}'" for c in service_classes)
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            f"""
            CREATE TEMP TABLE seg AS
            SELECT CAST(YEAR AS INTEGER) AS year, CAST(MONTH AS INTEGER) AS month,
                   upper(trim(ORIGIN)) AS origin, upper(trim(DEST)) AS dest,
                   ORIGIN_CITY_NAME AS origin_city, ORIGIN_STATE_ABR AS origin_state, ORIGIN_COUNTRY AS origin_country,
                   CAST(DEPARTURES_SCHEDULED AS DOUBLE) AS dep_sched, CAST(DEPARTURES_PERFORMED AS DOUBLE) AS dep_perf,
                   CAST(SEATS AS DOUBLE) AS seats, CAST(PASSENGERS AS DOUBLE) AS pax, CAST(DISTANCE AS DOUBLE) AS distance,
                   CLASS AS class
            FROM read_csv('{_sql_path(csv_path)}', header=true, auto_detect=true, null_padding=true,
                          types={{'YEAR': 'INTEGER', 'MONTH': 'INTEGER', 'DEPARTURES_SCHEDULED': 'DOUBLE', 'DEPARTURES_PERFORMED': 'DOUBLE',
                                 'SEATS': 'DOUBLE', 'PASSENGERS': 'DOUBLE', 'DISTANCE': 'DOUBLE', 'ORIGIN': 'VARCHAR', 'DEST': 'VARCHAR',
                                 'CLASS': 'VARCHAR', 'ORIGIN_COUNTRY': 'VARCHAR', 'DEST_COUNTRY': 'VARCHAR', 'ORIGIN_STATE_ABR': 'VARCHAR',
                                 'DEST_STATE_ABR': 'VARCHAR', 'ORIGIN_CITY_NAME': 'VARCHAR', 'DEST_CITY_NAME': 'VARCHAR', 'UNIQUE_CARRIER': 'VARCHAR'}})
            """
        )
        raw_rows = conn.execute("SELECT COUNT(*) FROM seg").fetchone()[0]
        if raw_rows == 0:
            raise DataValidationError(f"T-100 {year}: file produced zero rows")
        bad_year = conn.execute("SELECT COUNT(*) FROM seg WHERE year <> ?", [year]).fetchone()[0]
        if bad_year:
            raise DataValidationError(f"T-100 {year}: {bad_year} rows carry a different YEAR")
        traffic = conn.execute(
            f"""
            SELECT year, month, origin AS airport_code,
                   CAST(SUM(pax) AS BIGINT) AS passengers, CAST(SUM(seats) AS BIGINT) AS available_seats,
                   CAST(SUM(dep_perf) AS BIGINT) AS departures_performed, CAST(SUM(dep_sched) AS BIGINT) AS departures_scheduled
            FROM seg WHERE origin_country = 'US' AND class IN ({classes_sql})
            GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
            """
        ).df()
        routes = conn.execute(
            f"""
            SELECT year, month, origin AS origin_airport, dest AS destination_airport,
                   MAX(distance) AS distance_miles,
                   CAST(SUM(pax) AS BIGINT) AS passengers, CAST(SUM(seats) AS BIGINT) AS available_seats,
                   CAST(SUM(dep_perf) AS BIGINT) AS departures_performed
            FROM seg WHERE origin_country = 'US' AND class IN ({classes_sql})
            GROUP BY 1, 2, 3, 4 HAVING SUM(dep_perf) > 0 ORDER BY 1, 2, 3, 4
            """
        ).df()
        refs = conn.execute(
            """
            SELECT origin AS airport_code, arg_max(origin_city, pax_sum) AS city, arg_max(origin_state, pax_sum) AS state
            FROM (SELECT origin, origin_city, origin_state, SUM(pax) AS pax_sum FROM seg WHERE origin_country = 'US' GROUP BY 1, 2, 3)
            GROUP BY 1 ORDER BY 1
            """
        ).df()
    finally:
        conn.close()
    if traffic.empty:
        raise DataValidationError(f"T-100 {year}: no U.S.-origin rows for service classes {list(service_classes)}")
    traffic = traffic[traffic["airport_code"].str.match(AIRPORT_CODE_RE.pattern)].copy()
    routes = routes[routes["origin_airport"].str.match(AIRPORT_CODE_RE.pattern)].copy()
    traffic["year_month"] = [date(int(y), int(m), 1) for y, m in zip(traffic["year"], traffic["month"])]
    routes["year_month"] = [date(int(y), int(m), 1) for y, m in zip(routes["year"], routes["month"])]
    traffic["load_factor"] = (traffic["passengers"] / traffic["available_seats"].where(traffic["available_seats"] > 0)).astype(float)
    traffic["source"] = source
    routes["is_long_haul"] = routes["distance_miles"] >= long_haul_miles
    routes["source"] = source
    refs["city"] = refs["city"].fillna("").astype(str).str.split(",").str[0].str.strip()
    refs["state"] = refs["state"].fillna("").astype(str).str.strip().str.upper()
    months = sorted({date_to_year_month(d) for d in traffic["year_month"]})
    traffic_cols = ["year_month", "airport_code", "passengers", "available_seats", "departures_performed", "departures_scheduled", "load_factor", "source"]
    route_cols = ["year_month", "origin_airport", "destination_airport", "distance_miles", "passengers", "available_seats", "departures_performed", "is_long_haul", "source"]
    return T100YearResult(traffic=traffic[traffic_cols].reset_index(drop=True), routes=routes[route_cols].reset_index(drop=True),
                          airport_refs=refs, months=months, raw_rows=int(raw_rows))


# --------------------------------------------------------------------------------------
# OurAirports reference
# --------------------------------------------------------------------------------------

class OurAirportsAdapter:
    source = SOURCE_OURAIRPORTS

    def __init__(self, http: HttpClient, raw_dir: Path, url: str = OURAIRPORTS_URL):
        self.http = http
        self.raw_dir = raw_dir
        self.url = url

    def fetch(self) -> pd.DataFrame:
        dest = self.raw_dir / "ourairports_airports.csv"
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            self.http.download(self.url, dest)
        return load_ourairports_csv(dest)


def load_ourairports_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"ident", "type", "name", "iso_country", "iso_region", "municipality", "iata_code"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"OurAirports file missing columns {sorted(missing)}")
    us = df[(df["iso_country"] == "US") & (df["iata_code"].str.len() == 3)].copy()
    us["state"] = us["iso_region"].str.replace("US-", "", regex=False).str.upper()
    us["icao"] = us.get("icao_code", "").where(us.get("icao_code", "") != "", us["ident"]) if "icao_code" in us.columns else us["ident"]
    out = us.rename(columns={"iata_code": "airport_code", "name": "airport_name", "municipality": "city", "type": "airport_type",
                             "latitude_deg": "latitude", "longitude_deg": "longitude"})
    out["airport_code"] = out["airport_code"].str.upper()
    keep = ["airport_code", "airport_name", "city", "state", "airport_type", "icao", "latitude", "longitude"]
    for col in keep:
        if col not in out.columns:
            out[col] = ""
    out = out[keep]
    # prefer scheduled-service / larger airports when an IATA code is duplicated
    rank = {"large_airport": 0, "medium_airport": 1, "small_airport": 2}
    out["_rank"] = out["airport_type"].map(rank).fillna(3)
    out = out.sort_values(["airport_code", "_rank"]).drop_duplicates("airport_code").drop(columns="_rank")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Airport table assembly
# --------------------------------------------------------------------------------------

def assign_size_band(annual_passengers: float) -> str:
    if annual_passengers >= SIZE_BAND_THRESHOLDS["large"]:
        return "large"
    if annual_passengers >= SIZE_BAND_THRESHOLDS["medium"]:
        return "medium"
    if annual_passengers >= SIZE_BAND_THRESHOLDS["small"]:
        return "small"
    return "non_hub"


def build_airports_table(
    t100_refs: pd.DataFrame,
    otp_codes: Iterable[str],
    traffic: pd.DataFrame,
    reference: Optional[pd.DataFrame],
    end_month: Optional[str],
) -> pd.DataFrame:
    """Combine BTS-observed airports with the OurAirports reference into the `airports` table."""
    refs = t100_refs.copy() if t100_refs is not None else pd.DataFrame(columns=["airport_code", "city", "state"])
    refs = refs.set_index("airport_code") if not refs.empty else refs
    otp_set = {c for c in otp_codes if isinstance(c, str)}
    traffic_set = set(traffic["airport_code"]) if traffic is not None and not traffic.empty else set()
    codes = sorted(traffic_set | otp_set)

    latest_12m: dict[str, int] = {}
    if traffic is not None and not traffic.empty and end_month:
        start = month_start(month_add(end_month, -11))
        end = month_end(end_month)
        window = traffic[(traffic["year_month"] >= start) & (traffic["year_month"] <= end)]
        latest_12m = window.groupby("airport_code")["passengers"].sum().astype(int).to_dict()

    ref_idx = reference.set_index("airport_code") if reference is not None and not reference.empty else None
    rows = []
    for code in codes:
        city = state = ""
        if not refs.empty and code in refs.index:
            city = str(refs.loc[code, "city"] or "")
            state = str(refs.loc[code, "state"] or "")
        name = ""
        airport_type = ""
        icao = ""
        lat = lon = None
        if ref_idx is not None and code in ref_idx.index:
            r = ref_idx.loc[code]
            name = str(r["airport_name"] or "")
            airport_type = str(r["airport_type"] or "")
            icao = str(r["icao"] or "")
            city = city or str(r["city"] or "")
            state = state or str(r["state"] or "")
            try:
                lat = float(r["latitude"]) if r["latitude"] not in ("", None) else None
                lon = float(r["longitude"]) if r["longitude"] not in ("", None) else None
            except (TypeError, ValueError):
                lat = lon = None
        if not name:
            loc = ", ".join(p for p in (city, state) if p)
            name = f"{loc} ({code})" if loc else code
        pax12 = int(latest_12m.get(code, 0))
        rows.append({
            "airport_code": code,
            "airport_name": name,
            "city": city,
            "state": state,
            "country": "US",
            "iata_code": code,
            "icao_code": icao,
            "region": "new_england" if state in NEW_ENGLAND_STATES else None,
            "size_band": assign_size_band(pax12) if code in traffic_set else None,
            "aliases": json.dumps(generate_aliases(code, name, city, state)),
            "airport_type": airport_type or None,
            "has_traffic_data": code in traffic_set,
            "has_operations_data": code in otp_set,
            "latest_12m_passengers": pax12,
            "latitude": lat,
            "longitude": lon,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Refresh runner
# --------------------------------------------------------------------------------------

@dataclass
class RefreshSummary:
    scope: str
    start_month: str
    end_month_requested: Optional[str]
    otp_months_loaded: list[str] = field(default_factory=list)
    otp_months_failed: dict[str, str] = field(default_factory=dict)
    t100_years_loaded: list[int] = field(default_factory=list)
    t100_years_failed: dict[int, str] = field(default_factory=dict)
    traffic_end_month: Optional[str] = None
    ops_end_month: Optional[str] = None
    airports_rows: int = 0
    reference_status: str = "pending"
    db_path: Optional[str] = None
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.otp_months_loaded) and bool(self.t100_years_loaded) and not self.otp_months_failed and not self.t100_years_failed


class RefreshRunner:
    def __init__(
        self,
        db_path: Path,
        raw_dir: Path,
        *,
        http: Optional[HttpClient] = None,
        keep_raw: bool = False,
        workers: int = 4,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.db_path = Path(db_path)
        self.raw_dir = Path(raw_dir)
        self.http = http or HttpClient()
        self.keep_raw = keep_raw
        self.workers = max(1, workers)
        self.progress = progress or (lambda msg: log.info(msg))
        self.otp = BtsOnTimeAdapter(self.http, self.raw_dir, keep_raw)
        self.t100 = BtsT100Adapter(self.http, self.raw_dir, keep_raw)
        self.reference = OurAirportsAdapter(self.http, self.raw_dir)

    def run(self, scope: str, start_month: str, end_month: Optional[str] = None, fail_fast: bool = False) -> RefreshSummary:
        t0 = time.time()
        summary = RefreshSummary(scope=scope, start_month=start_month, end_month_requested=end_month)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        tmp_db = self.db_path.with_name(self.db_path.name + ".tmp")
        if tmp_db.exists():
            tmp_db.unlink()
        conn = dbmod.connect(tmp_db)
        dbmod.init_schema(conn)
        refreshed_at = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        try:
            resolved_end = end_month or self.otp.probe_latest_month()
            if resolved_end is None:
                raise SourceError("Could not determine the latest available On-Time Performance month (no PREZIP file found).")
            self.progress(f"Refresh scope={scope} months {start_month}..{resolved_end}")

            # ---- T-100 by year --------------------------------------------------------
            all_refs: list[pd.DataFrame] = []
            traffic_frames: list[pd.DataFrame] = []
            for year in range(int(start_month[:4]), int(resolved_end[:4]) + 1):
                try:
                    zip_path = self.t100.fetch_year(year)
                    result = self.t100.aggregate_year(zip_path, year)
                    in_range = result.traffic[(result.traffic["year_month"] >= month_start(start_month)) & (result.traffic["year_month"] <= month_end(resolved_end))]
                    routes_in = result.routes[(result.routes["year_month"] >= month_start(start_month)) & (result.routes["year_month"] <= month_end(resolved_end))]
                    dbmod.insert_dataframe(conn, "airport_monthly_traffic", in_range)
                    dbmod.insert_dataframe(conn, "route_monthly_metrics", routes_in)
                    traffic_frames.append(in_range)
                    all_refs.append(result.airport_refs)
                    summary.t100_years_loaded.append(year)
                    self.progress(f"T-100 {year}: {len(in_range):,} traffic rows, {len(routes_in):,} route rows, months {result.months[0]}..{result.months[-1]} ({result.raw_rows:,} raw rows)")
                    if not self.keep_raw:
                        zip_path.unlink(missing_ok=True)
                except (SourceError, DataValidationError, zipfile.BadZipFile) as exc:
                    summary.t100_years_failed[year] = str(exc)
                    log.error("T-100 %s failed: %s", year, exc)
                    if fail_fast:
                        raise
            traffic_all = pd.concat(traffic_frames, ignore_index=True) if traffic_frames else pd.DataFrame(columns=["year_month", "airport_code", "passengers"])
            if not traffic_all.empty:
                summary.traffic_end_month = date_to_year_month(max(traffic_all["year_month"]))
                t_start = date_to_year_month(min(traffic_all["year_month"]))
            else:
                t_start = None
            dbmod.record_refresh_metadata(
                conn, source=SOURCE_T100, data_scope=scope,
                coverage_start=month_start(t_start) if t_start else None,
                coverage_end=month_end(summary.traffic_end_month) if summary.traffic_end_month else None,
                record_count=int(len(traffic_all)), refreshed_at=refreshed_at,
                status="ok" if (summary.t100_years_loaded and not summary.t100_years_failed and not traffic_all.empty) else ("partial" if summary.t100_years_loaded else "failed"),
                notes=self._notes(summary.t100_years_failed, f"service classes {list(TRAFFIC_SERVICE_CLASSES)}; U.S. origins only"),
            )

            # ---- OTP by month (parallel download, sequential aggregate) ----------------
            months = months_between(start_month, resolved_end)
            ops_frames: list[pd.DataFrame] = []
            work_dir = self.raw_dir / "_work"
            with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self.otp.fetch_month, ym): ym for ym in months}
                pending = dict(futures)
                # process in month order as downloads complete
                for ym in months:
                    fut = next(f for f, m in pending.items() if m == ym)
                    try:
                        zip_path = fut.result()
                        t1 = time.time()
                        ops = self.otp.aggregate_month(zip_path, ym, work_dir=work_dir)
                        validate_national_volume(ops, ym)
                        dbmod.insert_dataframe(conn, "airport_monthly_operations", ops)
                        ops_frames.append(ops[["year_month", "airport_code", "total_flights"]])
                        summary.otp_months_loaded.append(ym)
                        done = len(summary.otp_months_loaded) + len(summary.otp_months_failed)
                        self.progress(f"OTP {ym}: {len(ops):,} airports, {int(ops['total_flights'].sum()):,} flights, aggregated in {time.time() - t1:.1f}s ({done}/{len(months)})")
                        if not self.keep_raw:
                            zip_path.unlink(missing_ok=True)
                    except (SourceError, DataValidationError, zipfile.BadZipFile) as exc:
                        summary.otp_months_failed[ym] = str(exc)
                        log.error("OTP %s failed: %s", ym, exc)
                        if fail_fast:
                            for f in pending:
                                f.cancel()
                            raise
            shutil.rmtree(work_dir, ignore_errors=True)
            ops_all = pd.concat(ops_frames, ignore_index=True) if ops_frames else pd.DataFrame(columns=["year_month", "airport_code"])
            if not ops_all.empty:
                summary.ops_end_month = date_to_year_month(max(ops_all["year_month"]))
                o_start = date_to_year_month(min(ops_all["year_month"]))
            else:
                o_start = None
            dbmod.record_refresh_metadata(
                conn, source=SOURCE_OTP, data_scope=scope,
                coverage_start=month_start(o_start) if o_start else None,
                coverage_end=month_end(summary.ops_end_month) if summary.ops_end_month else None,
                record_count=int(conn.execute("SELECT COUNT(*) FROM airport_monthly_operations").fetchone()[0]),
                refreshed_at=refreshed_at,
                status="ok" if summary.otp_months_loaded and not summary.otp_months_failed else ("partial" if summary.otp_months_loaded else "failed"),
                notes=self._notes(summary.otp_months_failed, "reporting carriers only; departures from U.S. airports"),
            )

            # ---- Airport reference ---------------------------------------------------
            reference_df: Optional[pd.DataFrame] = None
            try:
                reference_df = self.reference.fetch()
                summary.reference_status = "ok"
                dbmod.record_refresh_metadata(conn, source=SOURCE_OURAIRPORTS, data_scope=scope, coverage_start=None, coverage_end=None,
                                              record_count=int(len(reference_df)), status="ok", refreshed_at=refreshed_at, notes="U.S. airports with IATA codes")
            except (SourceError, DataValidationError, requests.RequestException) as exc:
                summary.reference_status = f"failed: {exc}"
                log.error("OurAirports reference failed: %s", exc)
                dbmod.record_refresh_metadata(conn, source=SOURCE_OURAIRPORTS, data_scope=scope, coverage_start=None, coverage_end=None,
                                              record_count=0, status="failed", refreshed_at=refreshed_at,
                                              notes=f"{exc}; airport names derived from BTS city/state fields")
            refs_df = pd.concat(all_refs, ignore_index=True).drop_duplicates("airport_code", keep="last") if all_refs else pd.DataFrame(columns=["airport_code", "city", "state"])
            end_for_size = summary.traffic_end_month
            airports_df = build_airports_table(refs_df, set(ops_all["airport_code"]) if not ops_all.empty else set(), traffic_all, reference_df, end_for_size)
            summary.airports_rows = dbmod.insert_dataframe(conn, "airports", airports_df)
            dbmod.record_refresh_metadata(conn, source=SOURCE_AIRPORTS_DERIVED, data_scope=scope,
                                          coverage_start=month_start(end_for_size and month_add(end_for_size, -11)) if end_for_size else None,
                                          coverage_end=month_end(end_for_size) if end_for_size else None,
                                          record_count=summary.airports_rows, status="ok", refreshed_at=refreshed_at,
                                          notes=f"size bands from passengers in the 12 months ending {end_for_size}")

            # ---- Default score snapshot (if the scoring layer is available) ------------
            try:
                from src.analytics import write_default_scores_snapshot  # noqa: WPS433 (lazy on purpose)
                n_scores = write_default_scores_snapshot(conn, scope)
                summary.notes.append(f"airport_scores snapshot rows: {n_scores}")
            except ImportError:
                summary.notes.append("airport_scores snapshot skipped (scoring layer not available); run scripts/refresh_data.py --scores-only later")
            except Exception as exc:  # pragma: no cover - defensive
                summary.notes.append(f"airport_scores snapshot failed: {exc}")
                log.exception("score snapshot failed")

            conn.close()
            conn = None
            if summary.otp_months_loaded or summary.t100_years_loaded:
                dbmod.atomic_replace(tmp_db, self.db_path)
                summary.db_path = str(self.db_path)
            else:
                summary.notes.append("No data loaded; existing database left untouched.")
        finally:
            if conn is not None:
                conn.close()
            if tmp_db.exists() and summary.db_path is None:
                tmp_db.unlink(missing_ok=True)
        summary.elapsed_seconds = round(time.time() - t0, 1)
        return summary

    @staticmethod
    def _notes(failures: dict, base: str) -> str:
        if not failures:
            return base
        items = "; ".join(f"{k}: {str(v)[:120]}" for k, v in list(failures.items())[:5])
        more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
        return f"{base}. FAILED: {items}{more}"
