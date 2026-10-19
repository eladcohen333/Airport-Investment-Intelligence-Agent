"""Typed contracts shared by analytics, tools, agent and UI.

Everything the LLM sees or the UI renders is one of these Pydantic models.
Month arithmetic helpers live here too because periods are part of the contracts.
"""
from __future__ import annotations

import calendar
import json
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------

Status = Literal["ok", "insufficient_data", "ambiguous", "unavailable", "error"]
DataScope = Literal["demo", "nationwide"]
Confidence = Literal["high", "medium", "low"]
SizeBand = Literal["large", "medium", "small", "non_hub"]
SIZE_BANDS: tuple[str, ...] = ("large", "medium", "small", "non_hub")

RankMetric = Literal[
    "expansion_opportunity_score",
    "demand_growth",
    "capacity_utilization",
    "operational_pressure",
    "delay_rate",
    "cancellation_rate",
]
ScreenMetric = Literal[
    "demand_growth",
    "capacity_utilization",
    "delay_rate",
    "cancellation_rate",
    "operational_pressure",
    "delay_rate_change",
    "cancellation_rate_change",
]
CompareMetric = Literal[
    "total_flights",
    "delayed_flights",
    "delay_rate",
    "avg_departure_delay_minutes",
    "cancellation_rate",
    "passengers",
    "capacity_utilization",
    "scores",
]
MethodologyTopic = Literal[
    "overview",
    "expansion_opportunity_score",
    "capacity_pressure_proxy",
    "demand_growth",
    "capacity_utilization",
    "operational_pressure",
    "long_haul",
    "size_band",
    "peer_group_and_normalization",
    "coverage_and_confidence",
    "data_sources",
]

NEW_ENGLAND_STATES: tuple[str, ...] = ("CT", "ME", "MA", "NH", "RI", "VT")
REGIONS: dict[str, tuple[str, ...]] = {"new_england": NEW_ENGLAND_STATES}


# --------------------------------------------------------------------------------------
# Month helpers ("YYYY-MM" strings are the canonical month representation)
# --------------------------------------------------------------------------------------

def parse_year_month(value: str) -> tuple[int, int]:
    text = str(value).strip()
    if len(text) != 7 or text[4] != "-":
        raise ValueError(f"Expected YYYY-MM, got {value!r}")
    year, month = int(text[:4]), int(text[5:])
    if not 1 <= month <= 12:
        raise ValueError(f"Month out of range in {value!r}")
    return year, month


def month_add(year_month: str, months: int) -> str:
    year, month = parse_year_month(year_month)
    index = year * 12 + (month - 1) + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def month_diff(later: str, earlier: str) -> int:
    ly, lm = parse_year_month(later)
    ey, em = parse_year_month(earlier)
    return (ly * 12 + lm) - (ey * 12 + em)


def month_start(year_month: str) -> date:
    year, month = parse_year_month(year_month)
    return date(year, month, 1)


def month_end(year_month: str) -> date:
    year, month = parse_year_month(year_month)
    return date(year, month, calendar.monthrange(year, month)[1])


def date_to_year_month(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


# --------------------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------------------

class DataPeriod(BaseModel):
    start: date
    end: date


class PeriodRequest(BaseModel):
    """What callers (LLM / router) may ask for. Both None means 'use the sidebar default'."""

    model_config = ConfigDict(extra="forbid")

    year: Optional[int] = Field(default=None, ge=1990, le=2100)
    end_month: Optional[str] = None

    @field_validator("end_month")
    @classmethod
    def _check_month(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        parse_year_month(v)
        return v

    def is_empty(self) -> bool:
        return self.year is None and self.end_month is None


class AnalysisPeriod(BaseModel):
    """Two adjacent rolling windows: latest = [end-11, end], prior = [end-23, end-12]."""

    model_config = ConfigDict(frozen=True)

    end_month: str
    window_months: int = 12
    requested: Optional[str] = None
    clamped: bool = False

    @field_validator("end_month")
    @classmethod
    def _check_end(cls, v: str) -> str:
        parse_year_month(v)
        return v

    @property
    def latest_start_month(self) -> str:
        return month_add(self.end_month, -(self.window_months - 1))

    @property
    def prior_end_month(self) -> str:
        return month_add(self.end_month, -self.window_months)

    @property
    def prior_start_month(self) -> str:
        return month_add(self.end_month, -(2 * self.window_months - 1))

    @property
    def latest_start(self) -> date:
        return month_start(self.latest_start_month)

    @property
    def latest_end(self) -> date:
        return month_end(self.end_month)

    @property
    def prior_start(self) -> date:
        return month_start(self.prior_start_month)

    @property
    def prior_end(self) -> date:
        return month_end(self.prior_end_month)

    @property
    def calendar_year(self) -> Optional[int]:
        year, month = parse_year_month(self.end_month)
        return year if (month == 12 and self.window_months == 12) else None

    @property
    def label(self) -> str:
        cy = self.calendar_year
        if cy is not None:
            return f"calendar {cy} vs calendar {cy - 1}"
        return (
            f"{self.window_months} months ending {self.end_month} "
            f"({self.latest_start_month} to {self.end_month}) vs prior {self.window_months} months "
            f"({self.prior_start_month} to {self.prior_end_month})"
        )

    @property
    def short_label(self) -> str:
        cy = self.calendar_year
        if cy is not None:
            return f"{cy} vs {cy - 1}"
        return f"{self.window_months}m to {self.end_month}"

    def data_period(self) -> DataPeriod:
        return DataPeriod(start=self.prior_start, end=self.latest_end)


# --------------------------------------------------------------------------------------
# Airports and resolution
# --------------------------------------------------------------------------------------

class AirportRef(BaseModel):
    airport_code: str
    airport_name: str
    city: str = ""
    state: str = ""
    size_band: Optional[str] = None
    has_traffic_data: bool = False
    has_operations_data: bool = False
    latest_12m_passengers: int = 0

    @property
    def display(self) -> str:
        loc = ", ".join(p for p in (self.city, self.state) if p)
        return f"{self.airport_name} ({self.airport_code})" + (f", {loc}" if loc else "")


class ResolveResult(BaseModel):
    status: Literal["resolved", "ambiguous", "unknown"]
    query: str
    airport: Optional[AirportRef] = None
    candidates: list[AirportRef] = Field(default_factory=list)
    match_type: Optional[Literal["code", "alias", "city", "city_dominant", "fuzzy"]] = None
    score: Optional[float] = None
    note: Optional[str] = None


# --------------------------------------------------------------------------------------
# Analytics payload building blocks
# --------------------------------------------------------------------------------------

class RawKPIs(BaseModel):
    airport_code: str
    pax_latest: int = 0
    pax_prior: int = 0
    seats_latest: int = 0
    seats_prior: int = 0
    departures_latest: int = 0
    departures_prior: int = 0
    flights_latest: int = 0
    flights_prior: int = 0
    operated_latest: int = 0
    operated_prior: int = 0
    delayed_latest: int = 0
    delayed_prior: int = 0
    cancelled_latest: int = 0
    cancelled_prior: int = 0
    delay_minutes_latest: float = 0.0
    delay_minutes_prior: float = 0.0
    traffic_months_latest: int = 0
    traffic_months_prior: int = 0
    ops_months_latest: int = 0
    ops_months_prior: int = 0
    demand_growth: Optional[float] = None
    capacity_utilization: Optional[float] = None
    capacity_utilization_prior: Optional[float] = None
    delay_rate: Optional[float] = None
    delay_rate_prior: Optional[float] = None
    cancellation_rate: Optional[float] = None
    cancellation_rate_prior: Optional[float] = None
    avg_departure_delay_minutes: Optional[float] = None
    avg_departure_delay_minutes_prior: Optional[float] = None
    operational_pressure: Optional[float] = None
    operational_pressure_prior: Optional[float] = None
    delay_rate_change: Optional[float] = None
    cancellation_rate_change: Optional[float] = None
    quality_flags: list[str] = Field(default_factory=list)


class CoverageCheck(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    detail: str = ""
    min_reported_flights: int = 500


class ComponentScores(BaseModel):
    demand_growth_score: Optional[float] = None
    capacity_utilization_score: Optional[float] = None
    operational_pressure_score: Optional[float] = None
    expansion_opportunity_score: Optional[float] = None


class PeerGroupSpec(BaseModel):
    data_scope: str
    states: Optional[list[str]] = None
    region: Optional[str] = None
    size_bands: Optional[list[str]] = None
    min_reported_flights: int = 500

    def describe(self) -> str:
        parts: list[str] = []
        if self.region == "new_england":
            parts.append("New England (" + ", ".join(NEW_ENGLAND_STATES) + ")")
        elif self.states:
            parts.append("states " + ", ".join(self.states))
        else:
            parts.append("all U.S. airports in scope")
        if self.size_bands:
            parts.append("size band " + "/".join(self.size_bands))
        else:
            parts.append("all size bands")
        parts.append(f"{self.data_scope} scope")
        return ", ".join(parts)


class Exclusion(BaseModel):
    airport: AirportRef
    reasons: list[str]
    detail: str
    raw: Optional[RawKPIs] = None


class RankedAirport(BaseModel):
    rank: int
    airport: AirportRef
    raw: RawKPIs
    scores: ComponentScores
    contributions: dict[str, float] = Field(default_factory=dict)
    confidence: Confidence = "medium"


class RankingResult(BaseModel):
    peer_group: PeerGroupSpec
    peer_group_size: int
    sort_by: str
    conditions: list["Condition"] = Field(default_factory=list)
    rows: list[RankedAirport]
    total_ranked: int
    exclusions: list[Exclusion] = Field(default_factory=list)
    normalization: dict[str, Any] = Field(default_factory=dict)


class CompareColumn(BaseModel):
    key: str
    label: str
    kind: Literal["absolute", "rate", "score"]
    unit: str = ""


class ComparisonResult(BaseModel):
    columns: list[CompareColumn]
    rows: list[dict[str, Any]]
    scores_peer_group: Optional[PeerGroupSpec] = None
    higher_operational_pressure: Optional[str] = None
    most_flights: Optional[str] = None
    note: str = ""
    coverage: dict[str, CoverageCheck] = Field(default_factory=dict)


class RouteRow(BaseModel):
    destination_airport: str
    distance_miles: float
    departures: int
    passengers: int


class RouteMixResult(BaseModel):
    airport: AirportRef
    long_haul_departures: int
    total_departures: int
    long_haul_share: Optional[float]
    threshold_miles: int
    months_covered: int
    top_long_haul_routes: list[RouteRow] = Field(default_factory=list)
    source: str = ""


class CPPComponent(BaseModel):
    name: str
    label: str
    raw_value: Optional[float]
    percentile: float
    weight: float
    contribution: float


class CapacityPressureResult(BaseModel):
    airport: AirportRef
    proxy_score: float
    components: list[CPPComponent]
    peer_group: PeerGroupSpec
    peer_group_size: int
    raw: RawKPIs
    coverage: CoverageCheck
    expansion: Optional[ComponentScores] = None


class MetricChange(BaseModel):
    metric: str
    label: str
    prior: Optional[float]
    latest: Optional[float]
    abs_delta: Optional[float]
    pct_change: Optional[float]
    direction: Literal["up", "down", "flat", "n/a"]
    unit: str = ""


class MonthlyPoint(BaseModel):
    year_month: str
    passengers: Optional[int] = None
    available_seats: Optional[int] = None
    load_factor: Optional[float] = None
    total_flights: Optional[int] = None
    delay_rate: Optional[float] = None
    cancellation_rate: Optional[float] = None


class AirportProfile(BaseModel):
    airport: AirportRef
    changes: list[MetricChange]
    series: list[MonthlyPoint]
    raw: RawKPIs
    coverage: CoverageCheck
    long_haul_threshold_miles: int
    long_haul_share_latest: Optional[float] = None


class LiveContext(BaseModel):
    airport_code: str
    provider: str
    retrieved_at: str
    flights_sampled: int
    scheduled: int = 0
    active: int = 0
    landed: int = 0
    cancelled: int = 0
    delayed_15m: int = 0
    avg_departure_delay_minutes: Optional[float] = None
    status_counts: dict[str, int] = Field(default_factory=dict)
    rate_limited: bool = False
    window_description: str = ""
    note: str = ""


class Methodology(BaseModel):
    topic: str
    title: str
    formulas: dict[str, str] = Field(default_factory=dict)
    weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Tool inputs
# --------------------------------------------------------------------------------------

class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: ScreenMetric
    op: Literal[">", ">=", "<", "<="]
    value: float


class ResolveAirportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)


class AirportProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airport: str = Field(min_length=1)
    period: Optional[PeriodRequest] = None


class RankAirportsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: Optional[str] = None
    states: Optional[list[str]] = None
    size_bands: Optional[list[SizeBand]] = None
    sort_by: RankMetric = "expansion_opportunity_score"
    conditions: Optional[list[Condition]] = None
    top_n: int = Field(default=10, ge=1, le=50)
    ascending: bool = False
    period: Optional[PeriodRequest] = None

    @field_validator("region")
    @classmethod
    def _check_region(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        key = v.strip().lower().replace(" ", "_").replace("-", "_")
        if key in ("", "none", "null", "all"):
            return None
        if key not in REGIONS:
            raise ValueError(f"Unknown region {v!r}; supported: {', '.join(REGIONS)}")
        return key

    @field_validator("states")
    @classmethod
    def _check_states(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if not v:
            return None
        cleaned = []
        for s in v:
            code = str(s).strip().upper()
            if len(code) != 2 or not code.isalpha():
                raise ValueError(f"State codes must be two letters, got {s!r}")
            cleaned.append(code)
        return cleaned


class CompareAirportsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airports: list[str] = Field(min_length=2, max_length=6)
    metrics: Optional[list[CompareMetric]] = None
    period: Optional[PeriodRequest] = None


class RouteMixInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airport: str = Field(min_length=1)
    period: Optional[PeriodRequest] = None
    long_haul_threshold_miles: Optional[int] = Field(default=None, ge=100, le=10000)


class CapacityPressureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airport: str = Field(min_length=1)
    period: Optional[PeriodRequest] = None
    same_size_band_peers: bool = True


class LiveContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    airport: str = Field(min_length=1)


class MethodologyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: MethodologyTopic = "overview"


# --------------------------------------------------------------------------------------
# Tool result envelope
# --------------------------------------------------------------------------------------

class ToolResult(BaseModel):
    status: Status
    tool_name: str
    data_scope: str
    data_period: Optional[DataPeriod] = None
    sources: list[str] = Field(default_factory=list)
    refresh_timestamp: Optional[str] = None
    definitions: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    message: Optional[str] = None
    period: Optional[AnalysisPeriod] = None
    payload: Any = None
    is_live: bool = False

    @model_validator(mode="after")
    def _non_ok_has_message(self) -> "ToolResult":
        if self.status != "ok" and not self.message:
            self.message = f"Tool {self.tool_name} returned status {self.status}."
        return self

    def envelope(self) -> dict[str, Any]:
        """The brief's required keys only."""
        return {
            "status": self.status,
            "data_scope": self.data_scope,
            "data_period": (
                {"start": self.data_period.start.isoformat(), "end": self.data_period.end.isoformat()}
                if self.data_period
                else None
            ),
            "sources": list(self.sources),
            "refresh_timestamp": self.refresh_timestamp,
            "definitions": dict(self.definitions),
            "metrics": dict(self.metrics),
            "limitations": list(self.limitations),
            "confidence": self.confidence,
        }

    def to_llm_dict(self, max_rows: int = 10, include_series: bool = False) -> dict[str, Any]:
        """Compact representation for the model: envelope + trimmed payload."""
        out = self.envelope()
        out["tool_name"] = self.tool_name
        out["message"] = self.message
        out["period_label"] = self.period.label if self.period else None
        payload = self.payload
        if payload is None:
            return out
        data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        if isinstance(data, dict):
            data = _compact_payload(data, max_rows=max_rows, include_series=include_series)
        out["payload"] = data
        return out

    def to_llm_json(self, max_rows: int = 10, include_series: bool = False) -> str:
        return json.dumps(self.to_llm_dict(max_rows=max_rows, include_series=include_series), default=str)


def _compact_payload(data: dict[str, Any], *, max_rows: int, include_series: bool) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in data.items():
        if key == "series" and not include_series:
            compact["series_points"] = len(value) if isinstance(value, list) else 0
            continue
        if key == "rows" and isinstance(value, list):
            compact[key] = [_compact_row(v) for v in value[:max_rows]]
            if len(value) > max_rows:
                compact["rows_truncated"] = len(value) - max_rows
            continue
        if key == "exclusions" and isinstance(value, list):
            compact[key] = [
                {"airport_code": (v.get("airport") or {}).get("airport_code"), "reasons": v.get("reasons"), "detail": v.get("detail")}
                for v in value[:max_rows]
            ]
            if len(value) > max_rows:
                compact["exclusions_truncated"] = len(value) - max_rows
            continue
        compact[key] = value
    return compact


def _compact_row(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    slim = dict(row)
    raw = slim.pop("raw", None)
    if isinstance(raw, dict):
        keep = (
            "pax_latest", "pax_prior", "seats_latest", "flights_latest", "delayed_latest", "cancelled_latest",
            "demand_growth", "capacity_utilization", "delay_rate", "cancellation_rate",
            "avg_departure_delay_minutes", "operational_pressure", "delay_rate_change", "cancellation_rate_change",
        )
        slim["raw"] = {k: raw.get(k) for k in keep}
    airport = slim.get("airport")
    if isinstance(airport, dict):
        slim["airport"] = {k: airport.get(k) for k in ("airport_code", "airport_name", "state", "size_band")}
    return slim


# --------------------------------------------------------------------------------------
# Agent-level contracts
# --------------------------------------------------------------------------------------

class RunDefaults(BaseModel):
    """Sidebar state handed to the agent and tools for every turn."""

    data_scope: str = "demo"
    period_end_month: Optional[str] = None
    states: Optional[list[str]] = None
    region: Optional[str] = None
    size_bands: Optional[list[str]] = None
    long_haul_threshold_miles: int = 2000
    min_reported_flights: int = 500
    live_enabled: bool = False


class SessionContext(BaseModel):
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    airport_codes: list[str] = Field(default_factory=list)
    comparison_set: list[str] = Field(default_factory=list)
    region: Optional[str] = None
    states: Optional[list[str]] = None
    selected_metric: Optional[str] = None
    period: Optional[AnalysisPeriod] = None
    size_bands: Optional[list[str]] = None
    long_haul_threshold_miles: Optional[int] = None
    last_tool: Optional[str] = None
    last_tool_input: Optional[dict[str, Any]] = None
    last_result_summary: dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0

    def compact(self) -> dict[str, Any]:
        return {
            "airport_codes": self.airport_codes,
            "comparison_set": self.comparison_set,
            "region": self.region,
            "states": self.states,
            "selected_metric": self.selected_metric,
            "period_end_month": self.period.end_month if self.period else None,
            "period_label": self.period.label if self.period else None,
            "size_bands": self.size_bands,
            "long_haul_threshold_miles": self.long_haul_threshold_miles,
            "last_tool": self.last_tool,
            "last_tool_input": self.last_tool_input,
            "last_result_summary": self.last_result_summary,
        }


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]
    result: ToolResult
    duration_ms: int = 0

    @property
    def is_error(self) -> bool:
        return self.result.status == "error"


class AgentTurn(BaseModel):
    mode: Literal["anthropic", "gemini", "demo"]
    answer_kind: Literal["answer", "clarification", "unsupported", "error"]
    conclusion: str
    answer_markdown: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    historical_calls: list[int] = Field(default_factory=list)
    live_calls: list[int] = Field(default_factory=list)
    grounding_warnings: list[str] = Field(default_factory=list)
    usage: Optional[dict[str, Any]] = None
    model: Optional[str] = None


RankingResult.model_rebuild()
