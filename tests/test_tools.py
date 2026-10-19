"""Phase 4: typed tools on the fixture database. Every result must carry the required envelope."""
from __future__ import annotations

import json
from datetime import date

import pytest

from src import tools as T
from src.live_provider import LiveSnapshot, NullLiveProvider
from src.models import (
    AirportProfile, CapacityPressureResult, ComparisonResult, LiveContext, Methodology, RankingResult, ResolveResult, RouteMixResult,
    RunDefaults, ToolResult,
)

REQUIRED_KEYS = {"status", "data_scope", "data_period", "sources", "refresh_timestamp", "definitions", "metrics", "limitations", "confidence"}


@pytest.fixture(scope="module")
def ctx(fixture_conn, today):
    defaults = RunDefaults(data_scope="demo", period_end_month=None, long_haul_threshold_miles=2000, min_reported_flights=500, live_enabled=False)
    return T.build_tool_context(fixture_conn, defaults, live_provider=NullLiveProvider(), today=today)


def _check_envelope(result: ToolResult):
    env = result.envelope()
    assert set(env) == REQUIRED_KEYS
    assert result.status in ("ok", "insufficient_data", "ambiguous", "unavailable", "error")
    assert result.data_scope == "demo"
    assert result.confidence in ("high", "medium", "low")
    json.dumps(result.to_llm_dict(), default=str)       # serializable for the model
    if result.status == "ok" and not result.is_live and result.tool_name != "get_methodology":
        assert result.data_period is not None and result.sources and result.definitions


# --------------------------------------------------------------------------------------
# required assignment airports
# --------------------------------------------------------------------------------------

def test_rank_new_england(ctx):
    res = T.execute_tool("rank_airports", {"region": "new_england", "sort_by": "expansion_opportunity_score", "top_n": 10}, ctx)
    _check_envelope(res)
    assert res.status == "ok"
    payload: RankingResult = res.payload
    assert payload.peer_group_size == 6 and [e.airport.airport_code for e in payload.exclusions] == ["TNY"]
    assert {r.airport.state for r in payload.rows} <= {"CT", "ME", "MA", "NH", "RI", "VT"}
    assert payload.rows[0].rank == 1 and payload.rows[0].scores.expansion_opportunity_score is not None
    assert res.metrics["top_airport"] == payload.rows[0].airport.airport_code
    assert res.period.end_month == "2025-12" and res.data_period.start == date(2024, 1, 1)
    assert any("excluded" in l for l in res.limitations) and "new_england" in res.definitions
    assert "weights" in payload.normalization and payload.normalization["weights"]["demand_growth"] == 0.40


def test_rank_with_size_and_state_filters_and_conditions(ctx):
    res = T.execute_tool("rank_airports", {"states": ["TX"], "size_bands": ["medium"], "sort_by": "demand_growth"}, ctx)
    assert res.status == "ok" and {r.airport.airport_code for r in res.payload.rows} == {"AUS", "DAL", "HOU"}
    growths = [r.raw.demand_growth for r in res.payload.rows]
    assert growths == sorted(growths, reverse=True)
    assert [r.airport.airport_code for r in res.payload.rows][-1] == "AUS"       # missing fixture month -> negative growth
    assert res.payload.peer_group_size == 3 and res.confidence == "low"
    res2 = T.execute_tool("rank_airports", {"sort_by": "demand_growth", "conditions": [{"metric": "demand_growth", "op": ">", "value": 0},
                                                                                  {"metric": "cancellation_rate_change", "op": ">", "value": 0}]}, ctx)
    assert res2.status in ("ok", "insufficient_data")
    if res2.status == "ok":
        assert all(r.raw.demand_growth > 0 and r.raw.cancellation_rate_change > 0 for r in res2.payload.rows)
    empty = T.execute_tool("rank_airports", {"states": ["WY"]}, ctx)
    assert empty.status == "insufficient_data" and "No airports" in empty.message


def test_compare_lax_sna_rate_based_congestion(ctx):
    res = T.execute_tool("compare_airports", {"airports": ["LAX", "Santa Ana Airport, SNA"]}, ctx)
    _check_envelope(res)
    assert res.status == "ok"
    payload: ComparisonResult = res.payload
    assert [r["airport_code"] for r in payload.rows] == ["LAX", "SNA"]
    assert payload.most_flights == "LAX" and payload.higher_operational_pressure == "SNA"      # fewer flights, higher rates
    keys = {c.key for c in payload.columns}
    assert {"total_flights", "delayed_flights", "delay_rate", "avg_departure_delay_minutes", "cancellation_rate", "passengers", "capacity_utilization",
            "expansion_opportunity_score"} <= keys
    assert res.metrics["LAX_delay_rate_pct"] == 20.0 and res.metrics["SNA_delay_rate_pct"] == 25.0
    assert res.metrics["LAX_total_flights"] == 240_000 and res.metrics["SNA_total_flights"] == 48_000
    assert any("rate-based" in l for l in res.limitations)


def test_compare_three_airports_and_period(ctx):
    res = T.execute_tool("compare_airports", {"airports": ["Denver", "Phoenix", "Las Vegas"], "period": {"year": 2024}}, ctx)
    assert res.status == "ok" and [r["airport_code"] for r in res.payload.rows] == ["DEN", "PHX", "LAS"]
    assert res.period.calendar_year == 2024


def test_route_mix_anc(ctx):
    res = T.execute_tool("analyze_route_mix", {"airport": "Anchorage, ANC"}, ctx)
    _check_envelope(res)
    assert res.status == "ok"
    payload: RouteMixResult = res.payload
    assert (payload.long_haul_departures, payload.total_departures) == (1200, 4800)
    assert res.metrics["long_haul_share_pct"] == 25.0 and res.metrics["threshold_miles"] == 2000 and res.metrics["months_covered"] == 12
    assert res.confidence == "high" and "2,000" in res.definitions["long_haul_flight"]
    res2 = T.execute_tool("analyze_route_mix", {"airport": "ANC", "long_haul_threshold_miles": 1400}, ctx)
    assert res2.metrics["long_haul_share_pct"] == 60.0 and res2.payload.threshold_miles == 1400
    none = T.execute_tool("analyze_route_mix", {"airport": "SNA"}, ctx)
    assert none.status == "insufficient_data" and none.data_period is not None


def test_capacity_pressure_sfo(ctx):
    res = T.execute_tool("analyze_capacity_pressure", {"airport": "San Francisco Airport"}, ctx)
    _check_envelope(res)
    assert res.status == "ok"
    payload: CapacityPressureResult = res.payload
    assert payload.airport.airport_code == "SFO" and 0 <= payload.proxy_score <= 100
    assert [c.name for c in payload.components] == ["capacity_utilization", "demand_growth", "operational_pressure"]
    assert sum(c.weight for c in payload.components) == pytest.approx(1.0)
    assert any("unmet demand" in l for l in res.limitations)
    assert res.metrics["capacity_pressure_proxy"] == pytest.approx(payload.proxy_score, abs=0.1)
    assert payload.peer_group.size_bands == ["large"]                # 12 large airports pass coverage -> same-band peers
    assert payload.peer_group_size >= 5


def test_capacity_pressure_insufficient(ctx):
    res = T.execute_tool("analyze_capacity_pressure", {"airport": "TNY"}, ctx)
    assert res.status == "insufficient_data" and res.payload is None
    assert "capacity_pressure_proxy" not in res.metrics
    assert any("unmet demand" in l for l in res.limitations)


def test_profile_generic_airport_and_change_question(ctx):
    res = T.execute_tool("get_airport_profile", {"airport": "Austin", "period": None}, ctx)
    _check_envelope(res)
    assert res.status == "ok"
    payload: AirportProfile = res.payload
    assert payload.airport.airport_code == "AUS" and len(payload.series) >= 23
    assert any(c.metric == "passengers" for c in payload.changes)
    assert res.confidence == "medium"                              # 11 of 12 traffic months in the latest window
    bos = T.execute_tool("get_airport_profile", {"airport": "Boston Logan", "period": {"year": 2024}}, ctx)
    assert bos.status == "ok" and bos.period.calendar_year == 2024
    assert bos.metrics["demand_growth_pct"] == 5.0 and bos.metrics["delay_rate_pct"] == 20.0 and bos.metrics["delay_rate_prior_pct"] == 18.8
    changes = {c.metric: c for c in bos.payload.changes}
    assert changes["cancellation_rate"].direction == "down" and changes["delay_rate"].direction == "up"


# --------------------------------------------------------------------------------------
# failure and integrity behaviour
# --------------------------------------------------------------------------------------

def test_ambiguous_and_unknown_airports(ctx):
    amb = T.execute_tool("compare_airports", {"airports": ["Washington", "BOS"]}, ctx)
    _check_envelope(amb)
    assert amb.status == "ambiguous" and amb.metrics == {} and isinstance(amb.payload, ResolveResult)
    assert {c.airport_code for c in amb.payload.candidates} == {"DCA", "IAD"}
    unknown = T.execute_tool("get_airport_profile", {"airport": "Atlantis Spaceport"}, ctx)
    assert unknown.status in ("error", "ambiguous") and unknown.payload is not None and unknown.metrics == {}
    resolved = T.execute_tool("resolve_airport", {"name": "Los Angeles Airport"}, ctx)
    assert resolved.status == "ok" and resolved.metrics["airport_code"] == "LAX"
    washington = T.execute_tool("resolve_airport", {"name": "Washington"}, ctx)
    assert washington.status == "ambiguous"


def test_period_before_coverage_and_clamping(ctx):
    early = T.execute_tool("compare_airports", {"airports": ["LAX", "SNA"], "period": {"year": 2023}}, ctx)
    assert early.status == "insufficient_data" and "coverage starts" in early.message
    late = T.execute_tool("compare_airports", {"airports": ["LAX", "SNA"], "period": {"year": 2026}}, ctx)
    assert late.status == "ok" and late.period.clamped and any("available through" in l for l in late.limitations)


def test_invalid_inputs_never_raise(ctx):
    bad = T.execute_tool("rank_airports", {"top_n": 0}, ctx)
    assert bad.status == "error" and "top_n" in bad.message
    one = T.execute_tool("compare_airports", {"airports": ["LAX"]}, ctx)
    assert one.status == "error"
    unknown_tool = T.execute_tool("run_sql", {"sql": "DROP TABLE airports"}, ctx)
    assert unknown_tool.status == "error" and "allow-list" in unknown_tool.message
    bad_region = T.execute_tool("rank_airports", {"region": "midwest"}, ctx)
    assert bad_region.status == "error" and "region" in bad_region.message
    extra = T.execute_tool("analyze_route_mix", {"airport": "ANC", "unexpected": 1}, ctx)
    assert extra.status == "error"


def test_live_tool_unavailable_without_key_and_with_toggle(ctx, fixture_conn, today):
    res = T.execute_tool("get_live_operational_context", {"airport": "LAX"}, ctx)
    _check_envelope(res)
    assert res.status == "unavailable" and res.is_live and "disabled" in res.message
    enabled = RunDefaults(data_scope="demo", live_enabled=True)
    ctx2 = T.build_tool_context(fixture_conn, enabled, live_provider=NullLiveProvider(), today=today)
    res2 = T.execute_tool("get_live_operational_context", {"airport": "LAX"}, ctx2)
    assert res2.status == "unavailable" and "AVIATIONSTACK_API_KEY" in res2.message
    assert res2.data_period is None and any("never modifies historical" in l for l in res2.limitations)

    class FakeProvider:
        name = "fake"

        def is_configured(self):
            return True

        def get_departure_snapshot(self, code):
            return LiveSnapshot(status="ok", provider="fake", context=LiveContext(airport_code=code, provider="fake", retrieved_at="2026-10-19T12:00:00+00:00",
                                                                                 flights_sampled=40, scheduled=20, active=15, cancelled=2, delayed_15m=9,
                                                                                 avg_departure_delay_minutes=12.5, status_counts={"scheduled": 20, "active": 15, "cancelled": 2, "landed": 3},
                                                                                 window_description="first 40 departures"))

    ctx3 = T.build_tool_context(fixture_conn, enabled, live_provider=FakeProvider(), today=today)
    res3 = T.execute_tool("get_live_operational_context", {"airport": "LAX"}, ctx3)
    assert res3.status == "ok" and res3.metrics["delayed_15m"] == 9 and res3.refresh_timestamp == "2026-10-19T12:00:00+00:00" and res3.confidence == "high"


def test_methodology_tool(ctx):
    res = T.execute_tool("get_methodology", {"topic": "expansion_opportunity_score"}, ctx)
    _check_envelope(res)
    assert res.status == "ok" and isinstance(res.payload, Methodology)
    assert res.payload.weights["expansion_opportunity_score"] == {"demand_growth": 0.40, "capacity_utilization": 0.35, "operational_pressure": 0.25}
    assert set(res.definitions) >= {"delayed_departure", "long_haul_flight", "new_england", "airport_congestion", "capacity_pressure", "airport_size"}


# --------------------------------------------------------------------------------------
# schema export
# --------------------------------------------------------------------------------------

def test_anthropic_tool_schemas_are_strict_and_complete():
    defs = {d["name"]: d for d in T.anthropic_tool_definitions(live_enabled=True)}
    assert set(defs) == {"resolve_airport", "get_airport_profile", "rank_airports", "compare_airports", "analyze_route_mix",
                         "analyze_capacity_pressure", "get_live_operational_context", "get_methodology"}
    assert "get_live_operational_context" not in {d["name"] for d in T.anthropic_tool_definitions(live_enabled=False)}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            assert "$ref" not in node and "title" not in node and "default" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for d in defs.values():
        assert d["strict"] is True and d["description"]
        walk(d["input_schema"])
    rank = defs["rank_airports"]["input_schema"]["properties"]
    assert rank["sort_by"]["enum"][0] == "expansion_opportunity_score"
    assert rank["top_n"]["minimum"] == 1 and rank["top_n"]["maximum"] == 50
    assert "null" in rank["states"]["type"] or "anyOf" in rank["states"]
    period = rank["period"]
    assert "anyOf" in period and any(b.get("type") == "null" for b in period["anyOf"])
    compare = defs["compare_airports"]["input_schema"]["properties"]["airports"]
    assert compare["minItems"] == 2 and compare["maxItems"] == 6


def test_gemini_tool_schemas_have_no_additional_properties():
    defs = {d["name"]: d for d in T.gemini_tool_definitions(live_enabled=False)}
    assert "get_live_operational_context" not in defs

    def walk(node):
        if isinstance(node, dict):
            assert "additionalProperties" not in node and "$ref" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for d in defs.values():
        walk(d["parameters"])
    rank = defs["rank_airports"]["parameters"]
    assert rank["properties"]["period"].get("nullable") is True
    assert "sort_by" not in rank.get("required", [])           # optional in Gemini variant
    assert rank["properties"]["period"]["type"] == "object"
