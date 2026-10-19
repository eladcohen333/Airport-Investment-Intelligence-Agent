"""Phase 5: deterministic router, follow-up context, grounding guard and the backend-neutral agent loop (fake backend)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src import agent as ag
from src import demo_router as dr
from src.live_provider import LiveSnapshot, NullLiveProvider
from src.llm_backends import BackendError, BackendResponse, ToolCallRequest
from src.models import LiveContext, RunDefaults, SessionContext
from src.tools import build_tool_context


@pytest.fixture(scope="module")
def defaults():
    return RunDefaults(data_scope="demo", long_haul_threshold_miles=2000, min_reported_flights=500, live_enabled=False)


@pytest.fixture(scope="module")
def tool_ctx(fixture_conn, defaults, today):
    return build_tool_context(fixture_conn, defaults, live_provider=NullLiveProvider(), today=today)


@pytest.fixture()
def demo(today):
    return dr.DemoAgent(today=today)


def route(question, tool_ctx, defaults, ctx=None):
    parsed = dr.parse_question(question, tool_ctx)
    return parsed, dr.DemoAgent().route(parsed, ctx or SessionContext(), defaults)


# --------------------------------------------------------------------------------------
# routing of the required and "also support" questions
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,tool,checks",
    [
        ("Which airports in New England are strong candidates for terminal expansion?", "rank_airports", {"region": "new_england", "sort_by": "expansion_opportunity_score"}),
        ("Compare LAX and Santa Ana Airport, SNA, congestion levels.", "compare_airports", {"airports": ["LAX", "SNA"]}),
        ("What percentage of departures from Anchorage, ANC, are long-haul?", "analyze_route_mix", {"airport": "ANC"}),
        ("What is the capacity-pressure signal at SFO and why?", "analyze_capacity_pressure", {"airport": "SFO"}),
        ("Which medium-sized airports in Texas have the strongest demand growth?", "rank_airports", {"states": ["TX"], "size_bands": ["medium"], "sort_by": "demand_growth"}),
        ("Compare Denver, Phoenix, and Las Vegas.", "compare_airports", {"airports": ["DEN", "PHX", "LAS"]}),
        ("What changed at Boston Logan in 2024?", "get_airport_profile", {"airport": "BOS", "period": {"year": 2024}}),
        ("What is the current operational status at LAX?", "get_live_operational_context", {"airport": "LAX"}),
        ("Which airports have growing passengers and worsening cancellations?", "rank_airports", {"sort_by": "demand_growth"}),
        ("How is the expansion opportunity score calculated?", "get_methodology", {"topic": "expansion_opportunity_score"}),
        ("What share of ANC departures are long-haul at 1,500 miles?", "analyze_route_mix", {"airport": "ANC", "long_haul_threshold_miles": 1500}),
    ],
)
def test_router_maps_questions_to_tools(question, tool, checks, tool_ctx, defaults):
    parsed, routed = route(question, tool_ctx, defaults)
    assert not isinstance(routed, dr.Clarification), (parsed.intent, routed)
    assert routed[0].tool == tool
    for key, value in checks.items():
        assert routed[0].input.get(key) == value, (key, routed[0].input)
    if "worsening cancellations" in question:
        conds = routed[0].input["conditions"]
        assert {"metric": "demand_growth", "op": ">", "value": 0} in conds
        assert {"metric": "cancellation_rate_change", "op": ">", "value": 0} in conds


def test_demo_runs_required_questions_end_to_end(demo, tool_ctx, defaults):
    for q in dr.SUGGESTED_QUESTIONS:
        turn = demo.run(q, SessionContext(), defaults, tool_ctx)
        assert turn.mode == "demo" and turn.answer_kind == "answer", (q, turn.answer_markdown)
        assert turn.tool_calls and turn.tool_calls[0].result.status == "ok"
        assert turn.answer_markdown.startswith("**Conclusion:**") and turn.conclusion
    live = demo.run("What is the current operational status at LAX?", SessionContext(), defaults, tool_ctx)
    assert live.tool_calls[0].result.status == "unavailable" and live.live_calls == [0]
    assert "unaffected" in live.answer_markdown


def test_follow_up_reuses_airports_and_changes_period(demo, tool_ctx, defaults):
    ctx = SessionContext()
    first = demo.run("Compare LAX and SNA congestion.", ctx, defaults, tool_ctx)
    assert first.tool_calls[0].name == "compare_airports" and first.tool_calls[0].input["airports"] == ["LAX", "SNA"]
    ctx = ag.update_context(ctx, "Compare LAX and SNA congestion.", first)
    assert ctx.comparison_set == ["LAX", "SNA"] and ctx.last_tool == "compare_airports"

    second = demo.run("What about 2024?", ctx, defaults, tool_ctx)
    call = second.tool_calls[0]
    assert call.name == "compare_airports"
    assert call.input["airports"] == ["LAX", "SNA"] and call.input["period"] == {"year": 2024}
    assert call.result.status == "ok" and call.result.period.calendar_year == 2024
    ctx = ag.update_context(ctx, "What about 2024?", second)
    assert ctx.period.end_month == "2024-12" and ctx.comparison_set == ["LAX", "SNA"]
    assert ctx.turn_count == 2 and len(ctx.recent_messages) == 4


def test_follow_up_edge_cases(demo, tool_ctx, defaults):
    empty = demo.run("What about 2024?", SessionContext(), defaults, tool_ctx)
    assert empty.answer_kind == "clarification" and "Which airport" in empty.conclusion

    ctx = ag.update_context(SessionContext(), "Compare LAX and SNA congestion.", demo.run("Compare LAX and SNA congestion.", SessionContext(), defaults, tool_ctx))
    one_more = demo.run("What about SFO?", ctx, defaults, tool_ctx)
    assert one_more.answer_kind == "clarification" and "SFO" in one_more.conclusion

    rank_turn = demo.run("Which airports in New England are strong candidates for terminal expansion?", SessionContext(), defaults, tool_ctx)
    ctx2 = ag.update_context(SessionContext(), "rank", rank_turn)
    assert ctx2.region == "new_england" and ctx2.selected_metric == "expansion_opportunity_score"
    boston = demo.run("What about Boston?", ctx2, defaults, tool_ctx)
    assert boston.tool_calls[0].name == "get_airport_profile" and boston.tool_calls[0].input["airport"] == "BOS"

    vague = demo.run("What about it?", ctx2, defaults, tool_ctx)
    assert vague.answer_kind == "clarification"


def test_ambiguous_unknown_and_unsupported(demo, tool_ctx, defaults):
    amb = demo.run("How congested is Washington?", SessionContext(), defaults, tool_ctx)
    assert amb.answer_kind == "clarification" and "DCA" in amb.conclusion and "IAD" in amb.conclusion
    unknown = demo.run("Tell me about Atlantis Spaceport", SessionContext(), defaults, tool_ctx)
    assert unknown.answer_kind == "clarification" and "could not find" in unknown.conclusion.lower() or "no airport" in unknown.conclusion.lower()
    unsupported = demo.run("Write me a poem about runways", SessionContext(), defaults, tool_ctx)
    assert unsupported.answer_kind == "unsupported" and "suggested questions" in unsupported.conclusion
    non_adjacent = demo.run("Compare LAX and SNA 2024 vs 2022", SessionContext(), defaults, tool_ctx)
    assert non_adjacent.answer_kind == "clarification" and "adjacent" in non_adjacent.conclusion


def test_combined_historical_and_live_sections(fixture_conn, today):
    class FakeProvider:
        name = "fake"

        def is_configured(self):
            return True

        def get_departure_snapshot(self, code):
            return LiveSnapshot(status="ok", provider="fake", context=LiveContext(airport_code=code, provider="fake", retrieved_at="2026-10-19T12:00:00+00:00",
                                                                                 flights_sampled=30, scheduled=10, active=15, cancelled=1, delayed_15m=6,
                                                                                 avg_departure_delay_minutes=9.0, status_counts={}, window_description="sample"))

    live_defaults = RunDefaults(data_scope="demo", live_enabled=True)
    ctx = build_tool_context(fixture_conn, live_defaults, live_provider=FakeProvider(), today=today)
    turn = dr.DemoAgent(today=today).run("What changed at LAX and what is its current status?", SessionContext(), live_defaults, ctx)
    assert [c.name for c in turn.tool_calls] == ["get_airport_profile", "get_live_operational_context"]
    assert turn.historical_calls == [0] and turn.live_calls == [1]
    assert f"### {dr.HISTORICAL_HEADING}" in turn.answer_markdown and f"### {dr.LIVE_HEADING}" in turn.answer_markdown
    assert turn.answer_markdown.index(dr.HISTORICAL_HEADING) < turn.answer_markdown.index(dr.LIVE_HEADING)


# --------------------------------------------------------------------------------------
# grounding guard
# --------------------------------------------------------------------------------------

def test_extract_numbers_and_grounding(demo, tool_ctx, defaults):
    assert ag.extract_numbers("In 2024 the delay rate was 23.4% across 240,000 flights (top 3).") == [23.4, 240000.0]
    assert ag.extract_numbers("LAX vs SNA") == []
    turn = demo.run("Compare LAX and SNA congestion.", SessionContext(), defaults, tool_ctx)
    results = [c.result for c in turn.tool_calls]
    assert ag.grounding_warnings("LAX delay rate 20% vs SNA 25%; LAX had 240,000 flights.", results) == []
    assert ag.grounding_warnings("LAX delay rate was 0.2 of flights", results) == []
    warnings = ag.grounding_warnings("SNA cancellation rate is 7.77%.", results)
    assert warnings and "7.77" in warnings[0]
    # rounding to significant figures is fine, a misquoted score is not
    assert ag.grounding_warnings("LAX handled about 240 thousand flights and 31.8 million passengers.", results) == []
    assert ag.grounding_warnings("LAX handled about 30 million passengers.", results)     # 31.8M -> 30M is too loose
    assert ag.grounding_warnings("LAX delay rate was 19.4%.", results)          # tool says 20.0


def test_grounding_rejects_misquoted_score(demo, tool_ctx, defaults):
    turn = demo.run("What is the capacity-pressure signal at SFO and why?", SessionContext(), defaults, tool_ctx)
    results = [c.result for c in turn.tool_calls]
    proxy = results[0].metrics["capacity_pressure_proxy"]
    assert ag.grounding_warnings(f"SFO's proxy is {proxy}.", results) == []
    assert ag.grounding_warnings(f"SFO's proxy is {round(proxy - 0.6, 1)}.", results)


# --------------------------------------------------------------------------------------
# agent loop with a scripted fake backend
# --------------------------------------------------------------------------------------

@dataclass
class FakeBackend:
    scripts: list
    name: str = "fake"
    model: str = "fake-model"
    calls: list = field(default_factory=list)

    def complete(self, system, history, tools):
        self.calls.append({"system": system, "history": [dict(h) for h in history], "tools": tools})
        item = self.scripts.pop(0) if self.scripts else BackendResponse(text="**Conclusion:** done.")
        if isinstance(item, Exception):
            raise item
        return item


def _tool_use(name, inp, call_id="toolu_1", text=""):
    return BackendResponse(text=text, tool_calls=[ToolCallRequest(id=call_id, name=name, input=inp)], stop_reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5})


def test_agent_executes_tool_and_grounds_answer(tool_ctx, defaults):
    backend = FakeBackend([
        _tool_use("analyze_route_mix", {"airport": "ANC", "period": None, "long_haul_threshold_miles": None}),
        BackendResponse(text="**Conclusion:** 25.0% of ANC departures (1,200 of 4,800) were long-haul.\n\n- threshold 2,000 miles", stop_reason="end_turn", usage={"input_tokens": 20, "output_tokens": 30}),
    ])
    agent = ag.AnalyticsAgent(backend, mode="anthropic")
    turn = agent.run("What percentage of departures from ANC are long-haul?", SessionContext(), defaults, tool_ctx)
    assert turn.answer_kind == "answer" and turn.mode == "anthropic"
    assert turn.conclusion.startswith("25.0% of ANC departures")
    assert [c.name for c in turn.tool_calls] == ["analyze_route_mix"] and turn.tool_calls[0].result.status == "ok"
    assert turn.grounding_warnings == []
    assert turn.usage == {"input_tokens": 30, "output_tokens": 35}
    second_call = backend.calls[1]["history"]
    assert second_call[-2]["role"] == "assistant" and second_call[-2]["tool_calls"][0].id == "toolu_1"
    assert second_call[-1]["role"] == "tool_results" and second_call[-1]["results"][0]["id"] == "toolu_1"
    assert "Session context" in backend.calls[0]["system"].lower() or "SESSION CONTEXT" in backend.calls[0]["system"]
    assert any(t["name"] == "rank_airports" for t in backend.calls[0]["tools"])


def test_agent_handles_parallel_calls_and_tool_errors(tool_ctx, defaults):
    resp = BackendResponse(text="", stop_reason="tool_use", tool_calls=[
        ToolCallRequest(id="a", name="analyze_route_mix", input={"airport": "ANC"}),
        ToolCallRequest(id="b", name="run_sql", input={"sql": "select 1"}),
    ])
    backend = FakeBackend([resp, BackendResponse(text="**Conclusion:** ANC long-haul share is 25%.")])
    turn = ag.AnalyticsAgent(backend, mode="gemini").run("ANC long haul", SessionContext(), defaults, tool_ctx)
    results = backend.calls[1]["history"][-1]["results"]
    assert [r["id"] for r in results] == ["a", "b"] and results[1]["is_error"] is True
    assert turn.tool_calls[1].result.status == "error" and "allow-list" in turn.tool_calls[1].result.message
    assert turn.answer_kind == "answer"


def test_no_tool_result_means_no_numeric_answer(tool_ctx, defaults):
    backend = FakeBackend([BackendResponse(text="LAX delay rate is 23.4% this year.")])
    turn = ag.AnalyticsAgent(backend, mode="anthropic").run("How delayed is LAX?", SessionContext(), defaults, tool_ctx)
    assert turn.answer_kind == "clarification" and turn.tool_calls == []
    assert "23.4" not in turn.answer_markdown and ag.extract_numbers(turn.answer_markdown) == []
    asking = FakeBackend([BackendResponse(text="Did you mean Washington Dulles (IAD) or Reagan National (DCA)?")])
    turn2 = ag.AnalyticsAgent(asking, mode="anthropic").run("Washington delays", SessionContext(), defaults, tool_ctx)
    assert turn2.answer_kind == "clarification" and "IAD" in turn2.answer_markdown


def test_refusal_and_backend_errors(tool_ctx, defaults):
    refusing = FakeBackend([BackendResponse(text="", stop_reason="refusal")])
    turn = ag.AnalyticsAgent(refusing, mode="anthropic").run("x", SessionContext(), defaults, tool_ctx)
    assert turn.answer_kind == "error" and "declined" in turn.conclusion

    failing = FakeBackend([BackendError("Anthropic rate limit reached; try again shortly.", retryable=True, kind="rate_limit")])
    turn = ag.AnalyticsAgent(failing, mode="anthropic").run("What percentage of departures from ANC are long-haul?", SessionContext(), defaults, tool_ctx)
    # model unavailable before any tool ran -> deterministic router answers, with a visible note
    assert turn.answer_kind == "answer" and turn.mode == "demo" and turn.tool_calls[0].name == "analyze_route_mix"
    assert any("deterministic router" in w and "rate limit" in w for w in turn.grounding_warnings)
    assert "Traceback" not in turn.answer_markdown

    after_tools = FakeBackend([_tool_use("analyze_route_mix", {"airport": "ANC"}), BackendError("network down", retryable=True, kind="network")])
    turn = ag.AnalyticsAgent(after_tools, mode="anthropic").run("ANC long haul", SessionContext(), defaults, tool_ctx)
    assert turn.answer_kind == "answer" and turn.tool_calls and "25.0%" in turn.answer_markdown
    assert any("deterministic summary" in w for w in turn.grounding_warnings)

    crashing = FakeBackend([RuntimeError("boom")])
    turn = ag.AnalyticsAgent(crashing, mode="anthropic").run("x", SessionContext(), defaults, tool_ctx)
    assert turn.answer_kind == "error" and "RuntimeError" in turn.conclusion and "boom" not in turn.answer_markdown


def test_iteration_cap_forces_final_answer(tool_ctx, defaults):
    scripts = [_tool_use("analyze_route_mix", {"airport": "ANC"}, call_id=f"c{i}") for i in range(ag.MAX_ITERATIONS)]
    scripts.append(BackendResponse(text="**Conclusion:** ANC long-haul share is 25.0%."))
    backend = FakeBackend(scripts)
    turn = ag.AnalyticsAgent(backend, mode="anthropic").run("ANC", SessionContext(), defaults, tool_ctx)
    assert len(backend.calls) == ag.MAX_ITERATIONS + 1
    assert backend.calls[-1]["tools"] == [] and backend.calls[-1]["history"][-1]["text"] == ag.ANSWER_NOW
    assert turn.answer_kind == "answer" and len(turn.tool_calls) == ag.MAX_ITERATIONS


def test_update_context_from_llm_turn(tool_ctx, defaults):
    backend = FakeBackend([_tool_use("rank_airports", {"region": "new_england", "sort_by": "demand_growth"}), BackendResponse(text="**Conclusion:** BDL leads.")])
    turn = ag.AnalyticsAgent(backend, mode="anthropic").run("New England growth", SessionContext(), defaults, tool_ctx)
    ctx = ag.update_context(SessionContext(), "New England growth", turn)
    assert ctx.region == "new_england" and ctx.selected_metric == "demand_growth" and ctx.last_tool == "rank_airports"
    assert ctx.last_result_summary["status"] == "ok" and ctx.period.end_month == "2025-12"
    for _ in range(8):
        ctx = ag.update_context(ctx, "again", turn)
    assert len(ctx.recent_messages) == 10


def test_system_prompt_contains_rules_and_context(tool_ctx, defaults):
    ctx = SessionContext(airport_codes=["LAX", "SNA"], comparison_set=["LAX", "SNA"])
    prompt = ag.build_system_prompt(defaults, ctx, tool_ctx.coverage)
    for needle in ("never calculate investment ROI", "insufficient_data", "**Conclusion:**", dr.HISTORICAL_HEADING, dr.LIVE_HEADING, "unmet demand", '"comparison_set": ["LAX", "SNA"]', "Delayed Departure"):
        assert needle in prompt
