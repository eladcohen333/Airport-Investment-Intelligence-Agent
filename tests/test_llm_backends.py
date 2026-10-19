"""Phase 5: provider adapters render history and parse tool calls correctly (fake clients, no network)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import llm_backends as lb
from src.config import Settings
from src.tools import anthropic_tool_definitions, gemini_tool_definitions

HISTORY = [
    {"role": "user", "text": "Compare LAX and SNA"},
    {"role": "assistant", "text": "Let me check.", "tool_calls": [lb.ToolCallRequest(id="t1", name="compare_airports", input={"airports": ["LAX", "SNA"]})]},
    {"role": "tool_results", "results": [{"id": "t1", "name": "compare_airports", "content": '{"status": "ok", "metrics": {"x": 1}}', "is_error": False}]},
    {"role": "user", "text": "What about 2024?"},
]


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------

class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _anthropic_response(blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=11, output_tokens=7))


def test_anthropic_render_messages_shape():
    backend = lb.AnthropicBackend(api_key="k", model="claude-opus-5-5", client=SimpleNamespace(messages=_FakeMessages(None)))
    msgs = backend.render_messages(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "user"]
    assert msgs[1]["content"][0] == {"type": "text", "text": "Let me check."}
    assert msgs[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "compare_airports", "input": {"airports": ["LAX", "SNA"]}}
    assert msgs[2]["content"][0]["type"] == "tool_result" and msgs[2]["content"][0]["tool_use_id"] == "t1" and msgs[2]["content"][0]["is_error"] is False
    raw_turn = {"role": "assistant", "text": "", "tool_calls": [], "raw": [{"type": "text", "text": "raw"}], "raw_backend": "anthropic"}
    assert backend.render_messages([raw_turn])[0]["content"] == [{"type": "text", "text": "raw"}]
    foreign = {**raw_turn, "raw_backend": "gemini", "text": "fallback"}
    assert backend.render_messages([foreign])[0]["content"] == [{"type": "text", "text": "fallback"}]


def test_anthropic_complete_passes_tools_and_parses_tool_use():
    blocks = [SimpleNamespace(type="text", text="Checking."),
              SimpleNamespace(type="tool_use", id="toolu_9", name="rank_airports", input={"region": "new_england"})]
    fake = _FakeMessages(_anthropic_response(blocks, "tool_use"))
    backend = lb.AnthropicBackend(api_key="k", model="claude-opus-5-5", client=SimpleNamespace(messages=fake), effort="medium")
    tools = anthropic_tool_definitions(False)
    resp = backend.complete("SYSTEM", HISTORY[:1], tools)
    assert fake.kwargs["model"] == "claude-opus-5-5" and fake.kwargs["system"] == "SYSTEM" and fake.kwargs["tools"] is tools
    assert fake.kwargs["output_config"] == {"effort": "medium"} and "tool_choice" not in fake.kwargs
    assert resp.stop_reason == "tool_use" and resp.text == "Checking."
    assert resp.tool_calls[0].id == "toolu_9" and resp.tool_calls[0].input == {"region": "new_england"}
    assert resp.usage == {"input_tokens": 11, "output_tokens": 7} and resp.raw_backend == "anthropic"
    refusal = backend.parse_response(_anthropic_response([], "refusal"))
    assert refusal.stop_reason == "refusal" and refusal.tool_calls == []
    stringy = backend.parse_response(_anthropic_response([SimpleNamespace(type="tool_use", id="x", name="get_methodology", input='{"topic": "overview"}')], "tool_use"))
    assert stringy.tool_calls[0].input == {"topic": "overview"}


# --------------------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------------------

class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_gemini_render_contents_and_declarations():
    from google.genai import types

    backend = lb.GeminiBackend(api_key="k", model="gemini-3.8-flash", client=SimpleNamespace(models=_FakeModels(None)))
    contents = backend.render_contents(HISTORY)
    assert [c.role for c in contents] == ["user", "model", "user", "user"]      # function responses use the user role
    assert contents[1].parts[0].text == "Let me check." and contents[1].parts[1].function_call.name == "compare_airports"
    fr = contents[2].parts[0].function_response
    assert fr.name == "compare_airports" and fr.response["status"] == "ok"
    decls = backend._declarations(gemini_tool_definitions(False))
    assert isinstance(decls[0], types.FunctionDeclaration) and decls[0].name == "resolve_airport"
    schema = decls[0].parameters_json_schema if hasattr(decls[0], "parameters_json_schema") else decls[0].parameters
    assert schema is not None


def test_gemini_complete_parses_function_calls():
    from google.genai import types

    part_call = SimpleNamespace(function_call=SimpleNamespace(name="analyze_route_mix", args={"airport": "ANC"}, id=None), text=None, thought=False)
    part_text = SimpleNamespace(function_call=None, text="Working on it.", thought=False)
    candidate = SimpleNamespace(content=SimpleNamespace(role="model", parts=[part_text, part_call]), finish_reason="STOP")
    response = SimpleNamespace(candidates=[candidate], usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=3))
    fake = _FakeModels(response)
    backend = lb.GeminiBackend(api_key="k", model="gemini-3.8-flash", client=SimpleNamespace(models=fake))
    resp = backend.complete("SYS", HISTORY[:1], gemini_tool_definitions(False))
    assert fake.kwargs["model"] == "gemini-3.8-flash"
    config = fake.kwargs["config"]
    assert isinstance(config, types.GenerateContentConfig) and config.system_instruction == "SYS"
    assert config.automatic_function_calling.disable is True and config.tools
    assert resp.stop_reason == "tool_use" and resp.text == "Working on it."
    assert resp.tool_calls[0].name == "analyze_route_mix" and resp.tool_calls[0].input == {"airport": "ANC"} and resp.tool_calls[0].id == "gemini_call_1"
    assert resp.usage == {"input_tokens": 5, "output_tokens": 3} and resp.raw_backend == "gemini"
    blocked = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]), finish_reason="SAFETY")], usage_metadata=None)
    assert backend.parse_response(blocked).stop_reason == "refusal"


def test_gemini_switches_to_fallback_model_on_overload(monkeypatch):
    monkeypatch.setattr(lb.time, "sleep", lambda s: None)

    class Overloaded(Exception):
        pass

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.8-flash":
                raise Overloaded("503 UNAVAILABLE. This model is currently experiencing high demand.")
            part = SimpleNamespace(function_call=None, text="**Conclusion:** fine.", thought=False)
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(role="model", parts=[part]), finish_reason="STOP")], usage_metadata=None)

    models = Models()
    backend = lb.GeminiBackend(api_key="k", model="gemini-3.8-flash", client=SimpleNamespace(models=models),
                               fallback_models=("gemini-3.6-flash", "gemini-3.5-flash-lite"), max_server_error_retries=1)
    resp = backend.complete("s", HISTORY[:1], [])
    assert resp.text.startswith("**Conclusion:**")
    assert models.calls == ["gemini-3.8-flash", "gemini-3.6-flash"]      # overload -> immediate switch, no retry on the overloaded model
    assert backend.model == "gemini-3.6-flash" and backend.primary_model == "gemini-3.8-flash"
    backend.complete("s", HISTORY[:1], [])
    assert models.calls[-1] == "gemini-3.6-flash"                                              # stays on the working model


def test_gemini_errors_map_to_backend_error():
    class Boom(Exception):
        pass

    class RaisingModels:
        def generate_content(self, **kwargs):
            raise Boom("429 RESOURCE_EXHAUSTED: quota")

    backend = lb.GeminiBackend(api_key="k", model="gemini-3.8-flash", client=SimpleNamespace(models=RaisingModels()))
    with pytest.raises(lb.BackendError) as exc:
        backend.complete("s", HISTORY[:1], [])
    assert exc.value.kind == "rate_limit" and exc.value.retryable


# --------------------------------------------------------------------------------------
# provider selection
# --------------------------------------------------------------------------------------

def _settings(**over):
    base = dict(project_root=".", llm_provider="auto", anthropic_api_key=None, anthropic_model="claude-opus-5-5", gemini_api_key=None,
                gemini_model="gemini-3.8-flash", gemini_fallback_models=("gemini-3.6-flash",), aviationstack_api_key=None, aviationstack_base_url="https://api.aviationstack.com/v1",
                duckdb_path="data/x.duckdb", long_haul_threshold_miles=2000, min_reported_flights=500, demo_scope_start="2023-01",
                nationwide_scope_start="2019-01", log_level="INFO")
    base.update(over)
    from pathlib import Path
    base["project_root"] = Path(base["project_root"])
    base["duckdb_path"] = Path(base["duckdb_path"])
    return Settings(**base)


def test_build_backend_selection():
    backend, provider = lb.build_backend(_settings())
    assert backend is None and provider == "demo"
    backend, provider = lb.build_backend(_settings(gemini_api_key="g"))
    assert provider == "gemini" and backend.model == "gemini-3.8-flash"
    backend, provider = lb.build_backend(_settings(anthropic_api_key="a", gemini_api_key="g"))
    assert provider == "anthropic" and backend.model == "claude-opus-5-5"
    backend, provider = lb.build_backend(_settings(anthropic_api_key="a", gemini_api_key="g", llm_provider="gemini"))
    assert provider == "gemini"
    backend, provider = lb.build_backend(_settings(anthropic_api_key="a", llm_provider="demo"))
    assert provider == "demo"
