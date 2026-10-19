"""Pluggable LLM backends behind one small interface.

The agent loop (src/agent.py) is provider-neutral. Each backend renders the neutral history into its own wire
format and parses tool calls back out. Anthropic is the default runtime; Gemini is the alternative when only a
Gemini key is available. Neither backend ever computes a metric.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from src.config import Settings, get_logger, resolve_provider

log = get_logger("llm")


@dataclass
class ToolCallRequest:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class BackendResponse:
    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"          # end_turn | tool_use | refusal | max_tokens | other
    usage: Optional[dict[str, Any]] = None
    raw: Any = None                        # provider-native assistant content, echoed back on the next call
    raw_backend: Optional[str] = None


class BackendError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, kind: str = "error"):
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind


class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> BackendResponse: ...


# --------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------

class AnthropicBackend:
    name = "anthropic"

    def __init__(self, api_key: Optional[str], model: str, client: Any = None, max_tokens: int = 4000, effort: Optional[str] = "medium"):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        if client is None:
            import anthropic  # local import keeps the SDK optional at import time

            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.client = client

    # ---- rendering --------------------------------------------------------------------
    def render_messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for turn in history:
            role = turn["role"]
            if role == "user":
                messages.append({"role": "user", "content": turn["text"]})
            elif role == "assistant":
                if turn.get("raw") is not None and turn.get("raw_backend") == self.name:
                    messages.append({"role": "assistant", "content": turn["raw"]})
                    continue
                blocks: list[dict[str, Any]] = []
                if turn.get("text"):
                    blocks.append({"type": "text", "text": turn["text"]})
                for tc in turn.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
                messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
            elif role == "tool_results":
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"], "is_error": bool(r.get("is_error"))}
                    for r in turn["results"]
                ]})
        return messages

    # ---- call -------------------------------------------------------------------------
    def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> BackendResponse:
        import anthropic

        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": self.render_messages(history)}
        if tools:
            kwargs["tools"] = tools
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            resp = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise BackendError("The Anthropic API key was rejected.", kind="auth") from exc
        except anthropic.RateLimitError as exc:
            raise BackendError("Anthropic rate limit reached; try again shortly.", retryable=True, kind="rate_limit") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"Anthropic API error (HTTP {exc.status_code}).", retryable=exc.status_code >= 500, kind="api") from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError("Could not reach the Anthropic API.", retryable=True, kind="network") from exc
        return self.parse_response(resp)

    def parse_response(self, resp: Any) -> BackendResponse:
        text_parts: list[str] = []
        calls: list[ToolCallRequest] = []
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                raw_input = block.input
                if isinstance(raw_input, str):
                    try:
                        raw_input = json.loads(raw_input)
                    except ValueError:
                        raw_input = {}
                calls.append(ToolCallRequest(id=block.id, name=block.name, input=dict(raw_input or {})))
        stop = getattr(resp, "stop_reason", None) or ("tool_use" if calls else "end_turn")
        if stop not in ("end_turn", "tool_use", "refusal", "max_tokens"):
            stop = "other"
        usage = None
        u = getattr(resp, "usage", None)
        if u is not None:
            usage = {"input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None)}
        return BackendResponse(text="\n".join(t for t in text_parts if t).strip(), tool_calls=calls, stop_reason=stop, usage=usage,
                               raw=getattr(resp, "content", None), raw_backend=self.name)


# --------------------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------------------

class GeminiBackend:
    name = "gemini"

    def __init__(self, api_key: Optional[str], model: str, client: Any = None, temperature: float = 0.0, max_output_tokens: int = 4000):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self.client = client

    @staticmethod
    def _types():
        from google.genai import types
        return types

    def render_contents(self, history: list[dict[str, Any]]) -> list[Any]:
        types = self._types()
        contents: list[Any] = []
        for turn in history:
            role = turn["role"]
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["text"])]))
            elif role == "assistant":
                if turn.get("raw") is not None and turn.get("raw_backend") == self.name:
                    contents.append(turn["raw"])
                    continue
                parts = []
                if turn.get("text"):
                    parts.append(types.Part.from_text(text=turn["text"]))
                for tc in turn.get("tool_calls", []):
                    parts.append(types.Part.from_function_call(name=tc.name, args=tc.input))
                contents.append(types.Content(role="model", parts=parts or [types.Part.from_text(text="(no content)")]))
            elif role == "tool_results":
                parts = []
                for r in turn["results"]:
                    try:
                        payload = json.loads(r["content"]) if isinstance(r["content"], str) else r["content"]
                    except ValueError:
                        payload = {"text": r["content"]}
                    if not isinstance(payload, dict):
                        payload = {"result": payload}
                    parts.append(types.Part.from_function_response(name=r["name"], response=payload))
                contents.append(types.Content(role="tool", parts=parts))
        return contents

    def _declarations(self, tools: list[dict[str, Any]]) -> list[Any]:
        types = self._types()
        decls = []
        supports_json_schema = "parameters_json_schema" in getattr(types.FunctionDeclaration, "model_fields", {})
        for t in tools:
            if supports_json_schema:
                decls.append(types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]))
            else:  # pragma: no cover - older SDKs
                decls.append(types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["parameters"]))
        return decls

    def complete(self, system: str, history: list[dict[str, Any]], tools: list[dict[str, Any]]) -> BackendResponse:
        types = self._types()
        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tools:
            config_kwargs["tools"] = [types.Tool(function_declarations=self._declarations(tools))]
        try:
            resp = self.client.models.generate_content(model=self.model, contents=self.render_contents(history),
                                                       config=types.GenerateContentConfig(**config_kwargs))
        except Exception as exc:  # the google-genai SDK raises a family of ClientError/ServerError types
            name = type(exc).__name__
            msg = str(exc)
            if "401" in msg or "403" in msg or "API key" in msg or "PERMISSION_DENIED" in msg:
                raise BackendError("The Gemini API key was rejected.", kind="auth") from exc
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise BackendError("Gemini rate limit or quota reached; try again shortly.", retryable=True, kind="rate_limit") from exc
            raise BackendError(f"Gemini API error ({name}).", retryable="5" == msg[:1], kind="api") from exc
        return self.parse_response(resp)

    def parse_response(self, resp: Any) -> BackendResponse:
        calls: list[ToolCallRequest] = []
        text_parts: list[str] = []
        candidates = getattr(resp, "candidates", None) or []
        content = candidates[0].content if candidates else None
        finish = str(getattr(candidates[0], "finish_reason", "") or "") if candidates else ""
        for i, part in enumerate(getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                args = fc.args or {}
                calls.append(ToolCallRequest(id=getattr(fc, "id", None) or f"gemini_call_{len(calls) + 1}", name=fc.name, input=dict(args)))
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                text_parts.append(part.text)
        stop = "tool_use" if calls else "end_turn"
        upper = finish.upper()
        if any(k in upper for k in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED")):
            stop = "refusal"
        elif "MAX_TOKENS" in upper:
            stop = "max_tokens"
        usage = None
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {"input_tokens": getattr(um, "prompt_token_count", None), "output_tokens": getattr(um, "candidates_token_count", None)}
        return BackendResponse(text="\n".join(text_parts).strip(), tool_calls=calls, stop_reason=stop, usage=usage, raw=content, raw_backend=self.name)


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------

def build_backend(settings: Settings) -> tuple[Optional[LLMBackend], str]:
    """Return (backend, provider_name). provider_name is 'demo' when no backend is configured."""
    provider = resolve_provider(settings)
    try:
        if provider == "anthropic":
            return AnthropicBackend(settings.anthropic_api_key, settings.anthropic_model), "anthropic"
        if provider == "gemini":
            return GeminiBackend(settings.gemini_api_key, settings.gemini_model), "gemini"
    except Exception as exc:  # SDK import/initialisation failures degrade to demo mode
        log.error("LLM backend %s unavailable: %s", provider, exc)
    return None, "demo"


__all__ = ["ToolCallRequest", "BackendResponse", "BackendError", "LLMBackend", "AnthropicBackend", "GeminiBackend", "build_backend"]
