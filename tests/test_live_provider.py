"""Phase 6: live provider error mapping and summarization (fake HTTP session; fixture JSON clearly named as a fixture)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from src import live_provider as lp

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class _Resp:
    def __init__(self, status_code, body=None, text="", headers=None):
        self.status_code = status_code
        self._body = body
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class _Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.resp


def test_missing_key_is_explicit_unavailable():
    snap = lp.AviationstackProvider(api_key=None).get_departure_snapshot("LAX")
    assert snap.status == "unavailable" and "AVIATIONSTACK_API_KEY" in snap.message and snap.context is None
    null = lp.NullLiveProvider()
    assert not null.is_configured() and null.get_departure_snapshot("LAX").status == "unavailable"
    assert isinstance(lp.build_live_provider(None, "https://api.aviationstack.com/v1"), lp.NullLiveProvider)
    assert isinstance(lp.build_live_provider("key", "https://api.aviationstack.com/v1"), lp.AviationstackProvider)


def test_happy_path_summarizes_fixture_payload():
    body = json.loads((FIXTURES / "fixture_aviationstack_flights.json").read_text(encoding="utf-8"))
    session = _Session(_Resp(200, body))
    provider = lp.AviationstackProvider(api_key="k", session=session, timeout=7)   # type: ignore[arg-type]
    snap = provider.get_departure_snapshot("lax")
    assert snap.ok and snap.provider == "aviationstack"
    call = session.calls[0]
    assert call["url"].endswith("/flights") and call["params"]["dep_iata"] == "LAX" and call["params"]["limit"] == 100 and call["timeout"] == 7
    ctx = snap.context
    assert ctx.airport_code == "LAX" and ctx.flights_sampled == 6
    assert ctx.delayed_15m == 2                 # delays 20 and 45; 10 is below the 15-minute threshold; nulls ignored
    assert ctx.cancelled == 1 and ctx.active == 2 and ctx.scheduled == 2 and ctx.landed == 1
    assert ctx.avg_departure_delay_minutes == pytest.approx((20 + 45 + 10 + 0) / 4)
    assert ctx.status_counts == {"scheduled": 2, "active": 2, "landed": 1, "cancelled": 1}
    assert "of 120 matching flights" in ctx.window_description and ctx.retrieved_at.endswith("+00:00")


@pytest.mark.parametrize(
    "status,code,expected_status,needle",
    [
        (401, "invalid_access_key", "unavailable", "rejected"),
        (401, "missing_access_key", "unavailable", "No Aviationstack API key"),
        (403, "function_access_restricted", "unavailable", "not available on the current plan"),
        (429, "usage_limit_reached", "unavailable", "monthly request quota"),
        (429, "rate_limit_reached", "unavailable", "rate limit"),
        (500, "internal_error", "error", "Live provider error"),
    ],
)
def test_api_error_codes_map_to_status(status, code, expected_status, needle):
    body = {"error": {"code": code, "message": "x"}}
    provider = lp.AviationstackProvider(api_key="k", session=_Session(_Resp(status, body, headers={"Retry-After": "30"} if status == 429 else None)))  # type: ignore[arg-type]
    snap = provider.get_departure_snapshot("LAX")
    assert snap.status == expected_status and needle in snap.message and snap.context is None
    if status == 429:
        assert snap.retry_after_seconds == 30


def test_http_failures_without_json_body():
    assert lp.AviationstackProvider(api_key="k", session=_Session(_Resp(429, None, text="slow down"))).get_departure_snapshot("LAX").status == "unavailable"  # type: ignore[arg-type]
    assert lp.AviationstackProvider(api_key="k", session=_Session(_Resp(503, None, text="<html>"))).get_departure_snapshot("LAX").status == "error"  # type: ignore[arg-type]
    malformed = lp.AviationstackProvider(api_key="k", session=_Session(_Resp(200, {"unexpected": True}))).get_departure_snapshot("LAX")  # type: ignore[arg-type]
    assert malformed.status == "error" and "unexpected response" in malformed.message
    timeout = lp.AviationstackProvider(api_key="k", session=_Session(exc=requests.Timeout())).get_departure_snapshot("LAX")  # type: ignore[arg-type]
    assert timeout.status == "error" and "timed out" in timeout.message
    conn = lp.AviationstackProvider(api_key="k", session=_Session(exc=requests.ConnectionError())).get_departure_snapshot("LAX")  # type: ignore[arg-type]
    assert conn.status == "error" and "Could not reach" in conn.message


def test_live_never_touches_historical_tables(fixture_conn):
    before = {t: fixture_conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("airport_monthly_traffic", "airport_monthly_operations", "airport_scores")}
    body = json.loads((FIXTURES / "fixture_aviationstack_flights.json").read_text(encoding="utf-8"))
    lp.AviationstackProvider(api_key="k", session=_Session(_Resp(200, body))).get_departure_snapshot("LAX")  # type: ignore[arg-type]
    after = {t: fixture_conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in before}
    assert before == after
