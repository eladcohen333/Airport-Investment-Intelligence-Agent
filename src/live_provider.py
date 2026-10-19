"""Live operational context providers.

Live data answers "what is happening now" questions only. It is never written to the historical tables and never
feeds a score. Every failure mode maps to an explicit status so the UI can show an honest unavailable/error state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

import requests

from src.config import get_logger
from src.models import LiveContext

log = get_logger("live")


def _use_system_trust_store() -> bool:
    """Verify TLS against the operating-system certificate store (same approach as the BTS refresh).

    `requests` ships certifi's CA bundle; on managed Windows machines behind TLS inspection the inspection root is
    only in the Windows store, so the live call fails with SSLError. `truststore` fixes that without disabling
    verification. Alternatively set REQUESTS_CA_BUNDLE / SSL_CERT_FILE to a bundle that includes the root.
    """
    try:
        import truststore  # type: ignore

        truststore.inject_into_ssl()
        return True
    except Exception:  # pragma: no cover - optional dependency
        return False

DELAY_THRESHOLD_MINUTES = 15
DEFAULT_LIMIT = 100


@dataclass
class LiveSnapshot:
    status: str                       # ok | unavailable | error
    context: Optional[LiveContext]
    message: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    provider: str = "none"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.context is not None


class LiveFlightProvider(Protocol):
    name: str

    def get_departure_snapshot(self, airport_code: str) -> LiveSnapshot: ...

    def is_configured(self) -> bool: ...


class NullLiveProvider:
    """Explicit 'not configured' provider used when no API key is present."""

    name = "none"

    def __init__(self, reason: str = "AVIATIONSTACK_API_KEY is not set; live operational context is unavailable."):
        self.reason = reason

    def is_configured(self) -> bool:
        return False

    def get_departure_snapshot(self, airport_code: str) -> LiveSnapshot:
        return LiveSnapshot(status="unavailable", context=None, message=self.reason, provider=self.name)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class AviationstackProvider:
    """Aviationstack /v1/flights adapter (real-time departures for one airport)."""

    name = "aviationstack"

    ERROR_MESSAGES = {
        "invalid_access_key": "The Aviationstack API key was rejected (invalid_access_key).",
        "missing_access_key": "No Aviationstack API key was supplied (missing_access_key).",
        "inactive_user": "The Aviationstack account is inactive (inactive_user).",
        "function_access_restricted": "This Aviationstack endpoint is not available on the current plan (function_access_restricted).",
        "usage_limit_reached": "The Aviationstack monthly request quota has been reached (usage_limit_reached).",
        "rate_limit_reached": "Aviationstack rate limit reached (rate_limit_reached).",
        "https_access_restricted": "HTTPS access is restricted on this Aviationstack plan.",
    }

    def __init__(self, api_key: Optional[str], base_url: str = "https://api.aviationstack.com/v1",
                 session: Optional[requests.Session] = None, timeout: float = 10.0, limit: int = DEFAULT_LIMIT):
        self.api_key = api_key or None
        self.base_url = base_url.rstrip("/")
        if session is None:
            _use_system_trust_store()
        self.session = session or requests.Session()
        self.timeout = timeout
        self.limit = limit

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ---- request ----------------------------------------------------------------------
    def get_departure_snapshot(self, airport_code: str) -> LiveSnapshot:
        if not self.api_key:
            return LiveSnapshot(status="unavailable", context=None, provider=self.name,
                                message="AVIATIONSTACK_API_KEY is not set; live operational context is unavailable.")
        params = {"access_key": self.api_key, "dep_iata": airport_code.upper(), "limit": self.limit}
        url = f"{self.base_url}/flights"
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
        except requests.Timeout:
            return LiveSnapshot(status="error", context=None, provider=self.name, message="The live flight provider timed out.")
        except requests.RequestException as exc:
            log.warning("live provider connection error: %s", type(exc).__name__)
            return LiveSnapshot(status="error", context=None, provider=self.name, message="Could not reach the live flight provider.")
        return self._interpret(resp, airport_code.upper())

    def _interpret(self, resp: requests.Response, airport_code: str) -> LiveSnapshot:
        retry_after = None
        try:
            if resp.headers.get("Retry-After"):
                retry_after = int(float(resp.headers["Retry-After"]))
        except (TypeError, ValueError):
            retry_after = None
        try:
            body: Any = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            code = str(body["error"].get("code") or "")
            message = self.ERROR_MESSAGES.get(code) or f"Live provider error: {body['error'].get('message') or code or resp.status_code}."
            status = "unavailable" if resp.status_code in (401, 403, 429) or code in self.ERROR_MESSAGES else "error"
            log.warning("live provider error %s (%s)", code, resp.status_code)
            return LiveSnapshot(status=status, context=None, provider=self.name, message=message, retry_after_seconds=retry_after)
        if resp.status_code == 429:
            return LiveSnapshot(status="unavailable", context=None, provider=self.name, message=self.ERROR_MESSAGES["rate_limit_reached"], retry_after_seconds=retry_after)
        if resp.status_code in (401, 403):
            return LiveSnapshot(status="unavailable", context=None, provider=self.name, message=f"Live provider rejected the request (HTTP {resp.status_code}).")
        if resp.status_code >= 500:
            return LiveSnapshot(status="error", context=None, provider=self.name, message=f"Live provider server error (HTTP {resp.status_code}).")
        if resp.status_code != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
            return LiveSnapshot(status="error", context=None, provider=self.name, message="Live provider returned an unexpected response.")
        return LiveSnapshot(status="ok", context=self._summarize(body, airport_code), provider=self.name)

    def _summarize(self, body: dict, airport_code: str) -> LiveContext:
        flights = body.get("data") or []
        counts: dict[str, int] = {}
        delayed = 0
        delays: list[float] = []
        for f in flights:
            status = str((f or {}).get("flight_status") or "unknown").lower()
            counts[status] = counts.get(status, 0) + 1
            dep = (f or {}).get("departure") or {}
            delay = dep.get("delay")
            if isinstance(delay, (int, float)):
                delays.append(float(delay))
                if delay >= DELAY_THRESHOLD_MINUTES:
                    delayed += 1
        pagination = body.get("pagination") or {}
        total = pagination.get("total")
        window = f"first {len(flights)} departures returned by the provider"
        if isinstance(total, int) and total > len(flights):
            window += f" of {total} matching flights (sample limited by the plan's page size)"
        return LiveContext(
            airport_code=airport_code,
            provider=self.name,
            retrieved_at=_now_iso(),
            flights_sampled=len(flights),
            scheduled=counts.get("scheduled", 0),
            active=counts.get("active", 0),
            landed=counts.get("landed", 0),
            cancelled=counts.get("cancelled", 0),
            delayed_15m=delayed,
            avg_departure_delay_minutes=(sum(delays) / len(delays)) if delays else None,
            status_counts=counts,
            rate_limited=False,
            window_description=window,
            note="Live sample reflects the provider's current data feed; it is not comparable to 12-month historical rates.",
        )


def build_live_provider(api_key: Optional[str], base_url: str) -> LiveFlightProvider:
    if api_key:
        return AviationstackProvider(api_key=api_key, base_url=base_url)
    return NullLiveProvider()


__all__ = ["LiveSnapshot", "LiveFlightProvider", "NullLiveProvider", "AviationstackProvider", "build_live_provider", "DELAY_THRESHOLD_MINUTES"]
