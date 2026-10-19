"""Live provider smoke test: one Aviationstack call for an airport, printed as the tool would see it.

Usage: python scripts/smoke_live.py [AIRPORT_CODE]      (default LAX; needs AVIATIONSTACK_API_KEY)
Exit code 0 when the provider returned a usable snapshot, 2 when the key is missing, 1 on provider errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import configure_logging, load_settings  # noqa: E402
from src.live_provider import build_live_provider  # noqa: E402


def main() -> int:
    code = (sys.argv[1] if len(sys.argv) > 1 else "LAX").upper()
    settings = load_settings()
    configure_logging(settings)
    provider = build_live_provider(settings.aviationstack_api_key, settings.aviationstack_base_url)
    snap = provider.get_departure_snapshot(code)
    print(f"provider={snap.provider} status={snap.status}")
    if not snap.ok:
        print(f"message: {snap.message}")
        if snap.retry_after_seconds:
            print(f"retry after: {snap.retry_after_seconds}s")
        return 2 if snap.status == "unavailable" and not provider.is_configured() else 1
    ctx = snap.context
    print(f"retrieved_at={ctx.retrieved_at} flights_sampled={ctx.flights_sampled} delayed_15m={ctx.delayed_15m} cancelled={ctx.cancelled} "
          f"active={ctx.active} scheduled={ctx.scheduled} avg_delay={ctx.avg_departure_delay_minutes}")
    print(f"status_counts={ctx.status_counts}")
    print(ctx.window_description)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
