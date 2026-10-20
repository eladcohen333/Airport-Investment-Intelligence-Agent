"""Live LLM tool-calling smoke test.

Runs the suggested questions, the brief's verbatim questions and one follow-up through the active LLM backend
against the local DuckDB store and prints which tools were called and the conclusions. Needs ANTHROPIC_API_KEY or
GEMINI_API_KEY in the environment / .env and a refreshed database.

Usage: python scripts/smoke_llm.py [--questions N] [--exam-only] [--provider auto|anthropic|gemini]
Exit code 0 when every question produced an answer or clarification grounded in tool calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db as dbmod  # noqa: E402
from src.agent import build_agent, update_context  # noqa: E402
from src.config import configure_logging, load_settings  # noqa: E402
from src.demo_router import SUGGESTED_QUESTIONS  # noqa: E402
from src.live_provider import build_live_provider  # noqa: E402
from src.models import RunDefaults, SessionContext  # noqa: E402
from src.tools import build_tool_context  # noqa: E402

# The assignment's wording as written in the brief; SUGGESTED_QUESTIONS are the code-friendly rephrasings shown in the UI.
EXAM_QUESTIONS = (
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
)
FOLLOW_UP = ("Compare LAX and SNA congestion.", "What about 2024?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=int, default=len(SUGGESTED_QUESTIONS))
    ap.add_argument("--provider", default=None, help="override LLM_PROVIDER for this run")
    ap.add_argument("--exam-only", action="store_true", help="run only the brief's verbatim questions plus the follow-up (fewer API calls)")
    args = ap.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    settings = load_settings(override=True)
    configure_logging(settings)
    bundle = build_agent(settings)
    print(f"LLM mode: {settings.active_model_label} (provider={bundle.mode})")
    if bundle.mode == "demo":
        print("No LLM key configured (ANTHROPIC_API_KEY / GEMINI_API_KEY). Nothing to smoke-test; demo mode is covered by pytest.")
        return 2
    if not dbmod.database_ready(settings.duckdb_path):
        print(f"No database at {settings.duckdb_path}; run scripts/refresh_data.py --scope demo first.")
        return 2
    conn = dbmod.connect(settings.duckdb_path, read_only=True)
    defaults = RunDefaults(data_scope="demo", long_haul_threshold_miles=settings.long_haul_threshold_miles,
                           min_reported_flights=settings.min_reported_flights, live_enabled=bool(settings.aviationstack_api_key))
    tool_ctx = build_tool_context(conn, defaults, live_provider=build_live_provider(settings.aviationstack_api_key, settings.aviationstack_base_url))
    failures = 0
    ctx = SessionContext()
    suggested = [] if args.exam_only else list(SUGGESTED_QUESTIONS[: args.questions])
    questions = suggested + list(EXAM_QUESTIONS) + list(FOLLOW_UP)
    for q in questions:
        t0 = time.time()
        turn = bundle.agent.run(q, ctx, defaults, tool_ctx)
        ctx = update_context(ctx, q, turn)
        calls = [f"{c.name}({json.dumps(c.input, default=str)[:90]}) -> {c.result.status}" for c in turn.tool_calls]
        ok = turn.answer_kind in ("answer", "clarification") and (turn.tool_calls or turn.answer_kind == "clarification")
        failures += 0 if ok else 1
        print("\n" + "=" * 100)
        print(f"Q: {q}\n   kind={turn.answer_kind} model={turn.model} {time.time() - t0:.1f}s usage={turn.usage}")
        for c in calls:
            print(f"   tool: {c}")
        print(f"   conclusion: {turn.conclusion[:400]}")
        if turn.grounding_warnings:
            print(f"   grounding warnings: {turn.grounding_warnings}")
    conn.close()
    print("\n" + ("SMOKE OK" if failures == 0 else f"SMOKE FAILURES: {failures}"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
