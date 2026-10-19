"""Airport Investment Intelligence Agent - Streamlit application.

Browser -> this app -> one analytics agent (Anthropic / Gemini / deterministic demo) -> allow-listed typed tools ->
deterministic DuckDB calculations -> local aggregated store (+ optional live provider).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import streamlit as st

from src import db as dbmod
from src.agent import build_agent, update_context
from src.analytics import airport_refs, airport_rows, get_coverage
from src.airport_resolver import AirportResolver
from src.config import configure_logging, get_logger, load_settings
from src.live_provider import build_live_provider
from src.models import AgentTurn, SessionContext
from src.tools import ToolContext
from src.ui_components import inject_css, render_header, render_missing_database, render_sidebar, render_suggested_questions, render_turn

st.set_page_config(page_title="Airport Investment Intelligence Agent", layout="wide", initial_sidebar_state="expanded")
settings = load_settings()
configure_logging(settings)
log = get_logger("app")
inject_css()


@st.cache_resource(show_spinner=False)
def load_catalog(db_path: str, mtime: float):
    """Open the store read-only and preload the airport catalog and coverage (keyed by file mtime so a refresh reloads)."""
    conn = dbmod.connect(db_path, read_only=True)
    rows = airport_rows(conn)
    return {"conn": conn, "resolver": AirportResolver.from_rows(rows), "refs": airport_refs(conn), "coverage": get_coverage(conn)}


@st.cache_resource(show_spinner=False)
def load_agent(provider: str, model: str | None, key_fingerprint: int):
    return build_agent(settings)


def _db_state():
    path = Path(settings.duckdb_path)
    if not dbmod.database_ready(path):
        return None
    return load_catalog(str(path), path.stat().st_mtime)


def _run_question(question: str, catalog, defaults, bundle) -> AgentTurn:
    conn = catalog["conn"].cursor()
    try:
        tool_ctx = ToolContext(conn=conn, resolver=catalog["resolver"], refs=catalog["refs"], defaults=defaults, coverage=catalog["coverage"],
                               live_provider=build_live_provider(settings.aviationstack_api_key, settings.aviationstack_base_url), today=date.today())
        return bundle.agent.run(question, st.session_state.ctx, defaults, tool_ctx)
    finally:
        conn.close()


def main() -> None:
    catalog = _db_state()
    coverage = catalog["coverage"] if catalog else None
    defaults, reset = render_sidebar(settings, coverage, live_configured=settings.live_configured)
    if "ctx" not in st.session_state or reset:
        st.session_state.ctx = SessionContext()
        st.session_state.turns = []
        st.session_state.pending = None
    render_header(coverage)
    if catalog is None:
        render_missing_database(str(settings.duckdb_path))
        return
    if not coverage.ready:
        st.warning("The loaded data does not yet cover two full 12-month windows; rankings and scores need at least 24 months. "
                   "Run the refresh with an earlier --start month.")

    fp = hash((bool(settings.anthropic_api_key), bool(settings.gemini_api_key), settings.llm_provider))
    bundle = load_agent(settings.resolved_provider, settings.anthropic_model if settings.resolved_provider == "anthropic" else settings.gemini_model, fp)

    clicked = render_suggested_questions()
    st.markdown('<div class="section-head">Conversation</div>', unsafe_allow_html=True)
    for question, turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            render_turn(turn)

    typed = st.chat_input("Ask about airport demand growth, utilization, operational pressure, long-haul mix or live status...")
    question = typed or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Running deterministic analysis..." if bundle.mode == "demo" else f"Asking the {bundle.mode} agent and running tools..."):
                try:
                    turn = _run_question(question, catalog, defaults, bundle)
                except Exception as exc:  # noqa: BLE001 - never show a stack trace to the analyst
                    log.exception("question failed: %s", question)
                    turn = AgentTurn(mode=bundle.mode if bundle.mode in ("anthropic", "gemini", "demo") else "demo", answer_kind="error",
                                     conclusion="The analysis could not be completed because of an internal error. Details were written to logs/app.log.",
                                     answer_markdown=f"Error type: {type(exc).__name__}. Try rephrasing the question or check the data refresh.")
            render_turn(turn)
        st.session_state.turns.append((question, turn))
        st.session_state.ctx = update_context(st.session_state.ctx, question, turn)


main()
