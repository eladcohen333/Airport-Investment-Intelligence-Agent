"""Phase 0 smoke tests: the package imports and the Streamlit entry point compiles."""
import py_compile
from pathlib import Path

from src.config import load_settings, resolve_provider

ROOT = Path(__file__).resolve().parent.parent


def test_app_compiles():
    py_compile.compile(str(ROOT / "app.py"), doraise=True)


def test_settings_without_keys_resolve_to_demo(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    settings = load_settings(env_file=ROOT / "nonexistent.env")
    assert resolve_provider(settings) == "demo"
    assert settings.active_model_label == "Demo / deterministic mode"
    assert settings.long_haul_threshold_miles == 2000
    assert settings.min_reported_flights == 500
