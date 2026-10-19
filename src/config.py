"""Runtime settings loaded from environment variables / .env.

Nothing in this module performs network access. Secrets are read but never logged.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ProviderName = Literal["anthropic", "gemini", "demo"]

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_AVIATIONSTACK_BASE_URL = "https://api.aviationstack.com/v1"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Ignoring non-integer %s=%r; using %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    llm_provider: str                      # auto | anthropic | gemini | demo
    anthropic_api_key: Optional[str]
    anthropic_model: str
    gemini_api_key: Optional[str]
    gemini_model: str
    aviationstack_api_key: Optional[str]
    aviationstack_base_url: str
    duckdb_path: Path
    long_haul_threshold_miles: int
    min_reported_flights: int
    demo_scope_start: str                  # YYYY-MM
    nationwide_scope_start: str            # YYYY-MM
    log_level: str

    @property
    def resolved_provider(self) -> ProviderName:
        return resolve_provider(self)

    @property
    def active_model_label(self) -> str:
        provider = self.resolved_provider
        if provider == "anthropic":
            return f"Anthropic Agent ({self.anthropic_model})"
        if provider == "gemini":
            return f"Gemini Agent ({self.gemini_model})"
        return "Demo / deterministic mode"

    @property
    def live_configured(self) -> bool:
        return bool(self.aviationstack_api_key)


def load_settings(env_file: Optional[Path] = None, override: bool = False) -> Settings:
    """Load `.env` (if present) and build immutable settings.

    `override=False` keeps real environment variables authoritative over the file.
    """
    env_path = env_file or (PROJECT_ROOT / ".env")
    if env_path.exists():
        load_dotenv(env_path, override=override)

    duckdb_raw = _env("DUCKDB_PATH", "data/airport_intelligence.duckdb") or "data/airport_intelligence.duckdb"
    duckdb_path = Path(duckdb_raw)
    if not duckdb_path.is_absolute():
        duckdb_path = PROJECT_ROOT / duckdb_path

    return Settings(
        project_root=PROJECT_ROOT,
        llm_provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL) or DEFAULT_ANTHROPIC_MODEL,
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL,
        aviationstack_api_key=_env("AVIATIONSTACK_API_KEY"),
        aviationstack_base_url=_env("AVIATIONSTACK_BASE_URL", DEFAULT_AVIATIONSTACK_BASE_URL) or DEFAULT_AVIATIONSTACK_BASE_URL,
        duckdb_path=duckdb_path,
        long_haul_threshold_miles=_env_int("LONG_HAUL_THRESHOLD_MILES", 2000),
        min_reported_flights=_env_int("MIN_REPORTED_FLIGHTS", 500),
        demo_scope_start=_env("DEMO_SCOPE_START", "2023-01") or "2023-01",
        nationwide_scope_start=_env("NATIONWIDE_SCOPE_START", "2019-01") or "2019-01",
        log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
    )


def resolve_provider(settings: Settings) -> ProviderName:
    """Decide which LLM runtime is active. Keys are checked for presence only."""
    choice = settings.llm_provider
    if choice == "anthropic":
        return "anthropic" if settings.anthropic_api_key else "demo"
    if choice == "gemini":
        return "gemini" if settings.gemini_api_key else "demo"
    if choice == "demo":
        return "demo"
    # auto
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.gemini_api_key:
        return "gemini"
    return "demo"


_LOGGING_CONFIGURED = False


def configure_logging(settings: Optional[Settings] = None) -> logging.Logger:
    """Developer-safe logging: rotating file under logs/, warnings+ to stderr. Idempotent."""
    global _LOGGING_CONFIGURED
    level_name = settings.log_level if settings else "INFO"
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger("airport_intel")
    if _LOGGING_CONFIGURED:
        root.setLevel(level)
        return root
    root.setLevel(level)
    log_dir = PROJECT_ROOT / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(file_handler)
    except OSError:  # pragma: no cover - read-only filesystems
        pass
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(stream_handler)
    root.propagate = False
    _LOGGING_CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"airport_intel.{name}")
