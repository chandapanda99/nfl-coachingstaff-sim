"""Single environment-backed source for application model defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from threading import Lock

from dotenv import find_dotenv, load_dotenv

from nfl_coaching_sim.providers.base import ModelProvider, REASONING_EFFORTS
from nfl_coaching_sim.runtime import user_config_path

_ENVIRONMENT_LOCK = Lock()
_ENVIRONMENT_LOADED = False
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def application_env_path() -> Path | None:
    """Resolve the one .env file used by source and packaged application runs."""

    if configured := os.environ.get("NFL_COACH_ENV_FILE"):
        return Path(configured).expanduser()
    if getattr(sys, "frozen", False):
        candidate = user_config_path()
        return candidate if candidate.is_file() else None
    discovered = find_dotenv()
    return Path(discovered) if discovered else None


def _load_environment() -> None:
    """Load the nearest .env once without overriding process-level settings."""

    global _ENVIRONMENT_LOADED
    if _ENVIRONMENT_LOADED:
        return
    with _ENVIRONMENT_LOCK:
        if not _ENVIRONMENT_LOADED:
            dotenv_path = application_env_path()
            if dotenv_path:
                load_dotenv(dotenv_path, override=False)
            _ENVIRONMENT_LOADED = True


def _first_environment_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


@dataclass(frozen=True)
class ApplicationSettings:
    """Resolved model defaults shared by the API, CLI, and provider adapters."""

    provider: str
    model: str
    base_url: str
    upstream_url: str
    model_license: str
    reasoning_effort: str | None
    log_level: str = "INFO"
    foundry_api_key: str | None = field(default=None, repr=False)


def get_application_settings(provider: str | None = None) -> ApplicationSettings:
    """Resolve provider-aware defaults from process variables and the nearest .env."""

    _load_environment()
    selected_provider = (
        (provider or _first_environment_value("NFL_COACH_MODEL_PROVIDER", default=ModelProvider.AZURE_FOUNDRY.value)).strip().lower()
    )
    provider_prefix = selected_provider.upper()

    if selected_provider == ModelProvider.AZURE_FOUNDRY.value:
        model = _first_environment_value("NFL_COACH_MODEL", "FOUNDRY_MODEL")
        base_url = _first_environment_value("NFL_COACH_MODEL_ENDPOINT", "FOUNDRY_ENDPOINT")
        upstream_url = _first_environment_value("NFL_COACH_MODEL_UPSTREAM_URL", "FOUNDRY_UPSTREAM_URL")
        model_license = _first_environment_value(
            "NFL_COACH_MODEL_LICENSE",
            "FOUNDRY_MODEL_LICENSE",
        )
    elif selected_provider == ModelProvider.OLLAMA.value:
        model = _first_environment_value("NFL_COACH_MODEL", "OLLAMA_MODEL")
        base_url = _first_environment_value(
            "NFL_COACH_MODEL_ENDPOINT",
            "OLLAMA_ENDPOINT",
            default="http://127.0.0.1:11434",
        )
        upstream_url = _first_environment_value("NFL_COACH_MODEL_UPSTREAM_URL", "OLLAMA_UPSTREAM_URL")
        model_license = _first_environment_value(
            "NFL_COACH_MODEL_LICENSE",
            "OLLAMA_MODEL_LICENSE",
        )
    else:
        model = _first_environment_value("NFL_COACH_MODEL", f"{provider_prefix}_MODEL")
        base_url = _first_environment_value("NFL_COACH_MODEL_ENDPOINT", f"{provider_prefix}_ENDPOINT")
        upstream_url = _first_environment_value("NFL_COACH_MODEL_UPSTREAM_URL", f"{provider_prefix}_UPSTREAM_URL")
        model_license = _first_environment_value(
            "NFL_COACH_MODEL_LICENSE",
            f"{provider_prefix}_MODEL_LICENSE",
        )

    reasoning_effort = (
        _first_environment_value(
            f"{provider_prefix}_REASONING_EFFORT",
            "NFL_COACH_REASONING_EFFORT",
        ).lower()
        or None
    )
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning effort must be one of {sorted(REASONING_EFFORTS)}")
    log_level = _first_environment_value("NFL_COACH_LOG_LEVEL", default="INFO").upper()
    if log_level not in _LOG_LEVELS:
        raise ValueError(f"NFL_COACH_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")

    return ApplicationSettings(
        provider=selected_provider,
        model=model,
        base_url=base_url,
        upstream_url=upstream_url,
        model_license=model_license,
        reasoning_effort=reasoning_effort,
        log_level=log_level,
        foundry_api_key=_first_environment_value("AZURE_FOUNDRY_API_KEY") or None,
    )
