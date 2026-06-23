"""Model configuration resolution.

Loads .env via python-dotenv, then resolves per-role model identifiers from
a three-level precedence chain:

    1. Explicit ``model_config`` arg to ``is_it_true()``
    2. ``IS_IT_TRUE_<ROLE>_MODEL`` env var
    3. ``IS_IT_TRUE_DEFAULT_MODEL`` env var

Also provides ``resolve_reasoning()`` for the optional per-role
reasoning/thinking effort parameter passed to litellm.
"""

from __future__ import annotations

import os

from .models import ModelConfigDict

_env_loaded = False


def _ensure_dotenv() -> None:
    """Load .env once per process — idempotent."""
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    _env_loaded = True


def _read_env(key: str) -> str | None:
    """Read an env var, returning None for empty / unset values."""
    value = os.getenv(key, "").strip()
    return value if value else None


DEFAULT_MODEL_ENV = "IS_IT_TRUE_DEFAULT_MODEL"

ROLE_ENV_MAP: dict[str, str] = {
    "query_planner": "IS_IT_TRUE_QUERY_PLANNER_MODEL",
    "evidence_extractor": "IS_IT_TRUE_EVIDENCE_EXTRACTOR_MODEL",
    "gap_detector": "IS_IT_TRUE_GAP_DETECTOR_MODEL",
    "contradiction_resolver": "IS_IT_TRUE_CONTRADICTION_RESOLVER_MODEL",
    "source_evaluator": "IS_IT_TRUE_SOURCE_EVALUATOR_MODEL",
    "verdict_judge": "IS_IT_TRUE_VERDICT_JUDGE_MODEL",
    "image_analyzer": "IS_IT_TRUE_IMAGE_ANALYZER_MODEL",
    "language_detector": "IS_IT_TRUE_LANGUAGE_DETECTOR_MODEL",
}


class ConfigError(Exception):
    """Raised when no model can be resolved for a role."""

    pass


def build_model_config(overrides: ModelConfigDict | dict | None = None) -> ModelConfigDict:
    """Build the effective ModelConfigDict from call-time overrides + env.

    Iterates every field on the model so newly added roles are automatically
    picked up without manual wiring.
    """
    _ensure_dotenv()

    # Normalise plain dict → ModelConfigDict
    if isinstance(overrides, dict):
        overrides = ModelConfigDict(**overrides)
    overrides = overrides or ModelConfigDict()

    resolved = ModelConfigDict()
    for field_name in ModelConfigDict.model_fields:
        override_val = getattr(overrides, field_name, None)
        if override_val:
            setattr(resolved, field_name, override_val)
            continue
        if field_name == "default":
            env_val = _read_env(DEFAULT_MODEL_ENV)
        else:
            env_var = ROLE_ENV_MAP.get(field_name, "")
            env_val = _read_env(env_var) if env_var else None
        if env_val:
            setattr(resolved, field_name, env_val)

    return resolved


# Valid reasoning effort values understood by litellm / provider APIs
_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "default"})


def resolve_reasoning(role: str) -> str | None:
    """Resolve reasoning effort for a role from environment.

    Checks ``IS_IT_TRUE_<ROLE>_REASONING_EFFORT`` first,
    then ``IS_IT_TRUE_REASONING_EFFORT`` as fallback.
    Returns ``None`` if neither is set (provider default behaviour).
    """
    _ensure_dotenv()
    key = f"IS_IT_TRUE_{role.upper()}_REASONING_EFFORT"
    value = _read_env(key)
    if value is not None and value in _REASONING_EFFORTS:
        return value
    value = _read_env("IS_IT_TRUE_REASONING_EFFORT")
    if value is not None and value in _REASONING_EFFORTS:
        return value
    return None


def resolve_model(role: str, config: ModelConfigDict) -> str:
    """Resolve the litellm model identifier for a specific sub-agent role.

    Uses the per-role field on ``config`` first, then ``config.default``.
    Raises ``ConfigError`` when nothing is configured.
    """
    specific = getattr(config, role, None)
    if specific:
        return specific
    if config.default:
        return config.default
    raise ConfigError(
        f"No model configured for role '{role}'. "
        "Set IS_IT_TRUE_DEFAULT_MODEL environment variable "
        "or pass a model_config to is_it_true()."
    )
