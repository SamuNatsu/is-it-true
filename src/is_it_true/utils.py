"""Shared utilities used by sub-agents and the orchestrator.

Includes finish-reason validation, json-repair + Pydantic response parsing,
verdict colour mapping, and UTC timestamp formatting.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, TypeAdapter

from . import logging as log

# litellm finish reasons that indicate a clean, complete response
_GOOD_FINISH_REASONS = frozenset({"stop", "eos"})

# Rich markup styles for each verdict — keep in sync with the Verdict type in models.py
_VERDICT_STYLES: dict[str, str] = {
    "true": "bold green",
    "mostly_true": "bold green",
    "false": "bold red",
    "mostly_false": "bold red",
    "misleading": "bold yellow",
    "unverified": "bold yellow",
}


def check_finish_reason(finish_reason: str | None, label: str) -> bool:
    """Validate that an LLM response completed cleanly.

    Logs a warning for non-clean finish reasons (length, content_filter, etc.).
    Returns ``True`` if the finish reason is acceptable.
    """
    if finish_reason is None or finish_reason in _GOOD_FINISH_REASONS:
        return True
    log.print(
        f"  [{label}] WARNING: finish_reason={finish_reason}, response may be truncated or filtered"
    )
    return False


def utc_now() -> str:
    """Return current UTC timestamp formatted for insertion into LLM prompts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ (%A, %d %B %Y)")


def parse_json_response(
    text: str,
    model: type[BaseModel] | TypeAdapter,
) -> Any | None:
    """Parse and validate LLM JSON output using ``json-repair`` + Pydantic.

    1. Repair malformed JSON via ``json_repair.repair_json()`` (fixes unescaped
       quotes, trailing commas, missing brackets, etc.).
    2. Parse the repaired string with ``json.loads``.
    3. Validate and coerce with the Pydantic *model*.
    4. Return ``dict`` (for ``BaseModel``) or the validated value (for
       ``TypeAdapter``). Returns ``None`` on any failure.

    Args:
        text: Raw LLM response content (may contain broken JSON).
        model: A ``BaseModel`` subclass or ``TypeAdapter`` to validate against.
    """
    content = text.strip()
    if not content:
        return None

    # Step 1: try json-repair first (handles everything from unescaped quotes
    #         to truncated responses)
    parsed = _load_and_validate(content, model)
    if parsed is not None:
        return parsed

    # Step 2: try regex-extract a JSON block and repair that
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        match = re.search(pattern, content)
        if match:
            parsed = _load_and_validate(match.group(0), model)
            if parsed is not None:
                return parsed

    return None


def _load_and_validate(
    json_text: str,
    model: type[BaseModel] | TypeAdapter,
) -> Any | None:
    """Try json-repair → json.loads → Pydantic validation.

    Falls back to plain ``json.loads`` if ``json-repair`` is unavailable or
    raises, so the function degrades gracefully without the optional dependency.
    """
    try:
        from json_repair import repair_json

        repaired = repair_json(json_text)
    except Exception:
        # json-repair unavailable or can't fix — try raw text
        repaired = json_text

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        return None

    try:
        if isinstance(model, TypeAdapter):
            return model.validate_python(data)
        validated = model.model_validate(data)
        return validated.model_dump()
    except Exception:
        return None


# Mapping from Evidence.supports_claim to display label
_EVIDENCE_LABELS = {True: "SUPPORTS", False: "CONTRADICTS", None: "NEUTRAL"}

# Domain credibility scores for known TLDs and news/science organisations
_DOMAIN_SCORES: dict[str, float] = {
    ".gov": 0.8,
    ".edu": 0.8,
    ".mil": 0.8,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bbc.com": 0.85,
    "npr.org": 0.85,
    "wikipedia.org": 0.75,
    "britannica.com": 0.75,
    "arxiv.org": 0.8,
    "nature.com": 0.8,
    "science.org": 0.8,
}


def domain_credibility_score(url: str, default: float = 0.5) -> float:
    """Fast credibility score from domain alone — no LLM call needed."""
    for domain, score in _DOMAIN_SCORES.items():
        if domain in url:
            return score
    return default


def evidence_label(supports_claim: bool | None) -> str:
    """Return a human-readable label for a tri-state supports_claim value."""
    return _EVIDENCE_LABELS.get(supports_claim, "NEUTRAL")


def verdict_style(verdict: str) -> str:
    """Return the Rich markup style string for a verdict."""
    return _VERDICT_STYLES.get(verdict, "")
