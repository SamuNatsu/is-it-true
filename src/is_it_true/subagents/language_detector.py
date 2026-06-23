"""Language detection sub-agent.

A lightweight LLM call that returns an ISO 639-1 language code for the claim.
Used once per investigation to tag the report and guide later sub-agents
(e.g. the verdict judge writes in the claim's language).
"""

from __future__ import annotations

import re

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import ModelConfigDict, record_token_usage
from ..utils import check_finish_reason

LANGUAGE_DETECTOR_SYSTEM = (
    "Detect the language of the given text. "
    "Return ONLY the ISO 639-1 two-letter language code. "
    "Examples: en, zh, fr, de, ja, es, ar, pt, ru, ko, hi, it, nl, pl, sv, tr, vi, th, id, ms. "
    "Output the code and nothing else — no punctuation, no explanation, no markdown."
)


def _parse_language_code(raw: str) -> str | None:
    """Extract a 2-letter ISO 639-1 code from model output.

    Tries exact 2-letter match, first word, and regex fallback
    in decreasing order of confidence.
    """
    text = raw.strip().lower()

    # Exact 2-letter alpha match
    if len(text) == 2 and text.isalpha():
        return text

    # First word (handles "en.", "en,", " English", etc.)
    word = re.sub(r"^[^a-z]+", "", text)
    word = re.split(r"[^a-z]", word)[0]
    if len(word) == 2:
        return word

    # Fallback: scan for the last 2-letter sequence (answer typically at end)
    matches = re.findall(r"\b([a-z]{2})\b", text)
    if matches:
        return matches[-1]

    return None


def _build_user_prompt(text: str) -> str:
    """Send only the first 500 characters of the claim."""
    return text[:500]


async def detect_language(
    text: str,
    model_config: ModelConfigDict | None = None,
) -> str:
    """Detect the language of *text*.

    Returns an ISO 639-1 code (e.g. ``"en"``). Defaults to ``"en"`` on failure.
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("language_detector", config)
    reasoning_effort = resolve_reasoning("language_detector")

    messages = [
        {"role": "system", "content": LANGUAGE_DETECTOR_SYSTEM},
        {"role": "user", "content": _build_user_prompt(text)},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=200,  # tiny — we only need 2 letters
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        check_finish_reason(response.choices[0].finish_reason, "language detector")
        content = response.choices[0].message.content
        if content:
            code = _parse_language_code(content)
            if code:
                return code
    except Exception:
        log.print("  language detector failed, defaulting to 'en'")

    return "en"
