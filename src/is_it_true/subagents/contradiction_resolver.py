"""Contradiction resolver sub-agent.

When two evidence items conflict, this sub-agent determines which source
is more credible and explains why. Resolution results are included in the
final report.
"""

from __future__ import annotations

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import (
    ContradictionResolution,
    ContradictionResolverResponse,
    Evidence,
    ModelConfigDict,
    record_token_usage,
)
from ..utils import check_finish_reason, parse_json_response, utc_now

CONTRADICTION_RESOLVER_SYSTEM = """You are a contradiction resolver for fact-checking. Two sources appear to
contradict each other. Your job is to determine which source is more credible and why.

Consider:
- Source credibility (domain authority, known biases, primary vs secondary)
- Publication date (newer may have corrected info)
- Specificity and detail level
- Whether either source cites primary evidence
- Whether the contradiction might be a misunderstanding or different context

Return a JSON object:
{
  "resolution": "Explanation of the contradiction and which source to trust",
  "trusted_source_index": 0,
  "reasoning": "Detailed reasoning for the choice"
}

trusted_source_index: 0 for source A, 1 for source B.

Return ONLY the JSON object — no markdown fences, no other text."""


def _build_user_prompt(
    claim: str,
    evidence_a: Evidence,
    evidence_b: Evidence,
) -> str:
    """Build prompt with both conflicting sources and their key passages."""
    now = utc_now()
    return (
        f"Current UTC: {now}\n\n"
        f"Claim: {claim}\n\n"
        f"Source A: {evidence_a.source.title}\n"
        f"URL A: {evidence_a.source.url}\n"
        f"Content A: {'; '.join(evidence_a.key_passages)[:2000]}\n\n"
        f"Source B: {evidence_b.source.title}\n"
        f"URL B: {evidence_b.source.url}\n"
        f"Content B: {'; '.join(evidence_b.key_passages)[:2000]}"
    )


async def resolve_contradiction(
    claim: str,
    evidence_a: Evidence,
    evidence_b: Evidence,
    model_config: ModelConfigDict | None = None,
) -> ContradictionResolution:
    """Resolve a contradiction between two evidence items.

    Returns a ``ContradictionResolution`` with the resolution, trusted
    source, and reasoning.
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("contradiction_resolver", config)
    reasoning_effort = resolve_reasoning("contradiction_resolver")

    messages = [
        {"role": "system", "content": CONTRADICTION_RESOLVER_SYSTEM},
        {"role": "user", "content": _build_user_prompt(claim, evidence_a, evidence_b)},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        content = response.choices[0].message.content or "{}"
        check_finish_reason(response.choices[0].finish_reason, "contradiction resolver")
        data = parse_json_response(content, ContradictionResolverResponse) or {}
    except Exception as e:
        log.print(f"  contradiction resolver failed: {e}")
        data = {"resolution": "Could not resolve", "trusted_source_index": 0, "reasoning": str(e)}

    trusted_idx = data.get("trusted_source_index", 0)
    return ContradictionResolution(
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        resolution=data.get("resolution", ""),
        trusted_source=evidence_a.source.url if trusted_idx == 0 else evidence_b.source.url,
        reasoning=data.get("reasoning", ""),
    )
