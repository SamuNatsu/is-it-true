"""Evidence extractor sub-agent.

Reads source content (up to 4000 chars per source) and extracts structured
evidence: whether the source supports, contradicts, or is neutral to the
claim, plus exact key passages from the text.

Sources are sent in a single batch to reduce LLM round-trips. Each source
is numbered and the response is a JSON array aligned by index.
"""

from __future__ import annotations

from dataclasses import dataclass

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import (
    Evidence,
    EvidenceExtractorResponse,
    ModelConfigDict,
    Source,
    record_token_usage,
)
from ..utils import check_finish_reason, parse_json_response, utc_now

EVIDENCE_EXTRACTOR_SYSTEM = """You are an evidence extractor. Your job is to read search results and web page
content, then extract key facts that support or contradict a given claim.

For each piece of evidence:
- Identify whether it SUPPORTS, CONTRADICTS, or is NEUTRAL to the claim
- Extract exact key passages from the source (quote verbatim when possible)
- Note the publication date and source credibility indicators
- If the content is too short or irrelevant, mark it as NEUTRAL with no key passages

Return a JSON object:
{
  "evidence": [
    {
      "supports_claim": true/false/null,
      "key_passages": ["exact quote 1", "exact quote 2"],
      "summary": "one-sentence summary of what this source says about the claim"
    }
  ]
}

supports_claim: true = supports the claim, false = contradicts, null = neutral/unrelated

Return ONLY the JSON object — no markdown fences, no other text."""


def _batch_user_prompt(claim: str, sources: list[dict]) -> str:
    """Build a prompt containing all sources with their content truncated to 4000 chars each."""
    now = utc_now()
    parts = [f"Current UTC: {now}\n\nClaim: {claim}\n\nSources:\n"]
    for i, s in enumerate(sources):
        content = s.get("content", "")[:4000]
        parts.append(f"--- Source {i + 1} ---")
        parts.append(f"Title: {s.get('title', '')}")
        parts.append(f"URL: {s.get('url', '')}")
        parts.append(f"Content: {content}")
    return "\n".join(parts)


@dataclass
class SourceInput:
    """Shaped input for the evidence extractor prompt."""

    title: str
    url: str
    content: str


async def extract_evidence(
    claim: str,
    sources: list[Source],
    model_config: ModelConfigDict | None = None,
) -> list[Evidence]:
    """Extract structured evidence from a batch of sources.

    Each source in the input list maps to one Evidence object in the output.
    Alignment is by index — evidence[i] corresponds to sources[i].
    """
    if not sources:
        return []

    config = model_config or ModelConfigDict()
    model = resolve_model("evidence_extractor", config)
    reasoning_effort = resolve_reasoning("evidence_extractor")

    # Build prompt inputs — use extracted_text when available, snippet as fallback
    source_inputs = [
        SourceInput(title=s.title, url=s.url, content=s.extracted_text or s.snippet)
        for s in sources
    ]
    source_dicts = [
        {"title": si.title, "url": si.url, "content": si.content} for si in source_inputs
    ]

    messages = [
        {"role": "system", "content": EVIDENCE_EXTRACTOR_SYSTEM},
        {"role": "user", "content": _batch_user_prompt(claim, source_dicts)},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=8000,
            temperature=0.0,  # deterministic for consistent extraction
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        check_finish_reason(response.choices[0].finish_reason, "evidence extractor")
        content = response.choices[0].message.content
        if not content:
            return []
        parsed_list = _parse_evidence_list(content)
    except Exception as e:
        log.print(f"  evidence extractor failed: {e}")
        return []

    # Align parsed evidence with source objects by index
    evidence_items = []
    for i, parsed in enumerate(parsed_list):
        if i >= len(sources):
            break
        source = sources[i]
        evidence_items.append(
            Evidence(
                supports_claim=parsed.get("supports_claim"),
                source=source,
                key_passages=parsed.get("key_passages", []),
            )
        )
    return evidence_items


def _parse_evidence_list(content: str) -> list[dict]:
    """Parse the evidence JSON array from LLM output.

    Expects ``{"evidence": [...]}`` — validated via json-repair + Pydantic.
    """
    data = parse_json_response(content, EvidenceExtractorResponse)
    if isinstance(data, dict) and "evidence" in data:
        return data["evidence"]
    return []
