"""Gap detector sub-agent.

Examines all evidence collected so far and identifies:
* Unsettled questions that need further investigation (gaps).
* Evidence pairs that contradict each other (needs resolution).

Runs once per round on the full accumulated evidence set.
"""

from __future__ import annotations

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import Evidence, Gap, GapDetectorResponse, ModelConfigDict, record_token_usage
from ..utils import check_finish_reason, evidence_label, parse_json_response, utc_now

GAP_DETECTOR_SYSTEM = """You are a rigorous gap detector for fact-checking. Your job is to examine all evidence
collected so far and identify what's still unsettled, unverified, or contradictory.

Look for:
- Claims within the main claim that have NO evidence at all
- Evidence that contradicts other evidence (flag for resolution)
- Evidence that is too weak, vague, or from untrustworthy sources
- Missing perspectives: were only supporting or only contradicting sources found?
- Facts that need confirmation from a second independent source
- Dates, numbers, statistics that need verification
- Logical gaps in the chain of evidence

Return a JSON object:
{
  "gaps": [
    {"question": "Specific question that needs investigation", "reason": "Why existing evidence is insufficient"}
  ],
  "contradictions": [
    {"source_a_index": 0, "source_b_index": 1, "description": "What contradicts"}
  ],
  "assessment": "Overall assessment of evidence completeness (1-2 sentences)"
}

If all evidence is sufficient and consistent, return {"gaps": [], "contradictions": [], "assessment": "..."}.

Return ONLY the JSON object — no markdown fences, no other text."""


def _build_user_prompt(claim: str, evidence_items: list[Evidence]) -> str:
    """Build the user prompt with claim and all evidence (up to 3 key passages per item)."""
    now = utc_now()
    parts = [f"Current UTC: {now}\n\nClaim: {claim}\n\nEvidence collected so far:\n"]
    for i, ev in enumerate(evidence_items):
        support = evidence_label(ev.supports_claim)
        passages = "; ".join(ev.key_passages[:3]) if ev.key_passages else "no key passages"
        parts.append(f"[{i}] {support} | {ev.source.title[:100]} | {ev.source.url}")
        parts.append(f"    Passages: {passages[:500]}")
        if ev.visual_findings:
            parts.append(f"    Visual: {ev.visual_findings[:300]}")
    return "\n".join(parts)


async def detect_gaps(
    claim: str,
    evidence_items: list[Evidence],
    model_config: ModelConfigDict | None = None,
) -> tuple[list[Gap], list[tuple[int, int, str]]]:
    """Detect gaps and contradictions in the collected evidence.

    Returns a tuple of ``(gaps, contradictions)`` where contradictions are
    ``(source_a_index, source_b_index, description)`` tuples pointing into
    the evidence_items list.
    """
    if not evidence_items:
        return [
            Gap(question="Find any evidence about this claim", reason="No evidence collected yet")
        ], []

    config = model_config or ModelConfigDict()
    model = resolve_model("gap_detector", config)
    reasoning_effort = resolve_reasoning("gap_detector")

    messages = [
        {"role": "system", "content": GAP_DETECTOR_SYSTEM},
        {"role": "user", "content": _build_user_prompt(claim, evidence_items)},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        check_finish_reason(response.choices[0].finish_reason, "gap detector")
        content = response.choices[0].message.content
        if not content:
            return [
                Gap(
                    question="Verify the claim with additional sources",
                    reason="Gap detector returned empty response",
                )
            ], []
        return _parse_gaps(content)
    except Exception as e:
        log.print(f"  gap detector failed: {e}")
        # Return a fallback gap so a transient LLM failure does not
        # prematurely terminate the investigation as "all gaps resolved".
        return [
            Gap(
                question="Verify the claim with additional sources",
                reason="Gap detector unavailable — retrying",
            )
        ], []


def _parse_gaps(content: str) -> tuple[list[Gap], list[tuple[int, int, str]]]:
    """Parse gaps and contradictions from JSON LLM response.

    Validated via json-repair + Pydantic.
    """
    data = parse_json_response(content, GapDetectorResponse)
    if not isinstance(data, dict):
        return [], []

    gaps = [
        Gap(question=g.get("question", ""), reason=g.get("reason", ""))
        for g in data.get("gaps", [])
    ]
    contradictions = [
        (c.get("source_a_index", 0), c.get("source_b_index", 0), c.get("description", ""))
        for c in data.get("contradictions", [])
    ]
    return gaps, contradictions
