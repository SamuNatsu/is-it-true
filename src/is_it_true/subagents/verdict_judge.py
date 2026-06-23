"""Verdict judge sub-agent — the final synthesis step.

Receives the complete chain of evidence collected across all investigation
rounds, plus any resolved contradictions, and delivers a definitive verdict
with a confidence score and a narrative summary.

This is the highest-value LLM call in the pipeline — consider using a more
capable model (e.g. gpt-4o) while keeping lighter models for other roles.
"""

from __future__ import annotations

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import (
    ContradictionResolution,
    Evidence,
    ModelConfigDict,
    VerdictJudgeResponse,
    record_token_usage,
)
from ..utils import check_finish_reason, evidence_label, parse_json_response, utc_now

VERDICT_JUDGE_SYSTEM = """You are the final verdict judge for a fact-checking investigation. You have the
complete chain of evidence collected across multiple investigation rounds. Your job is to deliver
a final, definitive verdict.

Available verdicts:
- "true": The claim is factually correct, supported by strong, consistent evidence
- "false": The claim is factually incorrect, contradicted by strong evidence
- "mostly_true": The claim is mostly correct but has minor inaccuracies or missing context
- "mostly_false": The claim is mostly incorrect but contains some truth
- "misleading": The claim is technically true but presented in a misleading way (missing context, cherry-picked)
- "unverified": There is insufficient evidence to make a determination

Confidence (0.0-1.0):
- 0.9+: Multiple authoritative sources, no contradictions, clear consensus
- 0.7-0.89: Good evidence but some gaps or minor contradictions resolved
- 0.5-0.69: Mixed evidence, significant gaps remain
- 0.3-0.49: Weak evidence, major gaps
- <0.3: Virtually no reliable evidence

Return a JSON object:
{
  "verdict": "one of the six verdict strings",
  "confidence": 0.85,
  "summary": "A clear, concise explanation of the verdict (2-4 sentences, in the claim's language)"
}

The summary field is REQUIRED and must be non-empty. Return ONLY the JSON object, no other text, markdown fences,
or commentary."""


def _build_user_prompt(
    claim: str,
    evidence_items: list[Evidence],
    contradictions: list[ContradictionResolution],
    language: str,
) -> str:
    """Build the complete evidence-chain prompt for the final verdict.

    Each evidence item includes source title, URL, up to 3 key passages
    (truncated to 500 chars each), visual findings, and credibility score.
    """
    now = utc_now()
    parts = [f"Current UTC: {now}\n\nClaim: {claim}\n\nComplete evidence chain:\n"]
    for i, ev in enumerate(evidence_items):
        support = evidence_label(ev.supports_claim)
        parts.append(f"[{i}] {support} | Credibility: {ev.source.credibility_score:.2f}")
        parts.append(f"    Source: {ev.source.title}")
        parts.append(f"    URL: {ev.source.url}")
        for p in ev.key_passages:
            parts.append(f"    Passage: {p[:500]}")
        if ev.visual_findings:
            parts.append(f"    Visual analysis: {ev.visual_findings[:500]}")
        parts.append("")
    if contradictions:
        parts.append("Resolved contradictions:")
        for c in contradictions:
            parts.append(f"  - {c}")
    parts.append(f"\nRespond in language: {language}")
    return "\n".join(parts)


async def deliver_verdict(
    claim: str,
    evidence_items: list[Evidence],
    contradictions: list[ContradictionResolution],
    language: str,
    model_config: ModelConfigDict | None = None,
) -> dict[str, str | float]:
    """Deliver the final verdict based on all collected evidence.

    Returns a dict with ``verdict``, ``confidence``, and ``summary`` keys.
    Falls back to ``unverified`` / 0.3 on failure.
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("verdict_judge", config)
    reasoning_effort = resolve_reasoning("verdict_judge")

    messages = [
        {"role": "system", "content": VERDICT_JUDGE_SYSTEM},
        {
            "role": "user",
            "content": _build_user_prompt(claim, evidence_items, contradictions, language),
        },
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
        content = response.choices[0].message.content or "{}"
        check_finish_reason(response.choices[0].finish_reason, "verdict judge")
        data = parse_json_response(content, VerdictJudgeResponse) or {}
        # Fallback: use raw content as summary if JSON parse didn't provide one
        if not data.get("summary"):
            data["summary"] = content[:500]
    except Exception as e:
        log.print(f"  verdict judge failed: {e}")
        data = {
            "verdict": "unverified",
            "confidence": 0.3,
            "summary": "Unable to synthesize a verdict due to an internal error",
        }

    return data
