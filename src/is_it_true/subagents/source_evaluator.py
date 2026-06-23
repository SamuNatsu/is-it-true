"""Source credibility evaluator sub-agent.

Rates web source credibility on a 0.0–1.0 scale using LLM judgment combined
with domain heuristics. Called for every source discovered in a round.

The domain heuristic (``domain_credibility_score``) provides a fast baseline for
well-known domains and serves as a fallback when the LLM call fails.
"""

from __future__ import annotations

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import ModelConfigDict, SourceEvaluatorResponse, record_token_usage
from ..utils import check_finish_reason, domain_credibility_score, parse_json_response, utc_now

SOURCE_EVALUATOR_SYSTEM = """You are a source credibility evaluator. Rate the credibility of a web source
for fact-checking purposes on a scale of 0.0 (completely untrustworthy) to 1.0 (authoritative primary source).

Consider:
- Domain authority: .gov > .edu > established news orgs > blogs > social media
- Known fact-checking or journalistic standards
- Primary vs secondary vs tertiary source
- Whether the content is opinion/editorial or factual reporting
- Publication date recency relative to the claim

Return a JSON object:
{
  "score": 0.85,
  "rationale": "Brief explanation of the score"
}

Return ONLY the JSON object — no markdown fences, no other text."""


def _build_user_prompt(source_title: str, source_url: str, content: str) -> str:
    """Build the evaluation prompt with the first 3000 chars of source content."""
    now = utc_now()
    content_snippet = content[:3000]
    return (
        f"Current UTC: {now}\n\n"
        f"Source: {source_title}\nURL: {source_url}\nContent excerpt:\n{content_snippet}"
    )


async def evaluate_source(
    source_title: str,
    source_url: str,
    content: str,
    model_config: ModelConfigDict | None = None,
) -> tuple[float, str]:
    """Evaluate a single source's credibility.

    Returns ``(score, rationale)`` where score is clamped to [0.0, 1.0].
    Falls back to the domain heuristic on LLM failure.
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("source_evaluator", config)
    reasoning_effort = resolve_reasoning("source_evaluator")

    messages = [
        {"role": "system", "content": SOURCE_EVALUATOR_SYSTEM},
        {"role": "user", "content": _build_user_prompt(source_title, source_url, content)},
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        content_raw = response.choices[0].message.content or "{}"
        check_finish_reason(response.choices[0].finish_reason, "source evaluator")
        data = parse_json_response(content_raw, SourceEvaluatorResponse) or {}
    except Exception as e:
        log.print(f"  source evaluator failed: {e}")
        return domain_credibility_score(source_url), str(e)

    score = data.get("score", 0.5)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    return max(0.0, min(1.0, score)), data.get("rationale", "")
