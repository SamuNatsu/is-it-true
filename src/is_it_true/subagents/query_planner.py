"""Query planner sub-agent.

Generates targeted English-language search queries from the claim and
optionally from unresolved gaps (round 2+). Always produces queries in
English regardless of the claim's language, since English search indexes
are richer and more authoritative.
"""

from __future__ import annotations

import litellm

from .. import logging as log
from ..config import resolve_model, resolve_reasoning
from ..models import Gap, ModelConfigDict, QueryPlannerResponse, record_token_usage
from ..utils import check_finish_reason, parse_json_response, utc_now

QUERY_PLANNER_SYSTEM = """You are a research query planner. Your job is to generate effective web search queries
to find evidence that verifies or refutes a claim.

Guidelines:
- You MUST generate at least 3 queries.
- Cover different angles: factual verification, opposing viewpoints, source authority
- For time-sensitive claims, include the current date/year in queries
- For claims about specific people/events, query for primary sources
- Prefer queries that target authoritative sources (news, academic, government)
- ALWAYS generate queries in English, regardless of the claim's language.
  English-language search results are far richer, more up-to-date, and more authoritative.
  Translate the search intent into English if the claim is not in English.

Example output:
{"queries": ["Is climate change caused by human activity scientific consensus", "climate change natural vs anthropogenic causes IPCC 2026", "global temperature rise data 2025 2026 NASA NOAA"]}

Return ONLY the JSON object — no markdown fences, no other text."""


def _looks_english(text: str, threshold: float = 0.7) -> bool:
    """Heuristic: if >=70% of letters are ASCII a-z/A-Z, treat as English.

    Used to add a translation hint to the prompt for non-English claims.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    ascii_count = sum(1 for c in letters if ord(c) < 128)
    return (ascii_count / len(letters)) >= threshold


def _build_user_prompt(claim: str, gaps: list[Gap] | None = None) -> str:
    """Build the user prompt with claim, optional gaps, and UTC timestamp."""
    now = utc_now()
    lang_hint = (
        "The claim is not in English — translate its meaning into English for your search queries."
        if not _looks_english(claim)
        else ""
    )
    if gaps:
        gap_text = "\n".join(f"- {g.question} (reason: {g.reason})" for g in gaps)
        return (
            f"Current UTC: {now}\n\n"
            f"Claim: {claim}\n"
            f"{lang_hint}\n\n"
            f"Unsettled questions from previous investigation:\n{gap_text}\n\n"
            f"Generate targeted search queries to investigate these gaps."
        )
    return (
        f"Current UTC: {now}\n\n"
        f"Claim: {claim}\n"
        f"{lang_hint}\n\n"
        f"Generate search queries to investigate this claim."
    )


async def plan_queries(
    claim: str,
    gaps: list[Gap] | None = None,
    model_config: ModelConfigDict | None = None,
) -> list[str]:
    """Generate search queries for the claim (and optionally gaps).

    Returns at least one query — falls back to the claim itself on failure.
    """
    config = model_config or ModelConfigDict()
    model = resolve_model("query_planner", config)
    reasoning_effort = resolve_reasoning("query_planner")
    messages = [
        {"role": "system", "content": QUERY_PLANNER_SYSTEM},
        {"role": "user", "content": _build_user_prompt(claim, gaps)},
    ]
    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.3,  # slight creativity for diverse query angles
            reasoning_effort=reasoning_effort,
        )
        record_token_usage(response.usage)
        check_finish_reason(response.choices[0].finish_reason, "query planner")
        content = response.choices[0].message.content
        if not content:
            return [claim]
        return _parse_queries(content) or [claim]
    except Exception as e:
        log.print(f"  query planner failed: {e}")
        return [claim]


def _parse_queries(content: str) -> list[str]:
    """Parse queries from LLM output via json-repair + Pydantic.

    Returns up to 5 queries, or an empty list on failure.
    """
    text = content.strip()
    parsed = parse_json_response(text, QueryPlannerResponse)
    if isinstance(parsed, dict):
        return [q for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()][:5]
    return []
