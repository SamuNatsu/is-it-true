"""Data models for the fact-checking pipeline.

All Pydantic models used throughout the investigation: typed literals, source
representations, evidence tracking, the final FactCheckReport, and the token
usage tracking infrastructure (contextvar-based, zero-signature-change recording).

The module-level functions ``begin_token_tracking``, ``end_token_tracking``, and
``record_token_usage`` form a context-variable accumulator so sub-agents can
record litellm response usage without changing their return signatures.
"""

from __future__ import annotations

import contextvars
from typing import Literal

from pydantic import BaseModel, Field

# --- Typed literals used across the package ---

Verdict = Literal["true", "false", "mostly_true", "mostly_false", "misleading", "unverified"]


class Source(BaseModel):
    """A single web source discovered during a search round.

    ``credibility_score`` starts at 0.5 and is refined by the source evaluator.
    ``extracted_text`` is the full-page content (or snippet fallback).
    """

    url: str
    title: str = ""
    snippet: str = ""
    extracted_text: str = ""
    publish_date: str | None = None
    images: list[str] = Field(default_factory=list)
    image_descriptions: list[str] = Field(default_factory=list)
    credibility_score: float = 0.5


class Evidence(BaseModel):
    """A single piece of evidence extracted from a source.

    ``supports_claim`` tri-state: True = supports, False = contradicts, None = neutral.
    ``visual_findings`` is populated only when image analysis runs.
    """

    supports_claim: bool | None = None
    source: Source
    key_passages: list[str] = Field(default_factory=list)
    visual_findings: str | None = None


class Gap(BaseModel):
    """An unanswered question uncovered during investigation."""

    question: str
    reason: str


class InvestigationRound(BaseModel):
    """A complete round of the iterative investigation loop.

    Contains the queries planned, sources found, evidence extracted, gaps
    identified, and the token usage incurred during this round.
    """

    round_number: int
    search_queries: list[str]
    search_engine_used: str
    model_used: str
    sources_found: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    gaps_identified: list[Gap] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class ContradictionResolution(BaseModel):
    """Resolution of conflicting evidence between two sources."""

    evidence_a: Evidence
    evidence_b: Evidence
    resolution: str
    trusted_source: str
    reasoning: str


class ModelConfigDict(BaseModel):
    """Per-role model overrides.

    Each optional field maps a sub-agent role name to a litellm-compatible model
    identifier (e.g. ``"openai/gpt-4o"``). The ``default`` field is the catch-all.

    Resolution order (highest wins):
    1. Explicit per-role value in this dict
    2. ``IS_IT_TRUE_<ROLE>_MODEL`` env var
    3. ``IS_IT_TRUE_DEFAULT_MODEL`` env var
    """

    default: str | None = None
    query_planner: str | None = None
    evidence_extractor: str | None = None
    gap_detector: str | None = None
    contradiction_resolver: str | None = None
    source_evaluator: str | None = None
    verdict_judge: str | None = None
    image_analyzer: str | None = None
    language_detector: str | None = None


# ---------------------------------------------------------------------------
# Sub-agent response models — used with json-repair + Pydantic validation
# ---------------------------------------------------------------------------


# query_planner returns a JSON object wrapping the query list
class QueryPlannerResponse(BaseModel):
    queries: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A single evidence item within the evidence extractor's response."""

    supports_claim: bool | None = None
    key_passages: list[str] = Field(default_factory=list)
    summary: str = ""


class EvidenceExtractorResponse(BaseModel):
    """Expected JSON shape from the evidence extractor sub-agent."""

    evidence: list[EvidenceItem] = Field(default_factory=list)


class GapItem(BaseModel):
    """A single gap within the gap detector's response."""

    question: str = ""
    reason: str = ""


class ContradictionItem(BaseModel):
    """A contradiction pair within the gap detector's response."""

    source_a_index: int = 0
    source_b_index: int = 0
    description: str = ""


class GapDetectorResponse(BaseModel):
    """Expected JSON shape from the gap detector sub-agent."""

    gaps: list[GapItem] = Field(default_factory=list)
    contradictions: list[ContradictionItem] = Field(default_factory=list)
    assessment: str = ""


class VerdictJudgeResponse(BaseModel):
    """Expected JSON shape from the verdict judge sub-agent."""

    verdict: str = "unverified"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    summary: str = ""


class SourceEvaluatorResponse(BaseModel):
    """Expected JSON shape from the source evaluator sub-agent."""

    score: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class ContradictionResolverResponse(BaseModel):
    """Expected JSON shape from the contradiction resolver sub-agent."""

    resolution: str = ""
    trusted_source_index: int = 0
    reasoning: str = ""


class ImageAnalyzerResponse(BaseModel):
    """Expected JSON shape from the image analyzer sub-agent."""

    supports_claim: bool | None = None
    description: str = ""
    assessment: str = ""


class TokenUsage(BaseModel):
    """Token consumption for one or more LLM calls.

    ``input_tokens`` — prompt/completion input tokens.
    ``output_tokens`` — generated output tokens.
    ``cache_read_tokens`` — tokens served from provider-side cache (not billed
    or billed at a discount on Anthropic / OpenAI).
    ``cache_creation_tokens`` — tokens written into provider-side cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def merge(self, other: TokenUsage) -> TokenUsage:
        """Return a new TokenUsage summing ``self`` and ``other``."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )

    @staticmethod
    def from_response(usage: object) -> TokenUsage:
        """Extract token counts from a litellm ``response.usage`` object.

        Handles OpenAI-style ``prompt_tokens_details.cached_tokens`` and
        Anthropic-style ``cache_read_input_tokens`` / ``cache_creation_input_tokens``.
        """
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        cache_read = 0
        cache_create = 0

        # OpenAI-style: prompt_tokens_details.cached_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cache_read = int(getattr(details, "cached_tokens", 0) or 0)

        # Anthropic-style: raw attributes on the usage object itself
        cache_read = cache_read or int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )


# ---------------------------------------------------------------------------
# Context-var-based token accumulator
#
# Every sub-agent that calls litellm.acompletion() calls
# record_token_usage(response.usage) — a one-liner with no signature impact.
# The orchestrator (agent.py) wraps each investigation round with
# begin_token_tracking() / end_token_tracking() to isolate per-round accounting.
#
# Context vars are asyncio-safe: each task / coroutine chain inherits and
# can mutate its own copy without leaking across tasks.
# ---------------------------------------------------------------------------

_token_acc: contextvars.ContextVar[list[TokenUsage] | None] = contextvars.ContextVar(
    "_token_acc", default=None
)


def record_token_usage(usage: object) -> None:
    """Record a litellm ``response.usage`` if tracking is active.

    Safe to call when no accumulator is set — silently a no-op.
    """
    acc = _token_acc.get()
    if acc is not None:
        acc.append(TokenUsage.from_response(usage))


def begin_token_tracking() -> list[TokenUsage]:
    """Start a new token-tracking scope.

    Returns a fresh mutable list that subsequent ``record_token_usage()`` calls
    will append to.
    """
    acc: list[TokenUsage] = []
    _token_acc.set(acc)
    return acc


def end_token_tracking() -> list[TokenUsage]:
    """Close the current token-tracking scope and return collected usages.

    Resets the accumulator so later calls (e.g. the next round) start fresh.
    """
    acc = _token_acc.get()
    _token_acc.set(None)
    return acc or []


class FactCheckReport(BaseModel):
    """The complete result of a fact-checking investigation.

    Serialisable to JSON for CLI ``--json`` / ``--json-lines`` output modes.
    """

    claim: str
    language: str = "en"
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    investigation_rounds: list[InvestigationRound] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    contradictions_resolved: list[ContradictionResolution] = Field(default_factory=list)
    model_config_used: ModelConfigDict = Field(default_factory=ModelConfigDict)
    total_token_usage: TokenUsage | None = None
