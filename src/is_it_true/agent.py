"""Async orchestrator for the multi-round fact-checking investigation.

The core loop in ``investigate()``:

  1. Detect language.
  2. For each round (1..max_rounds):
     a. Plan search queries from remaining gaps.
     b. Execute searches in parallel across the active engine.
     c. Filter, deduplicate, and enrich results (fetch + evaluate).
     d. Extract evidence from sources.
     e. Optionally run image analysis on visual evidence.
     f. Detect gaps and contradictions in collected evidence.
     g. Resolve contradictions.
     h. Record the round in the investigation state.
  3. Deliver a final verdict synthesising all evidence.

Termination conditions:
  * No gaps remain.
  * Evidence is consistent across rounds (> round 1) with no contradictions.
  * Maximum rounds exhausted.
  * Dead-end: no new sources found in a round.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from . import logging as log
from .config import build_model_config, resolve_model
from .display import print_evidence_summary, print_gaps_summary, print_round_token_usage, spin
from .engines import get_search_engine
from .engines.base import SearchEngine
from .models import (
    ContradictionResolution,
    Evidence,
    FactCheckReport,
    InvestigationRound,
    ModelConfigDict,
    Source,
    TokenUsage,
    Verdict,
    begin_token_tracking,
    end_token_tracking,
)
from .operations import enrich_sources, filter_results, search_queries_parallel
from .subagents.contradiction_resolver import resolve_contradiction
from .subagents.evidence_extractor import extract_evidence
from .subagents.gap_detector import detect_gaps
from .subagents.language_detector import detect_language
from .subagents.query_planner import plan_queries
from .subagents.verdict_judge import deliver_verdict


@dataclass
class _InvestigationState:
    """Mutable state carried across investigation rounds.

    Accumulates sources, evidence, rounds, contradictions, seen URLs,
    cached queries, and raw token usage records for final aggregation.
    """

    claim: str
    language: str
    model_config: ModelConfigDict
    engine_name: str
    multimedia: bool
    multimedia_types: list[str]
    max_rounds: int
    depth: str
    all_sources: list[Source] = field(default_factory=list)
    all_evidence: list[Evidence] = field(default_factory=list)
    rounds: list[InvestigationRound] = field(default_factory=list)
    contradictions: list[ContradictionResolution] = field(default_factory=list)
    search_queries_cache: set[str] = field(default_factory=set)
    seen_urls: set[str] = field(default_factory=set)
    token_usages: list[TokenUsage] = field(default_factory=list)
    resolved_contradiction_pairs: set[frozenset[int]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def investigate(
    claim: str,
    *,
    search_engine: str = "auto",
    max_rounds: int = 3,
    depth: str = "thorough",
    multimedia: bool = False,
    multimedia_types: list[str] | None = None,
    model_config: ModelConfigDict | dict | None = None,
    log_mode: str = "console",
) -> FactCheckReport:
    """Run the full multi-round investigation and return a report.

    This is the async core — called by ``is_it_true()``.
    """
    # Configure progress output mode
    if log_mode == "json":
        log.set_mode(log.OutputMode.JSON_LINES)
    elif log_mode == "none":
        log.set_mode(log.OutputMode.NONE)

    # Clamp rounds to safe range
    max_rounds = max(1, min(max_rounds, 5))

    # Mirror the public API's default for multimedia_types
    if multimedia and multimedia_types is None:
        multimedia_types = ["image"]

    config = build_model_config(model_config)

    # Resolve search engine (auto-discover or explicit)
    engine = await get_search_engine(search_engine)
    engine_name = engine.name

    # Language detection is a one-off LLM call before the main loop
    begin_token_tracking()
    language = await detect_language(claim, config)
    state_token_usages: list[TokenUsage] = end_token_tracking()

    log.print(f"[bold]Investigating:[/] {claim[:120]}{'...' if len(claim) > 120 else ''}")
    log.print(
        f"  engine=[bold]{engine_name}[/]  rounds={max_rounds}  "
        f"depth={depth}  language=[bold]{language}[/]"
    )
    log.print()
    log.event(
        "investigation_start",
        claim=claim,
        engine=engine_name,
        rounds=max_rounds,
        depth=depth,
        language=language,
    )

    state = _InvestigationState(
        claim=claim,
        language=language,
        model_config=config,
        engine_name=engine_name,
        # multimedia is only active when explicitly requested AND image type
        # is in the list (the only currently supported type)
        multimedia=multimedia
        and multimedia_types is not None
        and "image" in (multimedia_types or []),
        multimedia_types=multimedia_types or [],
        max_rounds=max_rounds,
        depth=depth,
    )
    state.token_usages.extend(state_token_usages)

    # --- Investigation loop ---
    for round_num in range(1, state.max_rounds + 1):
        should_continue = await _execute_round(state, engine, round_num)
        log.event("round_end", round=round_num, continued=should_continue)
        if not should_continue:
            break
        log.print()

    return await _build_report(state)


# ---------------------------------------------------------------------------
# Per-round execution
# ---------------------------------------------------------------------------


async def _execute_round(
    state: _InvestigationState,
    engine: SearchEngine,
    round_num: int,
) -> bool:
    """Execute one investigation round.

    Returns ``True`` if the investigation should continue to another round,
    ``False`` if it should stop (gaps resolved, consistent evidence, dead end).
    """
    log.print(f"[bold cyan]--- Round {round_num}/{state.max_rounds} ---[/]")
    log.event("round_start", round=round_num, max=state.max_rounds)

    # Start token tracking for this round — all sub-agent calls within the
    # round will append their usage to the context-var accumulator.
    begin_token_tracking()

    # 1. Plan queries (from gaps if this is round 2+)
    new_queries = await _plan_round_queries(state)
    if not new_queries:
        round_usages = end_token_tracking()
        state.token_usages.extend(round_usages)
        round_usage = _merge_token_usages(round_usages)
        round_record = InvestigationRound(
            round_number=round_num,
            search_queries=[],
            search_engine_used=state.engine_name,
            model_used=_resolve_model_safe("query_planner", state.model_config),
            sources_found=[],
            gaps_identified=[],
            token_usage=round_usage,
        )
        state.rounds.append(round_record)
        print_round_token_usage(round_usage)
        return False

    for i, q in enumerate(new_queries):
        log.print(f"    [dim]\\[{i + 1}][/] {q}")

    # 2. Search in parallel across the fallback engine chain
    raw_results = await spin(
        search_queries_parallel(engine, new_queries, state.seen_urls), "searching..."
    )

    # 3. Filter and deduplicate results (by URL, domain cap, quality scoring)
    filtered_results = filter_results(raw_results)

    # 4. Enrich: fetch full content for sources that need it, then evaluate
    #    credibility (LLM scoring or domain heuristics)
    round_sources = await spin(
        enrich_sources(filtered_results, engine, state.model_config), "evaluating..."
    )
    for s in round_sources:
        state.all_sources.append(s)

    log.print(f"  searched: {len(round_sources)} new [dim]({len(state.all_sources)} total)[/]")
    log.event("sources_enriched", new=len(round_sources), total=len(state.all_sources))

    # Dead-end check — no usable sources found
    if not round_sources:
        log.print("  [dim yellow]no new sources found, dead end[/]")
        round_usages = end_token_tracking()
        round_usage = _merge_token_usages(round_usages)
        round_record = InvestigationRound(
            round_number=round_num,
            search_queries=new_queries,
            search_engine_used=state.engine_name,
            model_used=_resolve_model_safe("query_planner", state.model_config),
            sources_found=[],
            gaps_identified=[],
            token_usage=round_usage,
        )
        state.rounds.append(round_record)
        state.token_usages.extend(round_usages)
        print_round_token_usage(round_usage)
        return False

    # 5. Extract evidence from sources (batched LLM call)
    evidence = await spin(
        extract_evidence(state.claim, round_sources, state.model_config),
        "extracting evidence...",
    )
    print_evidence_summary(evidence)
    log.event(
        "evidence_extracted", count=len(evidence), total=len(state.all_evidence) + len(evidence)
    )

    # 6. Optional image analysis
    if state.multimedia:
        await _run_image_analysis(state.claim, evidence, state.model_config)

    for ev in evidence:
        state.all_evidence.append(ev)

    # 7. Gap detection + contradiction flagging
    round_gaps, contradictions_raw = await spin(
        detect_gaps(state.claim, state.all_evidence, state.model_config),
        "detecting gaps...",
    )
    print_gaps_summary(round_gaps, contradictions_raw)
    log.event("gaps_detected", gaps=len(round_gaps), contradictions=len(contradictions_raw))

    # 8. Resolve contradictions in parallel
    if contradictions_raw:
        await spin(_run_contradiction_resolution(state, contradictions_raw), "resolving...")
        log.event("contradictions_resolved", count=len(state.contradictions))

    # Finalise round record and collect token usage
    round_usages = end_token_tracking()
    round_usage = _merge_token_usages(round_usages)

    # Build a summary of all models active in this round
    roles = ("query_planner", "evidence_extractor", "gap_detector", "source_evaluator")
    models_used: dict[str, str] = {}
    for role in roles:
        models_used.setdefault(_resolve_model_safe(role, state.model_config), role)
    model_summary = ", ".join(f"{role}:{model}" for model, role in sorted(models_used.items()))

    round_record = InvestigationRound(
        round_number=round_num,
        search_queries=new_queries,
        search_engine_used=state.engine_name,
        model_used=model_summary,
        sources_found=round_sources,
        evidence=evidence,
        gaps_identified=round_gaps,
        token_usage=round_usage,
    )
    state.rounds.append(round_record)
    state.token_usages.extend(round_usages)

    print_round_token_usage(round_usage)

    # Termination: no gaps means investigation is complete
    if not round_gaps:
        log.print("  [green]all gaps resolved, evidence complete[/]")
        return False

    # Termination: consistent evidence across rounds with no contradictions
    if _evidence_is_consistent(state.all_evidence, contradictions_raw, round_num):
        log.print("  [green]consistent evidence across rounds, investigation complete[/]")
        return False

    return True


# ---------------------------------------------------------------------------
# Round helpers
# ---------------------------------------------------------------------------


async def _plan_round_queries(state: _InvestigationState) -> list[str]:
    """Generate new search queries for this round.

    On round 1 queries are derived from the claim alone. On later rounds
    they target the gaps identified in the previous round.
    """
    gaps = state.rounds[-1].gaps_identified if state.rounds else None
    queries = await spin(plan_queries(state.claim, gaps, state.model_config), "planning queries...")

    # Filter out queries already used in previous rounds
    new_queries = [q for q in queries if q not in state.search_queries_cache]
    log.print(
        f"  queries planned: [bold]{len(new_queries)} new[/], "
        f"[dim]{len(queries) - len(new_queries)} cached[/]"
    )
    log.event("queries_planned", new=len(new_queries), cached=len(queries) - len(new_queries))
    if not new_queries:
        log.print("  [dim yellow]no new queries, stopping[/]")
        return []
    for q in new_queries:
        state.search_queries_cache.add(q)
    return new_queries


async def _run_image_analysis(
    claim: str,
    evidence: list[Evidence],
    model_config: ModelConfigDict,
) -> None:
    """Analyse images attached to evidence sources in parallel.

    Only runs when the claim contains visual keywords AND multimedia is enabled.
    Each source's first 3 images are analysed; results are written into
    ``evidence[].visual_findings``.
    """
    from .multimedia.image_analyzer import analyze_image, claim_warrants_visual_analysis

    # Bail out early if the claim doesn't warrant visual analysis
    if not claim_warrants_visual_analysis(claim):
        return

    image_items = []
    for ev_idx, ev in enumerate(evidence):
        for img_idx, img_url in enumerate(ev.source.images[:3]):
            desc = (
                ev.source.image_descriptions[img_idx]
                if img_idx < len(ev.source.image_descriptions)
                else ""
            )
            image_items.append((ev_idx, img_url, desc))

    if not image_items:
        return

    total = len(image_items)
    log.print(f"  analyzing images: [bold]{total}[/]")

    async def _analyze_one(ev_idx: int, img_url: str, desc: str, idx: int) -> None:
        try:
            result = await analyze_image(img_url, claim, desc, model_config)
            if result:
                finding = (
                    f"[{result.get('supports_claim')}] "
                    f"{result.get('description', '')[:200]} | "
                    f"{result.get('assessment', '')[:200]}"
                )
                existing = evidence[ev_idx].visual_findings or ""
                evidence[ev_idx].visual_findings = f"{existing}\n{finding}" if existing else finding
                log.print(f"    [dim]\\[{idx}/{total}][/] image analyzed: {img_url[:60]}")
        except Exception:
            log.print(
                f"    [dim]\\[{idx}/{total}][/] [yellow]image analysis failed[/]: {img_url[:60]}"
            )

    await asyncio.gather(
        *[
            _analyze_one(ev_idx, url, desc, i + 1)
            for i, (ev_idx, url, desc) in enumerate(image_items)
        ]
    )


async def _run_contradiction_resolution(
    state: _InvestigationState,
    contradictions_raw: list[tuple[int, int, str]],
) -> None:
    """Resolve contradictions in parallel.

    Each contradiction is a tuple of (evidence_index_a, evidence_index_b, description).
    Resolved ``ContradictionResolution`` objects are appended to ``state.contradictions``.
    """
    items = []
    for ci in contradictions_raw:
        idx_a, idx_b, desc = ci
        if 0 <= idx_a < len(state.all_evidence) and 0 <= idx_b < len(state.all_evidence):
            # Skip pairs already resolved in a previous round
            pair = frozenset({idx_a, idx_b})
            if pair in state.resolved_contradiction_pairs:
                continue
            state.resolved_contradiction_pairs.add(pair)
            items.append((state.all_evidence[idx_a], state.all_evidence[idx_b], desc))

    if not items:
        return

    total = len(items)
    log.print(f"  resolving: [bold]{total} contradictions[/]")

    async def _resolve_one(
        ev_a: Evidence,
        ev_b: Evidence,
        desc: str,
        idx: int,
    ) -> None:
        try:
            resolution = await resolve_contradiction(state.claim, ev_a, ev_b, state.model_config)
            state.contradictions.append(resolution)
            log.print(f"    [dim]\\[{idx}/{total}][/] resolved: {desc[:60]}")
        except Exception as exc:
            log.print(f"    [dim]\\[{idx}/{total}][/] [red]FAILED[/]: {desc[:60]} — {exc}")

    await asyncio.gather(
        *[_resolve_one(ev_a, ev_b, desc, i + 1) for i, (ev_a, ev_b, desc) in enumerate(items)]
    )


# ---------------------------------------------------------------------------
# Termination helpers
# ---------------------------------------------------------------------------


def _evidence_is_consistent(
    evidence_items: list[Evidence],
    contradictions_raw: list[tuple[int, int, str]],
    round_num: int,
) -> bool:
    """Determine whether all non-neutral evidence points in the same direction.

    Returns ``True`` when:
        * No raw contradictions remain to be resolved.
        * At least round 2 (round 1 evidence is always insufficient).
        * At least one piece of non-neutral evidence exists.
        * All non-neutral evidence has the same ``supports_claim`` direction.
    """
    if contradictions_raw:
        return False
    if round_num <= 1:
        return False
    non_neutral = [ev for ev in evidence_items if ev.supports_claim is not None]
    if not non_neutral:
        return False
    first_direction = non_neutral[0].supports_claim
    return all(ev.supports_claim == first_direction for ev in non_neutral)


def _merge_token_usages(usages: list[TokenUsage]) -> TokenUsage:
    """Aggregate a list of TokenUsage records into a single summary."""
    result = TokenUsage()
    for u in usages:
        result = result.merge(u)
    return result


def _resolve_model_safe(role: str, config: ModelConfigDict) -> str:
    """Resolve a role's model, falling back to ``"unknown"``.

    Graceful degradation so the round record always has a model name even
    if config resolution fails.
    """
    try:
        return resolve_model(role, config)
    except Exception:
        return str(config.default or "unknown")


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


async def _build_report(state: _InvestigationState) -> FactCheckReport:
    """Synthesise the final verdict and assemble the FactCheckReport.

    Makes one final LLM call (verdict_judge) and aggregates all token usage
    collected across language detection, investigation rounds, and the verdict.
    """
    log.print()
    log.print("[bold cyan]--- Verdict ---[/]")
    begin_token_tracking()
    verdict_data = await spin(
        deliver_verdict(
            state.claim,
            state.all_evidence,
            state.contradictions,
            state.language,
            state.model_config,
        ),
        "delivering verdict...",
    )
    verdict_usages = end_token_tracking()

    verdict: Verdict = verdict_data.get("verdict", "unverified")
    confidence = float(verdict_data.get("confidence", 0.3))
    # Deduplicate while preserving insertion order
    all_refs = list(dict.fromkeys(s.url for s in state.all_sources if s.url))

    # Total token usage = language detection + all rounds + verdict
    all_usages = list(state.token_usages) + verdict_usages
    total_usage = _merge_token_usages(all_usages)

    report = FactCheckReport(
        claim=state.claim,
        language=state.language,
        verdict=verdict,
        confidence=confidence,
        summary=verdict_data.get("summary", ""),
        investigation_rounds=state.rounds,
        references=all_refs,
        contradictions_resolved=state.contradictions,
        model_config_used=state.model_config,
        total_token_usage=total_usage,
    )
    log.event("verdict", verdict=verdict, confidence=confidence)
    return report
