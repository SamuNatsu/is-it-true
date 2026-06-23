"""Search execution and result enrichment pipeline.

The three main functions form the search phase of an investigation round:

1. ``search_queries_parallel`` — dispatch queries concurrently.
2. ``filter_results`` — deduplicate, quality-score, and cap results.
3. ``enrich_sources`` — fetch full page content and evaluate source credibility.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from . import logging as log
from .display import get_score_color
from .engines.base import SearchEngine, SearchResult
from .models import ModelConfigDict, Source
from .subagents.source_evaluator import evaluate_source
from .utils import domain_credibility_score


async def search_queries_parallel(
    engine: SearchEngine,
    queries: list[str],
    seen_urls: set[str],
) -> list[SearchResult]:
    """Execute up to 5 queries in parallel against the search engine.

    Deduplicates against *seen_urls* (mutated in place) so later rounds
    don't re-fetch the same results.
    """
    # Cap at 5 queries to avoid overwhelming the engine
    active_queries = queries[:5]
    total = len(active_queries)
    log.print(f"  searching: [bold]{total} queries[/]")

    results: list[SearchResult] = []

    async def _search_one(query: str, idx: int) -> None:
        try:
            found = await engine.search(query, num_results=5)
            eng_name = engine.name
            new = [r for r in found if r.url and r.url not in seen_urls]
            for r in new:
                seen_urls.add(r.url)
            results.extend(new)
            log.print(
                f"    [dim]\\[{idx}/{total}][/] "
                f"[bold]{len(new)} results[/]: "
                f"{query[:70]} [dim]({eng_name})[/]"
            )
            log.event(
                "search_result",
                idx=idx,
                total=total,
                results=len(new),
                query=query,
                engine=eng_name,
            )
        except Exception as exc:
            log.print(f"    [dim]\\[{idx}/{total}][/] [red]FAILED[/]: {query[:70]} — {exc}")

    await asyncio.gather(*[_search_one(q, i + 1) for i, q in enumerate(active_queries)])
    return results


async def enrich_sources(
    raw_results: list[SearchResult],
    engine: SearchEngine,
    model_config: ModelConfigDict,
) -> list[Source]:
    """Fetch full page content and evaluate source credibility.

    For results that lack raw_content (e.g. DuckDuckGo snippet-only),
    fetches the page via the engine's ``get_contents``.

    Then evaluates each source with the LLM-based source evaluator
    (or falls back to domain heuristics on error).
    """
    # 1. Fetch full content for results that only have snippets
    needs_fetch = [r for r in raw_results if not r.raw_content]
    fetch_map: dict[str, str] = {}
    if needs_fetch:
        log.print(f"  fetching: [bold]{len(needs_fetch)} pages[/]")
        fetch_coros = [
            _fetch_one(r, engine, i + 1, len(needs_fetch), fetch_map)
            for i, r in enumerate(needs_fetch)
        ]
        if fetch_coros:
            await asyncio.gather(*fetch_coros)

    # 2. Evaluate credibility: LLM-based if a model is configured,
    #    otherwise fast domain heuristic.
    should_evaluate = bool(model_config.source_evaluator or model_config.default)
    scores: dict[str, float] = {}

    if should_evaluate:
        log.print(f"  evaluating: [bold]{len(raw_results)} sources[/]")
        eval_coros = [
            _evaluate_one(r, fetch_map, i + 1, len(raw_results), scores, model_config)
            for i, r in enumerate(raw_results)
        ]
        await asyncio.gather(*eval_coros)
    else:
        for r in raw_results:
            scores[r.url] = domain_credibility_score(r.url)

    # 3. Build Source objects, preferring fetched content over snippets
    sources: list[Source] = []
    for r in raw_results:
        content = r.raw_content or fetch_map.get(r.url, "") or r.snippet
        sources.append(
            Source(
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                extracted_text=content,
                publish_date=r.publish_date,
                images=r.images,
                image_descriptions=r.image_descriptions,
                credibility_score=scores.get(r.url, 0.5),
            )
        )
    return sources


async def _fetch_one(
    result: SearchResult,
    engine: SearchEngine,
    idx: int,
    total: int,
    fetch_map: dict[str, str],
) -> None:
    """Fetch raw content for a single result and store in *fetch_map*."""
    try:
        fetched = await engine.get_contents([result.url])
        if fetched:
            fetch_map[result.url] = fetched[0].get("raw_content", "")
            log.print(f"    [dim]\\[{idx}/{total}][/] fetched: {result.url[:80]}")
    except Exception:
        log.print(f"    [dim]\\[{idx}/{total}][/] [yellow]fetch failed[/]: {result.url[:80]}")


async def _evaluate_one(
    result: SearchResult,
    fetch_map: dict[str, str],
    idx: int,
    total: int,
    scores: dict[str, float],
    model_config: ModelConfigDict,
) -> None:
    """Evaluate credibility for a single source.

    Sends the first 3000 characters of content to the source evaluator LLM.
    On failure defaults to 0.5.
    """
    try:
        content = result.raw_content or fetch_map.get(result.url, "") or result.snippet
        score, _ = await evaluate_source(
            result.title, result.url, content[:3000], model_config=model_config
        )
        scores[result.url] = score
        color = get_score_color(score)
        log.print(
            f"    [dim]\\[{idx}/{total}][/] evaluated: "
            f"{result.title[:60]} ([{color}]{score:.2f}[/])"
        )
    except Exception:
        scores[result.url] = 0.5


def filter_results(
    results: list[SearchResult],
    max_results: int = 12,
    max_per_domain: int = 3,
) -> list[SearchResult]:
    """Deduplicate and quality-filter search results.

    Rules (applied in order):
        1. Remove duplicate URLs.
        2. If still over *max_results*, quality-score each result:
           - +3 for raw_content present (full page fetched).
           - +1 for snippet > 200 chars.
           - +2 for .gov/.edu/.mil domains.
           - +1 for known authoritative news/scientific domains.
           - −2 for social media / low-quality sources.
        3. Cap at *max_per_domain* entries per domain.
    """
    # Always deduplicate by URL
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)

    results = unique
    if len(results) <= max_results:
        return results

    # Quality scoring function
    domain_counts: dict[str, int] = {}

    def _score(r: SearchResult) -> int:
        score = 1
        if r.raw_content:
            score += 3
        elif len(r.snippet) > 200:
            score += 1
        parsed = urlparse(r.url)
        domain = parsed.netloc.lower().removeprefix("www.")
        # Authoritative domains
        if any(d in domain for d in (".gov", ".edu", ".mil")):
            score += 2
        elif any(
            d in domain
            for d in (
                "reuters.com",
                "apnews.com",
                "bbc.com",
                "npr.org",
                "nature.com",
                "science.org",
                "wikipedia.org",
                "britannica.com",
            )
        ):
            score += 1
        # Low-quality / social domains
        if any(
            d in domain
            for d in (
                "reddit.com",
                "quora.com",
                "pinterest.com",
                "instagram.com",
                "facebook.com",
                "twitter.com",
                "x.com",
                "tiktok.com",
                "youtube.com",
                "medium.com",
            )
        ):
            score -= 2
        return score

    # Sort by descending quality score, then apply per-domain cap
    scored = sorted(results, key=_score, reverse=True)

    filtered: list[SearchResult] = []
    for r in scored:
        domain = urlparse(r.url).netloc.lower().removeprefix("www.")
        count = domain_counts.get(domain, 0)
        if count >= max_per_domain:
            continue
        domain_counts[domain] = count + 1
        filtered.append(r)
        if len(filtered) >= max_results:
            break

    dropped = len(results) - len(filtered)
    if dropped:
        log.print(f"  filter: [dim]kept {len(filtered)}, dropped {dropped}[/]")

    return filtered
