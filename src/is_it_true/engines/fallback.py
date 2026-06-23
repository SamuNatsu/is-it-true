"""Fallback engine — chains multiple engines with automatic failover.

On construction, auto-discovers available engines (Tavily, Exa, DuckDuckGo)
in priority order. Each ``search()`` or ``get_contents()`` call iterates
through the chain until one engine succeeds.

Discovery order:
    1. Tavily with TAVILY_API_KEY (advanced depth, raw content)
    2. Tavily keyless (free tier, basic depth)
    3. Exa with EXA_API_KEY (deep-reasoning context, highlights)
    4. DuckDuckGo (always available, no credentials)
"""

from __future__ import annotations

import os

from .. import logging as log
from .base import SearchEngine, SearchResult


class FallbackEngine(SearchEngine):
    """A search engine that delegates to a chain of engines in priority order.

    On construction, discovers all available engines and builds a fallback
    chain automatically. Each call to ``search()`` or ``get_contents()``
    tries each engine in sequence until one succeeds.
    """

    def __init__(self):
        self._engines = _discover_engines()

        if not self._engines:
            raise RuntimeError(
                "No search engine could be initialized. "
                "Set TAVILY_API_KEY or EXA_API_KEY, "
                "or ensure DuckDuckGo is reachable."
            )

        self._active_name = self._engines[0][0]

    @property
    def name(self) -> str:
        return "auto"

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Try each engine in the chain; return results from the first that succeeds."""
        for label, engine in self._engines:
            try:
                results = await engine.search(query, num_results)
                self._active_name = label
                return results
            except Exception as exc:
                log.engine_log(f"[engine] {label} failed: {exc}, trying next...")
                continue
        return []

    async def get_contents(self, urls: list[str]) -> list[dict]:
        """Try each engine for content fetching; return first non-empty result."""
        for label, engine in self._engines:
            try:
                results = await engine.get_contents(urls)
                if results:
                    return results
            except Exception as exc:
                log.engine_log(f"[engine] {label} get_contents failed: {exc}")
                continue
        return []


def _discover_engines() -> list[tuple[str, SearchEngine]]:
    """Discover available engines in priority order and build the chain."""
    engines: list[tuple[str, SearchEngine]] = []

    from .tavily import TavilyEngine

    # 1. Tavily with API key (advanced depth, raw content included)
    if os.getenv("TAVILY_API_KEY"):
        try:
            engines.append(("tavily", TavilyEngine()))
            log.engine_log("[engine] discovered: Tavily (API key)")
        except Exception as exc:
            log.engine_log(f"[engine] Tavily (API key) unavailable: {exc}")

    # 2. Tavily keyless (free tier: 1 000 credits/month, basic depth)
    if not engines:
        try:
            keyless = TavilyEngine()
            engines.append(("tavily", keyless))
            log.engine_log("[engine] discovered: Tavily (keyless)")
        except Exception as exc:
            log.engine_log(f"[engine] Tavily (keyless) unavailable: {exc}")

    # 3. Exa (deep-reasoning context with highlights)
    if os.getenv("EXA_API_KEY"):
        try:
            from .exa import ExaEngine

            engines.append(("exa", ExaEngine()))
            log.engine_log("[engine] discovered: Exa")
        except Exception as exc:
            log.engine_log(f"[engine] Exa unavailable: {exc}")

    # 4. DuckDuckGo (always available, no credentials needed)
    try:
        from .duckduckgo import DuckDuckGoEngine

        engines.append(("duckduckgo", DuckDuckGoEngine()))
        log.engine_log("[engine] discovered: DuckDuckGo")
    except Exception as exc:
        log.engine_log(f"[engine] DuckDuckGo unavailable: {exc}")

    if engines:
        log.engine_log(f"[engine] fallback chain: {' > '.join(label for label, _ in engines)}")
    return engines
