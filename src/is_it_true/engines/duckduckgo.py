"""DuckDuckGo search engine (always-available fallback).

Uses the ``duckduckgo-search`` library for queries and fetches full page
content via httpx + trafilatura extraction.

Trafilatura works on server-rendered HTML only — CSR/SPA pages return
empty content. The minimum acceptable extraction length is 200 characters;
shorter output is discarded in favour of the search snippet.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from .base import SearchEngine, SearchResult

# Minimum characters required from trafilatura extraction —
# below this threshold the snippet is preferred.
_MIN_CONTENT_LENGTH = 200


async def _extract_content(html: str) -> str:
    """Extract readable text from HTML using trafilatura.

    Runs in a thread pool since trafilatura is CPU-bound.
    Returns empty string on failure or if extracted content is too short.
    """
    try:
        import trafilatura

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=False,
                with_metadata=False,
            ),
        )
        if result and len(result.strip()) >= _MIN_CONTENT_LENGTH:
            return result.strip()
    except Exception:
        pass
    return ""


class DuckDuckGoEngine(SearchEngine):
    """DuckDuckGo-backed engine — no API key required.

    Search returns snippets only (no raw_content). Full content is fetched
    lazily by ``get_contents()`` which fetches each URL and runs trafilatura.
    """

    DEFAULT_MAX_RESULTS: ClassVar[int] = 5

    @property
    def name(self) -> str:
        return "duckduckgo"

    async def search(
        self, query: str, num_results: int = DEFAULT_MAX_RESULTS
    ) -> list[SearchResult]:
        """Search DuckDuckGo and return snippet-only results."""
        try:
            from duckduckgo_search import DDGS

            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: list(
                    DDGS().text(
                        query,
                        max_results=min(num_results, 20),
                    )
                ),
            )
            return [
                SearchResult(
                    url=r.get("href", ""),
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                )
                for r in raw
                if r.get("href")
            ]
        except Exception:
            return []

    async def get_contents(self, urls: list[str]) -> list[dict]:
        """Fetch and extract page content for each URL in parallel.

        Uses httpx with a User-Agent header to avoid bot-blocking.
        """
        import httpx

        results: list[dict] = []

        async def fetch_one(client: httpx.AsyncClient, url: str) -> dict | None:
            try:
                response = await client.get(url, follow_redirects=True, timeout=15.0)
                response.raise_for_status()
                text = await _extract_content(response.text)
                return {"url": url, "raw_content": text} if text else None
            except Exception:
                return None

        async with httpx.AsyncClient(
            headers={"User-Agent": "is-it-true/0.1.0"},
            timeout=httpx.Timeout(30.0),
        ) as client:
            tasks = [fetch_one(client, url) for url in urls]
            gathered = await asyncio.gather(*tasks)

        for r in gathered:
            if r:
                results.append(r)
        return results
