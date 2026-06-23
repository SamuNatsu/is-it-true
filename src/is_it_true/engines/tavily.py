"""Tavily search engine (primary — richest results).

Supports both API-key and keyless modes. In keyless mode, falls back to
``search_depth="basic"`` on failure. All search calls include raw content
and image metadata for downstream evidence extraction and image analysis.

The Tavily client is synchronous, so calls are wrapped in
``loop.run_in_executor`` to avoid blocking the async event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import ClassVar

from .base import SearchEngine, SearchResult


class TavilyEngine(SearchEngine):
    DEFAULT_MAX_RESULTS: ClassVar[int] = 5

    def __init__(self, api_key: str | None = None, search_depth: str = "advanced"):
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._search_depth = search_depth

    @property
    def name(self) -> str:
        return "tavily"

    def _get_client(self):
        """Create a TavilyClient — with API key if available, else keyless."""
        from tavily import TavilyClient

        if self._api_key:
            return TavilyClient(api_key=self._api_key)
        return TavilyClient()

    async def search(
        self, query: str, num_results: int = DEFAULT_MAX_RESULTS
    ) -> list[SearchResult]:
        """Search with advanced depth. Falls back to basic on keyless failure."""
        try:
            client = self._get_client()
            loop = asyncio.get_running_loop()
            # Tavily client is sync, run in thread pool
            response = await loop.run_in_executor(
                None,
                lambda: client.search(
                    query,
                    search_depth=self._search_depth,
                    max_results=min(num_results, 20),
                    include_raw_content=True,
                    include_images=True,
                    include_image_descriptions=True,
                ),
            )
            return self._parse_response(response)
        except Exception:
            # Keyless mode: retry with basic depth (lower cost / rate-limit)
            if not self._api_key:
                return await self._search_basic(query, num_results)
            raise

    async def _search_basic(self, query: str, num_results: int) -> list[SearchResult]:
        """Keyless fallback: search with basic depth only."""
        from tavily import TavilyClient

        client = TavilyClient()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query,
                search_depth="basic",
                max_results=min(num_results, 20),
                include_raw_content=True,
                include_images=True,
                include_image_descriptions=True,
            ),
        )
        return self._parse_response(response)

    def _parse_response(self, response: dict) -> list[SearchResult]:
        """Convert Tavily JSON response to SearchResult objects.

        Handles images as either a list of dicts or a JSON-encoded string.
        """
        results = []
        for r in response.get("results", []):
            images_list = r.get("images", []) or []
            # Tavily sometimes returns images as a JSON string
            if isinstance(images_list, str):
                try:
                    images_list = json.loads(images_list)
                except (json.JSONDecodeError, TypeError):
                    images_list = []
            images = [
                img.get("url", "") if isinstance(img, dict) else str(img)
                for img in images_list
                if img
            ]
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    raw_content=r.get("raw_content", ""),
                    publish_date=r.get("published_date"),
                    images=images,
                    image_descriptions=r.get("image_descriptions", []) or [],
                )
            )
        return results

    async def get_contents(self, urls: list[str]) -> list[dict]:
        """Extract raw content from a list of URLs via Tavily Extract."""
        client = self._get_client()
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.extract(urls=urls, include_images=True),
            )
        except Exception:
            # Keyless unsupported for extract — silently return empty
            if self._api_key:
                raise
            return []
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "url": r.get("url", ""),
                    "raw_content": r.get("raw_content", ""),
                }
            )
        return results
