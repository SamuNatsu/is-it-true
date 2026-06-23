"""Exa search engine (secondary — deep-reasoning context with highlights).

Requires ``EXA_API_KEY``. Returns full page text and highlighted passages
in search results. Exa client is synchronous, wrapped in
``loop.run_in_executor``.
"""

from __future__ import annotations

import asyncio
import os
from typing import ClassVar

from .base import SearchEngine, SearchResult


class ExaEngine(SearchEngine):
    DEFAULT_MAX_RESULTS: ClassVar[int] = 5

    def __init__(self, api_key: str | None = None, search_type: str = "auto"):
        self._api_key = api_key or os.getenv("EXA_API_KEY")
        self._search_type = search_type

    @property
    def name(self) -> str:
        return "exa"

    def _get_client(self):
        """Create an Exa client — raises if no API key is configured."""
        from exa_py import Exa

        if not self._api_key:
            raise RuntimeError("EXA_API_KEY is required for Exa search engine")
        return Exa(api_key=self._api_key)

    async def search(
        self, query: str, num_results: int = DEFAULT_MAX_RESULTS
    ) -> list[SearchResult]:
        """Search with full text and highlights via Exa API."""
        client = self._get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query,
                type=self._search_type,
                num_results=min(num_results, 20),
                # Request full text and highlighted passages
                contents={"text": True, "highlights": True},
            ),
        )
        return self._parse_response(response)

    def _parse_response(self, response) -> list[SearchResult]:
        """Convert Exa response objects to SearchResult objects.

        Highlights (the most relevant passages) become the snippet;
        full text becomes raw_content.
        """
        results = []
        for r in getattr(response, "results", []):
            text = getattr(r, "text", "") or ""
            highlights = getattr(r, "highlights", []) or []
            images_list = getattr(r, "image", None)
            if isinstance(images_list, str):
                images = [images_list]
            elif isinstance(images_list, list):
                images = [
                    img.get("url", "") if isinstance(img, dict) else str(img)
                    for img in images_list
                    if img
                ]
            else:
                images = []
            results.append(
                SearchResult(
                    url=getattr(r, "url", ""),
                    title=getattr(r, "title", ""),
                    snippet=" ".join(highlights) if highlights else "",
                    raw_content=text,
                    publish_date=getattr(r, "published_date", None),
                    images=images,
                )
            )
        return results

    async def get_contents(self, urls: list[str]) -> list[dict]:
        """Fetch full page text for a list of URLs via Exa contents API."""
        client = self._get_client()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.get_contents(urls, text=True),
        )
        results = []
        for r in getattr(response, "results", []) or []:
            results.append(
                {
                    "url": getattr(r, "url", ""),
                    "raw_content": getattr(r, "text", "") or "",
                }
            )
        return results
