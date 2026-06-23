"""Abstract search engine interface.

Every engine must implement ``search(query, num_results)`` and
``get_contents(urls)``. The FallbackEngine chains multiple engines so
callers never need to handle individual engine failures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single result from a web search.

    ``raw_content`` may be empty if the engine returns snippets only —
    ``enrich_sources()`` in operations.py fetches full content lazily.
    """

    url: str
    title: str
    snippet: str
    raw_content: str = ""
    publish_date: str | None = None
    images: list[str] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)


class SearchEngine(ABC):
    """Minimal async interface for a web search backend.

    ``name`` is used for logging and round records.
    """

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute a web search and return structured results."""

    @abstractmethod
    async def get_contents(self, urls: list[str]) -> list[dict]:
        """Fetch raw content for the given URLs.

        Returns a list of dicts with ``url`` and ``raw_content`` keys.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine identifier (e.g. ``"tavily"``)."""
