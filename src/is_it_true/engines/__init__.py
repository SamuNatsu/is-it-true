from __future__ import annotations

from .base import SearchEngine
from .fallback import FallbackEngine


async def get_search_engine(preferred: str = "auto") -> SearchEngine:
    if preferred == "tavily":
        from .tavily import TavilyEngine

        return TavilyEngine()

    if preferred == "exa":
        from .exa import ExaEngine

        return ExaEngine()

    if preferred == "duckduckgo":
        from .duckduckgo import DuckDuckGoEngine

        return DuckDuckGoEngine()

    return FallbackEngine()
