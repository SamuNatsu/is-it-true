"""is-it-true — AI-powered fact-checking with multi-round web investigation.

Usage::

    from is_it_true import is_it_true
    report = await is_it_true("The Eiffel Tower grows 15 cm in summer")
    print(report.verdict, report.confidence)
"""

from __future__ import annotations

from .agent import investigate
from .config import build_model_config
from .models import FactCheckReport, ModelConfigDict


async def is_it_true(
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
    """Investigate a claim and return a detailed fact-check report.

    Args:
        claim: The claim to fact-check.
        search_engine: ``"auto"``, ``"tavily"``, ``"exa"``, or ``"duckduckgo"``.
        max_rounds: Maximum investigation rounds (clamped 1–5, default 3).
        depth: ``"fast"`` or ``"thorough"`` — controls search depth.
        multimedia: Enable image analysis when the claim mentions visual content.
        multimedia_types: Which multimedia types to analyse (default: ``["image"]``).
        model_config: Per-role model overrides (see ``ModelConfigDict``).
        log_mode: ``"console"``, ``"json"``, or ``"none"`` — controls progress output.

    Returns:
        ``FactCheckReport`` with verdict, confidence, evidence chain, and references.
    """
    if multimedia_types is None:
        multimedia_types = ["image"]

    config = build_model_config(model_config)

    return await investigate(
        claim=claim,
        search_engine=search_engine,
        max_rounds=max_rounds,
        depth=depth,
        multimedia=multimedia,
        multimedia_types=multimedia_types,
        model_config=config,
        log_mode=log_mode,
    )
