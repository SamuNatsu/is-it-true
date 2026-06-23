"""Rich-console display helpers.

Spinners, colour codes, evidence/contradiction summaries, and token usage
formatting. All output goes through the ``logging`` module so it respects
the active OutputMode.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeVar

from rich.live import Live
from rich.spinner import Spinner

from . import logging as log
from .models import Evidence, Gap, TokenUsage

_T = TypeVar("_T")


async def spin(coro: Awaitable[_T], message: str) -> _T:
    """Show a rich spinner while awaiting *coro*, then return its result.

    No-op (awaits directly) in JSON / JSON_LINES output modes.
    """
    if log.get_mode() is not log.OutputMode.CONSOLE:
        return await coro
    spinner = Spinner("dots", text=message)
    with Live(spinner, refresh_per_second=10, transient=True, console=log.get_console()):
        return await coro


def get_score_color(score: float) -> str:
    """Map a 0–1 credibility score to a Rich colour name."""
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "yellow"
    return "red"


def format_evidence_icon(supports: bool | None) -> str:
    """Rich-markup icon for tri-state evidence direction."""
    if supports is True:
        return "[green]+[/]"
    if supports is False:
        return "[red]-[/]"
    return "[yellow]~[/]"


def print_evidence_summary(evidence: list[Evidence]) -> None:
    """Log a compact per-source summary with direction icons."""
    for i, ev in enumerate(evidence):
        icon = format_evidence_icon(ev.supports_claim)
        log.print(f"    [dim]\\[{i + 1}/{len(evidence)}][/] {icon} {ev.source.title[:60]}")
    log.print(f"  evidence: [bold]{len(evidence)} items[/] extracted")


def print_gaps_summary(gaps: list[Gap], contradictions_raw: list[tuple[int, int, str]]) -> None:
    """Log gaps and raw contradictions found in a round."""
    if gaps:
        log.print(f"  gaps: [bold yellow]{len(gaps)} found[/]")
        for i, g in enumerate(gaps, 1):
            log.print(f"    [dim]\\[{i}][/] [yellow]?[/] {g.question}")
    else:
        log.print("  gaps: [dim]none[/]")

    if contradictions_raw:
        log.print(f"  contradictions: [bold red]{len(contradictions_raw)} found[/]")
        for i, c in enumerate(contradictions_raw, 1):
            _idx_a, _idx_b, desc = c
            log.print(f"    [dim]\\[{i}][/] [red]![/] {desc[:120]}")
    else:
        log.print("  contradictions: [dim]none[/]")


def format_token_usage(usage: TokenUsage) -> str:
    """Format a TokenUsage as a Rich-markup string for display.

    Emits separate counts for input, output, cache-read, and cache-write tokens,
    plus a parenthesised total (input + output only, since cache tokens are
    typically a subset of input).
    """
    parts = [f"[bold]{usage.input_tokens:,}[/] in"]
    if usage.output_tokens:
        parts.append(f"[bold]{usage.output_tokens:,}[/] out")
    if usage.cache_read_tokens:
        parts.append(f"[bold]{usage.cache_read_tokens:,}[/] cache read")
    if usage.cache_creation_tokens:
        parts.append(f"[bold]{usage.cache_creation_tokens:,}[/] cache write")
    total = usage.input_tokens + usage.output_tokens
    if total:
        parts.append(f"([bold]{total:,}[/] total)")
    return " | ".join(parts)


def print_round_token_usage(usage: TokenUsage) -> None:
    """Log token usage for a single investigation round."""
    log.print(f"  tokens: {format_token_usage(usage)}")
