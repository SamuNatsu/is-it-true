"""CLI entry point for ``is-it-true`` console script.

Registered in pyproject.toml as ``is-it-true = "is_it_true.cli:main"``.

Two orthogonal options control output:

* ``--format`` — final report format: ``console``, ``json``, ``html``, or
  ``pdf`` (default: ``console``).
* ``--log`` — progress output during investigation: ``console``, ``json``
  (JSON lines), or ``none`` (default: ``console``).

The claim can be provided as a positional argument or piped via stdin.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from . import is_it_true
from . import logging as log
from .display import format_token_usage
from .models import FactCheckReport

_FORMAT_CHOICES = ("console", "json", "html", "pdf")
_LOG_CHOICES = ("console", "json", "none")


def _confidence_bar(confidence: float) -> str:
    """Render a 10-segment Rich confidence bar (0%–100%)."""
    filled = int(confidence * 10)
    bar = "█" * filled + "░" * (10 - filled)
    color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.4 else "red"
    return f"[{color}][{bar}][/] [bold]{confidence:.0%}[/]"


def _output_report(report: FactCheckReport, fmt: str, output_path: str | None) -> None:
    """Write the final report in the requested format."""
    if fmt == "json":
        _write_or_print(report.model_dump_json(), output_path)

    elif fmt == "html":
        from .formatters.html import render_html

        html_str = render_html(report)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    elif fmt == "pdf":
        from .formatters.pdf import render_pdf

        pdf_bytes = render_pdf(report)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    else:
        _print_console_report(report)


def _write_or_print(content: str, output_path: str | None) -> None:
    """Write content to output_path or print to stdout."""
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        sys.stdout.write(content)


def _print_console_report(report: FactCheckReport) -> None:
    """Render a rich console summary of the report."""
    from .utils import verdict_style

    verdict_text = report.verdict.replace("_", " ").title()
    style = verdict_style(report.verdict)

    log.get_console().print()
    log.get_console().print(f"  [bold]Claim:[/] {report.claim}")
    log.get_console().print()
    log.get_console().print(f"  [bold]Verdict:[/]    [{style}]{verdict_text}[/]")
    log.get_console().print(f"  [bold]Confidence:[/] {_confidence_bar(report.confidence)}")
    log.get_console().print()
    log.get_console().print(report.summary)
    log.get_console().print()

    if report.references:
        from rich.rule import Rule

        log.get_console().print(Rule("[bold]References[/]"))
        for i, ref in enumerate(report.references, 1):
            log.get_console().print(f"  [dim]\\[{i}][/] {ref}")
        log.get_console().print()

    if report.investigation_rounds:
        total_sources = sum(len(r.sources_found) for r in report.investigation_rounds)
        total_evidence = sum(len(r.evidence) for r in report.investigation_rounds)
        log.get_console().print(
            "[dim]"
            f"  Investigation: [bold]{len(report.investigation_rounds)} round(s)[/], "
            f"[bold]{total_sources} source(s)[/] searched, "
            f"[bold]{total_evidence} evidence item(s)[/] extracted"
            "[/]"
        )
        log.get_console().print()

    if report.total_token_usage:
        log.get_console().print(f"  [dim]Tokens: {format_token_usage(report.total_token_usage)}[/]")
        log.get_console().print()


def main() -> None:
    """Parse args, wire up output mode, and run the investigation.

    Exit codes: 0 on success, 1 on error (missing claim, exception).
    """
    parser = argparse.ArgumentParser(
        prog="is-it-true",
        description="AI-powered fact-checking from the terminal. "
        "Ask any claim and get a verdict with chain of evidence.",
    )
    parser.add_argument(
        "claim",
        nargs="?",
        help="The claim to fact-check. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--engine",
        "-e",
        choices=["auto", "tavily", "exa", "duckduckgo"],
        default="auto",
        help="Search engine to use (default: auto)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        choices=range(1, 6),
        default=3,
        metavar="1-5",
        help="Maximum investigation rounds (default: 3)",
    )
    parser.add_argument(
        "--depth",
        "-d",
        choices=["fast", "thorough"],
        default="thorough",
        help="Investigation depth (default: thorough)",
    )
    parser.add_argument(
        "--multimedia",
        action="store_true",
        help="Enable image analysis for visual claims",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=_FORMAT_CHOICES,
        default="console",
        help="Final report format (default: console)",
    )
    parser.add_argument(
        "--log",
        choices=_LOG_CHOICES,
        default="console",
        help="Progress log mode (default: console)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="is-it-true 0.1.0",
    )

    args = parser.parse_args()

    # Default output filename for HTML/PDF when not specified
    if args.format in ("html", "pdf") and not args.output:
        args.output = f"report.{args.format}"

    # Claim comes from positional arg or stdin
    claim = args.claim
    if claim is None:
        if sys.stdin.isatty():
            parser.print_help()
            print("\n  Provide a claim as an argument or pipe one via stdin.")
            sys.exit(1)
        claim = sys.stdin.read().strip()
        if not claim:
            print("Error: no claim provided.", file=sys.stderr)
            sys.exit(1)

    try:
        report = asyncio.run(
            is_it_true(
                claim,
                search_engine=args.engine,
                max_rounds=args.rounds,
                depth=args.depth,
                multimedia=args.multimedia,
                log_mode=args.log,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _output_report(report, args.format, args.output)
