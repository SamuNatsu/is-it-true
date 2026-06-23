"""Output-mode-aware logging layer.

Supports three modes:

* ``CONSOLE`` — rich console output (styled, coloured, spinners).
* ``JSON_LINES`` — write structured ``{"event": "...", ...}`` JSON lines for
  RPC / external consumers, then the final report as JSON.
* ``NONE`` — suppress all progress output; the caller handles the final report.

All progress output goes through this module so callers don't need to know the
active mode — ``print()`` and ``event()`` are safe no-ops in non-console modes.
"""

from __future__ import annotations

import json
import sys
from enum import Enum

from rich.console import Console


class OutputMode(Enum):
    CONSOLE = "console"
    NONE = "none"
    JSON_LINES = "json_lines"


# Module-level state — mutated once at startup by cli.py:main()
_mode: OutputMode = OutputMode.CONSOLE
_console: Console | None = None


def get_console() -> Console:
    """Lazy-construct the shared Rich Console (highlight=False)."""
    global _console
    if _console is None:
        _console = Console(highlight=False)
    return _console


def set_mode(mode: OutputMode) -> None:
    """Switch the global output mode (called once at CLI startup)."""
    global _mode
    _mode = mode


def get_mode() -> OutputMode:
    """Return the current output mode."""
    return _mode


def print(*args, **kwargs) -> None:
    """Rich-aware print — active only in CONSOLE mode."""
    if _mode is OutputMode.CONSOLE:
        get_console().print(*args, **kwargs)


def event(event_type: str, **data) -> None:
    """Emit a structured event.

    In JSON_LINES mode writes a single JSON line to stdout.
    In CONSOLE and NONE modes suppressed.
    """
    if _mode is OutputMode.JSON_LINES:
        payload = {"event": event_type, **data}
        out = json.dumps(payload, default=str, ensure_ascii=False)
        sys.stdout.write(out + "\n")
        sys.stdout.flush()


def json_report(report_json: str) -> None:
    """Write the final report JSON to stdout in JSON modes.

    In CONSOLE and NONE modes this is a no-op — the CLI handles display directly.
    """
    if _mode is OutputMode.JSON_LINES:
        sys.stdout.write(report_json + "\n")
        sys.stdout.flush()


def engine_log(message: str) -> None:
    """Write engine discovery / fallback messages.

    Always uses raw stdout.write (no Rich markup) so it works even before
    the Console is initialised.
    """
    if _mode is OutputMode.CONSOLE:
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
