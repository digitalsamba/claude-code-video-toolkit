"""UTF-8 console output on Windows.

Python on Windows picks the *locale* codec (usually cp1252) for `stdout`/`stderr`
whenever output is redirected rather than attached to a terminal — which is what
happens whenever a tool is piped, captured by a harness like Claude Code, or read
with `subprocess.run(..., capture_output=True)`. Printing any character outside
cp1252 then raises:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2713' ...

The traceback points at the `print()`, not at the encoding, so it reads like a bug
in whatever tool happened to emit an arrow or a check mark.

Two helpers:

* `ensure_utf8_output()` — call once at the top of a tool's `main()`. Re-opens
  `stdout`/`stderr` as UTF-8. No-op when they are already UTF-8 (macOS, Linux,
  Windows with `PYTHONUTF8=1`), so it is safe to call unconditionally.
* `utf8_env()` — environment for `subprocess`, so a *child* process (notably the
  `modal` CLI, whose progress output is full of box-drawing characters) writes
  UTF-8 too. Pair it with `encoding="utf-8"` on the `subprocess` call itself.

Python 3.15 makes UTF-8 mode the default (PEP 686), at which point both become
no-ops rather than wrong.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping


def _is_utf8(stream) -> bool:
    enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return enc in ("utf8", "utf8mb4")


def ensure_utf8_output() -> None:
    """Force `stdout`/`stderr` to UTF-8. Idempotent and never raises."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or _is_utf8(stream):
            continue
        try:
            # Python 3.7+. errors="replace" keeps a stray glyph from killing a
            # long-running render that is otherwise fine.
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Detached, already-wrapped, or non-reconfigurable stream. Printing
            # ASCII still works, so degrade quietly rather than break the tool.
            pass


def utf8_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment dict that makes a child process emit UTF-8.

    Pass to `subprocess.run(..., env=utf8_env(), encoding="utf-8")`.
    """
    env = dict(os.environ if base is None else base)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


__all__ = ["ensure_utf8_output", "utf8_env"]
