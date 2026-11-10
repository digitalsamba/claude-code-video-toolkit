#!/usr/bin/env python3
"""
Copy lint for anything public: dashes and stock AI phrasing, the marks of text nobody read before it shipped.

The same rules refuse a video job's narration and on-screen text before any spend (video_job.py). Run this on
every other word that goes out: site pages, README, YouTube titles and descriptions, release notes, listings.

Examples:
  uv run tools/copy_lint.py site/*.html README.md
  uv run tools/copy_lint.py --text "A game-changer for video"     # exit 1
  uv run tools/copy_lint.py --diff origin/main                    # only lines added on this branch (CI)
  uv run tools/copy_lint.py --commits origin/main                 # commit subjects, which become release notes
  uv run tools/youtube_stats.py ... ; uv run tools/copy_lint.py DESCRIPTION.md

HTML and Markdown are read as prose: code blocks, inline code, tags and comments are skipped, so a `--flag` or
a code comment's dash isn't flagged. Exit 1 on any finding.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

COPY_RULES = [
    (re.compile(r"—|\s–\s|\s--\s"), "a dash; use a comma, a full stop or a new sentence"),
    (re.compile(r"\bit'?s worth noting\b|\bit is (worth|important) (noting|to note)\b", re.I),
     "'it's worth noting'; say the point"),
    (re.compile(r"\bin today'?s (fast-paced |busy |modern )?(world|age|landscape)\b|\bmoving forward\b"
                r"|\bfirst and foremost\b|\blast but not least\b|\bwithout further ado\b|\ball in all\b"
                r"|\bin conclusion\b|\bshed light on\b|\bpave the way\b|\bat the forefront\b|\bdeep[- ]dive\b"
                r"|\bdelve\b|\btapestry\b|\bembark on a journey\b|\bunlock (your|the) (full )?potential\b"
                r"|\bgame[- ]changer\b|\bnavigat(e|ing) the (complexities|world|landscape)\b"
                r"|\bseamless(ly)?\b|\bresonate(s)? with\b|\btestament to\b", re.I),
     "stock AI phrasing; say it the way a person would"),
    # taste-skill's stock marketing phrases (scripts/lint.mjs, 'copy'), so one list covers both
    (re.compile(r"\b(unleash|supercharge|elevate[sd]?|empower(s|ing)?|revolutioni[sz]e[sd]?|game[- ]changing"
                r"|cutting[- ]edge|next[- ]gen(eration)?|world[- ]class|best[- ]in[- ]class|all[- ]in[- ]one"
                r"|effortless(ly)?|blazing(ly)?[- ]fast|lightning[- ]fast|reimagin(e|ed|ing)|leverag(e|es|ing)"
                r"|synergy|streamline[sd]?|holistic|turbocharge[sd]?|transform the way"
                r"|take your .{1,30} to the next level|say goodbye to|the future of|built for the modern)\b", re.I),
     "a stock marketing phrase; say the specific thing it does"),
]


def lint_text(text: str) -> list[tuple[str, str]]:
    """(matched text, reason) for every rule hit in a piece of prose."""
    out = []
    for rx, why in COPY_RULES:
        for m in rx.finditer(text):
            out.append((m.group(0).strip(), why))
    return out


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """The file's prose, line by line, with code, tags and comments blanked so line numbers still match."""
    src = path.read_text(encoding="utf-8")
    keep_newlines = lambda m: "\n" * m.group(0).count("\n")
    if path.suffix.lower() in (".html", ".htm"):
        src = re.sub(r"<!--.*?-->|<(pre|code|script|style)\b.*?</\1>", keep_newlines, src, flags=re.S | re.I)
        # the title and meta description are public too (search results, link previews): keep their text
        src = re.sub(r"<meta\b[^>]*\bcontent=\"([^\"]*)\"[^>]*>", r" \1 ", src, flags=re.I)
        src = re.sub(r"<[^>]+>", " ", src)
        src = html.unescape(src)
    elif path.suffix.lower() in (".md", ".markdown"):
        src = re.sub(r"```.*?```|<!--.*?-->", keep_newlines, src, flags=re.S)
        src = re.sub(r"`[^`\n]*`", " ", src)
        src = re.sub(r"\]\([^)]*\)", "]", src)  # link targets
    return list(enumerate(src.split("\n"), 1))


def lint_file(path: Path) -> list[dict]:
    return [{"file": str(path), "line": n, "match": m, "reason": why}
            for n, line in prose_lines(path) for m, why in lint_text(line)]


PROSE = (".md", ".markdown", ".html", ".htm", ".txt")
# Folders of saved command output (an example's capture/): what a tool printed, kept verbatim so the video can
# show it, not words anyone wrote. Linting them would ask for the record to be edited.
VERBATIM_DIRS = {"capture"}


def added_lines(ref: str, path: Path) -> set[int]:
    """Line numbers in the working copy of `path` that are new since `ref`."""
    out = subprocess.run(["git", "diff", "-U0", ref, "--", str(path)], capture_output=True, text=True).stdout
    lines: set[int] = set()
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def changed_files(ref: str, paths: list[Path]) -> list[Path]:
    cmd = ["git", "diff", "--name-only", "--diff-filter=AM", ref, "--"] + [str(p) for p in paths]
    names = subprocess.run(cmd, capture_output=True, text=True).stdout.split()
    return [Path(n) for n in names if Path(n).suffix.lower() in PROSE and Path(n).exists()
            and not VERBATIM_DIRS.intersection(Path(n).parts[:-1])]


def commit_subjects(ref: str) -> list[str]:
    out = subprocess.run(["git", "log", "--no-merges", "--format=%s", f"{ref}..HEAD"], capture_output=True,
                         text=True).stdout
    return [s for s in out.splitlines() if s.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", type=Path, help="HTML, Markdown or text files")
    p.add_argument("--text", action="append", default=[], help="lint a string (repeatable): a title, a tagline")
    p.add_argument("--diff", metavar="REF", help="lint only lines added since REF, in the given files or folders "
                   "(default: the whole repo); old text is fixed when it's next touched")
    p.add_argument("--commits", metavar="REF", help="lint the subjects of commits since REF (they become release notes)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    if not a.files and not a.text and not a.diff and not a.commits:
        p.error("give files, --text, --diff or --commits")
    if a.commits:
        a.text += commit_subjects(a.commits)

    findings = [{"file": "--text", "line": i, "match": m, "reason": why}
                for i, t in enumerate(a.text, 1) for m, why in lint_text(t)]
    if a.diff:
        for f in changed_files(a.diff, a.files or [Path(".")]):
            new = added_lines(a.diff, f)
            findings += [x for x in lint_file(f) if x["line"] in new]
    else:
        for f in a.files:
            if not f.exists():
                print(f"Error: {f} not found", file=sys.stderr)
                return 2
            findings += lint_file(f)

    if a.json:
        print(json.dumps({"ok": not findings, "findings": findings}, indent=2))
    else:
        for x in findings:
            print(f"{x['file']}:{x['line']}  '{x['match']}' is {x['reason']}")
        print(f"{len(findings)} finding(s)" if findings else "copy lint: clean")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
