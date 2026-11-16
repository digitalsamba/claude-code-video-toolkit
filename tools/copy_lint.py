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
  uv run tools/copy_lint.py --brand digital-samba narration/*.txt  # plus the brand's own rules (copy.json)

A brand can carry its own rules in brands/<name>/copy.json, beside its tone guide copy.md: `prefer` (a spelling
or word, and the one to use; `except` lists phrases where it stands, e.g. a product name), `avoid` (phrases it
doesn't use; `"level": "check"` reports a claim to verify at publish time without failing) and `names` (product
names and what to call them instead). A phrase with a capital letter matches case-sensitively, so "Zoom" the
product doesn't flag "zoom in"; the rest match whole words in any case.

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


ROOT = Path(__file__).resolve().parent.parent


def brand_rules(name: str) -> list[tuple[re.Pattern, str, list[str], str]]:
    """A brand's rules from brands/<name>/copy.json: (pattern, reason, exceptions, level) each."""
    path = ROOT / "brands" / name / "copy.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: a brand's copy rules live there (see brands/digital-samba)")
    data = json.loads(path.read_text(encoding="utf-8"))

    def rx(phrase: str) -> re.Pattern:
        flags = 0 if any(c.isupper() for c in phrase) else re.I
        return re.compile(r"(?<![\w-])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![\w-])", flags)

    out = []
    for e in data.get("prefer", []):
        out.append((rx(e["avoid"]), f"{name} writes '{e['use']}' ({e.get('why', '')})", e.get("except", []), e.get("level", "error")))
    for e in data.get("avoid", []):
        out.append((rx(e["phrase"]), f"not {name}'s words ({e.get('why', '')})", e.get("except", []), e.get("level", "error")))
    for e in data.get("names", []):
        out.append((rx(e["avoid"]), f"{name} calls it '{e['use']}' ({e.get('why', '')})", e.get("except", []), e.get("level", "error")))
    return out


def lint_text(text: str, brand: list | None = None) -> list[tuple[str, str]]:
    """(matched text, reason) for every rule hit in a piece of prose; a brand rule at level "check" is
    reported with its reason starting "check:" (see `failing`)."""
    out = []
    for rx, why in COPY_RULES:
        for m in rx.finditer(text):
            out.append((m.group(0).strip(), why))
    for rx, why, exceptions, level in brand or []:
        spared = [(m.start(), m.end()) for e in exceptions for m in re.finditer(re.escape(e), text)]
        for m in rx.finditer(text):
            if any(a <= m.start() and m.end() <= b for a, b in spared):
                continue
            out.append((m.group(0).strip(), f"check: {why}" if level == "check" else why))
    return out


def failing(findings: list[dict]) -> list[dict]:
    return [x for x in findings if not x["reason"].startswith("check:")]


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


def lint_file(path: Path, brand: list | None = None) -> list[dict]:
    return [{"file": str(path), "line": n, "match": m, "reason": why}
            for n, line in prose_lines(path) for m, why in lint_text(line, brand)]


PROSE = (".md", ".markdown", ".html", ".htm", ".txt")
# Folders of saved command output (an example's capture/): what a tool printed, kept verbatim so the video can
# show it, not words anyone wrote. Linting them would ask for the record to be edited.
VERBATIM_DIRS = {"capture"}
# A licence shipped beside a font or a model is someone else's text, kept word for word.
VERBATIM_NAMES = {"ofl.txt", "license.txt", "licence.txt", "license.md", "licence.md", "notice.txt"}


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
            and not VERBATIM_DIRS.intersection(Path(n).parts[:-1]) and Path(n).name.lower() not in VERBATIM_NAMES]


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
    p.add_argument("--brand", metavar="NAME", help="also apply brands/NAME/copy.json (spelling, words, product names)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        brand = brand_rules(a.brand) if a.brand else None
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if not a.files and not a.text and not a.diff and not a.commits:
        p.error("give files, --text, --diff or --commits")
    if a.commits:
        a.text += commit_subjects(a.commits)

    findings = [{"file": "--text", "line": i, "match": m, "reason": why}
                for i, t in enumerate(a.text, 1) for m, why in lint_text(t, brand)]
    if a.diff:
        for f in changed_files(a.diff, a.files or [Path(".")]):
            new = added_lines(a.diff, f)
            findings += [x for x in lint_file(f, brand) if x["line"] in new]
    else:
        for f in a.files:
            if not f.exists():
                print(f"Error: {f} not found", file=sys.stderr)
                return 2
            findings += lint_file(f, brand)

    bad = failing(findings)
    if a.json:
        print(json.dumps({"ok": not bad, "findings": findings}, indent=2))
    else:
        for x in findings:
            print(f"{x['file']}:{x['line']}  '{x['match']}' is {x['reason']}")
        checks = len(findings) - len(bad)
        print((f"{len(bad)} finding(s)" if bad else "copy lint: clean") + (f", {checks} claim(s) to check" if checks else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
