"""YouTube chapters from the episode's own timeline: writes out/publish/chapters.txt.

    uv run scripts/chapters.py                                   # from a project made from this template
    uv run scripts/chapters.py --video out/episode.loudnorm.mp4  # and check the render is this timeline
    uv run templates/howto/scripts/chapters.py --json            # from the toolkit root

A chapter starts where its card does (the typed question), and the stretch before the first card (the hook, the
ident, the title) is the chapter at 0:00, named by the hook's question. The times are read from the timeline the
render plays (scripts/timeline.mjs prints src/howto/timeline.ts), which is laid out from src/timing.json and
src/ident.json, so nothing is added up by hand.

`tools/publish_kit.py` makes a chapter per narration file, which is right for a video with a file per scene and
wrong here (a file is a sentence): run it with `--no-chapters` for the subtitles and use this for the chapters.

YouTube's rules are applied: the first chapter at 0:00, three or more, each at least 10 s. A chapter that would
be shorter joins the one before it, and with fewer than three nothing is written. With `--video` the render's
length is compared with the timeline's: a render made before the narration was re-timed has other cuts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "publish" / "chapters.txt"
MIN_CHAPTER_S = 10
MIN_CHAPTERS = 3
# YouTube reads whole seconds. A mark lands between 0.3 s before its card and 0.7 s into it, so a viewer who
# jumps to a chapter sees its question being typed, not the end of the scene before.
INTO_CARD_S = 0.7


def read_timeline() -> dict:
    node = shutil.which("node")
    if not node:
        raise SystemExit("node not found: the timeline is read from the project's own TypeScript")
    if not (ROOT / "node_modules").exists():
        raise SystemExit(f"no node_modules in {ROOT}: run npm install there first")
    run = subprocess.run([node, str(ROOT / "scripts" / "timeline.mjs")], capture_output=True, text=True, cwd=ROOT)
    if run.returncode != 0:
        raise SystemExit(run.stderr.strip() or "scripts/timeline.mjs failed")
    return json.loads(run.stdout)


def sentence(text: str) -> str:
    """A typed question as a chapter title: 'what do I need first?' -> 'What do I need first?'"""
    text = " ".join(text.split())
    return text[:1].upper() + text[1:]


def fmt(seconds: int) -> str:
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters_from(timeline: dict, titles: str = "title") -> tuple[list[dict], list[str]]:
    """(chapters, notes). `titles` is 'title' (the chapter's title) or 'ask' (its question, where it has one)."""
    fps = timeline["fps"]
    total = timeline["frames"] / fps
    notes: list[str] = []
    hook = (timeline.get("hook") or {}).get("ask")
    marks: list[dict] = [{"at": 0, "title": sentence(hook) if hook else " ".join(timeline["title"])}]
    for c in timeline["chapters"]:
        if c.get("start") is None:
            notes.append(f"chapter '{' '.join(c['title'])}' has no scenes, so it has no chapter mark")
            continue
        name = sentence(c["ask"]) if titles == "ask" and c.get("ask") else " ".join(c["title"])
        at = int(c["start"] / fps + INTO_CARD_S)
        if at == 0:  # no hook and no opening before it: the first chapter is the 0:00 one
            marks[0]["title"] = name
            continue
        if at - marks[-1]["at"] < MIN_CHAPTER_S:
            notes.append(f"'{name}' starts {at - marks[-1]['at']} s after '{marks[-1]['title']}', under YouTube's "
                         f"{MIN_CHAPTER_S} s, so it joins that chapter")
            continue
        marks.append({"at": at, "title": name})
    if len(marks) > 1 and total - marks[-1]["at"] < MIN_CHAPTER_S:
        last = marks.pop()
        notes.append(f"'{last['title']}' runs under {MIN_CHAPTER_S} s to the end, so it joins the chapter before")
    if len(marks) < MIN_CHAPTERS:
        notes.append(f"only {len(marks)} chapter(s) fit; YouTube needs {MIN_CHAPTERS}, so none are written")
        return [], notes
    return marks, notes


def video_seconds(path: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return float(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="YouTube chapters from the episode's timeline.")
    ap.add_argument("--video", help="The render: its length is checked against the timeline's")
    ap.add_argument("--titles", choices=["title", "ask"], default="title",
                    help="Name chapters by their title (default) or by the question typed on their card")
    ap.add_argument("--output", default=str(OUT), help="Where to write (default out/publish/chapters.txt)")
    ap.add_argument("--json", action="store_true", help="Print the chapters and notes as JSON")
    args = ap.parse_args()

    timeline = read_timeline()
    marks, notes = chapters_from(timeline, args.titles)
    total = timeline["frames"] / timeline["fps"]
    stale = False
    if args.video:
        video = Path(args.video)
        video = video if video.is_absolute() or video.exists() else ROOT / args.video
        got = video_seconds(video)
        if abs(got - total) > 0.25:
            stale = True
            notes.append(f"{video.name} is {got:.1f} s and the timeline is {total:.1f} s: render again, or these "
                         "chapters are for another cut")
    text = "".join(f"{fmt(m['at'])} {m['title']}\n" for m in marks)
    if marks and not stale:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if args.json:
        print(json.dumps({"seconds": round(total, 2), "chapters": marks, "notes": notes,
                          "file": str(args.output) if marks and not stale else None}, indent=1))
    else:
        sys.stdout.write(text)
        for n in notes:
            print(f"note: {n}", file=sys.stderr)
        if marks and not stale:
            print(f"wrote {args.output}", file=sys.stderr)
    return 1 if stale or not marks else 0


if __name__ == "__main__":
    sys.exit(main())
