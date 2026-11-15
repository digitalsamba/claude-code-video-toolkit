#!/usr/bin/env python3
"""
Publish kit: what a finished render needs on YouTube besides the file, built from the render itself.

    uv run tools/publish_kit.py --project projects/my-video                 # subtitles + chapters
    uv run tools/publish_kit.py --project projects/my-video --video out/video.loudnorm.mp4 --json
    uv run tools/publish_kit.py --project projects/my-video --no-chapters   # subtitles only

Writes `out/publish/`:

- `video.srt` (and `video.captions.json`): one subtitle file for the whole video. The render's audio is
  transcribed once and aligned to the scene scripts, so the timing is the render's own (no scene offsets to add
  up) and the words are spelled as the scripts spell them.
- `chapters.txt`: a chapter at each scene's cut (the middle of the pause before its first word), titled from `project.json` (`scenes[].title`) or
  the scene file name (`02_narrate.txt` -> "Narrate"). YouTube's rules are applied: the first chapter at 0:00,
  at least three, each at least 10 s; a scene too short for its own chapter joins the one before it. Edit the
  titles, then `/publish` puts the list in the description.
- `kit.json`: the paths, the chapters and the cost, for `/publish` to read.

Every part is optional: with no scene scripts the subtitles come from the raw transcript and there are no
chapters; `--no-captions` or `--no-chapters` skips one. Scene scripts are the `.txt` beside each scene MP3
(`public/audio/scenes/` by default, `--scenes DIR` otherwise), in file-name order, which must be the order they
play in.

Cost: one transcription of the render. Scribe is $0.40 an audio hour (about $0.01 for 90 s, needs
ELEVENLABS_API_KEY); `--engine whisper` is local and free (`uv sync --extra whisper`).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from win_encoding import ensure_utf8_output  # noqa: E402

SCRIBE_USD_PER_S = 0.40 / 3600
MIN_CHAPTER_S = 10
MIN_CHAPTERS = 3


def find_video(project: Path, given: str | None) -> Path:
    if given:
        p = Path(given)
        return p if p.is_absolute() or p.exists() else project / given
    meta = _project_json(project)
    rel = (meta.get("render") or {}).get("file")
    if rel and (project / rel).exists():
        return project / rel
    renders = sorted((project / "out").glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not renders:
        raise SystemExit(f"No render in {project / 'out'}; pass --video")
    return renders[0]


def _project_json(project: Path) -> dict:
    try:
        return json.loads((project / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def scene_scripts(scenes_dir: Path) -> list[Path]:
    """Each scene's .txt, in play order (file-name order). Only those beside an audio file count, so notes in the
    folder aren't read as narration."""
    if not scenes_dir.is_dir():
        return []
    audio = {".mp3", ".wav", ".m4a"}
    return sorted(t for t in scenes_dir.glob("*.txt")
                  if any(t.with_suffix(ext).exists() for ext in audio))


def title_from_name(path: Path) -> str:
    """'02_narrate.txt' -> 'Narrate', '03-the-bill.txt' -> 'The bill'."""
    words = re.sub(r"^[\d\s_.-]+", "", path.stem).replace("_", " ").replace("-", " ").strip()
    return (words[:1].upper() + words[1:]) if words else path.stem


def scene_titles(project: Path, scripts: list[Path]) -> list[str]:
    scenes = _project_json(project).get("scenes") or []
    titles = []
    for i, path in enumerate(scripts):
        given = scenes[i].get("title") if i < len(scenes) and isinstance(scenes[i], dict) else None
        titles.append(given or title_from_name(path))
    return titles


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def chapters_from(starts: list[float], titles: list[str], duration: float) -> tuple[list[dict], list[str]]:
    """YouTube chapters from scene start times. Returns (chapters, notes); chapters is empty when the video
    can't carry three of at least 10 s each."""
    notes = []
    chapters: list[dict] = []
    for start, title in zip(starts, titles):
        at = 0.0 if not chapters else float(int(start))   # YouTube reads whole seconds
        if chapters and at - chapters[-1]["at"] < MIN_CHAPTER_S:
            notes.append(f"'{title}' starts {at - chapters[-1]['at']:.0f} s after '{chapters[-1]['title']}', "
                         f"under YouTube's {MIN_CHAPTER_S} s, so it joins that chapter")
            continue
        chapters.append({"at": at, "title": title})
    if chapters and duration - chapters[-1]["at"] < MIN_CHAPTER_S:
        last = chapters.pop()
        notes.append(f"'{last['title']}' runs under {MIN_CHAPTER_S} s to the end, so it joins the chapter before")
    if len(chapters) < MIN_CHAPTERS:
        notes.append(f"only {len(chapters)} chapter(s) fit; YouTube needs {MIN_CHAPTERS}, so none are written")
        return [], notes
    return chapters, notes


def extract_audio(video: Path, out: Path) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-c:a", "libmp3lame",
                        "-b:a", "96k", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg couldn't read the audio of {video}: {r.stderr.strip()[-300:]}")


def duration_of(video: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        raise SystemExit(f"ffprobe couldn't read the length of {video}")


def transcribe(audio: Path, engine: str, language: str | None) -> tuple[list[dict], str]:
    import os

    import captions as cap
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("ELEVENLABS_API_KEY")
    if engine == "auto":
        engine = "scribe" if key else "whisper"
    if engine == "scribe":
        if not key:
            raise SystemExit("ELEVENLABS_API_KEY not set (or use --engine whisper)")
        from voiceover import resolve_model_id
        return cap.transcribe_scribe(audio, key, resolve_model_id("scribe_v2"), language), "scribe"
    return cap.transcribe_whisper(cap.load_whisper("auto"), audio, language), "whisper"


def build(args) -> dict:
    import captions as cap
    from pacing import spoken_text

    project = Path(args.project)
    if not project.is_dir():
        raise SystemExit(f"Not a project folder: {project}")
    video = find_video(project, args.video)
    if not video.exists():
        raise SystemExit(f"Render not found: {video}")
    out_dir = project / "out" / "publish"
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = duration_of(video)

    scripts = scene_scripts(Path(args.scenes) if args.scenes else project / "public" / "audio" / "scenes")
    scene_words = [spoken_text(p.read_text(encoding="utf-8")).split() for p in scripts]
    all_words = [w for ws in scene_words for w in ws]

    saved = out_dir / ("video" + cap.SUFFIX)
    reuse = not args.fresh and saved.exists() and saved.stat().st_mtime > video.stat().st_mtime
    match = None
    if reuse:
        # the transcript is what costs money: a re-run after editing titles reads it back instead
        words = [{"w": c["text"].strip(), "s": c["startMs"] / 1000, "e": c["endMs"] / 1000}
                 for c in json.loads(saved.read_text(encoding="utf-8"))]
        engine = "saved"
        if all_words and len(words) != len(all_words):
            raise SystemExit(f"{saved.name} has {len(words)} words but the scene scripts have {len(all_words)}: "
                             "they changed since it was made, so run again with --fresh")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "render.mp3"
            extract_audio(video, audio)
            heard, engine = transcribe(audio, args.engine, args.language)
        if not heard:
            raise SystemExit(f"No speech found in {video.name}")
        words = heard
        if all_words:
            words, match = cap.align(all_words, heard)

    result: dict = {
        "video": str(video), "durationSeconds": round(duration, 2), "engine": engine,
        "costUsd": round(duration * SCRIBE_USD_PER_S, 4) if engine == "scribe" else 0.0,
        "scenes": len(scripts), "matchRatio": round(match, 3) if match is not None else None,
        "notes": [],
    }
    if match is not None and match < cap.MIN_MATCH_RATIO:
        result["notes"].append(f"only {match:.0%} of the script words were heard: are the scene .txt files the "
                               "ones in this render?")
    if not scripts:
        result["notes"].append("no scene scripts found, so subtitles use the raw transcript and there are no chapters")

    # always written (local, free, and the paid transcript is kept); --no-captions leaves it out of the kit
    paths = cap.write_caption_outputs(saved, cap.to_remotion(words), srt=True, max_words=args.max_words)
    if not args.no_captions:
        result["captions"], result["srt"] = paths["captions"], paths["srt"]

    if scripts and not args.no_chapters:
        # a scene's cut sits in the pause between the last scene's last word and its own first word: take the
        # middle of it (measured on Made with itself, within 0.5 s of every cut; the first word alone ran 1-3 s late)
        starts, k = [], 0
        for ws in scene_words:
            if not ws or k >= len(words):
                starts.append(starts[-1] if starts else 0.0)
            elif k == 0:
                starts.append(0.0)
            else:
                starts.append((words[k - 1]["e"] + words[k]["s"]) / 2)
            k += len(ws)
        chapters, notes = chapters_from(starts, scene_titles(project, scripts), duration)
        result["notes"] += notes
        result["chapters"] = [{"at": fmt_time(c["at"]), "seconds": c["at"], "title": c["title"]} for c in chapters]
        if chapters:
            text = "\n".join(f"{fmt_time(c['at'])} {c['title']}" for c in chapters) + "\n"
            (out_dir / "chapters.txt").write_text(text, encoding="utf-8")
            result["chaptersFile"] = str(out_dir / "chapters.txt")

    (out_dir / "kit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["kit"] = str(out_dir / "kit.json")
    return result


def main(argv=None) -> None:
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description="Subtitles and chapters for YouTube, from a finished render.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--project", required=True, help="Project folder (projects/NAME)")
    ap.add_argument("--video", help="The render (default: project.json render.file, else the newest out/*.mp4)")
    ap.add_argument("--scenes", help="Folder of scene scripts (default: public/audio/scenes)")
    ap.add_argument("--engine", choices=["auto", "scribe", "whisper"], default="auto",
                    help="Transcription (default: scribe if ELEVENLABS_API_KEY is set, else whisper)")
    ap.add_argument("--language", help="Language code, e.g. en (default: auto-detect)")
    ap.add_argument("--max-words", type=int, default=7, help="Words per subtitle cue (default 7)")
    ap.add_argument("--no-captions", action="store_true", help="Leave the subtitles out of the kit")
    ap.add_argument("--no-chapters", action="store_true", help="Skip the chapters")
    ap.add_argument("--fresh", action="store_true",
                    help="Transcribe again even if out/publish/video.captions.json is newer than the render")
    ap.add_argument("--json", action="store_true", help="Print kit.json on stdout")
    args = ap.parse_args(argv)

    r = build(args)
    if args.json:
        print(json.dumps(r, indent=2))
        return
    print(f"{Path(r['video']).name}: {r['durationSeconds']:.1f} s, {r['engine']}, ${r['costUsd']:.4f}", file=sys.stderr)
    if r.get("matchRatio") is not None:
        print(f"  {r['scenes']} scene scripts, {r['matchRatio']:.0%} of their words heard", file=sys.stderr)
    if r.get("srt"):
        print(f"  subtitles: {r['srt']}", file=sys.stderr)
    for c in r.get("chapters") or []:
        print(f"  {c['at']} {c['title']}", file=sys.stderr)
    for n in r["notes"]:
        print(f"  note: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
