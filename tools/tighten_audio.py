#!/usr/bin/env python3
"""
Tighten narration and music audio: shorten pauses, clamp pace, trim silent intros.

TTS leaves pauses of 0.4-0.9s between sentences, and generated music often opens
with a beat of near-silence. Both read as dead air in a short. Run this after
voiceover.py and before captions.py (captions hold times relative to the audio).

Usage:
    # Per-scene VO: cap pauses inside each clip, clamp fast lines (rewrites the MP3s)
    uv run tools/tighten_audio.py --scene-dir projects/my-video/public/audio/scenes \
        --max-pause 0.22 --max-wpm 180

    # Music: cut the silent opening so the bed starts on frame 0
    uv run tools/tighten_audio.py --input projects/my-video/public/audio/music.mp3 --trim-lead

Originals are kept in a .tighten-backup/ folder beside the files.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pacing
from win_encoding import ensure_utf8_output

EDGE = 0.05  # silence kept at the head and tail of a narration clip
SILENCE_DB = -42


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out)


def silences(path: Path, min_len: float) -> list[tuple[float, float]]:
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"silencedetect=n={SILENCE_DB}dB:d={min_len}", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    if len(ends) < len(starts):
        ends.append(duration(path))
    return list(zip(starts, ends))


def backup(path: Path) -> None:
    bdir = path.parent / ".tighten-backup"
    bdir.mkdir(exist_ok=True)
    dest = bdir / path.name
    if not dest.exists():  # keep the first original, not an already-tightened take
        shutil.copy2(path, dest)


def render_segments(path: Path, keep: list[tuple[float, float]]) -> None:
    filt = "".join(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[s{i}];" for i, (a, b) in enumerate(keep))
    filt += "".join(f"[s{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1[out]"
    tmp = path.with_suffix(".tmp.mp3")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-filter_complex", filt, "-map", "[out]",
         "-c:a", "libmp3lame", "-b:a", "192k", str(tmp)],
        check=True,
    )
    tmp.replace(path)


def tighten_pauses(path: Path, max_pause: float) -> None:
    """Trim head/tail silence to EDGE and cap every internal silence at max_pause."""
    total = duration(path)
    keep: list[tuple[float, float]] = []
    cursor: float | None = 0.0
    for s, e in silences(path, 0.12):
        if s <= 0.01:  # leading silence
            cursor = max(0.0, e - EDGE)
            continue
        if e >= total - 0.01:  # trailing silence
            keep.append((cursor, min(total, s + EDGE)))
            cursor = None
            break
        if e - s > max_pause:
            half = max_pause / 2
            keep.append((cursor, s + half))
            cursor = e - half
    if cursor is not None:
        keep.append((cursor, total))
    render_segments(path, keep)


def trim_lead(path: Path) -> float:
    """Cut near-silence at the start (music intros). Returns seconds removed."""
    sil = silences(path, 0.1)
    # Sparse intros register as several short silences; cut through those that sit in the first second.
    cut = 0.0
    for s, e in sil:
        if s <= cut + 0.3 and e <= 1.5:
            cut = e
        else:
            break
    if cut > 0.02:
        render_segments(path, [(cut, duration(path))])
    return round(cut, 3)


def main():
    ensure_utf8_output()
    ap = argparse.ArgumentParser(description="Shorten pauses, clamp pace, trim silent intros")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene-dir", help="Directory of scene MP3s (each with a sibling .txt script)")
    src.add_argument("--input", help="A single audio file")
    ap.add_argument("--max-pause", type=float, default=0.22, help="Longest pause kept inside a clip (default 0.22s)")
    ap.add_argument("--max-wpm", type=float, help="Clamp lines faster than this (pitch-preserving atempo)")
    ap.add_argument("--min-tempo", type=float, default=0.8, help="Slowest atempo the clamp may use (default 0.8)")
    ap.add_argument("--trim-lead", action="store_true", help="--input only: just cut the silent opening (music)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    pacing.set_min_tempo(args.min_tempo)

    results = []
    if args.input:
        path = Path(args.input)
        backup(path)
        before = duration(path)
        if args.trim_lead:
            cut = trim_lead(path)
            results.append({"file": str(path), "trimmed_lead_seconds": cut,
                            "before": round(before, 2), "after": round(duration(path), 2)})
        else:
            tighten_pauses(path, args.max_pause)
            results.append({"file": str(path), "before": round(before, 2), "after": round(duration(path), 2)})
    else:
        for mp3 in sorted(Path(args.scene_dir).glob("*.mp3")):
            backup(mp3)
            before = duration(mp3)
            tighten_pauses(mp3, args.max_pause)
            tightened = duration(mp3)
            entry = {"file": mp3.name, "before": round(before, 2), "tightened": round(tightened, 2)}
            txt = mp3.with_suffix(".txt")
            if txt.exists():
                script = txt.read_text(encoding="utf-8").strip()
                if args.max_wpm:
                    r = pacing.clamp_pace(str(mp3), script, args.max_wpm, verbose=False)
                    if r.get("applied"):
                        entry["atempo"] = r["atempo"]
                entry["wpm"], entry["pacing"] = pacing.pace_label(script, duration(mp3))
            entry["after"] = round(duration(mp3), 2)
            results.append(entry)

    if args.json:
        print(json.dumps({"success": True, "results": results}, indent=2))
    else:
        for r in results:
            extra = f"  {r['wpm']} wpm" if r.get("wpm") else ""
            extra += f"  atempo {r['atempo']:.3f}" if r.get("atempo") else ""
            extra += f"  cut {r['trimmed_lead_seconds']}s" if "trimmed_lead_seconds" in r else ""
            print(f"{Path(r['file']).name}: {r['before']}s -> {r['after']}s{extra}")
        print("Originals in .tighten-backup/. Re-run captions.py now: caption times are relative to the audio.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
