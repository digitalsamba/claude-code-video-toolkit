#!/usr/bin/env python3
"""Step 2 — word-level caption timing (youtube-longform).

Whisper provides word *timestamps*; scenes.json provides ground-truth *text*.
Script words are aligned onto whisper's timeline (difflib word matching,
interpolating unmatched runs) so captions are word-perfect even when whisper
mishears the TTS — never use whisper's own transcription.

Writes captions/words_{id}.json — one entry per script word, seconds relative
to that scene's audio file. build.py chunks these two ways: into the uploadable
out/captions.srt (always) and into burned-in captions (if captions.burn is on).

Requires openai-whisper (pulls in torch):    uv sync --extra whisper

Run from this project directory after gen_vo.py:
    uv run gen_captions.py
    uv run gen_captions.py --force    # redo scenes that already have timing
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

try:
    import whisper
except ImportError:
    sys.exit("openai-whisper is required for captions:\n"
             "    uv sync --extra whisper\n"
             "(captions are optional — build.py renders without them)")


def pick_model(requested: str | None) -> str:
    """Resolve "auto"/unset to turbo on CUDA, base on CPU; explicit names pass through."""
    if requested and requested != "auto":
        return requested
    import torch  # whisper already depends on it
    return "turbo" if torch.cuda.is_available() else "base"


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def align(script_words: list[str], heard: list[dict]) -> list[dict]:
    """Map script words onto heard-word timings; interpolate unmatched runs."""
    sm = difflib.SequenceMatcher(
        a=[norm(w) for w in script_words],
        b=[norm(h["w"]) for h in heard],
        autojunk=False,
    )
    starts: list[float | None] = [None] * len(script_words)
    ends: list[float | None] = [None] * len(script_words)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            starts[a + k] = heard[b + k]["s"]
            ends[a + k] = heard[b + k]["e"]
    total_end = heard[-1]["e"] if heard else 0.0
    i = 0
    while i < len(script_words):
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < len(script_words) and starts[j] is None:
            j += 1
        lo = ends[i - 1] if i > 0 else 0.0
        hi = starts[j] if j < len(script_words) else total_end
        span = max(hi - lo, 0.12 * (j - i))
        for k in range(i, j):
            starts[k] = lo + span * (k - i) / (j - i)
            ends[k] = lo + span * (k - i + 1) / (j - i)
        i = j
    return [{"w": w, "s": round(starts[idx], 3), "e": round(ends[idx], 3)}
            for idx, w in enumerate(script_words)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Redo scenes that already have timing")
    args = ap.parse_args()

    config = json.loads((HERE / "config.json").read_text())
    cap = config.get("captions", {})
    scenes = json.loads((HERE / "scenes.json").read_text())["scenes"]
    out_dir = HERE / "captions"
    out_dir.mkdir(exist_ok=True)

    todo = []
    for s in scenes:
        mp3 = HERE / "audio" / "scenes" / f"{s['id']}_{s['slug']}.mp3"
        out = out_dir / f"words_{s['id']}.json"
        if not mp3.exists():
            print(f"missing {mp3.name} — run gen_vo.py first; skipping")
        elif args.force or not out.exists() or out.stat().st_mtime < mp3.stat().st_mtime:
            todo.append((s, mp3, out))
    if not todo:
        print("All scenes already have caption timing — use --force to redo.")
        return

    model_name = pick_model(cap.get("whisperModel", "auto"))
    print(f"whisper model: {model_name} ({len(todo)} scene(s))")
    model = whisper.load_model(model_name)
    for s, mp3, out in todo:
        r = model.transcribe(str(mp3), word_timestamps=True,
                             language=cap.get("language", "en"))
        heard = [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]}
                 for seg in r["segments"] for w in seg["words"]]
        words = align(s["text"].split(), heard)
        out.write_text(json.dumps(words, indent=1))
        print(f"  {out.name}: {len(words)} words")

    print("Next: uv run build.py")


if __name__ == "__main__":
    main()
