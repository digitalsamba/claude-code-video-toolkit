#!/usr/bin/env python3
"""Step 2 — word-level caption timing for burned karaoke captions.

Whisper provides word *timestamps*; scenes.json provides ground-truth *text*.
Script words are aligned onto whisper's timeline (difflib word matching,
interpolating unmatched runs) so the burned captions are word-perfect even
when whisper mishears the TTS — never burn whisper's own transcription.

Requires openai-whisper (not in the toolkit's base requirements — it pulls
in torch):    pip install openai-whisper

Run from this project directory after gen_vo.py:
    python3 gen_captions.py
Writes captions/words_{id}.json — chunks with start/end seconds relative to
each scene's audio file. Captions are optional: build.py renders without them.
"""
from __future__ import annotations

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
             "    pip install openai-whisper\n"
             "(or set captions.enabled=false in config.json and skip this step)")


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
    return [{"w": w, "s": starts[idx], "e": ends[idx]}
            for idx, w in enumerate(script_words)]


def chunk(words: list[dict], max_words: int, max_chars: int) -> list[dict]:
    chunks, cur = [], []
    for w in words:
        cand = " ".join([c["w"] for c in cur] + [w["w"]])
        if cur and (len(cur) >= max_words or len(cand) > max_chars):
            chunks.append({"text": " ".join(c["w"] for c in cur),
                           "start": cur[0]["s"], "end": cur[-1]["e"]})
            cur = []
        cur.append(w)
    if cur:
        chunks.append({"text": " ".join(c["w"] for c in cur),
                       "start": cur[0]["s"], "end": cur[-1]["e"]})
    # stretch each chunk to meet the next so captions never flicker off
    for i in range(len(chunks) - 1):
        chunks[i]["end"] = max(chunks[i]["end"], chunks[i + 1]["start"])
    return chunks


def main() -> None:
    config = json.loads((HERE / "config.json").read_text())
    cap = config.get("captions", {})
    scenes = json.loads((HERE / "scenes.json").read_text())["scenes"]
    out_dir = HERE / "captions"
    out_dir.mkdir(exist_ok=True)

    model = whisper.load_model(cap.get("whisperModel", "base"))
    for s in scenes:
        mp3 = HERE / "audio" / "scenes" / f"{s['id']}_{s['slug']}.mp3"
        if not mp3.exists():
            print(f"missing {mp3.name} — run gen_vo.py first; skipping")
            continue
        r = model.transcribe(str(mp3), word_timestamps=True, language="en")
        heard = [{"w": w["word"].strip(), "s": w["start"], "e": w["end"]}
                 for seg in r["segments"] for w in seg["words"]]
        words = align(s["text"].split(), heard)
        chunks = chunk(words, cap.get("maxWords", 3), cap.get("maxChars", 22))
        out = out_dir / f"words_{s['id']}.json"
        out.write_text(json.dumps(chunks, indent=1))
        print(f"  {out.name}: {len(chunks)} chunks")

    print("Next: python3 build.py")


if __name__ == "__main__":
    main()
