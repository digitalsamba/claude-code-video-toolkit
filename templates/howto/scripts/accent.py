"""Has a British voice gone American on a word? Measures it, so the line is caught before anyone listens.

    uv run --extra prosody scripts/accent.py                       # card, dollar(s), your, more, first, four, ...
    uv run --extra prosody scripts/accent.py --words card,your --max-hz 5500

A library voice can change accent on one word (the pilot's narrator said "dollars" and "card" with an American
r). An American r pulls the third formant down at the end of the vowel: under about 2,200 Hz in a woman's voice,
where a British ending leaves it near 2,700. This reads each word's times from the `.captions.json` beside the
narration, takes the last 45% of the word and reports the median third formant there, lowest first. A word under
the line is one to listen to and probably re-voice with `[crisp British English accent]` at the start of its
text. It is a screen, not a verdict: a short or creaky word can read low, and so does a word before a vowel
("your account", "for anything"), where British speech sounds a linking r as well. For a man's voice use `--max-hz 5000`
and expect both numbers a few hundred hertz lower.

Needs the prosody extra (Praat through parselmouth): uv sync --extra prosody
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "public" / "audio" / "scenes"
WORDS = "card,cards,dollar,dollars,your,yours,more,first,four,for,after,order,start,part,here,there,transfers"


def third_formant(sound, start: float, end: float, max_hz: float) -> float | None:
    import parselmouth  # noqa: F401  (imported here so --help works without the extra)

    a = start + (end - start) * 0.55
    if end - a < 0.03:
        return None
    part = sound.extract_part(from_time=max(0, a - 0.03), to_time=min(sound.duration, end + 0.03), preserve_times=True)
    formant = part.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=max_hz)
    values = []
    t = a
    while t <= end:
        v = formant.get_value_at_time(3, t)
        if v == v:  # not NaN
            values.append(v)
        t += 0.005
    return statistics.median(values) if len(values) >= 3 else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Screen narration for words said with an American r.")
    ap.add_argument("--scenes", default=str(SCENES), help="Folder of <name>.mp3 + <name>.captions.json")
    ap.add_argument("--words", default=WORDS, help="Words to measure, comma separated")
    ap.add_argument("--line", type=float, default=2200, help="Third formant under this is flagged (default 2200 Hz)")
    ap.add_argument("--max-hz", type=float, default=5500, help="Formant ceiling: 5500 for a woman, 5000 for a man")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        import parselmouth
    except ImportError:
        print("needs the prosody extra: uv sync --extra prosody", file=sys.stderr)
        return 2
    want = {w.strip().lower() for w in args.words.split(",") if w.strip()}
    rows = []
    for caps in sorted(Path(args.scenes).glob("*.captions.json")):
        mp3 = caps.with_name(caps.name.replace(".captions.json", ".mp3"))
        if not mp3.exists():
            continue
        sound = None
        for w in json.loads(caps.read_text()):
            word = re.sub(r"[^a-z']", "", w["text"].lower())
            if word not in want:
                continue
            sound = sound or parselmouth.Sound(str(mp3))
            f3 = third_formant(sound, w["startMs"] / 1000, w["endMs"] / 1000, args.max_hz)
            if f3 is not None:
                rows.append({"file": mp3.stem, "word": word, "at": round(w["startMs"] / 1000, 2), "f3": round(f3), "flag": f3 < args.line})
    rows.sort(key=lambda r: r["f3"])
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{'LOW ' if r['flag'] else '    '}{r['f3']:5d} Hz  {r['word']:10} {r['file']} @ {r['at']} s")
        low = sum(r["flag"] for r in rows)
        print(f"{len(rows)} words measured, {low} under {args.line:.0f} Hz" + (": listen to those" if low else ""))
    return 1 if any(r["flag"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
