"""Lay the narration out on the timeline: writes src/timing.json.

Run after voicing or re-voicing any phrase, and after captions.py (the word times come from the
`.captions.json` beside each MP3):

    uv run templates/howto/scripts/timing.py            # from the toolkit root, or
    uv run scripts/timing.py                             # from a project made from this template

Phrases are `public/audio/scenes/<NN><letter>_<scene>.mp3` (e.g. 04b_spent.mp3): the scene id is the part
after the underscore, and phrases play in file-name order. Within a scene, LEAD frames come before the
first phrase, GAP between phrases and TAIL after the last. `public/audio/layout.json` can override them per
scene, e.g. {"open": {"lead": 72}, "spent": {"gap": 6}}. The scene components never hold a frame number for
narration; they look words up with cue(), so this file is the only place timing lives.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "public" / "audio" / "scenes"
LAYOUT = ROOT / "public" / "audio" / "layout.json"
OUT = ROOT / "src" / "timing.json"
FPS = 30
DEFAULTS = {"lead": 18, "gap": 12, "tail": 30}
NAME = re.compile(r"^(\d+)([a-z]?)_([a-z0-9-]+)$")


def frames(path: Path) -> int:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    )
    return round(float(out) * FPS)


def main() -> int:
    layout = json.loads(LAYOUT.read_text()) if LAYOUT.exists() else {}
    phrases: dict[str, list[Path]] = {}
    for mp3 in sorted(SCENES.glob("*.mp3")):
        m = NAME.match(mp3.stem)
        if not m:
            print(f"skipped {mp3.name}: not <NN><letter>_<scene>.mp3", file=sys.stderr)
            continue
        phrases.setdefault(m.group(3), []).append(mp3)

    out: dict = {"fps": FPS, "scenes": {}}
    for scene, files in phrases.items():
        cfg = {**DEFAULTS, **layout.get(scene, {})}
        t = cfg["lead"]
        lines = []
        for mp3 in files:
            caps = mp3.with_suffix(".captions.json")
            if not caps.exists():
                print(f"missing {caps.name}: run tools/captions.py on {SCENES}", file=sys.stderr)
                return 1
            words = [
                {"text": w["text"], "startMs": w["startMs"], "endMs": w["endMs"]}
                for w in json.loads(caps.read_text())
            ]
            n = frames(mp3)
            lines.append({"file": mp3.stem, "at": t, "frames": n, "words": words})
            t += n + cfg["gap"]
        out["scenes"][scene] = {"frames": t - cfg["gap"] + cfg["tail"], "phrases": lines}

    OUT.write_text(json.dumps(out, indent=1))
    total = sum(s["frames"] for s in out["scenes"].values())
    print(f"{OUT.relative_to(ROOT)}: {len(out['scenes'])} scenes, {total / FPS:.1f} s of narrated scenes")
    for name, s in out["scenes"].items():
        print(f"  {name:10} {s['frames'] / FPS:5.1f} s  {len(s['phrases'])} phrases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
