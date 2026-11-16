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

A project with several episodes keeps each one's phrases in `public/audio/scenes/<id>/` (the episode's `id` in
src/episode.ts); they are written under `episodes.<id>`, and layout.json takes them as {"<id>": {"open": {...}}}.
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


def lay(scenes: Path, layout: dict) -> dict | None:
    """The scenes of one folder of phrases, each phrase placed in its scene."""
    phrases: dict[str, list[Path]] = {}
    for mp3 in sorted(scenes.glob("*.mp3")):
        m = NAME.match(mp3.stem)
        if not m:
            print(f"skipped {mp3.name}: not <NN><letter>_<scene>.mp3", file=sys.stderr)
            continue
        phrases.setdefault(m.group(3), []).append(mp3)

    out: dict = {}
    for scene, files in phrases.items():
        cfg = {**DEFAULTS, **layout.get(scene, {})}
        t = cfg["lead"]
        lines = []
        for mp3 in files:
            caps = mp3.with_suffix(".captions.json")
            if not caps.exists():
                print(f"missing {caps.name}: run tools/captions.py on {scenes}", file=sys.stderr)
                return None
            words = [
                {"text": w["text"], "startMs": w["startMs"], "endMs": w["endMs"]}
                for w in json.loads(caps.read_text())
            ]
            n = frames(mp3)
            lines.append({"file": mp3.stem, "at": t, "frames": n, "words": words})
            t += n + cfg["gap"]
        out[scene] = {"frames": t - cfg["gap"] + cfg["tail"], "phrases": lines}
    return out


def report(name: str, scenes: dict) -> None:
    total = sum(s["frames"] for s in scenes.values())
    print(f"{name}: {len(scenes)} scenes, {total / FPS:.1f} s of narrated scenes")
    for scene, s in scenes.items():
        print(f"  {scene:10} {s['frames'] / FPS:5.1f} s  {len(s['phrases'])} phrases")


def main() -> int:
    layout = json.loads(LAYOUT.read_text()) if LAYOUT.exists() else {}
    top = lay(SCENES, layout)
    if top is None:
        return 1
    out: dict = {"fps": FPS, "scenes": top}
    # a project with several episodes keeps each one's phrases in a folder named by its id
    for folder in sorted(d for d in SCENES.iterdir() if d.is_dir() and any(d.glob("*.mp3"))) if SCENES.exists() else []:
        scenes = lay(folder, layout.get(folder.name, {}))
        if scenes is None:
            return 1
        out.setdefault("episodes", {})[folder.name] = {"scenes": scenes}

    OUT.write_text(json.dumps(out, indent=1))
    if top or "episodes" not in out:
        report(str(OUT.relative_to(ROOT)), top)
    for name, ep in out.get("episodes", {}).items():
        report(f"{OUT.relative_to(ROOT)} episode {name}", ep["scenes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
