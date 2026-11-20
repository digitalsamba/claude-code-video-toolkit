#!/usr/bin/env python3
"""Render both idents in both shapes and collect them as a brand's ident folder.

    uv run examples/blender-ident/build.py                    # render everything (about 35 min on an M4), then collect
    uv run examples/blender-ident/build.py --collect-only     # collect what examples/blender-ident/out/ already has
    uv run examples/blender-ident/build.py --to brands/my-brand/ident

Each clip is one `tools/blender_render.py render --clips` run of clap.json or key.json; this only names the
files and writes ident.json, which lists them with their length and the frame of the hit. Needs Blender.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SHAPES = {"16x9": "1920x1080", "9x16": "1080x1920"}
# (spec, shape, transparent)
RUNS = [(spec, shape, False) for spec in ("clap", "key") for shape in SHAPES] + [("clap", shape, True) for shape in SHAPES]


DEFAULT = "clap"  # the ident a template plays when it is not told which


def rest_frame(clip: Path, after: int) -> int | None:
    """The first frame after `after` from which the picture no longer changes: where a title may come in."""
    import numpy as np
    w, h = 96, 96
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(float)
    if len(frames) < 2:
        return None
    change = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))  # change[i]: frame i to i + 1
    still = change < max(0.15, change.max() * 0.03)
    for f in range(after, len(frames)):
        if still[f:].all():
            return f
    return None


def out_dir(spec, shape, alpha):
    return HERE / "out" / f"{spec}-{shape}{'-alpha' if alpha else ''}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", type=Path, default=ROOT / "brands" / "video-toolkit" / "ident")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    failed = []
    if not a.collect_only:
        for spec, shape, alpha in RUNS:
            cmd = [sys.executable, str(ROOT / "tools" / "blender_render.py"), "render", str(HERE / f"{spec}.json"), "--clips",
                   "--alpha" if alpha else "--with-sound", "--res", SHAPES[shape], "--out", str(out_dir(spec, shape, alpha))]
            if subprocess.run(cmd).returncode:
                failed.append(f"{spec} {shape}{' alpha' if alpha else ''}")
    a.to.mkdir(parents=True, exist_ok=True)
    idents = {}
    for spec, shape, alpha in RUNS:
        folder = out_dir(spec, shape, alpha) / "clips"
        if not (folder / "clips.json").exists():
            print(f"not rendered yet: {folder}")
            continue
        events = json.loads((folder.parent / "events.json").read_text())
        checks = json.loads((folder.parent / "render_checks.json").read_text()) if (folder.parent / "render_checks.json").exists() else {}
        shots = {s["id"]: s for s in json.loads((folder.parent / "checks.json").read_text())["shots"]}
        for clip in json.loads((folder / "clips.json").read_text())["clips"]:
            part = clip["id"]  # open or close
            name = f"{spec}-{part}-{shape}{'-alpha' if alpha else ''}{Path(clip['file']).suffix}"
            shutil.copy(folder / clip["file"], a.to / name)
            entry = idents.setdefault(spec, {}).setdefault(part, {"seconds": clip["seconds"], "frames": clip["frames"],
                                                                  "fps": clip["fps"], "files": {}})
            entry["files"][f"{shape}{'-alpha' if alpha else ''}"] = name
            if alpha:  # a transparent clip keeps the colours of the stage it was lit on
                stage = json.loads((HERE / f"{spec}.json").read_text()).get("set", {}).get("stage", "sweep")
                entry["alpha_ground"] = "dark" if stage == "dark" else "light"
            if clip.get("sound") and not alpha:
                shutil.copy(folder / clip["sound"], a.to / f"{spec}-{part}.wav")
                entry["sound"] = f"{spec}-{part}.wav"
            hits = [e["frame"] for e in events["events"] if e.get("hit") and e["shot"] == part]
            if hits:  # as a frame of the clip, counting from 0
                entry["hit_frame"] = hits[0] - shots[part]["frames"][0]
                if not alpha and shape == "16x9":
                    rest = rest_frame(a.to / name, entry["hit_frame"])
                    if rest is not None:
                        entry["title_frame"] = rest
            seen = [h for h in checks.get("hits", []) if h["shot"] == part]
            if seen and not alpha:
                entry["hit_checked"] = all(h["passed"] for h in seen)
    if not idents:  # nothing rendered here: leave the manifest that is there alone
        print("nothing to collect: render first (run without --collect-only)")
        return 1
    (a.to / "ident.json").write_text(json.dumps({"made_by": "examples/blender-ident/build.py", "default": DEFAULT, "idents": idents}, indent=2) + "\n")
    print(f"collected {sum(len(p['files']) for i in idents.values() for p in i.values())} clips in {a.to}")
    for f in failed:
        print(f"  FAILED its checks: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
