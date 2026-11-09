#!/usr/bin/env python3
"""
Check that a sound effect lands on the movement it belongs to.

A sound laid over generated footage (a sneeze over an LTX clip, a thud when a jacket lands) is placed at a frame
by hand, and the movement in the clip can come earlier or later than it looks in a quick preview. This measures
both from the render: the sound's peak (its loudest 50 ms, from the effect file at its placed time) and the
biggest movement in the screen region it belongs to, nearby. It reports the offset and the shift that would line
them up. People notice sound ahead of picture from ~45 ms and behind it from ~125 ms; the default tolerance is
0.1 s.

  # one cue: the sneeze placed at 34.10 s, over Theo's tile
  uv run tools/sync_check.py --video out/video.mp4 --sfx public/audio/sfx/sneeze.mp3 --at 34.10 \\
      --region 1000:700:460:150

  # before rendering: where in a clip should the sound start? (prevention: place it, don't guess)
  uv run tools/sync_check.py --clip public/people/theo-sneeze.mp4 --sfx public/audio/sfx/sneeze.mp3 --window 2.5:4.5

  # every cue in a JSON list: [{"name": "sneeze", "sfx": "...", "at": 34.1, "region": "w:h:x:y", "kind": "peak"}, ...]
  # (optional "window": "a:b" in render seconds, where the movement is; default: while the sound plays, +-0.5 s)
  uv run tools/sync_check.py --video out/video.mp4 --cues sync-cues.json

Regions are ffmpeg crop strings (w:h:x:y) in the video's pixels. Exit 1 if any cue is out by more than
--tolerance, so it can gate a delivery.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

WIN = 0.05        # audio window, seconds
FPS = 30          # picture sampling
SEARCH = 0.5      # seconds of picture searched either side of the sound


def sound_peak(sfx: Path) -> tuple[float, float]:
    """(seconds into the file of its loudest 50 ms, length of the file)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(sfx), "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    n = int(8000 * WIN)
    rms = np.array([np.sqrt((a[i:i + n] ** 2).mean()) for i in range(0, max(1, len(a) - n), n)])
    return float(rms.argmax() * WIN + WIN / 2), len(a) / 8000


def motion_peak(video: Path, region: str, t0: float, t1: float, kind: str = "peak") -> tuple[float, float]:
    """The moment in the region between t0 and t1 that the sound belongs to, and the movement's size.
    kind 'peak': the biggest frame-to-frame change (a sneeze, a laugh, a jingle while a badge is lifted).
    kind 'impact': where the movement stops hardest (a jacket landing, a stamp), the sharpest fall after a peak.
    kind 'onset': where a sustained movement starts (a badge lifted slowly, a door swinging open): the first frame
    at half the window's biggest change. 'peak' picks noise on a movement with no single spike."""
    t0 = max(0.0, t0)
    s = 64
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(video),
                          "-vf", f"crop={region},fps={FPS},scale={s}:{s},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    n = len(raw) // (s * s)
    if n < 3:
        return t0, 0.0
    fr = np.frombuffer(raw[: n * s * s], np.uint8).reshape(n, s * s).astype(np.float32)
    m = np.abs(np.diff(fr, axis=0)).mean(axis=1)
    # a 24 fps clip in a 30 fps video repeats every fifth frame: a zero there is a duplicate, not a stop
    dup = np.zeros(len(m), bool)
    dup[1:-1] = (m[1:-1] < 0.02) & (m[:-2] > 0.3) & (m[2:] > 0.3)   # one still frame between two moving ones
    m = m.copy()
    m[dup] = (np.roll(m, 1)[dup] + np.roll(m, -1)[dup]) / 2
    m = np.convolve(m, np.ones(3) / 3, mode="same")  # one noisy frame isn't a movement
    if kind == "onset":
        i = int(np.argmax(m >= 0.5 * m.max()))
        return t0 + (i + 1) / FPS, float(m.max())
    if kind == "impact":
        fall = m[:-3] - m[3:]                        # drop over 0.1 s
        i = int(fall.argmax()) + 3
        return t0 + (i + 1) / FPS, float(m.max())
    i = int(m.argmax())
    return t0 + (i + 1) / FPS, float(m[i])


def check(video: Path, cue: dict, tol: float) -> dict:
    peak, length = sound_peak(Path(cue["sfx"]))
    heard = cue["at"] + peak
    w0, w1 = ((float(x) for x in cue["window"].split(":")) if cue.get("window")
              else (cue["at"] - SEARCH, cue["at"] + length + SEARCH))
    seen, size = motion_peak(video, cue["region"], w0, w1, cue.get("kind", "peak"))
    off = seen - heard
    return {"name": cue.get("name", Path(cue["sfx"]).stem), "sound_peak": round(heard, 2), "movement_peak": round(seen, 2),
            "offset": round(off, 2), "shift_sound_by": round(off, 2), "ok": abs(off) <= tol or size < 0.5,
            "movement": round(size, 2)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, help="a render to check")
    p.add_argument("--clip", type=Path, help="a clip to place a sound on, before rendering")
    p.add_argument("--window", help="with --clip: seconds a:b of the clip where the movement is (default: all)")
    p.add_argument("--rate", type=float, default=1.0, help="with --clip: the clip's playback rate in the video")
    p.add_argument("--trim", type=int, default=0,
                   help="with --clip: frames skipped at the clip's start (LivingPhoto trim, 30 fps)")
    p.add_argument("--sfx", type=Path)
    p.add_argument("--at", type=float, help="seconds: where the effect starts in the video")
    p.add_argument("--region", help="ffmpeg crop w:h:x:y of the movement the sound belongs to")
    p.add_argument("--name")
    p.add_argument("--kind", default="peak", choices=["peak", "impact", "onset"],
                   help="peak: the sound goes on the biggest movement; impact: where the movement stops; "
                        "onset: where a sustained movement starts")
    p.add_argument("--cues", type=Path, help="JSON list of {name, sfx, at, region}")
    p.add_argument("--tolerance", type=float, default=0.1)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()

    if a.clip:
        if not a.sfx:
            p.error("--clip needs --sfx")
        peak, _ = sound_peak(a.sfx)
        w0, w1 = (float(x) for x in a.window.split(":")) if a.window else (0.0, 600.0)
        info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-select_streams",
                               "v:0", "-of", "csv=p=0", str(a.clip)], capture_output=True, text=True).stdout.strip()
        w, h = info.split(",")
        moment, size = motion_peak(a.clip, a.region or f"{w}:{h}:0:0", w0, w1, a.kind)
        start = (moment - a.trim / 30) / a.rate - peak   # in video seconds after the clip's shot starts
        print(json.dumps({"clip": str(a.clip), "movement_at": round(moment, 2), "sound_peak_in_file": round(peak, 2),
                          "start_sound_after_clip_s": round(start, 2), "start_sound_after_clip_frames_30fps": round(start * 30)}))
        return 0
    if not a.video:
        p.error("give --video (check a render) or --clip (place a sound before rendering)")
    if a.cues:
        cues = json.loads(a.cues.read_text())
        # a cue's sfx path may be relative to the working directory, the cue file's folder, or a folder above it
        # (the toolkit root), so the same cue file works from the project or from the root
        bases = [Path.cwd(), *a.cues.resolve().parents]
        for c in cues:
            found = next((b / c["sfx"] for b in bases if (b / c["sfx"]).exists()), None)
            if found is None:
                p.error(f"cue {c.get('name', '?')}: sound file {c['sfx']} not found from here or from {a.cues}")
            c["sfx"] = str(found)
    elif a.sfx and a.at is not None and a.region:
        cues = [{"name": a.name or a.sfx.stem, "sfx": str(a.sfx), "at": a.at, "region": a.region, "kind": a.kind}]
    else:
        p.error("give --cues, or --sfx, --at and --region")
    results = [check(a.video, c, a.tolerance) for c in cues]
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            mark = "ok  " if r["ok"] else "OUT "
            note = "" if r["ok"] else f"  -> move the sound {'later' if r['offset'] > 0 else 'earlier'} by {abs(r['offset']):.2f}s"
            if r["movement"] < 0.5:
                note = "  (no clear movement in the region: check the region)"
            print(f"{mark}{r['name']:<16} sound peak {r['sound_peak']:7.2f}s  movement peak {r['movement_peak']:7.2f}s  "
                  f"offset {r['offset']:+.2f}s{note}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
