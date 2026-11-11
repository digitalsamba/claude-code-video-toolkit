#!/usr/bin/env python3
"""Contact sheet of a rendered video: one settled frame per scene, tiled into one PNG.

The design critique in the video-design skill looks at a video the way a juror looks at a page: every
scene at once, at rest. This finds the scene cuts (ffmpeg scene detection), takes the frame a moment
before each cut (after the scene's entrances have played, before the next scene starts), and tiles
them in order. Works on any MP4, whatever built it.

    uv run tools/contact_sheet.py out/video.mp4                      # -> out/video.sheet.png
    uv run tools/contact_sheet.py out/video.mp4 --at 4.2,13.8,31     # frames at chosen seconds
    uv run tools/contact_sheet.py out/video.mp4 --count 12           # evenly spaced, no detection
    uv run tools/contact_sheet.py out/video.mp4 --keep-frames        # also keep each frame as a PNG

Detection compares colour, not just brightness, so a cut between two fields of the same lightness is
found; a slow cross-fade or a camera move may not be: lower --threshold, or pass --at.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# The container's duration can run past the last video frame (audio is longer), so a frame is never asked for
# closer to the end than this; a seek beyond the last frame writes no image.
END_MARGIN = 0.5


def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def scene_cuts(path: Path, threshold: float, fps: int = 8) -> list[float]:
    """Timestamps (s) of scene boundaries: hard changes in the picture, colour included, and blanks.

    ffmpeg's own scene score is mostly brightness, so a cut between two fields of the same lightness
    (orange to sky blue) scores near zero. Instead compare tiny RGB thumbnails `fps` times a second: the
    mean absolute difference per channel, 0-1, above `threshold` is a cut. Scenes that share one ground
    often fade their content out and the next one in, with no hard change at all; a run of near-uniform
    frames (the bare ground) between two busier ones counts as a boundary too.
    """
    w, h = 32, 18
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf", f"fps={fps},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    size = w * h * 3
    data = proc.stdout
    frames = [data[i:i + size] for i in range(0, len(data) - size + 1, size)]

    def spread(f: bytes) -> float:
        # mean distance of each pixel from the frame's mean colour, 0-1: near 0 on a bare field
        n = len(f) // 3
        mean = [sum(f[c::3]) / n for c in range(3)]
        return sum(abs(f[i] - mean[i % 3]) for i in range(len(f))) / (len(f) * 255)

    cuts = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        diff = sum(abs(x - y) for x, y in zip(a, b)) / (size * 255)
        if diff > threshold:
            cuts.append(round(i / fps, 3))
    blank = [spread(f) < 0.01 for f in frames]
    i = 0
    while i < len(blank):
        if blank[i]:
            j = i
            while j + 1 < len(blank) and blank[j + 1]:
                j += 1
            if i > 0 and j < len(blank) - 1:  # a blank between content, not the very start or end
                # the scene ends where the blank starts; its fade-out runs into it, so step back past that
                cuts.append(round(i / fps, 3))
            i = j + 1
        else:
            i += 1
    return sorted(cuts)


def settled_times(cuts: list[float], duration: float, before: float, min_scene: float) -> list[float]:
    """One time per scene, `before` seconds ahead of the scene's end (its cut, or the video's end)."""
    bounds = [0.0] + [c for c in cuts if c > min_scene] + [duration]
    # merge cuts closer than min_scene (a flash, a fast transition) into one boundary
    merged = [bounds[0]]
    for b in bounds[1:]:
        if b - merged[-1] >= min_scene:
            merged.append(b)
        else:
            merged[-1] = b
    times = []
    for start, end in zip(merged, merged[1:]):
        t = max(start + (end - start) / 2, end - before)
        times.append(round(min(t, duration - END_MARGIN), 3))
    return times


def grab(path: Path, t: float, out: Path, width: int) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", f"scale={width}:-2", str(out)],
        check=True,
    )


def tile(frames: list[Path], cols: int, out: Path) -> None:
    n = len(frames)
    rows = math.ceil(n / cols)
    # pad the grid with black so xstack always gets a full rectangle
    inputs: list[str] = []
    for f in frames:
        inputs += ["-i", str(f)]
    pad = rows * cols - n
    layout = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        x = "+".join(["w0"] * c) or "0"
        y = "+".join(["h0"] * r) or "0"
        layout.append(f"{x}_{y}")
    chain = "".join(f"[{i}:v]" for i in range(n))
    if pad:
        chain = f"{chain}" + "".join(f"[b{i}]" for i in range(pad))
        pads = ";".join(f"[0:v]drawbox=c=black:t=fill[b{i}]" for i in range(pad))
        filt = f"{pads};{chain}xstack=inputs={rows * cols}:layout={'|'.join(layout)}"
    else:
        filt = f"{chain}xstack=inputs={n}:layout={'|'.join(layout)}" if n > 1 else "[0:v]null"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filt, str(out)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="sheet PNG (default: <video>.sheet.png)")
    ap.add_argument("--at", action="append", help="seconds instead of scene detection: comma-separated, "
                    "or repeat the flag")
    ap.add_argument("--count", type=int, help="this many evenly spaced frames instead of scene detection")
    ap.add_argument("--threshold", type=float, default=0.12, help="picture change between samples, 0-1 (default 0.12)")
    ap.add_argument("--before", type=float, default=0.8, help="seconds before each cut to sample (default 0.8)")
    ap.add_argument("--min-scene", type=float, default=1.5, help="ignore cuts closer together than this (s)")
    ap.add_argument("--cols", type=int, help="columns (default: 3, or 4 above 12 frames)")
    ap.add_argument("--width", type=int, default=640, help="width of each tile in px (default 640)")
    ap.add_argument("--keep-frames", action="store_true", help="also keep each frame beside the sheet")
    ap.add_argument("--json", action="store_true", help="print the times and paths as JSON")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"{tool} not found on PATH", file=sys.stderr)
            return 1
    if not args.video.exists():
        print(f"{args.video} not found", file=sys.stderr)
        return 1

    duration = ffprobe_duration(args.video)
    if args.at:
        times = [min(max(0.0, float(t)), round(duration - END_MARGIN, 3)) for t in ",".join(args.at).split(",") if t.strip()]
        how = "chosen times"
    elif args.count is not None:
        if args.count < 1:
            print("--count must be 1 or more", file=sys.stderr)
            return 1
        times = [round(duration * (i + 0.5) / args.count, 3) for i in range(args.count)]
        how = f"{args.count} evenly spaced"
    else:
        cuts = scene_cuts(args.video, args.threshold)
        times = settled_times(cuts, duration, args.before, args.min_scene)
        how = f"{len(times)} scenes (cut threshold {args.threshold})"
        if len(times) < 3 and duration > 12:
            # soft fades on one ground leave no cuts to find: sample evenly instead, about every 5s
            n = max(4, min(12, round(duration / 5)))
            times = [round(duration * (i + 0.5) / n, 3) for i in range(n)]
            how = f"no clear cuts found, so {n} evenly spaced frames (pass --at for exact scenes)"

    out = args.output or args.video.with_suffix(".sheet.png")
    cols = args.cols or (4 if len(times) > 12 else 3)
    with tempfile.TemporaryDirectory() as tmp:
        frames = []
        for i, t in enumerate(times):
            f = Path(tmp) / f"{i:03d}.png"
            grab(args.video, t, f, args.width)
            frames.append(f)
        tile(frames, min(cols, len(frames)), out)
        kept = []
        if args.keep_frames:
            folder = out.with_suffix("")
            folder.mkdir(parents=True, exist_ok=True)
            for f, t in zip(frames, times):
                dest = folder / f"{t:08.3f}s.png"
                shutil.copy(f, dest)
                kept.append(str(dest))

    if args.json:
        print(json.dumps({"video": str(args.video), "sheet": str(out), "times": times, "frames": kept}, indent=1))
    else:
        print(f"{args.video.name}: {how} -> {out}")
        print("  at " + ", ".join(f"{t:.1f}s" for t in times))
    return 0


if __name__ == "__main__":
    sys.exit(main())
