#!/usr/bin/env python3
"""
Extend an LTX image-to-video clip by continuing it from one of its own frames.

An LTX clip is ~5 s; a shot is often longer. Holding the last frame reads as a pause, and playing the tail back
and forth (a ping-pong) reads as a loop. This continues the motion instead: it takes the frame at --at (the
clip's last good frame, before any late cut to a stranger or smear), runs ltx2.py image-to-video from it, and
joins the original up to that frame with the new clip, with a short cross-fade at the seam. Repeat with --times
for longer shots. Each extension costs one LTX clip (~$0.08 on Modal).

Examples:
  # one extension from 4.8 s: a ~10 s clip
  uv run tools/ltx_extend.py --input people/rob.mp4 --at 4.8 --output people/rob-long.mp4 \\
      --prompt "A fixed view that never moves: the framing stays exactly the same, no zoom, no pull back, no cuts. ... the same man keeps laughing ... no text"

  # from a clip's last good frame (LivingPhoto hold={105} at 30 fps is 3.5 s), two extensions
  uv run tools/ltx_extend.py --input dana.mp4 --at 3.5 --times 2 --prompt "..." --output dana-long.mp4

Never continue from a close-up frame (LTX invents a new scene), and don't say "tripod" (it draws one).
Check the result with tools/contact_sheet.py like any LTX clip: a continuation can still cut away at its end.
"""

import argparse
import json
import subprocess
import sys
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den, "duration": float(j["format"]["duration"])}


def extend_once(src: Path, at: float, prompt: str, seed: int, out: Path, crossfade: float, frames: int,
                cloud: str, work: Path, n: int) -> None:
    info = probe(src)
    at = min(at, info["duration"] - 1 / info["fps"])
    still = work / f"seam{n}.png"
    # the last frame before `at` (the original plays up to it): seeking exactly to the clip's end yields no frame
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, at - 0.5):.3f}", "-i", str(src), "-t",
                    f"{min(0.5, at):.3f}", "-update", "1", str(still)], check=True)
    if not still.exists():
        raise SystemExit(f"Error: no frame at {at:.2f}s of {src}")
    key = hashlib.sha1(f"{src.name}|{at:.3f}|{seed}|{frames}|{prompt}".encode()).hexdigest()[:10]
    cont = work / f"cont{n}-{key}.mp4"
    if cont.exists():   # paid for on an earlier run that failed later: reuse it
        print(f"-> reusing {cont}", file=sys.stderr)
        return join(src, at, cont, out, crossfade, info)
    cmd = [sys.executable, str(ROOT / "tools/ltx2.py"), "--cloud", cloud, "--no-open", "--input", str(still),
           "--width", str(info["width"]), "--height", str(info["height"]), "--num-frames", str(frames),
           "--seed", str(seed), "--prompt", prompt, "--output", str(cont)]
    print(f"-> continuing from {at:.2f}s ({n}) ...", file=sys.stderr)
    subprocess.run(cmd, check=True, stdout=sys.stderr)
    join(src, at, cont, out, crossfade, info)


def join(src: Path, at: float, cont: Path, out: Path, crossfade: float, info: dict) -> None:
    fps = info["fps"]
    w, h = info["width"], info["height"]
    # the original up to the seam, then the continuation (its first frame is the seam frame again, so skip it)
    filt = (f"[0:v]trim=0:{at:.3f},setpts=PTS-STARTPTS,fps={fps},scale={w}:{h},format=yuv420p[a];"
            f"[1:v]trim=start_frame=1,setpts=PTS-STARTPTS,fps={fps},scale={w}:{h},format=yuv420p[b];"
            f"[a][b]xfade=transition=fade:duration={crossfade}:offset={max(0.0, at - crossfade):.3f}[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(cont), "-filter_complex", filt,
                    "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)], check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="the LTX clip to extend")
    p.add_argument("--at", type=float, help="seconds: continue from this frame (default: the clip's last frame)")
    p.add_argument("--prompt", required=True, help="the continuation prompt (same locked-off framing as the clip)")
    p.add_argument("--times", type=int, default=1, help="extensions to chain (each continues from the new end)")
    p.add_argument("--frames", type=int, default=121, help="frames per continuation (LTX: 8n+1, default 121)")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--crossfade", type=float, default=0.12, help="seconds of cross-fade at each seam")
    p.add_argument("--cloud", default="modal", choices=["modal"])
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()

    if not a.input.exists():
        print(f"Error: {a.input} not found", file=sys.stderr)
        return 1
    # kept beside the output until the whole chain succeeds, so a failed run doesn't lose the clips it paid for
    work = a.output.parent / f".{a.output.stem}-extend"
    work.mkdir(parents=True, exist_ok=True)
    src = a.input
    at = a.at if a.at is not None else probe(src)["duration"]
    for n in range(1, a.times + 1):
        dst = a.output if n == a.times else work / f"step{n}.mp4"
        extend_once(src, at, a.prompt, a.seed + n - 1, dst, a.crossfade, a.frames, a.cloud, work, n)
        src = dst
        at = probe(src)["duration"]  # later links continue from the new end
    shutil.rmtree(work, ignore_errors=True)
    info = probe(a.output)
    print(json.dumps({"output": str(a.output), "duration": round(info["duration"], 2),
                      "frames_at_30fps": round(info["duration"] * 30)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
