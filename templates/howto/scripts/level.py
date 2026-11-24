"""Level every narration phrase to the same loudness, in place, so no line jumps out of the episode.

    uv run scripts/level.py                     # public/audio/scenes/*.mp3
    uv run scripts/level.py --episode ep1       # public/audio/scenes/ep1/*.mp3, in a project that keeps several
    uv run scripts/level.py --check             # print each phrase's loudness, change nothing

A voice made a sentence at a time, cloned voices most of all, comes out a few dB louder or softer from line to
line. A phrase is one to four seconds long, too short for ffmpeg's one-pass loudnorm to settle, so each is
measured (ebur128) and given a plain gain, with a limiter for the peaks. The default target, -16 LUFS, is the
toolkit's scene level; the render is then brought to -14 LUFS by `qc.py --render --fix`.

Run it after voicing and before scripts/timing.py, so the timing is measured from the files the render plays.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOSE = 0.3  # within this many dB of the target a phrase is left alone


def loudness(path: Path) -> float:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    if not found:
        raise SystemExit(f"{path.name}: ffmpeg measured no loudness (is it audio?)")
    return float(found[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", help="In a project that keeps several episodes, the id of one")
    ap.add_argument("--target", type=float, default=-16.0, help="LUFS (default -16)")
    ap.add_argument("--check", action="store_true", help="Measure only")
    a = ap.parse_args()

    folder = ROOT / "public" / "audio" / "scenes" / (a.episode or "")
    phrases = sorted(p for p in folder.glob("*.mp3") if not p.name.endswith(".lvl.mp3"))
    if not phrases:
        raise SystemExit(f"no phrases in {folder.relative_to(ROOT)}")
    for mp3 in phrases:
        before = loudness(mp3)
        gain = a.target - before
        if a.check or abs(gain) < CLOSE:
            print(f"{mp3.name:32} {before:6.1f} LUFS{'' if a.check else '  ok'}")
            continue
        tmp = mp3.with_suffix(".lvl.mp3")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-af", f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false", "-c:a", "libmp3lame", "-q:a", "2", str(tmp)],
            check=True,
        )
        tmp.replace(mp3)
        print(f"{mp3.name:32} {before:6.1f} -> {loudness(mp3):6.1f} LUFS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
