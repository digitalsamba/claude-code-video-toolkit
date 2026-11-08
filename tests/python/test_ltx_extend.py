"""ltx_extend.py without Modal: the seam-frame grab, and the join with a fake continuation clip.

The source clip's frames are flat grey at a level that counts up with the frame index, so a grabbed frame says
which frame it was. The fake ltx2.py writes a clip that starts on the seam image and then goes white, and logs
each call, so a test can see what was asked and make a later link fail.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ltx_extend  # noqa: E402

FPS = 30
SIZE = 64
STEP = 4  # grey level per frame: frame i is 4*i

FAKE_LTX = r'''
import argparse, json, os, subprocess, sys
from pathlib import Path
p = argparse.ArgumentParser()
for f in ("--cloud", "--input", "--width", "--height", "--num-frames", "--seed", "--prompt", "--output"):
    p.add_argument(f)
p.add_argument("--no-open", action="store_true")
a = p.parse_args()
log = Path(os.environ["FAKE_LTX_LOG"])
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({"input": a.input, "seed": a.seed, "frames": a.num_frames})
log.write_text(json.dumps(calls))
if len(calls) >= int(os.environ.get("FAKE_LTX_FAIL_AT", "999")):
    sys.exit(3)
n = int(a.num_frames)
# first frame is the seam image, the rest white
subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-t", f"{1/30:.4f}",
                "-i", a.input, "-f", "lavfi", "-i", f"color=white:s={a.width}x{a.height}:r=30:d={(n-1)/30:.4f}",
                "-filter_complex", "[0:v]scale={w}:{h},fps=30,format=yuv420p[a];[1:v]format=yuv420p[b];[a][b]concat=n=2:v=1[v]".format(w=a.width, h=a.height),
                "-map", "[v]", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", a.output], check=True)
'''


def make_ramp(path: Path, frames: int) -> None:
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s",
                             f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-qp", "0",
                             "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(frames):
        proc.stdin.write(np.full((SIZE, SIZE), STEP * i, dtype=np.uint8).tobytes())
    proc.stdin.close()
    assert proc.wait() == 0


def grey(path: Path) -> float:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return float(np.frombuffer(out, dtype=np.uint8).mean())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class LtxExtendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "clip.mp4"
        make_ramp(self.src, 60)  # 2 s
        fake_root = self.tmp / "fake"
        (fake_root / "tools").mkdir(parents=True)
        (fake_root / "tools" / "ltx2.py").write_text(FAKE_LTX)
        self.log = self.tmp / "calls.json"
        os.environ["FAKE_LTX_LOG"] = str(self.log)
        os.environ.pop("FAKE_LTX_FAIL_AT", None)
        self._root = ltx_extend.ROOT
        ltx_extend.ROOT = fake_root

    def tearDown(self):
        ltx_extend.ROOT = self._root
        os.environ.pop("FAKE_LTX_FAIL_AT", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_main(self, *args: str) -> int:
        old = sys.argv
        sys.argv = ["ltx_extend.py", "--input", str(self.src), "--prompt", "same framing", *args]
        try:
            return ltx_extend.main()
        finally:
            sys.argv = old

    def calls(self) -> list:
        return json.loads(self.log.read_text()) if self.log.exists() else []

    def test_seam_is_the_frame_before_at(self):
        work = self.tmp / "work"
        work.mkdir()
        out = self.tmp / "out.mp4"
        ltx_extend.extend_once(self.src, 1.0, "p", 5, out, 0.12, 25, "modal", work, 1)
        # frame 29 is the last one shown before 1.0 s
        self.assertAlmostEqual(grey(work / "seam1.png"), STEP * 29, delta=3)

    def test_seam_at_the_clip_end_still_grabs_a_frame(self):
        # the --times 2 bug: seeking to the very last frame wrote no image. `at` clamps to the last frame's start
        # (1.967 s), so the original is trimmed to frames 0-58 and the seam is frame 58, the last one it shows
        work = self.tmp / "work"
        work.mkdir()
        out = self.tmp / "out.mp4"
        ltx_extend.extend_once(self.src, ltx_extend.probe(self.src)["duration"], "p", 5, out, 0.12, 25,
                               "modal", work, 1)
        self.assertAlmostEqual(grey(work / "seam1.png"), STEP * 58, delta=3)

    def test_join_length_and_continuation_follows_the_seam(self):
        out = self.tmp / "long.mp4"
        self.assertEqual(self.run_main("--at", "1.0", "--frames", "31", "--output", str(out)), 0)
        # 1.0 s of original + 30 new frames (the seam duplicate dropped) - the cross-fade
        self.assertAlmostEqual(ltx_extend.probe(out)["duration"], 1.0 + 30 / FPS - 0.12, delta=0.07)
        frames = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                                capture_output=True, check=True).stdout
        means = np.frombuffer(frames, dtype=np.uint8).reshape(-1, SIZE * SIZE).mean(axis=1)
        self.assertAlmostEqual(means[0], 0, delta=3)       # starts on the original
        self.assertAlmostEqual(means[20], STEP * 20, delta=3)
        self.assertGreater(means[-1], 245)                 # ends on the continuation
        self.assertFalse((self.tmp / ".long-extend").exists(), "work dir is removed on success")

    def test_chain_continues_from_each_new_end(self):
        out = self.tmp / "chain.mp4"
        self.assertEqual(self.run_main("--times", "2", "--frames", "25", "--output", str(out)), 0)
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual([c["seed"] for c in calls], ["5", "6"])
        self.assertAlmostEqual(ltx_extend.probe(out)["duration"], 2.0 + 2 * (24 / FPS - 0.12), delta=0.1)

    def test_failed_chain_keeps_paid_clips_and_rerun_reuses_them(self):
        out = self.tmp / "chain.mp4"
        os.environ["FAKE_LTX_FAIL_AT"] = "2"
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main("--times", "2", "--frames", "25", "--output", str(out))
        work = self.tmp / ".chain-extend"
        self.assertEqual(len(list(work.glob("cont1-*.mp4"))), 1, "the first paid clip is kept")
        os.environ.pop("FAKE_LTX_FAIL_AT")
        self.assertEqual(self.run_main("--times", "2", "--frames", "25", "--output", str(out)), 0)
        # 2 calls from the failed run, 1 new one: link 1 was reused
        self.assertEqual(len(self.calls()), 3)


if __name__ == "__main__":
    unittest.main()
