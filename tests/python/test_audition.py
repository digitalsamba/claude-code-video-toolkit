"""Tests for tools/audition.py: each take mixed under a synthetic picture. No keys, no network; needs ffmpeg.

    uv run python -m unittest tests/python/test_audition.py -v
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import audition as au  # noqa: E402


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def band(path: Path, freq: float) -> float:
    """Share of the spectrum's energy within 30 Hz of freq."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 8000)
    return float(spec[np.abs(f - freq) < 30].sum() / spec.sum())


def lufs(path: Path) -> float:
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=framelog=quiet",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def run(*args: str) -> dict:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["audition.py", *args, "--json"]), redirect_stdout(out), redirect_stderr(err):
        au.main()
    return json.loads(out.getvalue())


class Audition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = Path(self.tmp.name)
        # a 6s picture whose own audio (the "narration") is a 1 kHz tone
        ff("-f", "lavfi", "-i", "color=c=0x56140F:s=320x568:r=30:d=6", "-f", "lavfi",
           "-i", "sine=frequency=1000:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", str(d / "video.mp4"))
        # two 8s takes: 220 Hz and 330 Hz
        for name, hz in (("take_a", 220), ("take_b", 330)):
            ff("-f", "lavfi", "-i", f"sine=frequency={hz}:duration=8", str(d / f"{name}.wav"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_render_per_take_trimmed_and_normalised(self):
        r = run("--video", str(self.d / "video.mp4"), "--music", str(self.d / "take_a.wav"), str(self.d / "take_b.wav"),
                "--volume", "0.5", "--fade-out", "1")
        self.assertEqual(len(r["renders"]), 2)
        for x in r["renders"]:
            out = Path(x["render"])
            self.assertEqual(out.parent.name, "audition")
            dur, has_audio = au.probe(out)
            self.assertTrue(has_audio)
            self.assertAlmostEqual(dur, 6.0, delta=0.15)
            self.assertAlmostEqual(lufs(out), -14.0, delta=1.0)
        self.assertTrue(r["renders"][0]["render"].endswith("video.take_a.mp4"))

    def test_keeps_the_videos_own_audio_unless_dropped(self):
        kept = run("--video", str(self.d / "video.mp4"), "--music", str(self.d / "take_a.wav"))
        out = Path(kept["renders"][0]["render"])
        self.assertTrue(kept["kept_video_audio"])
        self.assertGreater(band(out, 1000), 0.2)
        self.assertGreater(band(out, 220), 0.2)
        dropped = run("--video", str(self.d / "video.mp4"), "--music", str(self.d / "take_a.wav"), "--drop-audio",
                      "--out-dir", str(self.d / "dropped"))
        out = Path(dropped["renders"][0]["render"])
        self.assertFalse(dropped["kept_video_audio"])
        self.assertLess(band(out, 1000), 0.01)
        self.assertGreater(band(out, 220), 0.8)

    def test_recommend_names_a_take_and_its_render(self):
        r = run("--video", str(self.d / "video.mp4"), "--music", str(self.d / "take_a.wav"), str(self.d / "take_b.wav"),
                "--recommend", "--under", "text")
        rec = r["recommended"]
        self.assertIn(rec["take"], [str(self.d / "take_a.wav"), str(self.d / "take_b.wav")])
        self.assertTrue(Path(rec["render"]).exists())


if __name__ == "__main__":
    unittest.main()
