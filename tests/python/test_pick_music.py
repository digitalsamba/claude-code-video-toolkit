"""Tests for tools/pick_music.py on synthetic takes with known defects. No keys, no network; needs ffmpeg.

    uv run python -m unittest tests/python/test_pick_music.py -v
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import pick_music as pm  # noqa: E402

SR = pm.SR


def plucks(seconds: float, every: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    """Decaying notes every `every` seconds over a quiet pad, like a harp bed."""
    t = np.arange(int(seconds * SR)) / SR
    pad = 0.05 * np.sin(2 * np.pi * 110 * t)
    x = pad.copy()
    for start in np.arange(0, seconds, every):
        n = t >= start
        x[n] += amp * np.sin(2 * np.pi * freq * (t[n] - start)) * np.exp(-(t[n] - start) * 6)
    return x


def fade(x: np.ndarray, seconds: float = 2.0) -> np.ndarray:
    n = int(seconds * SR)
    y = x.copy()
    y[-n:] *= np.linspace(1, 0, n) ** 2
    return y


def write(path: Path, x: np.ndarray) -> Path:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def run(*args: str) -> tuple[int, dict]:
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(sys, "argv", ["pick_music.py", *args, "--json"]), redirect_stdout(out), redirect_stderr(err):
        rc = pm.main()
    return rc, json.loads(out.getvalue())


class PickMusic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def take(self, name: str, x: np.ndarray) -> str:
        return str(write(self.d / name, x))

    def test_clean_take_beats_one_with_a_dropout(self):
        clean = self.take("clean.wav", fade(plucks(20, 0.5)))
        gap = plucks(20, 0.5)
        gap[int(8 * SR):int(9.2 * SR)] = 0
        holed = self.take("holed.wav", fade(gap))
        rc, r = run(holed, clean, "--target", "19.5")
        self.assertEqual(rc, 0)
        self.assertEqual(r["recommended"], clean)
        out = next(t for t in r["takes"] if t["file"] == holed)
        self.assertFalse(out["eligible"])
        self.assertTrue(any("drops out at 8." in d for d in out["defects"]), out["defects"])

    def test_short_take_is_ruled_out(self):
        short = self.take("short.wav", fade(plucks(15, 0.5)))
        rc, r = run(short, "--target", "20")
        self.assertEqual(rc, 1)
        self.assertIn("5.0s shorter than the video", r["takes"][0]["defects"])

    def test_ending_cut_off_unless_the_video_fades_it(self):
        abrupt = self.take("abrupt.wav", plucks(20, 0.5))
        faded = self.take("faded.wav", fade(plucks(20, 0.5)))
        _, r = run(abrupt, faded)
        self.assertEqual(r["recommended"], faded)
        a = next(t for t in r["takes"] if t["file"] == abrupt)
        self.assertEqual(a["ending"], "cuts off")
        _, r2 = run(abrupt, "--fade-out", "1.5")
        self.assertTrue(r2["takes"][0]["fade_covers_it"])
        self.assertFalse(any("doesn't fade" in b for b in r2["takes"][0]["bad"]))

    def test_busy_take_loses_under_text(self):
        calm = self.take("calm.wav", fade(plucks(20, 0.6)))
        busy = self.take("busy.wav", fade(plucks(20, 0.12)))
        _, r = run(busy, calm, "--under", "text")
        self.assertEqual(r["recommended"], calm)
        b = next(t for t in r["takes"] if t["file"] == busy)
        self.assertGreater(b["onsets_per_s"], pm.CALM["text"])

    def test_a_note_on_the_cut_counts(self):
        t = np.arange(20 * SR) / SR
        pad = 0.1 * np.sin(2 * np.pi * 110 * t)
        on = pad.copy()
        n = t >= 5.0
        on[n] += 0.4 * np.sin(2 * np.pi * 440 * (t[n] - 5.0)) * np.exp(-(t[n] - 5.0) * 4)
        lands = self.take("lands.wav", fade(on))
        _, r = run(lands, "--cuts", "5.0,12.0")
        hits = {h["at"]: h["lands"] for h in r["takes"][0]["cuts"]}
        self.assertTrue(hits[5.0])
        self.assertFalse(hits[12.0])

    def test_a_tie_goes_to_the_calmer_take_and_says_so(self):
        same = fade(plucks(20, 0.5))
        a = self.take("a.wav", same)
        b = self.take("b.wav", same)
        with mock.patch.object(pm, "score", side_effect=lambda m, under: (1.0, [], [])):
            with mock.patch.object(pm, "analyse", wraps=pm.analyse) as an:
                _, r = run(a, b, "--under", "text")
                self.assertEqual(an.call_count, 2)
        self.assertTrue(any(w.startswith("ties with") and "calmer" in w for w in r["why"]), r["why"])

    def test_pick_copies_the_winner(self):
        clean = self.take("clean.wav", fade(plucks(20, 0.5)))
        dest = self.d / "out" / "music.wav"
        _, r = run(clean, "--pick", str(dest))
        self.assertTrue(dest.exists())
        self.assertEqual(r["picked"], str(dest))


if __name__ == "__main__":
    unittest.main()
