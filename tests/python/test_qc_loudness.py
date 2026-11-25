import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import qc  # noqa: E402


class FailedFixTest(unittest.TestCase):
    def test_a_clipping_candidate_cannot_replace_a_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "render.mp4"
            final = source.with_name("render.loudnorm.mp4")
            source.write_bytes(b"input")
            final.write_bytes(b"previous output")
            measured = {"lufs": -18.0, "true_peak": 0.5, "lra": 1.0}

            def encode(_src, dst, *_args, **_kwargs):
                dst.write_bytes(b"clips after encoding")

            report = qc.Report()
            with (mock.patch.object(qc, "normalize_loudness", side_effect=encode),
                  mock.patch.object(qc, "measure_loudness", return_value=measured)):
                self.assertIsNone(qc.fix_render(source, measured, -14.0, report))
            self.assertEqual(report.count("error"), 1)
            self.assertEqual(final.read_bytes(), b"previous output")
            self.assertEqual(source.read_bytes(), b"input")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs FFmpeg")
class FixRenderTest(unittest.TestCase):
    def test_render_fix_reports_the_encoded_mix_and_its_true_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "render.mp4"
            # Quiet narration surrogate with a short, high-crest transient.
            mix = (r"0.045*sin(2*PI*211*t)+if(between(t\,2.847\,2.947)\,"
                   r"0.99*(0.6*sin(2*PI*4700*t)+0.4*(2*random(0)-1))\,0)")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=96x64:r=30:d=7",
                            "-f", "lavfi", "-i", f"aevalsrc={mix}:s=48000:d=7", "-c:v", "libx264",
                            "-c:a", "aac", "-b:a", "128k", str(source)], check=True, capture_output=True)
            original = source.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / "tools/qc.py"), "--render", str(source),
                                     "--fix", "--skip-frames", "--json"], capture_output=True, text=True, encoding="utf-8")
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 0, report["findings"])
            final = Path(report["fixed_render"])
            self.assertEqual(Path(report["render"]["file"]), final)
            self.assertLessEqual(qc.measure_loudness(final)["true_peak"], qc.YOUTUBE_TRUE_PEAK)
            self.assertEqual(report["render"]["true_peak"], qc.measure_loudness(final)["true_peak"])
            self.assertEqual(source.read_bytes(), original)
