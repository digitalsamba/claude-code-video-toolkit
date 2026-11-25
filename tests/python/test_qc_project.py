import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[2] / "tools/qc.py"


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs FFmpeg")
class StandaloneProjectTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name)

    def check(self, *args):
        result = subprocess.run([sys.executable, str(TOOL), "--project", str(self.project), "--json", *args],
                                capture_output=True, text=True, encoding="utf-8")
        return result.returncode, json.loads(result.stdout)

    def test_checks_narration_captions_and_manims_event_times(self):
        phrases = self.project / "public/audio/phrases/intro"
        phrases.mkdir(parents=True)
        wav = phrases / "intro.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=5", str(wav)],
                       check=True, capture_output=True)
        wav.with_suffix(".txt").write_text("Pick a lower gear before the hill and pedal lightly through the shift.", encoding="utf-8")
        captions = self.project / "captions"
        captions.mkdir()
        (captions / "intro.captions.json").write_text('[{"text":"Pick","startMs":4500,"endMs":5500}]', encoding="utf-8")
        (self.project / "diagram.events.json").write_text('[{"at":1.4,"what":"pan"}]', encoding="utf-8")
        (self.project / "timeline.json").write_text('{"segments":[{"start":0,"dur":5}],"end":5}', encoding="utf-8")
        code, report = self.check()
        self.assertEqual(code, 1)
        self.assertEqual(report["template_type"], "standalone")
        self.assertEqual(report["pacing_measured"], 1)
        self.assertIn("public/audio/phrases/intro/intro.wav", report["scene_loudness"])
        self.assertEqual(report["timing_files_checked"], 2)
        errors = [f["message"] for f in report["findings"] if f["level"] == "error"]
        self.assertEqual(len(errors), 1, report["findings"])
        self.assertIn("caption ends after", errors[0])

    def test_explains_skips_for_an_empty_project(self):
        code, report = self.check()
        self.assertEqual(code, 0)
        self.assertEqual(report["audio_files_checked"], 0)
        notes = [f for f in report["findings"] if f["level"] == "info" and "skipped" in f["message"]]
        self.assertEqual({f["check"] for f in notes}, {"assets", "timing", "pacing"})

    def test_an_explicit_missing_config_is_an_error(self):
        code, report = self.check("--config", str(self.project / "missing.ts"))
        self.assertEqual(code, 1)
        self.assertIn("config file not found", report["findings"][0]["message"])
