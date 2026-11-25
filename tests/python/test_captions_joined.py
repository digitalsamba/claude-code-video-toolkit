import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import captions  # noqa: E402


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs FFmpeg")
class JoinedCaptionsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.phrases = self.root / "phrases"
        self.phrases.mkdir()
        self.output = self.root / "narration.captions.json"

    def write_wav(self, filename, frames):
        path = self.phrases / filename
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 48000, frames, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * frames)
        return path

    def run_join(self, responses):
        with (mock.patch.object(captions, "load_whisper", return_value=object()),
              mock.patch.object(captions, "transcribe_whisper", side_effect=responses) as transcribe,
              redirect_stdout(io.StringIO()) as stdout,
              redirect_stderr(io.StringIO())):
            captions.main([str(self.phrases), "--engine", "whisper", "--joined", str(self.output),
                           "--gap-ms", "137", "--srt", "--vtt", "--json"])
        return json.loads(stdout.getvalue()), [call.args[1].name for call in transcribe.call_args_list]

    def test_join_uses_decoded_duration_and_keeps_words_inside_their_clips(self):
        self.write_wav("02-next.wav", 48000)
        source = self.write_wav("uncompressed.wav", 24013)
        mp3 = self.phrases / "01-start.mp3"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "libmp3lame", str(mp3)],
                       check=True, capture_output=True)
        source.unlink()
        summary, order = self.run_join([[{"w": "First", "s": 0.42, "e": 0.9}],
                                        [{"w": "Second", "s": 0.06, "e": 0.19}]])
        self.assertEqual(order, ["01-start.mp3", "02-next.wav"])
        words = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual([(w["startMs"], w["endMs"], w["text"]) for w in words],
                         [(420, 500, "First"), (697, 827, " Second")])
        self.assertEqual(summary["joined"]["words"], 2)
        self.assertTrue((self.root / "narration.srt").is_file())
        self.assertTrue((self.root / "narration.vtt").is_file())

    def test_a_failed_clip_preserves_an_existing_joined_file(self):
        self.write_wav("01-first.wav", 48000)
        self.write_wav("02-silent.wav", 48000)
        self.output.write_text("[]\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as error:
            self.run_join([[{"w": "First", "s": 0.05, "e": 0.25}], []])
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "[]\n")
