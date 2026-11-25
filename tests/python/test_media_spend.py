import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import captions  # noqa: E402
import cloud_gpu  # noqa: E402
import media_spend  # noqa: E402
import video_job  # noqa: E402
import voiceover  # noqa: E402


class MediaSpendTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log = self.root / "logs/media-spend.log"
        env = mock.patch.dict(os.environ, {"MEDIA_SPEND_LOG": str(self.log), "MEDIA_SPEND_MANAGED": "0",
                                           "VIDEO_JOBS_DIR": str(self.root / "jobs")})
        env.start()
        self.addCleanup(env.stop)

    def receipts(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def test_standalone_voiceover_records_one_character_estimate(self):
        client, save = mock.Mock(), mock.Mock()
        with (mock.patch.object(voiceover, "_get_elevenlabs_imports", return_value=(None, mock.Mock(), save)),
              mock.patch.object(voiceover, "get_audio_duration", return_value=1.0)):
            voiceover.generate_single_audio(client, "A brief line.", self.root / "line.mp3", "voice",
                                            "eleven_multilingual_v2", 0.5, 0.5, 0.0, 1.0)
        row, = self.receipts()
        self.assertEqual((row["tool"], row["provider"], row["unit"], row["units"]),
                         ("voiceover.py", "elevenlabs", "characters", 13))
        self.assertEqual(row["cost_usd"], 0.0026)
        self.assertEqual(row["cost_source"], "estimate")
        self.assertNotIn("A brief line.", self.log.read_text(encoding="utf-8"))

    def test_scribe_records_audio_seconds_even_when_no_words_are_detected(self):
        source = self.root / "line.wav"
        source.write_bytes(b"test fixture")
        client = mock.Mock()
        client.speech_to_text.convert.return_value.words = []
        with (mock.patch("elevenlabs.client.ElevenLabs", return_value=client),
              mock.patch.object(captions, "get_audio_duration", return_value=9.0)):
            self.assertEqual(captions.transcribe_scribe(source, "test", "scribe_v2", "en"), [])
        row, = self.receipts()
        self.assertEqual(row["unit"], "audio_seconds")
        self.assertEqual(row["units"], 9.0)
        self.assertEqual(row["cost_usd"], 0.001)

    def test_cloud_receipts_do_not_depend_on_progress_output(self):
        responses = [({"model": "example"}, 9.0), ({"error": "request rejected"}, 0.0)]
        with mock.patch.object(cloud_gpu, "_call_modal", side_effect=responses):
            cloud_gpu.call_cloud_endpoint("modal", {}, "ltx2", verbose=False)
            cloud_gpu.call_cloud_endpoint("modal", {}, "ltx2", verbose=False)
        row, = self.receipts()
        self.assertEqual((row["tool"], row["provider"], row["units"]), ("ltx2.py", "modal", 9.0))
        self.assertGreater(row["cost_usd"], 0)

    def test_worker_owns_the_receipt_for_a_managed_child(self):
        fake = self.root / "paid.py"
        fake.write_text(f"import sys\nsys.path.insert(0, {str(TOOLS)!r})\n"
                        "from media_spend import record_tts\n"
                        "record_tts('paid.py', 50, 'fixture')\n"
                        "print('Est. cost: $0.0100')\n", encoding="utf-8")
        job = {"id": "20261125-090000-1234"}
        (video_job.jobs_dir() / job["id"]).mkdir(parents=True)
        result = video_job.run_tool(job, str(fake), [])
        self.assertEqual(result["returncode"], 0, result["stderr"])
        row, = self.receipts()
        self.assertEqual(row["cost_usd"], 0.01)
        self.assertEqual(row["job_id"], job["id"])
        self.assertEqual(job["actual_usd"], 0.01)

    def test_an_unpriced_request_remains_visible_with_a_null_cost(self):
        media_spend.record_spend("music.py", "elevenlabs", "audio_seconds", 30)
        row, = self.receipts()
        self.assertIsNone(row["cost_usd"])
        self.assertEqual(row["cost_source"], "unpriced")
