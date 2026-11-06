"""Host capabilities: an agent is told what this host can make, and submit refuses the rest before any spend.

    uv run python -m unittest tests/python/test_video_job_host.py -v
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import video_job as vj  # noqa: E402
from test_video_job import Isolated, spec  # noqa: E402


def keys_only_spec() -> dict:
    """A video a host with only hosted API keys (no Modal, no RunPod) can make."""
    sp = spec()
    sp["voice"] = {"provider": "elevenlabs", "voiceId": "x"}
    for s in sp["scenes"]:
        s["visual"] = {"kind": "ideogram", "caption": {"headline": "HELLO", "object": "a blue circle"}}
    sp["music"] = False
    return sp


HAS_LOCAL_TOOLS = {"ffmpeg": True, "ffprobe": True, "node": 24,
                   "python_packages": {"moviepy": True, "numpy": True}}


class TestHost(Isolated):
    def setUp(self):
        super().setUp()
        os.environ.pop("VIDEO_JOB_SKIP_HOST_CHECK", None)
        self._real_local_tools = vj._local_tools
        vj._local_tools = lambda: HAS_LOCAL_TOOLS      # the key logic, not this machine's PATH

    def tearDown(self):
        vj._local_tools = self._real_local_tools
        super().tearDown()

    def test_music_is_soft(self):
        os.environ["ELEVENLABS_API_KEY"] = "k"
        os.environ["IDEOGRAM_API_KEY"] = "k"
        sp = keys_only_spec(); sp.pop("music")             # music on by default, no ACEMUSIC_API_KEY
        n = vj.normalize_spec(sp)
        self.assertEqual(vj.missing_for(n), [])
        soft = vj.missing_for(n, include_soft=True)
        self.assertEqual([(m["needed_for"], m.get("soft")) for m in soft], [("the music bed", True)])
        job = vj.submit(sp, max_usd=5, start_worker=False)
        self.assertEqual(job["phase"], "queued")
        vj.cancel(job["id"])

    def test_unknown_provider_is_a_spec_error(self):
        sp = keys_only_spec(); sp["music"] = {"provider": "suno"}
        with self.assertRaises(vj.SpecError) as ctx:
            vj.missing_for(vj.normalize_spec(sp), include_soft=True)
        self.assertIn("suno", str(ctx.exception))

    def test_message_names_every_var_needed(self):
        sp = keys_only_spec(); sp["voice"] = {"provider": "qwen3", "cloud": "runpod", "speaker": "Ryan"}
        os.environ["IDEOGRAM_API_KEY"] = "k"
        os.environ["RUNPOD_API_KEY"] = "k"
        with self.assertRaises(vj.Unavailable) as ctx:
            vj.submit(sp, max_usd=5, start_worker=False)
        self.assertIn("RUNPOD_QWEN3_TTS_ENDPOINT_ID and RUNPOD_API_KEY", str(ctx.exception))

    def test_resume_checks_the_host_first(self):
        os.environ["VIDEO_JOB_SKIP_HOST_CHECK"] = "1"
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        jf = vj.job_dir(job["id"]) / "job.json"
        j = json.loads(jf.read_text())
        j.update(phase="failed", error_code="worker_never_started"); jf.write_text(json.dumps(j))
        vj.clear_stale_lock()
        os.environ.pop("VIDEO_JOB_SKIP_HOST_CHECK")
        with self.assertRaises(vj.Unavailable):
            vj.resume(job["id"], start_worker=False)

    def test_skip_switch_quiets_estimate_too(self):
        os.environ["VIDEO_JOB_SKIP_HOST_CHECK"] = "1"
        self.assertEqual(vj.missing_for(vj.normalize_spec(spec()), include_soft=True), [])

    def test_bare_host_offers_only_placeholders(self):
        h = vj.host_capabilities()
        self.assertEqual(h["available_visual_kinds"], ["none"])
        self.assertFalse(h["voices"]["qwen3"]["modal"])
        self.assertIn("host", vj.catalog())
        self.assertIn("host", vj.presets())

    def test_submit_refuses_before_anything_is_written(self):
        with self.assertRaises(vj.Unavailable) as ctx:
            vj.submit(spec(), max_usd=5, start_worker=False)
        needed = " ".join(m["needed_for"] for m in ctx.exception.missing)
        self.assertIn("qwen3 voice", needed)
        self.assertFalse((self.tmp / "jobs").exists() and any((self.tmp / "jobs").iterdir()))

    def test_keys_only_host_can_make_a_card_video(self):
        os.environ["ELEVENLABS_API_KEY"] = "k"
        os.environ["IDEOGRAM_API_KEY"] = "k"
        h = vj.host_capabilities()
        self.assertEqual(sorted(h["available_visual_kinds"]), ["ideogram", "none"])
        self.assertEqual(vj.missing_for(vj.normalize_spec(keys_only_spec())), [])
        job = vj.submit(keys_only_spec(), max_usd=5, start_worker=False)
        self.assertEqual(job["phase"], "queued")
        vj.cancel(job["id"])

    def test_missing_qwen_points_at_the_other_cloud_when_it_is_set_up(self):
        os.environ["RUNPOD_API_KEY"] = "k"
        os.environ["RUNPOD_QWEN3_TTS_ENDPOINT_ID"] = "e"
        miss = vj.missing_for(vj.normalize_spec(spec()))
        voice = next(m for m in miss if "voice" in m["needed_for"])
        self.assertIn('"cloud": "runpod"', voice["hint"])

    def test_cli_estimate_reports_and_submit_refuses_with_exit_2(self):
        p = self.tmp / "spec.json"
        p.write_text(json.dumps(spec()))
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(vj.main(["estimate", "--spec", str(p), "--json"]), 0)
        self.assertTrue(json.loads(out.getvalue())["host_missing"])
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(vj.main(["submit", "--spec", str(p), "--json"]), 2)
        res = json.loads(out.getvalue())
        self.assertEqual(res["error_code"], "unavailable")
        self.assertTrue(res["missing"])

    def test_routes_follow_the_local_tools(self):
        h = vj.host_capabilities()
        self.assertIn("job", h["routes"])
        real = vj._local_tools
        vj._local_tools = lambda: {"ffmpeg": False, "ffprobe": True, "node": None,
                                   "python_packages": {"moviepy": True, "numpy": True}}
        try:
            h = vj.host_capabilities()
            self.assertEqual(h["routes"], {"job": False, "session": False})
            os.environ["ELEVENLABS_API_KEY"] = "k"
            os.environ["IDEOGRAM_API_KEY"] = "k"
            miss = vj.missing_for(vj.normalize_spec(keys_only_spec()))
            self.assertEqual([m["install"] for m in miss], ["ffmpeg"])
            vj._local_tools = lambda: {"ffmpeg": True, "ffprobe": True, "node": 16,
                                       "python_packages": {"moviepy": True, "numpy": True}}
            self.assertEqual(vj.host_capabilities()["routes"], {"job": True, "session": False})
        finally:
            vj._local_tools = real

    def test_skip_switch(self):
        os.environ["VIDEO_JOB_SKIP_HOST_CHECK"] = "1"
        vj.check_host(vj.normalize_spec(spec()))   # no raise


if __name__ == "__main__":
    import unittest
    unittest.main()
