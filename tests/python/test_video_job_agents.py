"""Tests for the agent-runtime seams of tools/video_job.py: inline specs, the compact
caption form, the reserve-at-submit ledger mode and the ETA. No cloud keys, no network."""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import video_job as vj  # noqa: E402
from test_video_job import Isolated, spec  # noqa: E402


class TestAgentSeams(Isolated):
    def test_compact_caption_expands_into_the_template(self):
        n = vj.normalize_spec({"scenes": [{"text": "hi", "visual": {
            "kind": "ideogram", "caption": {"headline": "ONE EXTRA DAY", "object": "an amber calendar"}}}]})
        cap = n["scenes"][0]["visual"]["caption"]
        self.assertEqual(set(cap), set(vj.IDEOGRAM_CAPTION_TEMPLATE))
        els = cap["compositional_deconstruction"]["elements"]
        self.assertEqual(els[0]["text"], "ONE EXTRA DAY")
        self.assertIn("amber calendar", els[1]["desc"])
        self.assertIn("ONE EXTRA DAY", cap["high_level_description"])

    def test_spec_json_and_progress_json_on_the_cli(self):
        s = json.dumps(spec())
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = vj.main(["estimate", "--spec-json", s, "--progress", "json", "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out.getvalue())["scenes"], 3)
        ev = json.loads(err.getvalue().strip().splitlines()[-1])
        self.assertEqual(ev["stage"], "cost")
        self.assertIn("$0.0000", ev["msg"])

    def test_reserve_mode_books_the_estimate_once(self):
        os.environ["VIDEO_JOB_RESERVE_AT_SUBMIT"] = "1"
        try:
            s = json.dumps(spec())
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                rc = vj.main(["submit", "--spec-json", s, "--max-usd", "5", "--progress", "json", "--json",
                              "--output", str(self.tmp / "files" / "x.mp4")])
            self.assertEqual(rc, 0)
            job_id = json.loads(out.getvalue())["id"]
            job = vj.load_job(job_id)
            self.assertEqual(job["ledger"], "reserve")
            self.assertTrue(job["publish_to"].endswith("x.mp4"))
            ev = json.loads(err.getvalue().strip().splitlines()[-1])
            self.assertIn(f"${job['estimate_usd']:.4f}", ev["msg"])
            # the worker's own receipts stay in the job dir, not the shared ledger
            vj.write_receipt(job, "ltx2.py", 0.2, 100.0, 0, [])
            self.assertTrue((vj.job_dir(job_id) / "receipts.jsonl").exists())
            self.assertFalse(vj.spend_log_path().exists())
            # a worker that was started elsewhere is gone: cancel so the lock is free
            vj.cancel(job_id)
        finally:
            os.environ.pop("VIDEO_JOB_RESERVE_AT_SUBMIT", None)

    def test_actual_mode_is_the_default_outside_longarm(self):
        os.environ.pop("LONGARM_WORKSPACE", None)
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        self.assertEqual(job["ledger"], "actual")
        vj.write_receipt(job, "ltx2.py", 0.2, 100.0, 0, [])
        self.assertTrue(vj.spend_log_path().exists())

    def test_spend_log_follows_the_longarm_workspace(self):
        os.environ.pop("MEDIA_SPEND_LOG", None)
        os.environ["LONGARM_WORKSPACE"] = str(self.tmp / "ws")
        try:
            self.assertEqual(vj.spend_log_path(), self.tmp / "ws" / "logs" / "media-spend.log")
            self.assertTrue(vj.reserve_at_submit())
        finally:
            os.environ.pop("LONGARM_WORKSPACE", None)

    def test_eta_counts_down_through_the_phases(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        queued = vj.eta_seconds(job)
        job["phase"] = "build"
        build = vj.eta_seconds(job)
        self.assertGreater(queued, build)
        self.assertGreater(build, 0)
        job["phase"] = "done"
        self.assertIsNone(vj.eta_seconds(job))
