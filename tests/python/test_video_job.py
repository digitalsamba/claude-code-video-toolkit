"""Unit tests for tools/video_job.py: the estimate, the state machine, the lock and the cap.

They run without cloud keys and without a network, in a temporary jobs directory:

    uv run python -m unittest tests/python/test_video_job.py -v
    uv run pytest tests/python -q          # if pytest is installed

Paid steps are injected as fakes through run_pipeline(steps=...), so nothing here calls a
cloud tool. The template copy (create_project) is real.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import video_job as vj  # noqa: E402

KEYS = ("ELEVENLABS_API_KEY", "IDEOGRAM_API_KEY", "MODAL_LTX2_ENDPOINT_URL", "MODAL_QWEN3_TTS_ENDPOINT_URL",
        "R2_ACCOUNT_ID", "VIDEO_JOB_EXPANDER", "VIDEO_JOB_MAX_USD")


def spec(**over) -> dict:
    s = json.loads(json.dumps(vj.SPEC_EXAMPLE))
    s.update(over)
    return s


class Isolated(unittest.TestCase):
    """Every test gets its own jobs dir and spend log, and no cloud keys."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="video-job-test-"))
        self._env = {k: os.environ.pop(k, None) for k in KEYS}
        os.environ["VIDEO_JOBS_DIR"] = str(self.tmp / "jobs")
        os.environ["MEDIA_SPEND_LOG"] = str(self.tmp / "logs" / "media-spend.log")

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        os.environ.pop("VIDEO_JOBS_DIR", None)
        os.environ.pop("MEDIA_SPEND_LOG", None)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestEstimate(Isolated):
    def test_is_a_pure_function_of_the_spec(self):
        a, b = vj.estimate(spec()), vj.estimate(spec())
        self.assertEqual(a, b)
        words = sum(len(s["text"].split()) for s in vj.SPEC_EXAMPLE["scenes"])
        narration_s = round(words / vj.WORDS_PER_SECOND, 1)
        expected = (3 * vj.PRICES["voiceover"]["qwen3"] + 2 * vj.PRICES["ideogram_image"]["DEFAULT"]
                    + 1 * vj.PRICES["ltx_clip"] + narration_s * vj.PRICES["captions_per_audio_second"]["scribe"])
        self.assertAlmostEqual(a["estimate_usd"], round(expected, 4), places=4)
        self.assertEqual(a["scenes"], 3)
        self.assertEqual({r["step"] for r in a["breakdown"]}, {"voiceover", "visuals", "captions"})
        self.assertEqual(a["prices_version"], vj.PRICES_VERSION)

    def test_prices_follow_the_spec_knobs(self):
        cheap = vj.estimate(spec(ideogram={"speed": "TURBO"}, captions={"enabled": False}))
        self.assertAlmostEqual(cheap["estimate_usd"], 3 * 0.01 + 2 * 0.03 + 0.20, places=4)
        el = vj.estimate(spec(voice={"provider": "elevenlabs"}))
        chars = sum(len(s["text"]) for s in vj.SPEC_EXAMPLE["scenes"])
        vo = next(r for r in el["breakdown"] if r["step"] == "voiceover")
        self.assertAlmostEqual(vo["usd"], round(chars / 1000 * 0.20, 4), places=4)
        music = vj.estimate(spec(music={"enabled": True, "provider": "modal"}))
        self.assertIn("music", {r["step"] for r in music["breakdown"]})

    def test_brief_without_expander_estimates_a_skeleton(self):
        s, source = vj.resolve_spec("Why is the sky blue?", None, "16:9", 45, allow_skeleton=True)
        self.assertEqual(source, "skeleton")
        est = vj.estimate(s)
        self.assertEqual(est["format"], "16:9")
        self.assertEqual(est["scenes"], 4)             # 45 s / 12 s per scene
        self.assertGreater(est["estimate_usd"], 0)
        with self.assertRaises(vj.NoExpander):          # but it can't be submitted
            vj.resolve_spec("Why is the sky blue?", None, "9:16", 45, allow_skeleton=False)

    def test_normalize_validates(self):
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec({"scenes": []})
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec(spec(format="4:3"))
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec({"scenes": [{"text": "hi", "visual": {"kind": "ltx"}}]})  # no prompt
        n = vj.normalize_spec({"scenes": [{"text": "one", "visual": {"kind": "ltx", "prompt": "p"}},
                                          {"text": "two", "visual": {"kind": "ideogram", "prompt": "p"}},
                                          {"text": "three"}]})
        self.assertEqual([s["visual"]["kind"] for s in n["scenes"]], ["ltx", "ideogram", "none"])
        self.assertEqual(n["scenes"][0]["asset"], "clips/01_one.mp4")
        self.assertEqual(n["scenes"][1]["asset"], "images/02_two.png")
        self.assertEqual(n["scenes"][2]["asset"], "")              # placeholder card, costs nothing


class TestCapAndLock(Isolated):
    def test_cap_below_estimate_is_refused_before_any_spend(self):
        est = vj.estimate(spec())["estimate_usd"]
        with self.assertRaises(vj.BudgetExceeded):
            vj.submit(spec(), max_usd=round(est - 0.01, 2), start_worker=False)
        self.assertFalse(vj.lock_path().exists())
        self.assertEqual(list(vj.jobs_dir().glob("*/job.json")) if vj.jobs_dir().exists() else [], [])
        self.assertFalse(vj.spend_log_path().exists())

    def test_cli_refuses_with_exit_2(self):
        spec_path = self.tmp / "spec.json"
        spec_path.write_text(json.dumps(spec()))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = vj.main(["submit", "--spec", str(spec_path), "--max-usd", "0.05", "--json"])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue())["error_code"], "budget_exceeded")

    def test_only_one_job_at_a_time(self):
        first = vj.submit(spec(), max_usd=5, start_worker=False)
        self.assertTrue(vj.lock_path().exists())
        with self.assertRaises(vj.LockHeld) as ctx:
            vj.submit(spec(), max_usd=5, start_worker=False)
        self.assertEqual(ctx.exception.holder["job_id"], first["id"])
        # a finished job's lock is stale and gets cleared
        vj.set_phase(first, "failed", error="x")
        second = vj.submit(spec(), max_usd=5, start_worker=False)
        self.assertNotEqual(second["id"], first["id"])
        self.assertEqual(json.loads(vj.lock_path().read_text())["job_id"], second["id"])

    def test_dead_worker_releases_the_lock_and_fails_the_job(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        job["worker_pid"] = 2 ** 22 + 12345            # not a live pid on any sane box
        vj.save_job(job)
        s = vj.status(job["id"])
        self.assertEqual(s["phase"], "failed")
        self.assertEqual(s["error_code"], "worker_died")
        self.assertFalse(vj.lock_path().exists())
        self.assertIsNotNone(vj.submit(spec(), max_usd=5, start_worker=False))


class TestStateMachine(Isolated):
    def fakes(self, seen: list, fail_at: str | None = None, spend: dict | None = None):
        spend = spend or {}

        def step(name, ret):
            def f(job, *a, **k):
                seen.append(job["phase"])
                if name in spend:
                    vj.write_receipt(job, f"{name}.py", spend[name], 1.0, 0, [])
                if fail_at == name:
                    raise vj.StepFailed(f"{name} broke")
                return ret(job) if callable(ret) else ret
            return f

        def build(job):
            out = Path(job["project_dir"]) / "out" / "short.mp4"
            out.write_bytes(b"not really an mp4")
            seen.append(job["phase"])
            return out

        return {
            "voiceover": step("voiceover", {"narration_s": 20.0, "durations": {"01": 7, "02": 7, "03": 6}}),
            "visuals": step("visuals", {"assets": ["a", "b", "c"]}),
            "captions": step("captions", {"scenes": 3}),
            "build": build,
            "qc": step("qc", lambda job: (Path(job["project_dir"]) / "out" / "short.mp4",
                                          {"passed": True, "errors": 0, "warnings": 1})),
            "publish": step("publish", lambda job: {"mp4": str(Path(job["project_dir"]) / "out" / "short.mp4"),
                                                    "url": None, "duration_s": 24.1}),
        }

    def test_happy_path_walks_every_phase(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        seen: list = []
        vj.run_pipeline(job, json.loads(Path(job["spec_path"]).read_text()), self.fakes(seen))
        self.assertEqual(seen, ["voiceover", "visuals", "captions", "build", "qc", "publish"])
        on_disk = vj.load_job(job["id"])
        self.assertEqual(on_disk["phase"], "done")
        self.assertEqual(on_disk["progress"], 100)
        self.assertTrue(on_disk["output"]["mp4"].endswith("short.mp4"))
        self.assertTrue(on_disk["output"]["qc"]["passed"])
        self.assertTrue((Path(job["project_dir"]) / "scenes.json").exists())
        cfg = json.loads((Path(job["project_dir"]) / "config.json").read_text())
        self.assertEqual((cfg["format"]["width"], cfg["format"]["height"]), (1080, 1920))
        r = vj.result(job["id"])
        self.assertTrue(r["ready"])

    def test_16x9_writes_a_landscape_config(self):
        job = vj.submit(spec(format="16:9"), max_usd=5, start_worker=False)
        vj.run_pipeline(job, json.loads(Path(job["spec_path"]).read_text()), self.fakes([]))
        cfg = json.loads((Path(job["project_dir"]) / "config.json").read_text())
        self.assertEqual((cfg["format"]["width"], cfg["format"]["height"]), (1920, 1080))
        self.assertLess(cfg["captions"]["y"], 1080)

    def test_failed_step_is_reported_not_lost(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        seen: list = []
        vj.run_pipeline(job, json.loads(Path(job["spec_path"]).read_text()), self.fakes(seen, fail_at="visuals"))
        self.assertEqual(seen, ["voiceover", "visuals"])
        on_disk = vj.load_job(job["id"])
        self.assertEqual(on_disk["phase"], "failed")
        self.assertEqual(on_disk["error_code"], "step_failed")
        self.assertIn("visuals broke", on_disk["error"])
        self.assertFalse(vj.result(job["id"])["ready"])

    def test_worker_stops_before_the_next_paid_step_at_the_cap(self):
        est = vj.estimate(spec())["estimate_usd"]
        job = vj.submit(spec(), max_usd=round(est + 0.01, 2), start_worker=False)
        seen: list = []
        # voiceover "actually" costs far more than estimated, so visuals would pass the cap
        vj.run_pipeline(job, json.loads(Path(job["spec_path"]).read_text()),
                        self.fakes(seen, spend={"voiceover": est}))
        self.assertEqual(seen, ["voiceover"])
        on_disk = vj.load_job(job["id"])
        self.assertEqual(on_disk["phase"], "failed")
        self.assertEqual(on_disk["error_code"], "budget_exceeded")
        self.assertAlmostEqual(on_disk["actual_usd"], est, places=4)

    def test_receipts_match_the_ledger_shape(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        vj.write_receipt(job, "ltx2.py", 0.2, 101.5, 0, ["--prompt", "x"])
        for path in (vj.job_dir(job["id"]) / "receipts.jsonl", vj.spend_log_path()):
            line = json.loads(path.read_text().strip().splitlines()[-1])
            self.assertEqual(set(line) >= {"ts", "tool", "cost_usd", "elapsed_s", "exit", "args"}, True)
            self.assertEqual(line["tool"], "ltx2.py")
            self.assertEqual(line["job_id"], job["id"])
        self.assertAlmostEqual(vj.load_job(job["id"])["actual_usd"], 0.2)


class TestCli(Isolated):
    def test_estimate_and_presets_need_no_keys(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = vj.main(["estimate", "--brief", "A 45 second explainer about prompt caching", "--json"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["spec_source"], "skeleton")
        self.assertGreater(out["estimate_usd"], 0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(vj.main(["presets"]), 0)
        self.assertIn("spec_example", json.loads(buf.getvalue()))
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(vj.main(["list", "--json"]), 0)
        self.assertEqual(json.loads(buf.getvalue()), [])

    def test_submit_brief_without_expander_is_refused(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = vj.main(["submit", "--brief", "anything", "--json"])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(buf.getvalue())["error_code"], "no_expander")

    def test_expander_command_is_pluggable(self):
        script = self.tmp / "expander.py"
        script.write_text(
            "import json,sys\nreq=json.load(sys.stdin)\n"
            "print('```json')\n"
            "print(json.dumps({'title': req['brief'], 'scenes': [\n"
            "  {'text': 'hello there', 'visual': {'kind': 'flux', 'prompt': 'soft gradient'}},\n"
            "  {'text': 'bye now', 'visual': {'kind': 'none'}}]}))\n"
            "print('```')\n")
        os.environ["VIDEO_JOB_EXPANDER"] = f'"{sys.executable}" "{script}"'
        s, source = vj.resolve_spec("Say hello", None, "16:9", 20, allow_skeleton=False)
        self.assertEqual(source, "expander")
        self.assertEqual(s["title"], "Say hello")
        self.assertEqual(s["format"], "16:9")
        self.assertEqual(vj.estimate(s)["estimate_usd"], round(2 * 0.01 + 0.01 + (4 / 2.4) * 0.40 / 3600, 4))


if __name__ == "__main__":
    unittest.main()
