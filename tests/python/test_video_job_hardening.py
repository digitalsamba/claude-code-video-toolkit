"""Hardening tests for tools/video_job.py: resume, prune, the lock window, the cost event on
every exit, the per-host ETA, events and warnings, and the build's size cap. No keys, no
network: cloud tools are replaced by a fake run_tool that writes the files a real one would,
so the real step functions (and their skip-what-exists logic) run.

    uv run python -m unittest tests/python/test_video_job_hardening.py -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import video_job as vj  # noqa: E402
from test_video_job import Isolated, spec  # noqa: E402


class FakeTools:
    """Stands in for run_tool: writes each tool's output file, books the fallback cost like
    the real one, and can fail one named visual to simulate a job dying mid-way."""

    def __init__(self, fail_scene: str | None = None):
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_scene = fail_scene

    def __call__(self, job, tool, args, cwd=None, timeout=None, fallback_cost=None):
        name = Path(tool).name
        self.calls.append((name, list(args)))
        out, rc = "", 0
        if name == "voiceover.py":
            scene_dir = Path(args[args.index("--scene-dir") + 1])
            rows = []
            for txt in sorted(scene_dir.glob("*.txt")):
                mp3 = txt.with_suffix(".mp3")
                mp3.write_bytes(b"ID3fake")
                rows.append({"output": str(mp3), "success": True, "duration_seconds": 5.0,
                             "wpm": 182.0 if txt.name.startswith("02") else 150.0,
                             "pacing": "fast" if txt.name.startswith("02") else "ok"})
            out = json.dumps({"scenes": rows})
        elif name in ("ideogram4.py", "ltx2.py", "flux2.py"):
            target = Path(args[args.index("--output") + 1])
            if self.fail_scene and target.name.startswith(self.fail_scene + "_"):
                rc = 1
            else:
                target.write_bytes(b"asset")
        elif name == "captions.py":
            for mp3 in Path(args[0]).glob("*.mp3"):
                mp3.with_suffix(".captions.json").write_text(json.dumps(
                    [{"text": "hello", "startMs": 0, "endMs": 400}, {"text": "/", "startMs": 400, "endMs": 450},
                     {"text": "there", "startMs": 450, "endMs": 800}]))
        elif name == "music_gen.py":
            Path(args[args.index("--output") + 1]).write_bytes(b"music")
        cost = fallback_cost if rc == 0 else None
        if name not in ("build.py", "qc.py"):
            vj.write_receipt(job, name, cost, 0.1, rc, args)
        return {"returncode": rc, "stdout": out, "stderr": "", "cost": cost, "elapsed": 0.1}

    def names(self):
        return [c[0] for c in self.calls]


def free_tail():
    """build/qc/publish fakes (the free steps)."""
    def build(job):
        out = Path(job["project_dir"]) / "out" / "short.mp4"
        out.write_bytes(b"mp4")
        return out

    def qc(job, render):
        return render, {"passed": True, "errors": 0, "warnings": 0, "findings": []}

    def publish(job, spec_, final):
        return {"mp4": str(final), "duration_s": 20.0, "size_bytes": 3, "url": None}
    return {"build": build, "qc": qc, "publish": publish}


def run(job, tools):
    with mock.patch.object(vj, "run_tool", tools):
        return vj.run_pipeline(job, json.loads(Path(job["spec_path"]).read_text()), free_tail())


class TestResume(Isolated):
    def test_resume_pays_only_for_what_is_missing(self):
        s = spec(music={"enabled": True, "provider": "modal"})
        job = vj.submit(s, max_usd=5, start_worker=False)
        first = FakeTools(fail_scene="03")          # dies on the third visual
        run(job, first)
        failed = vj.status(job["id"])
        self.assertEqual(failed["phase"], "failed")
        self.assertTrue(failed["resumable"])
        spent_first = failed["actual_usd"]
        self.assertEqual(first.names(), ["voiceover.py", "ltx2.py", "ideogram4.py", "ideogram4.py"])

        job = vj.resume(job["id"], start_worker=False)
        self.assertEqual(job["phase"], "queued")
        self.assertIsNone(job["error"])
        second = FakeTools()
        run(job, second)
        done = vj.load_job(job["id"])
        self.assertEqual(done["phase"], "done")
        # no second voiceover, only the missing visual, then captions and music
        self.assertEqual(second.names(), ["ideogram4.py", "captions.py", "music_gen.py"])
        unit = vj.PRICES["ideogram_image"]["DEFAULT"]
        self.assertAlmostEqual(done["actual_usd"], round(spent_first + unit + done["steps"]["captions"].get("cost", 0)
                                                         + vj.PRICES["music_track"]["modal"], 4), places=3)
        self.assertEqual(done["resumes"], 1)
        kinds = [e["kind"] for e in done["events"]]
        self.assertIn("resumed", kinds)
        # a second resume of a finished job is refused
        with self.assertRaises(vj.SpecError):
            vj.resume(job["id"], start_worker=False)

    def test_killed_worker_is_resumable_and_steps_are_idempotent(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        tools = FakeTools()
        run(job, tools)
        self.assertEqual(vj.load_job(job["id"])["phase"], "done")
        # pretend the container died during build on a later run: phase build, dead pid
        j = vj.load_job(job["id"])
        j.update(phase="build", worker_pid=2 ** 22 + 4321, finished=None)
        vj.save_job(j)
        s = vj.status(job["id"])
        self.assertEqual(s["error_code"], "worker_died")
        self.assertTrue(s["resumable"])
        again = FakeTools()
        run(vj.resume(job["id"], start_worker=False), again)
        self.assertEqual(again.names(), [])             # every paid output was on disk
        self.assertEqual(vj.load_job(job["id"])["phase"], "done")

    def test_reserve_ledger_nets_to_actual_across_a_resume(self):
        os.environ["VIDEO_JOB_RESERVE_AT_SUBMIT"] = "1"
        try:
            job = vj.submit(spec(), max_usd=5, start_worker=False)
            ledger = vj.spend_log_path()
            ledger.parent.mkdir(parents=True, exist_ok=True)

            def book(usd):   # what a runtime's spend wrapper writes from our cost event
                with open(ledger, "a") as f:
                    f.write(json.dumps({"tool": "video_job.py", "cost_usd": usd}) + "\n")
            book(job["estimate_usd"])
            run(job, FakeTools(fail_scene="03"))
            job = vj.resume(job["id"], start_worker=False)
            self.assertGreater(job["resume_reserved_usd"], 0)
            book(job["resume_reserved_usd"])
            run(job, FakeTools())
            done = vj.load_job(job["id"])
            total = sum(json.loads(line)["cost_usd"] for line in ledger.read_text().splitlines())
            self.assertAlmostEqual(total, done["actual_usd"], places=4)
        finally:
            os.environ.pop("VIDEO_JOB_RESERVE_AT_SUBMIT", None)

    def test_budget_stop_resumes_with_a_higher_cap(self):
        est = vj.estimate(spec())["estimate_usd"]
        job = vj.submit(spec(), max_usd=est + 0.01, start_worker=False)
        tools = FakeTools()
        job["actual_usd"] = est      # as if voiceover had cost far more than estimated
        vj.save_job(job)
        run(job, tools)
        self.assertEqual(vj.load_job(job["id"])["error_code"], "budget_exceeded")
        job = vj.resume(job["id"], max_usd=10, start_worker=False)
        self.assertEqual(job["max_usd"], 10)
        run(job, FakeTools())
        self.assertEqual(vj.load_job(job["id"])["phase"], "done")

    def test_qwen3_voiceover_asks_for_the_cost_event(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        tools = FakeTools()
        run(job, tools)
        vo_args = next(a for n, a in tools.calls if n == "voiceover.py")
        self.assertEqual(vo_args[vo_args.index("--progress") + 1], "json")


class TestPrune(Isolated):
    def make_job(self, n: int, phase="done", days_ago=30.0, published=False, project=True):
        jid = f"20260101-{n:06d}-abcd"
        d = vj.jobs_dir() / jid
        d.mkdir(parents=True)
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days_ago * 86400))
        job = {"id": jid, "phase": phase, "created": finished, "finished": finished,
               "output": {"mp4": str(d / "x.mp4"), "published": "/srv/files/x.mp4" if published else None}}
        (d / "job.json").write_text(json.dumps(job))
        (d / "x.mp4").write_bytes(b"keep me")
        if project:
            (d / "project" / "clips").mkdir(parents=True)
            (d / "project" / "clips" / "a.mp4").write_bytes(b"x" * 1000)
        return jid

    def test_prune_keeps_recent_running_and_published(self):
        ids = [self.make_job(i) for i in range(1, 14)]          # 13 old finished jobs
        running = ids[0]
        j = json.loads((vj.jobs_dir() / running / "job.json").read_text())
        j["phase"] = "build"
        (vj.jobs_dir() / running / "job.json").write_text(json.dumps(j))
        fresh_pub = self.make_job(0, days_ago=2, published=True)   # oldest id, published 2 days ago
        dry = vj.prune(keep=10, dry_run=True)
        self.assertTrue(all((vj.jobs_dir() / i / "project").exists() for i in ids))
        r = vj.prune(keep=10)
        self.assertEqual(r["pruned"], dry["pruned"])
        newest_ten = sorted(ids + [fresh_pub], reverse=True)[:10]
        expect = sorted(set(ids) - set(newest_ten) - {running}, reverse=True)
        self.assertEqual(r["pruned"], expect)
        self.assertEqual(r["skipped"][running], "running")
        self.assertEqual(r["skipped"][fresh_pub], "published in the last 7 days")
        self.assertEqual(r["freed_bytes"], 1000 * len(expect))
        for i in expect:
            d = vj.jobs_dir() / i
            self.assertFalse((d / "project").exists())
            self.assertTrue((d / "x.mp4").exists())
            self.assertTrue(json.loads((d / "job.json").read_text())["pruned"])
            self.assertFalse(vj.is_resumable({**json.loads((d / "job.json").read_text()), "phase": "failed"}))

    def test_older_than_and_submit_runs_a_conservative_prune(self):
        for i in range(1, 13):
            self.make_job(i, days_ago=1 if i > 6 else 40)
        r = vj.prune(keep=0, older_than_days=7, dry_run=True)
        self.assertEqual(len(r["pruned"]), 6)
        vj.submit(spec(), max_usd=5, start_worker=False)          # auto-prunes: keeps the 10 newest
        left = [p.parent.name for p in vj.jobs_dir().glob("*/project")]
        self.assertEqual(len(left), 10)
        os.environ["VIDEO_JOB_KEEP"] = "0"
        try:
            self.assertIsNone(vj.auto_prune())
        finally:
            os.environ.pop("VIDEO_JOB_KEEP")


class TestReviewFixes(Isolated):
    def test_a_partial_file_from_a_failed_tool_is_never_reused(self):
        class Partial(FakeTools):
            def __call__(self, job, tool, args, **kw):
                r = super().__call__(job, tool, args, **kw)
                if r["returncode"]:
                    Path(args[args.index("--output") + 1]).write_bytes(b"half a file")
                return r
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        run(job, Partial(fail_scene="03"))
        job = vj.resume(job["id"], start_worker=False)
        again = FakeTools()
        run(job, again)
        self.assertIn("ideogram4.py", again.names())        # scene 03 bought again, not reused
        self.assertIn("images/03_cta.png", vj.load_job(job["id"])["made"])

    def test_resume_lock_is_not_stolen_by_a_status_poll(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        run(job, FakeTools(fail_scene="03"))
        vj.release_lock(job["id"])
        vj.acquire_lock(job["id"])            # what resume does while job.json still says failed
        vj.clear_stale_lock()
        self.assertTrue(vj.lock_path().exists())

    def test_reserve_ledger_nets_when_the_first_run_overspent(self):
        os.environ["VIDEO_JOB_RESERVE_AT_SUBMIT"] = "1"
        try:
            job = vj.submit(spec(), max_usd=5, start_worker=False)
            ledger = vj.spend_log_path()
            ledger.parent.mkdir(parents=True, exist_ok=True)
            with open(ledger, "a") as f:
                f.write(json.dumps({"cost_usd": job["estimate_usd"]}) + "\n")
            job["project_dir"] = str(vj.create_project(job, json.loads(Path(job["spec_path"]).read_text())))
            job["actual_usd"] = job["estimate_usd"] + 0.20           # overspent, then failed
            vj.set_phase(job, "failed", error="x", error_code="step_failed")
            job = vj.resume(job["id"], start_worker=False)
            self.assertEqual(job["resume_reserved_usd"], 0.0)
            job["actual_usd"] += 0.10
            vj.set_phase(job, "done")
            total = sum(json.loads(l)["cost_usd"] for l in ledger.read_text().splitlines())
            self.assertAlmostEqual(total, vj.load_job(job["id"])["actual_usd"], places=4)
        finally:
            os.environ.pop("VIDEO_JOB_RESERVE_AT_SUBMIT", None)

    def test_error_tail_drops_progress_lines(self):
        r = {"stderr": '{"ts": "1", "stage": "heartbeat"}\nTraceback: boom\n{"ts": "2", "stage": "cost"}'}
        self.assertEqual(vj._error_tail(r), "Traceback: boom")

    @unittest.skipIf(os.name == "nt", "ps check is POSIX")
    def test_cancel_leaves_a_recycled_pid_alone(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            job = vj.submit(spec(), max_usd=5, start_worker=False)
            job.update(worker_pid=other.pid, phase="visuals")
            vj.save_job(job)
            vj.cancel(job["id"])
            self.assertTrue(vj.pid_alive(other.pid))
        finally:
            other.kill()


class TestLockWindow(Isolated):
    def test_a_submit_that_never_spawned_frees_the_lock_after_120s(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        with self.assertRaises(vj.LockHeld):                      # fresh: still held
            vj.submit(spec(), max_usd=5, start_worker=False)
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 200))
        j = vj.load_job(job["id"])
        j["created"] = old
        j["phase_started_ts"] = time.time() - 200
        vj.save_job(j)
        lock = json.loads(vj.lock_path().read_text())
        lock["ts"] = time.time() - 200
        vj.lock_path().write_text(json.dumps(lock))
        s = vj.status(job["id"])
        self.assertEqual(s["error_code"], "worker_never_started")
        self.assertTrue(s["resumable"])
        self.assertFalse(vj.lock_path().exists())
        self.assertIsNotNone(vj.submit(spec(), max_usd=5, start_worker=False))

    def test_unreadable_lock_is_cleared_by_list(self):
        vj.jobs_dir().mkdir(parents=True)
        vj.lock_path().write_text("{not json")
        vj.list_jobs()
        self.assertFalse(vj.lock_path().exists())


class TestCostEventOnEveryExit(Isolated):
    def call(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = vj.main(list(argv) + ["--progress", "json"])
        costs = [json.loads(l) for l in err.getvalue().splitlines() if l.startswith("{") and '"cost"' in l]
        return rc, costs, out.getvalue()

    def test_errors_and_refusals_print_one_zero_cost_event(self):
        for argv, want_rc in ((["submit", "--spec-json", "{not json at all", "--json"], 1),
                              (["submit", "--brief", "x", "--json"], 2),
                              (["status", "20990101-000000-abcd", "--json"], 1),
                              (["resume", "20990101-000000-abcd", "--json"], 1),
                              (["status"], 2),                                  # argparse usage error
                              (["spec", "--brief", "x"], 2)):
            rc, costs, _ = self.call(*argv)
            self.assertEqual(rc, want_rc, argv)
            self.assertEqual(len(costs), 1, argv)
            self.assertIn("$0.0000", costs[0]["msg"])

    def test_lock_held_and_success_print_exactly_one(self):
        vj.submit(spec(), max_usd=5, start_worker=False)
        p = self.tmp / "spec.json"
        p.write_text(json.dumps(spec()))
        rc, costs, out = self.call("submit", "--spec", str(p), "--json")
        self.assertEqual(rc, 3)
        self.assertEqual(json.loads(out)["error_code"], "lock_held")
        self.assertEqual(len(costs), 1)
        rc, costs, _ = self.call("estimate", "--spec", str(p), "--json")
        self.assertEqual((rc, len(costs)), (0, 1))

    def test_unreadable_job_is_an_error_not_a_traceback(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        (vj.job_dir(job["id"]) / "job.json").write_text("{half a jso")
        rc, costs, out = self.call("status", job["id"], "--json")
        self.assertEqual(rc, 1)
        self.assertEqual(json.loads(out)["error_code"], "job_unreadable")
        self.assertEqual(len(costs), 1)

    def test_missing_spec_fails_the_job_cleanly(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        Path(job["spec_path"]).unlink()
        self.assertEqual(vj.worker_main(job["id"]), 1)
        j = vj.load_job(job["id"])
        self.assertEqual(j["phase"], "failed")
        self.assertIn("spec.json", j["error"])
        self.assertFalse(vj.lock_path().exists())


class TestEta(Isolated):
    def job(self, phase="queued", **over):
        s = vj.normalize_spec({"format": "16:9", "music": False, "scenes": [
            {"text": "one two three four", "visual": {"kind": "ltx", "prompt": "slow pan over a harbour at dawn"}},
            {"text": "five six seven", "visual": {"kind": "text", "headline": "HI"}}]})
        p = self.tmp / f"spec-{phase}.json"
        p.write_text(json.dumps(s))
        return {"id": "20260101-000000-abcd", "phase": phase, "spec_path": str(p), **over}, s

    HISTORY = [
        {"steps": {"voiceover": 100, "captions": 4, "qc": 6, "publish": 2, "build": 100}, "video_s": 50,
         "assets": [{"kind": "ltx", "s": 300}, {"kind": "text", "s": 10}]},
        {"steps": {"voiceover": 200, "captions": 6, "qc": 8, "publish": 4, "build": 400}, "video_s": 100,
         "assets": [{"kind": "ltx", "s": 500}, {"kind": "ltx", "s": 400}, {"kind": "text", "s": 30}]},
    ]

    def test_medians_of_this_hosts_history(self):
        job, s = self.job()
        now = 1_000_000.0
        m = vj.step_model(self.HISTORY)
        self.assertEqual(m["steps"]["voiceover"], 150)
        self.assertEqual(m["kinds"]["ltx"], 400)
        self.assertEqual(m["kinds"]["text"], 20)
        self.assertEqual(m["build_rate"], 3.0)                    # median of 2.0 and 4.0 s per video s
        video_s = vj.estimate(s)["video_s_approx"]
        warm = 150 + (400 + 20) + 5 + 3.0 * video_s + 7 + 3       # music off
        self.assertEqual(vj.eta_seconds(job, now=now, history=self.HISTORY, last_ltx=now - 60), int(warm))
        cold = vj.eta_seconds(job, now=now, history=self.HISTORY, last_ltx=now - 3600)
        self.assertEqual(cold, int(warm + vj.LTX_COLD_START_S))
        self.assertEqual(vj.eta_seconds(job, now=now, history=self.HISTORY, last_ltx=None), cold)  # never run

    def test_table_without_history_and_time_in_phase_counts_down(self):
        job, s = self.job()
        now = 1_000_000.0
        t = vj._STEP_SECONDS
        video_s = vj.estimate(s)["video_s_approx"]
        table = (t["voiceover"] + t["ltx"] + t["text"] + vj.LTX_COLD_START_S + t["captions"]
                 + vj._BUILD_S_PER_VIDEO_S * video_s + t["qc"] + t["publish"])
        self.assertEqual(vj.eta_seconds(job, now=now, history=[], last_ltx=None), int(table))
        build, _ = self.job("voiceover", phase_started_ts=now - 30)
        full, _ = self.job("voiceover", phase_started_ts=now)
        a = vj.eta_seconds(full, now=now, history=self.HISTORY, last_ltx=now)
        b = vj.eta_seconds(build, now=now, history=self.HISTORY, last_ltx=now)
        self.assertEqual(a - b, 30)
        done, _ = self.job("done")
        self.assertIsNone(vj.eta_seconds(done))

    def test_history_is_per_host_and_finished_jobs_only(self):
        for n, (host, phase) in enumerate([("a", "done"), ("b", "done"), ("a", "failed"), ("a", "done")]):
            d = vj.jobs_dir() / f"20260101-00000{n}-abcd"
            d.mkdir(parents=True)
            (d / "job.json").write_text(json.dumps({"id": d.name, "host": host, "phase": phase,
                                                    "timings": {"steps": {"voiceover": n}}}))
        self.assertEqual([h["steps"]["voiceover"] for h in vj.host_history("a")], [3, 0])


class TestEventsAndWarnings(Isolated):
    def test_status_tells_what_happened(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        run(job, FakeTools())
        s = vj.status(job["id"])
        kinds = [e["kind"] for e in s["events"]]
        for k in ("submitted", "phase", "voiced", "asset", "captions", "music"):
            self.assertIn(k, kinds)
        phases = [e["msg"] for e in s["events"] if e["kind"] == "phase"]
        self.assertEqual(phases, ["voiceover", "visuals", "captions", "music", "build", "qc", "publish", "done"])
        voiced = [e for e in s["events"] if e["kind"] == "voiced"]
        self.assertEqual(len(voiced), 3)
        self.assertEqual(voiced[1]["wpm"], 182.0)
        self.assertTrue(all("ts" in e for e in s["events"]))
        self.assertTrue(any("02" in w and "fast" in w for w in s["warnings"]))
        self.assertTrue(s["log_tail"])
        words = json.loads((Path(job["project_dir"]) / "captions" / "words_01.json").read_text())
        self.assertEqual(words[0]["text"], "hello there")          # the lone "/" is dropped
        t = vj.load_job(job["id"])["timings"]
        self.assertEqual({a["kind"] for a in t["assets"]}, {"ltx", "ideogram"})
        self.assertIn("build", t["steps"])

    def test_skipped_captions_become_a_warning_in_result(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        job["steps"]["captions"] = {"skipped": True, "reason": "error: no caption engine\nset a key"}
        job["output"] = {"size_bytes": 12_000_000, "over_size_cap": True}
        vj.save_job(job)
        r = vj.result(job["id"])
        self.assertTrue(any("Captions were skipped" in w and "ELEVENLABS_API_KEY" in w for w in r["warnings"]))
        self.assertTrue(any("12.0 MB" in w for w in r["warnings"]))


class TestExpander(Isolated):
    def test_a_refused_spec_goes_back_to_the_expander_once_with_the_error(self):
        script = self.tmp / "expander.py"
        script.write_text(
            "import json,sys\nreq=json.load(sys.stdin)\n"
            "prompt = 'slow pan over a harbour at dawn' if 'previous_error' in req else 'a laptop screen glowing'\n"
            "print(json.dumps({'title': 't', 'scenes': [{'text': 'Boats at first light.', "
            "'visual': {'kind': 'ltx', 'prompt': prompt}}]}))\n")
        os.environ["VIDEO_JOB_EXPANDER"] = f'"{sys.executable}" "{script}"'
        s, source = vj.resolve_spec("boats", None, "16:9", 10, allow_skeleton=False)
        self.assertEqual(source, "expander")
        self.assertIn("harbour", s["scenes"][0]["visual"]["prompt"])
        with self.assertRaises(vj.SpecError) as cm:
            vj.expand_brief("boats", "16:9", 10, attempts=1)
        self.assertIn("expander's spec was refused", str(cm.exception))
        self.assertIn("screen", vj.EXPANDER_INSTRUCTIONS)


class TestSmallSeams(Isolated):
    def test_output_must_be_an_mp4_inside_the_allowed_roots(self):
        with self.assertRaises(vj.SpecError):
            vj.resolve_output(str(self.tmp / "out.txt"))
        os.environ["VIDEO_JOB_OUTPUT_ROOTS"] = str(self.tmp / "files")
        try:
            self.assertTrue(vj.resolve_output(str(self.tmp / "files" / "a.mp4")).endswith("a.mp4"))
            with self.assertRaises(vj.SpecError):
                vj.resolve_output(str(self.tmp / "elsewhere" / "a.mp4"))
            with self.assertRaises(vj.SpecError):
                vj.resolve_output(str(self.tmp / "files" / ".." / "a.mp4"))
        finally:
            os.environ.pop("VIDEO_JOB_OUTPUT_ROOTS")

    def test_windows_command_split_drops_quotes_and_keeps_backslashes(self):
        with mock.patch.object(vj.os, "name", "nt"):
            self.assertEqual(vj.split_command(r'"C:\Program Files\py.exe" "C:\x\expand.py" --fast'),
                             [r"C:\Program Files\py.exe", r"C:\x\expand.py", "--fast"])
        self.assertEqual(vj.split_command("claude -p --output-format text"), ["claude", "-p", "--output-format", "text"])

    def test_size_cap_is_written_to_the_job_config(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        run(job, FakeTools())
        cfg = json.loads((Path(job["project_dir"]) / "config.json").read_text())
        self.assertEqual(cfg["encode"], {"preset": "veryfast", "maxMB": 9.5})
        self.assertEqual(cfg["music"]["file"], "audio/music.mp3")        # music on by default
        quiet = vj.submit({**spec(music=False), "allowCopy": True}, max_usd=5,
                          start_worker=False)   # the first job is done; same words on purpose
        run(quiet, FakeTools())
        qcfg = json.loads((Path(quiet["project_dir"]) / "config.json").read_text())
        self.assertIsNone(qcfg["music"]["file"])         # so qc.py doesn't warn about a missing bed
        self.assertIsNone(vj.max_mb({"maxMB": 0}))
        os.environ["VIDEO_JOB_MAX_MB"] = "24"
        try:
            self.assertEqual(vj.max_mb({}), 24.0)
        finally:
            os.environ.pop("VIDEO_JOB_MAX_MB")

    def test_build_bitrate_cap_maths(self):
        spec_ = importlib.util.spec_from_file_location(
            "ces_build", ROOT / "templates" / "concept-explainer-short" / "build.py")
        build = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(build)
        self.assertEqual(build.encode_params(50.0, None), [])
        p = build.encode_params(50.0, 9.5)
        kbps = int(9.5 * 8000 * 0.95 / 50) - 192
        self.assertEqual(p, ["-crf", "23", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k"])
        # the ceiling holds the file under the cap with the audio on top
        self.assertLess((kbps + 192) * 50 / 8 / 1000, 9.5)

    @unittest.skipIf(os.name == "nt", "process groups are POSIX")
    def test_cancel_stops_the_tool_the_worker_is_running(self):
        code = ("import subprocess,sys,time\n"
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                "print(p.pid,flush=True)\ntime.sleep(60)\n")
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        # argv carries "_worker <id>" like the real worker, which cancel checks for
        parent = subprocess.Popen([sys.executable, "-c", code, "_worker", job["id"]], stdout=subprocess.PIPE,
                                  text=True, start_new_session=True)
        child = int(parent.stdout.readline())
        job["worker_pid"] = parent.pid
        job["phase"] = "visuals"
        vj.save_job(job)
        vj.cancel(job["id"])
        parent.wait(timeout=5)
        for _ in range(50):
            if not vj.pid_alive(child):
                break
            time.sleep(0.1)
        self.assertFalse(vj.pid_alive(child))


if __name__ == "__main__":
    unittest.main()
