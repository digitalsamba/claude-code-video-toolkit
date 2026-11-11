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

    def test_bare_host_offers_what_needs_no_provider(self):
        h = vj.host_capabilities()
        self.assertEqual(h["available_visual_kinds"], ["text", "file", "none"])
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
        self.assertEqual(sorted(h["available_visual_kinds"]), ["file", "ideogram", "none", "text"])
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


class TestKitsAndLocal(Isolated):
    """A host with an ElevenLabs key and nothing else, and a kit made once in a session."""

    def setUp(self):
        super().setUp()
        os.environ.pop("VIDEO_JOB_SKIP_HOST_CHECK", None)
        os.environ["ELEVENLABS_API_KEY"] = "k"
        self.ws = self.tmp / "ws"
        kit = self.ws / "kits" / "candles"
        (kit / "assets").mkdir(parents=True)
        from PIL import Image
        Image.new("RGB", (64, 96), (200, 120, 40)).save(kit / "assets" / "jar.jpg")
        (kit / "brand.json").write_text(json.dumps({"colors": {"primary": "#C2410C", "bgDark": "#1C1917"}}))
        (kit / "kit.json").write_text(json.dumps({
            "description": "Small candle shop, warm and plain",
            "voice": {"provider": "elevenlabs", "voiceId": "v"}, "format": "9:16", "music": False,
            "describe": {"jar.jpg": "an amber candle jar"}, "focus": {"jar.jpg": 0.8}}))
        os.environ["VIDEO_JOB_WORKSPACE"] = str(self.ws)

    def tearDown(self):
        os.environ.pop("VIDEO_JOB_WORKSPACE", None)
        super().tearDown()

    def kit_spec(self):
        return {"title": "New candles", "kit": "candles", "scenes": [
            {"text": "Our autumn jars are poured by hand, one at a time.",
             "visual": {"kind": "file", "path": "jar.jpg", "headline": "AUTUMN JARS"}},
            {"text": "Each one burns for forty hours.", "visual": {"kind": "text", "headline": "40 HOURS"}},
            {"text": "Order by Friday for the weekend."}]}

    def test_kit_fills_voice_format_palette_and_finds_files(self):
        sp = vj.normalize_spec(self.kit_spec())
        self.assertEqual(sp["voice"]["provider"], "elevenlabs")
        self.assertEqual(sp["palette"]["accent1"], "#C2410C")
        self.assertFalse(sp["music"]["enabled"])
        first = sp["scenes"][0]
        self.assertTrue(first["visual"]["src"].endswith("jar.jpg"))
        self.assertEqual(first["asset"], "images/01_" + first["slug"] + ".jpg")
        self.assertEqual(first["overlay"]["headline"], "AUTUMN JARS")
        self.assertEqual(first["focus"], 0.8)
        # no FLUX here: the text scene and the scene with no visual draw locally, in the kit's ground colour
        for sc in sp["scenes"][1:]:
            self.assertEqual(sc["visual"]["source"], "local")
            self.assertEqual(sc["visual"]["color"], "#1C1917")
        self.assertEqual(vj.missing_for(sp), [])
        first_kit = vj.catalog()["kits"][0]
        self.assertEqual(first_kit["name"], "candles")   # workspace kits first
        self.assertEqual(first_kit["assets"], [{"file": "jar.jpg", "shows": "an amber candle jar", "usedRecently": 0}])

    def test_estimate_charges_nothing_for_local_and_own_files(self):
        est = vj.estimate(self.kit_spec())
        vis = [r for r in est["breakdown"] if r["step"] == "visuals"]
        self.assertEqual(vis[0]["count"], 3)
        self.assertEqual(vis[0]["usd"], 0.0)

    def test_visuals_step_copies_and_draws_without_a_tool(self):
        job = vj.submit(self.kit_spec(), max_usd=5, start_worker=False)
        spec = json.loads((vj.job_dir(job["id"]) / "spec.json").read_text())
        job["project_dir"] = str(vj.create_project(job, spec))
        proj = vj._project(job)
        calls = []
        real = vj.run_tool
        vj.run_tool = lambda *a, **k: calls.append(a) or {"returncode": 1}
        try:
            res = vj.step_visuals(job, spec)
        finally:
            vj.run_tool = real
        self.assertEqual(calls, [])
        self.assertEqual(len(res["assets"]), 3)
        from PIL import Image
        bg = Image.open(proj / spec["scenes"][1]["asset"])
        self.assertEqual(bg.size, (720, 1280))
        # the kit's ground colour, a step off the caption pill when the pill is drawn in the same ink
        want = vj._hex_rgb(vj.field_clear_of("#1C1917", (spec.get("palette") or {}).get("ink") or "#0A0A14")
                           if spec["captions"]["enabled"] else "#1C1917")
        r, g, b = bg.resize((1, 1)).getpixel((0, 0))
        self.assertLess(abs(r - want[0]) + abs(g - want[1]) + abs(b - want[2]), 12)
        vj.cancel(job["id"])

    def test_resume_arithmetic_prices_own_files_and_local_backgrounds_at_zero(self):
        # the worker prices what is left before each run (caught by a real job: ValueError: file)
        job = vj.submit(self.kit_spec(), max_usd=5, start_worker=False)
        spec = json.loads((vj.job_dir(job["id"]) / "spec.json").read_text())
        job["project_dir"] = str(vj.create_project(job, spec))
        self.assertEqual(vj.remaining_estimates(job, spec)["visuals"], 0.0)
        vj.cancel(job["id"])

    def test_explicit_backgrounds_flux_still_needs_flux(self):
        sp = self.kit_spec()
        sp["backgrounds"] = "flux"
        miss = vj.missing_for(vj.normalize_spec(sp))
        self.assertTrue(any("FLUX" in m["needed_for"] for m in miss))

    def test_missing_file_and_unknown_kit_are_spec_errors(self):
        sp = self.kit_spec()
        sp["scenes"][0]["visual"]["path"] = "nope.jpg"
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec(sp)
        with self.assertRaises(vj.SpecError) as ctx:
            vj.normalize_spec({**self.kit_spec(), "kit": "nobody"})
        self.assertIn("candles", str(ctx.exception))

    def test_cli_kit_flag_takes_the_kit_format_voice_and_files(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = vj.main(["spec", "--kit", "candles", "--title", "Jars",
                          "--scene", "Poured by hand, one at a time. :: file: jar.jpg ; AUTUMN JARS"])
        self.assertEqual(rc, 0)
        sp = json.loads(out.getvalue())
        self.assertEqual(sp["kit"]["name"], "candles")
        self.assertEqual(sp["voice"]["provider"], "elevenlabs")
        self.assertTrue(sp["scenes"][0]["visual"]["src"].endswith("jar.jpg"))

    def test_expander_is_told_the_kit_and_its_assets(self):
        seen = {}
        real = vj.subprocess.run

        def fake_run(args, input=None, **kw):
            seen.update(json.loads(input))
            spec = {"title": "Jars", "scenes": [{"text": "Poured by hand, one at a time.",
                                                 "visual": {"kind": "file", "path": "jar.jpg"}}]}
            return type("R", (), {"returncode": 0, "stdout": json.dumps(spec), "stderr": ""})()
        os.environ["VIDEO_JOB_EXPANDER"] = "fake"
        vj.subprocess.run = fake_run
        try:
            sp = vj.expand_brief("a post about the autumn jars", "9:16", 20, kit="candles")
        finally:
            vj.subprocess.run = real
            os.environ.pop("VIDEO_JOB_EXPANDER", None)
        self.assertEqual(seen["kit"]["assets"], [{"file": "jar.jpg", "shows": "an amber candle jar", "usedRecently": 0}])
        self.assertIn("kit.assets", seen["instructions"])
        self.assertEqual(sp["kit"]["name"], "candles")

    def test_flat_form_file_scene(self):
        sp = vj.spec_from_scene_args("Jars", ["Poured by hand. :: file: jar.jpg ; AUTUMN JARS ; small batch"], "9:16")
        sp["kit"] = "candles"
        n = vj.normalize_spec(sp)
        self.assertEqual(n["scenes"][0]["visual"]["kind"], "file")
        self.assertEqual(n["scenes"][0]["overlay"]["subline"], "small batch")

    def test_files_from_recent_videos_are_counted(self):
        job = vj.submit(self.kit_spec(), max_usd=5, start_worker=False)
        jf = vj.job_dir(job["id"]) / "job.json"
        j = json.loads(jf.read_text()); j["phase"] = "done"; jf.write_text(json.dumps(j))
        self.assertEqual(vj.catalog()["kits"][0]["assets"][0]["usedRecently"], 1)


class TestCopy(Isolated):
    def test_dashes_and_stock_phrases_are_refused_before_spend(self):
        sp = spec()
        sp["scenes"][0]["text"] = "Light bends in the air — and it's worth noting the sky is blue."
        with self.assertRaises(vj.SpecError) as ctx:
            vj.submit(sp, max_usd=5, start_worker=False)
        msg = str(ctx.exception)
        self.assertIn("dash", msg)
        self.assertIn("worth noting", msg)
        sp2 = spec(); sp2["title"] = "Turas — a daily walk"
        with self.assertRaises(vj.SpecError) as ctx2:
            vj.submit(sp2, max_usd=5, start_worker=False)
        self.assertIn("the title", str(ctx2.exception))
        self.assertFalse(any((self.tmp / "jobs").glob("*/job.json")))

    def test_a_line_from_a_finished_job_is_refused(self):
        job = vj.submit(spec(), max_usd=5, start_worker=False)
        jf = vj.job_dir(job["id"]) / "job.json"
        j = json.loads(jf.read_text()); j["phase"] = "done"; jf.write_text(json.dumps(j))
        os.environ.pop("VIDEO_JOB_LOCK_FAKE", None)
        vj.clear_stale_lock()
        with self.assertRaises(vj.SpecError) as ctx:
            vj.submit(spec(), max_usd=5, start_worker=False)
        self.assertIn("repeats a line from job " + job["id"], str(ctx.exception))


class TestInputsAndReview(TestKitsAndLocal):
    """Fixes from the PR review: where files may come from, what the worker re-reads, and the expander loop."""

    def test_a_file_outside_the_kit_workspace_and_cwd_is_refused(self):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as other:
            secret = Path(other) / "private.jpg"
            Image.new("RGB", (8, 8)).save(secret)
            sp = self.kit_spec(); sp["scenes"][0]["visual"]["path"] = str(secret)
            with self.assertRaises(vj.SpecError) as ctx:
                vj.normalize_spec(sp)
            self.assertIn("outside the kit", str(ctx.exception))
            sp = self.kit_spec(); sp["kit"] = {"dir": other, "assets": other}
            with self.assertRaises(vj.SpecError):             # a hand-made kit dict is read from its kit.json
                vj.normalize_spec(sp)

    def test_the_worker_keeps_the_path_submit_resolved(self):
        n = vj.normalize_spec(self.kit_spec())
        n["scenes"][0]["visual"]["path"] = "only-resolvable-at-submit.jpg"
        vj._TRUSTED_SPEC = True
        try:
            again = vj.normalize_spec(n)
        finally:
            vj._TRUSTED_SPEC = False
        self.assertEqual(again["scenes"][0]["visual"]["src"], n["scenes"][0]["visual"]["src"])
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec(n)                                # outside the worker, it is resolved again

    def test_source_and_focus_are_checked(self):
        sp = self.kit_spec()
        sp["scenes"][0]["visual"].pop("focus", None)
        sp["scenes"][0]["focus"] = 0.2                           # scene-level, as the skill says
        sp["scenes"][2]["visual"] = {"kind": "ideogram", "source": "local", "caption": {"headline": "FRIDAY"}}
        sp["scenes"][2]["focus"] = "right"
        n = vj.normalize_spec(sp)
        self.assertEqual(n["scenes"][0]["focus"], 0.2)
        self.assertNotIn("source", n["scenes"][2]["visual"])
        self.assertNotIn("focus", n["scenes"][2])
        sp["scenes"][1]["visual"]["source"] = "gpu"
        with self.assertRaises(vj.SpecError):
            vj.normalize_spec(sp)

    def test_used_recently_counts_by_folder_and_file_name(self):
        kj = self.ws / "kits" / "candles" / "kit.json"
        k = json.loads(kj.read_text()); k["name"] = "The Candle Shop"; kj.write_text(json.dumps(k))
        sp = self.kit_spec(); sp["scenes"][0]["visual"]["path"] = "assets/jar.jpg"
        job = vj.submit(sp, max_usd=5, start_worker=False)
        jf = vj.job_dir(job["id"]) / "job.json"
        j = json.loads(jf.read_text()); j["phase"] = "done"; jf.write_text(json.dumps(j))
        self.assertEqual(vj.catalog()["kits"][0]["assets"][0]["usedRecently"], 1)

    def test_the_expander_is_asked_again_about_its_copy(self):
        calls = []
        real = vj.subprocess.run

        def fake_run(args, input=None, **kw):
            req = json.loads(input)
            calls.append(req)
            line = ("In today's fast-paced world, jars matter." if len(calls) == 1
                    else "Poured by hand, one at a time.")
            spec = {"title": "Jars", "scenes": [{"text": line, "visual": {"kind": "file", "path": "jar.jpg"}}]}
            return type("R", (), {"returncode": 0, "stdout": json.dumps(spec), "stderr": ""})()
        os.environ["VIDEO_JOB_EXPANDER"] = "fake"
        vj.subprocess.run = fake_run
        try:
            sp = vj.expand_brief("a post about the autumn jars", "9:16", 20, kit=str(self.ws / "kits" / "candles"))
        finally:
            vj.subprocess.run = real
            os.environ.pop("VIDEO_JOB_EXPANDER", None)
        self.assertEqual(len(calls), 2)
        self.assertIn("stock AI phrasing", calls[1]["previous_error"])
        self.assertEqual(calls[0]["kit"]["description"], "Small candle shop, warm and plain")   # kit named by path
        self.assertEqual(sp["scenes"][0]["text"], "Poured by hand, one at a time.")


if __name__ == "__main__":
    import unittest
    unittest.main()
