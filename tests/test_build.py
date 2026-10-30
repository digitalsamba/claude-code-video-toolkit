"""Exercise real local subprocesses; no generation services or network required."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build import TOOLKIT, builtin_plan
from production import BuildError, build_lock, canonical, load_state, plan_build, run_build, validate_plan


def task(task_id, code, outputs, **kwargs):
    return {"id": task_id, "command": ["{python}", "-c", code], "outputs": outputs, **kwargs}


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)

    def run_plan(self, tasks, **kwargs):
        return run_build(self.project, {"version": 1, "tasks": tasks}, TOOLKIT, **kwargs)

    def cli(self, *args):
        return subprocess.run([sys.executable, str(TOOLKIT / "tools/build.py"), "--project", str(self.project),
                               *args], capture_output=True, text=True, encoding="utf-8")

    def test_plan_and_dry_run_do_not_write_or_execute(self):
        plan = {"version": 1, "tasks": [task("a", "from pathlib import Path; Path('out').write_text('ran')", ["out"])]}
        plan_path = self.project / "production.json"
        plan_path.write_text(json.dumps(plan))
        before = sorted(p.name for p in self.project.iterdir())
        for flags in [("--json",), ("--run", "--dry-run", "--json")]:
            result = self.cli("--plan", str(plan_path), *flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["tasks"][0]["action"], "run")
        self.assertEqual(sorted(p.name for p in self.project.iterdir()), before)

    def test_failure_checkpoint_resumes_successful_dependencies(self):
        first = task("first", "from pathlib import Path; p=Path('count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); Path('a').write_text('a')", ["a"])
        second = task("second", "from pathlib import Path; import sys; sys.exit(3) if not Path('gate').exists() else Path('b').write_text(Path('a').read_text()+'b')", ["b"], dependsOn=["first"])
        final = task("final", "from pathlib import Path; Path('video').write_text(Path('b').read_text()+'c')", ["video"], dependsOn=["second"])
        failed = self.run_plan([final, second, first])
        self.assertFalse(failed["success"])
        self.assertEqual(failed["failed"]["id"], "second")
        self.assertFalse((self.project / "video").exists())
        self.assertEqual(load_state(self.project)["tasks"]["second"]["status"], "failed")
        (self.project / "gate").write_text("ready")
        resumed = self.run_plan([final, second, first])
        self.assertTrue(resumed["success"], resumed)
        self.assertEqual(resumed["resumed"], ["first"])
        self.assertEqual((self.project / "count").read_text(), "1")
        self.assertEqual((self.project / "video").read_text(), "abc")
        self.assertEqual(load_state(self.project)["tasks"]["second"]["attempts"], 2)
        self.assertEqual(self.run_plan([final, second, first])["resumed"], ["first", "second", "final"])

    def test_restart_and_missing_output_rerun_dependencies(self):
        tasks = [task("a", "from pathlib import Path; Path('a').write_text('done')", ["a"]),
                 task("b", "from pathlib import Path; Path('b').write_text(Path('a').read_text())", ["b"], dependsOn=["a"])]
        self.assertTrue(self.run_plan(tasks)["success"])
        (self.project / "a").unlink()
        self.assertEqual(self.run_plan(tasks)["completed"], ["a", "b"])
        self.assertEqual(self.run_plan(tasks, restart=True)["completed"], ["a", "b"])

    def test_output_verification_and_timeout(self):
        missing = self.run_plan([task("missing", "pass", ["missing"])])
        self.assertFalse(missing["success"])
        self.assertIn("expected nonempty output", missing["failed"]["error"])
        timeout = self.run_plan([task("timeout", "import time; time.sleep(1)", ["out"], timeoutSeconds=0.05)])
        self.assertFalse(timeout["success"])
        self.assertEqual(load_state(self.project)["tasks"]["timeout"]["status"], "failed")

    def test_shell_metacharacters_are_literal_and_stdout_is_an_output(self):
        special = "hello & echo dangerous; $(whoami) > injected.txt"
        t = {"id": "literal", "command": ["{python}", "-c", "import sys; print(sys.argv[1])", special],
             "outputs": [], "stdoutFile": "reports/literal.txt"}
        self.assertTrue(self.run_plan([t])["success"])
        self.assertEqual((self.project / "reports/literal.txt").read_text().strip(), special)
        self.assertFalse((self.project / "injected.txt").exists())
        self.assertEqual(self.run_plan([t])["resumed"], ["literal"])

    def test_lock_blocks_another_process_and_releases_after_error(self):
        plan = {"version": 1, "tasks": [task("a", "print('ok')", [], stdoutFile="out")]}
        path = self.project / "production.json"
        path.write_text(json.dumps(plan))
        with build_lock(self.project):
            result = self.cli("--plan", str(path), "--run", "--json")
            self.assertEqual(result.returncode, 2)
            self.assertIn("Another build", json.loads(result.stdout)["error"])
        self.assertEqual(self.cli("--plan", str(path), "--run", "--json").returncode, 0)

    def test_invalid_graphs_and_paths(self):
        cases = [
            [task("a", "pass", [], dependsOn=["missing"])],
            [task("a", "pass", [], dependsOn=["b"]), task("b", "pass", [], dependsOn=["a"])],
            [task("a", "pass", []), task("a", "pass", [])],
            [task("a", "pass", ["../escape"])],
            [task("a", "pass", ["out"], cwd="../")],
            [task("a", "pass", ["out"]), task("b", "pass", ["./out"])],
            [task("a", "pass", [".video-toolkit/build-state.json"])],
            [task("a", "pass", ["*.mp4"])],
        ]
        for tasks in cases:
            with self.subTest(tasks=tasks), self.assertRaises(BuildError):
                validate_plan({"version": 1, "tasks": tasks}, self.project)
        self.assertFalse((self.project / ".video-toolkit").exists())

    def test_changed_definition_invalidates_dependents(self):
        t = task("a", "print('one')", [], stdoutFile="a")
        self.run_plan([t])
        t["command"][2] = "print('two')"
        self.assertEqual(self.run_plan([t])["completed"], ["a"])
        self.assertEqual((self.project / "a").read_text().strip(), "two")

    def test_corrupt_checkpoint_is_reported_without_replacement(self):
        directory = self.project / ".video-toolkit"
        directory.mkdir()
        state = directory / "build-state.json"
        state.write_text("broken")
        with self.assertRaises(BuildError):
            self.run_plan([task("a", "print('ok')", [], stdoutFile="out")])
        self.assertEqual(state.read_text(), "broken")

    def test_builtin_wires_existing_tools_to_real_project_paths(self):
        (self.project / "src/config").mkdir(parents=True)
        (self.project / "src/config/demo-config.ts").write_text("export const demoConfig = {}")
        scene_dir = self.project / "public/audio/scenes"
        scene_dir.mkdir(parents=True)
        (scene_dir / "01_intro.txt").write_text("Hello")
        (self.project / "package.json").write_text(json.dumps({"scripts": {"render": "remotion render Demo out/video.mp4"}}))
        plan = builtin_plan(self.project, provider="qwen3", captions=True)
        tasks = {t["id"]: t for t in plan["tasks"]}
        voice_id = next(key for key in tasks if key.startswith("voiceover-"))
        captions_id = next(key for key in tasks if key.startswith("captions-"))
        self.assertEqual(tasks["timing"]["dependsOn"], [voice_id])
        self.assertIn("--engine", tasks[captions_id]["command"])
        self.assertIn("tools/pacing.py", tasks[voice_id]["toolkitInputs"])
        self.assertEqual(tasks["render"]["outputs"], ["out/video.mp4"])
        preview = plan_build(self.project, plan, TOOLKIT)
        self.assertTrue(all(Path(t["cwd"]) == canonical(self.project) for t in preview["tasks"]))
        self.assertFalse((self.project / ".video-toolkit").exists())
        self.assertFalse(any("publish" in str(t["command"]) for t in preview["tasks"]))

    def test_content_change_rebuilds_only_affected_branch_and_dependents(self):
        for name in ("left.txt", "right.txt"):
            (self.project / name).write_text(name)
        tasks = [task("left", "from pathlib import Path; print(Path('left.txt').read_text())", [],
                      inputs=["left.txt"], stdoutFile="left.out"),
                 task("right", "from pathlib import Path; print(Path('right.txt').read_text())", [],
                      inputs=["right.txt"], stdoutFile="right.out"),
                 task("final", "from pathlib import Path; print(Path('left.out').read_text()+Path('right.out').read_text())", [],
                      inputs=["left.out", "right.out"], dependsOn=["left", "right"], stdoutFile="final.out")]
        self.assertTrue(self.run_plan(tasks)["success"])
        stamp = (self.project / "left.txt").stat().st_mtime_ns
        (self.project / "left.txt").write_text("new content")
        os.utime(self.project / "left.txt", ns=(stamp, stamp))
        result = self.run_plan(tasks)
        self.assertEqual(result["completed"], ["left", "final"])
        self.assertEqual(result["resumed"], ["right"])
        self.assertIn("new content", (self.project / "final.out").read_text())
        os.utime(self.project / "left.txt", None)
        self.assertEqual(self.run_plan(tasks)["completed"], [])

    def test_output_tampering_and_forced_task_invalidate_dependents(self):
        tasks = [task("a", "print('original')", [], stdoutFile="a"),
                 task("b", "print('final')", [], dependsOn=["a"], stdoutFile="b")]
        self.run_plan(tasks)
        (self.project / "a").write_text("tampered")
        self.assertEqual(self.run_plan(tasks)["completed"], ["a", "b"])
        self.assertEqual(self.run_plan(tasks, force_tasks=["a"])["completed"], ["a", "b"])
        with self.assertRaises(BuildError):
            self.run_plan(tasks, force_tasks=["missing"])

    def test_glob_membership_missing_inputs_and_tool_source_changes(self):
        code = self.project / "generate.py"
        code.write_text("print('one')")
        tasks = [{"id": "a", "command": ["{python}", "{project}/generate.py"], "outputs": [],
                  "stdoutFile": "out", "inputs": ["notes/*.txt", "optional.txt"]}]
        self.run_plan(tasks)
        (self.project / "notes").mkdir()
        (self.project / "notes/new.txt").write_text("new")
        self.assertEqual(self.run_plan(tasks)["completed"], ["a"])
        (self.project / "optional.txt").write_text("present")
        self.assertEqual(self.run_plan(tasks)["completed"], ["a"])
        (self.project / "optional.txt").unlink()
        self.assertEqual(self.run_plan(tasks)["completed"], ["a"])
        code.write_text("print('two')")
        self.assertEqual(self.run_plan(tasks)["completed"], ["a"])
        self.assertEqual((self.project / "out").read_text().strip(), "two")

    def test_toolkit_and_environment_inputs_invalidate_without_recording_secrets(self):
        toolkit = self.project / "toolkit"
        toolkit.mkdir()
        setting = toolkit / "voice.json"
        setting.write_text('{"speed":1}')
        tasks = [task("a", "print('done')", [], stdoutFile="out", toolkitInputs=["voice.json"],
                      environmentInputs=["TOOLKIT_TEST_VOICE"])]
        plan = {"version": 1, "tasks": tasks}
        with patch.dict(os.environ, {"TOOLKIT_TEST_VOICE": "secret-one"}):
            run_build(self.project, plan, toolkit)
            self.assertEqual(run_build(self.project, plan, toolkit)["completed"], [])
        with patch.dict(os.environ, {"TOOLKIT_TEST_VOICE": "secret-two"}):
            self.assertEqual(run_build(self.project, plan, toolkit)["completed"], ["a"])
            setting.write_text('{"speed":2}')
            self.assertEqual(run_build(self.project, plan, toolkit)["completed"], ["a"])
        self.assertNotIn("secret-", (self.project / ".video-toolkit/build-state.json").read_text())

    def test_in_place_inputs_snapshot_after_success(self):
        (self.project / "config").write_text("first")
        tasks = [task("timing", "from pathlib import Path; Path('config').write_text('synced')", ["config"], inputs=["config"])]
        self.run_plan(tasks)
        self.assertEqual(self.run_plan(tasks)["resumed"], ["timing"])

    def test_runtime_manifest_validation_and_portable_path_guards(self):
        (self.project / "project.json").write_text(json.dumps({"name": "bad", "scenes": []}))
        result = self.cli("--json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("template", json.loads(result.stdout)["error"])
        for value in ("C:relative", "name\nfile", "../escape", "https://example.test/a"):
            with self.subTest(path=value), self.assertRaises(BuildError):
                validate_plan({"version": 1, "tasks": [task("a", "pass", [], inputs=[value])]}, self.project)
        with self.assertRaises(BuildError):
            validate_plan({"version": True, "tasks": [task("a", "pass", [])]}, self.project)


if __name__ == "__main__":
    unittest.main()
