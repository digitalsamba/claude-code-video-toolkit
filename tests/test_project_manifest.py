"""Offline manifest boundary and real CLI tests. Run with unittest discovery."""
from __future__ import annotations

import copy
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from project_manifest import (  # noqa: E402
    ProjectValidationError, SCHEMA_PATH, load_project, resolve_project_path,
    save_project, validate_project,
)


def minimal() -> dict:
    return {"name": "Example", "template": "custom-template", "scenes": []}


class CanonicalPathTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows aliases')
    def test_extended_prefix_does_not_reject_a_valid_project_reference(self):
        root = Path('C:/toolkit/project')
        target = Path('\\\\?\\C:\\toolkit\\project\\public\\image.png')
        with patch.object(Path, 'resolve', side_effect=[root, target]):
            self.assertEqual(resolve_project_path(root, 'public/image.png'), root / 'public/image.png')


class ManifestTests(unittest.TestCase):
    def test_legacy_round_trip_preserves_unknown_fields_without_mutating_input(self):
        original = json.loads((ROOT / "tests/fixtures/project-legacy.json").read_text())
        before = copy.deepcopy(original)
        normalized = validate_project(original)
        self.assertEqual(original, before)
        self.assertEqual(normalized.pop("schemaVersion"), 1)
        self.assertEqual(normalized, original)
        with tempfile.TemporaryDirectory() as directory:
            save_project(directory, original)
            loaded = load_project(directory)
            self.assertEqual(loaded, dict(original, schemaVersion=1))

    def test_custom_scene_and_visual_types_and_per_scene_audio(self):
        project = minimal()
        project.update(engine="moviepy", format={"width": 1080, "height": 1920, "fps": 30})
        project["scenes"] = [{"id": "first", "type": "chapter", "durationSeconds": 2.5,
                              "visual": {"type": "synthetic-ui", "assets": ["images/a.png"]},
                              "audio": {"file": "audio/scenes/first.mp3", "durationSeconds": 2.2},
                              "scriptFile": "public/audio/scenes/first.txt", "narration": None}]
        self.assertEqual(validate_project(project)["engine"], "moviepy")
        project["format"] = "9:16"
        validate_project(project)

    def assert_invalid(self, project: object, path: str):
        with self.assertRaises(ProjectValidationError) as caught:
            validate_project(project)
        self.assertIn(path, [error["path"] for error in caught.exception.errors])

    def test_schema_errors_report_field_paths(self):
        project = minimal()
        project["schemaVersion"] = 2
        project["scenes"] = [{"id": "a", "type": "hook", "durationSeconds": 0}]
        with self.assertRaises(ProjectValidationError) as caught:
            validate_project(project)
        paths = {error["path"] for error in caught.exception.errors}
        self.assertEqual(paths, {"/schemaVersion", "/scenes/0/durationSeconds"})
        self.assert_invalid([], "/")
        self.assert_invalid({"name": "missing template"}, "/template")

    def test_duplicate_scene_ids(self):
        project = minimal()
        project["scenes"] = [{"id": "same", "type": "hook"}, {"id": "same", "type": "custom"}]
        self.assert_invalid(project, "/scenes/1/id")

    def test_invalid_numbers_and_timestamps(self):
        for value in (-1, 0, True, "3", float("nan"), float("inf")):
            with self.subTest(value=value):
                project = minimal()
                project["scenes"] = [{"id": "a", "type": "title", "durationSeconds": value}]
                with self.assertRaises(ProjectValidationError):
                    validate_project(project)
        for value in ("yesterday", "2026-10-31T12:00:00", "2026-02-30T12:00:00Z"):
            project = dict(minimal(), created=value)
            self.assert_invalid(project, "/created")
        project = dict(minimal(), format={"width": 0, "fps": -1})
        self.assert_invalid(project, "/format/width")

    def test_cross_platform_path_containment(self):
        bad_paths = ("../secret.mp4", "public/../../secret", "/tmp/out.mp4", "C:\\temp\\out.mp4",
                     "C:out.mp4", "\\\\server\\share\\out.mp4", "https://example.com/a", "a\nfile", ".")
        for value in bad_paths:
            with self.subTest(value=value):
                self.assert_invalid(dict(minimal(), render={"file": value}), "/render/file")
        for value in ("out/a.mp4", "out\\a.mp4", "public/fonts/Café.ttf"):
            validate_project(dict(minimal(), render={"file": value}))
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_project_path(directory, "out/a.mp4"), Path(directory).resolve() / "out/a.mp4")
            self.assertEqual(resolve_project_path(directory, ".", allow_root=True), Path(directory).resolve())

    def test_escaping_symlink_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            link = Path(directory) / "linked"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("This environment cannot create symlinks")
            with self.assertRaises(ProjectValidationError):
                resolve_project_path(directory, "linked/video.mp4")

    def test_production_graph_and_paths(self):
        project = dict(minimal(), production={"version": 1, "tasks": [
            {"id": "voice", "command": ["{python}", "{toolkit}/tools/voiceover.py"],
             "inputs": ["public/audio/*.txt"], "outputs": ["public/audio/one.mp3"], "cwd": "."},
            {"id": "qc", "command": ["{python}", "{toolkit}/tools/qc.py"], "dependsOn": ["voice"],
             "outputs": [], "stdoutFile": "out/qc.json", "estimatedCostUsd": 0}
        ]})
        validate_project(project)
        project["production"]["tasks"][0]["cwd"] = "public/"
        validate_project(project)
        project["production"]["tasks"][1]["dependsOn"] = ["missing"]
        self.assert_invalid(project, "/production/tasks/1/dependsOn/0")
        project["production"]["tasks"][1]["dependsOn"] = ["voice"]
        project["production"]["tasks"][0]["dependsOn"] = ["qc"]
        self.assert_invalid(project, "/production/tasks")
        project["production"]["tasks"][0].pop("dependsOn")
        project["production"]["tasks"][0]["outputs"] = ["out/*.mp3"]
        self.assert_invalid(project, "/production/tasks/0/outputs/0")
        project["production"]["tasks"][0]["outputs"] = []
        project["production"]["tasks"][1]["id"] = "voice"
        self.assert_invalid(project, "/production/tasks/1/id")

    def test_bad_json_duplicate_properties_and_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "project.json"
            for value in ('{"name":', '{"name":"a","name":"b"}'):
                target.write_text(value, encoding="utf-8")
                with self.assertRaises(ProjectValidationError):
                    load_project(target)
            target.write_text(json.dumps(minimal()), encoding="utf-8-sig")
            self.assertEqual(load_project(target)["schemaVersion"], 1)

    def test_json_integer_semantics_do_not_accept_booleans(self):
        validate_project(dict(minimal(), schemaVersion=1.0, format={"width": 1080.0}))
        self.assert_invalid(dict(minimal(), schemaVersion=True), "/schemaVersion")
        self.assert_invalid(dict(minimal(), format={"width": True}), "/format/width")

    def test_failed_save_does_not_damage_existing_file_or_leave_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            target = save_project(directory, minimal())
            original = target.read_bytes()
            with self.assertRaises(ProjectValidationError):
                save_project(target, {"name": "broken"})
            self.assertEqual(target.read_bytes(), original)
            with patch("project_manifest.os.replace", side_effect=OSError("blocked")):
                with self.assertRaises(OSError):
                    save_project(target, minimal())
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_schema_uses_only_implemented_keywords(self):
        supported = {"$schema", "$id", "$ref", "$defs", "title", "description", "default", "type",
                     "required", "properties", "items", "enum", "minimum", "exclusiveMinimum",
                     "minLength", "maxLength", "minItems", "pattern", "format", "anyOf"}
        def check(schema):
            self.assertFalse(set(schema) - supported)
            for key in ("$defs", "properties"):
                for child in schema.get(key, {}).values():
                    check(child)
            if "items" in schema:
                check(schema["items"])
            for child in schema.get("anyOf", []):
                check(child)
        check(json.loads(SCHEMA_PATH.read_text()))


class CLITests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "tools/project.py"), *arguments],
                              text=True, encoding="utf-8", capture_output=True, cwd=ROOT.parent)

    def test_validate_and_migrate_from_outside_toolkit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "project.json"
            target.write_text(json.dumps(minimal()), encoding="utf-8")
            original = target.read_bytes()
            for command in ("validate", "migrate"):
                result = self.run_cli(command, directory, "--json")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(json.loads(result.stdout)["valid"])
                self.assertFalse(json.loads(result.stdout)["written"])
                self.assertEqual(target.read_bytes(), original)
            result = self.run_cli("migrate", str(target), "--write", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["written"])
            self.assertEqual(json.loads(target.read_text())["schemaVersion"], 1)
            first = target.read_bytes()
            result = self.run_cli("migrate", directory, "--write", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_bytes(), first)

    def test_validation_error_and_missing_file_are_json_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "project.json"
            for contents in (None, "{", json.dumps(dict(minimal(), output="../escape.mp4"))):
                if contents is not None:
                    target.write_text(contents)
                result = self.run_cli("validate", str(target), "--json")
                self.assertEqual(result.returncode, 1)
                self.assertFalse(json.loads(result.stdout)["valid"])
                self.assertTrue(json.loads(result.stdout)["errors"])
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
