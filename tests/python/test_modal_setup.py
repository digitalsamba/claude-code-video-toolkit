"""modal_setup.py on a workspace whose image builder version the Modal client refuses.

    uv run python -m unittest tests/python/test_modal_setup.py -v

The deploy subprocess, the URL lookup and the .env write are faked: nothing here talks to Modal.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import modal_setup as ms  # noqa: E402

# What modal 1.5.0's CLI prints, piped (80 columns), for a workspace created in November 2026:
# the phrase the tool looks for is split across two panel lines.
REFUSED_DEFAULT = """\
┌─ Error ──────────────────────────────────────────────────────────────────────┐
│ This version of the modal client supports the following image builder        │
│ versions: {'2024.10', '2023.12', '2025.06', '2024.04'}.                      │
│                                                                              │
│ You are using '2026.10'. Please update your client library (pip install      │
│ --upgrade modal).                                                            │
└──────────────────────────────────────────────────────────────────────────────┘
"""
# The same, without rich (the CLI then writes the bare message).
REFUSED_PLAIN = (
    "This version of the modal client supports the following image builder versions: "
    "{'2024.10', '2023.12', '2025.06', '2024.04'}.\n\n"
    "You are using '2026.10'. Please update your client library (pip install --upgrade modal).\n"
)
# The version came from the user: MODAL_IMAGE_BUILDER_VERSION, or ~/.modal.toml.
REFUSED_ENV = """\
┌─ Error ──────────────────────────────────────────────────────────────────────┐
│ This version of the modal client supports the following image builder        │
│ versions: {'2024.10', '2023.12', '2025.06', '2024.04'}.                      │
│                                                                              │
│ You are using '2026.10' (based on your `MODAL_IMAGE_BUILDER_VERSION`         │
│ environment variable). Please update or remove your local configuration.     │
└──────────────────────────────────────────────────────────────────────────────┘
"""
REFUSED_TOML = """\
┌─ Error ──────────────────────────────────────────────────────────────────────┐
│ This version of the modal client supports the following image builder        │
│ versions: {'2024.10', '2023.12', '2025.06', '2024.04'}.                      │
│                                                                              │
│ You are using '2026.10' (based on your local config file at                  │
│ `/home/you/.modal.toml`). Please update or remove your local configuration.  │
└──────────────────────────────────────────────────────────────────────────────┘
"""


class FakeModal:
    """Stands in for `_run_streamed`: answers each call from a script, records the env it got."""

    def __init__(self, *answers: tuple[int, str]):
        self.answers = list(answers)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, env):
        self.calls.append((cmd, dict(env)))
        return self.answers.pop(0) if self.answers else (0, "deployed\n")

    def builder_versions(self) -> list[str | None]:
        return [env.get(ms.BUILDER_VAR) for _, env in self.calls]


class BuilderVersionTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop(ms.BUILDER_VAR, None)
        self._real = (ms._run_streamed, ms.lookup_url, ms.write_env_value)
        ms.lookup_url = lambda app: (f"https://example--{app.app_name}.modal.run", None)
        self.written: dict[str, str] = {}
        ms.write_env_value = self.written.__setitem__

    def tearDown(self):
        ms._run_streamed, ms.lookup_url, ms.write_env_value = self._real
        os.environ.pop(ms.BUILDER_VAR, None)
        if self._saved is not None:
            os.environ[ms.BUILDER_VAR] = self._saved

    def deploy(self, fake: FakeModal, *tools: str) -> list[dict]:
        ms._run_streamed = fake
        return [ms.deploy(ms.APPS[t]) for t in tools]

    def test_refused_default_retries_once_and_the_run_keeps_the_fallback(self):
        fake = FakeModal((1, REFUSED_DEFAULT))
        first, second = self.deploy(fake, "qwen3_tts", "flux2")
        self.assertTrue(first["success"] and second["success"])
        # qwen3_tts fails, is retried with the fallback; flux2 starts with it.
        self.assertEqual(fake.builder_versions(), [None, "2025.06", "2025.06"])
        self.assertEqual(first["image_builder_version"], "2025.06")
        self.assertEqual(second["image_builder_version"], "2025.06")
        self.assertIn("MODAL_FLUX2_ENDPOINT_URL", self.written)

    def test_the_bare_message_falls_back_too(self):
        fake = FakeModal((1, REFUSED_PLAIN))
        (result,) = self.deploy(fake, "music_gen")
        self.assertTrue(result["success"])
        self.assertEqual(fake.builder_versions(), [None, "2025.06"])

    def test_soulx_weights_step_falls_back_before_its_deploy(self):
        fake = FakeModal((1, REFUSED_DEFAULT))
        (result,) = self.deploy(fake, "soulx")
        self.assertTrue(result["success"])
        self.assertEqual([cmd[3] for cmd, _ in fake.calls], ["run", "run", "deploy"])
        self.assertEqual(fake.builder_versions(), [None, "2025.06", "2025.06"])

    def test_a_version_the_user_set_is_left_alone(self):
        os.environ[ms.BUILDER_VAR] = "2026.10"
        fake = FakeModal((1, REFUSED_ENV))
        (result,) = self.deploy(fake, "qwen3_tts")
        self.assertFalse(result["success"])
        self.assertEqual(fake.builder_versions(), ["2026.10"])
        self.assertIn(ms.BUILDER_VAR, result["hint"])

    def test_a_version_from_modal_toml_is_left_alone(self):
        fake = FakeModal((1, REFUSED_TOML))
        (result,) = self.deploy(fake, "qwen3_tts")
        self.assertFalse(result["success"])
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn(ms.BUILDER_VAR, os.environ)
        self.assertIn("modal.toml", result["hint"])

    def test_other_failures_are_not_retried(self):
        fake = FakeModal((1, "Please add a payment method to use A10G GPU functions\n"))
        (result,) = self.deploy(fake, "qwen3_tts")
        self.assertFalse(result["success"])
        self.assertEqual(fake.builder_versions(), [None])
        self.assertNotIn("image_builder_version", result)
        self.assertIn("payment method", result["hint"])

    def test_accepted_workspace_version_is_never_pinned(self):
        fake = FakeModal()
        (result,) = self.deploy(fake, "upscale")
        self.assertTrue(result["success"])
        self.assertEqual(fake.builder_versions(), [None])
        self.assertNotIn(ms.BUILDER_VAR, os.environ)

    def test_dry_run_shows_a_set_version(self):
        os.environ[ms.BUILDER_VAR] = "2025.06"
        result = ms.deploy(ms.APPS["music_gen"], dry_run=True)
        app_file = Path("docker", "modal-music-gen", "app.py")
        self.assertEqual(result["commands"],
                         [f"MODAL_IMAGE_BUILDER_VERSION=2025.06 modal deploy {app_file}"])


if __name__ == "__main__":
    unittest.main()
