import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import verify_setup


class VerifySetupTests(unittest.TestCase):
    def test_local_remote_and_smoke_test_modes(self):
        config = {
            "MODAL_QWEN3_TTS_ENDPOINT_URL": "https://example.test/tts",
            "MODAL_FLUX2_ENDPOINT_URL": "https://example.test/flux",
            "R2_ACCOUNT_ID": "test-account",
            "R2_ACCESS_KEY_ID": "test-key",
            "R2_SECRET_ACCESS_KEY": "test-secret",
        }
        for flags, remote, smoke in (
            ([], False, False),
            (["--remote"], True, False),
            (["--test"], True, True),
        ):
            with self.subTest(flags=flags):
                argv = ["verify_setup.py", "--json", *flags]
                stdout = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    patch.dict(os.environ, config, clear=True),
                    patch.object(verify_setup, "check_command", return_value=(True, "v24.0.0")),
                    patch.object(verify_setup.shutil, "which", return_value="modal"),
                    patch.object(verify_setup, "check_modal_apps", return_value={
                        "ok": True, "apps": ["video-toolkit-soulx"], "detail": "configured",
                    }) as apps,
                    patch.object(verify_setup, "check_soulx_weights", return_value={"ok": True}) as weights,
                    patch.object(verify_setup, "test_r2_connectivity", return_value={"ok": True}) as r2,
                    patch.object(verify_setup, "test_cloud_endpoint", return_value={"ok": True}) as cloud,
                    redirect_stdout(stdout),
                ):
                    verify_setup.main()

                result = json.loads(stdout.getvalue())
                self.assertTrue(result["summary"]["ready"])
                self.assertTrue(result["r2"]["ok"])
                self.assertTrue(result["voice"]["qwen3_tts"]["ok"])
                self.assertIn("runpod_tools", result)
                self.assertEqual(remote, "modal_apps" in result)
                self.assertEqual(remote, "soulx_weights" in result)
                self.assertEqual(remote, "r2_test" in result)
                self.assertEqual(smoke, "test_qwen3_tts" in result)
                self.assertEqual(smoke, "test_flux2" in result)
                if remote:
                    apps.assert_called_once_with()
                    weights.assert_called_once_with()
                    r2.assert_called_once_with()
                else:
                    apps.assert_not_called()
                    weights.assert_not_called()
                    r2.assert_not_called()
                expected_smoke_calls = [call("qwen3_tts", "modal"), call("flux2", "modal")] if smoke else []
                self.assertEqual(expected_smoke_calls, cloud.call_args_list)
