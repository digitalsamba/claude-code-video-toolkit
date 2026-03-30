#!/usr/bin/env python3
"""
Integration tests for MiniMax TTS provider.

These tests require MINIMAX_API_KEY to be set and make real API calls.
Skip with: python -m pytest tests/test_minimax_tts_integration.py -k "not integration"
Or run only integration tests: MINIMAX_API_KEY=xxx python -m pytest tests/test_minimax_tts_integration.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add tools/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set — skipping integration tests"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxTTSIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax TTS API."""

    def test_generate_audio_hd(self):
        """Test generating audio with speech-2.8-hd model."""
        from minimax_tts import generate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_hd.mp3")
            result = generate_audio(
                text="Hello, this is a test of MiniMax text to speech.",
                output_path=output_path,
                voice="English_Graceful_Lady",
                model="hd",
                verbose=False,
            )

            self.assertTrue(result["success"], f"Failed: {result.get('error')}")
            self.assertTrue(Path(output_path).exists())
            self.assertGreater(Path(output_path).stat().st_size, 1000)

    def test_generate_audio_turbo(self):
        """Test generating audio with speech-2.8-turbo model."""
        from minimax_tts import generate_audio

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_turbo.mp3")
            result = generate_audio(
                text="Quick turbo test of MiniMax.",
                output_path=output_path,
                voice="English_Persuasive_Man",
                model="turbo",
                verbose=False,
            )

            self.assertTrue(result["success"], f"Failed: {result.get('error')}")
            self.assertTrue(Path(output_path).exists())

    def test_voiceover_minimax_single_file(self):
        """Test voiceover.py with --provider minimax in single-file mode."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.txt")
            output_path = os.path.join(tmpdir, "voiceover.mp3")

            Path(script_path).write_text("Integration test voiceover with MiniMax.")

            result = subprocess.run(
                [
                    sys.executable, "tools/voiceover.py",
                    "--provider", "minimax",
                    "--script", script_path,
                    "--output", output_path,
                    "--json",
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent),
                env={**os.environ, "MINIMAX_API_KEY": MINIMAX_API_KEY},
            )

            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
            output = json.loads(result.stdout)
            self.assertTrue(output.get("success"))
            self.assertEqual(output["provider"], "minimax")


if __name__ == "__main__":
    unittest.main()
