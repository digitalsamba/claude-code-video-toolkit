#!/usr/bin/env python3
"""
Unit tests for MiniMax TTS provider integration.

Tests cover:
- minimax_tts.py standalone module (generate_audio, voice/model validation)
- voiceover.py MiniMax provider integration (CLI args, dry-run, brand config)
- config.py get_minimax_api_key()
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add tools/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestMiniMaxTTSConstants(unittest.TestCase):
    """Test minimax_tts.py constants and configuration."""

    def test_voices_dict_not_empty(self):
        from minimax_tts import MINIMAX_VOICES

        self.assertGreater(len(MINIMAX_VOICES), 0)

    def test_english_voices_exist(self):
        from minimax_tts import MINIMAX_VOICES

        english_voices = [v for v, lang in MINIMAX_VOICES.items() if lang == "English"]
        self.assertGreaterEqual(len(english_voices), 5)

    def test_chinese_voices_exist(self):
        from minimax_tts import MINIMAX_VOICES

        chinese_voices = [v for v, lang in MINIMAX_VOICES.items() if lang == "Chinese"]
        self.assertGreaterEqual(len(chinese_voices), 7)

    def test_default_voice_is_valid(self):
        from minimax_tts import DEFAULT_VOICE, MINIMAX_VOICES

        self.assertIn(DEFAULT_VOICE, MINIMAX_VOICES)

    def test_models_dict(self):
        from minimax_tts import MINIMAX_TTS_MODELS

        self.assertIn("hd", MINIMAX_TTS_MODELS)
        self.assertIn("turbo", MINIMAX_TTS_MODELS)
        self.assertEqual(MINIMAX_TTS_MODELS["hd"], "speech-2.8-hd")
        self.assertEqual(MINIMAX_TTS_MODELS["turbo"], "speech-2.8-turbo")

    def test_api_url(self):
        from minimax_tts import MINIMAX_TTS_API_URL

        self.assertEqual(MINIMAX_TTS_API_URL, "https://api.minimax.io/v1/t2a_v2")

    def test_default_model(self):
        from minimax_tts import DEFAULT_MODEL

        self.assertEqual(DEFAULT_MODEL, "hd")


class TestMiniMaxTTSGenerateAudio(unittest.TestCase):
    """Test minimax_tts.generate_audio() function."""

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_successful_generation(self, mock_post):
        from minimax_tts import generate_audio

        # Create mock response with hex-encoded MP3 data
        fake_audio_hex = b"fake_audio_bytes".hex()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {"audio": fake_audio_hex},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.mp3")
            result = generate_audio(
                text="Hello world",
                output_path=output_path,
                voice="English_Graceful_Lady",
                model="hd",
                verbose=False,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["output"], output_path)
            self.assertEqual(result["script_chars"], 11)
            self.assertTrue(Path(output_path).exists())

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_api_error_response(self, mock_post):
        from minimax_tts import generate_audio

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 1001, "status_msg": "Invalid API key"},
            "data": {},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.mp3")
            result = generate_audio(
                text="Hello",
                output_path=output_path,
                verbose=False,
            )

            self.assertFalse(result["success"])
            self.assertIn("1001", result["error"])

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_http_error(self, mock_post):
        from minimax_tts import generate_audio

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.mp3")
            result = generate_audio(
                text="Hello",
                output_path=output_path,
                verbose=False,
            )

            self.assertFalse(result["success"])
            self.assertIn("500", result["error"])

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        from minimax_tts import generate_audio

        # Remove MINIMAX_API_KEY from env
        os.environ.pop("MINIMAX_API_KEY", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.mp3")
            result = generate_audio(
                text="Hello",
                output_path=output_path,
                verbose=False,
            )

            self.assertFalse(result["success"])
            self.assertIn("MINIMAX_API_KEY", result["error"])

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_no_audio_in_response(self, mock_post):
        from minimax_tts import generate_audio

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "data": {},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output.mp3")
            result = generate_audio(
                text="Hello",
                output_path=output_path,
                verbose=False,
            )

            self.assertFalse(result["success"])
            self.assertIn("No audio", result["error"])

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_request_payload_format(self, mock_post):
        from minimax_tts import generate_audio

        fake_audio_hex = b"audio".hex()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0},
            "data": {"audio": fake_audio_hex},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.mp3")
            generate_audio(
                text="Test text",
                output_path=output_path,
                voice="English_Persuasive_Man",
                model="turbo",
                speed=1.2,
                volume=0.8,
                pitch=-2,
                verbose=False,
            )

            # Verify the API was called with correct payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]

            self.assertEqual(payload["model"], "speech-2.8-turbo")
            self.assertEqual(payload["text"], "Test text")
            self.assertEqual(payload["voice_setting"]["voice_id"], "English_Persuasive_Man")
            self.assertEqual(payload["voice_setting"]["speed"], 1.2)
            self.assertEqual(payload["voice_setting"]["vol"], 0.8)
            self.assertEqual(payload["voice_setting"]["pitch"], -2)
            self.assertEqual(payload["audio_setting"]["format"], "mp3")

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_authorization_header(self, mock_post):
        from minimax_tts import generate_audio

        fake_audio_hex = b"audio".hex()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0},
            "data": {"audio": fake_audio_hex},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.mp3")
            generate_audio(text="Hi", output_path=output_path, verbose=False)

            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            self.assertEqual(headers["Authorization"], "Bearer test_key_123")
            self.assertEqual(headers["Content-Type"], "application/json")

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_timeout_handling(self, mock_post):
        import requests as req
        from minimax_tts import generate_audio

        mock_post.side_effect = req.exceptions.Timeout("Connection timed out")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.mp3")
            result = generate_audio(
                text="Hello",
                output_path=output_path,
                verbose=False,
            )

            self.assertFalse(result["success"])
            self.assertIn("timed out", result["error"])

    @patch("minimax_tts.requests.post")
    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key_123"})
    def test_output_directory_creation(self, mock_post):
        from minimax_tts import generate_audio

        fake_audio_hex = b"audio_data".hex()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "base_resp": {"status_code": 0},
            "data": {"audio": fake_audio_hex},
        }
        mock_post.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "dir", "output.mp3")
            result = generate_audio(
                text="Hello",
                output_path=nested_path,
                verbose=False,
            )

            self.assertTrue(result["success"])
            self.assertTrue(Path(nested_path).exists())


class TestConfigMiniMaxAPIKey(unittest.TestCase):
    """Test config.py get_minimax_api_key()."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "my_key_abc"})
    def test_returns_key_from_env(self):
        from config import get_minimax_api_key

        self.assertEqual(get_minimax_api_key(), "my_key_abc")

    @patch.dict(os.environ, {}, clear=True)
    def test_returns_none_when_missing(self):
        os.environ.pop("MINIMAX_API_KEY", None)
        from config import get_minimax_api_key

        result = get_minimax_api_key()
        self.assertIsNone(result)


class TestVoiceoverMiniMaxCLI(unittest.TestCase):
    """Test voiceover.py CLI argument parsing for MiniMax provider."""

    def test_minimax_provider_accepted(self):
        from voiceover import parse_args

        with patch("sys.argv", [
            "voiceover.py",
            "--provider", "minimax",
            "--script", "test.txt",
            "--output", "out.mp3",
        ]):
            args = parse_args()
            self.assertEqual(args.provider, "minimax")

    def test_minimax_voice_default(self):
        from voiceover import parse_args

        with patch("sys.argv", [
            "voiceover.py",
            "--provider", "minimax",
            "--script", "test.txt",
            "--output", "out.mp3",
        ]):
            args = parse_args()
            self.assertEqual(args.minimax_voice, "English_Graceful_Lady")

    def test_minimax_voice_custom(self):
        from voiceover import parse_args

        with patch("sys.argv", [
            "voiceover.py",
            "--provider", "minimax",
            "--minimax-voice", "English_Persuasive_Man",
            "--script", "test.txt",
            "--output", "out.mp3",
        ]):
            args = parse_args()
            self.assertEqual(args.minimax_voice, "English_Persuasive_Man")

    def test_minimax_model_choices(self):
        from voiceover import parse_args

        for model_choice in ["hd", "turbo"]:
            with patch("sys.argv", [
                "voiceover.py",
                "--provider", "minimax",
                "--minimax-model", model_choice,
                "--script", "test.txt",
                "--output", "out.mp3",
            ]):
                args = parse_args()
                self.assertEqual(args.minimax_model, model_choice)

    def test_minimax_volume_and_pitch(self):
        from voiceover import parse_args

        with patch("sys.argv", [
            "voiceover.py",
            "--provider", "minimax",
            "--volume", "2.0",
            "--pitch", "3",
            "--script", "test.txt",
            "--output", "out.mp3",
        ]):
            args = parse_args()
            self.assertEqual(args.volume, 2.0)
            self.assertEqual(args.pitch, 3)


class TestVoiceoverMiniMaxDryRun(unittest.TestCase):
    """Test voiceover.py dry-run output for MiniMax provider."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test_key"})
    def test_single_file_dry_run(self):
        from voiceover import parse_args

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test script content")
            script_path = f.name

        try:
            import io
            from contextlib import redirect_stdout

            with patch("sys.argv", [
                "voiceover.py",
                "--provider", "minimax",
                "--script", script_path,
                "--output", "out.mp3",
                "--dry-run",
                "--json",
            ]):
                captured = io.StringIO()
                with redirect_stdout(captured):
                    from voiceover import main
                    from dotenv import load_dotenv
                    load_dotenv()
                    main()

                output = captured.getvalue()
                result = json.loads(output)
                self.assertTrue(result["dry_run"])
                self.assertEqual(result["provider"], "minimax")
                self.assertEqual(result["voice"], "English_Graceful_Lady")
                self.assertEqual(result["model"], "hd")
        finally:
            os.unlink(script_path)


class TestVoiceoverMiniMaxBrand(unittest.TestCase):
    """Test voiceover.py brand config resolution for MiniMax."""

    def test_brand_voice_config_with_minimax(self):
        """Verify that voice.json with minimax section can be loaded."""
        from config import load_brand_voice_config

        config = load_brand_voice_config("default")
        if config:
            self.assertIn("minimax", config)
            minimax_cfg = config["minimax"]
            self.assertIn("voice", minimax_cfg)
            self.assertIn("model", minimax_cfg)


class TestMiniMaxTTSGetAudioDuration(unittest.TestCase):
    """Test audio duration helper."""

    def test_nonexistent_file(self):
        from minimax_tts import get_audio_duration

        result = get_audio_duration("/nonexistent/path.mp3")
        self.assertIsNone(result)


class TestGenerateSingleAudioMiniMax(unittest.TestCase):
    """Test voiceover.py generate_single_audio_minimax wrapper."""

    @patch("minimax_tts.generate_audio")
    def test_delegates_to_minimax_tts(self, mock_gen):
        mock_gen.return_value = {"success": True, "output": "/tmp/test.mp3"}

        from voiceover import generate_single_audio_minimax

        result = generate_single_audio_minimax(
            script="Hello world",
            output_path=Path("/tmp/test.mp3"),
            voice="English_Persuasive_Man",
            model="turbo",
            speed=1.5,
            volume=2.0,
            pitch=3,
        )

        mock_gen.assert_called_once_with(
            text="Hello world",
            output_path="/tmp/test.mp3",
            voice="English_Persuasive_Man",
            model="turbo",
            speed=1.5,
            volume=2.0,
            pitch=3,
            verbose=False,
        )


class TestMiniMaxTTSListVoices(unittest.TestCase):
    """Test minimax_tts.py --list-voices CLI."""

    def test_list_voices_output(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "minimax_tts", "--list-voices"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent / "tools"),
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("English_Graceful_Lady", result.stdout)
        self.assertIn("English_Persuasive_Man", result.stdout)
        self.assertIn("Deep_Voice_Man", result.stdout)
        self.assertIn("speech-2.8-hd", result.stdout)


if __name__ == "__main__":
    unittest.main()
