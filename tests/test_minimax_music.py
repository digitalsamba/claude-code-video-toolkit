import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "minimax_music.py"
SPEC = importlib.util.spec_from_file_location("minimax_music", MODULE_PATH)
minimax_music = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(minimax_music)


class FakeResponse:
    def __init__(self, *, payload=None, content=b"", lines=None, status_code=200):
        self.payload = payload
        self.content = content
        self.lines = lines or []
        self.status_code = status_code

    def json(self):
        return self.payload

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise minimax_music.requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, post_response, get_response=None):
        self.post_response = post_response
        self.get_response = get_response
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.post_response

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.get_response


def base_args(**overrides):
    values = {
        "model": "music-3.0",
        "prompt": "Cinematic ambient score",
        "lyrics": None,
        "stream": False,
        "output_format": "hex",
        "sample_rate": 44100,
        "bitrate": 256000,
        "audio_format": "mp3",
        "lyrics_optimizer": False,
        "is_instrumental": True,
        "region": "global_en",
        "aigc_watermark": None,
    }
    values.update(overrides)
    return values


class BuildPayloadTests(unittest.TestCase):
    def test_builds_documented_global_payload(self):
        payload = minimax_music.build_payload(**base_args())
        self.assertEqual("music-3.0", payload["model"])
        self.assertEqual("Cinematic ambient score", payload["prompt"])
        self.assertEqual(
            {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"},
            payload["audio_setting"],
        )
        self.assertTrue(payload["is_instrumental"])
        self.assertNotIn("aigc_watermark", payload)

    def test_supports_all_generation_models(self):
        for model in minimax_music.MODELS:
            with self.subTest(model=model):
                payload = minimax_music.build_payload(**base_args(model=model))
                self.assertEqual(model, payload["model"])

    def test_adds_watermark_only_for_mainland_china(self):
        payload = minimax_music.build_payload(
            **base_args(region="cn_zh", aigc_watermark=True)
        )
        self.assertTrue(payload["aigc_watermark"])
        with self.assertRaisesRegex(ValueError, "only available for the cn_zh"):
            minimax_music.build_payload(**base_args(aigc_watermark=True))

    def test_rejects_url_streaming(self):
        with self.assertRaisesRegex(ValueError, "only supports hex"):
            minimax_music.build_payload(**base_args(stream=True, output_format="url"))

    def test_validates_prompt_and_lyrics_requirements(self):
        with self.assertRaisesRegex(ValueError, "requires a prompt"):
            minimax_music.build_payload(**base_args(prompt=None))
        with self.assertRaisesRegex(ValueError, "requires lyrics"):
            minimax_music.build_payload(
                **base_args(prompt="A pop song", lyrics=None, is_instrumental=False)
            )


class GenerateMusicTests(unittest.TestCase):
    def test_decodes_hex_response_and_uses_global_endpoint(self):
        response = FakeResponse(
            payload={
                "data": {"audio": "494433", "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
        )
        session = FakeSession(response)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "track.mp3"
            result = minimax_music.generate_music(
                api_key="test-key", output=output, session=session, **base_args()
            )
            self.assertEqual(b"ID3", output.read_bytes())
        self.assertEqual(minimax_music.ENDPOINTS["global_en"], session.posts[0][0])
        self.assertEqual("Bearer test-key", session.posts[0][1]["headers"]["Authorization"])
        self.assertEqual(2, result["status"])

    def test_downloads_url_response_from_mainland_china_endpoint(self):
        response = FakeResponse(
            payload={
                "data": {"audio": "https://files.example/track.wav", "status": 2},
                "base_resp": {"status_code": 0},
            }
        )
        session = FakeSession(response, FakeResponse(content=b"RIFFaudio"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "track.wav"
            minimax_music.generate_music(
                api_key="test-key",
                output=output,
                session=session,
                **base_args(region="cn_zh", output_format="url", audio_format="wav"),
            )
            self.assertEqual(b"RIFFaudio", output.read_bytes())
        self.assertEqual(minimax_music.ENDPOINTS["cn_zh"], session.posts[0][0])
        self.assertEqual("https://files.example/track.wav", session.gets[0][0])

    def test_collects_streaming_hex_chunks(self):
        events = [
            {"data": {"audio": "4944", "status": 1}, "base_resp": {"status_code": 0}},
            {"data": {"audio": "33", "status": 2}, "base_resp": {"status_code": 0}},
        ]
        response = FakeResponse(lines=[f"data: {json.dumps(event)}" for event in events])
        session = FakeSession(response)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "track.mp3"
            result = minimax_music.generate_music(
                api_key="test-key",
                output=output,
                session=session,
                **base_args(stream=True),
            )
            self.assertEqual(b"ID3", output.read_bytes())
        self.assertTrue(result["stream"])
        self.assertTrue(session.posts[0][1]["stream"])

    def test_reports_api_error(self):
        response = FakeResponse(
            payload={"base_resp": {"status_code": 1001, "status_msg": "invalid key"}}
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(minimax_music.MiniMaxMusicError, "invalid key"):
                minimax_music.generate_music(
                    api_key="test-key",
                    output=Path(directory) / "track.mp3",
                    session=FakeSession(response),
                    **base_args(),
                )

    def test_rejects_incomplete_response(self):
        response = FakeResponse(
            payload={"data": {"audio": "494433", "status": 1}, "base_resp": {"status_code": 0}}
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(minimax_music.MiniMaxMusicError, "did not complete"):
                minimax_music.generate_music(
                    api_key="test-key",
                    output=Path(directory) / "track.mp3",
                    session=FakeSession(response),
                    **base_args(),
                )


if __name__ == "__main__":
    unittest.main()
