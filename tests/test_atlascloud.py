from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import atlascloud


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", status_code=200):
        self.payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise atlascloud.requests.HTTPError(f"HTTP {self.status_code}")


class FakeHTTP:
    def __init__(self, post_response, get_responses):
        self.post_response = post_response
        self.get_responses = list(get_responses)
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.post_response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_responses.pop(0)


class GenerateImageTests(unittest.TestCase):
    @patch("atlascloud.time.sleep", return_value=None)
    def test_submits_polls_and_downloads(self, _sleep):
        http = FakeHTTP(
            FakeResponse(
                {
                    "code": 200,
                    "data": {
                        "id": "prediction-123",
                        "status": "processing",
                        "urls": {"get": "https://api.example/prediction-123"},
                    },
                }
            ),
            [
                FakeResponse(
                    {
                        "code": 200,
                        "data": {
                            "status": "completed",
                            "outputs": ["https://cdn.example/image.jpg"],
                        },
                    }
                ),
                FakeResponse(content=b"image-bytes"),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "image.jpg"
            result = atlascloud.generate_image(
                "secret",
                prompt="A clean title background",
                output_path=str(output),
                extra_params={"seed": 7, "prompt": "ignored"},
                poll_interval=0,
                http=http,
            )

            self.assertEqual(output.read_bytes(), b"image-bytes")
            self.assertEqual(result["prediction_id"], "prediction-123")

        submit_url, submit_kwargs = http.post_calls[0]
        self.assertTrue(submit_url.endswith("/model/generateImage"))
        self.assertEqual(submit_kwargs["json"]["prompt"], "A clean title background")
        self.assertEqual(submit_kwargs["json"]["seed"], 7)
        self.assertNotIn("size", submit_kwargs["json"])
        self.assertNotIn("output_format", submit_kwargs["json"])
        self.assertEqual(http.get_calls[0][0], "https://api.example/prediction-123")

    @patch("atlascloud.time.sleep", return_value=None)
    def test_reports_failed_prediction_without_downloading(self, _sleep):
        http = FakeHTTP(
            FakeResponse(
                {"code": 200, "data": {"id": "failed-123", "status": "processing"}}
            ),
            [
                FakeResponse(
                    {"code": 200, "data": {"status": "failed", "error": "bad prompt"}}
                )
            ],
        )

        result = atlascloud.generate_image(
            "secret",
            prompt="A prompt",
            output_path="unused.jpg",
            poll_interval=0,
            http=http,
        )

        self.assertIsNone(result)
        self.assertEqual(len(http.get_calls), 1)


class ExtraParamsTests(unittest.TestCase):
    def test_requires_json_object(self):
        with self.assertRaises(TypeError):
            atlascloud._extra_params('["not", "an", "object"]')


if __name__ == "__main__":
    unittest.main()
