import base64
import os
import unittest
from unittest.mock import patch

from tools import cloud_gpu


class FakeResponse:
    def __init__(self, status_code=200, *, payload=None, text="", headers=None, chunks=()):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.headers = headers or {}
        self._chunks = list(chunks)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def iter_content(self, chunk_size):
        del chunk_size
        return iter(self._chunks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class MuapiCloudTests(unittest.TestCase):
    def test_muapi_size_maps_supported_aspect_ratios_and_rejects_others(self):
        self.assertEqual(cloud_gpu._muapi_size(1024, 1024), "1024x1024")
        self.assertEqual(cloud_gpu._muapi_size(1920, 1080), "1792x1024")
        self.assertEqual(cloud_gpu._muapi_size(1080, 1920), "1024x1792")
        with self.assertRaisesRegex(ValueError, "only 1:1, 16:9, and 9:16"):
            cloud_gpu._muapi_size(4, 3)

    def test_muapi_generation_downloads_https_image_without_forwarding_auth(self):
        requests_seen = {}
        image_bytes = b"\x89PNG\r\nfixture"

        def fake_post(url, **kwargs):
            requests_seen["post"] = (url, kwargs)
            return FakeResponse(payload={"data": [{"url": "https://cdn.example/image.png"}]})

        def fake_get(url, **kwargs):
            requests_seen["get"] = (url, kwargs)
            return FakeResponse(
                headers={"Content-Length": str(len(image_bytes)), "Content-Type": "image/png"},
                chunks=(image_bytes,),
            )

        with patch.dict(os.environ, {"MUAPI_API_KEY": "secret"}), patch.object(
            cloud_gpu.requests, "post", side_effect=fake_post
        ), patch.object(cloud_gpu.requests, "get", side_effect=fake_get):
            result, elapsed = cloud_gpu.call_cloud_endpoint(
                provider="muapi",
                payload={
                    "input": {
                        "operation": "generate",
                        "prompt": "a lantern in rain",
                        "width": 1920,
                        "height": 1080,
                    }
                },
                tool_name="flux2",
                verbose=False,
            )

        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(base64.b64decode(result["image_base64"]), image_bytes)
        post_url, post_kwargs = requests_seen["post"]
        self.assertEqual(post_url, cloud_gpu._MUAPI_IMAGE_ENDPOINT)
        self.assertEqual(
            post_kwargs["headers"],
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        self.assertEqual(
            post_kwargs["json"],
            {
                "model": "flux-schnell",
                "prompt": "a lantern in rain",
                "n": 1,
                "size": "1792x1024",
            },
        )
        get_url, get_kwargs = requests_seen["get"]
        self.assertEqual(get_url, "https://cdn.example/image.png")
        self.assertFalse(get_kwargs["allow_redirects"])
        self.assertNotIn("headers", get_kwargs)

    def test_muapi_rejects_editing_before_network_request(self):
        with patch.dict(os.environ, {"MUAPI_API_KEY": "secret"}), patch.object(
            cloud_gpu.requests,
            "post",
            side_effect=AssertionError("MuAPI editing must not make a request"),
        ):
            result, elapsed = cloud_gpu.call_cloud_endpoint(
                provider="muapi",
                payload={"input": {"operation": "edit", "prompt": "add a hat"}},
                tool_name="flux2",
                verbose=False,
            )

        self.assertGreaterEqual(elapsed, 0)
        self.assertIn("generation only", result["error"])

    def test_muapi_rejects_non_https_result_url(self):
        with patch.dict(os.environ, {"MUAPI_API_KEY": "secret"}), patch.object(
            cloud_gpu.requests,
            "post",
            return_value=FakeResponse(payload={"data": [{"url": "http://cdn.example/image.png"}]}),
        ), patch.object(
            cloud_gpu.requests,
            "get",
            side_effect=AssertionError("Unsafe MuAPI URL must not be fetched"),
        ):
            result, _ = cloud_gpu.call_cloud_endpoint(
                provider="muapi",
                payload={"input": {"operation": "generate", "prompt": "a lantern"}},
                tool_name="flux2",
                verbose=False,
            )

        self.assertEqual(result["error"], "MuAPI returned an unsafe image URL")


if __name__ == "__main__":
    unittest.main()
