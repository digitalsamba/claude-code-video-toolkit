from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools import minimax_video


class FakeResponse:
    def __init__(self, payload=None, content=b"video"):
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.content


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class MiniMaxVideoTests(unittest.TestCase):
    def test_builds_text_to_video_payload(self):
        payload = minimax_video.build_payload(
            prompt="A city skyline at dusk",
            model=minimax_video.DEFAULT_MODEL,
            duration=6,
            resolution="1080P",
            prompt_optimizer=False,
            fast_pretreatment=True,
        )

        self.assertEqual(payload["model"], minimax_video.DEFAULT_MODEL)
        self.assertEqual(payload["prompt"], "A city skyline at dusk")
        self.assertFalse(payload["prompt_optimizer"])
        self.assertTrue(payload["fast_pretreatment"])
        self.assertNotIn("first_frame_image", payload)

    def test_encodes_local_image_as_data_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image = Path(tmpdir) / "frame.png"
            image.write_bytes(b"png-data")

            source = minimax_video.image_source(str(image))

        self.assertTrue(source.startswith("data:image/png;base64,"))

    def test_rejects_unsupported_image_data_url(self):
        with self.assertRaisesRegex(ValueError, "Unsupported image data URL type"):
            minimax_video.image_source("data:image/gif;base64,AAAA")

    def test_fast_model_requires_image_input(self):
        with self.assertRaisesRegex(ValueError, "requires --input"):
            minimax_video.build_payload(
                prompt="A moving camera",
                model=minimax_video.FAST_MODEL,
                duration=6,
                resolution="768P",
            )

    def test_512p_hailuo_02_requires_image_input(self):
        with self.assertRaisesRegex(ValueError, "only for image-to-video"):
            minimax_video.build_payload(
                prompt="A moving camera",
                model=minimax_video.HAILUO_02_MODEL,
                duration=6,
                resolution="512P",
            )

    def test_create_task_uses_cn_endpoint(self):
        session = FakeSession([
            FakeResponse({"task_id": "task-1", "base_resp": {"status_code": 0}}),
        ])

        task_id = minimax_video.create_video_task(
            "test-key",
            {"model": minimax_video.DEFAULT_MODEL, "prompt": "Test"},
            region="cn_zh",
            request_timeout=30,
            session=session,
        )

        self.assertEqual(task_id, "task-1")
        method, url, kwargs = session.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://api.minimaxi.com/v1/video_generation")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_poll_returns_file_id_after_processing(self):
        session = FakeSession([
            FakeResponse({"status": "Processing", "base_resp": {"status_code": 0}}),
            FakeResponse({"status": "Success", "file_id": "file-1", "base_resp": {"status_code": 0}}),
        ])
        clock = iter([0.0, 0.0, 1.0])

        file_id = minimax_video.poll_video_task(
            "test-key",
            "task-1",
            region="global_en",
            request_timeout=30,
            generation_timeout=60,
            poll_interval=10,
            session=session,
            sleep=lambda _: None,
            monotonic=lambda: next(clock),
        )

        self.assertEqual(file_id, "file-1")
        self.assertEqual(session.calls[0][1], "https://api.minimax.io/v1/query/video_generation")

    def test_api_error_is_raised(self):
        session = FakeSession([
            FakeResponse({"base_resp": {"status_code": 1004, "status_msg": "authentication failed"}}),
        ])

        with self.assertRaisesRegex(minimax_video.MiniMaxVideoError, "authentication failed"):
            minimax_video.create_video_task(
                "test-key",
                {"model": minimax_video.DEFAULT_MODEL, "prompt": "Test"},
                region="global_en",
                request_timeout=30,
                session=session,
            )

    def test_download_writes_video(self):
        session = FakeSession([FakeResponse(content=b"mp4-bytes")])
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "out.mp4"

            result = minimax_video.download_video(
                "https://download.example/video.mp4",
                str(output),
                request_timeout=30,
                session=session,
            )

            self.assertEqual(result, str(output))
            self.assertEqual(output.read_bytes(), b"mp4-bytes")


if __name__ == "__main__":
    unittest.main()
