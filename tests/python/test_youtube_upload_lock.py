"""youtube_upload.py's private-lock warning: a scheduled upload is private by design."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))


@unittest.skipUnless(importlib.util.find_spec("googleapiclient"), "needs uv sync --extra youtube")
class LockedPrivateTest(unittest.TestCase):
    def setUp(self):
        import youtube_upload
        self.f = youtube_upload.locked_private

    def test_scheduled_and_private_is_fine(self):
        args = SimpleNamespace(privacy="private", publish_at="2026-11-10T09:00:00Z")
        self.assertIsNone(self.f(args, {"privacyStatus": "private", "publishAt": "2026-11-10T09:00:00Z"}))

    def test_scheduled_with_publish_time_dropped_warns(self):
        args = SimpleNamespace(privacy="private", publish_at="2026-11-10T09:00:00Z")
        self.assertIn("dropped the scheduled", self.f(args, {"privacyStatus": "private"}))

    def test_public_kept_private_warns(self):
        args = SimpleNamespace(privacy="public", publish_at=None)
        self.assertIn("public upload", self.f(args, {"privacyStatus": "private"}))
        self.assertIsNone(self.f(args, {"privacyStatus": "public"}))

    def test_private_upload_never_warns(self):
        self.assertIsNone(self.f(SimpleNamespace(privacy="private", publish_at=None), {"privacyStatus": "private"}))


if __name__ == "__main__":
    unittest.main()
