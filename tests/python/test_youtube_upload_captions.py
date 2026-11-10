"""youtube_upload.py: captions need youtube.force-ssl, and --video-id adds them to an uploaded video."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

OLD_TOKEN = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


@unittest.skipUnless(importlib.util.find_spec("googleapiclient"), "needs uv sync --extra youtube")
class CaptionScopeTest(unittest.TestCase):
    def setUp(self):
        import youtube_upload
        self.yu = youtube_upload

    def test_captions_scope_is_requested(self):
        self.assertIn("https://www.googleapis.com/auth/youtube.force-ssl", self.yu.SCOPES)

    def test_old_token_lacks_it_only_when_captions_asked(self):
        creds = SimpleNamespace(scopes=OLD_TOKEN)
        self.assertEqual(self.yu.missing_scopes(creds, self.yu.needed_scopes(SimpleNamespace(captions=None))), [])
        self.assertEqual(self.yu.missing_scopes(creds, self.yu.needed_scopes(SimpleNamespace(captions="v.srt"))),
                         [self.yu.CAPTION_SCOPE])
        self.assertEqual(self.yu.missing_scopes(SimpleNamespace(scopes=self.yu.SCOPES), [self.yu.CAPTION_SCOPE]), [])


@unittest.skipUnless(importlib.util.find_spec("googleapiclient"), "needs uv sync --extra youtube")
class VideoIdTest(unittest.TestCase):
    def setUp(self):
        import youtube_upload
        self.parse = youtube_upload.build_parser().parse_args
        self.validate = youtube_upload.validate_args

    def test_captions_on_uploaded_video_needs_no_title_or_file(self):
        with tempfile.NamedTemporaryFile(suffix=".srt") as srt:
            self.assertIsNone(self.validate(self.parse(["--video-id", "abc", "--captions", srt.name])))

    def test_video_id_needs_something_to_add(self):
        self.assertIn("needs something", self.validate(self.parse(["--video-id", "abc"])))

    def test_video_id_and_video_conflict(self):
        self.assertIn("leave out --video", self.validate(self.parse(["--video-id", "abc", "--video", "v.mp4"])))

    def test_missing_caption_file(self):
        self.assertIn("not found", self.validate(self.parse(["--video-id", "abc", "--captions", "nope.srt"])))


if __name__ == "__main__":
    unittest.main()
