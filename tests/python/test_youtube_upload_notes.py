"""youtube_upload.py's metadata notes: advice on the title, description, tags and category before an upload."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

GOOD = "A chart drawn three ways.\n\nGet it: https://example.com/repo\n\n#remotion #dataviz"
WIDE = {"width": 1920, "height": 1080, "seconds": 90.0}


@unittest.skipUnless(importlib.util.find_spec("googleapiclient"), "needs uv sync --extra youtube")
class MetadataNotesTest(unittest.TestCase):
    def setUp(self):
        import youtube_upload
        self.yu = youtube_upload
        self.f = youtube_upload.metadata_notes

    def test_complete_metadata_has_no_notes(self):
        self.assertEqual(self.f("One chart, three designs in Remotion", GOOD, ["remotion", "charts"], "28", WIDE), [])

    def test_missing_hashtags(self):
        notes = self.f("Remotion chart", "Get it: https://example.com", ["remotion"], "28", WIDE)
        self.assertEqual(len(notes), 1)
        self.assertIn("hashtags", notes[0])

    def test_a_url_fragment_is_not_a_hashtag(self):
        notes = self.f("Remotion chart", "Docs: https://example.com/page#setup", ["remotion"], "28", WIDE)
        self.assertTrue(any("hashtags" in n for n in notes))

    def test_missing_link(self):
        notes = self.f("Remotion chart", "A chart. #remotion", ["remotion"], "28", WIDE)
        self.assertEqual(len(notes), 1)
        self.assertIn("no link", notes[0])

    def test_no_tag_in_the_title(self):
        notes = self.f("Just a Quick One", GOOD, ["blender", "3d character"], "28", WIDE)
        self.assertEqual(len(notes), 1)
        self.assertIn("None of the tags", notes[0])

    def test_tag_match_ignores_case_and_no_tags_is_quiet(self):
        self.assertEqual(self.f("A 3D Character acts", GOOD, ["3d character"], "28", WIDE), [])
        self.assertEqual(self.f("Anything", GOOD, [], "28", WIDE), [])

    def test_category_left_out(self):
        notes = self.f("Remotion chart", GOOD, ["remotion"], None, WIDE)
        self.assertEqual(len(notes), 1)
        self.assertIn("People & Blogs", notes[0])

    def test_a_vertical_cut_under_three_minutes_is_a_short(self):
        notes = self.f("Remotion chart", GOOD, ["remotion"], "28", {"width": 1080, "height": 1920, "seconds": 27.0})
        self.assertEqual(len(notes), 1)
        self.assertIn("related video", notes[0])

    def test_square_counts_and_long_or_wide_does_not(self):
        self.assertTrue(self.yu.is_short({"width": 1080, "height": 1080, "seconds": 60.0}))
        self.assertFalse(self.yu.is_short({"width": 1080, "height": 1920, "seconds": 181.0}))
        self.assertFalse(self.yu.is_short(WIDE))
        self.assertFalse(self.yu.is_short(None))

    def test_unreadable_file_gives_no_shape(self):
        self.assertIsNone(self.yu.probe_shape(None))
        self.assertIsNone(self.yu.probe_shape(str(Path(__file__))))


if __name__ == "__main__":
    unittest.main()
