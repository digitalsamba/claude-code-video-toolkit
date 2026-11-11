"""video_job: a local background never matches the caption pill's ink, or the pill disappears on it."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import video_job as vj  # noqa: E402


class FieldClearOfInkTest(unittest.TestCase):
    def test_dark_field_equal_to_ink_lifts(self):
        out = vj.field_clear_of("#0A0A14", "#0A0A14")
        self.assertNotEqual(out, "#0A0A14")
        self.assertGreater(sum(vj._hex_rgb(out)), sum(vj._hex_rgb("#0A0A14")) + 60)

    def test_light_field_equal_to_ink_darkens(self):
        self.assertLess(sum(vj._hex_rgb(vj.field_clear_of("#F4F1EA", "#F4F1EA"))), sum(vj._hex_rgb("#F4F1EA")))

    def test_a_field_already_apart_is_kept(self):
        self.assertEqual(vj.field_clear_of("#3771E0", "#0A0A14"), "#3771E0")


if __name__ == "__main__":
    unittest.main()
