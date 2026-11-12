"""publish_kit.py: chapters follow YouTube's rules, titles come from file names, scripts in play order."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import publish_kit as pk  # noqa: E402


class ChaptersTest(unittest.TestCase):
    def test_first_at_zero_and_whole_seconds(self):
        ch, notes = pk.chapters_from([0.4, 11.2, 25.8, 37.5], ["Open", "Narrate", "Score", "Generate"], 60)
        self.assertEqual([c["at"] for c in ch], [0.0, 11.0, 25.0, 37.0])
        self.assertEqual(notes, [])

    def test_short_scene_joins_the_one_before(self):
        ch, notes = pk.chapters_from([0, 12, 16, 30], ["A", "B", "C", "D"], 60)
        self.assertEqual([c["title"] for c in ch], ["A", "B", "D"])
        self.assertIn("'C'", notes[0])

    def test_short_last_chapter_joins_the_one_before(self):
        ch, _ = pk.chapters_from([0, 12, 24, 55], ["A", "B", "C", "D"], 60)
        self.assertEqual([c["title"] for c in ch], ["A", "B", "C"])

    def test_fewer_than_three_writes_none(self):
        ch, notes = pk.chapters_from([0, 15], ["A", "B"], 40)
        self.assertEqual(ch, [])
        self.assertIn("needs 3", notes[-1])

    def test_time_format(self):
        self.assertEqual(pk.fmt_time(83.9), "1:23")
        self.assertEqual(pk.fmt_time(3725), "1:02:05")


class ScenesTest(unittest.TestCase):
    def test_titles_from_file_names(self):
        self.assertEqual(pk.title_from_name(Path("02_narrate.txt")), "Narrate")
        self.assertEqual(pk.title_from_name(Path("03-the-bill.txt")), "The bill")

    def test_only_scripts_beside_audio_in_name_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name in ("02_b", "01_a"):
                (d / f"{name}.txt").write_text("words")
                (d / f"{name}.mp3").write_bytes(b"")
            (d / "notes.txt").write_text("not narration")
            self.assertEqual([p.name for p in pk.scene_scripts(d)], ["01_a.txt", "02_b.txt"])

    def test_project_titles_win(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "project.json").write_text(json.dumps({"scenes": [{"title": "The prompt"}, {}]}))
            self.assertEqual(pk.scene_titles(d, [Path("01_open.txt"), Path("02_narrate.txt")]),
                             ["The prompt", "Narrate"])


@unittest.skipUnless(importlib.util.find_spec("googleapiclient"), "needs uv sync --extra youtube")
class CheckHelpersTest(unittest.TestCase):
    def test_project_number_from_client_id(self):
        import youtube_upload as yu
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"installed": {"client_id": "495036436430-abc.apps.googleusercontent.com"}}, f)
        self.assertEqual(yu.cloud_project_number(f.name), "495036436430")
        self.assertIsNone(yu.cloud_project_number("/nope.json"))

    def test_enable_url(self):
        import youtube_upload as yu
        self.assertEqual(yu.enable_url("analytics", "1"),
                         "https://console.developers.google.com/apis/api/youtubeanalytics.googleapis.com/overview?project=1")


if __name__ == "__main__":
    unittest.main()
