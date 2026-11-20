"""YouTube chapters from the howto template's timeline (templates/howto/scripts/chapters.py)."""
import importlib.util
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOWTO = ROOT / "templates" / "howto"

spec = importlib.util.spec_from_file_location("howto_chapters", HOWTO / "scripts" / "chapters.py")
chapters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chapters)


def timeline(starts, frames=5400, hook="will this cost me money?", asks=True):
    return {
        "fps": 30,
        "frames": frames,
        "title": ["What it costs:", "mostly nothing"],
        "hook": {"ask": hook} if hook else None,
        "chapters": [
            {"title": [f"Chapter {i + 1}"], "ask": f"question {i + 1}?" if asks else None, "start": s}
            for i, s in enumerate(starts)
        ],
    }


class Rules(unittest.TestCase):
    def test_first_chapter_is_the_hook_at_zero(self):
        marks, notes = chapters.chapters_from(timeline([634, 1283, 2401, 3740]))
        self.assertEqual(marks[0], {"at": 0, "title": "Will this cost me money?"})
        self.assertEqual([m["at"] for m in marks], [0, 21, 43, 80, 125])
        self.assertEqual(notes, [])

    def test_no_hook_names_the_opening_by_the_episode(self):
        marks, _ = chapters.chapters_from(timeline([600, 1500, 3000], hook=None))
        self.assertEqual(marks[0]["title"], "What it costs: mostly nothing")

    def test_a_mark_is_never_more_than_a_third_of_a_second_before_its_card(self):
        for start in range(300, 3000, 7):
            marks, _ = chapters.chapters_from(timeline([start, start + 900, start + 1800]))
            self.assertGreaterEqual(marks[1]["at"], start / 30 - 0.31)
            self.assertLessEqual(marks[1]["at"], start / 30 + 0.7)

    def test_a_chapter_under_ten_seconds_joins_the_one_before(self):
        marks, notes = chapters.chapters_from(timeline([600, 780, 1800, 3000]))
        self.assertEqual([m["title"] for m in marks], ["Will this cost me money?", "Chapter 1", "Chapter 3", "Chapter 4"])
        self.assertIn("joins that chapter", notes[0])

    def test_a_last_chapter_under_ten_seconds_joins_the_one_before(self):
        marks, notes = chapters.chapters_from(timeline([600, 1800, 3000, 5200]))
        self.assertEqual(marks[-1]["title"], "Chapter 3")
        self.assertIn("to the end", notes[0])

    def test_fewer_than_three_writes_none(self):
        marks, notes = chapters.chapters_from(timeline([900], frames=1800))
        self.assertEqual(marks, [])
        self.assertIn("YouTube needs 3", notes[-1])

    def test_a_chapter_at_frame_zero_is_the_first_mark(self):
        marks, _ = chapters.chapters_from(timeline([0, 900, 1800], hook=None))
        self.assertEqual([m["title"] for m in marks], ["Chapter 1", "Chapter 2", "Chapter 3"])

    def test_titles_can_be_the_typed_questions(self):
        marks, _ = chapters.chapters_from(timeline([634, 1283, 2401]), titles="ask")
        self.assertEqual(marks[1]["title"], "Question 1?")

    def test_times_over_an_hour(self):
        self.assertEqual(chapters.fmt(59), "0:59")
        self.assertEqual(chapters.fmt(125), "2:05")
        self.assertEqual(chapters.fmt(3725), "1:02:05")


@unittest.skipUnless(shutil.which("node") and (HOWTO / "node_modules").exists(), "needs node and templates/howto/node_modules")
class Pilot(unittest.TestCase):
    """The shipped pilot: the timeline the render plays gives the chapters its published description carries."""

    def test_timeline_reads_and_starts_every_chapter(self):
        run = subprocess.run(["node", str(HOWTO / "scripts" / "timeline.mjs")], capture_output=True, text=True, cwd=HOWTO)
        self.assertEqual(run.returncode, 0, run.stderr)
        t = json.loads(run.stdout)
        self.assertEqual(sum(s["frames"] for s in t["segments"]), t["frames"])
        self.assertTrue(all(c["start"] is not None for c in t["chapters"]))
        marks, _ = chapters.chapters_from(t)
        self.assertEqual(marks[0]["at"], 0)
        self.assertEqual(len(marks), len(t["chapters"]) + 1)


if __name__ == "__main__":
    unittest.main()
