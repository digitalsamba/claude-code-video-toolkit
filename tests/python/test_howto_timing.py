"""The howto template's timing.py: phrases laid out per scene, and a folder of phrases per episode."""
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("howto_timing", ROOT / "templates" / "howto" / "scripts" / "timing.py")
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def phrase(folder: Path, stem: str, seconds: float, words: list[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(seconds), str(folder / f"{stem}.mp3")],
        check=True,
    )
    step = int(seconds * 1000 / len(words))
    caps = [{"text": w, "startMs": i * step, "endMs": (i + 1) * step} for i, w in enumerate(words)]
    (folder / f"{stem}.captions.json").write_text(json.dumps(caps))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
class Layout(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.scenes = self.tmp / "public" / "audio" / "scenes"
        timing.ROOT = self.tmp
        timing.SCENES = self.scenes
        timing.LAYOUT = self.tmp / "public" / "audio" / "layout.json"
        timing.OUT = self.tmp / "src" / "timing.json"
        timing.OUT.parent.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def read(self):
        self.assertEqual(timing.main(), 0)
        return json.loads(timing.OUT.read_text())

    def test_one_episode_is_laid_out_at_the_top(self):
        phrase(self.scenes, "01a_free", 1.0, ["all", "free"])
        phrase(self.scenes, "01b_free", 2.0, ["every", "part"])
        out = self.read()
        self.assertNotIn("episodes", out)
        free = out["scenes"]["free"]
        self.assertEqual([p["at"] for p in free["phrases"]], [18, 18 + 30 + 12])
        self.assertEqual(free["frames"], 18 + 30 + 12 + 60 + 30)

    def test_a_folder_of_phrases_is_an_episode(self):
        phrase(self.scenes / "ep2", "01a_form", 1.0, ["schedule", "a", "call"])
        phrase(self.scenes / "ep3", "01a_form", 2.0, ["one", "link"])
        out = self.read()
        self.assertEqual(out["scenes"], {})
        self.assertEqual(sorted(out["episodes"]), ["ep2", "ep3"])
        self.assertEqual(out["episodes"]["ep2"]["scenes"]["form"]["frames"], 18 + 30 + 30)
        self.assertEqual(out["episodes"]["ep3"]["scenes"]["form"]["phrases"][0]["words"][1]["text"], "link")

    def test_layout_is_per_episode(self):
        phrase(self.scenes / "ep2", "01a_form", 1.0, ["schedule"])
        phrase(self.scenes / "ep3", "01a_form", 1.0, ["schedule"])
        timing.LAYOUT.write_text(json.dumps({"ep3": {"form": {"lead": 40}}}))
        out = self.read()
        self.assertEqual(out["episodes"]["ep2"]["scenes"]["form"]["phrases"][0]["at"], 18)
        self.assertEqual(out["episodes"]["ep3"]["scenes"]["form"]["phrases"][0]["at"], 40)

    def test_a_phrase_with_no_captions_fails(self):
        phrase(self.scenes, "01a_free", 1.0, ["free"])
        (self.scenes / "01a_free.captions.json").unlink()
        self.assertEqual(timing.main(), 1)


if __name__ == "__main__":
    unittest.main()
