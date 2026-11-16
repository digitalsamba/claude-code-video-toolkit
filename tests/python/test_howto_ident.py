"""The brand ident slot and the howto template's reader of it (templates/howto/scripts/ident.py). No Blender."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRAND = ROOT / "brands" / "video-toolkit"

spec = importlib.util.spec_from_file_location("howto_ident", ROOT / "templates" / "howto" / "scripts" / "ident.py")
ident = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ident)


class Slot(unittest.TestCase):
    """brands/video-toolkit/ident is the filled slot the docs point at: it has to be what they say."""

    def setUp(self):
        self.manifest = json.loads((BRAND / "ident" / "ident.json").read_text())

    def test_default_is_one_of_the_idents(self):
        self.assertIn(self.manifest["default"], self.manifest["idents"])

    def test_every_listed_file_is_there(self):
        for name, parts in self.manifest["idents"].items():
            for part, entry in parts.items():
                for f in [*entry["files"].values(), entry["sound"]]:
                    self.assertTrue((BRAND / "ident" / f).exists(), f"{name} {part}: {f}")

    def test_open_has_a_hit_then_a_title_frame_inside_the_clip(self):
        for name, parts in self.manifest["idents"].items():
            o = parts["open"]
            self.assertEqual(o["fps"], 30)
            self.assertLess(o["hit_frame"], o["title_frame"], name)
            self.assertLess(o["title_frame"], o["frames"], name)


class Reader(unittest.TestCase):
    def test_brand_comes_from_the_import_not_the_comment(self):
        self.assertEqual(ident.brand_name(), "video-toolkit")

    def test_mark_has_a_fill_for_each_ground(self):
        mark = ident.read_mark(BRAND / "assets")
        self.assertEqual([p["id"] for p in mark["parts"]], ["slash", "play"])
        self.assertEqual(mark["parts"][0]["light"], "#3771e0")
        self.assertEqual(mark["parts"][1]["dark"], "#ffffff")
        self.assertTrue(all(p["mono"] == "#ffffff" for p in mark["parts"]))
        self.assertTrue(all(p["d"].startswith("M") and p["d"].endswith("Z") for p in mark["parts"]))
        x, y, w, h = mark["box"]
        self.assertGreater(w, h)  # the slash and the triangle side by side

    def test_mark_alone_leaves_other_grounds_to_the_template(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "mark.svg").write_text((BRAND / "assets" / "mark.svg").read_text())
            mark = ident.read_mark(Path(d))
        self.assertTrue(all(p["dark"] is None and p["mono"] is None for p in mark["parts"]))

    def test_no_mark_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ident.read_mark(Path(d)))

    def test_template_ships_filled_from_the_slot(self):
        shipped = json.loads((ROOT / "templates" / "howto" / "src" / "ident.json").read_text())
        manifest = json.loads((BRAND / "ident" / "ident.json").read_text())
        for use in ("episode", "short"):
            for part, clip in shipped[use].items():
                if part == "name":
                    continue
                self.assertTrue((ROOT / "templates" / "howto" / "public" / clip["file"]).exists(), clip["file"])
                self.assertEqual(clip["frames"], manifest["idents"][shipped[use]["name"]][part]["frames"])
        self.assertNotIn("open", shipped["short"])


if __name__ == "__main__":
    unittest.main()
