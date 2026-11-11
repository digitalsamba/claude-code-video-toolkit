"""A brand's voice.json can name several voices; --voice picks one."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import config  # noqa: E402


def brand(root: Path, name: str, voice: dict) -> None:
    d = root / "brands" / name
    d.mkdir(parents=True)
    (d / "brand.json").write_text("{}")
    (d / "voice.json").write_text(json.dumps(voice))


class BrandVoices(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        brand(root, "single", {"voiceId": "ONE"})
        brand(root, "multi", {
            "default": "b", "settings": {"stability": 0.5},
            "voices": {"a": {"voiceId": "A"}, "b": {"voiceId": "B", "settings": {"stability": 0.9}}},
        })
        brand(root, "nodefault", {"voices": {"first": {"voiceId": "F"}, "second": {"voiceId": "S"}}})
        patch = mock.patch.object(config, "find_workspace_root", return_value=root)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_single_voice_unchanged(self):
        self.assertEqual(config.load_brand_voice_config("single"), {"voiceId": "ONE"})
        self.assertEqual(config.list_brand_voices("single"), [])

    def test_single_voice_rejects_a_name(self):
        with self.assertRaises(ValueError):
            config.load_brand_voice_config("single", "a")

    def test_default_and_named(self):
        self.assertEqual(config.load_brand_voice_config("multi")["voiceId"], "B")
        a = config.load_brand_voice_config("multi", "a")
        self.assertEqual((a["voiceId"], a["voiceName"]), ("A", "a"))
        self.assertEqual(a["settings"], {"stability": 0.5})  # shared top-level settings carry over
        self.assertEqual(config.load_brand_voice_config("multi", "b")["settings"], {"stability": 0.9})
        self.assertEqual(config.list_brand_voices("multi"), ["a", "b"])

    def test_no_default_takes_first(self):
        self.assertEqual(config.load_brand_voice_config("nodefault")["voiceId"], "F")

    def test_unknown_name_lists_the_voices(self):
        with self.assertRaisesRegex(ValueError, "a, b"):
            config.load_brand_voice_config("multi", "zed")

    def test_toolkit_brand_ships_two_voices(self):
        mock.patch.stopall()
        self.assertEqual(config.list_brand_voices("video-toolkit"), ["marilyn", "bren"])


if __name__ == "__main__":
    unittest.main()
