"""music_gen.py picks a cloud that is set up, and makes takes on it."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import music_gen  # noqa: E402


class DefaultCloudTest(unittest.TestCase):
    def test_a_modal_endpoint_in_the_env_is_used_before_the_hosted_api(self):
        self.assertEqual(music_gen.default_cloud({"MODAL_MUSIC_GEN_ENDPOINT_URL": "https://x"}), "modal")

    def test_modal_comes_before_runpod_when_both_are_set_up(self):
        env = {"MODAL_MUSIC_GEN_ENDPOINT_URL": "https://x", "RUNPOD_ACESTEP_ENDPOINT_ID": "abc", "RUNPOD_API_KEY": "k"}
        self.assertEqual(music_gen.default_cloud(env), "modal")

    def test_runpod_needs_its_key_as_well_as_its_endpoint(self):
        self.assertEqual(music_gen.default_cloud({"RUNPOD_ACESTEP_ENDPOINT_ID": "abc", "RUNPOD_API_KEY": "k"}), "runpod")
        self.assertEqual(music_gen.default_cloud({"RUNPOD_ACESTEP_ENDPOINT_ID": "abc"}), "acemusic")

    def test_with_nothing_set_up_the_hosted_api_is_the_last_resort(self):
        self.assertEqual(music_gen.default_cloud({}), "acemusic")
        self.assertEqual(music_gen.default_cloud({"MODAL_MUSIC_GEN_ENDPOINT_URL": ""}), "acemusic")


class TakePathsTest(unittest.TestCase):
    def test_one_take_keeps_the_path_it_was_given(self):
        self.assertEqual(music_gen.take_paths("public/audio/bg.mp3", 1), ["public/audio/bg.mp3"])

    def test_takes_are_numbered_the_way_pick_music_expects(self):
        got = [Path(p).as_posix() for p in music_gen.take_paths("public/audio/bg.mp3", 3)]
        self.assertEqual(got, ["public/audio/bg_1.mp3", "public/audio/bg_2.mp3", "public/audio/bg_3.mp3"])


if __name__ == "__main__":
    unittest.main()
