"""Phrase-to-motion behavior and the real CLI, without network or audio providers."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.cues import main, resolve_cues


ROOT = Path(__file__).resolve().parents[1]


def captions(text="Open settings then apply changes then apply again", shift=0):
    return [{"text": word, "startMs": i * 400 + shift, "endMs": i * 400 + 300 + shift}
            for i, word in enumerate(text.split())]


class NarrationCueTests(unittest.TestCase):
    def test_retiming_audio_moves_the_same_named_action(self):
        spec = {"cues": {"open": {"phrase": "open settings", "offsetMs": -100}}}
        original = resolve_cues(captions(), spec, audio_offset_ms=500, fps=30)
        retimed = resolve_cues(captions(shift=1000), spec, audio_offset_ms=500, fps=30)
        self.assertEqual(original["cues"]["open"]["frame"], 12)
        self.assertEqual(retimed["cues"]["open"]["frame"], 42)
        self.assertEqual(retimed["cues"]["open"]["timeMs"], 1400)

    def test_repeated_word_requires_an_explicit_occurrence(self):
        with self.assertRaisesRegex(ValueError, "matches 2 times; set occurrence"):
            resolve_cues(captions(), {"cues": {"apply": {"phrase": "apply"}}})
        result = resolve_cues(captions(), {"cues": {"apply": {
            "phrase": "apply", "occurrence": 2, "anchor": "end", "offsetMs": 50,
        }}}, fps=60)
        self.assertEqual(result["cues"]["apply"]["timeMs"], 2750)
        self.assertEqual(result["cues"]["apply"]["frame"], 165)

    def test_case_punctuation_and_unicode_apostrophes(self):
        result = resolve_cues(captions('Choose "CAFÉ," then it’s ready.'),
                              {"cues": {"ready": {"phrase": "THEN it's ready"}}})
        self.assertEqual(result["cues"]["ready"]["startMs"], 800)
        self.assertEqual(result["cues"]["ready"]["endMs"], 1900)
        cafe = resolve_cues(captions('Choose "CAFÉ,"'), {"cues": {"cafe": {"phrase": "café"}}})
        self.assertEqual(cafe["cues"]["cafe"]["timeMs"], 400)

    def test_phrase_must_be_contiguous_whole_words(self):
        for phrase in ("app", "open apply"):
            with self.subTest(phrase=phrase), self.assertRaisesRegex(ValueError, "was not found"):
                resolve_cues(captions(), {"cues": {"bad": {"phrase": phrase}}})

    def test_nearest_frame_rounding_and_fractional_fps(self):
        spec = {"cues": {"a": {"phrase": "open", "offsetMs": 50}}}
        self.assertEqual(resolve_cues(captions(), spec, fps=30)["cues"]["a"]["frame"], 2)
        self.assertEqual(resolve_cues(captions(), spec, fps=29.97)["cues"]["a"]["frame"], 1)

    def test_invalid_word_times_fail(self):
        bad_words = [
            [{"text": "Open", "startMs": -1, "endMs": 100}],
            [{"text": "Open", "startMs": 101, "endMs": 100}],
            [{"text": "Open", "startMs": "0", "endMs": 100}],
            [{"text": "Open", "startMs": True, "endMs": 100}],
            [{"text": "Open", "startMs": float("nan"), "endMs": 100}],
            [{"text": "Open", "endMs": 100}],
            list(reversed(captions())),
        ]
        for words in bad_words:
            with self.subTest(words=words), self.assertRaises(ValueError):
                resolve_cues(words, {"cues": {"a": {"phrase": "open"}}})

    def test_invalid_specs_and_offsets_fail(self):
        definitions = [
            {}, {"phrase": "!!!"}, {"phrase": "open", "anchor": "middle"},
            {"phrase": "open", "occurrence": 0}, {"phrase": "open", "occurrence": True},
            {"phrase": "open", "occurrence": None},
            {"phrase": "apply", "occurrence": 3}, {"phrase": "open", "occurence": 1},
            {"phrase": "open", "offsetMs": -1}, {"phrase": "open", "offsetMs": float("inf")},
            {"phrase": "open", "offsetMs": 10 ** 400},
        ]
        for definition in definitions:
            with self.subTest(definition=definition), self.assertRaises(ValueError):
                resolve_cues(captions(), {"cues": {"bad": definition}})
        for fps in (0, -1, True, float("nan")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                resolve_cues(captions(), {"cues": {"a": {"phrase": "open"}}}, fps=fps)

    def test_checked_in_example_matches_real_cli_output(self):
        example = ROOT / "examples" / "narration-cues"
        process = subprocess.run([
            sys.executable, str(ROOT / "tools" / "cues.py"),
            "--captions", str(example / "public" / "narration.captions.json"),
            "--spec", str(example / "cue-spec.json"), "--fps", "30",
        ], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(process.stdout),
                         json.loads((example / "src" / "cues.json").read_text(encoding="utf-8")))

    def test_failed_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            words, spec, output = (folder / name for name in ("words.json", "spec.json", "cues.json"))
            words.write_text(json.dumps(captions()), encoding="utf-8")
            spec.write_text('{"cues":{"missing":{"phrase":"delete"}}}', encoding="utf-8")
            output.write_text("previous cues", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(["--captions", str(words), "--spec", str(spec), "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "previous cues")

    def test_cli_writes_nested_output_and_rejects_source_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            words, spec = folder / "words.json", folder / "spec.json"
            words.write_text(json.dumps(captions()), encoding="utf-8")
            spec.write_text('{"cues":{"open":{"phrase":"open"}}}', encoding="utf-8")
            output = folder / "generated" / "cues.json"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--captions", str(words), "--spec", str(spec),
                                       "--output", str(output)]), 0)
                self.assertEqual(main(["--captions", str(words), "--spec", str(spec),
                                       "--output", str(words)]), 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["cues"]["open"]["timeMs"], 0)
            self.assertEqual(json.loads(words.read_text(encoding="utf-8")), captions())


if __name__ == "__main__":
    unittest.main()
