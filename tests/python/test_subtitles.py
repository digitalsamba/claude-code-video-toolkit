import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from subtitles import export_subtitles


class SubtitleTests(unittest.TestCase):
    def test_srt_and_vtt_preserve_text_and_timing(self):
        words = [{'text': ' Hola', 'startMs': 0, 'endMs': 500},
                 {'text': ' mundo', 'startMs': 600, 'endMs': 1000}]
        self.assertEqual(export_subtitles(words), '1\n00:00:00,000 --> 00:00:01,000\nHola mundo\n')
        vtt = export_subtitles(words, format='vtt', max_words=1)
        self.assertTrue(vtt.startswith('WEBVTT\n\n'))
        self.assertIn('2\n00:00:00.600 --> 00:00:01.000\nmundo', vtt)

    def test_invalid_timing_and_word_limits_are_rejected(self):
        for value in (True, float('nan'), 10**1000, -1):
            with self.subTest(value=str(value)[:20]), self.assertRaises(ValueError):
                export_subtitles([{'text': 'hello', 'startMs': value, 'endMs': 500}])
        for limit in (True, 1.5, 0, 31):
            with self.assertRaises(ValueError):
                export_subtitles([], max_words=limit)
        with self.assertRaises(ValueError):
            export_subtitles([{'text': 'a', 'startMs': 0, 'endMs': 500},
                              {'text': 'b', 'startMs': 200, 'endMs': 600}])

    def test_untrusted_text_cannot_create_extra_cues_or_markup(self):
        result = export_subtitles([{'text': '<b>one</b>\n\n --> two', 'startMs': 0, 'endMs': 500}])
        self.assertEqual(result.count('-->'), 1)
        self.assertIn('&lt;b&gt;one&lt;/b&gt;', result)

    def test_cli_writes_unicode_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'words.json', Path(directory) / 'captions.srt'
            source.write_text(json.dumps([{'text': 'مرحبا', 'startMs': 0, 'endMs': 500}]), encoding='utf-8')
            command = [sys.executable, str(ROOT / 'tools/subtitles.py'), str(source), '--output', str(target)]
            result = subprocess.run(command, capture_output=True, encoding='utf-8',
                                    env={**os.environ, 'PYTHONIOENCODING': 'ascii'})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('مرحبا', target.read_text(encoding='utf-8'))
            saved = target.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(target.read_bytes(), saved)


if __name__ == '__main__':
    unittest.main()
