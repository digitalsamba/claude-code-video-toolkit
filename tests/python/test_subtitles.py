import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import captions
from captions import export_subtitles


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

    def test_only_vtt_escapes_markup_and_ampersands(self):
        words = [{'text': 'A & <b>two</b>\n\n --> three', 'startMs': 0, 'endMs': 500}]
        srt = export_subtitles(words)
        self.assertIn('A & <b>two</b>', srt)
        self.assertNotIn('&amp;', srt)
        self.assertEqual(srt.count('-->'), 1)
        vtt = export_subtitles(words, format='vtt')
        self.assertIn('A &amp; &lt;b&gt;two&lt;/b&gt;', vtt)
        self.assertEqual(vtt.count('-->'), 1)

    def test_sentence_and_pause_boundaries_respect_word_limit(self):
        words = [{'text': text, 'startMs': start, 'endMs': end} for text, start, end in [
            (' One.', 0, 300), (' Two', 300, 500), (' three', 1200, 1400),
            (' four', 1400, 1500), (' five', 1500, 1700)]]
        cues = export_subtitles(words, max_words=2).strip().split('\n\n')
        self.assertEqual([cue.splitlines()[-1] for cue in cues], ['One.', 'Two', 'three four', 'five'])
        self.assertIn('00:00:00,300 --> 00:00:00,500', cues[1])

    def test_cli_offline_export_overwrites_without_transcription(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'words.captions.json'
            target = Path(directory) / 'words.srt'
            source.write_text(json.dumps([{'text': 'مرحبا & hello', 'startMs': 0, 'endMs': 500}]), encoding='utf-8')
            command = [sys.executable, str(ROOT / 'tools/captions.py'), str(source), '--srt', '--vtt', '--json']
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, encoding='utf-8',
                                        env={**os.environ, 'PYTHONIOENCODING': 'ascii', 'ELEVENLABS_API_KEY': ''})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(json.loads(result.stdout)['engine'])
            self.assertIn('مرحبا & hello', target.read_text(encoding='utf-8'))
            self.assertIn('&amp;', target.with_suffix('.vtt').read_text(encoding='utf-8'))
            saved = target.read_bytes()
            source.write_text('[{"text":"bad","startMs":500,"endMs":0}]')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(target.read_bytes(), saved)

    def test_transcription_creates_json_and_both_subtitles_in_one_call(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / 'voice.wav'
            audio.write_bytes(b'fixture')
            heard = [{'w': 'Author', 's': 0., 'e': .4}, {'w': 'text.', 's': .4, 'e': .8}]
            with patch.dict(os.environ, {'ELEVENLABS_API_KEY': 'test-key'}), patch.object(captions, 'transcribe_scribe', return_value=heard) as transcribe, redirect_stdout(io.StringIO()):
                captions.main([str(audio), '--engine', 'scribe', '--srt', '--vtt', '--json'])
                transcribe.assert_called_once()
            self.assertEqual(json.loads(audio.with_suffix('.captions.json').read_text())[1]['text'], ' text.')
            self.assertIn('Author text.', audio.with_suffix('.srt').read_text())
            self.assertTrue(audio.with_suffix('.vtt').read_text().startswith('WEBVTT'))

    def test_failed_atomic_write_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'words.srt'
            target.write_text('previous')
            with patch.object(captions.os, 'replace', side_effect=OSError('write denied')):
                with self.assertRaises(OSError):
                    captions.write_text_atomic(target, 'replacement')
            self.assertEqual(target.read_text(), 'previous')
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
