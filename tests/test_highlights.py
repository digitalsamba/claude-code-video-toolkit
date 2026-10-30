import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from highlights import render, select
from media_ops import probe, run
from transcript import segments

WORDS = [{'text': text, 'startMs': start, 'endMs': end} for text, start, end in [
    ('Useful', 100, 400), ('result.', 500, 900),
    ('Useful', 1200, 1500), ('result.', 1600, 1900),
    ('New', 2200, 2500), ('feature', 2600, 2900), ('works.', 4500, 4900),
    ('Unfinished', 5200, 5500)]]


class HighlightTests(unittest.TestCase):
    def test_complete_sentences_dedup_and_pause_removal(self):
        plan = select(WORDS, 6, budget=3)
        self.assertEqual([s['text'] for s in plan['sentences']], ['Useful result.', 'New feature works.'])
        self.assertTrue(plan['omittedIncompleteTail'])
        self.assertLessEqual(plan['durationSeconds'], 3)
        self.assertEqual(len(plan['segments']), 3)
        self.assertLess(plan['segments'][1]['endSeconds'], 3.2)
        self.assertGreater(plan['segments'][2]['startSeconds'], 4)

    def test_keywords_and_bad_timing(self):
        plan = select(WORDS, 6, budget=1.8, keywords=['feature'])
        self.assertEqual(plan['sentences'][0]['text'], 'New feature works.')
        for budget in [float('nan'), -1, True]:
            with self.assertRaises(ValueError):
                select(WORDS, 6, budget=budget)
        with self.assertRaises(ValueError):
            select(WORDS, 1)
        with self.assertRaises(ValueError):
            segments({'segments': [{'start': 1, 'end': 0, 'text': 'No.'}]})
        with self.assertRaises(ValueError):
            select([{'text': 'No punctuation', 'startMs': 0, 'endMs': 1000}], 2)

    def test_padding_does_not_include_the_next_sentences_words(self):
        words = [{'text': 'Release.', 'startMs': 100, 'endMs': 900},
                 {'text': 'Other.', 'startMs': 950, 'endMs': 1800}]
        plan = select(words, 2, budget=1.1, keywords=['release'])
        self.assertEqual(len(plan['sentences']), 1)
        self.assertLessEqual(plan['segments'][0]['endSeconds'], .95)
        words[1]['startMs'] = 800
        with self.assertRaises(ValueError):
            select(words, 2)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_actual_render_matches_edl_duration_with_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'input.mp4'
            output = Path(directory) / 'highlights.mp4'
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=20:d=6',
                 '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6', '-c:v', 'libx264', '-c:a', 'aac', video])
            plan = select(WORDS, 6, budget=3)
            render(video, output, plan)
            media = probe(output)
            self.assertAlmostEqual(media['duration'], plan['durationSeconds'], delta=.15)
            self.assertTrue(any(s['codec_type'] == 'audio' for s in media['streams']))
