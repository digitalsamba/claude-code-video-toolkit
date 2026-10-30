import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from data_video import chart_frames, dataset, prepare, render
from media_ops import probe, run


class DataVideoTests(unittest.TestCase):
    def fixture(self, root, kind='line'):
        (root / 'data.csv').write_text('period,series,value\nJan,A,10\nFeb,A,20\nJan,B,15\nFeb,B,12\n')
        spec = root / 'chart.json'
        spec.write_text(json.dumps({'version': 1, 'type': kind, 'dataset': 'data.csv', 'title': 'Release usage',
                                    'unit': 'users', 'width': 320, 'height': 180, 'fps': 10, 'durationSeconds': 1}))
        return spec

    def test_frames_animate_and_narration_uses_actual_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = prepare(self.fixture(root))
            self.assertIn('10 in Jan, 20 in Feb', report['narrationText'])
            self.assertIn('Change: -3 users', report['narrationText'])
            frames = list(chart_frames(report))
            self.assertEqual(len(frames), 10)
            self.assertEqual(frames[0].shape, (180, 320, 3))
            self.assertGreater(np.mean(np.abs(frames[0].astype(float) - frames[-1])), .3)
            (root / 'data.csv').write_text('period,series,value\nJan,A,10\nJan,A,20\n')
            with self.assertRaises(ValueError):
                prepare(root / 'chart.json')

    def test_json_bad_numbers_and_missing_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            for rows in [[{'p': 'Jan', 's': 'A', 'v': True}],
                         [{'p': 'Jan', 's': 'A', 'v': 'NaN'}],
                         [{'p': 'Jan', 's': 'A', 'v': 1}, {'p': 'Feb', 's': 'B', 'v': 2}]]:
                path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    dataset(path, {'period': 'p', 'series': 's', 'value': 'v'})

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_narrated_bar_video_has_correct_duration_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            narration = root / 'narration.wav'
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=duration=1.4', narration])
            report = prepare(self.fixture(root, 'bar'), narration)
            render(report, root / 'chart.mp4')
            result = probe(root / 'chart.mp4')
            self.assertAlmostEqual(result['duration'], 1.4, delta=.15)
            self.assertTrue(any(s['codec_type'] == 'audio' for s in result['streams']))
