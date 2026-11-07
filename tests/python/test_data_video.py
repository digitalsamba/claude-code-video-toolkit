import json
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from data_video import chart_frames, dataset, prepare, render
from data_video import text_overlay


def run(command):
    return subprocess.run(list(map(str, command)), check=True, capture_output=True).stdout


def probe(path):
    info = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]))
    return {**info, 'duration': float(info['format']['duration'])}


class DataVideoTests(unittest.TestCase):
    def fixture(self, root, kind='line'):
        (root / 'data.csv').write_text('period,series,value\nJan,A,10\nFeb,A,20\nJan,B,15\nFeb,B,12\n')
        spec = root / 'chart.json'
        spec.write_text(json.dumps({'version': 1, 'type': kind, 'dataset': 'data.csv', 'title': 'Release usage',
                                    'unit': 'users', 'width': 320, 'height': 180, 'fps': 10, 'durationSeconds': 1}))
        return spec

    def test_frames_animate_without_derived_narration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = prepare(self.fixture(root))
            self.assertNotIn('narrationText', report)
            self.assertEqual(report['headline'], '')
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
            render(report, root / 'chart.mp4')  # Rerender replaces a completed output.
            result = probe(root / 'chart.mp4')
            self.assertAlmostEqual(result['duration'], 1.4, delta=.15)
            self.assertTrue(any(s['codec_type'] == 'audio' for s in result['streams']))
            video = next(s for s in result['streams'] if s['codec_type'] == 'video')
            self.assertEqual((video['width'], video['height']), (320, 180))
            (root / 'data.csv').write_text('period,series,value\nJan,A,50\nFeb,A,60\n')
            with self.assertRaisesRegex(ValueError, 'Dataset changed'):
                render(report, root / 'stale.mp4')
            self.assertFalse((root / 'stale.mp4').exists())


    def test_brand_and_author_text_are_separate_overlay_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            data = json.loads(spec.read_text())
            data.update(source='Source: release metrics', headline='42 active users')
            spec.write_text(json.dumps(data))
            report = prepare(spec, brand='digital-samba')
            self.assertEqual(report['theme']['palette'][0], '#F06859')
            self.assertIn('Inter', report['theme']['fontFamily'])
            self.assertTrue(Path(report['theme']['fontPath']).is_file())
            overlay = np.asarray(text_overlay(report))
            self.assertGreater(overlay[:35, :, 3].sum(), 0)
            self.assertGreater(overlay[165:, :, 3].sum(), 0)
            self.assertEqual(report['headline'], '42 active users')
            self.assertNotIn('narrationText', report)
            self.assertEqual(len(report['datasetSha256']), 64)

    def test_brand_path_traversal_and_unknown_brand_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.fixture(Path(directory))
            for brand in ('../default', 'not-a-real-brand'):
                with self.assertRaises((ValueError, OSError)):
                    prepare(spec, brand=brand)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_example_delegates_to_shared_renderer_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            output = root / 'example.mp4'
            example = Path(__file__).resolve().parents[2] / 'examples/data-viz-chart/build.py'
            command = [sys.executable, str(example), '--spec', str(spec), '--output', str(output), '--brand', 'digital-samba']
            result = subprocess.run([*command, '--dry-run'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['executed'])
            self.assertAlmostEqual(probe(output)['duration'], 1, delta=.15)
            report = subprocess.run([sys.executable, str(example), '--dry-run'], capture_output=True, text=True)
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertEqual(json.loads(report.stdout)['durationSeconds'], 15)


class CommandLineTests(unittest.TestCase):
    def test_help_without_models_or_credentials(self):
        import os
        import subprocess
        command = Path(__file__).resolve().parents[2] / "tools/data_video.py"
        result = subprocess.run([sys.executable, str(command), "--help"],
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
            capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
