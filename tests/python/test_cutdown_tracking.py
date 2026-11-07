import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import cutdown
from cutdown import Segment, Source
from cutdown_tracking import crop_dimensions, rectangle, track_subject, write_crop_commands


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], check=True, capture_output=True)
    return json.loads(result.stdout)


class GeometryTests(unittest.TestCase):
    def test_crop_geometry_and_pan_speed_and_initial_focus(self):
        src = Source(320, 180, 2, '10/1', False, None)
        self.assertEqual(crop_dimensions(src, 90, 160, 1), (102, 180))
        with self.assertRaises(ValueError):
            rectangle([300, 0, 100, 20], 320, 180)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'track.cmd'
            write_crop_commands(path, [{'center': None}, {'center': None}], src, 90, 160,
                                focus=.25, zoom=1, smoothing=.2, max_speed=100, label='track0')
            self.assertIn('crop@track0 x 54', path.read_text())
            write_crop_commands(path, [{'center': [60, 90]}, {'center': [300, 90]}], src, 90, 160,
                                focus=.5, zoom=1, smoothing=1, max_speed=100, label='track0')
            self.assertIn('0.100000000 crop@track0 x 19', path.read_text())


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class DependencyTests(unittest.TestCase):
    def test_missing_vision_does_not_affect_other_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'input.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=160x90:d=0.5', str(video)], check=True, capture_output=True)
            err = io.StringIO()
            with patch.dict(sys.modules, {'cv2': None}), redirect_stdout(io.StringIO()), redirect_stderr(err):
                self.assertEqual(cutdown.main(['-i', str(video), '--mode', 'track']), 1)
                self.assertIn('uv sync --extra vision', err.getvalue())
                self.assertEqual(cutdown.main(['-i', str(video), '--mode', 'crop', '--dry-run']), 0)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


@unittest.skipUnless(importlib.util.find_spec('cv2') and shutil.which('ffmpeg') and shutil.which('ffprobe'), 'vision and FFmpeg required')
class TrackingTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv2 = cv2
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video = self.folder / 'moving.mp4'
        texture = np.random.default_rng(7).integers(60, 255, (40, 40, 3), dtype=np.uint8)
        frames = []
        for i in range(20):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            frame[70:110, 30 + i * 5:70 + i * 5] = texture
            frames.append(frame)
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '320x180', '-r', '10', '-i', 'pipe:0', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(self.video)], input=np.stack(frames).tobytes(), check=True, capture_output=True)
        self.src = cutdown.probe(self.video)

    def invoke(self, *args, formats=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch.dict(cutdown.FORMATS, formats or {'9:16': (90, 160), '1:1': (100, 100)}):
            result = cutdown.main(['-i', str(self.video), *args, '--json'])
        self.assertEqual(result, 0, err.getvalue())
        return json.loads(out.getvalue())

    def test_csrt_motion_renders_and_overwrites_with_audio(self):
        tracks = track_subject(self.video, self.src, [Segment(0, 2)], [30, 70, 40, 40])
        self.assertGreater(tracks[0][-1]['center'][0], 130)
        output = self.folder / 'tracked.mp4'
        for _ in range(2):
            self.invoke('--mode', 'track', '--box', '30,70,40,40', '--track-smoothing', '1', '-o', str(output))
        info = probe(output)
        video = next(s for s in info['streams'] if s['codec_type'] == 'video')
        self.assertEqual((video['width'], video['height']), (90, 160))
        self.assertTrue(any(s['codec_type'] == 'audio' for s in info['streams']))
        capture = self.cv2.VideoCapture(str(output))
        capture.set(self.cv2.CAP_PROP_POS_FRAMES, 19)
        ok, last = capture.read()
        capture.release()
        self.assertTrue(ok)
        self.assertGreater(last[60:100, 30:60].mean(), 25)

    def test_segments_platform_and_multiple_formats_keep_joined_audio(self):
        report = self.invoke('--mode', 'track', '--platform', 'shorts', '-f', '9:16', '-f', '1:1', '--segment', '0-0.9@0.2', '--segment', '1.1-2@0.8', '--output-dir', str(self.folder / 'outputs'))
        self.assertEqual(len(report['outputs']), 2)
        for output in report['outputs']:
            info = probe(output['output'])
            self.assertAlmostEqual(float(info['format']['duration']), 1.8, delta=.15)
            self.assertTrue(any(s['codec_type'] == 'audio' for s in info['streams']))
            self.assertEqual(output['segments'], 2)

    def test_occlusion_multiple_faces_and_distant_face_hold_last_position(self):
        detector = Mock()
        detector.empty.return_value = False
        detector.detectMultiScale.side_effect = [np.array([[30, 70, 40, 40]]), np.array([[30, 70, 40, 40], [220, 70, 40, 40]]), (), np.array([[240, 70, 40, 40]])] + [()] * 16
        with patch.object(self.cv2, 'CascadeClassifier', return_value=detector):
            entries = track_subject(self.video, self.src, [Segment(0, 2)])[0]
        self.assertEqual(entries[0]['center'], [50, 90])
        self.assertTrue(all(entry['held'] for entry in entries[1:]))
        self.assertTrue(all(entry['center'] == [50, 90] for entry in entries))

    def test_hard_scene_cut_holds_and_vfr_is_rejected(self):
        detector = Mock()
        detector.empty.return_value = False
        detector.detectMultiScale.return_value = np.array([[30, 70, 40, 40]])
        with patch.object(self.cv2, 'CascadeClassifier', return_value=detector), patch.object(self.cv2, 'absdiff', return_value=np.full((18, 32), 255, np.uint8)):
            entries = track_subject(self.video, self.src, [Segment(0, 2)])[0]
        self.assertTrue(all(entry['sceneCut'] and entry['held'] for entry in entries[1:]))
        self.assertTrue(all(entry['center'] == entries[0]['center'] for entry in entries))
        self.src.avg_fps = '8/1'
        with self.assertRaisesRegex(ValueError, 'constant FPS'):
            track_subject(self.video, self.src, [Segment(0, 2)])

    def test_other_modes_still_render_and_failed_render_preserves_output(self):
        for mode in ('blur', 'crop', 'pad'):
            self.invoke('--mode', mode, '--start', '0.2', '--end', '0.8', '-o', str(self.folder / f'{mode}.mp4'), formats={'9:16': (360, 640)})
        target = self.folder / 'existing.mp4'
        target.write_bytes(b'old video')
        original_run = subprocess.run
        def run(command, **kwargs):
            if command[0] == 'ffmpeg':
                return subprocess.CompletedProcess(command, 1, '', 'simulated encoder failure')
            return original_run(command, **kwargs)
        with patch.object(subprocess, 'run', side_effect=run), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cutdown.main(['-i', str(self.video), '--mode', 'crop', '-o', str(target)]), 1)
        self.assertEqual(target.read_bytes(), b'old video')
        self.assertEqual(list(self.folder.glob('.existing-*')), [])
