import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from media_ops import cv2_module, encode_frames, probe
from reframe import crop_box, rectangle, reframe, track, validate_track


class CropTests(unittest.TestCase):
    def test_crop_clamps_and_retains_target_aspect(self):
        self.assertEqual(crop_box((10, 50), (320, 180), (90, 160)), [0, 0, 101, 180])
        self.assertEqual(crop_box((400, 50), (320, 180), (90, 160)), [219, 0, 101, 180])
        with self.assertRaises(ValueError):
            rectangle([300, 0, 100, 100], 320, 180)


@unittest.skipUnless(importlib.util.find_spec('cv2') and shutil.which('ffmpeg'), 'vision extra and FFmpeg required')
class TrackingTests(unittest.TestCase):
    def test_csrt_follows_motion_and_reframes_real_video(self):
        cv2 = cv2_module()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'moving.mp4', Path(directory) / 'vertical.mp4'
            texture = np.random.default_rng(7).integers(60, 255, (40, 40, 3), dtype=np.uint8)
            def frames():
                for i in range(20):
                    image = np.zeros((180, 320, 3), dtype=np.uint8)
                    image[70:110, 30 + i * 5:70 + i * 5] = texture
                    yield image
            encode_frames(frames(), source, width=320, height=180, fps=10)
            report = track(source, box=[30, 70, 40, 40], max_lost=0)
            self.assertGreater(report['frames'][-1]['box'][0], 110)
            self.assertLess(abs(report['frames'][-1]['box'][0] - 125), 10)
            self.assertEqual(len(report['frames']), 20)
            reframe(source, output, report, width=90, height=160, smoothing=1)
            stream = next(s for s in probe(output)['streams'] if s['codec_type'] == 'video')
            self.assertEqual((stream['width'], stream['height']), (90, 160))
            capture = cv2.VideoCapture(str(output))
            capture.set(cv2.CAP_PROP_POS_FRAMES, 19)
            ok, last = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(last[60:100, 30:60].mean(), 25)
            report['sourceSha256'] = 'stale'
            with self.assertRaises(ValueError):
                validate_track(report, source)
