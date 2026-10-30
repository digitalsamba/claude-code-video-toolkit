import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from composite import RvmMatte, composite
from media_ops import cv2_module, encode_frames, probe, sha256


@unittest.skipUnless(importlib.util.find_spec('cv2') and shutil.which('ffmpeg'), 'vision extra and FFmpeg required')
class CompositingTests(unittest.TestCase):
    def fixture(self, root):
        foreground = root / 'presenter.mp4'
        def frames():
            for i in range(10):
                frame = np.zeros((90, 160, 3), dtype=np.uint8)
                frame[:, :, 1] = 255
                frame[35:65, 20 + i * 5:50 + i * 5] = [0, 0, 255]
                yield frame
        encode_frames(frames(), foreground, width=160, height=90, fps=10)
        Image.new('RGB', (160, 90), 'blue').save(root / 'background.png')
        Image.new('RGBA', (20, 10), 'yellow').save(root / 'label.png')
        report = {'version': 1, 'sourceSha256': sha256(foreground), 'width': 160, 'height': 90, 'fps': 10,
                  'frames': [{'frame': i, 'box': [20 + i * 5, 35, 30, 30]} for i in range(10)]}
        return foreground, report

    def test_chromakey_replaces_background_and_graphic_follows_subject(self):
        cv2 = cv2_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, report = self.fixture(root)
            layers = [{'path': 'label.png', 'attachToTrack': True}]
            output = root / 'composited.mp4'
            preview = composite(video, root / 'background.png', output, key='#00ff00',
                                overlays=layers, overlay_root=root, report=report)
            self.assertFalse(preview['executed'])
            self.assertFalse(output.exists())
            result = composite(video, root / 'background.png', output, key='#00ff00',
                               overlays=layers, overlay_root=root, report=report, execute=True)
            self.assertEqual(result['frames'], 10)
            capture = cv2.VideoCapture(str(output))
            ok, first = capture.read()
            capture.set(cv2.CAP_PROP_POS_FRAMES, 9)
            ok2, last = capture.read()
            capture.release()
            self.assertTrue(ok and ok2)
            self.assertGreater(first[5, 5, 0], 200)  # blue background
            self.assertLess(first[5, 5, 1], 30)
            self.assertGreater(first[45, 30, 2], 180)  # red foreground
            self.assertGreater(first[28, 33, 1], 180)  # yellow attached label
            self.assertGreater(last[28, 78, 1], 180)
            self.assertLess(last[28, 33, 1], 40)
            self.assertAlmostEqual(probe(output)['duration'], 1, delta=.1)

    def test_supplied_alpha_and_stale_tracks(self):
        cv2 = cv2_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, report = self.fixture(root)
            mask = np.zeros((90, 160), dtype=np.uint8)
            mask[35:65, 20:95] = 255
            Image.fromarray(mask).save(root / 'alpha.png')
            output = root / 'alpha-composite.mp4'
            composite(video, root / 'background.png', output, alpha=root / 'alpha.png', execute=True)
            capture = cv2.VideoCapture(str(output))
            ok, image = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(image[45, 30, 2], 180)
            self.assertGreater(image[5, 5, 0], 200)
            report['sourceSha256'] = 'stale'
            with self.assertRaises(ValueError):
                composite(video, root / 'background.png', root / 'stale.mp4', key='#00ff00', report=report, execute=True)
            self.assertFalse((root / 'stale.mp4').exists())


@unittest.skipUnless(os.environ.get('VIDEO_TOOLKIT_RVM_MODEL') and importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('cv2'), 'set VIDEO_TOOLKIT_RVM_MODEL for real optional model inference')
class RvmInferenceTests(unittest.TestCase):
    def test_real_model_recycles_states_and_produces_foreground_alpha(self):
        cv2 = cv2_module()
        image = cv2.imread(str(ROOT / 'examples/ds-release-649/public/people/elena.jpg'))
        image = cv2.resize(image, (256, 256))
        engine = RvmMatte(os.environ['VIDEO_TOOLKIT_RVM_MODEL'], 1)
        for _ in range(3):
            foreground, alpha = engine.matte(image)
            self.assertEqual(foreground.shape, image.shape)
            self.assertEqual(alpha.shape, (256, 256, 1))
            self.assertTrue(np.isfinite(alpha).all())
        self.assertTrue(all(state.shape != (1, 1, 1, 1) for state in engine.states))
        self.assertGreater(float(alpha.max()), .5)
        self.assertLess(float(alpha.min()), .5)

    def test_real_model_full_video_composition(self):
        cv2 = cv2_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = cv2.imread(str(ROOT / 'examples/ds-release-649/public/people/elena.jpg'))
            image = cv2.resize(image, (256, 256))
            source, output = root / 'person.mp4', root / 'set.mp4'
            encode_frames((image for _ in range(4)), source, width=256, height=256, fps=10)
            Image.new('RGB', (256, 256), 'blue').save(root / 'background.png')
            result = composite(source, root / 'background.png', output,
                               rvm_model=os.environ['VIDEO_TOOLKIT_RVM_MODEL'], execute=True)
            self.assertEqual(result['frames'], 4)
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(np.abs(frame.astype(float) - image).mean(), 5)
