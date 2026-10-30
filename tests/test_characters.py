import base64
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from characters import checked_lock, lock, production_plan, render_shot
from media_ops import probe, run
from production import validate_plan


class CharacterTests(unittest.TestCase):
    def fixture(self, root):
        Image.new('RGB', (64, 64), 'orange').save(root / 'reference.png')
        spec = {'version': 1, 'style': 'Soft daylight, watercolor', 'characters': {'lea': {
            'description': 'Short brown hair, round glasses', 'wardrobe': 'Blue jacket', 'references': ['reference.png']}},
            'shots': [{'id': 'intro', 'characters': ['lea'], 'prompt': 'Standing in a studio',
                       'motionPrompt': 'A small wave', 'imageCostUsd': .05, 'videoCostUsd': .1}]}
        locked = lock(spec, root)
        path = root / 'lookbook.lock.json'
        path.write_text(json.dumps(locked))
        return spec, locked, path

    def test_reference_hashes_seeds_plan_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, locked, path = self.fixture(root)
            self.assertEqual(lock(spec, root), locked)
            plan = production_plan(root, path)
            self.assertEqual(len(plan['tasks']), 2)
            self.assertEqual(plan['tasks'][1]['dependsOn'], ['intro-image'])
            validate_plan(plan, root)
            result = render_shot(root, path, 'intro', 'intro.png')
            self.assertFalse(result['executed'])
            self.assertIn('Blue jacket', result['prompt'])
            self.assertIn('Soft daylight', result['prompt'])
            self.assertFalse((root / 'intro.png').exists())
            Image.new('RGB', (64, 64), 'blue').save(root / 'reference.png')
            with self.assertRaises(ValueError):
                checked_lock(locked, root)

    def test_existing_flux_transport_receives_references_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, locked, path = self.fixture(root)
            png = io.BytesIO()
            Image.new('RGB', (64, 64), 'orange').save(png, format='PNG')
            # Exercise the real existing FLUX encode/decode path, without a paid API request.
            with patch('cloud_gpu.call_cloud_endpoint', return_value=({'image_base64': base64.b64encode(png.getvalue()).decode()}, .01)) as endpoint, patch('flux2.get_r2_payload_config', return_value=None):
                report = render_shot(root, path, 'intro', 'intro.png', execute=True)
            payload = endpoint.call_args.kwargs['payload']['input']
            self.assertEqual(payload['seed'], locked['shots'][0]['seed'])
            self.assertEqual(payload['operation'], 'edit')
            self.assertEqual(base64.b64decode(payload['image_base64']), (root / 'reference.png').read_bytes())
            self.assertTrue(report['assetVersion'])
            self.assertEqual(len(list((root / '.video-toolkit/assets/records').glob('*.json'))), 1)
            with self.assertRaises(ValueError):
                render_shot(root, path, 'intro', '../escape.png')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_existing_ltx_transport_receives_conditioning_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, path = self.fixture(root)
            sample = root / 'response.mp4'
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=64x64:r=24:d=0.5', '-c:v', 'libx264', sample])
            response = {'video_base64': base64.b64encode(sample.read_bytes()).decode()}
            with patch('cloud_gpu.call_cloud_endpoint', return_value=(response, .01)) as endpoint, patch('ltx2.get_r2_payload_config', return_value=None):
                render_shot(root, path, 'intro', 'motion.mp4', kind='video', image='reference.png', execute=True)
            payload = endpoint.call_args.kwargs['payload']['input']
            self.assertIn('A small wave', payload['prompt'])
            self.assertEqual(base64.b64decode(payload['image_base64']), (root / 'reference.png').read_bytes())
            self.assertGreater(probe(root / 'motion.mp4')['duration'], .4)
