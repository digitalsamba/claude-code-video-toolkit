import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from production import BuildError
from provider import run_adapter
from extensions import scaffold, validate_template
ROOT = Path(__file__).resolve().parents[1]


class ExtensionTests(unittest.TestCase):
    def test_local_provider_preview_execute_history_and_output_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = ROOT / 'examples/providers/local-card/adapter.json'
            request = {'capability': 'image', 'prompt': 'A <reviewed> title', 'parameters': {'seed': 2}}
            result = run_adapter(adapter, request, root, 'public/title.svg')
            self.assertFalse(result['executed'])
            self.assertEqual(list(root.iterdir()), [])
            result = run_adapter(adapter, request, root, 'public/title.svg', run=True)
            self.assertTrue(result['executed'])
            self.assertIn('&lt;reviewed&gt;', (root / 'public/title.svg').read_text())
            self.assertTrue((root / '.video-toolkit/assets/records' / (result['assetVersion'] + '.json')).is_file())
            with self.assertRaisesRegex(BuildError, 'already exists'):
                run_adapter(adapter, request, root, 'public/title.svg', run=True)
            with self.assertRaises(BuildError):
                run_adapter(adapter, request, root, '../outside.svg')

    def test_missing_env_and_invalid_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = json.loads((ROOT / 'examples/providers/local-card/adapter.json').read_text())
            data['requiredEnv'] = ['TOOLKIT_TEST_NONEXISTENT_KEY']
            adapter = root / 'adapter.json'
            adapter.write_text(json.dumps(data))
            request = {'capability': 'image', 'prompt': 'text'}
            self.assertFalse(run_adapter(adapter, request, root, 'image.svg')['ready'])
            with self.assertRaisesRegex(BuildError, 'Missing adapter environment'):
                run_adapter(adapter, request, root, 'image.svg', run=True)
            request['capability'] = 'video'
            with self.assertRaises(BuildError): run_adapter(adapter, request, root, 'video.mp4')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_scaffold_renders_real_video(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'template'
            scaffold(target, 'new-template')
            self.assertEqual(validate_template(target)['engine'], 'python')
            result = subprocess.run([sys.executable, 'build.py'], cwd=target, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((target / 'out/video.mp4').stat().st_size, 1000)
            with self.assertRaises(BuildError): scaffold(target, 'new-template')


if __name__ == '__main__': unittest.main()
