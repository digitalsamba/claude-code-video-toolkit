import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UnicodeCliTests(unittest.TestCase):
    def test_capability_help_needs_no_model_or_credentials(self):
        for tool in ('audio_mix', 'highlights', 'reframe', 'dubbing', 'characters', 'data_video', 'evidence', 'composite'):
            with self.subTest(tool=tool):
                result = subprocess.run([sys.executable, str(ROOT / f'tools/{tool}.py'), '--help'],
                    env={**os.environ, 'PYTHONIOENCODING': 'ascii'}, capture_output=True, encoding='utf-8', timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_dubbing_import_outputs_unicode_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'transcript.json'
            source.write_text(json.dumps({'segments': [{'start': 0, 'end': 1, 'text': 'مرحبا بالعالم.',
                                                        'speaker': 'محدث'}]}), encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT / 'tools/dubbing.py'), 'import', str(source),
                '--output', str(root / 'dub.json')], env={**os.environ, 'PYTHONIOENCODING': 'ascii'},
                capture_output=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['segments'][0]['speaker'], 'محدث')

    def test_redirected_console_keeps_unicode_json_under_ascii_locale(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'title.svg').write_text('local image', encoding='utf-8')
            metadata = project / 'metadata.json'
            metadata.write_text(json.dumps({'prompt': 'مرحبا بالعالم'}), encoding='utf-8')
            result = subprocess.run([sys.executable, str(ROOT / 'tools/asset_history.py'), '--project', directory,
                '--json', 'record', 'title.svg', '--metadata', str(metadata)],
                env={**os.environ, 'PYTHONIOENCODING': 'ascii'}, capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['metadata']['prompt'], 'مرحبا بالعالم')


if __name__ == '__main__': unittest.main()
