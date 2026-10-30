import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UnicodeCliTests(unittest.TestCase):
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
