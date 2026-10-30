import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from production import BuildError
from subtitles import export_subtitles
from variants import derive_variants, materialize

ROOT = Path(__file__).resolve().parents[1]


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.base = {'name': 'Demo', 'template': 'storyboard', 'publish': {'privacy': 'public'},
            'scenes': [{'id': 'intro', 'type': 'title', 'title': 'Hello Remotion', 'onScreenText': 'Click here',
                'narration': 'Welcome', 'audioFile': 'old.mp3', 'captionsFile': 'old.json', 'durationSeconds': 2}]}
        self.spec = {'version': 1, 'glossary': {'Remotion': 'Remotion'}, 'variants': [
            {'id': 'es', 'locale': 'es', 'scenes': {'intro': {'title': 'Hola Remotion', 'onScreenText': 'Haz clic', 'narration': 'Bienvenido'}}},
            {'id': 'hook-b', 'kind': 'campaign', 'scenes': {'intro': {'title': 'Ship your next video'}}}]}

    def test_translations_keep_sources_and_drop_stale_audio(self):
        original = copy.deepcopy(self.base)
        variants = derive_variants(self.base, self.spec)
        self.assertEqual(self.base, original)
        self.assertEqual(variants[0]['scenes'][0]['narration'], 'Bienvenido')
        self.assertNotIn('audioFile', variants[0]['scenes'][0])
        self.assertNotIn('publish', variants[0])
        self.assertEqual(variants[1]['scenes'][0]['audioFile'], 'old.mp3')

    def test_incomplete_translations_glossary_and_ids_fail(self):
        for modify in ('missing', 'glossary', 'duplicate', 'path', 'unknown'):
            spec = copy.deepcopy(self.spec)
            if modify == 'missing': del spec['variants'][0]['scenes']['intro']['title']
            if modify == 'glossary': spec['variants'][0]['scenes']['intro']['title'] = 'Hola'
            if modify == 'duplicate': spec['variants'][1]['id'] = 'ES'
            if modify == 'path': spec['variants'][0]['id'] = '../escape'
            if modify == 'unknown': spec['variants'][0]['scenes']['missing'] = {}
            with self.subTest(modify=modify), self.assertRaises(BuildError):
                derive_variants(self.base, spec)

    def test_materialize_is_readonly_by_default_and_audio_plan_is_coordinated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variants = derive_variants(self.base, self.spec)
            result = materialize(root, variants, with_audio=True)
            self.assertEqual(list(root.iterdir()), [])
            self.assertTrue(any(task['id'].startswith('vo-es') for task in result['production']['tasks']))
            materialize(root, variants, write=True, with_audio=True)
            props = json.loads((root / 'variants/es/render-props.json').read_text())
            self.assertIn('variants/es/audio/', props['project']['scenes'][0]['audioFile'])
            self.assertEqual((root / 'variants/es/scripts/intro-1.txt').read_text().strip(), 'Bienvenido')
            with self.assertRaisesRegex(BuildError, 'already exists'):
                materialize(root, variants, write=True)

    def test_cli_creates_variant_props_and_downloadable_subtitles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'project.json').write_text(json.dumps(self.base))
            spec = root / 'variants-spec.json'
            spec.write_text(json.dumps(self.spec))
            result = subprocess.run([sys.executable, str(ROOT / 'tools/variants.py'), '--project', directory,
                                     '--spec', str(spec), '--write'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)['variants']), 2)
            words = [{'text': ' Hola', 'startMs': 0, 'endMs': 500}, {'text': ' mundo', 'startMs': 600, 'endMs': 1000}]
            self.assertIn('00:00:00,000 --> 00:00:01,000\nHola mundo', export_subtitles(words))
            self.assertTrue(export_subtitles(words, format='vtt').startswith('WEBVTT'))
            words[1]['startMs'] = -1
            with self.assertRaises(ValueError): export_subtitles(words)


if __name__ == '__main__': unittest.main()
