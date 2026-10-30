import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build import builtin_plan
from production import BuildError
from storyboard_plan import prepare_props
ROOT = Path(__file__).resolve().parents[1]


class StoryboardPlanTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffprobe'), 'FFprobe required')
    def test_explicit_audio_sets_actual_duration_and_is_not_regenerated(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'public').mkdir()
            with wave.open(str(project / 'public/voice.wav'), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b'\0\0' * 6400)
            data = {'name': 'Demo', 'template': 'storyboard', 'scenes': [
                {'id': 'intro', 'type': 'title', 'narration': 'Existing recording', 'audioFile': 'voice.wav'}]}
            (project / 'project.json').write_text(json.dumps(data))
            props = prepare_props(project)
            self.assertAlmostEqual(props['project']['scenes'][0]['durationSeconds'], 1.0, places=2)
            self.assertFalse(any(task['id'].startswith('voice-') for task in builtin_plan(project)['tasks']))

    def test_manifest_narration_generates_per_scene_tasks_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            data = {'name': 'Demo', 'template': 'storyboard', 'format': 'portrait', 'scenes': [
                {'id': 'intro', 'type': 'title', 'narration': 'Hello'}, {'id': 'demo', 'type': 'demo', 'narration': 'Try it'}]}
            (project / 'project.json').write_text(json.dumps(data))
            plan = builtin_plan(project, captions=True)
            scripts = [task for task in plan['tasks'] if task['id'].startswith('script-')]
            self.assertEqual(len(scripts), 2)
            self.assertEqual(scripts[0]['command'][-1], 'Hello')
            self.assertIn('StoryboardPortrait', next(task for task in plan['tasks'] if task['id'] == 'render')['command'])
            data['scenes'][0]['narration'] = 'Changed words'
            (project / 'project.json').write_text(json.dumps(data))
            new = builtin_plan(project, captions=True)
            new_scripts = [task for task in new['tasks'] if task['id'].startswith('script-')]
            self.assertNotEqual(scripts[0], new_scripts[0])
            self.assertEqual(scripts[1], new_scripts[1])
            self.assertEqual([item.name for item in project.iterdir()], ['project.json'])

    def test_missing_assets_block_props_and_silent_output_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            data = {'name': 'Demo', 'template': 'storyboard', 'scenes': [{'id': 'intro', 'type': 'title', 'durationSeconds': 1}]}
            (project / 'project.json').write_text(json.dumps(data))
            self.assertEqual(prepare_props(project)['project']['scenes'][0]['durationSeconds'], 1)
            self.assertIn('--allow-silent', builtin_plan(project)['tasks'][-1]['command'])
            data['scenes'][0]['visual'] = {'type': 'image', 'asset': 'missing.png'}
            (project / 'project.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(BuildError, 'missing storyboard asset'):
                prepare_props(project)


if __name__ == '__main__': unittest.main()
