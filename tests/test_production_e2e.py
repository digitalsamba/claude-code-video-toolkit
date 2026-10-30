"""Offline integration: real local provider, captions, FFmpeg, QC and checkpoints."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ProductionEndToEndTests(unittest.TestCase):
    def test_local_pipeline_renders_checks_resumes_and_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            request = project / 'request.json'
            request.write_text(json.dumps({'capability': 'image', 'prompt': 'Initial story'}))
            (project / 'words.json').write_text(json.dumps([
                {'text': 'Hello', 'startMs': 0, 'endMs': 400}, {'text': 'world', 'startMs': 500, 'endMs': 900}]))
            tasks = [
                {'id': 'card', 'command': ['{python}', '{toolkit}/tools/provider.py', '--adapter',
                    '{toolkit}/examples/providers/local-card/adapter.json', '--input', '{project}/request.json',
                    '--project', '{project}', '--output', 'public/card.svg', '--run'],
                 'inputs': ['request.json'], 'outputs': ['public/card.svg'], 'estimatedCostUsd': 0},
                {'id': 'subtitles', 'command': ['{python}', '{toolkit}/tools/subtitles.py', '{project}/words.json',
                    '--output', '{project}/out/captions.srt'], 'inputs': ['words.json'], 'outputs': ['out/captions.srt'], 'estimatedCostUsd': 0},
                {'id': 'render', 'dependsOn': ['card', 'subtitles'], 'command': ['ffmpeg', '-y', '-v', 'error',
                    '-f', 'lavfi', '-i', 'testsrc2=s=320x180:r=30:d=1.5', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=1.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                    '-shortest', '{project}/out/video.mp4'], 'inputs': ['public/card.svg', 'out/captions.srt'],
                 'outputs': ['out/video.mp4'], 'estimatedCostUsd': 0},
                {'id': 'qc', 'dependsOn': ['render'], 'command': ['{python}', '{toolkit}/tools/qc.py',
                    '--render', '{project}/out/video.mp4', '--json'], 'inputs': ['out/video.mp4'],
                 'outputs': [], 'stdoutFile': 'out/qc.json', 'estimatedCostUsd': 0}]
            plan = project / 'production.json'
            plan.write_text(json.dumps({'version': 1, 'budgetUsd': 0, 'tasks': tasks}))
            cli = [sys.executable, str(ROOT / 'tools/build.py'), '--project', directory,
                   '--plan', str(plan), '--json']
            def run(*options):
                result = subprocess.run([*cli, *options], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return json.loads(result.stdout)
            preview = run()
            self.assertTrue(preview['budget']['allowed'])
            self.assertFalse((project / '.video-toolkit').exists())
            first = run('--run')
            self.assertEqual(set(first['completed']), {'card', 'subtitles', 'render', 'qc'})
            self.assertGreater((project / 'out/video.mp4').stat().st_size, 1000)
            qc = json.loads((project / 'out/qc.json').read_text())
            self.assertFalse(any(item['level'] == 'error' for item in qc['findings']))
            resumed = run('--run')
            self.assertEqual(set(resumed['resumed']), {'card', 'subtitles', 'render', 'qc'})
            # Rebuilding uses a new filename, since provider adapters protect outputs.
            tasks[0]['command'][tasks[0]['command'].index('public/card.svg')] = 'public/card-v2.svg'
            tasks[0]['outputs'] = ['public/card-v2.svg']
            tasks[2]['inputs'][0] = 'public/card-v2.svg'
            request.write_text(json.dumps({'capability': 'image', 'prompt': 'Revised story'}))
            plan.write_text(json.dumps({'version': 1, 'budgetUsd': 0, 'tasks': tasks}))
            revised = run('--run')
            self.assertEqual(set(revised['completed']), {'card', 'render', 'qc'})
            self.assertEqual(revised['resumed'], ['subtitles'])
            versions = list((project / '.video-toolkit/assets/records').glob('*.json'))
            self.assertEqual(len(versions), 2)


if __name__ == '__main__': unittest.main()
