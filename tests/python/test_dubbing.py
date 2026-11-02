import json
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from dubbing import dub, import_transcript, prepare
from media_ops import probe, run


class DubbingSpecTests(unittest.TestCase):
    def test_import_preserves_speaker_ids_and_requires_labels(self):
        transcript = {'segments': [{'start': 0, 'end': 1, 'text': 'Hello.', 'speaker': 'A'},
                                    {'start': 2, 'end': 3, 'text': 'Hola.', 'speaker': 'B'}]}
        spec = import_transcript(transcript)
        self.assertEqual(list(spec['speakers']), ['A', 'B'])
        self.assertEqual(spec['segments'][1]['speaker'], 'B')
        with self.assertRaises(ValueError):
            prepare(spec, Path('.'), 4)
        transcript['segments'][0].pop('speaker')
        with self.assertRaises(ValueError):
            import_transcript(transcript)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_real_timeline_has_two_voices_and_preserved_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, output = root / 'video.mp4', root / 'dub.mp4'
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=160x90:r=10:d=4', '-c:v', 'libx264', video])
            for name, frequency, duration in [('a', 440, .9), ('b', 880, 1.1), ('background', 220, 4)]:
                run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'sine=frequency={frequency}:duration={duration}', root / f'{name}.wav'])
            spec = {'version': 1, 'speakers': {'A': {}, 'B': {}}, 'background': 'background.wav', 'segments': [
                {'id': 'a', 'speaker': 'A', 'startSeconds': .5, 'endSeconds': 1.5, 'text': 'First.', 'audio': 'a.wav'},
                {'id': 'b', 'speaker': 'B', 'startSeconds': 2.5, 'endSeconds': 3.5, 'text': 'Second.', 'audio': 'b.wav'}]}
            spec_file = root / 'dub.json'
            spec_file.write_text(json.dumps(spec))
            self.assertFalse(dub(video, spec_file, output)['executed'])
            self.assertFalse(output.exists())
            supplied = spec['segments'][0].pop('audio')
            spec['speakers']['A'] = {'provider': 'elevenlabs', 'voiceId': 'configured-voice'}
            spec_file.write_text(json.dumps(spec))
            with patch('dubbing.synthesize', side_effect=AssertionError('preview must not synthesize')):
                self.assertEqual(dub(video, spec_file, output)['synthesizedSegments'], 1)
            spec['segments'][0]['audio'] = supplied
            spec_file.write_text(json.dumps(spec))
            report = dub(video, spec_file, output, execute=True)
            self.assertEqual([t['speaker'] for t in report['timing']], ['A', 'B'])
            self.assertTrue(report['backgroundPreserved'])
            self.assertAlmostEqual(probe(output)['duration'], 4, delta=.15)
            decoded = root / 'result.wav'
            run(['ffmpeg', '-v', 'error', '-i', output, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', decoded])
            with wave.open(str(decoded)) as stream:
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(float)
            def power(start, frequency):
                window = samples[int(start * 16000):int((start + .2) * 16000)]
                return abs(np.sum(window * np.exp(-2j * np.pi * frequency * np.arange(len(window)) / 16000)))
            self.assertGreater(power(.8, 440), power(2.8, 440) * 5)
            self.assertGreater(power(2.8, 880), power(.8, 880) * 5)
            self.assertGreater(power(1.8, 220), 10000)
            spec['segments'][0]['endSeconds'] = .6
            spec_file.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                dub(video, spec_file, root / 'too-fast.mp4', execute=True)
            self.assertFalse((root / 'too-fast.mp4').exists())


class CommandLineTests(unittest.TestCase):
    def test_import_preserves_unicode_under_redirected_ascii_output(self):
        import os
        import subprocess
        command = Path(__file__).resolve().parents[2] / 'tools/dubbing.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'transcript.json'
            source.write_text(json.dumps({'segments': [
                {'start': 0, 'end': 1, 'text': 'مرحبا بالعالم.', 'speaker': 'محدث'}
            ]}), encoding='utf-8')
            result = subprocess.run([sys.executable, str(command), 'import', str(source),
                '--output', str(root / 'dub.json')], env={**os.environ, 'PYTHONIOENCODING': 'ascii'},
                capture_output=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['segments'][0]['speaker'], 'محدث')

    def test_help_without_models_or_credentials(self):
        import os
        import subprocess
        command = Path(__file__).resolve().parents[2] / "tools/dubbing.py"
        result = subprocess.run([sys.executable, str(command), "--help"],
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
            capture_output=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
