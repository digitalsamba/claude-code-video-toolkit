import json
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audio_mix import mix
from media_ops import load_json, number, probe, run, staged_output


@unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
class AudioMixTests(unittest.TestCase):
    def test_music_ducks_under_speech_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, music, voice = [root / x for x in ['video.mp4', 'music.wav', 'voice.wav']]
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=160x90:r=10:d=4', '-c:v', 'libx264', video])
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=4', music])
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                 "sine=frequency=880:duration=4,volume=4,volume=enable='lt(t,1)+gte(t,2)':volume=0", voice])
            output = root / 'mix.mp4'
            preview = mix(video, music, output, speech=voice)
            self.assertFalse(preview['executed'])
            self.assertFalse(output.exists())
            # Mute audible speech while keeping it as the sidechain control.
            mix(video, music, output, speech=voice, speech_gain=0, music_gain=1, release=100, execute=True)
            decoded = root / 'decoded.wav'
            run(['ffmpeg', '-v', 'error', '-i', output, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', decoded])
            with wave.open(str(decoded)) as stream:
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(float)
            def rms(start, end):
                return np.sqrt(np.mean(samples[int(start * 16000):int(end * 16000)] ** 2))
            self.assertLess(rms(1.3, 1.8), rms(.3, .8) * .4)
            self.assertGreater(rms(3, 3.5), rms(1.3, 1.8) * 2)
            self.assertAlmostEqual(probe(output)['duration'], 4, delta=.15)
            with self.assertRaises(ValueError):
                mix(video, music, output, speech=voice, execute=True)
            with self.assertRaises(ValueError):
                mix(video, music, root / 'bad.mp4', speech=voice, threshold=float('nan'))
            self.assertFalse((root / 'bad.mp4').exists())

    def test_timed_effect_and_invalid_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, music = root / 'video.mp4', root / 'music.wav'
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=160x90:r=10:d=2',
                 '-f', 'lavfi', '-i', 'anullsrc=r=16000:cl=mono', '-t', '2', '-c:v', 'libx264', '-c:a', 'aac', video])
            run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.2', music])
            mix(video, music, root / 'out.mp4', music_gain=0,
                events=[{'path': str(music), 'atSeconds': 1}], execute=True)
            with self.assertRaises(ValueError):
                mix(video, music, root / 'bad.mp4', events=[{'path': str(music), 'atSeconds': 10}])


class MediaBoundaryTests(unittest.TestCase):
    def test_nonfinite_or_huge_numbers_are_validation_errors(self):
        for value in (True, float('inf'), float('nan'), 10**1000, '1'):
            with self.assertRaises(ValueError):
                number(value, 'test')

    def test_json_and_atomic_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'data.json'
            target.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                load_json(target)
            target.write_text(json.dumps({'a': 1}))
            with self.assertRaises(ValueError):
                with staged_output(target):
                    pass
            other = target.with_name('other.json')
            with self.assertRaises(RuntimeError):
                with staged_output(other) as temporary:
                    temporary.write_text('incomplete')
                    raise RuntimeError('interrupted')
            self.assertFalse(other.exists())
