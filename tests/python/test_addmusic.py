import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from addmusic import build_mix_command, load_events, mix_audio_with_video
from qc import measure_loudness


def ff(*args):
    return subprocess.run(['ffmpeg', '-y', '-v', 'error', *map(str, args)], capture_output=True, check=True)


def samples(path, folder):
    wav = folder / 'decoded.wav'
    ff('-i', path, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav)
    with wave.open(str(wav)) as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16).astype(float)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class AddMusicTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.video, self.music, self.voice, self.output = [self.folder / p for p in ('video.mp4', 'music.wav', 'voice.wav', 'mix.mp4')]
        ff('-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10:d=6', '-c:v', 'libx264', self.video)
        ff('-f', 'lavfi', '-i', 'sine=frequency=220:duration=6', self.music)
        ff('-f', 'lavfi', '-i', "sine=frequency=880:duration=6,volume=4,volume=enable='lt(t,1)+gte(t,2)':volume=0", self.voice)

    def test_ducking_recovers_and_fixed_mix_stays_fixed(self):
        options = dict(speech=self.voice, original_volume=0, music_volume=1, release=100, verbose=False)
        self.assertTrue(mix_audio_with_video(self.video, self.music, self.output, duck=True, **options))
        data = samples(self.output, self.folder)
        rms = lambda a, b: np.sqrt(np.mean(data[int(a * 16000):int(b * 16000)] ** 2))
        self.assertLess(rms(1.3, 1.8), rms(.3, .8) * .4)
        self.assertGreater(rms(3, 3.5), rms(1.3, 1.8) * 2)
        self.assertTrue(mix_audio_with_video(self.video, self.music, self.output, **options))
        data = samples(self.output, self.folder)
        self.assertAlmostEqual(rms(1.3, 1.8) / rms(.3, .8), 1, delta=.1)

    def test_events_are_relative_and_timed_and_music_loops_on_silent_video(self):
        effect = self.folder / 'chime.wav'
        ff('-f', 'lavfi', '-i', 'sine=frequency=440:duration=0.2', effect)
        events_file = self.folder / 'effects.json'
        events_file.write_text(json.dumps([{'path': 'chime.wav', 'atSeconds': 1}]))
        events = load_events(events_file, 6)
        self.assertEqual(Path(events[0]['path']), effect.resolve())
        self.assertTrue(mix_audio_with_video(self.video, self.music, self.output, music_volume=0, events=events, verbose=False))
        data = samples(self.output, self.folder)
        self.assertGreater(np.sqrt(np.mean(data[16480:18560] ** 2)), 100)
        self.assertLess(np.max(np.abs(data[3200:6400])), 5)
        self.assertLess(np.max(np.abs(data[32000:40000])), 5)
        self.assertTrue(mix_audio_with_video(self.video, effect, self.output, verbose=False))
        data = samples(self.output, self.folder)
        self.assertGreater(np.sqrt(np.mean(data[80000:88000] ** 2)), 50)

    def test_failures_preserve_output_and_reject_input_collisions(self):
        self.output.write_bytes(b'existing output')
        self.assertFalse(mix_audio_with_video(self.video, self.folder / 'missing.wav', self.output, verbose=False))
        self.assertEqual(self.output.read_bytes(), b'existing output')
        self.assertEqual(list(self.folder.glob('.mix-*')), [])
        for invalid in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                build_mix_command(self.video, self.music, self.output, threshold=invalid)
        with self.assertRaises(ValueError):
            build_mix_command(self.video, self.music, self.video)
        with self.assertRaises(ValueError):
            build_mix_command(self.video, self.music, self.output, duck=True)

    def test_cli_defaults_overwrite_and_dry_run_is_side_effect_free(self):
        command = [sys.executable, str(ROOT / 'tools/addmusic.py'), '-i', str(self.video), '-m', str(self.music), '--speech', str(self.voice), '--duck', '-o', str(self.output), '--json']
        for _ in range(2):
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['success'])
        before = self.output.read_bytes()
        result = subprocess.run([*command, '--dry-run'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.output.read_bytes(), before)
        new_output = self.folder / 'uncreated' / 'mix.mp4'
        result = subprocess.run([sys.executable, str(ROOT / 'tools/addmusic.py'), '-i', str(self.video), '-p', 'ambient', '-o', str(new_output), '--dry-run', '--json'], capture_output=True, text=True, env={**os.environ, 'ELEVENLABS_API_KEY': ''})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(new_output.parent.exists())

    def test_mono_low_rate_narration_keeps_stereo_music_at_48k(self):
        voice = self.folder / 'voice-mono.wav'
        music = self.folder / 'music-stereo.wav'
        ff('-f', 'lavfi', '-i', 'sine=frequency=880:duration=6:sample_rate=22050', '-ac', '1', voice)
        ff('-f', 'lavfi', '-i', 'sine=frequency=220:duration=6:sample_rate=44100',
           '-f', 'lavfi', '-i', 'sine=frequency=330:duration=6:sample_rate=44100',
           '-filter_complex', '[0:a][1:a]join=inputs=2:channel_layout=stereo', music)
        self.assertTrue(mix_audio_with_video(self.video, music, self.output, speech=voice, original_volume=0, duck=True, verbose=False))
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels',
                                '-of', 'json', str(self.output)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)['streams'][0]
        self.assertEqual((int(stream['sample_rate']), stream['channels']), (48000, 2))
        left, right = (self.folder / 'left.wav', self.folder / 'right.wav')
        ff('-i', self.output, '-af', 'pan=mono|c0=c0', '-ar', '16000', left)
        ff('-i', self.output, '-af', 'pan=mono|c0=c1', '-ar', '16000', right)
        read = lambda p: np.frombuffer(wave.open(str(p)).readframes(10 ** 6), dtype=np.int16).astype(float)
        self.assertGreater(np.max(np.abs(read(left) - read(right))), 100)  # still two different channels

    def test_missing_effect_file_is_named(self):
        events_file = self.folder / 'effects.json'
        events_file.write_text(json.dumps([{'path': 'nope.wav', 'atSeconds': 1}]))
        with self.assertRaisesRegex(ValueError, 'Effect file not found: .*nope.wav'):
            load_events(events_file, 6)

    def test_ducked_mix_passes_delivery_loudness_after_qc_fix(self):
        self.assertTrue(mix_audio_with_video(self.video, self.music, self.output, speech=self.voice, duck=True, verbose=False))
        result = subprocess.run([sys.executable, str(ROOT / 'tools/qc.py'), '--render', str(self.output), '--fix', '--skip-frames', '--json'], capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        normalized = self.output.with_name('mix.loudnorm.mp4')
        self.assertTrue(normalized.is_file())
        self.assertAlmostEqual(measure_loudness(normalized)['lufs'], -14, delta=1)


class CliHelpTests(unittest.TestCase):
    def test_help_without_credentials(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/addmusic.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--duck', result.stdout)
