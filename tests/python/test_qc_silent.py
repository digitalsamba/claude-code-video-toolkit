import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SilentOutputTests(unittest.TestCase):
    def check_video(self, video, *flags):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/qc.py'),
            '--project', str(video.parent), '--render', str(video), '--skip-frames', '--json', *flags],
            capture_output=True, encoding='utf-8', timeout=30)
        return result.returncode, json.loads(result.stdout)

    def test_missing_audio_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'silent.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=160x90:r=10:d=1', '-c:v', 'libx264', str(video)], check=True)
            code, report = self.check_video(video)
            self.assertEqual(code, 1)
            self.assertGreater(report['errors'], 0)
            code, report = self.check_video(video, '--allow-silent')
            self.assertEqual(code, 0)
            self.assertEqual(report['errors'], 0)

    def test_digital_silence_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'silence.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'color=c=blue:s=160x90:r=10:d=1', '-f', 'lavfi', '-i',
                'anullsrc=r=48000:cl=stereo', '-t', '1', '-c:v', 'libx264', '-c:a', 'aac', str(video)], check=True)
            self.assertEqual(self.check_video(video)[0], 1)
            self.assertEqual(self.check_video(video, '--allow-silent')[0], 0)

    def test_missing_video_still_fails_when_silence_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.check_video(Path(directory) / 'missing.mp4', '--allow-silent')
            self.assertEqual(code, 1)
            self.assertGreater(report['errors'], 0)
