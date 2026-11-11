import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVING = 'mandelbrot=s=160x180:r=10'


def make(path: Path, left: str, seconds: int):
    """Two halves side by side: the right always moves, the left is `left` (a filter chain)."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y',
                    '-f', 'lavfi', '-i', left,
                    '-f', 'lavfi', '-i', f'{MOVING},trim=duration={seconds}',
                    '-f', 'lavfi', '-i', f'sine=f=440:d={seconds}',
                    '-filter_complex', '[0][1]hstack', '-t', str(seconds), '-c:v', 'libx264', '-c:a', 'aac',
                    str(path)], check=True)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MotionStopTests(unittest.TestCase):
    def motion_stops(self, video):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/qc.py'), '--render', str(video), '--json'],
                                capture_output=True, encoding='utf-8', timeout=120)
        report = json.loads(result.stdout)
        return [i for i in report['findings'] if i.get('kind') == 'motion_stop']

    def test_a_clip_that_runs_out_and_freezes_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / 'freeze.mp4'
            make(video, f'{MOVING},trim=duration=6,tpad=stop_mode=clone:stop_duration=8', 14)
            stops = self.motion_stops(video)
            self.assertEqual(len(stops), 1)
            self.assertAlmostEqual(stops[0]['start'], 6, delta=0.5)

    def test_motion_that_keeps_going_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / 'moving.mp4'
            make(video, f'{MOVING},trim=duration=14', 14)
            self.assertEqual(self.motion_stops(video), [])

    def test_a_region_that_never_moved_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / 'static.mp4'
            make(video, 'color=c=navy:s=160x180:r=10:d=14', 14)
            self.assertEqual(self.motion_stops(video), [])


if __name__ == '__main__':
    unittest.main()
