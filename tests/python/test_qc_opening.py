"""qc.py --render measures the opening of a short vertical video: movement, the first shot, the first sound."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIZE = '180x320'
MOVING = f'testsrc2=s={SIZE}:r=10'
OTHER = f'mandelbrot=s={SIZE}:r=10'


def make(path: Path, video: str, audio: str, seconds: float):
    """A video from a lavfi picture chain and a lavfi sound chain."""
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', video, '-f', 'lavfi', '-i', audio,
                    '-t', str(seconds), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path)], check=True)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class OpeningTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.video = Path(self.folder.name) / 'short.mp4'

    def tearDown(self):
        self.folder.cleanup()

    def qc(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/qc.py'), '--render', str(self.video), '--json'],
                                capture_output=True, encoding='utf-8', timeout=180)
        report = json.loads(result.stdout)
        return report.get('opening'), {i.get('kind') for i in report['findings'] if i['check'] == 'opening'}

    def test_a_short_that_moves_cuts_and_sounds_at_once_passes(self):
        # two different moving pictures, cut at 2 s, a tone throughout
        make(self.video, f'{MOVING},trim=duration=2[a];{OTHER},trim=duration=6[b];[a][b]concat', 'sine=f=440:d=8', 8)
        opening, kinds = self.qc()
        self.assertLessEqual(opening['moves_at'], 0.3)
        self.assertAlmostEqual(opening['first_cut'], 2.0, delta=0.2)
        self.assertEqual(opening['first_sound'], 0.0)
        self.assertEqual(kinds, {None})   # only the line of numbers

    def test_a_still_opening_is_flagged(self):
        make(self.video, f'color=gray:s={SIZE}:r=10:d=3[a];{MOVING},trim=duration=5[b];[a][b]concat', 'sine=f=440:d=8', 8)
        opening, kinds = self.qc()
        self.assertIn('opening_still', kinds)
        self.assertAlmostEqual(opening['moves_at'], 3.0, delta=0.2)

    def test_a_first_shot_that_never_changes_is_flagged(self):
        make(self.video, MOVING, 'sine=f=440:d=9', 9)
        opening, kinds = self.qc()
        self.assertIsNone(opening['first_cut'])
        self.assertIn('opening_long_shot', kinds)
        self.assertNotIn('opening_still', kinds)

    def test_a_silent_opening_is_flagged(self):
        make(self.video, MOVING, 'anullsrc=r=44100:cl=mono,atrim=duration=3[a];sine=f=440:d=5[b];[a][b]concat=v=0:a=1', 8)
        opening, kinds = self.qc()
        self.assertIn('opening_silent', kinds)
        self.assertAlmostEqual(opening['first_sound'], 3.0, delta=0.3)

    def test_an_opening_far_quieter_than_the_rest_is_flagged(self):
        make(self.video, MOVING, 'sine=f=440:d=2,volume=-20dB[a];sine=f=440:d=6[b];[a][b]concat=v=0:a=1', 8)
        opening, kinds = self.qc()
        self.assertIn('opening_quiet', kinds)
        self.assertLess(opening['level_vs_rest_db'], -10)

    def test_a_landscape_video_is_not_measured(self):
        make(self.video, 'testsrc2=s=320x180:r=10', 'sine=f=440:d=4', 4)
        opening, kinds = self.qc()
        self.assertIsNone(opening)
        self.assertEqual(kinds, set())


if __name__ == '__main__':
    unittest.main()
