import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class SyncCheckTests(unittest.TestCase):
    """A white square jumps across a black frame at 2.0 s; a short beep peaks 0.1 s into its file."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = Path(cls.dir.name)
        cls.video, cls.beep = d / 'jump.mp4', d / 'beep.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=320x180:r=30:d=4',
                        '-vf', "drawbox=x=40:y=60:w=60:h=60:color=white:t=fill:enable='lt(t,2)',"
                               "drawbox=x=220:y=60:w=60:h=60:color=white:t=fill:enable='gte(t,2)'",
                        '-c:v', 'libx264', str(cls.video)], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=f=880:d=0.2',
                        '-af', 'adelay=50|50,afade=t=out:st=0.15:d=0.1', str(cls.beep)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_check(self, *args):
        r = subprocess.run([sys.executable, str(ROOT / 'tools/sync_check.py'), *args, '--json'] if '--clip' not in args
                           else [sys.executable, str(ROOT / 'tools/sync_check.py'), *args],
                           capture_output=True, encoding='utf-8', timeout=60)
        return r.returncode, json.loads(r.stdout)

    def test_a_sound_half_a_second_early_is_out(self):
        code, res = self.run_check('--video', str(self.video), '--sfx', str(self.beep), '--at', '1.4',
                                   '--region', '320:180:0:0')
        self.assertEqual(code, 1)
        self.assertAlmostEqual(res[0]['offset'], 0.5, delta=0.1)

    def test_a_sound_on_the_movement_is_ok(self):
        code, res = self.run_check('--video', str(self.video), '--sfx', str(self.beep), '--at', '1.9',
                                   '--region', '320:180:0:0')
        self.assertEqual(code, 0)

    def test_clip_mode_says_where_to_start_the_sound(self):
        code, res = self.run_check('--clip', str(self.video), '--sfx', str(self.beep))
        self.assertEqual(code, 0)
        self.assertAlmostEqual(res['start_sound_after_clip_s'], 1.9, delta=0.1)

    def test_clip_mode_allows_for_a_trimmed_start(self):
        code, res = self.run_check('--clip', str(self.video), '--sfx', str(self.beep), '--trim', '15')
        self.assertEqual(code, 0)
        self.assertAlmostEqual(res['start_sound_after_clip_s'], 1.4, delta=0.1)


    def test_onset_finds_where_a_sustained_movement_starts(self):
        # a square that slides steadily from 1.5 s to 3.5 s: no single spike, so the sound goes on the start
        video = Path(self.dir.name) / 'slide.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=320x180:r=30:d=4',
                        '-f', 'lavfi', '-i', 'color=c=white:s=60x60:r=30:d=4', '-filter_complex',
                        "[0][1]overlay=x='40+clip(t-1.5,0,2)*100':y=60", '-c:v', 'libx264', str(video)], check=True)
        code, res = self.run_check('--clip', str(video), '--sfx', str(self.beep), '--kind', 'onset')
        self.assertAlmostEqual(res['movement_at'], 1.5, delta=0.15)

    def test_cue_sounds_resolve_from_the_cue_files_folder(self):
        cues = Path(self.dir.name) / 'cues.json'
        cues.write_text(json.dumps([{'name': 'beep', 'sfx': self.beep.name, 'at': 1.9, 'region': '320:180:0:0'}]))
        r = subprocess.run([sys.executable, str(ROOT / 'tools/sync_check.py'), '--video', str(self.video), '--cues',
                            str(cues), '--json'], capture_output=True, encoding='utf-8', timeout=60, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
