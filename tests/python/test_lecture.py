"""tools/lecture.py on the host (no Blender): the narration's words, the timeline, and the specs it writes.

    uv run python -m unittest tests/python/test_lecture.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import lecture  # noqa: E402


def silence(path: Path, seconds: float):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono', '-t', str(seconds),
                    '-q:a', '9', str(path)], check=True)


def captions(words: str, step_ms: int = 300) -> list:
    return [{'text': w, 'startMs': i * step_ms, 'endMs': i * step_ms + 250} for i, w in enumerate(words.split())]


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class LectureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        scenes = self.tmp / 'public' / 'audio' / 'scenes'
        scenes.mkdir(parents=True)
        lines = {'01_open': 'Hello there. One thing first, and then the board.',
                 '02_next': "So here's the second part, and it's short. Then we look closer at one, two, three, and back "
                            "out to the whole board for the end of it."}
        for i, (sid, text) in enumerate(lines.items()):
            silence(scenes / f'{sid}.mp3', 4.0 + 3 * i)
            (scenes / f'{sid}.captions.json').write_text(json.dumps(captions(text)))
        (self.tmp / 'cast').mkdir()
        (self.tmp / 'cast' / 'p.json').write_text(json.dumps({'id': 'p', 'model': 'p-fitted.blend'}))
        (self.tmp / 'cast' / 'p-fitted.blend').write_text('')
        self.lec = {'title': 'A test', 'presenter': {'cast': 'cast/p.json'}, 'segments': [
            {'id': '01_open', 'heading': 'Open', 'acting': [['Hello', 'gesture', 'open'], ['the board', 'show', [400, 500]]]},
            {'id': '02_next', 'heading': 'Next', 'acting': [['one two three', 'gesture', 'count'], ['short', 'look', 'happy']],
             'windows': [{'in': 'look closer', 'out': 'and back'}]}]}
        self.write()

    def write(self):
        (self.tmp / 'lecture.json').write_text(json.dumps(self.lec))

    def test_words_are_compared_as_the_board_compares_them(self):
        """The cue words match lib/components/cues.ts: lower case, punctuation gone, apostrophes kept inside."""
        self.assertEqual(lecture.norm("Here's the 'second' part \u2014 isn\u2019t it?"), ['here\'s', 'the', 'second', 'part', "isn't", 'it'])
        self.assertEqual(lecture.norm('Blender 4.2, or later'), ['blender', '4', '2', 'or', 'later'])

    def test_the_timeline_follows_the_audio_and_the_gaps(self):
        tl = lecture.cmd_timeline(self.tmp, None)
        a, b = tl['segments']
        self.assertEqual(a['startFrame'], 30)                               # a second before the first word
        self.assertAlmostEqual(b['start'], a['start'] + a['dur'] + 1.2, places=3)  # the rules' gap
        self.assertEqual(a['words'][0], {'w': 'hello', 'f': 30, 'e': 38})   # each word by frame, as the board reads it
        self.assertEqual(len(tl['windows']), 1)
        w = tl['windows'][0]
        self.assertLess(w['from'], w['to'])
        self.assertEqual([m['s'] for m in tl['moves']], [1.4, 1])          # the camera goes in and comes back out

    def test_a_join_is_crossed_by_one_move_keyed_alike_in_both_clips(self):
        """The second clip starts inside the turn to its heading that the first one began: both clips hold the same
        beat at the same master frame, and the second has a lead-in shot it is not rendered from."""
        lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        p01 = json.loads((self.tmp / 'blender' / 'presenter' / 'P01.json').read_text())
        p02 = json.loads((self.tmp / 'blender' / 'presenter' / 'P02.json').read_text())
        index = json.loads((self.tmp / 'blender' / 'presenter' / 'index.json').read_text())
        self.assertEqual([s['id'] for s in p02['shots']], ['pre', 'P02'])
        self.assertEqual([s['id'] for s in p01['shots']], ['P01'])
        fps = p01['fps']
        last = [b for b in p01['beats'] if b['do'] == 'present'][-1]
        first = [b for b in p02['beats'] if b['do'] == 'present'][0]
        self.assertEqual(round(p01['master_start_frame'] + last['at'] * fps), round(p02['master_start_frame'] + first['at'] * fps))
        self.assertEqual(last['target'], first['target'])
        self.assertEqual(last['dur'], first['dur'])
        self.assertEqual(index[1]['pre'], round(p02['shots'][0]['dur'] * fps))
        self.assertAlmostEqual(p02['clock'] * fps, p02['master_start_frame'], places=3)  # (breathing keeps the global clock)
        self.assertNotIn('look_at', [b['do'] for b in p02['beats'] if b['at'] < 0.5])  # (no look at the lens to snap to)

    def test_a_beat_that_would_collide_is_dropped_and_said(self):
        """A gesture on the first words, while the arm shows the heading, is left out, and the run says so; a turn
        to the board while the camera is in a window becomes a gesture to the hall."""
        self.lec['segments'][1]['acting'].append(['closer at', 'show', [300, 300]])
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        out = self._specs()
        self.assertIn("dropped the open gesture on 'Hello'", out)
        self.assertIn("'closer at' faces the hall", out)

    def test_a_cue_not_in_the_narration_names_what_is(self):
        self.lec['segments'][0]['acting'].append(['the blackboard', 'show', [100, 100]])
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        with self.assertRaises(SystemExit) as ctx:
            lecture.cmd_specs(self.tmp, None)
        self.assertIn('"the blackboard" is not in the narration', str(ctx.exception))
        self.assertIn('hello there one thing first', str(ctx.exception))

    def test_a_presenter_not_yet_fitted_says_how(self):
        (self.tmp / 'cast' / 'p-fitted.blend').unlink()
        lecture.cmd_timeline(self.tmp, None)
        with self.assertRaises(SystemExit) as ctx:
            lecture.cmd_specs(self.tmp, None)
        self.assertIn('blender_render.py fit', str(ctx.exception))

    def _specs(self) -> str:
        import contextlib
        import io
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            lecture.cmd_specs(self.tmp, None)
        return said.getvalue()


if __name__ == '__main__':
    unittest.main()
