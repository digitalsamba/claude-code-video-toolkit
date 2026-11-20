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

    def test_a_segment_that_opens_in_a_window_is_joined_on_a_look_at_the_lens(self):
        """When the next segment opens in a window, the carried move is a gesture to the hall, and both clips also
        look at the lens, the second look landing on the join's frame: the head was the one part left to each
        clip's own earlier beats, 72 mm apart at a join in the first lecture to do this."""
        self.lec['segments'][1]['windows'] = [{'in': "So here's", 'out': 'and back'}]
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        d = self.tmp / 'blender' / 'presenter'
        index = json.loads((d / 'index.json').read_text())
        join = index[1]['from']
        for clip in ('P01', 'P02'):
            spec = json.loads((d / f'{clip}.json').read_text())
            fps, m = spec['fps'], spec['master_start_frame']
            looks = [b for b in spec['beats'] if b['do'] == 'look_at' and b['target'] == 'camera'
                     and join - 40 < m + b['at'] * fps < join]
            self.assertEqual(round(m + (looks[-1]['at'] + looks[-1]['dur']) * fps), join, clip)
            self.assertIn('gesture', [b['do'] for b in spec['beats'] if abs(m + b['at'] * fps - join) < 30])

    def test_a_panes_video_is_measured_from_its_file(self):
        """A pane can play a video (public/): the timeline writes its length in frames, read from the file, so the
        board holds its last frame without a typed length; a missing file is named before anything renders."""
        clip = self.tmp / 'public' / 'clip.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=64x36:r=30', '-t', '2.5',
                        '-pix_fmt', 'yuv420p', str(clip)], check=True)
        self.lec['segments'][1]['board'] = [{'at': 'look closer', 'pane': {'x': 60, 'y': 300, 'w': 800, 'h': 500,
                                             'title': 'A clip', 'video': 'clip.mp4', 'out': 'and back'}}]
        self.write()
        tl = lecture.cmd_timeline(self.tmp, None)
        self.assertEqual(tl['videos'], {'clip.mp4': 75})
        self.lec['segments'][1]['board'][0]['pane']['video'] = 'gone.mp4'
        self.write()
        with self.assertRaises(SystemExit) as ctx:
            lecture.cmd_timeline(self.tmp, None)
        self.assertIn('gone.mp4', str(ctx.exception))

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

    def test_an_opening_is_the_first_clips_lead_in_drawn_in_the_hall(self):
        """With "opening", the lecture starts on the room shot: its spec is the first clip's (the same cast, beats
        and clock, so the pose at the cut is the same in both) with the hall drawn and dressed and a camera that
        lands on the board as the walk ends; the first clip has the opening as its lead-in, not rendered, and the
        first word comes half a second after the cut."""
        self.lec['opening'] = {}
        self.write()
        tl = lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        d = self.tmp / 'blender' / 'presenter'
        e1, p01 = json.loads((d / 'E1.json').read_text()), json.loads((d / 'P01.json').read_text())
        index = json.loads((d / 'index.json').read_text())
        fps, o = tl['fps'], lecture.DEFAULTS['opening']
        self.assertEqual(tl['roomFrames'], round(o['seconds'] * fps))
        self.assertEqual(tl['segments'][0]['startFrame'], round((o['seconds'] + o['first_word']) * fps))
        self.assertEqual([c['id'] for c in index], ['E1', 'P01', 'P02'])
        self.assertTrue(index[0]['room'])
        self.assertEqual(index[0]['frames'], index[1]['from'])
        self.assertEqual([s['id'] for s in p01['shots']], ['pre', 'P01'])
        self.assertEqual([s['id'] for s in e1['shots']], ['E1', 'tail'])
        self.assertEqual(e1['shots'][0]['dur'], p01['shots'][0]['dur'])          # the room shot is the lead-in
        self.assertEqual(e1['shots'][0]['to'], {'cam': 'board', 'lens': 108})
        self.assertEqual((e1['shots'][0]['start'], e1['shots'][0]['arrive']), ('last_walk', 'walk'))
        for k in ('beats', 'cast', 'clock', 'fps'):
            self.assertEqual(e1[k], p01[k], k)
        self.assertEqual((e1['set']['walls'], p01['set']['walls']), (True, False))
        self.assertEqual(e1['set']['hall'], {})
        self.assertIs(p01['set']['table'], False)                                 # (nothing where the walk goes)
        self.assertEqual(p01['cast'][0]['at'], {'door': o['behind']})
        verbs = [b['do'] for b in p01['beats']]
        self.assertIn('open_door', verbs)
        self.assertIn('walk', verbs)
        self.lec['opening'] = {'door': 0.5}                                      # before the picture is up
        self.write()
        with self.assertRaises(SystemExit) as ctx:
            lecture.cmd_timeline(self.tmp, None)
        self.assertIn('in that order', str(ctx.exception))

    def test_the_opening_switches_the_lights_on_or_keeps_the_lit_hall(self):
        """By default the hall starts dark: the first clip and the room shot both have the set's lights off, a
        step in to where the presenter's own arm presses the switch, the press with the door hand, and then the
        walk to the mark; no latch. "lights": "on" is the lit hall's opening as it was (its own times, the latch,
        one walk), and the order of the opening's times is checked for each."""
        self.lec['opening'] = {}
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        d = self.tmp / 'blender' / 'presenter'
        e1, p01 = json.loads((d / 'E1.json').read_text()), json.loads((d / 'P01.json').read_text())
        o = lecture.DEFAULTS['opening']
        self.assertEqual((e1['set']['lights'], p01['set']['lights']), ('off', 'off'))
        beats = [b for b in p01['beats'] if b['do'] in ('open_door', 'walk', 'lights', 'settle')]
        self.assertEqual([b['do'] for b in beats], ['open_door', 'walk', 'lights', 'walk', 'settle'])
        door, step, press, walk = beats[:4]
        self.assertNotIn('latch', door)                                          # the switch pulls you in now
        self.assertEqual(step['to'], {'switch': 'reach'})
        self.assertEqual((press['by'], press['hand']), ('presenter', door['hand']))  # the hand that opened the door
        self.assertEqual((press['at'], walk['at']), (o['switch'], o['walk'][0]))
        self.assertTrue(8.5 <= o['seconds'] <= 9.0)
        self.lec['opening'] = {'lights': 'on'}
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        e1, p01 = json.loads((d / 'E1.json').read_text()), json.loads((d / 'P01.json').read_text())
        self.assertNotIn('lights', p01['set'])
        self.assertEqual(e1['shots'][0]['start'], 'walk')
        verbs = [b['do'] for b in p01['beats']]
        self.assertNotIn('lights', verbs)
        self.assertEqual(verbs.count('walk'), 1)
        self.assertEqual(next(b for b in p01['beats'] if b['do'] == 'open_door')['latch'],
                         round(o['lit']['door'] - o['lit']['latch'], 3))
        for bad in ({'lights': 'dim'}, {'switch': 9.0}):
            self.lec['opening'] = bad
            self.write()
            with self.assertRaises(SystemExit) as ctx:
                lecture.cmd_timeline(self.tmp, None)
            self.assertIn('"lights"' if 'lights' in bad else 'in that order', str(ctx.exception))

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

    def test_a_close_is_its_clip_seen_by_a_second_camera_on_its_own_frames(self):
        """A segment's "closes" cut to a camera placed round the presenter: the clip's own spec (cast, beats, clock)
        with the hall drawn, rendered only on the close's frames, and never joined to its neighbours."""
        self.lec['segments'][1].pop('windows')
        self.lec['segments'][1]['closes'] = [{'in': 'one two three', 'out': 'end', 'type': 'closeup', 'az': 12}]
        self.write()
        tl = lecture.cmd_timeline(self.tmp, None)
        (c,) = tl['closes']
        seg = tl['segments'][1]
        self.assertEqual((c['id'], c['from'], c['to']), ('C02a', seg['startFrame'] + 117 - 3, seg['startFrame'] + seg['frames']))
        index = lecture.cmd_specs(self.tmp, None)
        entry = next(i for i in index if i['id'] == 'C02a')
        self.assertTrue(entry['close'])
        d = self.tmp / 'blender' / 'presenter'
        clip, close = (json.loads((d / f'{n}.json').read_text()) for n in ('P02', 'C02a'))
        self.assertEqual([s['id'] for s in close['shots']], ['pre', 'C02a', 'tail'])
        self.assertEqual([bool(s.get('built_only')) for s in close['shots']], [True, False, True])
        self.assertAlmostEqual(sum(s['dur'] for s in close['shots']), sum(s['dur'] for s in clip['shots']), places=3)
        self.assertEqual(entry['pre'], round(close['shots'][0]['dur'] * 30))
        shot = close['shots'][1]
        self.assertEqual((shot['type'], shot['subject'], shot['az']), ('closeup', 'presenter', 12))
        self.assertEqual((close['resolution'], close['transparent'], close['clock']), ('1920x1080', False, clip['clock']))
        self.assertTrue(close['set']['walls'])
        self.assertEqual(close['set']['board'], {'image': '../../public/board/C02a.png'})
        for b in clip['beats']:  # (every beat of the clip is in the close: the body is in the same pose)
            if not (b['do'] == 'look_at' and b['target'] != 'camera'):  # (but for looks out at the hall during it)
                self.assertIn(b, close['beats'])
        self.assertTrue(all(b['target'] == 'camera' for b in close['beats']
                            if b['do'] == 'look_at' and close['shots'][0]['dur'] <= b['at'] < close['shots'][0]['dur'] + shot['dur']))
        extra = [b for b in close['beats'] if b not in clip['beats']]
        self.assertEqual([(b['do'], b['target']) for b in extra], [('look_at', 'camera')])  # they find the new lens
        # taken out of lecture.json, its spec goes too
        self.lec['segments'][1].pop('closes')
        self.write()
        lecture.cmd_timeline(self.tmp, None)
        lecture.cmd_specs(self.tmp, None)
        self.assertFalse((d / 'C02a.json').exists())

    def test_a_close_that_cannot_be_there_is_refused_with_the_reason(self):
        for close, why in (({'in': 'look closer', 'out': 'the whole board'}, 'window'),
                           ({'in': "here's", 'out': 'short'}, 'heading'),
                           ({'in': 'the end', 'out': 'end'}, 'under 1.5 s'),
                           ({'in': 'the end', 'type': 'wide'}, 'closeup, medium')):
            self.lec['segments'][1]['closes'] = [close]
            self.write()
            with self.assertRaises(SystemExit) as e:
                lecture.cmd_timeline(self.tmp, None)
            self.assertIn(why, str(e.exception))

    def test_a_line_over_the_walk_in_is_said_by_the_presenter_and_ends_before_the_first_segment(self):
        scenes = self.tmp / 'public' / 'audio' / 'scenes'
        silence(scenes / '00_hook.mp3', 3.0)
        (scenes / '00_hook.captions.json').write_text(json.dumps(captions('Nobody animated this.')))
        self.lec['opening'] = {'voice': {'id': '00_hook', 'at': 2.0}}
        self.write()
        tl = lecture.cmd_timeline(self.tmp, None)
        v = tl['opening']['voice']
        self.assertEqual((v['id'], v['at'], v['startFrame']), ('00_hook', 2.0, 60))
        self.assertAlmostEqual(v['dur'], 3.0, delta=0.1)
        lecture.cmd_specs(self.tmp, None)
        d = self.tmp / 'blender' / 'presenter'
        for name in ('P01', 'E1'):  # (the room shot is the first clip's spec: they talk in both)
            spec = json.loads((d / f'{name}.json').read_text())
            self.assertEqual(spec['voice'][0], {'captions': '../../public/audio/scenes/00_hook.captions.json', 'at': 2.0})
            talks = [b for b in spec['beats'] if b['do'] == 'talk']
            self.assertEqual(talks[0]['at'], 2.0)
            self.assertEqual(len(talks), 2)
        self.lec['opening'] = {'voice': {'id': '00_hook', 'at': 7.5}}
        self.write()
        with self.assertRaises(SystemExit) as e:
            lecture.cmd_timeline(self.tmp, None)
        self.assertIn('first segment', str(e.exception))

if __name__ == '__main__':
    unittest.main()
