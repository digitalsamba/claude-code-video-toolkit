"""Shot specs for tools/blender_render.py: the parts that run without Blender, and one check that needs it."""
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'lib' / 'blender'))
import blender_render
import sound
import cast
from timeline import SpecError, Timeline

SHOTS = [{'id': 'S1', 'dur': 2, 'type': 'wide'}, {'id': 'S2_face', 'dur': 3.5, 'type': 'closeup'}]


def have_blender():
    binary = next((c for c in blender_render.blender_candidates() if Path(c).exists()), None)
    return bool(binary) and blender_render.blender_version(binary)[0] >= blender_render.MIN_VERSION


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        words = [('The', 0, 200), ('disk', 250, 600), ('is ', 650, 800), ('full.', 850, 1300),
                 ('Full,', 2000, 2400), ('again', 2450, 2900)]
        (self.folder / 'vo.captions.json').write_text(json.dumps(
            [{'text': t, 'startMs': a, 'endMs': b} for t, a, b in words]))

    def test_shot_relative_times(self):
        tl = Timeline(SHOTS, 30)
        self.assertEqual(tl.time(1.25), 1.25)
        self.assertEqual(tl.time('S2_face'), 2.0)
        self.assertAlmostEqual(tl.time('S2_face+0.4'), 2.4)
        self.assertAlmostEqual(tl.time('S2_face.end-0.2'), 5.3)
        self.assertEqual(tl.time('end'), 5.5)
        self.assertEqual(tl.frame(0), 1)
        self.assertEqual(tl.frames - 1, 165)
        self.assertEqual(tl.shot_at(1.99), 'S1')
        self.assertEqual(tl.shot_at(2.0), 'S2_face')

    def test_a_cue_with_a_hyphen_and_a_number_is_the_words_when_they_are_said(self):
        (self.folder / 'n.captions.json').write_text(json.dumps(
            [{'text': t, 'startMs': a, 'endMs': a + 300} for t, a in (('The', 0), ('top-10', 12000), ('list', 12500))]))
        tl = Timeline([{'id': 'S1', 'dur': 20, 'type': 'wide'}], 30, [{'captions': 'n.captions.json'}], self.folder)
        self.assertAlmostEqual(tl.time('cue:top-10'), 12.0)        # the words "top 10", not "top" minus ten seconds
        self.assertAlmostEqual(tl.time('cue:list-10'), 2.5)        # "list 10" is never said: an offset

    def test_bad_references_name_the_problem(self):
        tl = Timeline(SHOTS, 24)
        for ref in ('S9+1', 'S1 plus 2', 'cue:disk'):
            with self.assertRaises(SpecError):
                tl.time(ref)
        with self.assertRaises(SpecError):
            Timeline([{'id': 'S1', 'dur': 1}], 24)
        with self.assertRaises(SpecError):
            Timeline(SHOTS + [SHOTS[0]], 24)
        for dur in (0, -1, 'x', 0.01):   # shorter than a frame, or not a number
            with self.assertRaises(SpecError):
                Timeline([{'id': 'S1', 'dur': dur, 'type': 'wide'}], 24)

    def test_a_shot_id_may_hold_a_hyphen_and_a_number(self):
        tl = Timeline([{'id': 'take', 'dur': 2, 'type': 'wide'}, {'id': 'take-2', 'dur': 3, 'type': 'wide'}], 30)
        self.assertEqual(tl.time('take-2'), 2.0)          # the shot, not "take" minus two seconds
        self.assertEqual(tl.time('take-2.end'), 5.0)
        self.assertAlmostEqual(tl.time('take-2+0.5'), 2.5)
        self.assertAlmostEqual(tl.time('take.end-0.5'), 1.5)

    def test_cues_come_from_the_narration(self):
        voice = [{'captions': 'vo.captions.json', 'at': 'S1+0.5'}]
        tl = Timeline(SHOTS, 30, voice, self.folder)
        self.assertAlmostEqual(tl.time('cue:disk'), 0.75)
        self.assertAlmostEqual(tl.time('cue:disk is full'), 0.75)
        self.assertAlmostEqual(tl.time('cue:Full'), 1.35)          # case and punctuation don't matter
        self.assertAlmostEqual(tl.time('cue:disk-0.25'), 0.5)      # an offset, when the words aren't "disk 0 25"
        self.assertAlmostEqual(tl.time('cue:full#2'), 2.5)         # the second time it is said
        self.assertAlmostEqual(tl.time('cue:full#2-0.1'), 2.4)
        self.assertAlmostEqual(tl.time('cue:again+0.25'), 3.2)
        with self.assertRaises(SpecError) as ctx:
            tl.time('cue:empty')
        self.assertIn('the disk is full', str(ctx.exception))      # the error says what the narration does say
        with self.assertRaises(SpecError):
            Timeline(SHOTS, 30, [{'captions': 'missing.json'}], self.folder)


class CastTests(unittest.TestCase):
    """Character descriptions (lib/blender/cast.py): no Blender needed."""

    def test_nothing_given_is_the_character_the_examples_use(self):
        look = cast.resolve({'id': 'dev'})
        self.assertEqual(look['hair'], {'style': 'tufts', 'color': '#3A2513'})
        self.assertEqual(look['clothes']['top'], 'hoodie')
        self.assertEqual(look['accessories'], [])
        old = cast.resolve({'id': 'dev', 'colors': {'top': '#3771E0', 'hair': '#2B211B'}})   # the first format
        self.assertEqual((old['clothes']['top_color'], old['hair']['color']), ('#3771E0', '#2B211B'))

    def test_a_seed_is_a_person_and_the_same_person_every_time(self):
        self.assertEqual(cast.resolve({'id': 'a', 'seed': 41}), cast.resolve({'id': 'a', 'seed': 41}))
        looks = [json.dumps(cast.resolve({'id': 'a', 'seed': n}), sort_keys=True) for n in range(40, 50)]
        self.assertGreaterEqual(len(set(looks)), 9)
        kept = cast.resolve({'id': 'a', 'seed': 41, 'hair': {'style': 'bob'}})                # what is given is kept
        self.assertEqual(kept['hair']['style'], 'bob')

    def test_names_and_hex_colours(self):
        look = cast.resolve({'id': 'a', 'body': {'skin': 'brown'}, 'hair': {'color': 'grey'}, 'face': {'eyes': '#112233'}})
        self.assertEqual((look['body']['skin'], look['hair']['color'], look['face']['eyes']),
                         (cast.SKIN['brown'], cast.HAIR_COLORS['grey'], '#112233'))

    def test_a_wrong_field_names_the_choices(self):
        for entry, said in (({'hair': {'style': 'mullet'}}, 'bob'), ({'body': {'skin': 'green'}}, 'hex colour'),
                            ({'accessories': ['monocle']}, 'glasses'), ({'face': {'chin': 'big'}}, 'chin'),
                            ({'from': 'nobody.json'}, 'not found')):
            with self.assertRaises(SpecError) as ctx:
                cast.resolve({'id': 'a', **entry}, Path('.'))
            self.assertIn(said, str(ctx.exception))

    def test_a_stored_character_can_be_recostumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            stored = {'id': 'maya', 'hair': {'style': 'bob', 'color': 'black'}, 'clothes': {'top': 'shirt'},
                      'accessories': ['glasses']}
            (Path(tmp) / 'maya.json').write_text(json.dumps(stored))
            look = cast.resolve({'id': 'lead', 'from': 'maya.json', 'clothes': {'top_color': '#B23A48'}}, Path(tmp))
            self.assertEqual(look['id'], 'lead')
            self.assertEqual(look['hair']['style'], 'bob')
            self.assertEqual((look['clothes']['top'], look['clothes']['top_color']), ('shirt', '#B23A48'))
            self.assertEqual(look['accessories'], ['glasses'])

    def test_the_kit_example_character_resolves(self):
        cast.resolve({'id': 'maya', 'from': str(ROOT / 'kits' / 'example' / 'cast' / 'maya.json')})


class HostSideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_exposure_ignores_transparent_pixels(self):
        from PIL import Image
        im = Image.new('RGBA', (240, 135), (0, 0, 0, 0))
        im.paste((128, 128, 128, 255), (0, 0, 60, 135))            # a quarter of the frame, mid grey
        im.save(self.folder / 'a.png')
        mean, clipped = blender_render.exposure(self.folder / 'a.png')
        self.assertAlmostEqual(mean, 128 / 255, places=2)
        self.assertEqual(clipped, 0)

    def test_clip_sound_is_cut_to_the_shot(self):
        events = {'fps': 30, 'duration': 6.0, 'score': [{'from': 0, 'to': 6, 'mood': 'calm'}],
                  'events': [{'t': 0.5, 'frame': 16, 'kind': 'alert'}, {'t': 4.2, 'frame': 127, 'kind': 'tap'}]}
        (self.folder / 'events.json').write_text(json.dumps(events))
        out = self.folder / 'clip.wav'
        sound.render(self.folder / 'events.json', out, start=4.0, end=5.5, score=False, room=False)
        with wave.open(str(out)) as w:
            n, rate = w.getnframes(), w.getframerate()
            data = w.readframes(n)
        self.assertAlmostEqual(n / rate, 1.5, places=2)
        samples = [int.from_bytes(data[i:i + 2], 'little', signed=True) for i in range(0, len(data), 2)]
        first = next(i for i, s in enumerate(samples) if abs(s) > 200)
        self.assertAlmostEqual(first / rate, 0.2, delta=0.03)      # the tap, 0.2 s into the clip; no alert, no bed

    def test_check_builds_at_the_films_own_shape(self):
        spec = self.folder / 'spec.json'
        for resolution, override, want in (('1920x1080', None, '960x540'), ('1080x1920', None, '540x960'),
                                           ('1080x1080', None, '960x960'), ('1920x1080', '1080x1920', '540x960')):
            spec.write_text(json.dumps({'resolution': resolution}))
            self.assertEqual(blender_render.still_size(spec, override), want)
        spec.write_text(json.dumps({}))
        self.assertEqual(blender_render.still_size(spec, None), '960x540')

    def test_a_white_picture_on_a_screen_is_not_blown_out(self):
        shows = blender_render.shows_a_picture
        self.assertFalse(shows({'cast': [{'id': 'dev'}]}))
        self.assertFalse(shows({'set': {'kind': 'studio', 'props': [{'id': 'p', 'kind': 'phone'}]}}))   # accent colour
        self.assertTrue(shows({'set': {'kind': 'studio', 'props': [{'id': 'c', 'kind': 'card', 'screen': {'image': 'a.png'}}]}}))
        self.assertTrue(shows({'set': {'kind': 'desk_room', 'screen': {'video': 'a.mp4'}}}))
        self.assertFalse(shows({'set': {'kind': 'desk_room', 'screen': {'title': 'Storage'}}}))         # the drawn UI

    def test_shot_lists_ignore_spaces(self):
        self.assertEqual(blender_render.shot_ids(' A, B ,'), ['A', 'B'])
        self.assertEqual(blender_render.shot_ids(None), [])

    def test_blender_set_to_a_folder_is_not_a_blender(self):
        binary = self.folder / 'blender'
        binary.write_text('')
        with mock.patch.dict(os.environ, {'BLENDER': str(binary)}):
            self.assertEqual(blender_render.installed_blender(), str(binary))   # BLENDER wins when it is a file
        with mock.patch.dict(os.environ, {'BLENDER': str(self.folder)}):
            self.assertNotEqual(blender_render.installed_blender(), str(self.folder))

    def test_specs_sharing_a_manifest_keep_their_own_clips(self):
        path = self.folder / 'clips.json'

        def clip(i, fps, spec):
            return {'id': i, 'file': f'{i}.mp4', 'fps': fps, 'resolution': [1920, 1080], 'spec': spec}

        def ids():
            return sorted(c['id'] for c in json.loads(path.read_text())['clips'])

        blender_render.merge_clips(path, [clip('A1', 30, 'b/room.json'), clip('A2', 30, 'b/room.json')], 'b/room.json', True)
        blender_render.merge_clips(path, [clip('B1', 24, 'b/desk.json')], 'b/desk.json', True)
        got = {c['id']: c['fps'] for c in json.loads(path.read_text())['clips']}
        self.assertEqual(got, {'A1': 30, 'A2': 30, 'B1': 24})                                     # each its own rate
        blender_render.merge_clips(path, [clip('A2', 30, 'b/room.json')], 'b/room.json', False)   # --shots A2
        self.assertEqual(ids(), ['A1', 'A2', 'B1'])
        blender_render.merge_clips(path, [clip('A9', 30, 'b/room.json')], 'b/room.json', True)    # A1, A2 renamed away
        self.assertEqual(ids(), ['A9', 'B1'])
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_a_manifest_from_before_clips_carried_their_rate_is_read(self):
        path = self.folder / 'clips.json'
        path.write_text(json.dumps({'fps': 30, 'resolution': [1920, 1080], 'clips': [{'id': 'old', 'file': 'old.mp4'}]}))
        blender_render.merge_clips(path, [{'id': 'new', 'file': 'new.mp4', 'fps': 24, 'resolution': [1080, 1920],
                                           'spec': 'x/spec.json'}], 'x/spec.json', True)
        got = {c['id']: c for c in json.loads(path.read_text())['clips']}
        self.assertEqual((got['old']['fps'], got['old']['resolution']), (30, [1920, 1080]))
        self.assertEqual(got['new']['fps'], 24)


@unittest.skipUnless(have_blender() and shutil.which('ffmpeg'), 'Blender and FFmpeg required')
class BlenderCheckTests(unittest.TestCase):
    def test_example_spec_passes_its_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = blender_render.run_compiler(ROOT / 'examples' / 'blender-disk-panic' / 'spec.json',
                                                 Path(tmp), 'check', '480x270', None)
            self.assertTrue(checks['passed'], checks)
            self.assertEqual(len(checks['shots']), 8)
            self.assertTrue((Path(tmp) / 'stills' / 'S3_face.png').exists())

    def test_the_product_shot_example_passes_its_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = blender_render.run_compiler(ROOT / 'examples' / 'blender-product-shot' / 'spec.json',
                                                 Path(tmp), 'check', '480x270', None)
            self.assertTrue(checks['passed'], checks)
            self.assertEqual([s['type'] for s in checks['shots']], ['reveal', 'orbit', 'product', 'orbit'])

    def test_unknown_verb_is_a_spec_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = json.loads((ROOT / 'examples' / 'blender-disk-panic' / 'spec.json').read_text())
            spec['beats'].append({'at': 'S1_wide+1', 'do': 'moonwalk'})
            path = Path(tmp) / 'spec.json'
            path.write_text(json.dumps(spec))
            with self.assertRaises(RuntimeError) as ctx:
                blender_render.run_compiler(path, Path(tmp) / 'out', 'check', '480x270', None)
            self.assertIn('moonwalk', str(ctx.exception))

    # ---- small specs written for one question each
    def compile(self, spec, files=None, res='480x270'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        for name, content in (files or {}).items():
            (folder / name).write_text(json.dumps(content))
        (folder / 'spec.json').write_text(json.dumps({'cast': [{'id': 'dev'}], **spec}))
        checks = blender_render.run_compiler(folder / 'spec.json', folder / 'out', 'check', res, None)
        return checks, json.loads((folder / 'out' / 'events.json').read_text())

    def test_mistakes_in_a_spec_are_named(self):
        shots = [{'id': 'A', 'dur': 1, 'type': 'wide'}]
        for beats, shot, said in (
                ([{'at': 'A', 'do': 'expression', 'face': 'smug'}], shots, 'shocked'),       # lists the faces it has
                ([{'at': 'A', 'do': 'pose', 'pose': 'dab'}], shots, 'hands_on_head'),
                ([{'at': 'A', 'do': 'blink', 'who': 'sam'}], shots, 'sam'),
                ([{'do': 'blink'}], shots, '"at"'),
                ([{'at': 'A', 'do': 'look', 'target': 'left'}], shots, 'degrees'),
                ([], [{'id': 'A', 'dur': 1, 'type': 'wide', 'subject': 'cat'}], 'cat'),
                ([], [{'id': 'A', 'dur': 1, 'type': 'pull_out'}], 'screen'),
                ([{'at': 'A', 'do': 'blink', 'dur': 'long'}], shots, 'number of seconds'),
                ([], [{'id': 'A', 'dur': 'x', 'type': 'wide'}], 'at least one frame')):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile({'shots': shot, 'beats': beats})
            self.assertIn(said, str(ctx.exception))
            self.assertNotIn('Traceback', str(ctx.exception))
        for extra, said in (({'score': [{'from': 0, 'to': 'nope', 'mood': 'calm'}]}, 'nope'),
                            ({'score': [{'from': 0, 'to': 'end', 'mood': 'jazz'}]}, 'tension'),
                            ({'fps': 29.97}, 'whole number'),
                            ({'resolution': '1920X1080'}, 'WIDTHxHEIGHT')):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile({'shots': shots, **extra})
            self.assertIn(said, str(ctx.exception))
            self.assertNotIn('Traceback', str(ctx.exception))

    def test_acting_takes_the_same_time_at_any_frame_rate(self):
        spec = {'shots': [{'id': 'A', 'dur': 4, 'type': 'wide'}],
                'beats': [{'at': 0.5, 'do': 'type', 'until': 2.5}, {'at': 3, 'do': 'tap'},
                          {'at': 2.6, 'do': 'shake_head', 'until': 3.8}]}
        by_rate = {}
        for fps in (24, 30, 60):
            _, events = self.compile({**spec, 'fps': fps})
            by_rate[fps] = events['events']
        for fps in (30, 60):
            for kind in ('key_click', 'whoosh', 'tap'):
                a = [e['t'] for e in by_rate[24] if e['kind'] == kind]
                b = [e['t'] for e in by_rate[fps] if e['kind'] == kind]
                self.assertEqual(len(a), len(b), f'{kind} at {fps} fps')
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, delta=0.09)

    def test_a_beat_lands_on_the_narration_word(self):
        captions = [{'text': t, 'startMs': a, 'endMs': a + 200} for t, a in
                    (('The', 0), ('disk', 300), ('is', 600), ('full.', 900))]
        checks, events = self.compile(
            {'fps': 30, 'shots': [{'id': 'A', 'dur': 3, 'type': 'over_shoulder', 'subject': 'screen'}],
             'voice': [{'captions': 'vo.captions.json', 'at': 'A+0.5'}],
             'beats': [{'at': 'cue:full', 'do': 'alert'}, {'at': 'cue:disk-0.1', 'do': 'gulp'}]},
            {'vo.captions.json': captions})
        when = {e['kind']: e['t'] for e in events['events']}
        self.assertAlmostEqual(when['alert'], 1.4, delta=0.02)
        self.assertAlmostEqual(when['gulp'], 0.7, delta=0.02)

    def test_moves_pass_in_every_frame_shape(self):
        for res, small in (('1920x1080', '480x270'), ('1080x1920', '270x480'), ('1080x1080', '360x360')):
            checks, _ = self.compile(
                {'resolution': res, 'set': {'kind': 'desk_room'},
                 'shots': [{'id': 'out', 'dur': 3, 'type': 'pull_out', 'subject': 'screen'},
                           {'id': 'face', 'dur': 1, 'type': 'closeup'},
                           {'id': 'in', 'dur': 3, 'type': 'push_in', 'subject': 'screen'}]}, res=small)
            self.assertTrue(checks['passed'], (res, checks['shots']))
            self.assertGreater(checks['shots'][0]['fill_distance_m'], 0.15)

    def test_a_blocked_camera_is_moved_and_says_why(self):
        # level with the desk and straight behind the chair, the chair's back hides the screen: the solver moves it
        checks, _ = self.compile({'shots': [{'id': 'A', 'dur': 2, 'type': 'over_shoulder', 'subject': 'screen',
                                             'az': 0, 'h': 0.0, 'dist': 1.4}]})
        shot = checks['shots'][0]
        self.assertTrue(shot['adjusted'])
        self.assertIn('view blocked by chair_back', ' '.join(shot['nominal_issues']))
        self.assertTrue(shot['passed'])
        self.assertTrue(blender_render.describe_fix(shot))   # says what it did: swung, rose or pulled back

    def test_every_described_character_builds_and_acts(self):
        beats = [{'at': 0.2, 'do': 'pose', 'pose': 'hands_on_head', 'dur': 0.4},
                 {'at': 0.3, 'do': 'expression', 'face': 'shocked'}]
        people = [{'id': 'a', 'body': {'build': 'broad'}, 'face': {'shape': 'square', 'beard': 'full', 'eyes': 'blue'},
                   'hair': {'style': 'short'}, 'clothes': {'top': 'tee'}},
                  {'id': 'b', 'body': {'build': 'slim', 'skin': 'deep'}, 'face': {'shape': 'oval', 'nose': 'long'},
                   'hair': {'style': 'curly'}, 'clothes': {'top': 'shirt'}, 'accessories': ['glasses', 'headphones', 'earrings']},
                  {'id': 'c', 'hair': {'style': 'long'}, 'face': {'beard': 'moustache', 'brows': 'thick'}},
                  {'id': 'd', 'hair': {'style': 'bald'}}, {'id': 'e', 'hair': {'style': 'bun'}}, {'id': 'f', 'seed': 5}]
        for person in people:
            checks, _ = self.compile({'cast': [person], 'beats': beats,
                                      'shots': [{'id': 'A', 'dur': 1, 'type': 'medium'},
                                                {'id': 'B', 'dur': 1, 'type': 'closeup'}]})
            self.assertTrue(checks['passed'], (person, checks['shots']))
            self.assertLess(checks['ik_max_error_m'], 0.01)

    # ---- the studio set: props, no cast
    CUBE = 'v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nv -1 -1 3\nv 1 -1 3\nv 1 1 3\nv -1 1 3\n' \
           'f 1 2 3 4\nf 5 8 7 6\nf 1 5 6 2\nf 2 6 7 3\nf 3 7 8 4\nf 5 1 4 8\n'

    def studio(self, shots, props=None, res='480x270', resolution='1920x1080'):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        (folder / 'tower.obj').write_text(self.CUBE)
        spec = {'fps': 30, 'resolution': resolution, 'shots': shots,
                'set': {'kind': 'studio', 'props': props or [
                    {'id': 'app', 'kind': 'phone'}, {'id': 'ui', 'kind': 'card', 'at': [0.5, 0]},
                    {'id': 'tower', 'kind': 'model', 'file': 'tower.obj', 'size': 0.3, 'at': [-0.4, 0]}]}}
        (folder / 'spec.json').write_text(json.dumps(spec))
        return blender_render.run_compiler(folder / 'spec.json', folder / 'out', 'check', res, None)

    def test_a_studio_frames_each_prop_from_its_size(self):
        checks = self.studio([{'id': 'phone', 'dur': 2, 'type': 'orbit', 'subject': 'app', 'az': -30, 'az_to': 30},
                              {'id': 'model', 'dur': 1, 'type': 'product', 'subject': 'tower'},
                              {'id': 'reveal', 'dur': 3, 'type': 'reveal', 'subject': 'ui'}])
        self.assertTrue(checks['passed'], checks['shots'])
        phone, model, reveal = checks['shots']
        self.assertLess(phone['distance_m'], model['distance_m'])       # a 15 cm phone is filmed closer than a 30 cm model
        self.assertEqual(phone['screen_readable'], 1.0)
        self.assertNotIn('screen_readable', model)
        self.assertGreater(reveal['fill_distance_m'], 0.1)

    def test_a_studio_works_in_portrait(self):
        checks = self.studio([{'id': 'phone', 'dur': 2, 'type': 'orbit', 'subject': 'app', 'az': -25, 'az_to': 25},
                              {'id': 'reveal', 'dur': 3, 'type': 'reveal', 'subject': 'app'}],
                             props=[{'id': 'app', 'kind': 'phone'}], res='270x480', resolution='1080x1920')
        self.assertTrue(checks['passed'], checks['shots'])

    def test_an_orbit_that_turns_the_screen_away_fails(self):
        checks = self.studio([{'id': 'spin', 'dur': 3, 'type': 'orbit', 'subject': 'app', 'az': -170, 'az_to': 170}],
                             props=[{'id': 'app', 'kind': 'phone'}])
        self.assertFalse(checks['passed'])
        self.assertIn('faces the lens', ' '.join(checks['shots'][0]['issues']))

    def test_studio_mistakes_are_named(self):
        for shots, props, said in (
                ([{'id': 'A', 'dur': 1, 'type': 'closeup'}], None, 'desk set'),
                ([{'id': 'A', 'dur': 1, 'type': 'product', 'subject': 'cat'}], None, 'cat'),
                ([{'id': 'A', 'dur': 1, 'type': 'reveal', 'subject': 'tower'}], None, 'screen'),
                ([{'id': 'A', 'dur': 1, 'type': 'product'}], [{'id': 'm', 'kind': 'model', 'file': 'gone.glb'}], 'not found'),
                ([{'id': 'A', 'dur': 1, 'type': 'product'}], [{'id': 'm', 'kind': 'hologram'}], 'phone')):
            with self.assertRaises(RuntimeError) as ctx:
                self.studio(shots, props)
            self.assertIn(said, str(ctx.exception))
            self.assertNotIn('Traceback', str(ctx.exception))

    def test_a_screen_video_at_another_frame_rate_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            import subprocess
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=25:d=1',
                            '-pix_fmt', 'yuv420p', str(Path(tmp) / 'v.mp4')], check=True)
            spec = {'fps': 30, 'cast': [{'id': 'dev'}], 'set': {'screen': {'video': 'v.mp4'}},
                    'shots': [{'id': 'A', 'dur': 1, 'type': 'screen_fill', 'subject': 'screen'}]}
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            with self.assertRaises(RuntimeError) as ctx:
                blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '480x270', None)
            self.assertIn('25 fps', str(ctx.exception))
            # the same on a studio prop
            spec = {'fps': 30, 'set': {'kind': 'studio', 'props': [{'id': 'ui', 'kind': 'card', 'screen': {'video': 'v.mp4'}}]},
                    'shots': [{'id': 'A', 'dur': 1, 'type': 'product', 'subject': 'ui'}]}
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            with self.assertRaises(RuntimeError) as ctx:
                blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '480x270', None)
            self.assertIn('25 fps', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
