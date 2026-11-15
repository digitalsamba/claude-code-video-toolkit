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


class PaintedFaceTests(unittest.TestCase):
    """Reading a model's painted face (lib/blender/facepaint.py): plain numpy, no Blender needed."""

    def painted_eye(self, wide=0.032, tall=0.018, iris=0.0095, covered=0.15):
        """A patch of a face with an eye painted on it, 0.2 mm a sample: skin, a white almond between two lid
        lines, a lash line over it, a brown iris with a dark pupil whose top the upper lid covers."""
        import numpy as np
        step, h, w = 0.0002, 225, 320
        zz, xx = np.mgrid[0:h, 0:w]
        x, z = (xx - w / 2) * step, (zz - h / 2) * step
        P = np.empty((h, w, 3), dtype=np.float32)
        P[:] = (0.94, 0.72, 0.63)
        edge = tall / 2 * (1 - (x / (wide / 2)) ** 2)                       # the lid lines: two arcs, corner to corner
        opening = (np.abs(x) < wide / 2) & (np.abs(z) <= edge)
        P[(np.abs(x) < wide / 2) & (z > edge) & (z < edge + 0.0016)] = (0.30, 0.18, 0.14)   # the lash line
        P[opening] = (0.86, 0.84, 0.83)
        centre = tall / 2 + 2 * iris * covered - iris                        # the lid over `covered` of the iris
        r = np.hypot(x, z - centre)
        P[opening & (r < iris)] = (0.32, 0.21, 0.15)
        P[opening & (r < 0.45 * iris)] = (0.04, 0.03, 0.03)
        return P

    def test_a_painted_eye_is_read_for_its_size_its_iris_and_its_lid(self):
        import facepaint
        eye = facepaint.measure_eye(self.painted_eye())
        self.assertIsNotNone(eye)
        self.assertAlmostEqual(eye['open'][0], 0.032, delta=0.002)
        self.assertAlmostEqual(eye['open'][1], 0.018, delta=0.002)
        self.assertAlmostEqual(eye['iris_r'], 0.0095, delta=0.001)
        self.assertAlmostEqual(eye['covered'], 0.15, delta=0.07)             # the lid rests over the top of the iris
        self.assertLess(eye['pupil_r'], 0.7 * eye['iris_r'])
        self.assertGreater(eye['iris_colour'][0], eye['iris_colour'][2])     # brown, not the pupil's black
        for plate in ('fill', 'iris', 'data'):                               # what a kept face's eye is cut from
            self.assertEqual(eye['plates'][plate].ndim, 3)
        import numpy as np
        blank = np.empty((225, 320, 3), dtype=np.float32)
        blank[:] = (0.94, 0.72, 0.63)
        self.assertIsNone(facepaint.measure_eye(blank))                      # bare skin is not an eye

    def test_eyes_twice_as_tall_as_the_painted_ones_with_no_lid_are_a_stare(self):
        """The face that was sent back: the rig's white ovals were 30 mm tall over painted eyes about half that, wide
        open where the paint has a lid over the iris. A cartoon face whose rig eyes match its paint passes, and
        so does the same face once its paint is kept."""
        import facepaint
        eye = {k: v for k, v in facepaint.measure_eye(self.painted_eye()).items() if k != 'plates'}
        seen = {'eyes': [eye, eye]}
        stared = facepaint.stare(seen, [0.04, 0.03], kept=False)
        self.assertGreater(stared['eye_tall'], 1.5)
        self.assertIn('stares', ' '.join(stared['problems']))
        self.assertIn('"paint": "keep"', ' '.join(stared['problems']))
        self.assertNotIn('problems', facepaint.stare(seen, [0.04, 0.03], kept=True))
        self.assertNotIn('problems', facepaint.stare(seen, [0.034, 0.02], kept=False))
        self.assertEqual(facepaint.stare({'eyes': [None, None]}, [0.04, 0.03], kept=False), {})


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

    def test_a_character_comes_in_walks_sits_down_and_winks(self):
        """The clay cast on its feet: through a door, across the room, onto a pulled-out chair, in to the desk."""
        checks, events = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 3.7]}],
            'set': {'kind': 'desk_room', 'door': {}, 'chair_out': 0.8},
            'shots': [{'id': 'feet', 'dur': 3, 'type': 'feet', 'reveal': True},
                      {'id': 'up', 'dur': 2, 'type': 'tilt_up'},
                      {'id': 'walk', 'dur': 5, 'type': 'full', 'follow': True},
                      {'id': 'sit', 'dur': 4, 'type': 'wide'},
                      {'id': 'face', 'dur': 2, 'type': 'closeup'}],
            'beats': [{'at': 'feet+0.2', 'do': 'open_door', 'dur': 0.7},
                      {'at': 'feet+0.9', 'do': 'walk', 'to': [2.6, 2.0], 'until': 'feet.end'},
                      {'at': 'walk+0.1', 'do': 'walk', 'to': [1.1, 0.33], 'until': 'walk+2.5'},
                      {'at': 'walk+2.6', 'do': 'walk', 'to': [0, 0.33], 'facing': 0, 'until': 'walk+4.8'},
                      {'at': 'sit+0.1', 'do': 'sit_down', 'dur': 1.2},
                      {'at': 'sit+1.5', 'do': 'pull_in', 'dur': 0.9},
                      {'at': 'sit+2.6', 'do': 'pose', 'pose': 'type', 'dur': 0.5},
                      {'at': 'face+0.2', 'do': 'look_at', 'target': 'camera', 'dur': 0.4},
                      {'at': 'face+1', 'do': 'wink', 'side': 'right'}]})
        self.assertTrue(checks['passed'], checks)
        self.assertLess(checks['ik_max_error_m'], 0.01)  # every step was one the legs could make
        self.assertLessEqual(checks['under_floor_m'], 0.015)  # and no foot went through the floor on the way
        kinds = [e['kind'] for e in events['events']]
        self.assertIn('door_open', kinds)
        self.assertGreater(kinds.count('step'), 8)
        # contact: no hand inside the body, nothing passed through the furniture, no planted foot slid
        for kind in ('hand_in_body_m', 'through_m', 'foot_slide_m', 'touch_miss_m'):
            self.assertLessEqual(checks[kind], 0.03, kind)
        # nothing jumps in a single frame: the chest comes to rest after its last sway, a hand travels from one
        # spot to the next (in a move this short the keys used to land out of order)
        self.assertLessEqual(checks['snap_m'], 0.02)
        self.assertNotIn('body_problems', checks)

    def test_a_body_does_not_move_like_a_machine(self):
        """Standing up, turning, sitting down and pulling in: the parts of the body do not all start and stop
        on one frame, and the left side does not mirror the right."""
        checks, _ = self.compile({
            'cast': [{'id': 'dev', 'posture': 'sit', 'at': [0, 0.8]}],
            'set': {'kind': 'desk_room', 'chair_out': 0.8},
            'shots': [{'id': 'A', 'dur': 11, 'type': 'wide'}],
            'beats': [{'at': 'A+0.3', 'do': 'stand_up', 'dur': 1.7},
                      {'at': 'A+2.4', 'do': 'turn', 'facing': 110, 'dur': 1.3},
                      {'at': 'A+4.0', 'do': 'turn', 'facing': 0, 'dur': 1.3},
                      {'at': 'A+5.8', 'do': 'sit_down', 'dur': 1.8},
                      {'at': 'A+8.0', 'do': 'pull_in', 'dur': 1.5}]})
        self.assertTrue(checks['passed'], checks)
        self.assertEqual([m['do'] for m in checks['moves']], ['stand_up', 'turn', 'turn', 'sit_down', 'pull_in'])
        for move in checks['moves']:
            self.assertGreaterEqual(move['starts_spread_s'] + move['stops_spread_s'], 0.2, move)  # something leads, something settles
            self.assertGreater(move['sides_differ_m'], 0.02, move)

    def test_a_hand_opens_the_door_and_the_check_says_if_it_cannot(self):
        shots = [{'id': 'A', 'dur': 4, 'type': 'full', 'reveal': True}]
        door = {'kind': 'desk_room', 'door': {}}
        checks, events = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 3.5]}], 'set': door, 'shots': shots,
            'beats': [{'at': 'A+0.5', 'do': 'open_door', 'dur': 0.8, 'by': 'dev'},
                      {'at': 'A+1.4', 'do': 'walk', 'to': [2.6, 1.8]}]})
        self.assertLessEqual(checks['touch_miss_m'], 0.03)  # the knuckles are on the handle when it starts to move
        self.assertIn('touch', [e['kind'] for e in events['events']])
        for cast_at, by, said in (([2.6, 4.6], 'dev', 'stand them closer'), ([2.6, 3.5], 'sam', 'sam')):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile({'cast': [{'id': 'dev', 'posture': 'stand', 'at': cast_at}], 'set': door, 'shots': shots,
                              'beats': [{'at': 'A+0.5', 'do': 'open_door', 'by': by}]})
            self.assertIn(said, str(ctx.exception))

    def test_a_point_lifts_the_arm_at_the_target_with_the_hand_on_its_side(self):
        """A point raises the arm along the line to the target (so it reads from across a room), with the hand
        on the target's side unless the beat names one, and says how true the aim is."""
        checks, events = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [1.6, 1.6]}], 'set': {'kind': 'desk_room', 'door': {}},
            'shots': [{'id': 'A', 'dur': 4, 'type': 'full'}],
            'beats': [{'at': 'A+0.3', 'do': 'point', 'target': [3.5, 0.0, 2.2], 'dur': 1.2},
                      {'at': 'A+2.2', 'do': 'point', 'target': 'camera', 'hand': 'right'}]})
        self.assertTrue(checks['passed'], checks)
        points = [e for e in events['events'] if e['kind'] == 'point']
        self.assertEqual([p['hand'] for p in points], ['left', 'right'])  # (the target at +x is on dev's left)
        for p in points:
            self.assertLessEqual(p['aim_deg'], 12)
        with self.assertRaises(RuntimeError) as ctx:
            self.compile({'shots': [{'id': 'A', 'dur': 2, 'type': 'wide'}],
                          'beats': [{'at': 'A+0.3', 'do': 'point', 'target': 'the door'}]})
        self.assertIn('target', str(ctx.exception))

    def test_a_presenter_carries_a_cup_in_puts_it_down_and_talks_at_a_board(self):
        """A lecture hall: someone comes to the table carrying a cup, puts it down, takes the chalk, shows the
        board, makes a gesture, dusts their hands and talks, filmed by a camera put where the spec says. The cup
        and the chalk are each in the hand until they are put down, and what is wrong in a beat is named."""
        spec = {
            'set': {'kind': 'lecture_hall', 'table': {'at': [1.6, 1.3], 'cup': [1.45, 1.5], 'chalk': [1.4, 1.52]}},
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 1.9], 'facing': -90}],
            'shots': [{'id': 'A', 'dur': 9, 'type': 'fixed', 'cam': [-0.5, -3.0, 1.6], 'aim': [1.2, 2.0, 1.3],
                       'lens': 30, 'push': 0.1}],
            'beats': [{'at': 0, 'do': 'carry', 'prop': 'cup', 'hand': 'left'},
                      {'at': 'A+0.2', 'do': 'walk', 'to': [1.25, 1.82], 'facing': 20, 'until': 'A+2.0'},
                      {'at': 'A+2.2', 'do': 'put_down', 'prop': 'cup', 'dur': 0.7},
                      {'at': 'A+3.5', 'do': 'pick_up', 'prop': 'chalk', 'hand': 'left', 'dur': 0.6},
                      {'at': 'A+4.8', 'do': 'present', 'target': [-1.0, 2.9, 2.2], 'dur': 1.2},
                      {'at': 'A+6.6', 'do': 'gesture', 'kind': 'open', 'hand': 'right', 'dur': 0.8},
                      {'at': 'A+7.6', 'do': 'dust_hands', 'dur': 0.8},
                      {'at': 'A+0.1', 'do': 'blinks'}, {'at': 'A+4.0', 'do': 'talk'}]}
        checks, events = self.compile(spec)
        self.assertTrue(checks['shots'][0]['passed'], checks)
        kinds = [e['kind'] for e in events['events']]
        for kind in ('put_down', 'pick_up', 'point', 'dust'):
            self.assertIn(kind, kinds)
        self.assertLess(kinds.index('put_down'), kinds.index('pick_up'))
        self.assertLessEqual(checks['through_m'], 0.03)                    # nothing goes through the table
        far = {**spec, 'cast': [{'id': 'dev', 'posture': 'stand', 'at': [0.0, 0.0]}], 'beats': [
            {'at': 0, 'do': 'carry', 'prop': 'cup'}, {'at': 'A+1', 'do': 'put_down', 'prop': 'cup'}]}
        with self.assertRaises(RuntimeError) as ctx:
            self.compile(far)
        self.assertIn('short of putting it down', str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            self.compile({**spec, 'beats': [{'at': 0, 'do': 'carry', 'prop': 'umbrella'}]})
        self.assertIn('cup, chalk', str(ctx.exception))
        with self.assertRaises(RuntimeError) as ctx:
            self.compile({**spec, 'shots': [{'id': 'A', 'dur': 2, 'type': 'fixed', 'cam': [0, -3, 1.5]}], 'beats': []})
        self.assertIn('"cam" and "aim"', str(ctx.exception))

    def test_a_kitchen_has_a_lid_a_mug_and_claude_at_his_keys(self):
        """A kitchen: the laptop's lid starts shut and opens, Claude the CRT types on the worktop with a click
        for each key and turns, and a verb or a prop that belongs to another set is named."""
        spec = {
            'set': {'kind': 'kitchen', 'lid': 'closed', 'claude': {}},
            'shots': [{'id': 'A', 'dur': 2, 'type': 'medium'},
                      {'id': 'B', 'dur': 2, 'type': 'fixed', 'cam': [-0.6, 0.6, 1.25], 'aim': [-2.2, 0.0, 1.1], 'lens': 40}],
            'beats': [{'at': 0, 'do': 'pose', 'pose': 'hands_on_head', 'dur': 0.1},
                      {'at': 'A+0.3', 'do': 'open_lid', 'dur': 0.6},
                      {'at': 'B+0.2', 'do': 'claude_type', 'until': 'B+1.2'},
                      {'at': 'B+1.3', 'do': 'claude_turn', 'facing': 70, 'dur': 0.4}]}
        checks, events = self.compile(spec)
        for shot in checks['shots']:
            self.assertTrue(shot['passed'], shot)
        self.assertNotIn('set_problems', checks)                           # no cupboard faces in one plane
        kinds = [e['kind'] for e in events['events']]
        self.assertIn('lid_open', kinds)
        clicks = [e for e in events['events'] if e['kind'] == 'key_click']
        self.assertGreaterEqual(len(clicks), 8)                            # ten keys a second, for a second
        self.assertTrue(all(e['shot'] == 'B' for e in clicks))
        desk = {'shots': spec['shots'][:1]}
        for bad, said in (({**desk, 'beats': [{'at': 'A', 'do': 'close_lid'}]}, 'kitchen verb'),
                          ({**spec, 'set': {'kind': 'kitchen'}, 'beats': [{'at': 'A', 'do': 'claude_type'}]}, 'needs Claude'),
                          ({**spec, 'beats': [{'at': 'A', 'do': 'claude_turn'}]}, 'facing'),
                          ({**spec, 'beats': [{'at': 'A', 'do': 'open_lid', 'by': 'sam'}]}, 'sam'),
                          ({**spec, 'beats': [{'at': 'A', 'do': 'sip'}]}, 'no mug'),
                          ({**spec, 'beats': [{'at': 0, 'do': 'carry', 'prop': 'chalk'}]}, 'one of cup')):
            with self.assertRaises(RuntimeError) as ctx:
                self.compile(bad)
            self.assertIn(said, str(ctx.exception))
            self.assertNotIn('Traceback', str(ctx.exception))

    def test_a_walk_can_bend_and_the_body_leans_into_it(self):
        """A curved walk: the path bows away from the straight line, the body banks into the bend, and the
        check reports each walk. A straight walk reports no bow."""
        checks, _ = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 2.0]}],
            'set': {'kind': 'desk_room', 'door': {}, 'chair_out': 0.8},
            'shots': [{'id': 'A', 'dur': 6, 'type': 'full', 'follow': True}],
            'beats': [{'at': 'A+0.2', 'do': 'walk', 'to': [0, 0.41], 'facing': 0, 'curve': True, 'heading': -90,
                       'until': 'A+4.6'}]})
        self.assertTrue(checks['passed'], checks)
        bend = checks['walks'][0]
        self.assertTrue(bend['curve'])
        self.assertGreater(bend['bow_m'], 0.2)
        self.assertGreaterEqual(bend['into_bend'], 0.5)
        self.assertGreater(bend['lean_deg'], 1.0)
        self.assertLessEqual(checks['foot_slide_m'], 0.03)  # feet land on the bend and stay where they land
        with self.assertRaises(RuntimeError) as ctx:
            self.compile({'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 2.0]}],
                          'set': {'kind': 'desk_room', 'door': {}}, 'shots': [{'id': 'A', 'dur': 3, 'type': 'full'}],
                          'beats': [{'at': 'A+0.2', 'do': 'walk', 'to': [0, 0.41], 'curve': 'yes'}]})
        self.assertIn('curve', str(ctx.exception))

    def test_a_longer_stride_takes_fewer_steps_and_a_whistle_purses_the_lips(self):
        """`stride` makes the same walk in fewer, longer steps; outside 0.3 to 0.85 of a leg it is a named error.
        `whistle` compiles beside a walk, and the sound's `gains` turns the steps down and leaves the rest."""
        def walk(stride):
            checks, events = self.compile({
                'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 2.0]}],
                'set': {'kind': 'desk_room', 'door': {}},
                'shots': [{'id': 'A', 'dur': 5, 'type': 'full', 'follow': True}],
                'beats': [{'at': 'A+0.2', 'do': 'walk', 'to': [0, 0.41], 'until': 'A+4.4', 'stride': stride},
                          {'at': 'A+0.3', 'do': 'whistle', 'until': 'A+4.0'}]})
            return checks, [e for e in events['events'] if e['kind'] == 'step']
        short, short_steps = walk(0.5)
        long, long_steps = walk(0.8)
        self.assertLess(len(long_steps), len(short_steps))
        self.assertLessEqual(long['foot_slide_m'], 0.03)
        with self.assertRaises(RuntimeError) as ctx:
            walk(1.4)
        self.assertIn('stride', str(ctx.exception))
        events = {'fps': 30, 'duration': 1.0, 'events': [{'t': 0.1, 'frame': 3, 'kind': 'step'}]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'steps.json').write_text(json.dumps(events))

            def peak(**kw):
                out = folder / 'steps.wav'
                sound.render(folder / 'steps.json', out, score=False, room=False, **kw)
                with wave.open(str(out)) as w:
                    data = w.readframes(w.getnframes())
                return max(abs(int.from_bytes(data[i:i + 2], 'little', signed=True)) for i in range(0, len(data), 2))
            self.assertAlmostEqual(peak(gains={'step': 0.35}) / peak(), 0.35, delta=0.02)

    def test_a_door_and_a_screen_graphic_have_no_faces_in_one_plane(self):
        """The set is checked for surfaces that would flicker: the door's frame against the wall does not, and
        an alert laid 0.4 mm over the screen does not count."""
        checks, _ = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [2.6, 2.4]}],
            'set': {'kind': 'desk_room', 'door': {}},
            'shots': [{'id': 'A', 'dur': 2, 'type': 'full'}, {'id': 'B', 'dur': 2, 'type': 'over_shoulder', 'subject': 'screen'}],
            'beats': [{'at': 'A+0.2', 'do': 'open_door', 'dur': 0.8}, {'at': 'B+0.3', 'do': 'alert'}]})
        self.assertNotIn('surfaces_fight', checks)
        self.assertNotIn('set_problems', checks)

    def test_hand_shapes_are_named_and_a_mitten_keeps_its_shape(self):
        """The clay cast's hands are mittens: a hand beat is accepted and changes nothing, and none of the
        checks that need fingers is reported. A shape that does not exist is named with the ones that do."""
        checks, _ = self.compile({
            'cast': [{'id': 'dev', 'posture': 'stand', 'at': [1.6, 1.6]}], 'set': {'kind': 'desk_room', 'door': {}},
            'shots': [{'id': 'A', 'dur': 2, 'type': 'full'}],
            'beats': [{'at': 'A+0.3', 'do': 'hand', 'shape': 'point', 'hand': 'right', 'dur': 0.3},
                      {'at': 'A+1.0', 'do': 'hand', 'shape': 'fist'}]})
        self.assertTrue(checks['passed'], checks)
        for key in ('grip_open', 'hands_apart_m', 'palm_tilt_deg', 'palm_turn_deg', 'face', 'skin'):
            self.assertNotIn(key, checks)
        with self.assertRaises(RuntimeError) as ctx:
            self.compile({'shots': [{'id': 'A', 'dur': 2, 'type': 'wide'}],
                          'beats': [{'at': 'A+0.3', 'do': 'hand', 'shape': 'claw'}]})
        for word in ('shape', 'grip', 'point', 'fist', 'flat'):
            self.assertIn(word, str(ctx.exception))

    CONAL = ROOT / 'tests' / 'fixtures' / 'blender-fitted'
    _conal = None

    @classmethod
    def conal(cls):
        """The toolkit's author as a fitted character (a caricature he shares, in tests/fixtures beside Mara):
        fitted once for the tests that use him, into a folder that lasts as long as the class."""
        if cls._conal is None:
            import contextlib
            import io
            tmp = Path(tempfile.mkdtemp())
            cls.addClassCleanup(shutil.rmtree, tmp, ignore_errors=True)
            shutil.copy(cls.CONAL / 'conal.json', tmp / 'conal.json')
            shutil.copy(cls.CONAL / 'enter-conal.json', tmp / 'enter.json')
            args = mock.Mock(model=str(cls.CONAL / 'conal.glb'), fit=str(cls.CONAL / 'conal-fit.json'),
                             out=str(tmp / 'conal-fitted.blend'), preview=None)
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(args)
            assert code == 0, said.getvalue()
            cls._conal = tmp
        return cls._conal


    def test_a_fitted_character_with_fingers_passes_its_own_checks(self):
        """A generated model fitted to the rig (the author's, the body the rig was worked out on): its face, its
        skin and its hands are measured, and the walk-in passes."""
        with tempfile.TemporaryDirectory() as tmp:
            checks = blender_render.run_compiler(self.conal() / 'enter.json', Path(tmp) / 'out', 'check', '480x270', None)
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertGreaterEqual(checks['face']['mouth_share'], 0.18)   # a mouth wide enough to read
        self.assertEqual(checks['skin']['leg_above_hips'], 0)           # no hem that tears when the thighs bend
        self.assertLessEqual(checks['skin']['stretch'], 4.0)
        self.assertLessEqual(checks['grip_open'], 0.75)                 # the hand closes on the door handle
        self.assertGreaterEqual(checks['hands_apart_m'], 0.005)         # the hands never go into each other
        self.assertLessEqual(checks['palm_tilt_deg'], 35)               # and lie palm down on the desk
        self.assertLessEqual(checks['palm_turn_deg'], 35)               # and stay palm to the leg in a walk
        for walk in checks['walks']:                                     # arms hang and swing from the shoulder
            self.assertLessEqual(walk['arms']['elbow_deg'], 35)
            self.assertGreaterEqual(walk['arms']['forearm_deg'], 40)
            self.assertLessEqual(walk['arms']['hand_off_m'], 0.04)
        self.assertLessEqual(checks['typing']['rest_m'], 0.008)         # the fingers rest on the keys
        self.assertGreaterEqual(checks['typing']['finger_travel_m'], 0.006)  # and press them
        for kind in ('hand_in_body_m', 'through_m', 'foot_slide_m', 'touch_miss_m'):
            self.assertLessEqual(checks[kind], 0.03, kind)
        self.assertLessEqual(checks['snap_m'], 0.02)                    # nothing jumps in a single frame

    def test_the_kitchen_example_passes_on_its_second_character(self):
        """examples/blender-kitchen has a film for each fitted character in the repo; this is the author's."""
        film = json.loads((ROOT / 'examples' / 'blender-kitchen' / 'spec-conal.json').read_text())
        film['cast'][0]['from'] = str(self.conal() / 'conal.json')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'film.json').write_text(json.dumps(film))
            checks = blender_render.run_compiler(Path(tmp) / 'film.json', Path(tmp) / 'out', 'check', '480x270', None)
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertLessEqual(checks['skin']['stretch'], 4.0)
        self.assertLessEqual(checks['carried_tilt_deg'], 12.0)
        self.assertLessEqual(checks['carried_through_m'], 0.006)

    def test_someone_at_a_table_shuts_a_lid_without_going_through_it(self):
        """A laptop's lid is further off than a seated arm: the chest turns the shoulder to it and leans only
        as far as the table's edge lets it, the fingertips take the lid's top edge and come down with it, the
        other hand gets off the keys first, and both end flat on the table. Nothing passes through the table,
        the lid, the laptop or the body on the way (a lean given in the spec took both hands through the table)."""
        cast = self.conal() / 'conal.json'
        spec = {'fps': 30, 'set': {'kind': 'kitchen', 'mug': {'on_table': True}}, 'cast': [{'id': 'conal', 'from': str(cast)}],
                'shots': [{'id': 'A', 'dur': 3.0, 'type': 'medium'}],
                'beats': [{'at': 0, 'do': 'pose', 'pose': 'rest', 'dur': 0.1},
                          {'at': 'A+0.8', 'do': 'close_lid', 'dur': 0.6, 'by': 'conal'}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            checks = blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '480x270', None)
            events = json.loads((Path(tmp) / 'out' / 'events.json').read_text())
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertIn('lid_close', [e['kind'] for e in events['events']])
        for kind in ('hand_in_body_m', 'through_m', 'touch_miss_m'):
            self.assertLessEqual(checks[kind], 0.03, kind)
        self.assertLessEqual(checks['snap_m'], 0.02)

    def test_a_mug_is_taken_by_its_handle_drunk_from_and_put_down_level(self):
        """Someone at the kitchen table takes the mug by its handle, lifts it, starts a sip that stops short,
        drinks, and puts it down. The mug is level in the hand throughout (the tip of a sip is the mug's own),
        clear of the table and the shut laptop, and the hand's turn has no frame that spins (rotation keys
        written out of frame order held opposite signs, and the mug turned 139 degrees between two of them)."""
        cast = self.conal() / 'conal.json'
        spec = {'fps': 30, 'set': {'kind': 'kitchen', 'lid': 'closed', 'mug': {'at': [0.5, -0.465], 'on_table': True}},
                'cast': [{'id': 'conal', 'from': str(cast)}],
                'shots': [{'id': 'A', 'dur': 4.5, 'type': 'medium'}, {'id': 'B', 'dur': 5.5, 'type': 'closeup'}],
                'beats': [{'at': 0, 'do': 'pose', 'pose': 'rest', 'dur': 0.1},
                          {'at': 'A+0.8', 'do': 'pick_up', 'prop': 'cup', 'hand': 'left', 'dur': 0.8},
                          {'at': 'A+2.6', 'do': 'sip', 'hand': 'left', 'part': 0.6, 'stay': True, 'dur': 0.8},
                          {'at': 'B+0.3', 'do': 'sip', 'hand': 'left', 'dur': 2.0},
                          {'at': 'B+2.9', 'do': 'put_down', 'prop': 'cup', 'hand': 'left', 'dur': 0.8}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            checks = blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '480x270', None)
            events = json.loads((Path(tmp) / 'out' / 'events.json').read_text())
        self.assertEqual([e['kind'] for e in events['events'] if e['kind'] in ('pick_up', 'sip', 'put_down')],
                         ['pick_up', 'sip', 'sip', 'put_down'])
        self.assertLessEqual(checks['carried_tilt_deg'], 12.0)
        self.assertLessEqual(checks['carried_through_m'], 0.006)
        self.assertGreaterEqual(checks['hands_apart_m'], 0.005)
        self.assertLessEqual(checks['hand_in_body_m'], 0.03)
        self.assertLessEqual(checks['snap_m'], 0.02)

    def test_a_pint_is_held_where_it_stands_let_go_of_taken_up_and_drunk_from_on_a_call(self):
        """Someone at a pub table holds a phone up for a video call in one hand; the other lies round a pint
        that stands on the table (no handle: the hand goes round the glass), comes off it, takes it up and
        drinks. The phone and the glass are sized to the hand that holds them, a hand round a glass is on its
        side on purpose (not a hand rolled off the table), the glass stays level in the hand, and a camera can
        look back from the phone at the face."""
        cast = self.conal() / 'conal.json'
        spec = {'fps': 30, 'resolution': '1080x1920', 'set': {'kind': 'pub'}, 'cast': [{'id': 'conal', 'from': str(cast)}],
                'shots': [{'id': 'P0', 'dur': 1.0, 'type': 'medium', 'az': 30},
                          {'id': 'A', 'dur': 2.0, 'type': 'fixed', 'cam': 'phone', 'aim': 'face', 'lens': 17, 'clip': 0.08},
                          {'id': 'B', 'dur': 4.5, 'type': 'medium', 'az': 30, 'dof': 2.8}],
                'beats': [{'at': 0, 'do': 'pose', 'pose': 'rest', 'dur': 0.0},
                          {'at': 0, 'do': 'phone', 'hand': 'right', 'dist': 0.6, 'drop': 0.0},
                          {'at': 0.05, 'do': 'pick_up', 'prop': 'pint', 'hand': 'left', 'lift': False, 'dur': 0.3},
                          {'at': 0.7, 'do': 'look_at', 'target': 'phone', 'dur': 0.2},
                          {'at': 'A+0.5', 'do': 'let_go', 'hand': 'left', 'dur': 0.7},
                          {'at': 'B+0.2', 'do': 'look_at', 'target': 'pint', 'dur': 0.3},
                          {'at': 'B+0.3', 'do': 'pick_up', 'prop': 'pint', 'hand': 'left', 'dur': 0.7},
                          {'at': 'B+1.9', 'do': 'sip', 'hand': 'left', 'dur': 2.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            checks = blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '270x480', None)
            events = json.loads((Path(tmp) / 'out' / 'events.json').read_text())
        kinds = [e['kind'] for e in events['events']]
        self.assertEqual([k for k in kinds if k in ('phone', 'touch', 'pick_up', 'sip')], ['phone', 'touch', 'pick_up', 'sip'])
        held = next(e for e in events['events'] if e['kind'] == 'phone')
        self.assertGreater(held['from_face_m'], 0.15)                  # as far out as a short arm holds it
        self.assertTrue(all(s['passed'] for s in checks['shots']), checks['shots'])
        self.assertLessEqual(checks['carried_tilt_deg'], 12.0)
        self.assertLessEqual(checks.get('carried_through_m', 0.0), 0.006)
        self.assertLessEqual(checks['palm_tilt_deg'], 35)              # the hand round the glass is not counted
        self.assertLessEqual(checks['hand_in_body_m'], 0.03)
        self.assertLessEqual(checks['through_m'], 0.03)
        self.assertLessEqual(checks['snap_m'], 0.02)

    def test_someone_on_a_sofa_holds_a_phone_and_wags_a_finger_with_hands_that_start_in_the_lap(self):
        """A living room has no desk: the hands start on the thighs. One lifts a phone for a call and the other
        wags a finger at it; a hand that holds something can't be the one that wags, and the error says so."""
        cast = self.conal() / 'conal.json'
        spec = {'fps': 30, 'resolution': '1080x1920', 'set': {'kind': 'living_room'}, 'cast': [{'id': 'conal', 'from': str(cast)}],
                'shots': [{'id': 'A', 'dur': 4.0, 'type': 'medium', 'az': -24, 'dof': 2.2}],
                'beats': [{'at': 0.3, 'do': 'phone', 'hand': 'left', 'dur': 0.6},
                          {'at': 1.0, 'do': 'look_at', 'target': 'phone', 'dur': 0.3},
                          {'at': 1.6, 'do': 'wag', 'hand': 'right', 'dur': 1.6, 'toward': 'phone'}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            checks = blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out', 'check', '270x480', None)
            events = json.loads((Path(tmp) / 'out' / 'events.json').read_text())
            self.assertTrue(all(s['passed'] for s in checks['shots']), checks['shots'])
            self.assertIn('phone', [e['kind'] for e in events['events']])
            self.assertLessEqual(checks['hand_in_body_m'], 0.03)       # the lap hands lie on the legs, not in them
            self.assertLessEqual(checks['through_m'], 0.03)            # and the body is on the sofa, not in it
            self.assertLessEqual(checks['snap_m'], 0.02)
            spec['beats'][2]['hand'] = 'left'
            (Path(tmp) / 'spec.json').write_text(json.dumps(spec))
            with self.assertRaises(RuntimeError) as ctx:
                blender_render.run_compiler(Path(tmp) / 'spec.json', Path(tmp) / 'out2', 'check', '270x480', None)
            self.assertIn('holding something', str(ctx.exception))

    MARA = ROOT / 'tests' / 'fixtures' / 'blender-fitted'

    def test_an_invented_model_is_fitted_with_a_face_and_fingers_and_acts(self):
        """Mara (tests/fixtures/blender-fitted, made up, not a likeness) goes through the fit as anyone's model
        would: five fingers found on each hand, sized to her own, the wrist put at the cuff, the thumb laid in to
        the palm, legs that touched parted; then she takes hand shapes, points, pulls faces and winks, and the
        checks pass. Then the walk-in: through the door, across the room, onto the chair, in to the desk and
        typing, on a body nothing in the rig was tuned on (short arms, long legs, narrow shoulders)."""
        import contextlib
        import io
        import re
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ('spec.json', 'enter.json', 'kitchen.json', 'mara.json'):
                shutil.copy(self.MARA / name, tmp / name)
            said = io.StringIO()
            args = mock.Mock(model=str(self.MARA / 'mara.glb'), fit=str(self.MARA / 'fit.json'),
                             out=str(tmp / 'mara-fitted.blend'), preview=None)
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(args)
            said = said.getvalue()
            self.assertEqual(code, 0, said)
            self.assertEqual(len(re.findall(r'fit: hands, [LR] 5 fingers', said)), 2, said)
            depth = re.search(r"thumb root (\d+)% of the hand's length out of the palm \(made (\d+)%\)", said)
            self.assertIsNotNone(depth, said)
            self.assertLessEqual(int(depth.group(1)), 10)                  # laid in from the generator's 20%
            self.assertGreater(int(depth.group(2)), int(depth.group(1)))
            self.assertIn('legs touched', said)                            # her shoes are one piece in the model
            # the donor's hands are a fifth too long: the fit brings the donor to the body's arm span first, then
            # sizes the hands, and the two together are the hand's scale
            sized = re.search(r"scaled (\d\.\d+)x to match", said)
            self.assertIsNotNone(sized, said)
            span = re.search(r"the hands model is (\d\.\d+)x the body's arm span", said)
            total = float(sized.group(1)) / (float(span.group(1)) if span else 1.0)
            self.assertLess(total, 0.95, said)
            self.assertIn('wrist joint moved', said)                       # read off a still, it was up the sleeve
            self.assertEqual(len(re.findall(r'fit: painted face, (?:right|left) eye opens', said)), 2, said)
            checks = blender_render.run_compiler(tmp / 'spec.json', tmp / 'out', 'check', '480x270', None)
            events = json.loads((tmp / 'out' / 'events.json').read_text())
            enter = blender_render.run_compiler(tmp / 'enter.json', tmp / 'enter', 'check', '480x270', None)
            kitchen = blender_render.run_compiler(tmp / 'kitchen.json', tmp / 'kitchen', 'check', '480x270', None)
            kitchen_events = json.loads((tmp / 'kitchen' / 'events.json').read_text())
            film = json.loads((ROOT / 'examples' / 'blender-kitchen' / 'spec.json').read_text())
            film['cast'][0]['from'] = 'mara.json'                          # the copy here, fitted above
            (tmp / 'film.json').write_text(json.dumps(film))
            example = blender_render.run_compiler(tmp / 'film.json', tmp / 'film', 'check', '480x270', None)
            self.assertTrue(example['passed'], example)                    # the example passes outright: its
            self.assertLessEqual(example['skin']['stretch'], 4.0)          # sip is part height for the sleeve
            self.assertLessEqual(example['carried_tilt_deg'], 12.0)
            at_table = json.loads((tmp / 'kitchen.json').read_text())
            shut = {**at_table, 'beats': at_table['beats'] + [{'at': 'T9_claude+0.5', 'do': 'close_lid', 'by': 'mara'}]}
            (tmp / 'shut.json').write_text(json.dumps(shut))
            with self.assertRaises(RuntimeError) as cannot_shut:
                blender_render.run_compiler(tmp / 'shut.json', tmp / 'shut', 'check', '480x270', None)
            stood = {**at_table, 'shots': at_table['shots'][:1], 'cast': [{**at_table['cast'][0], 'at': [0.75, -0.35],
                                                                            'facing': 90}],
                     'beats': [{'at': 0, 'do': 'carry', 'prop': 'cup'}, {'at': 0.5, 'do': 'put_down', 'prop': 'cup'}]}
            (tmp / 'stood.json').write_text(json.dumps(stood))
            with self.assertRaises(RuntimeError) as too_low:
                blender_render.run_compiler(tmp / 'stood.json', tmp / 'stood', 'check', '480x270', None)
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertGreaterEqual(checks['face']['mouth_share'], 0.18)
        self.assertGreater(checks['face']['mouth_travel_m'], 0.005)        # the mouth moves for the faces
        self.assertLessEqual(checks['face']['eye_tall'], 1.5)              # the rig's eyes are the size of her painted ones
        self.assertFalse(checks['face']['paint_kept'])
        self.assertEqual(checks['skin']['leg_above_hips'], 0)
        self.assertLessEqual(checks['skin']['stretch'], 4.0)
        self.assertGreaterEqual(checks['hands_apart_m'], 0.005)
        points = [e for e in events['events'] if e['kind'] == 'point']
        self.assertEqual(len(points), 1)
        self.assertLessEqual(points[0]['aim_deg'], 12)
        for shot in checks['shots']:                                       # the close-up finds her own head
            self.assertTrue(shot['passed'], shot)
        self.assertTrue(enter['passed'], enter.get('body_problems'))
        self.assertLessEqual(enter['ik_max_error_m'], 0.01)                # every reach is one her arms can make
        self.assertLessEqual(enter['grip_open'], 0.75)                     # the hand closes on the door handle
        self.assertGreaterEqual(enter['hands_apart_m'], 0.005)             # broad hands on narrow shoulders stay apart
        self.assertLessEqual(enter['snap_m'], 0.02)                        # nothing jumps in a single frame
        self.assertLessEqual(enter['skin']['stretch'], 4.0)
        self.assertLessEqual(enter['typing']['rest_m'], 0.008)
        self.assertGreaterEqual(enter['typing']['into_m'], -0.004)
        for walk in enter['walks']:
            self.assertLessEqual(walk['arms']['elbow_deg'], 35)
            self.assertLessEqual(walk['arms']['hand_off_m'], 0.04)
        for kind in ('hand_in_body_m', 'through_m', 'foot_slide_m', 'touch_miss_m'):
            self.assertLessEqual(enter[kind], 0.03, kind)
        # the kitchen: in with a mug, sat down, the mug put down from the chair with the other hand left on
        # the table, the lid opened by its edge with her fingertips, the chest turned as far as her reach needs
        for shot in kitchen['shots']:
            self.assertTrue(shot['passed'], shot)
        kinds = [e['kind'] for e in kitchen_events['events']]
        for kind in ('put_down', 'lid_open', 'key_click', 'pick_up', 'sip'):
            self.assertIn(kind, kinds)
        for kind in ('hand_in_body_m', 'through_m', 'touch_miss_m'):     # no hand in the table, the lid or her
            self.assertLessEqual(kitchen[kind], 0.03, kind)
        # the mug: carried by its handle through the door, the sitting down and the chair pulled in, put down,
        # taken up again and drunk from, level all the way and never in the door, the table or the laptop
        self.assertLessEqual(kitchen['carried_tilt_deg'], 12.0)
        self.assertLessEqual(kitchen['carried_through_m'], 0.006)
        self.assertGreaterEqual(kitchen['hands_apart_m'], 0.005)          # the carrying hand keeps to its side
        self.assertLessEqual(kitchen['snap_m'], 0.02)                      # the chest's turn is not a jump
        self.assertIn('short of the lid', str(cannot_shut.exception))     # her arm is too short to shut it
        self.assertIn('leave out "by"', str(cannot_shut.exception))
        self.assertIn('sit them down first', str(too_low.exception))      # a standing arm can't reach a table

    def test_a_face_fitted_to_keep_its_paint_acts_with_its_own_eyes(self):
        """The same model fitted with "paint": "keep": nothing is erased, each eye is a patch cut from the paint
        with its images packed in the fitted file, and the face pulls its expressions, blinks, winks and
        glances without the rig's own eyes. A face with no painted eye where the fit says is refused."""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fit = json.loads((self.MARA / 'fit.json').read_text())
            fit.pop('hands', None)
            fit['face']['paint'] = 'keep'
            (tmp / 'fit.json').write_text(json.dumps(fit))
            shutil.copy(self.MARA / 'mara.json', tmp / 'mara.json')
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(mock.Mock(model=str(self.MARA / 'mara.glb'), fit=str(tmp / 'fit.json'),
                                                    out=str(tmp / 'mara-fitted.blend'), preview=None))
            self.assertEqual(code, 0, said.getvalue())
            spec = {'title': 'Kept face', 'fps': 24, 'resolution': '1280x720', 'engine': 'eevee',
                    'set': {'kind': 'desk_room'}, 'cast': [{'id': 'mara', 'from': 'mara.json'}],
                    'shots': [{'id': 'F', 'dur': 4.0, 'type': 'closeup'}],
                    'beats': [{'at': 0.1, 'do': 'look_at', 'target': 'camera', 'dur': 0.3},
                              {'at': 0.6, 'do': 'expression', 'face': 'happy', 'dur': 0.3},
                              {'at': 1.2, 'do': 'expression', 'face': 'worried', 'dur': 0.3},
                              {'at': 1.8, 'do': 'expression', 'face': 'grin', 'dur': 0.3},
                              {'at': 2.4, 'do': 'blink'}, {'at': 2.8, 'do': 'wink'},
                              {'at': 3.2, 'do': 'glance', 'yaw': 20, 'dur': 0.2}]}
            (tmp / 'face.json').write_text(json.dumps(spec))
            checks = blender_render.run_compiler(tmp / 'face.json', tmp / 'out', 'check', '480x270', None)
            off = json.loads(json.dumps(fit))
            off['face']['eyes'] = [[x, z + 0.12] for x, z in off['face']['eyes']]   # in her hair: no eye there
            (tmp / 'off.json').write_text(json.dumps(off))
            refused = io.StringIO()
            with contextlib.redirect_stdout(refused), contextlib.redirect_stderr(refused):
                code = blender_render.fit(mock.Mock(model=str(self.MARA / 'mara.glb'), fit=str(tmp / 'off.json'),
                                                    out=str(tmp / 'off.blend'), preview=None))
        self.assertTrue(checks['face']['paint_kept'])
        self.assertNotIn('eye_tall', checks['face'])                       # her own eyes: nothing to hold them against
        for shot in checks['shots']:
            self.assertTrue(shot['passed'], shot)
        self.assertNotEqual(code, 0)
        self.assertIn('needs a painted eye', refused.getvalue())

    NELL = ROOT / 'tests' / 'fixtures' / 'blender-bare'

    def test_a_bare_armed_model_is_cut_at_the_wrist_sealed_and_keeps_its_painted_face(self):
        """Nell (tests/fixtures/blender-bare, made up) has no cuffs: each hand is found by the wrist's own
        narrowing, the arm's end is drawn onto the hand's ring and the two share their normals there, the hands'
        texture is cleared of the picture's background, and a hands model made from a closer picture is brought
        to the body's size first. Her face is painted nearer to life, so it is kept: both painted eyes are read
        (one has a liner whose outer edge is a sharper step than the iris's), and she pulls faces, blinks,
        glances and winks with them."""
        import contextlib
        import io
        import re
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ('spec.json', 'nell.json'):
                shutil.copy(self.NELL / name, tmp / name)
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(mock.Mock(model=str(self.NELL / 'nell.glb'), fit=str(self.NELL / 'fit.json'),
                                                    out=str(tmp / 'nell-fitted.blend'), preview=None))
            said = said.getvalue()
            self.assertEqual(code, 0, said)
            checks = blender_render.run_compiler(tmp / 'spec.json', tmp / 'out', 'check', '480x270', None)
        self.assertEqual(len(re.findall(r'fit: hands, [LR] 5 fingers \(thumb, index, middle, ring, little\)', said)), 2, said)
        sealed = re.findall(r"a bare arm's end drawn in onto the hand: the ring stood ([\d.]+) mm off the hand's "
                            r"skin, now ([\d.]+) mm", said)
        self.assertEqual(len(sealed), 2, said)                             # both arms, with no cuff over the join
        for before, after in sealed:
            self.assertGreater(float(before), 1.0)
            self.assertLessEqual(float(after), 0.1)
        self.assertRegex(said, r'\d+ vertices on the ring where a bare arm meets its hand share their normals')
        self.assertRegex(said, r"the hands model is 1\.\d+x the body's arm span")
        grey = re.search(r"([\d.]+)% of their texture was not skin's colour", said)
        self.assertIsNotNone(grey, said)
        self.assertLess(float(grey.group(1)), 2.0)
        stretch = re.search(r'edges stretched ([\d.]+)x', said)
        self.assertLessEqual(float(stretch.group(1)), 3.0, said)           # the thumb's web is not torn
        eyes = re.findall(r'(?:right|left) eye opens ([\d.]+) x ([\d.]+) mm, iris ([\d.]+) mm', said)
        self.assertEqual(len(eyes), 2, said)
        for wide, tall, iris in eyes:                                      # (misread, the left iris came out 42 mm)
            self.assertLess(float(iris), 25.0, said)
            self.assertLess(float(tall), 0.6 * float(wide), said)          # a lidded eye, wider than it is tall
        self.assertNotIn(' FAIL ', said)
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertTrue(checks['face']['paint_kept'])
        self.assertGreaterEqual(checks['face']['brow_squeeze'], 0.5)       # the brows do not crease the skin
        for kind in ('hand_in_body_m', 'through_m'):
            self.assertLessEqual(checks[kind], 0.03, kind)

    def test_an_arm_off_its_mirrored_elbow_is_failed_and_its_own_joint_frees_it(self):
        """Nell's right arm hangs 3 cm nearer her body than her left. With the elbow mirrored from the left, the
        inside of her right forearm moved with her spine: skin hung from the arm to the hip, and typing, the
        forearm sagged 3.2 cm into the desk. The fit fails that by counting edges from forearm straight to body
        (none on a free arm). With `joints_right` the arm is free; the forearm keeps none of the hip's weight,
        and it stays out of the desk through the walk-in's typing."""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fit = json.loads((self.NELL / 'fit.json').read_text())
            mirrored = {k: v for k, v in fit.items() if k != 'joints_right'}
            (tmp / 'mirrored.json').write_text(json.dumps(mirrored))
            shutil.copy(self.NELL / 'nell_hands.glb', tmp / 'nell_hands.glb')
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                blender_render.fit(mock.Mock(model=str(self.NELL / 'nell.glb'), fit=str(tmp / 'mirrored.json'),
                                             out=str(tmp / 'mirrored.blend'), preview=None))
            self.assertIn('the inside of the right forearm moves with the body', said.getvalue())
            self.assertNotIn('the inside of the left forearm', said.getvalue())
            for name in ('enter.json', 'nell.json'):
                shutil.copy(self.NELL / name, tmp / name)
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(mock.Mock(model=str(self.NELL / 'nell.glb'), fit=str(self.NELL / 'fit.json'),
                                                    out=str(tmp / 'nell-fitted.blend'), preview=None))
            self.assertEqual(code, 0, said.getvalue())
            self.assertNotIn(' FAIL ', said.getvalue())
            enter = blender_render.run_compiler(tmp / 'enter.json', tmp / 'enter', 'check', '480x270', None)
        self.assertFalse([p for p in enter.get('body_problems', []) if 'desk' in p or 'lap_base' in p], enter.get('body_problems'))
        self.assertLessEqual(enter['through_m'], 0.03)
        self.assertLessEqual(enter['typing']['into_m'], 0.003)

    def test_a_model_is_made_smaller_without_painting_faces_from_the_wrong_place(self):
        """`reduce` thins a model inside the islands of its texture: half of Mara's faces, with next to none of
        them left reading the texture from across a seam (0.04%; reduced across the seams it is 1.08%). Not a
        third: a body of 20,000 faces is coarser than any the rig keeps, and there the measure reads 0.22% even
        with every seam held, because a face's nearest point on the model lands in the next island."""
        import contextlib
        import io
        import re
        with tempfile.TemporaryDirectory() as tmp:
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.reduce(mock.Mock(model=str(self.MARA / 'mara.glb'), out=str(Path(tmp) / 'small.glb'),
                                                       faces=30000, texture=512))
            self.assertEqual(code, 0, said.getvalue())
            self.assertTrue((Path(tmp) / 'small.glb').exists())
        got = re.search(r'REDUCED (\d+) to (\d+) faces, (\d+) vertices .* ([\d.]+)% of the faces painted', said.getvalue())
        self.assertIsNotNone(got, said.getvalue())
        self.assertLess(int(got.group(2)), 0.55 * int(got.group(1)))
        self.assertGreater(int(got.group(3)), 100)
        self.assertLess(float(got.group(4)), 0.2)

    def test_a_deep_chest_reaches_from_its_own_shoulder_and_a_hand_on_the_lap_stays_on_the_leg(self):
        """Mara fitted again with her shoulder joints 4.5 cm behind the chest (a deep chest, as a third body
        had): a reach is measured from the joint where it is, so sitting down her hands come to her thighs
        with the arm they have; and pulling in, a hand waiting on her lap rides the leg while the chest leans
        over it. Measured from in front of the joint, the wrist missed its target; held to the chest, the hand
        went near 3 cm into the thigh."""
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(self.MARA / 'mara.json', tmp / 'mara.json')
            fit = json.loads((self.MARA / 'fit.json').read_text())
            fit['joints']['sh'][1] += 0.045
            fit['hands']['model'] = str(self.MARA / fit['hands']['model'])
            (tmp / 'fit.json').write_text(json.dumps(fit))
            args = mock.Mock(model=str(self.MARA / 'mara.glb'), fit=str(tmp / 'fit.json'),
                             out=str(tmp / 'mara-fitted.blend'), preview=None)
            said = io.StringIO()
            with contextlib.redirect_stdout(said):
                code = blender_render.fit(args)
            self.assertEqual(code, 0, said.getvalue())
            (tmp / 'sit.json').write_text(json.dumps({
                'fps': 24, 'resolution': '960x540', 'engine': 'eevee', 'set': {'kind': 'desk_room', 'chair_out': 0.75},
                'cast': [{'id': 'mara', 'from': 'mara.json', 'posture': 'stand', 'at': [0, 0.41], 'facing': 0}],
                'shots': [{'id': 'A', 'dur': 4.5, 'type': 'wide'}],
                'beats': [{'at': 'A+0.2', 'do': 'sit_down', 'dur': 1.8}, {'at': 'A+2.3', 'do': 'pull_in', 'dur': 1.5}]}))
            checks = blender_render.run_compiler(tmp / 'sit.json', tmp / 'out', 'check', '480x270', None)
        self.assertTrue(checks['passed'], checks.get('body_problems'))
        self.assertLessEqual(checks['ik_max_error_m'], 0.001)    # (3.4 mm with the reach measured from the chest)
        self.assertLessEqual(checks['hand_in_body_m'], 0.015)    # (2.8 cm with the hand held to the chest)
        self.assertLessEqual(checks['snap_m'], 0.02)

    def test_flicker_is_a_still_point_that_jumps_for_a_frame_and_comes_back(self):
        """The render check reads still points of the set out of the frames: one that jumps and comes back on
        five frames of a shot fails, a single blip is only noted, and a smooth change is neither."""
        import numpy as np
        frames = [np.full((40, 40), 100.0, dtype=np.float32) for _ in range(30)]
        for f in (4, 8, 12, 16, 20):
            frames[f][8:13, 8:13] = 130.0       # the door frame: five one-frame jumps
        for f in (6, 14):
            frames[f][28:33, 8:13] = 130.0      # the wall: two
        for f in range(30):
            frames[f][18:23, 28:33] = 100 + 3 * f   # a lamp coming up smoothly
        rows = [[f, k, x, y, x, y, x, y] for f in range(1, 31) for k, (x, y) in enumerate(((0.25, 0.25), (0.25, 0.75), (0.75, 0.5)))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'still_points.json').write_text(json.dumps(
                {'objects': ['door_frame', 'wall', 'lamp'], 'shots': {'A': {'frames': [1, 30], 'points': rows}}}))
            (out / 'A.mp4').write_bytes(b'')
            real = blender_render.gray_frames
            blender_render.gray_frames = lambda video, size: iter(frames)
            try:
                found = blender_render.flicker(out, {'A': (out / 'A.mp4', 1)}, (40, 40))
            finally:
                blender_render.gray_frames = real
        by = {r['object']: r for r in found}
        self.assertTrue(by['door_frame']['fails'])
        self.assertEqual(by['door_frame']['frames'], [5, 9, 13, 17, 21])
        self.assertFalse(by['wall']['fails'])
        self.assertNotIn('lamp', by)

    def test_flicker_is_not_a_picture_sliding_under_a_slow_camera(self):
        """A push-in moves a still point a fraction of a pixel a frame. Over a board with writing on it, a
        patch that snaps to whole pixels reads each snap as a jump (the lecture hall's board: 104 frames
        reported, none to see). Read at the point's own place, the level changes smoothly; two faces
        in one plane on the same surface, under the same move, are still caught."""
        import numpy as np
        yy, xx = np.mgrid[0:60, 0:200].astype(np.float32)
        writing = 60.0 + 120.0 * ((xx.astype(int) // 3) % 2)     # chalk strokes: 3 px light, 3 px dark
        frames = [writing.copy() for _ in range(60)]
        for f in (10, 20, 30, 40, 50):
            frames[f][:, 120:] += 25.0                            # the right half fights with another face
        drift = lambda f: 0.3 * f                                 # pixels: the camera pushing in
        at = lambda f, x: (x + drift(f)) / 200
        rows = [[f, k, at(f, x), 0.5, at(f - 1, x), 0.5, at(f + 1, x), 0.5]
                for f in range(1, 61) for k, x in enumerate((40.0, 150.0))]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'still_points.json').write_text(json.dumps(
                {'objects': ['board_face', 'board_patch'], 'shots': {'A': {'frames': [1, 60], 'points': rows}}}))
            (out / 'A.mp4').write_bytes(b'')
            real = blender_render.gray_frames
            # the picture slides with the camera: frame f shows the writing moved by drift(f)
            moved = [np.stack([np.interp(np.arange(200) - drift(i + 1), np.arange(200), fr[r]) for r in range(60)])
                     for i, fr in enumerate(frames)]
            blender_render.gray_frames = lambda video, size: iter(moved)
            try:
                found = blender_render.flicker(out, {'A': (out / 'A.mp4', 1)}, (200, 60))
            finally:
                blender_render.gray_frames = real
        by = {r['object']: r for r in found}
        self.assertNotIn('board_face', by)
        self.assertTrue(by['board_patch']['fails'])
        self.assertEqual(by['board_patch']['frames'], [11, 21, 31, 41, 51])

    def test_mistakes_with_postures_are_named(self):
        shots = [{'id': 'A', 'dur': 3, 'type': 'wide'}]
        for extra, said in (
                ({'beats': [{'at': 'A', 'do': 'walk', 'to': [1, 1]}]}, 'stand_up first'),     # still on the chair
                ({'beats': [{'at': 'A', 'do': 'walk'}]}, '[x, y]'),
                ({'beats': [{'at': 'A', 'do': 'turn'}]}, 'degrees'),
                ({'beats': [{'at': 'A', 'do': 'open_door'}]}, 'no door'),
                ({'beats': [{'at': 'A', 'do': 'wink', 'side': 'both'}]}, 'left'),
                ({'beats': [{'at': 'A', 'do': 'reach', 'target': [0, -9, 1]}]}, 'walk them closer'),
                ({'beats': [{'at': 'A', 'do': 'look_at', 'target': 'moon'}]}, 'camera'),
                ({'cast': [{'id': 'dev', 'posture': 'lying'}]}, 'stand'),
                ({'shots': [{'id': 'A', 'dur': 3, 'type': 'feet'}]}, 'on their feet'),
                ({'set': {'kind': 'desk', 'door': {}}}, 'desk_room')):
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
