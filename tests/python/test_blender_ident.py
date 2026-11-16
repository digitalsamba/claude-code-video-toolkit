"""A logo as a studio prop, the beats that move its parts, and the checks on an ident.

Most of this runs without Blender: the SVG reader, the tracks a beat makes, the named sounds, the colour half of
the last-frame test, the hit check and the mark sheet. The rest builds small specs in Blender when it is installed.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'lib' / 'blender'))
import blender_render
import mark_sheet
import motion
import sound
import svg
from timeline import SpecError

EXAMPLE = ROOT / 'examples' / 'blender-ident'
FFMPEG = bool(shutil.which('ffmpeg'))


def have_blender():
    binary = next((c for c in blender_render.blender_candidates() if Path(c).exists()), None)
    return bool(binary) and blender_render.blender_version(binary)[0] >= blender_render.MIN_VERSION


class Folder(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)

    def svg(self, body, name='m.svg', box='0 0 100 100'):
        path = self.folder / name
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{box}">{body}</svg>')
        return path


class SvgTests(Folder):
    def test_each_filled_shape_is_a_part_with_its_id_and_colour(self):
        parts, box, notes = svg.read(self.svg(
            '<polygon id="bar" points="0,0 10,0 10,50 0,50" fill="#3771E0"/>'
            '<rect x="20" y="0" width="30" height="50" style="fill: rgb(255, 0, 0)"/>'
            '<circle cx="80" cy="50" r="10" fill="navy"/>'))
        self.assertEqual([p.id for p in parts], ['bar', 'p2', 'p3'])
        self.assertEqual([p.fill for p in parts], ['#3771e0', '#ff0000', '#000080'])
        self.assertEqual(box, (0, 0, 100, 100))
        self.assertEqual(notes, [])
        self.assertEqual(parts[1].box(), (20, 0, 50, 50))
        x0, y0, x1, y1 = parts[2].box()
        self.assertAlmostEqual(x1 - x0, 20, places=3)

    def test_a_path_keeps_its_holes_and_its_curves_are_flattened(self):
        # a square with a square hole, then a quarter disc drawn with an arc
        parts, _, _ = svg.read(self.svg(
            '<path id="o" d="M0 0 H40 V40 H0 Z M10 10 h20 v20 h-20 z" fill="#000"/>'
            '<path id="q" d="M50,50 L90,50 A40,40 0 0 1 50,90 Z" fill="#000"/>'))
        o, q = parts
        self.assertEqual(len(o.rings), 2)
        self.assertAlmostEqual(abs(svg.area(o.rings[0])), 1600)
        self.assertAlmostEqual(abs(svg.area(o.rings[1])), 400)
        self.assertAlmostEqual(abs(svg.area(q.rings[0])), 3.14159 * 40 * 40 / 4, delta=2)

    def test_transforms_on_groups_and_shapes_are_applied(self):
        parts, _, _ = svg.read(self.svg(
            '<g transform="translate(10 20) scale(2)"><rect width="5" height="5" transform="rotate(90)" fill="#000"/></g>'))
        x0, y0, x1, y1 = parts[0].box()
        self.assertEqual([round(v, 4) for v in (x0, y0, x1, y1)], [0, 20, 10, 30])

    def test_what_is_not_read_is_said(self):
        parts, _, notes = svg.read(self.svg(
            '<rect width="10" height="10" fill="#000"/><text x="0" y="0">hi</text>'
            '<path d="M0 0 L5 5" fill="none" stroke="#000"/><rect width="4" height="4" fill="url(#g)"/>'))
        self.assertEqual(len(parts), 2)
        said = ' '.join(notes)
        for word in ('text', 'stroke', 'gradient'):
            self.assertIn(word, said)
        self.assertEqual(parts[1].fill, '#808080')

    def test_a_file_with_nothing_to_build_is_an_error_that_says_why(self):
        with self.assertRaises(svg.SvgError) as ctx:
            svg.read(self.svg('<path d="M0 0 L5 5" fill="none" stroke="#000"/>'))
        self.assertIn('stroke', str(ctx.exception))
        (self.folder / 'bad.svg').write_text('<html></html>')
        with self.assertRaises(svg.SvgError):
            svg.read(self.folder / 'bad.svg')
        with self.assertRaises(svg.SvgError):
            svg.path_rings('M0 0 L5')

    def test_the_ident_draws_the_same_mark_as_the_brand(self):
        ident = {p.id: p for p in svg.read(EXAMPLE / 'assets' / 'mark-parts.svg')[0]}
        for name in ('mark.svg', 'mark-on-dark.svg', 'mark-white.svg'):
            flat = svg.read(ROOT / 'brands' / 'video-toolkit' / 'assets' / name)[0]
            self.assertEqual([p.id for p in flat], ['slash', 'play'])
            for p in flat:
                self.assertEqual(p.rings, ident[p.id].rings, f'{name}: {p.id}')


class MotionTests(unittest.TestCase):
    def test_a_track_holds_then_moves_then_holds(self):
        t = motion.Track(1.0)
        t.to(10, 20, 3.0, 'linear')
        self.assertEqual([t.at(f) for f in (1, 10, 15, 20, 99)], [1.0, 1.0, 2.0, 3.0, 3.0])
        self.assertEqual(t.span(), (10, 20))

    def test_a_move_starts_from_where_the_last_one_had_got_to(self):
        t = motion.Track(0.0)
        t.to(0, 10, 10.0, 'linear')
        t.to(5, 15, 0.0, 'linear')   # cut in half-way: from 5, back to 0
        self.assertAlmostEqual(t.at(5), 5.0)
        self.assertAlmostEqual(t.at(10), 2.5)
        self.assertAlmostEqual(t.at(15), 0.0)

    def test_a_move_with_no_length_is_a_jump(self):
        t = motion.Track(0.0)
        t.to(7, 7, 4.0)
        self.assertEqual([t.at(6), t.at(7)], [0.0, 4.0])

    def test_every_ease_starts_at_nought_and_ends_at_one_and_a_spring_goes_past(self):
        for name, fn in motion.EASES.items():
            self.assertAlmostEqual(fn(0.0), 0.0, places=6, msg=name)
            self.assertAlmostEqual(fn(1.0), 1.0, places=6, msg=name)
        self.assertGreater(max(motion.EASES['spring'](i / 50) for i in range(51)), 1.03)
        self.assertLess(motion.EASES['in'](0.5), 0.2)  # still slow half-way: it arrives at speed

    def test_a_blink_starts_on_and_ends_inside_its_time(self):
        changes = motion.blink_frames(10, 70, 30, every=0.8, off=0.27)
        self.assertEqual(changes[0], (10, True))
        self.assertEqual([v for _, v in changes], [True, False, True, False, True])
        self.assertTrue(all(10 <= f < 70 for f, _ in changes))

    def test_a_beat_names_a_prop_or_one_of_its_parts(self):
        props = {'mark': ['slash', 'play'], 'app': None}
        self.assertEqual(motion.split_target('mark.slash', props), ('mark', 'slash'))
        self.assertEqual(motion.split_target('app', props), ('app', None))
        for ref, said in (('logo', 'mark, app'), ('mark.dot', 'slash, play'), ('app.screen', 'no parts')):
            with self.assertRaises(ValueError) as ctx:
                motion.split_target(ref, props)
            self.assertIn(said, str(ctx.exception))

    def test_mistakes_in_a_beat_are_named(self):
        props = {'mark': ['slash', 'play'], 'app': None}
        ok = {'at': 1, 'do': 'shear', 'part': 'mark.slash', 'to': -20}
        motion.check_beat(ok, props)
        motion.check_beat({'at': 1, 'do': 'hit', 'sound': 'clap'}, props, ('clap',))
        for beat, said in (
                ({'at': 1, 'do': 'moonwalk'}, 'shear'),                                  # lists the verbs
                ({**ok, 'part': 'mark'}, 'one part'),
                ({**ok, 'part': 'mark.dot'}, 'slash, play'),
                ({**ok, 'to': 'far'}, 'degrees'),
                ({**ok, 'ease': 'bouncy'}, 'spring'),
                ({**ok, 'foot': 'side'}, 'bottom'),
                ({'at': 1, 'do': 'grow', 'part': 'mark.play', 'to': 1, 'from': 'middle'}, 'bottom_left'),
                ({'at': 1, 'do': 'grow', 'part': 'mark.play', 'to': -1}, 'below 0'),
                ({'at': 1, 'do': 'move', 'part': 'app', 'to': [1, 2]}, 'toward the viewer'),
                ({'at': 1, 'do': 'press', 'part': 'app'}, 'depth'),
                ({'at': 1, 'do': 'blink', 'part': 'mark.play'}, 'how long'),
                ({'at': 1, 'do': 'light', 'light': 'sun', 'to': 1}, 'key, fill, rim'),
                ({'at': 1, 'do': 'sweep', 'from': 'above'}, 'left'),
                ({'at': 1, 'do': 'hit', 'sound': 'bang'}, 'clap'),
                ({'at': 1, 'do': 'sound'}, 'name of a sound')):
            with self.assertRaises(SpecError, msg=beat) as ctx:
                motion.check_beat(beat, props, ('clap',))
            self.assertIn(said, str(ctx.exception))
            self.assertIn('beat at 1', str(ctx.exception))

    def test_a_glow_past_the_colours_limit_is_noted(self):
        self.assertIsNone(motion.glow_note({'at': 1, 'do': 'glow', 'to': 1.0}))
        self.assertIn('clips', motion.glow_note({'at': 1, 'do': 'glow', 'to': 2.2}))


def click(path, at=0.2, seconds=0.5, rate=44100):
    """A WAV that is silent but for a short burst at `at` seconds."""
    n, i0 = int(seconds * rate), int(at * rate)
    with wave.open(str(path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b''.join((20000 if i0 <= i < i0 + 400 and i % 2 else -20000 if i0 <= i < i0 + 400 else 0)
                               .to_bytes(2, 'little', signed=True) for i in range(n)))


def samples(path):
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return [int.from_bytes(raw[i:i + 2], 'little', signed=True) for i in range(0, len(raw), 2)]


@unittest.skipUnless(FFMPEG, 'ffmpeg reads the recordings')
class SoundTests(Folder):
    def events(self, events, sounds, duration=2.0, mix=None):
        path = self.folder / 'events.json'
        path.write_text(json.dumps({'fps': 30, 'duration': duration, 'events': events, 'score': [],
                                    'sounds': sounds, 'mix': mix or {}}))
        return path

    def test_a_recordings_loudest_moment_lands_on_the_events_frame(self):
        click(self.folder / 'c.wav', at=0.2)
        ev = self.events([{'t': 1.0, 'kind': 'clap', 'frame': 31}], {'clap': [{'file': str(self.folder / 'c.wav')}]})
        sound.render(ev, self.folder / 'out.wav')
        xs = samples(self.folder / 'out.wav')
        loud = [i for i, x in enumerate(xs) if abs(x) > 5000]
        middle = (loud[0] + loud[-1]) / 2 / sound.SR
        self.assertAlmostEqual(middle, 1.0, delta=0.012)       # not 1.2: the file's own lead-in is taken off
        self.assertEqual(max(abs(x) for x in xs[: int(0.5 * sound.SR)]), 0)   # and no room tone under it

    def test_a_sound_is_the_same_level_in_every_clip(self):
        click(self.folder / 'c.wav', at=0.05)
        sounds = {'clap': [{'file': str(self.folder / 'c.wav'), 'gain': 0.5}],
                  'hum': [{'tones': [220.0], 'gain': 0.1, 'attack': 0.01, 'decay': 0.3}]}
        ev = self.events([{'t': 0.5, 'kind': 'clap', 'frame': 16}, {'t': 2.5, 'kind': 'clap', 'frame': 76},
                          {'t': 2.6, 'kind': 'hum', 'frame': 79}], sounds, duration=4.0, mix={'gain': 0.8})
        sound.render(ev, self.folder / 'a.wav', start=0.0, end=2.0, score=False, room=False)
        sound.render(ev, self.folder / 'b.wav', start=2.0, end=4.0, score=False, room=False)
        a, b = samples(self.folder / 'a.wav'), samples(self.folder / 'b.wav')
        self.assertAlmostEqual(max(abs(x) for x in a) / 32767, 0.5 * 0.8 * 20000 / 32767, delta=0.03)
        # the second clip has a hum as well, and its clap is no quieter for it: nothing is normalised
        self.assertGreaterEqual(max(abs(x) for x in b), max(abs(x) for x in a) - 300)

    def test_a_held_tone_is_cut_where_the_beat_ends(self):
        ev = self.events([{'t': 0.0, 'kind': 'hum', 'frame': 1, 'until': 1.0}],
                         {'hum': [{'tones': [220.0], 'gain': 0.3, 'attack': 0.05, 'hold': 5.0, 'decay': 0.5}]})
        sound.render(ev, self.folder / 'out.wav')
        xs = samples(self.folder / 'out.wav')
        held = max(abs(x) for x in xs[int(0.5 * sound.SR): int(0.9 * sound.SR)])
        after = max(abs(x) for x in xs[int(1.5 * sound.SR): int(1.9 * sound.SR)])
        self.assertGreater(held, 5000)
        self.assertLess(after, held * 0.01)

    def test_the_loudest_moment_is_measured_as_sync_check_measures_it(self):
        xs = [0.0] * 44100
        for i in range(22050, 22050 + 441):
            xs[i] = 1.0
        self.assertAlmostEqual(sound.loudest(xs) / sound.SR, 0.5 + 0.005, delta=0.03)


class MarkColourTests(Folder):
    def still(self, slash, play=(255, 255, 255)):
        from PIL import Image, ImageDraw
        im = Image.new('RGB', (400, 300), (12, 17, 31))
        d = ImageDraw.Draw(im)
        d.polygon([(100, 250), (140, 250), (180, 50), (140, 50)], fill=slash)
        d.polygon([(170, 250), (210, 50), (330, 150)], fill=play)
        im.save(self.folder / 'still.png')
        outline = lambda pts: [[[x / 400, y / 300] for x, y in pts]]  # noqa: E731
        return {'frame': 60, 'facing_deg': 3.0, 'parts': [
            {'id': 'slash', 'color': '#3771e0', 'off': 0.0, 'outline': outline([(100, 250), (140, 250), (180, 50), (140, 50)])},
            {'id': 'play', 'color': '#ffffff', 'off': 0.0, 'outline': outline([(170, 250), (210, 50), (330, 150)])}]}

    def test_a_part_in_its_own_colour_passes(self):
        mark = self.still((55, 113, 224))
        self.assertEqual(blender_render.mark_colours(self.folder / 'still.png', mark), [])
        self.assertEqual(mark['parts'][0]['seen'], '#3771e0')
        self.assertEqual(mark['parts'][0]['colour_off'], 0.0)

    def test_a_blue_clipped_toward_cyan_fails_and_says_which_way(self):
        mark = self.still((90, 200, 255))
        said = ' '.join(blender_render.mark_colours(self.folder / 'still.png', mark))
        self.assertIn('slash reads #5ac8ff where the mark has #3771e0', said)
        self.assertIn('lighter or clipped', said)

    def test_a_part_left_in_the_dark_fails_and_says_so(self):
        mark = self.still((55, 113, 224), play=(150, 150, 150))
        said = ' '.join(blender_render.mark_colours(self.folder / 'still.png', mark))
        self.assertIn('play reads', said)
        self.assertIn('darker', said)
        self.assertNotIn('slash', said)


@unittest.skipUnless(FFMPEG, 'ffmpeg makes the test clip')
class HitTests(Folder):
    def clip(self, flash_at):
        """Twenty grey frames, with something appearing on frame `flash_at` (the first frame is number 1)."""
        from PIL import Image, ImageDraw
        for i in range(1, 21):
            im = Image.new('RGB', (160, 90), (20, 20, 30))
            d = ImageDraw.Draw(im)
            d.rectangle([10 + i, 40, 14 + i, 60], fill=(60, 110, 220))  # something moving a little all the way through
            if i >= flash_at:
                d.rectangle([60, 20, 130, 70], fill=(255, 255, 255))
            im.save(self.folder / f'f_{i:04d}.png')
        out = self.folder / 'clip.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-start_number', '1', '-i',
                        str(self.folder / 'f_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '12', str(out)],
                       check=True)
        return out

    def test_a_hit_on_the_frame_of_the_biggest_change_passes(self):
        found = blender_render.hit_frames(self.clip(12), 1, [12])
        self.assertEqual((found[0]['seen'], found[0]['passed']), (12, True))

    def test_a_hit_keyed_early_is_caught(self):
        found = blender_render.hit_frames(self.clip(12), 1, [8])
        self.assertEqual((found[0]['seen'], found[0]['passed']), (12, False))

    def test_a_clip_that_starts_part_way_into_the_film_counts_from_there(self):
        found = blender_render.hit_frames(self.clip(12), 61, [72])
        self.assertTrue(found[0]['passed'])


class MarkSheetTests(Folder):
    MARK = ('<polygon id="slash" points="20,80 32,80 50,20 38,20" fill="#3771e0"/>'
            '<polygon id="play" points="38,80 56,20 85,50" fill="#121418"/>')

    def test_contrast_is_measured_on_every_ground(self):
        parts, box, _ = svg.read(self.svg(self.MARK))
        report, warnings = mark_sheet.measure(parts, box, {'as drawn': parts},
                                              [('light', '#ffffff'), ('dark', '#232d49')])
        self.assertGreater(report['contrast']['#ffffff']['slash'], 3)
        said = ' '.join(warnings)
        self.assertIn('play (#121418) on the dark ground #232d49', said)
        self.assertIn('--on-dark', said)
        self.assertNotIn('#ffffff:', said)

    def test_a_ground_gets_the_first_version_that_reads_on_it(self):
        parts, box, _ = svg.read(self.svg(self.MARK))
        dark = svg.read(self.svg(self.MARK.replace('#3771e0', '#93b4f2').replace('#121418', '#ffffff'), 'd.svg'))[0]
        mono = svg.read(self.svg(self.MARK.replace('#3771e0', '#ffffff').replace('#121418', '#ffffff'), 'w.svg'))[0]
        versions = {'as drawn': parts, 'on dark': dark, 'one colour': mono}
        self.assertEqual(mark_sheet.pick(versions, '#ffffff')[0], 'as drawn')
        self.assertEqual(mark_sheet.pick(versions, '#232d49')[0], 'on dark')
        self.assertEqual(mark_sheet.pick(versions, '#3771e0')[0], 'one colour')
        _, warnings = mark_sheet.measure(parts, box, versions, [('a', '#ffffff'), ('b', '#232d49'), ('c', '#3771e0')])
        self.assertEqual(warnings, [])

    def test_a_part_the_circle_cuts_and_a_part_lost_at_16_px_are_warned_about(self):
        parts, box, _ = svg.read(self.svg(
            '<rect id="corner" x="2" y="2" width="20" height="20" fill="#000"/>'
            '<rect id="dot" x="50" y="50" width="3" height="3" fill="#000"/>'))
        report, warnings = mark_sheet.measure(parts, box, {'as drawn': parts}, [('light', '#ffffff')])
        said = ' '.join(warnings)
        self.assertEqual(report['circle'], ['corner'])
        self.assertIn('a round avatar cuts it', said)
        self.assertIn('dot covers', said)
        self.assertLess(report['at_16px']['dot'], 2)

    def test_small_sizes_are_drawn_with_part_covered_pixels(self):
        parts, box, _ = svg.read(self.svg('<rect x="0" y="0" width="50" height="100" fill="#000"/>', box='0 0 100 100'))
        m = mark_sheet.masks(parts, box, 3)[0]   # half of the middle column is covered
        row = [m.getpixel((x, 1)) for x in range(3)]
        self.assertEqual((row[0], row[2]), (255, 0))
        self.assertAlmostEqual(row[1], 127, delta=10)

    def test_the_tool_writes_a_sheet_and_says_what_it_found(self):
        path = self.svg(self.MARK)
        done = subprocess.run([sys.executable, str(ROOT / 'tools' / 'mark_sheet.py'), '--svg', str(path), '--ground',
                               '#ffffff', '--ground', '#232d49', '--json'], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout)
        self.assertTrue(Path(report['sheet']).exists())
        self.assertTrue(report['warnings'])
        strict = subprocess.run([sys.executable, str(ROOT / 'tools' / 'mark_sheet.py'), '--svg', str(path), '--ground',
                                 '#232d49', '--strict'], capture_output=True, text=True)
        self.assertEqual(strict.returncode, 1)
        bad = subprocess.run([sys.executable, str(ROOT / 'tools' / 'mark_sheet.py'), '--svg', str(path), '--ground', 'navy'],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)
        self.assertIn('#rrggbb', bad.stdout)


@unittest.skipUnless(have_blender(), 'Blender 4.2 or later is not installed')
class BlenderIdentTests(Folder):
    def spec(self, change=None, name='clap.json'):
        spec = json.loads((EXAMPLE / name).read_text())
        spec['set']['props'][0]['file'] = str(EXAMPLE / 'assets' / 'mark-parts.svg')
        for layers in spec.get('sounds', {}).values():
            for layer in layers:
                if 'file' in layer:
                    layer['file'] = str(EXAMPLE / layer['file'])
        spec['look'] = {}  # (bloom is a look, not a check: the tests leave it off)
        if change:
            change(spec)
        return spec

    def check(self, spec, res='480x270', alpha=False):
        (self.folder / 'spec.json').write_text(json.dumps(spec))
        return blender_render.run_compiler(self.folder / 'spec.json', self.folder / 'out', 'check', res, None, alpha=alpha)

    def problems(self, checks):
        """The compiler's issues plus the picture half of the mark test, as tools/blender_render.py check adds it."""
        said = []
        for shot in checks['shots']:
            said += shot['issues']
            if shot.get('mark'):
                said += blender_render.mark_colours(self.folder / 'out' / 'stills' / f'{shot["id"]}.png', shot['mark'])
        return said

    def test_the_example_ends_on_the_flat_mark_in_its_own_colours(self):
        for name in ('clap.json', 'key.json'):
            checks = self.check(self.spec(name=name))
            self.assertTrue(checks['passed'], checks)
            self.assertEqual(self.problems(checks), [], name)
            opening, closing = checks['shots']
            self.assertEqual(opening['mark']['frame'], opening['frames'][1])    # the open is judged on its last frame
            self.assertEqual(closing['mark']['frame'], closing['frames'][0])    # the close on its first
            self.assertEqual([(p['id'], p['off']) for p in opening['mark']['parts']], [('slash', 0.0), ('play', 0.0)])
            self.assertLessEqual(opening['mark']['facing_deg'], 8)

    def test_sounds_are_on_the_frames_the_beats_were_keyed_on(self):
        self.check(self.spec())
        heard = json.loads((self.folder / 'out' / 'events.json').read_text())
        clap = next(e for e in heard['events'] if e['kind'] == 'clap')
        self.assertEqual((clap['frame'], clap['shot'], clap.get('hit')), (21, 'open', True))   # open+0.667 s at 30 fps
        self.assertEqual([e['kind'] for e in heard['events'] if e['shot'] == 'close'], ['hum', 'tap', 'stand'])
        self.assertTrue(Path(heard['sounds']['clap'][0]['file']).is_absolute())

    def test_a_last_frame_that_is_not_the_mark_fails_and_says_what_is_wrong(self):
        def unfinished(spec):  # the bar never finishes its fall, and the caret is never put away
            spec['beats'] = [b for b in spec['beats'] if not (b['do'] == 'hide' and b['at'] == 'open+0.667')
                             and not (b['do'] == 'shear' and b.get('to') in (1.5, 0))]
        said = ' '.join(self.problems(self.check(self.spec(unfinished))))
        self.assertIn('caret is still on screen', said)
        self.assertIn('slash is', said)
        self.assertIn("from where the flat mark has it", said)

        def gone(spec):
            spec['beats'] = [b for b in spec['beats'] if not (b['do'] == 'grow' and b.get('to') == 1)]
        self.assertIn('play is not on screen', ' '.join(self.problems(self.check(self.spec(gone)))))

    def test_a_brand_colour_lost_to_the_light_fails(self):
        def dim(spec):
            spec['set']['lights'] = {'key': 0.15, 'fill': 0.0}
        self.assertIn('darker', ' '.join(self.problems(self.check(self.spec(dim)))))

        def tinted(spec):  # the default view for a film shifts a saturated colour: a logo wants Standard
            spec['look'] = {'view': 'AgX', 'exposure': -0.9}
        self.assertTrue(self.problems(self.check(self.spec(tinted))))

    def test_a_camera_that_does_not_end_square_on_fails(self):
        def askew(spec):
            spec['shots'][0]['az'] = 25
            spec['shots'][0].pop('from')
        self.assertIn('off square', ' '.join(self.problems(self.check(self.spec(askew)))))

    def test_it_holds_in_portrait_and_on_a_transparent_ground(self):
        tall = self.check(self.spec(), res='270x480')
        self.assertTrue(tall['passed'], tall)
        self.assertEqual(self.problems(tall), [])
        clear = self.check(self.spec(), alpha=True)
        self.assertTrue(clear['alpha'] and clear['passed'], clear)
        from PIL import Image
        still = Image.open(self.folder / 'out' / 'stills' / 'open.png')
        self.assertEqual(still.mode, 'RGBA')
        self.assertEqual(still.getpixel((5, 5))[3], 0)                       # no stage behind it
        self.assertEqual(self.problems(clear), [])                           # and the colours still hold

    def test_any_svg_becomes_a_logo_and_a_whole_prop_can_be_moved(self):
        path = self.svg('<path id="o" d="M10 10 H90 V90 H10 Z M30 30 h40 v40 h-40 z" fill="#e0457b"/>'
                        '<circle id="dot" cx="50" cy="50" r="12" fill="#ffffff"/>')
        spec = {'fps': 30, 'set': {'kind': 'studio', 'props': [
                    {'id': 'logo', 'kind': 'logo', 'file': str(path), 'material': 'gloss', 'parts': {'dot': {'material': 'flat'}}},
                    {'id': 'ui', 'kind': 'card', 'at': [0.5, 0]}]},
                'shots': [{'id': 'A', 'dur': 1.5, 'type': 'ident', 'subject': 'logo'},
                          {'id': 'B', 'dur': 1, 'type': 'product', 'subject': 'ui'}],
                'beats': [{'at': 0, 'do': 'turn', 'part': 'logo.o', 'to': 90, 'hinge': 'left'},
                          {'at': 0.2, 'dur': 0.6, 'do': 'turn', 'part': 'logo.o', 'to': 0, 'ease': 'spring'},
                          {'at': 0, 'do': 'grow', 'part': 'logo.dot', 'to': 0},
                          {'at': 0.5, 'dur': 0.4, 'do': 'grow', 'part': 'logo.dot', 'to': 1},
                          {'at': 'B', 'dur': 0.5, 'do': 'move', 'part': 'ui', 'to': [0, 0.05, 0]},
                          {'at': 'B+0.5', 'dur': 0.2, 'do': 'press', 'part': 'ui', 'depth': 0.02}]}
        checks = self.check(spec)
        self.assertTrue(checks['passed'], checks)
        mark = checks['shots'][0]['mark']
        self.assertEqual([p['id'] for p in mark['parts']], ['o', 'dot'])
        self.assertEqual(len(mark['parts'][0]['outline']), 2)                # the hole is kept
        self.assertEqual(self.problems(checks), [])                          # a flat part reads as its colour exactly

    def test_mistakes_in_an_ident_spec_are_named(self):
        for change, said in (
                (lambda s: s['beats'].append({'at': 1, 'do': 'moonwalk'}), 'shear'),
                (lambda s: s['beats'].append({'at': 1, 'do': 'shear', 'part': 'mark.dot', 'to': 1}), 'slash, play, caret'),
                (lambda s: s['set']['props'][0].update(extras=['dot']), 'no part called'),
                (lambda s: s['set']['props'][0].update(material='chrome'), 'satin'),
                (lambda s: s['set']['props'][0].update(file='gone.svg'), 'not found'),
                (lambda s: s['set']['props'][0].update(base={'kind': 'plinth'}), 'key'),
                (lambda s: s['set'].update(stage='moon'), 'sweep, dark'),
                (lambda s: s['set'].update(lights={'sun': 1}), 'key, fill, rim'),
                (lambda s: s['shots'][0].update(mark='middle'), '"start", "end" or "none"'),
                (lambda s: s['shots'][0].update({'from': {'height': 3}}), 'az'),
                (lambda s: s['sounds'].update(clap=[{'file': 'gone.mp3'}]), 'recording not found'),
                (lambda s: s['sounds'].update(clap=[{'gain': 1}]), '"file"'),
                (lambda s: s['beats'].append({'at': 1, 'do': 'hit', 'sound': 'bang'}), 'clap')):
            with self.assertRaises(RuntimeError, msg=said) as ctx:
                self.check(self.spec(change))
            self.assertIn(said, str(ctx.exception))
            self.assertNotIn('Traceback', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
