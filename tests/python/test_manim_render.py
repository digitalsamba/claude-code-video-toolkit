import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'lib' / 'manim'))
import manim_render
from core import Brand, Voice

WORDS = [('A', 0, 60), (' request', 80, 300), (' asks', 400, 600), (' the', 620, 700), (' cache,', 720, 1000),
         (' then', 1100, 1300), (' the', 1320, 1400), (' Cache', 1420, 1700), (' again.', 1720, 2100)]


def captions(root):
    path = root / 'v.captions.json'
    path.write_text(json.dumps([{'text': t, 'startMs': a, 'endMs': b} for t, a, b in WORDS]))
    return path


class CueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.voice = Voice(captions(Path(self.tmp.name)))

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_phrase_starts_where_its_first_word_does(self):
        self.assertAlmostEqual(self.voice.cue('the cache'), 0.62)

    def test_punctuation_and_case_are_ignored(self):
        self.assertAlmostEqual(self.voice.cue('CACHE', nth=2), 1.42)

    def test_lead_starts_before_the_word(self):
        self.assertAlmostEqual(self.voice.cue('again', lead=0.3), 1.42)

    def test_a_phrase_that_is_not_said_raises(self):
        with self.assertRaises(KeyError):
            self.voice.cue('database')
        with self.assertRaises(KeyError):
            self.voice.cue('cache', nth=3)

    def test_end_is_the_last_word(self):
        self.assertAlmostEqual(self.voice.end, 2.1)

    def test_no_captions_is_an_error_that_says_what_to_pass(self):
        with mock.patch.dict('os.environ', {}, clear=False) as env:
            env.pop('MANIM_TOOLKIT_CAPTIONS', None)
            with self.assertRaisesRegex(FileNotFoundError, '--captions'):
                Voice()


class BrandTests(unittest.TestCase):
    def test_colours_and_mark_come_from_the_brand(self):
        brand = Brand('video-toolkit')
        self.assertEqual(brand.c('primary'), '#3771e0')
        self.assertTrue(brand.mark().name == 'mark.svg' and brand.mark().exists())
        self.assertEqual(brand.c('nope', '#000000'), '#000000')
        with self.assertRaisesRegex(KeyError, 'primary'):
            brand.c('nope')

    def test_an_unknown_brand_raises(self):
        with self.assertRaises(FileNotFoundError):
            Brand('no-such-brand')


class PlanTests(unittest.TestCase):
    def test_size(self):
        self.assertEqual(manim_render.parse_size('1080x1920'), (1080, 1920))
        for bad in ('1080', '1081x1920', 'wide'):
            with self.assertRaises(Exception):
                manim_render.parse_size(bad)

    def test_an_opaque_clip_is_mp4_and_a_transparent_one_is_webm_or_mov(self):
        self.assertEqual(manim_render.output_plan(Path('a/b.mp4'), False)['events'], Path('a/b.events.json'))
        self.assertEqual(manim_render.output_plan(Path('b.webm'), True)['kind'], 'webm')
        self.assertEqual(manim_render.output_plan(Path('b.mov'), True)['kind'], 'mov')
        for out, alpha in ((Path('b.webm'), False), (Path('b.mp4'), True)):
            with self.assertRaises(SystemExit):
                manim_render.output_plan(out, alpha)

    def test_the_command_always_sets_fps_and_size(self):
        cmd = manim_render.manim_command('manim', Path('s.py'), 'Flow', (1080, 1920), 30, True, Path('/tmp/m'))
        self.assertEqual(cmd[cmd.index('--fps') + 1], '30')
        self.assertEqual(cmd[cmd.index('-r') + 1], '1080,1920')
        self.assertIn('-t', cmd)
        self.assertEqual(cmd[-2:], ['s.py', 'Flow'])


class DoctorTests(unittest.TestCase):
    def test_without_manim_it_fails_and_says_how_to_install(self):
        out = io.StringIO()
        with mock.patch.object(manim_render, 'find_manim', return_value=None), redirect_stdout(out):
            code = manim_render.doctor(as_json=True)
        report = json.loads(out.getvalue())
        self.assertEqual(code, 1)
        self.assertFalse(report['ok'])
        self.assertIn('Install', next(c for c in report['checks'] if c['check'] == 'manim')['detail'])

    def test_render_without_manim_exits_2(self):
        with mock.patch.object(manim_render, 'find_manim', return_value=None):
            self.assertEqual(manim_render.main(['render', 'x.py', 'X', '--out', 'x.mp4']), 2)


@unittest.skipUnless(shutil.which('manim') and shutil.which('ffmpeg'), 'Manim is optional and not installed')
class RenderTests(unittest.TestCase):
    def test_the_typed_ident_renders_with_its_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'typed.mp4'
            code = manim_render.main(['render', str(ROOT / 'examples/manim-ident/idents.py'), 'Typed', '--brand', 'video-toolkit',
                                      '--size', '480x270', '--out', str(out), '--json'])
            self.assertEqual(code, 0)
            events = json.loads(out.with_suffix('.events.json').read_text())
            self.assertEqual([e['what'] for e in events], ['key'] * 6 + ['fold:0.55', 'land'])
            gaps = [round(b['at'] - a['at'], 3) for a, b in zip(events, events[1:6])]
            self.assertEqual(gaps, [0.1] * 5)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'csv=p=0', str(out)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertEqual(probe, '480,270,30/1')

    def test_a_transparent_render_is_a_webm_with_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'typed.webm'
            self.assertEqual(manim_render.main(['render', str(ROOT / 'examples/manim-ident/idents.py'), 'Typed', '--brand', 'video-toolkit',
                                                '--size', '480x270', '--alpha', '--out', str(out), '--json']), 0)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name:stream_tags=alpha_mode', '-of', 'csv=p=0', str(out)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertIn('vp9', probe)
            self.assertIn('1', probe.split(',')[1:])

    def test_a_scene_that_overruns_a_cue_fails_the_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'late.py'
            scene.write_text('from toolkit_manim import *\nclass Late(ToolkitScene):\n    def construct(self):\n'
                             '        self.play(FadeIn(Square()), run_time=1.0)\n        self.at(0.5)\n')
            self.assertEqual(manim_render.main(['render', str(scene), 'Late', '--size', '480x270', '--out', str(Path(tmp) / 'late.mp4')]), 1)


if __name__ == '__main__':
    unittest.main()
