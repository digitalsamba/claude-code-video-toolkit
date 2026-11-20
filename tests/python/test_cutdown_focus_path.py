"""A crop focus that moves across a segment: --segment START-END@0.1,0.9."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import cutdown
from cutdown import Source, focus_x, parse_segment, reframe_filter


def colour_at(video: Path, seconds: float) -> tuple[int, int, int]:
    """The middle pixel of the frame at a time."""
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{seconds}', '-i', str(video), '-frames:v', '1',
                          '-vf', 'crop=2:2:iw/2:ih/2,scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
                         check=True, capture_output=True).stdout
    return out[0], out[1], out[2]


class ParseTests(unittest.TestCase):
    def test_one_value_is_a_fixed_focus(self):
        seg = parse_segment('4-11@0.25')
        self.assertEqual((seg.start, seg.end, seg.focus, seg.path), (4.0, 11.0, 0.25, None))

    def test_several_values_are_a_path(self):
        seg = parse_segment('0:04-0:11@0.57,0.7,0.4')
        self.assertEqual(seg.path, (0.57, 0.7, 0.4))
        self.assertEqual(seg.focus, 0.57)

    def test_a_value_outside_the_frame_is_refused(self):
        with self.assertRaises(ValueError):
            parse_segment('4-11@0.5,1.4')


class FilterTests(unittest.TestCase):
    def test_fixed_focus_is_a_number_and_a_path_is_an_expression_of_t(self):
        self.assertEqual(focus_x(200, 0.25), '50')
        expr = focus_x(200, 0.0, (0.0, 1.0, 0.5), 4.0)
        self.assertTrue(expr.startswith("'if(lt(t,2.0000),0.00+(200.00)*(t-0.0000)/2.0000,"))
        self.assertTrue(expr.endswith("100.00))'"))

    def test_only_a_path_changes_the_crop_chain(self):
        src = Source(320, 180, 2, '10/1', False, None)
        fixed = reframe_filter(src, 90, 160, 'crop', 0.5, 1.0, 'black')
        self.assertEqual(fixed, reframe_filter(src, 90, 160, 'crop', 0.5, 1.0, 'black', None, 2.0))
        self.assertIn("'if(lt(t,", reframe_filter(src, 90, 160, 'crop', 0.1, 1.0, 'black', (0.1, 0.9), 2.0))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.video = Path(self.folder.name) / 'halves.mp4'
        # red on the left half, blue on the right, 2 s
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=red:s=160x180:r=10:d=2', '-f', 'lavfi', '-i',
                        'color=blue:s=160x180:r=10:d=2', '-filter_complex', 'hstack', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        str(self.video)], check=True, capture_output=True)

    def tearDown(self):
        self.folder.cleanup()

    def test_the_crop_moves_from_the_left_half_to_the_right(self):
        out = Path(self.folder.name) / 'moved.mp4'
        self.assertEqual(cutdown.main(['-i', str(self.video), '--mode', 'crop', '-f', '9:16', '--segment', '0-2@0.1,0.9', '-o', str(out)]), 0)
        first, last = colour_at(out, 0.0), colour_at(out, 1.85)
        self.assertGreater(first[0], 150)
        self.assertLess(first[2], 80)
        self.assertGreater(last[2], 150)
        self.assertLess(last[0], 80)

    def test_a_fixed_focus_stays_where_it_is(self):
        out = Path(self.folder.name) / 'fixed.mp4'
        self.assertEqual(cutdown.main(['-i', str(self.video), '--mode', 'crop', '-f', '9:16', '--segment', '0-2@0.1', '-o', str(out)]), 0)
        self.assertGreater(colour_at(out, 1.85)[0], 150)

    def test_track_refuses_a_moving_focus(self):
        self.assertEqual(cutdown.main(['-i', str(self.video), '--mode', 'track', '--segment', '0-2@0.1,0.9', '--dry-run']), 1)


if __name__ == '__main__':
    unittest.main()
