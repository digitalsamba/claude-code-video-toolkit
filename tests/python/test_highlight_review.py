import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from highlights import main, select
from cutdown import parse_segment
from media_ops import staged_output


class ReviewRegressionTests(unittest.TestCase):
    def test_last_take_and_first_override(self):
        words = [{'text': ' Result.', 'startMs': 100, 'endMs': 400},
                 {'text': ' Result.', 'startMs': 600, 'endMs': 900}]
        self.assertEqual(select(words, 2)['sentences'][0]['sourceStartSeconds'], .6)
        self.assertEqual(select(words, 2, keep='first')['sentences'][0]['sourceStartSeconds'], .1)
        self.assertEqual(select(words, 2)['sentences'][0]['text'], 'Result.')

    def test_zero_length_words_use_indices_for_padding(self):
        words = [{'text': 'First.', 'startMs': 100, 'endMs': 500},
                 {'text': ' Zero', 'startMs': 500, 'endMs': 500},
                 {'text': ' keeper.', 'startMs': 500, 'endMs': 900},
                 {'text': ' Next.', 'startMs': 900, 'endMs': 1200}]
        plan = select(words, 2, budget=.5, keywords=['keeper'])
        self.assertEqual(plan['sentences'][0]['text'], 'Zero keeper.')
        self.assertEqual(plan['segments'][0]['startSeconds'], .5)
        self.assertEqual(plan['segments'][0]['endSeconds'], .9)
        self.assertEqual(plan['cutdownArgs'], ['--segment', '0.5-0.9'])

    def test_cutdown_arguments_avoid_scientific_notation(self):
        plan = select([{'text': 'Keep.', 'startMs': 120.0002, 'endMs': 900}], 2)
        argument = plan['cutdownArgs'][1]
        self.assertNotIn('e-', argument)
        segment = parse_segment(argument)
        self.assertAlmostEqual(segment.start, .0000002, delta=1e-9)

    def test_atomic_replace_overwrites_without_hardlinks_and_preserves_failed_output(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'output.txt'
            target.write_text('old')
            with patch.object(os, 'link', side_effect=OSError('hardlinks unsupported')):
                with staged_output(target) as temporary:
                    self.assertEqual(temporary.parent.resolve(), target.parent.resolve())
                    temporary.write_text('new')
            self.assertEqual(target.read_text(), 'new')
            with self.assertRaises(RuntimeError):
                with staged_output(target) as temporary:
                    temporary.write_text('partial')
                    raise RuntimeError('render failed')
            self.assertEqual(target.read_text(), 'new')
            self.assertEqual(list(Path(folder).iterdir()), [target])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_default_cli_renders_twice_and_dry_run_writes_nothing(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video, output, transcript, plan = [folder / name for name in ('input.mp4', 'cuts.mp4', 'words.json', 'plan.json')]
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=160x90:r=20:d=2', '-c:v', 'libx264', str(video)], check=True, capture_output=True)
            transcript.write_text(json.dumps([{'text': ' Done.', 'startMs': 100, 'endMs': 900}]))
            args = ['--video', str(video), '--transcript', str(transcript), '--output', str(output), '--save-plan', str(plan)]
            for _ in range(2):
                with redirect_stdout(StringIO()):
                    self.assertEqual(main(args), 0)
                self.assertGreater(output.stat().st_size, 0)
                self.assertTrue(json.loads(plan.read_text())['executed'])
            before = {p: p.read_bytes() for p in folder.iterdir()}
            with redirect_stdout(StringIO()):
                self.assertEqual(main([*args, '--dry-run']), 0)
            self.assertEqual(before, {p: p.read_bytes() for p in folder.iterdir()})
