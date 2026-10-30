#!/usr/bin/env python3
"""Create a new, entirely local production demo with synthetic test media."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from media_ops import encode_frames, run


def create(project):
    project = Path(project).resolve()
    if project.exists():
        raise ValueError('Choose a new project directory. This example preserves earlier runs.')
    inputs = project / 'input'
    inputs.mkdir(parents=True)
    def save(name, data):
        (inputs / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    texture = np.random.default_rng(12).integers(30, 240, (40, 40, 3), dtype=np.uint8)
    def frames():
        for frame in range(40):
            pixels = np.zeros((180, 320, 3), dtype=np.uint8)
            pixels[:, :, 1] = 255
            x = 30 + frame * 3
            pixels[70:110, x:x + 40] = texture
            yield pixels
    encode_frames(frames(), inputs / 'silent.mp4', width=320, height=180, fps=10)
    run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
         "sine=frequency=440:duration=4,volume=enable='lt(t,0.2)+between(t,1.5,2.2)+gte(t,3.5)':volume=0", inputs / 'narration.wav'])
    run(['ffmpeg', '-v', 'error', '-i', inputs / 'silent.mp4', '-i', inputs / 'narration.wav',
         '-c:v', 'copy', '-c:a', 'aac', inputs / 'source.mp4'])
    for name, frequency, duration in [('speaker-a', 440, 1.1), ('speaker-b', 880, 1.1), ('music', 220, 4), ('click', 1200, .12)]:
        run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'sine=frequency={frequency}:duration={duration}', inputs / f'{name}.wav'])
    Image.new('RGB', (320, 180), '#122345').save(inputs / 'studio.png')
    Image.fromarray(texture[:, :, ::-1]).save(inputs / 'reference.png')
    label = Image.new('RGBA', (70, 22), '#ffd166')
    ImageDraw.Draw(label).text((5, 3), 'Subject', fill='#111827')
    label.save(inputs / 'label.png')
    save('labels.json', [{'path': 'label.png', 'attachToTrack': True, 'offset': [0, -5]}])
    save('effects.json', [{'path': 'click.wav', 'atSeconds': 1.1, 'gain': .7}])
    save('words.json', [{'text': word, 'startMs': start, 'endMs': end} for word, start, end in [
        ('First', 200, 650), ('sentence.', 700, 1500), ('Second', 2200, 2700), ('sentence.', 2800, 3500)]])
    save('dub.json', {'version': 1, 'speakers': {'A': {}, 'B': {}}, 'segments': [
        {'id': 'first', 'speaker': 'A', 'startSeconds': .2, 'endSeconds': 1.5, 'text': 'First sentence.', 'audio': 'speaker-a.wav'},
        {'id': 'second', 'speaker': 'B', 'startSeconds': 2.2, 'endSeconds': 3.5, 'text': 'Second sentence.', 'audio': 'speaker-b.wav'}]})
    (inputs / 'data.csv').write_text('period,series,value\nJan,Teams,10\nFeb,Teams,20\nMar,Teams,35\n', encoding='utf-8')
    save('chart.json', {'version': 1, 'type': 'line', 'dataset': 'data.csv', 'title': 'Synthetic sample data',
                        'unit': 'teams', 'width': 320, 'height': 180, 'fps': 10, 'durationSeconds': 4})
    (inputs / 'source.md').write_text('The example uses synthetic media.\n', encoding='utf-8')
    (inputs / 'script.md').write_text('The example uses synthetic media.\n', encoding='utf-8')
    save('evidence.json', {'version': 1, 'script': 'script.md', 'sources': {'readme': {'path': 'source.md'}},
                           'claims': [{'id': 'synthetic', 'text': 'The example uses synthetic media.',
                                       'source': 'readme', 'quote': 'The example uses synthetic media.'}]})
    save('characters.json', {'version': 1, 'style': 'Bright geometric illustration', 'characters': {'sample': {
        'description': 'A patterned square character', 'wardrobe': 'Fixed multicolor texture', 'references': ['input/reference.png']}},
        'shots': [{'id': 'wave', 'characters': ['sample'], 'prompt': 'On a blue stage', 'motionPrompt': 'Gently move right'}]})
    tasks = []
    def task(name, tool, args, inputs, outputs, depends=(), stdout=None):
        item = {'id': name, 'command': ['{python}', '{toolkit}/tools/' + tool + '.py', *args],
                'inputs': inputs, 'outputs': outputs, 'dependsOn': list(depends), 'estimatedCostUsd': 0,
                'toolkitInputs': ['tools/' + tool + '.py', 'tools/media_ops.py'], 'timeoutSeconds': 180}
        if stdout:
            item['stdoutFile'] = stdout
        tasks.append(item)
    p = '{project}/'
    task('evidence-lock', 'evidence', ['lock', '--spec', p + 'input/evidence.json', '--output', p + 'evidence.lock.json'],
         ['input/evidence.json', 'input/source.md', 'input/script.md'], ['evidence.lock.json'])
    task('evidence-check', 'evidence', ['check', '--spec', p + 'evidence.lock.json', '--strict'],
         ['evidence.lock.json', 'input/source.md', 'input/script.md'], [], ['evidence-lock'], 'out/evidence.json')
    task('characters-lock', 'characters', ['lock', '--project', '{project}', '--spec', p + 'input/characters.json', '--output', p + 'characters.lock.json'],
         ['input/characters.json', 'input/reference.png'], ['characters.lock.json'])
    task('characters-preview', 'characters', ['plan', '--project', '{project}', '--lock', p + 'characters.lock.json'],
         ['characters.lock.json', 'input/reference.png'], [], ['characters-lock'], 'out/characters.plan.json')
    task('chart', 'data_video', ['--spec', p + 'input/chart.json', '--narration', p + 'input/narration.wav',
         '--output', p + 'out/chart.mp4', '--save-report', p + 'out/chart.json', '--run'],
         ['input/chart.json', 'input/data.csv', 'input/narration.wav'], ['out/chart.mp4', 'out/chart.json'])
    task('dub', 'dubbing', ['render', '--video', p + 'input/source.mp4', '--spec', p + 'input/dub.json', '--output', p + 'out/dubbed.mp4', '--run'],
         ['input/source.mp4', 'input/dub.json', 'input/speaker-a.wav', 'input/speaker-b.wav'], ['out/dubbed.mp4'])
    task('mix', 'audio_mix', ['--video', p + 'out/dubbed.mp4', '--music', p + 'input/music.wav',
         '--events', p + 'input/effects.json', '--output', p + 'out/mixed.mp4', '--run'],
         ['out/dubbed.mp4', 'input/music.wav', 'input/click.wav', 'input/effects.json'], ['out/mixed.mp4'], ['dub'])
    task('highlights', 'highlights', ['--video', p + 'out/mixed.mp4', '--transcript', p + 'input/words.json', '--seconds', '3.2',
         '--output', p + 'out/highlights.mp4', '--save-plan', p + 'out/edits.json', '--run'],
         ['out/mixed.mp4', 'input/words.json'], ['out/highlights.mp4', 'out/edits.json'], ['mix'])
    task('reframe', 'reframe', ['--video', p + 'input/source.mp4', '--box', '30,70,40,40', '--width', '90', '--height', '160',
         '--output', p + 'out/portrait.mp4', '--run'], ['input/source.mp4'], ['out/portrait.mp4', 'out/portrait.track.json'])
    task('composite', 'composite', ['--foreground', p + 'input/source.mp4', '--background', p + 'input/studio.png', '--key', '#00ff00',
         '--overlays', p + 'input/labels.json', '--track', p + 'out/portrait.track.json', '--output', p + 'out/composite.mp4', '--run'],
         ['input/source.mp4', 'input/studio.png', 'input/label.png', 'input/labels.json', 'out/portrait.track.json'], ['out/composite.mp4'], ['reframe'])
    task('qc', 'qc', ['--render', p + 'out/composite.mp4', '--json'], ['out/composite.mp4'], [], ['composite'], 'out/qc.json')
    # Include imported helpers in content fingerprints, not only the entry script.
    for item in tasks:
        item['toolkitInputs'] += ['tools/transcript.py', 'tools/reframe.py', 'tools/production.py']
    plan = {'version': 1, 'budgetUsd': 0, 'tasks': tasks}
    (project / 'production.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path, help='A new directory; media and plan will be written here')
    args = parser.parse_args()
    create(args.project)
    print(args.project.resolve() / 'production.json')
