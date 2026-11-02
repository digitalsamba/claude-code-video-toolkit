#!/usr/bin/env python3
"""Duck music under speech and place sound effects using FFmpeg. Preview by default."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from media_ops import (
    audio_stream,
    load_json,
    number,
    output_path,
    probe,
    run,
    staged_output,
    video_stream,
)


def mix(video, music, output, *, speech=None, events=None, music_gain=.25,
        speech_gain=1., threshold=.03, ratio=8., attack=20., release=400., execute=False):
    source = probe(video)
    video_stream(source)
    voice = probe(speech or video)
    audio_stream(voice)
    audio_stream(probe(music))
    output = output_path(output, video, speech, music)
    duration = source['duration']
    for name, value, low, high in [('music gain', music_gain, 0, 4), ('speech gain', speech_gain, 0, 4),
                                  ('threshold', threshold, .000976, 1), ('ratio', ratio, 1, 20),
                                  ('attack', attack, .01, 2000), ('release', release, .01, 9000)]:
        number(value, name, low, high)
    if events is None:
        events = []
    if not isinstance(events, list) or len(events) > 128:
        raise ValueError('events must be an array of at most 128 sound effects')
    command = ['ffmpeg', '-y', '-v', 'error', '-i', str(Path(video).resolve()),
               '-i', str(Path(speech or video).resolve()), '-stream_loop', '-1', '-i', str(Path(music).resolve())]
    filters = [f'[1:a:0]apad,atrim=duration={duration},asplit=2[voice][control]',
               f'[2:a:0]volume={music_gain},atrim=duration={duration}[music]',
               f'[music][control]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[ducked]',
               f'[voice]volume={speech_gain}[spoken]']
    labels = ['[spoken]', '[ducked]']
    normalized = []
    for i, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get('path'), str):
            raise ValueError('Each sound effect needs path and atSeconds')
        when = number(event.get('atSeconds'), 'sound effect atSeconds', 0, duration - .001)
        gain = number(event.get('gain', 1), 'sound effect gain', 0, 4)
        audio_stream(probe(event['path']))
        if Path(event['path']).resolve() == output:
            raise ValueError('Output cannot replace an effect input')
        command += ['-i', str(Path(event['path']).resolve())]
        filters.append(f'[{i + 3}:a:0]volume={gain},adelay={round(when * 1000)}:all=1[fx{i}]')
        labels.append(f'[fx{i}]')
        normalized.append({**event, 'atSeconds': when, 'gain': gain})
    filters.append(''.join(labels) + f'amix=inputs={len(labels)}:normalize=0:duration=first,alimiter=limit=0.95:latency=1[out]')
    command += ['-filter_complex', ';'.join(filters), '-map', '0:v:0', '-map', '[out]', '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k', '-t', str(duration)]
    result = {'version': 1, 'executed': execute, 'output': str(output), 'durationSeconds': duration,
              'ducking': {'threshold': threshold, 'ratio': ratio, 'attackMs': attack, 'releaseMs': release},
              'events': normalized, 'command': [*command, str(output)]}
    if execute:
        with staged_output(output) as target:
            run([*command, target])
    return result


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--music', required=True, type=Path)
    parser.add_argument('--speech', type=Path, help='Isolated narration; otherwise use video audio')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--events', type=Path, help='JSON array of {path, atSeconds, gain}; paths relative to this file')
    parser.add_argument('--music-gain', type=float, default=.25)
    parser.add_argument('--speech-gain', type=float, default=1)
    parser.add_argument('--threshold', type=float, default=.03)
    parser.add_argument('--ratio', type=float, default=8)
    parser.add_argument('--attack', type=float, default=20)
    parser.add_argument('--release', type=float, default=400)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        events = load_json(args.events) if args.events else None
        if isinstance(events, list):
            events = [{**e, 'path': str(args.events.parent / e['path'])} if isinstance(e, dict) and isinstance(e.get('path'), str) else e for e in events]
        result = mix(args.video, args.music, args.output, speech=args.speech, events=events,
                     music_gain=args.music_gain, speech_gain=args.speech_gain, threshold=args.threshold,
                     ratio=args.ratio, attack=args.attack, release=args.release, execute=args.run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
