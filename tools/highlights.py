#!/usr/bin/env python3
"""Assemble transcript-ranked highlights with complete sentences and accurate FFmpeg cuts."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from media_ops import (
    load_json,
    number,
    output_path,
    probe,
    run,
    sha256,
    staged_output,
    video_stream,
    write_json,
)
from transcript import segments, transcribe


def tokens(text):
    return re.findall(r'\w+', text.casefold())


def sentences(transcript):
    units = []
    pending = []
    for segment in segments(transcript):
        words = segment['words'] or [{k: segment[k] for k in ('text', 'start', 'end')}]
        for word in words:
            pending.append(word)
            if re.search(r'[.!?。！？]["\'”’]*$', word['text']):
                units.append({'text': ' '.join(w['text'] for w in pending), 'words': pending})
                pending = []
    return units, pending


def select(transcript, duration, *, budget=60., keywords=(), padding=.12, max_pause=.7, dedup=.85):
    number(duration, 'source duration', .001)
    number(budget, 'budget', .1, 36000)
    number(padding, 'padding', 0, 1)
    number(max_pause, 'max pause', .1, 10)
    number(dedup, 'dedup similarity', .5, 1)
    units, incomplete = sentences(transcript)
    if not units:
        raise ValueError('No complete punctuated sentences found. Supply a punctuated transcript.')
    all_words = [word for unit in units for word in unit['words']] + incomplete
    previous = 0
    for word in all_words:
        if word['start'] < previous - .001:
            raise ValueError('Resolve overlapping speech timing before selecting highlights')
        if word['end'] > duration + .001:
            raise ValueError('Transcript extends beyond the source media')
        previous = word['end']
    starts = {word['start']: i for i, word in enumerate(all_words)}
    ends = {word['end']: i for i, word in enumerate(all_words)}
    def padded(start, end):
        left_index, right_index = starts[start], ends[end]
        before = all_words[left_index - 1]['end'] if left_index else 0
        after = all_words[right_index + 1]['start'] if right_index + 1 < len(all_words) else duration
        return [max(0, start - padding, before), min(duration, end + padding, after)]
    vocabulary = Counter(t for unit in units for t in set(tokens(unit['text'])))
    wanted = {t for term in keywords for t in tokens(term)}
    candidates, seen = [], []
    for unit in units:
        terms = tokens(unit['text'])
        normalized = ' '.join(terms)
        if not terms or any(SequenceMatcher(None, normalized, earlier).ratio() >= dedup for earlier in seen):
            continue
        seen.append(normalized)
        words = unit['words']
        ranges = []
        start, end = words[0]['start'], words[0]['end']
        for word in words[1:]:
            if word['start'] - end > max_pause:
                ranges.append(padded(start, end))
                start = word['start']
            end = word['end']
        ranges.append(padded(start, end))
        length = sum(b - a for a, b in ranges)
        score = sum(1 / vocabulary[t] for t in set(terms)) / max(1, len(terms))
        score += len(wanted.intersection(terms)) * 3
        candidates.append({**unit, 'ranges': ranges, 'durationSeconds': length, 'score': round(score, 6)})
    selected, used = [], 0
    for unit in sorted(candidates, key=lambda u: (-u['score'], u['words'][0]['start'])):
        if used + unit['durationSeconds'] <= budget + .000001:
            selected.append(unit)
            used += unit['durationSeconds']
    selected.sort(key=lambda u: u['words'][0]['start'])
    if not selected:
        raise ValueError('No complete sentence fits the requested duration budget')
    # Adjacent sentence padding may overlap. Merge source ranges without repeating frames.
    cuts = []
    for unit in selected:
        for start, end in unit['ranges']:
            if cuts and start <= cuts[-1]['endSeconds']:
                cuts[-1]['endSeconds'] = max(cuts[-1]['endSeconds'], end)
            else:
                cuts.append({'startSeconds': start, 'endSeconds': end})
    offset = 0
    for cut in cuts:
        cut['outputStartSeconds'] = offset
        offset += cut['endSeconds'] - cut['startSeconds']
    if len(cuts) > 128:
        raise ValueError('More than 128 cuts selected. Use a shorter budget.')
    return {'version': 1, 'durationSeconds': offset, 'segments': cuts,
            'sentences': [{'text': u['text'], 'score': u['score'], 'sourceStartSeconds': u['words'][0]['start']} for u in selected],
            'omittedIncompleteTail': bool(incomplete), 'ranking': 'keyword matches and uncommon transcript terms'}


def render(video, output, plan):
    info = probe(video)
    video_stream(info)
    output_path(output, video)
    has_audio = any(s['codec_type'] == 'audio' for s in info['streams'])
    filters, joined = [], []
    for index, cut in enumerate(plan['segments']):
        start, end = cut['startSeconds'], cut['endSeconds']
        filters.append(f'[0:v:0]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]')
        joined.append(f'[v{index}]')
        if has_audio:
            filters.append(f'[0:a:0]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]')
            joined.append(f'[a{index}]')
    filters.append(''.join(joined) + f'concat=n={len(plan["segments"])}:v=1:a={int(has_audio)}[v]' + ('[a]' if has_audio else ''))
    with staged_output(output) as target:
        command = ['ffmpeg', '-y', '-v', 'error', '-i', str(Path(video).resolve()), '-filter_complex', ';'.join(filters), '-map', '[v]']
        if has_audio:
            command += ['-map', '[a]', '-c:a', 'aac']
        run([*command, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', target])


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--transcript', type=Path)
    source.add_argument('--transcribe', action='store_true', help='Run installed OpenAI Whisper locally; may download model weights')
    parser.add_argument('--model', default='base')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--keyword', action='append', default=[])
    parser.add_argument('--max-pause', type=float, default=.7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-plan', type=Path, help='Explicitly write the edit decision JSON, even in preview')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        info = probe(args.video)
        output_path(args.output, args.video, args.transcript)
        transcript = load_json(args.transcript) if args.transcript else transcribe(args.video, args.model)
        plan = select(transcript, info['duration'], budget=args.seconds, keywords=args.keyword, max_pause=args.max_pause)
        plan.update(source=str(args.video.resolve()), sourceSha256=sha256(args.video), output=str(args.output.resolve()), executed=args.run)
        if args.save_plan:
            output_path(args.save_plan, args.video, args.transcript, args.output)
            write_json(args.save_plan, plan)
        if args.run:
            render(args.video, args.output, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
