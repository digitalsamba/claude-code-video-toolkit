#!/usr/bin/env python3
"""Export word caption JSON to downloadable SRT or WebVTT subtitles."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from production import atomic_write


def export_subtitles(words: object, *, format: str = 'srt', max_words: int = 7) -> str:
    if format not in ('srt', 'vtt') or not 1 <= max_words <= 30 or not isinstance(words, list):
        raise ValueError('Use srt/vtt, a word limit of 1–30 and a caption array')
    validated = []
    previous = -1
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get('text'), str):
            raise ValueError('Caption words require text')
        start, end = word.get('startMs'), word.get('endMs')
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (start, end)) or not 0 <= start < end or start < previous:
            raise ValueError('Caption times must be finite, ordered and positive in duration')
        previous = end
        validated.append(word)
    def timestamp(value):
        millis = round(value)
        seconds, millis = divmod(millis, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:02}:{minutes:02}:{seconds:02}{"," if format == "srt" else "."}{millis:03}'
    blocks = []
    for index in range(0, len(validated), max_words):
        group = validated[index:index + max_words]
        content = ' '.join(word['text'].strip() for word in group).replace('-->', '→')
        content = content.replace('\r', ' ').replace('\n', ' ').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        blocks.append(f'{len(blocks) + 1}\n{timestamp(group[0]["startMs"])} --> {timestamp(group[-1]["endMs"])}\n{content}')
    return ('WEBVTT\n\n' if format == 'vtt' else '') + '\n\n'.join(blocks) + '\n'


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=['srt', 'vtt'], default='srt')
    parser.add_argument('--max-words', type=int, default=7)
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError('Output cannot replace caption JSON')
        content = export_subtitles(json.loads(args.input.read_text(encoding='utf-8')), format=args.format, max_words=args.max_words)
        atomic_write(args.output, content)
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
