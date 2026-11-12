#!/usr/bin/env python3
"""Render reusable CSV/JSON chart videos with Matplotlib, optional narration and provenance."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from media_ops import audio_stream, load_json, number, output_path, probe, sha256, write_json


def dataset(path, columns, default_series='Value'):
    path = Path(path)
    if not isinstance(columns, dict) or set(columns) != {'period', 'series', 'value'} or any(not isinstance(v, str) or not v for v in columns.values()):
        raise ValueError('columns must map period, series and value to column names')
    if len(set(columns.values())) != 3:
        raise ValueError('Column names must be distinct')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV needs unique header names')
            rows = list(reader)
    elif path.suffix.lower() == '.json':
        rows = load_json(path)
    else:
        raise ValueError('Dataset must be .csv or a JSON array of row objects')
    if not isinstance(rows, list) or not rows or len(rows) > 12000:
        raise ValueError('Dataset needs 1 to 12000 rows')
    if all(isinstance(row, list) and len(row) == 2 for row in rows):
        rows = [{columns['period']: row[0], columns['series']: default_series, columns['value']: row[1]} for row in rows]
    periods, names, cells = [], [], {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(key not in row for key in columns.values()):
            raise ValueError(f'Row {index} is missing required columns')
        period, name, value = [row[columns[key]] for key in ('period', 'series', 'value')]
        if not isinstance(period, (str, int, float)) or isinstance(period, bool):
            raise ValueError('Period labels must be strings or numbers')
        if isinstance(period, (int, float)):
            number(period, 'period label')
        period = str(period).strip()
        if not period or not isinstance(name, str) or not name.strip():
            raise ValueError('Period and series labels cannot be empty')
        name = name.strip()
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as error:
                raise ValueError(f'Row {index} value must be numeric') from error
        number(value, f'row {index} value')
        if (period, name) in cells:
            raise ValueError(f'Duplicate value for {period}/{name}')
        cells[period, name] = value
        if period not in periods:
            periods.append(period)
        if name not in names:
            names.append(name)
    if len(periods) > 500 or len(names) > 24:
        raise ValueError('Use at most 500 periods and 24 series per chart')
    if len(cells) != len(periods) * len(names):
        raise ValueError('Every series needs a value in every period. Fill missing values explicitly.')
    return {'periods': periods, 'series': names, 'values': [[cells[p, n] for p in periods] for n in names]}


def line_axis(data):
    """Use calendar coordinates for dated periods, preserving categorical labels otherwise."""
    from matplotlib import dates as mdates
    parsed = []
    for label in data['periods']:
        date = None
        if re.match(r'^\d{4}-\d{2}-\d{2}', label):
            try:
                date = datetime.fromisoformat(label.replace('Z', '+00:00'))
            except ValueError:
                pass
        if date is None:
            for pattern in ('%Y-%m-%d', '%Y-%m', '%b %Y', '%B %Y'):
                try:
                    date = datetime.strptime(label, pattern)
                    break
                except ValueError:
                    continue
        parsed.append(date)
    if not any(parsed):
        return {**data, 'positions': list(range(len(parsed))), 'dateAxis': False}
    if not all(parsed):
        raise ValueError('Line periods must all be valid dates or all categorical labels')
    positions = mdates.date2num(parsed).tolist()
    if len(set(positions)) != len(positions):
        raise ValueError('Date periods must refer to distinct times')
    order = sorted(range(len(positions)), key=positions.__getitem__)
    return {**data, 'positions': [positions[i] for i in order], 'dateAxis': True,
            'periods': [data['periods'][i] for i in order],
            'values': [[row[i] for i in order] for row in data['values']]}


def prepare(spec_file, narration=None, *, brand='default'):
    spec = load_json(spec_file)
    if not isinstance(spec, dict) or type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Chart spec version must be 1')
    if spec.get('type') not in ('line', 'bar') or not isinstance(spec.get('dataset'), str):
        raise ValueError('Chart needs type line/bar and a dataset path')
    source = (Path(spec_file).resolve().parent / spec['dataset']).resolve()
    data = dataset(source, spec.get('columns', {'period': 'period', 'series': 'series', 'value': 'value'}), spec.get('seriesName', 'Value'))
    if spec['type'] == 'line' and len(data['periods']) < 2:
        raise ValueError('Line charts need at least two periods')
    if spec['type'] == 'line':
        data = line_axis(data)
    for key in ('title', 'unit', 'source', 'headline'):
        if not isinstance(spec.get(key, ''), str):
            raise ValueError(f'{key} must be text')
        if '\n' in spec.get(key, '') or '\r' in spec.get(key, ''):
            raise ValueError(f'{key} must be a single line')
    duration = number(spec.get('durationSeconds', 6), 'durationSeconds', .5, 3600)
    if narration:
        info = probe(narration)
        audio_stream(info)
        duration = number(info['duration'], 'narration duration', .5, 3600)
    fps = spec.get('fps', 30)
    width, height = spec.get('width', 1280), spec.get('height', 720)
    for name, value, low, high in [('fps', fps, 1, 60), ('width', width, 64, 4096), ('height', height, 64, 4096)]:
        if type(value) is not int:
            raise ValueError(f'{name} must be an integer')
        number(value, name, low, high)
    if width % 2 or height % 2:
        raise ValueError('Chart dimensions must be even')
    headline_start = number(spec.get('headlineStartSeconds', 0), 'headlineStartSeconds', 0, duration)
    headline_fade = number(spec.get('headlineFadeSeconds', .5), 'headlineFadeSeconds', 0, duration)
    if spec.get('headline') and (headline_start >= duration or headline_start + headline_fade > duration):
        raise ValueError('The headline must start and finish fading before the video ends')
    theme = brand_theme(brand)
    return {'version': 1, 'type': spec['type'], 'title': spec.get('title', ''), 'unit': spec.get('unit', ''),
            'dataset': str(source), 'datasetSha256': sha256(source), 'data': data,
            'durationSeconds': duration, 'fps': fps, 'width': width, 'height': height, 'theme': theme,
            'source': spec.get('source', ''), 'headline': spec.get('headline', ''),
            'headlineStartSeconds': headline_start, 'headlineFadeSeconds': headline_fade, 'narrationAudio': str(Path(narration).resolve()) if narration else None}


def brand_theme(name):
    from matplotlib.font_manager import FontProperties, findfont
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a brand name from brands/<name>/brand.json')
    path = Path(__file__).resolve().parents[1] / 'brands' / name / 'brand.json'
    brand = load_json(path)
    colors = brand.get('colors', {})
    for key in ('primary', 'bgLight', 'textDark', 'textLight'):
        if not isinstance(colors.get(key), str) or not re.fullmatch(r'#[0-9A-Fa-f]{6}', colors[key]):
            raise ValueError(f'Brand needs a #RRGGBB colors.{key}')
    family = brand.get('fonts', {}).get('primary')
    if not isinstance(family, str) or not family.strip():
        raise ValueError('Brand needs fonts.primary')
    requested = [part.strip().strip('"\'') for part in family.split(',')]
    font_path = None
    for index, candidate in enumerate([*requested, 'DejaVu Sans']):
        if candidate.casefold() == 'system-ui':
            aliases = (['Segoe UI', 'DejaVu Sans'] if sys.platform == 'win32' else
                       ['Helvetica Neue', 'DejaVu Sans'] if sys.platform == 'darwin' else ['DejaVu Sans'])
        elif candidate.casefold() in ('-apple-system', 'blinkmacsystemfont'):
            aliases = ['Helvetica Neue', 'DejaVu Sans'] if sys.platform == 'darwin' else []
        else:
            aliases = [candidate]
        for alias in aliases:
            try:
                font_path = findfont(FontProperties(family=[alias]), fallback_to_default=False)
                break
            except ValueError:
                continue
        if font_path is not None:
            break
    if font_path is None:
        raise ValueError('No usable brand font found. Install a font from fonts.primary.')
    font_name = FontProperties(fname=font_path).get_name()
    palette = [colors['primary'], colors['textLight']]
    return {'name': name, 'path': str(path), 'sha256': sha256(path), 'background': colors['bgLight'],
            'text': colors['textDark'], 'muted': colors['textLight'], 'palette': palette,
            'fontFamily': family, 'fontPath': font_path, 'resolvedFont': font_name,
            'fontFallback': index != 0}


def text_overlay(report, *, fields=('title', 'headline', 'source')):
    """Render author-written title, attribution and headline independently of Matplotlib."""
    from PIL import Image, ImageDraw, ImageFont
    width, height = report['width'], report['height']
    layer = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(layer)
    margin = round(width * .04)
    for key, top, scale, color in [('title', .035, .040, report['theme']['text']),
                                    ('headline', .12, .070, report['theme']['text']),
                                    ('period', .20, .025, report['theme']['muted']),
                                    ('source', .93, .023, report['theme']['muted'])]:
        text = report.get(key, '')
        if key not in fields or not text:
            continue
        size = max(8, round(height * scale))
        while True:
            font = ImageFont.truetype(report['theme']['fontPath'], size)
            box = draw.textbbox((0, 0), text, font=font)
            if box[2] - box[0] <= width - margin * 2:
                break
            size -= 1
            if size < 8:
                raise ValueError(f'{key} is too long for this chart size; shorten the text')
        draw.text((margin, round(height * top)), text, font=font, fill=color, anchor='lt')
    return layer


def chart_renderer(report):
    from matplotlib import dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from PIL import Image
    theme = report['theme']
    figure = Figure(figsize=(report['width'] / 100, report['height'] / 100), dpi=100, facecolor=theme['background'])
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    figure.subplots_adjust(left=.16, right=.95, bottom=.23, top=.76)
    axes.set_facecolor(theme['background'])
    font = FontProperties(fname=theme['fontPath'], size=max(6, report['height'] / 55))
    axes.tick_params(colors=theme['text'], labelsize=font.get_size())
    for spine in axes.spines.values():
        spine.set_color(theme['muted'])
    values = np.array(report['data']['values'], dtype=float)
    periods, names = report['data']['periods'], report['data']['series']
    minimum, maximum = min(0, values.min()), max(0, values.max())
    span = maximum - minimum or 1
    axes.set_ylim(minimum - span * .05, maximum + span * .15)
    axes.set_ylabel(report['unit'], color=theme['text'], fontproperties=font)
    artists = []
    if report['type'] == 'line':
        positions = np.asarray(report['data']['positions'], dtype=float)
        axes.set_xlim(positions[0], positions[-1])
        if report['data']['dateAxis']:
            axes.xaxis_date()
            first, last = mdates.num2date(positions[[0, -1]])
            months = (last.year - first.year) * 12 + last.month - first.month
            locator = mdates.MonthLocator(interval=max(1, math.ceil(months / 6)))
            ticks = [tick for tick in locator.tick_values(first, last)
                     if positions[0] <= tick <= positions[-1] and mdates.num2date(tick).day == 1]
            # MonthLocator falls back to the endpoints when a range contains no month start.
            axes.set_xticks(ticks or [positions[0]])
            axes.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:
            indices = np.unique(np.linspace(0, len(periods) - 1, min(8, len(periods))).astype(int))
            axes.set_xticks(indices, [periods[i] for i in indices], rotation=15)
        for i, name in enumerate(names):
            color = theme['palette'][0 if i == 0 else 1]
            artists.append(axes.plot([], [], label=name, color=color, linewidth=2.5)[0])
        if len(names) > 1:
            axes.legend(facecolor=theme['background'], labelcolor=theme['text'], prop=font)
    else:
        artists = axes.bar(names, values[:, 0], color=theme['palette'][0])
        axes.tick_params(axis='x', rotation=15)
    for tick in [*axes.get_xticklabels(), *axes.get_yticklabels()]:
        tick.set_fontproperties(font)
    overlay = text_overlay(report, fields=('title', 'source'))
    headline = text_overlay(report, fields=('headline',))
    period_text, period_overlay = None, None
    frame_count = math.ceil(report['durationSeconds'] * report['fps'])
    hold = min(round(report['fps'] * .5), frame_count // 3)

    def frame_at(seconds):
        nonlocal period_text, period_overlay
        frame = min(frame_count - 1, max(0, int(seconds * report['fps'] + 1e-6)))
        amount = min(1, frame / max(1, frame_count - hold - 1))
        if report['type'] == 'line':
            head = positions[0] + amount * (positions[-1] - positions[0])
            left = max(0, int(np.searchsorted(positions, head, side='right')) - 1)
            right = min(left + 1, len(periods) - 1)
            fraction = (head - positions[left]) / (positions[right] - positions[left]) if right > left else 0
        else:
            progress = amount * (len(periods) - 1)
            left = int(progress)
            right = min(left + 1, len(periods) - 1)
            fraction = progress - left
        current = values[:, left] * (1 - fraction) + values[:, right] * fraction
        for i, artist in enumerate(artists):
            if report['type'] == 'line':
                artist.set_data([*positions[:left + 1], head], [*values[i, :left + 1], current[i]])
            else:
                artist.set_height(current[i])
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba()).copy())
        image.alpha_composite(overlay)
        elapsed = frame / report['fps'] - report['headlineStartSeconds']
        fade = report['headlineFadeSeconds']
        opacity = max(0, min(1, elapsed / fade)) if fade else float(elapsed >= 0)
        if report['headline'] and opacity >= 1:
            image.alpha_composite(headline)
        elif report['headline'] and opacity > 0:
            faded = headline.copy()
            faded.putalpha(headline.getchannel('A').point(lambda a: round(a * opacity)))
            image.alpha_composite(faded)
        if report['type'] == 'bar':
            label = periods[left] if fraction == 0 or left == right else f'{periods[left]} → {periods[right]}'
            if label != period_text:
                period_text = label
                period_overlay = text_overlay({**report, 'period': label}, fields=('period',))
            image.alpha_composite(period_overlay)
        return np.asarray(image.convert('RGB')).copy()
    return figure, frame_at


def chart_frames(report):
    figure, frame_at = chart_renderer(report)
    try:
        for frame in range(math.ceil(report['durationSeconds'] * report['fps'])):
            yield frame_at(frame / report['fps'])
    finally:
        figure.clear()


def render(report, output):
    from moviepy import VideoClip
    output = output_path(output, report['dataset'], report.get('narrationAudio'), report['theme']['path'])
    if sha256(report['dataset']) != report['datasetSha256']:
        raise ValueError('Dataset changed after planning. Rebuild the chart report.')
    if sha256(report['theme']['path']) != report['theme']['sha256']:
        raise ValueError('Brand changed after planning. Rebuild the chart report.')
    figure, frame_at = chart_renderer(report)
    video = None
    try:
        video = VideoClip(frame_function=frame_at, duration=report['durationSeconds'])
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{output.stem}-', dir=output.parent) as folder:
            target = Path(folder) / output.name
            picture = Path(folder) / 'picture.mp4' if report.get('narrationAudio') else target
            video.write_videofile(str(picture), fps=report['fps'], codec='libx264', audio=False,
                                  threads=2, logger=None, ffmpeg_params=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            if report.get('narrationAudio'):
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(picture), '-i', report['narrationAudio'],
                                '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
                                '-t', str(report['durationSeconds']), '-movflags', '+faststart', str(target)],
                               check=True, capture_output=True, text=True, encoding='utf-8')
            if not target.is_file() or target.stat().st_size == 0:
                raise ValueError('Encoder produced an empty chart')
            os.replace(target, output)
    finally:
        if video is not None:
            video.close()
        figure.clear()


def main(argv=None, *, default_spec=None, default_output=None, default_brand='default'):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=default_spec, required=default_spec is None)
    parser.add_argument('--output', type=Path, default=default_output, required=default_output is None)
    parser.add_argument('--brand', default=default_brand, help='Brand name from brands/<name>/brand.json (default: default)')
    parser.add_argument('--narration', type=Path, help='Author-provided narration audio; sets chart duration')
    parser.add_argument('--save-report', type=Path, help='Write data provenance JSON after rendering')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print a report without writes')
    args = parser.parse_args(argv)
    try:
        report = prepare(args.spec, args.narration, brand=args.brand)
        inputs = [args.spec, report['dataset'], args.narration, report['theme']['path']]
        output_path(args.output, *inputs)
        if args.save_report:
            output_path(args.save_report, *inputs, args.output)
        report.update(executed=not args.dry_run, output=str(args.output.resolve()))
        if not args.dry_run:
            render(report, args.output)
            if args.save_report:
                write_json(args.save_report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except ImportError as error:
        print(f'Missing dependency: {error}. Install with uv sync from the toolkit root.', file=sys.stderr)
        return 1
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
