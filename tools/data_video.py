#!/usr/bin/env python3
"""Render reusable CSV/JSON chart videos with Matplotlib, optional narration and provenance."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from media_ops import (
    audio_stream,
    encode_frames,
    load_json,
    number,
    output_path,
    probe,
    sha256,
    staged_output,
    write_json,
)


def dataset(path, columns):
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


def prepare(spec_file, narration=None):
    spec = load_json(spec_file)
    if not isinstance(spec, dict) or type(spec.get('version')) is not int or spec['version'] != 1:
        raise ValueError('Chart spec version must be 1')
    if spec.get('type') not in ('line', 'bar') or not isinstance(spec.get('dataset'), str):
        raise ValueError('Chart needs type line/bar and a dataset path')
    source = (Path(spec_file).resolve().parent / spec['dataset']).resolve()
    data = dataset(source, spec.get('columns', {'period': 'period', 'series': 'series', 'value': 'value'}))
    if spec['type'] == 'line' and len(data['periods']) < 2:
        raise ValueError('Line charts need at least two periods')
    for key in ('title', 'unit'):
        if not isinstance(spec.get(key, ''), str):
            raise ValueError(f'{key} must be text')
    duration = number(spec.get('durationSeconds', 6), 'durationSeconds', .5, 3600)
    if narration:
        audio_info = probe(narration)
        audio_stream(audio_info)
        duration = number(audio_info['duration'], 'narration duration', .5, 3600)
    fps = spec.get('fps', 30)
    width, height = spec.get('width', 1280), spec.get('height', 720)
    for name, value, low, high in [('fps', fps, 1, 60), ('width', width, 64, 4096), ('height', height, 64, 4096)]:
        if type(value) is not int:
            raise ValueError(f'{name} must be an integer')
        number(value, name, low, high)
    if width % 2 or height % 2:
        raise ValueError('Chart dimensions must be even')
    colors = spec.get('colors', ['#64b5f6', '#ffb74d', '#81c784', '#e57373', '#ba68c8', '#4dd0e1'])
    if not isinstance(colors, list) or not colors or any(not isinstance(c, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', c) for c in colors):
        raise ValueError('colors must be a nonempty array of #RRGGBB colors')
    narration_text = []
    for name, values in zip(data['series'], data['values']):
        if len(values) == 1:
            narration_text.append(f'{name}: {values[0]:g} {spec.get("unit", "")} in {data["periods"][0]}.')
        else:
            narration_text.append(f'{name}: {values[0]:g} in {data["periods"][0]}, {values[-1]:g} in {data["periods"][-1]}. Change: {values[-1] - values[0]:+g} {spec.get("unit", "")}.')
    return {'version': 1, 'type': spec['type'], 'title': spec.get('title', ''), 'unit': spec.get('unit', ''),
            'dataset': str(source), 'datasetSha256': sha256(source), 'data': data,
            'durationSeconds': duration, 'fps': fps, 'width': width, 'height': height, 'colors': colors,
            'narrationText': '\n'.join(narration_text), 'narrationAudio': str(Path(narration).resolve()) if narration else None}


def chart_frames(report):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ValueError('Matplotlib is a core dependency. Install with uv sync') from error
    figure = Figure(figsize=(report['width'] / 100, report['height'] / 100), dpi=100, facecolor='#111827')
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    figure.subplots_adjust(left=.14, right=.95, bottom=.20, top=.78)
    axes.set_facecolor('#111827')
    axes.tick_params(colors='#e5e7eb', labelsize=10)
    for spine in axes.spines.values():
        spine.set_color('#64748b')
    values = np.array(report['data']['values'], dtype=float)
    periods, names = report['data']['periods'], report['data']['series']
    minimum, maximum = min(0, values.min()), max(0, values.max())
    span = maximum - minimum or 1
    axes.set_ylim(minimum - span * .05, maximum + span * .15)
    axes.set_ylabel(report['unit'], color='#e5e7eb')
    figure.suptitle(report['title'], color='#f9fafb', fontsize=max(10, min(24, report['width'] / 50)), y=.94)
    artists = []
    if report['type'] == 'line':
        axes.set_xlim(0, max(1, len(periods) - 1))
        tick_indices = np.unique(np.linspace(0, len(periods) - 1, min(8, len(periods))).astype(int))
        axes.set_xticks(tick_indices, [periods[i] for i in tick_indices], rotation=15)
        for i, name in enumerate(names):
            artists.append(axes.plot([], [], label=name, color=report['colors'][i % len(report['colors'])], linewidth=2.5)[0])
        axes.legend(facecolor='#1f2937', labelcolor='#f9fafb', fontsize=9)
    else:
        artists = axes.bar(names, values[:, 0], color=[report['colors'][i % len(report['colors'])] for i in range(len(names))])
        axes.tick_params(axis='x', rotation=15)
    frame_count = math.ceil(report['durationSeconds'] * report['fps'])
    hold = min(round(report['fps'] * .5), frame_count // 3)
    for frame in range(frame_count):
        progress = min(1, frame / max(1, frame_count - hold - 1)) * (len(periods) - 1)
        left = int(progress)
        right = min(left + 1, len(periods) - 1)
        fraction = progress - left
        current = values[:, left] * (1 - fraction) + values[:, right] * fraction
        axes.set_title(periods[left], color='#cbd5e1', fontsize=11)
        for i, artist in enumerate(artists):
            if report['type'] == 'line':
                artist.set_data([*range(left + 1), progress], [*values[i, :left + 1], current[i]])
            else:
                artist.set_height(current[i])
        canvas.draw()
        # Agg produces RGBA; the shared FFmpeg encoder consumes BGR.
        yield np.asarray(canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()


def render(report, output):
    output_path(output, report['dataset'], report.get('narrationAudio'))
    if sha256(report['dataset']) != report['datasetSha256']:
        raise ValueError('Dataset changed after planning. Rebuild the chart report.')
    encode_frames(chart_frames(report), output, width=report['width'], height=report['height'],
                  fps=report['fps'], audio=report.get('narrationAudio'))


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--narration', type=Path, help='Existing narration audio; sets exact chart duration')
    parser.add_argument('--script-output', type=Path, help='Explicitly write a data-derived narration script')
    parser.add_argument('--save-report', type=Path, help='Explicitly write data/provenance JSON')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = prepare(args.spec, args.narration)
        output_path(args.output, args.spec, report['dataset'], args.narration)
        outputs = [p for p in [args.output, args.script_output, args.save_report] if p]
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError('Output paths must be distinct')
        for path in outputs:
            output_path(path, args.spec, report['dataset'], args.narration)
        if args.script_output:
            with staged_output(args.script_output) as target:
                target.write_text(report['narrationText'] + '\n', encoding='utf-8')
        report.update(executed=args.run, output=str(args.output.resolve()))
        if args.run:
            render(report, args.output)
        if args.save_report:
            write_json(args.save_report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
