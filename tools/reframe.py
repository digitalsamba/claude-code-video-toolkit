#!/usr/bin/env python3
"""Track a subject with OpenCV and render a smoothed crop. Preview by default."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from media_ops import (
    cv2_module,
    encode_frames,
    number,
    output_path,
    probe,
    sha256,
    video_stream,
    write_json,
)


def rectangle(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('box must be [x, y, width, height] in source pixels')
    x, y, w, h = [number(v, 'box coordinate', 0) for v in value]
    if w < 2 or h < 2 or x + w > width or y + h > height:
        raise ValueError('Subject box must fit inside the source frame')
    return [round(x), round(y), round(w), round(h)]


def crop_box(center, source, target):
    width, height = source
    target_w, target_h = target
    if width / height >= target_w / target_h:
        crop_h, crop_w = height, round(height * target_w / target_h)
    else:
        crop_w, crop_h = width, round(width * target_h / target_w)
    x = max(0, min(width - crop_w, round(center[0] - crop_w / 2)))
    y = max(0, min(height - crop_h, round(center[1] - crop_h / 2)))
    return [x, y, crop_w, crop_h]


def dimensions(width, height):
    for name, value in [('width', width), ('height', height)]:
        if type(value) is not int or value % 2:
            raise ValueError(f'{name} must be an even integer')
        number(value, name, 2, 4096)


def track(video, *, box=None, faces=False, max_lost=12):
    cv2 = cv2_module()
    info = probe(video)
    stream = video_stream(info)
    fps = float(Fraction(stream['avg_frame_rate']))
    nominal = float(Fraction(stream['r_frame_rate']))
    number(fps, 'frame rate', 1, 240)
    if abs(nominal - fps) > fps * .01:
        raise ValueError('Variable-frame-rate input must be converted to constant FPS before tracking')
    if (box is None) == (not faces):
        raise ValueError('Choose either an initial box or automatic faces')
    number(max_lost, 'max lost frames', 0, 300)
    if type(max_lost) is not int:
        raise ValueError('max lost frames must be an integer')
    capture = cv2.VideoCapture(str(Path(video).resolve()))
    entries, last, lost = [], None, 0
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') if faces else None
    tracker = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            found = True
            if faces:
                detections = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(24, 24))
                if len(detections):
                    if last is None:
                        chosen = max(detections, key=lambda b: int(b[2]) * int(b[3]))
                    else:
                        cx, cy = last[0] + last[2] / 2, last[1] + last[3] / 2
                        chosen = min(detections, key=lambda b: (b[0] + b[2] / 2 - cx) ** 2 + (b[1] + b[3] / 2 - cy) ** 2)
                        # Do not silently switch to a distant face after an occlusion.
                        if ((chosen[0] + chosen[2] / 2 - cx) ** 2 + (chosen[1] + chosen[3] / 2 - cy) ** 2) ** .5 > max(last[2], last[3]) * 2:
                            found = False
                    if found:
                        last = [int(v) for v in chosen]
                else:
                    found = False
            elif tracker is None:
                last = rectangle(box, width, height)
                if not hasattr(cv2, 'TrackerCSRT_create'):
                    raise ValueError('CSRT requires opencv-contrib-python-headless, installed by the vision extra')
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, tuple(last))
            else:
                found, current = tracker.update(frame)
                if found:
                    last = rectangle(list(current), width, height)
            lost = 0 if found else lost + 1
            if last is None or lost > max_lost:
                raise ValueError(f'Subject lost at frame {len(entries)}. Supply a clearer box or split at the scene cut.')
            entries.append({'frame': len(entries), 'timeSeconds': len(entries) / fps,
                            'box': list(last), 'held': not bool(found)})
            if len(entries) > 1_000_000:
                raise ValueError('Tracking is limited to one million frames per job')
    finally:
        capture.release()
    if not entries:
        raise ValueError('No video frames decoded')
    return {'version': 1, 'sourceSha256': sha256(video), 'source': str(Path(video).resolve()),
            'width': stream['width'], 'height': stream['height'], 'fps': fps,
            'method': 'opencv-haar-faces' if faces else 'opencv-csrt', 'frames': entries}


def validate_track(report, video):
    if not isinstance(report, dict) or type(report.get('version')) is not int or report['version'] != 1 or report.get('sourceSha256') != sha256(video):
        raise ValueError('Track does not match the source video hash')
    info = probe(video)
    stream = video_stream(info)
    width, height = stream['width'], stream['height']
    if (report.get('width'), report.get('height')) != (width, height):
        raise ValueError('Track dimensions do not match video')
    fps = number(report.get('fps'), 'track FPS', 1, 240)
    if abs(fps - float(Fraction(stream['avg_frame_rate']))) > .001:
        raise ValueError('Track FPS does not match video')
    entries = report.get('frames')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Track needs per-frame boxes')
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or type(entry.get('frame')) is not int or entry['frame'] != index:
            raise ValueError('Track frames must be consecutive from zero')
        rectangle(entry.get('box'), width, height)
    return report


def reframe(video, output, report, *, width=1080, height=1920, smoothing=.2, max_speed=600):
    cv2 = cv2_module()
    validate_track(report, video)
    dimensions(width, height)
    number(smoothing, 'smoothing', .001, 1)
    number(max_speed, 'max pan speed', 1, 10000)
    output_path(output, video)
    capture = cv2.VideoCapture(str(Path(video).resolve()))
    def frames():
        center = None
        try:
            for entry in report['frames']:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Track is longer than the decoded video')
                x, y, w, h = entry['box']
                desired = [x + w / 2, y + h / 2]
                if center is None:
                    center = desired
                else:
                    limit = max_speed / report['fps']
                    center = [old + max(-limit, min(limit, (new - old) * smoothing)) for old, new in zip(center, desired)]
                left, top, cw, ch = crop_box(center, (report['width'], report['height']), (width, height))
                yield cv2.resize(frame[top:top + ch, left:left + cw], (width, height), interpolation=cv2.INTER_AREA)
            if capture.read()[0]:
                raise ValueError('Track is shorter than the decoded video')
        finally:
            capture.release()
    encode_frames(frames(), output, width=width, height=height, fps=report['fps'], audio=video)


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--box', help='x,y,width,height in first-frame source pixels')
    source.add_argument('--faces', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-track', type=Path, help='Defaults to <output>.track.json')
    parser.add_argument('--width', type=int, default=1080)
    parser.add_argument('--height', type=int, default=1920)
    parser.add_argument('--smoothing', type=float, default=.2)
    parser.add_argument('--max-lost', type=int, default=12)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        info = probe(args.video)
        stream = video_stream(info)
        dimensions(args.width, args.height)
        number(args.smoothing, 'smoothing', .001, 1)
        box = rectangle([float(x) for x in args.box.split(',')], stream['width'], stream['height']) if args.box else None
        output_path(args.output, args.video)
        track_file = args.save_track or args.output.with_suffix('.track.json')
        output_path(track_file, args.video, args.output)
        result = {'version': 1, 'executed': args.run, 'output': str(args.output.resolve()), 'track': str(track_file.resolve()),
                  'dimensions': [args.width, args.height], 'initialBox': box, 'automaticFaces': args.faces}
        if args.run:
            report = track(args.video, box=box, faces=args.faces, max_lost=args.max_lost)
            reframe(args.video, args.output, report, width=args.width, height=args.height, smoothing=args.smoothing)
            write_json(track_file, report)
            result['frames'] = len(report['frames'])
            result['heldFrames'] = sum(e['held'] for e in report['frames'])
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
