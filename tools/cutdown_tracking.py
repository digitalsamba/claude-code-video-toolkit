"""Optional subject tracking for cutdown.py --mode track. No separate CLI."""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path


def cv2_module():
    try:
        import cv2
        return cv2
    except ImportError as error:
        raise ValueError('Tracking requires the optional vision extra: uv sync --extra vision. Other cutdown modes need only FFmpeg.') from error


def rectangle(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('--box needs x,y,width,height in source pixels')
    x, y, w, h = [round(v) for v in box]
    if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
        raise ValueError('Subject box must fit inside the source frame')
    return [x, y, w, h]


def crop_dimensions(src, width, height, zoom):
    scale = max(width / src.width, height / src.height) * zoom
    return (max(2, min(src.width, round(width / scale / 2) * 2)),
            max(2, min(src.height, round(height / scale / 2) * 2)))


def track_subject(video, src, segments, box=None):
    """Track each selected segment independently; hold position on ambiguity or loss."""
    cv2 = cv2_module()
    try:
        fps = float(Fraction(src.fps))
        average = float(Fraction(src.avg_fps or src.fps))
    except (ValueError, ZeroDivisionError):
        raise ValueError('Tracking needs a valid constant frame rate') from None
    if not 1 <= fps <= 240 or abs(average - fps) > fps * .01:
        raise ValueError('Tracking needs constant FPS. Convert variable-frame-rate input with FFmpeg first.')
    if box is not None:
        box = rectangle(box, src.width, src.height)
        if not hasattr(cv2, 'TrackerCSRT_create'):
            raise ValueError('CSRT requires opencv-contrib-python-headless: uv sync --extra vision')
    detector = None if box is not None else cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if detector is not None and detector.empty():
        raise ValueError('OpenCV face detector data is missing. Reinstall the vision extra.')
    reports, total = [], 0
    for segment in segments:
        capture = cv2.VideoCapture(str(Path(video).resolve()))
        first = math.ceil(segment.start * fps - 1e-6)
        count = math.ceil(segment.end * fps - 1e-6) - first
        if count < 1 or total + count > 1_000_000:
            capture.release()
            raise ValueError('Tracking needs 1 to one million selected frames')
        total += count
        if first and not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
            capture.release()
            raise ValueError('Could not seek to the selected tracking segment')
        tracker, last, previous, scene_cut = None, None, None, False
        entries = []
        try:
            for index in range(count):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Video ended before the selected tracking segment')
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18))
                # A hard cut invalidates the subject identity. Keep its last good position.
                if previous is not None and cv2.absdiff(previous, small).mean() > 45:
                    scene_cut = True
                previous = small
                found = False
                current = None
                if not scene_cut and box is not None:
                    if tracker is None:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, tuple(box))
                        found, current = True, box
                    else:
                        found, current = tracker.update(frame)
                elif not scene_cut:
                    faces = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(24, 24))
                    # Multiple faces are ambiguous. Never switch speakers automatically.
                    if len(faces) == 1:
                        current = [int(v) for v in faces[0]]
                        found = True
                if found and last is not None:
                    found = math.hypot(current[0] + current[2] / 2 - last[0] - last[2] / 2,
                                       current[1] + current[3] / 2 - last[1] - last[3] / 2) <= max(last[2:]) * 2
                if found:
                    try:
                        last = rectangle(list(current), src.width, src.height)
                    except (ValueError, TypeError):
                        found = False
                center = [last[0] + last[2] / 2, last[1] + last[3] / 2] if last else None
                entries.append({'center': center, 'held': not found, 'sceneCut': scene_cut})
        except cv2.error as error:
            raise ValueError(f'OpenCV tracking failed: {error}') from error
        finally:
            capture.release()
        reports.append(entries)
    return reports


def write_crop_commands(path, entries, src, width, height, *, focus, zoom, smoothing, max_speed, label):
    """Create source-pixel FFmpeg crop commands on the trimmed segment's timeline."""
    fps = float(Fraction(src.fps))
    crop_w, crop_h = crop_dimensions(src, width, height, zoom)
    center = entries[0]['center'] or [crop_w / 2 + (src.width - crop_w) * focus, src.height / 2]
    commands = []
    for index, entry in enumerate(entries):
        desired = entry['center'] or center
        limit = max_speed / fps
        center = [old + max(-limit, min(limit, (new - old) * smoothing)) for old, new in zip(center, desired)]
        x = max(0, min(src.width - crop_w, round(center[0] - crop_w / 2)))
        y = max(0, min(src.height - crop_h, round(center[1] - crop_h / 2)))
        commands.append(f'{index / fps:.9f} crop@{label} x {x}, crop@{label} y {y};')
    Path(path).write_text('\n'.join(commands) + '\n', encoding='utf-8', newline='\n')
