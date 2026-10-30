#!/usr/bin/env python3
"""Composite moving presenters with backgrounds and tracked graphics using existing media engines."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from media_ops import (
    cv2_module,
    encode_frames,
    load_json,
    number,
    output_path,
    probe,
    sha256,
    video_stream,
)
from reframe import validate_track


class RvmMatte:
    """Use the published RVM FP32 ONNX interface; carry recurrent state across frames."""
    def __init__(self, model, ratio):
        number(ratio, 'downsample ratio', .05, 1)
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ValueError('RVM requires uv sync --extra matting and a local FP32 ONNX model') from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = {item.name: item.type for item in self.session.get_inputs()}
        expected = {'src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'}
        outputs = {item.name for item in self.session.get_outputs()}
        if set(inputs) != expected or any(t != 'tensor(float)' for t in inputs.values()) or outputs != {'fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'}:
            raise ValueError('Use the official RVM FP32 ONNX model interface')
        self.states = [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]
        self.ratio = np.array([ratio], dtype=np.float32)

    def matte(self, bgr):
        source = bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        foreground, alpha, *states = self.session.run(['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'], {
            'src': source, 'downsample_ratio': self.ratio,
            **{f'r{i + 1}i': value for i, value in enumerate(self.states)}})
        if foreground.shape != source.shape or alpha.shape != (1, 1, bgr.shape[0], bgr.shape[1]):
            raise ValueError('RVM returned unexpected frame dimensions')
        if not np.isfinite(foreground).all() or not np.isfinite(alpha).all():
            raise ValueError('RVM returned nonfinite pixels')
        self.states = states
        return np.clip(foreground[0].transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255, np.clip(alpha[0, 0], 0, 1)[:, :, None]


class FrameSource:
    """Sample a still or constant-FPS video at the foreground's timestamps."""
    def __init__(self, path, *, loop=False, gray=False):
        self.cv2 = cv2_module()
        self.path = Path(path).resolve()
        self.still = None
        self.capture = None
        self.cached = None
        self.index = -1
        self.gray = gray
        self.loop = loop
        if self.path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
            self.still = self.cv2.imread(str(self.path), self.cv2.IMREAD_GRAYSCALE if gray else self.cv2.IMREAD_COLOR)
            if self.still is None:
                raise ValueError(f'Cannot read image: {self.path}')
        else:
            stream = video_stream(probe(self.path))
            self.fps = float(Fraction(stream['avg_frame_rate']))
            number(self.fps, 'background/mask FPS', .1, 240)
            if abs(float(Fraction(stream['r_frame_rate'])) - self.fps) > self.fps * .01:
                raise ValueError('Background/mask video must have constant FPS')
            self.capture = self.cv2.VideoCapture(str(self.path))
            self.count = int(self.capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
            if self.count < 1:
                self.close()
                raise ValueError('Cannot determine background/mask frame count')

    def at(self, seconds):
        if self.still is not None:
            return self.still
        index = math.floor(seconds * self.fps + .000001)
        if self.loop:
            index %= self.count
        elif index >= self.count:
            raise ValueError('Background or mask is shorter than foreground. Supply full media or enable background looping.')
        if index != self.index:
            if index != self.index + 1:
                self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = self.capture.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {index} of {self.path}')
            self.cached = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY) if self.gray else frame
            self.index = index
        return self.cached

    def close(self):
        if self.capture is not None:
            self.capture.release()


def cover(frame, width, height):
    cv2 = cv2_module()
    scale = max(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (math.ceil(frame.shape[1] * scale), math.ceil(frame.shape[0] * scale)))
    x, y = (resized.shape[1] - width) // 2, (resized.shape[0] - height) // 2
    return resized[y:y + height, x:x + width]


def overlay_images(spec, root, duration, report):
    cv2 = cv2_module()
    if not isinstance(spec, list) or len(spec) > 64:
        raise ValueError('Overlays must be an array of at most 64 entries')
    result = []
    for item in spec:
        if not isinstance(item, dict) or not isinstance(item.get('path'), str):
            raise ValueError('Each overlay needs a path')
        path = (root / item['path']).resolve()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError('Overlay images must have four RGBA channels, such as transparent PNG')
        start = number(item.get('startSeconds', 0), 'overlay start', 0, duration)
        end = number(item.get('endSeconds', duration), 'overlay end', start + .000001, duration + .001)
        size = item.get('size', [image.shape[1], image.shape[0]])
        if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or not 1 <= x <= 4096 for x in size):
            raise ValueError('Overlay size must be [width, height] integers from 1 to 4096')
        position = item.get('position')
        attached = item.get('attachToTrack', False)
        if type(attached) is not bool or (position is None) == (not attached):
            raise ValueError('Choose position or attachToTrack for every overlay')
        if attached and report is None:
            raise ValueError('Attached overlays require --track')
        offset = item.get('offset', [0, 0])
        for name, pair in [('position', position if position is not None else [0, 0]), ('offset', offset)]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f'{name} must be [x,y]')
            for value in pair:
                number(value, name, -16384, 16384)
        result.append({**item, 'path': str(path), 'image': cv2.resize(image, tuple(size)),
                       'startSeconds': start, 'endSeconds': end, 'position': position,
                       'attachToTrack': attached, 'offset': offset})
    return result


def paste(frame, image, x, y):
    x, y = round(x), round(y)
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame.shape[1], x + image.shape[1]), min(frame.shape[0], y + image.shape[0])
    if right <= left or bottom <= top:
        return
    cropped = image[top - y:bottom - y, left - x:right - x]
    alpha = cropped[:, :, 3:4].astype(np.float32) / 255
    frame[top:bottom, left:right] = np.clip(cropped[:, :, :3] * alpha + frame[top:bottom, left:right] * (1 - alpha), 0, 255).astype(np.uint8)


def keyed_frames(video, width, height, key, similarity, blend):
    """Use FFmpeg chromakey, including its chroma-space edge handling."""
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(video), '-vf',
            f'chromakey=0x{key[1:]}:{similarity}:{blend}', '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors)
        try:
            length = width * height * 4
            while True:
                parts, size = [], 0
                while size < length:
                    part = process.stdout.read(length - size)
                    if not part:
                        break
                    parts.append(part)
                    size += len(part)
                if not size:
                    break
                if size != length:
                    raise ValueError('Incomplete FFmpeg keying frame')
                rgba = np.frombuffer(b''.join(parts), dtype=np.uint8).reshape(height, width, 4)
                yield rgba[:, :, :3][:, :, ::-1].astype(np.float32), rgba[:, :, 3:4].astype(np.float32) / 255
            if process.wait(timeout=1800):
                errors.seek(0)
                raise ValueError(errors.read().decode('utf-8', errors='replace')[-2000:])
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def composite(foreground, background, output, *, alpha=None, key=None, rvm_model=None,
              overlays=None, overlay_root=Path('.'), report=None, ratio=None,
              similarity=.15, blend=.05, loop_background=False, execute=False):
    cv2 = cv2_module()
    info = probe(foreground)
    stream = video_stream(info)
    width, height = stream['width'], stream['height']
    fps = float(Fraction(stream['avg_frame_rate']))
    number(fps, 'foreground FPS', 1, 240)
    if width % 2 or height % 2:
        raise ValueError('Foreground must have even dimensions for H.264 output')
    if abs(float(Fraction(stream['r_frame_rate'])) - fps) > fps * .01:
        raise ValueError('Convert variable-frame-rate foreground to constant FPS before compositing')
    if sum(x is not None for x in (alpha, key, rvm_model)) != 1:
        raise ValueError('Choose exactly one supplied alpha, chroma key or RVM model')
    if key and (not isinstance(key, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', key)):
        raise ValueError('Chroma key must be #RRGGBB')
    number(similarity, 'key similarity', .01, 1)
    number(blend, 'key blend', 0, 1)
    ratio = number(ratio if ratio is not None else min(1, 512 / max(width, height)), 'downsample ratio', .05, 1)
    if rvm_model and not Path(rvm_model).is_file():
        raise ValueError('Supply a local RVM FP32 ONNX model')
    if report is not None:
        validate_track(report, foreground)
    layers = overlay_images(overlays if overlays is not None else [], overlay_root, info['duration'], report)
    output_path(output, foreground, background, alpha, rvm_model, *[o['path'] for o in layers])
    backdrop = FrameSource(background, loop=loop_background)
    mask = None
    capture = None
    keyed = None
    count = 0
    try:
        if alpha:
            mask = FrameSource(alpha, gray=True)
            if mask.still is None and abs(mask.fps - fps) > .001:
                raise ValueError('Alpha video must have the same FPS as foreground')
            if mask.at(0).shape != (height, width):
                raise ValueError('Alpha dimensions must exactly match foreground')
        result = {'version': 1, 'executed': execute, 'output': str(Path(output).resolve()),
                  'sourceSha256': sha256(foreground), 'dimensions': [width, height],
                  'mode': 'rvm' if rvm_model else ('chromakey' if key else 'supplied-alpha'),
                  'overlays': len(layers), 'modelSha256': sha256(rvm_model) if rvm_model else None}
        if not execute:
            return result
        matte = RvmMatte(rvm_model, ratio) if rvm_model else None
        if key:
            keyed = keyed_frames(foreground, width, height, key, similarity, blend)
        else:
            capture = cv2.VideoCapture(str(Path(foreground).resolve()))
        def frames():
            nonlocal count
            while True:
                if keyed is not None:
                    try:
                        fgr, opacity = next(keyed)
                    except StopIteration:
                        break
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if matte:
                        fgr, opacity = matte.matte(frame)
                    else:
                        fgr, opacity = frame.astype(np.float32), mask.at(count / fps).astype(np.float32)[:, :, None] / 255
                bgr = cover(backdrop.at(count / fps), width, height)
                composed = np.clip(fgr * opacity + bgr * (1 - opacity), 0, 255).astype(np.uint8)
                for layer in layers:
                    if layer['startSeconds'] <= count / fps < layer['endSeconds']:
                        if layer['attachToTrack']:
                            if count >= len(report['frames']):
                                raise ValueError('Track is shorter than foreground')
                            x, y, w, _height = report['frames'][count]['box']
                            position = [x + w / 2 - layer['image'].shape[1] / 2, y - layer['image'].shape[0]]
                        else:
                            position = layer['position']
                        paste(composed, layer['image'], position[0] + layer['offset'][0], position[1] + layer['offset'][1])
                count += 1
                yield composed
            if count == 0:
                raise ValueError('No foreground frames decoded')
            if report and count != len(report['frames']):
                raise ValueError('Track frame count does not match foreground')
        encode_frames(frames(), output, width=width, height=height, fps=fps, audio=foreground)
        return {**result, 'frames': count}
    finally:
        backdrop.close()
        if mask:
            mask.close()
        if capture is not None:
            capture.release()
        if keyed is not None:
            keyed.close()


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foreground', type=Path, required=True)
    parser.add_argument('--background', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--alpha', type=Path, help='White is opaque, black is transparent; still or same-FPS video')
    mode.add_argument('--key', help='Chroma key color, such as #00ff00')
    mode.add_argument('--rvm-model', type=Path, help='Local official FP32 RVM ONNX model; no weights downloaded automatically')
    parser.add_argument('--overlays', type=Path)
    parser.add_argument('--track', type=Path)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--similarity', type=float, default=.15)
    parser.add_argument('--blend', type=float, default=.05)
    parser.add_argument('--loop-background', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = composite(args.foreground, args.background, args.output, alpha=args.alpha, key=args.key,
            rvm_model=args.rvm_model, overlays=load_json(args.overlays) if args.overlays else None,
            overlay_root=args.overlays.resolve().parent if args.overlays else Path('.'),
            report=load_json(args.track) if args.track else None, ratio=args.ratio,
            similarity=args.similarity, blend=args.blend, loop_background=args.loop_background, execute=args.run)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
