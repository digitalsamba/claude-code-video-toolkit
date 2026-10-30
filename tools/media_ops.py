"""Shared media boundaries for local production tools, using installed FFmpeg."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from win_encoding import utf8_env


def number(value, label, low=None, high=None):
    try:
        finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{label} must be a finite number')
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f'{label} must be between {low} and {high}')
    return value


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON property: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'Invalid JSON number: {x}')))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command, *, timeout=1800):
    result = subprocess.run([str(x) for x in command], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=timeout, shell=False, check=False, env=utf8_env())
    if result.returncode:
        raise ValueError(f'{command[0]} failed ({result.returncode}): {result.stderr[-3000:]}')
    return result.stdout


def probe(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Media file does not exist: {path}')
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', path]))
    duration = float(data.get('format', {}).get('duration', 0))
    number(duration, 'media duration', 0.000001)
    return {**data, 'duration': duration, 'path': str(path)}


def video_stream(info):
    stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
    if stream is None:
        raise ValueError('Input must have a video stream')
    return stream


def audio_stream(info):
    if not any(s['codec_type'] == 'audio' for s in info['streams']):
        raise ValueError('Input must have an audio stream')


def output_path(path, *inputs):
    path = Path(path).resolve()
    if path in [Path(p).resolve() for p in inputs if p is not None]:
        raise ValueError('Output cannot replace an input')
    if path.exists():
        raise ValueError(f'Output already exists: {path}. Choose a new filename.')
    return path


@contextmanager
def staged_output(path):
    """Publish a finished file without overwriting an existing file, even on a race."""
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError('Media engine did not produce a nonempty output')
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    with staged_output(path) as target:
        target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def cv2_module():
    try:
        import cv2
        return cv2
    except ImportError as error:
        raise ValueError('OpenCV is optional. Install with uv sync --extra vision') from error


def encode_frames(frames, output, *, width, height, fps, audio=None):
    """Encode a bounded BGR frame iterator; mux the original audio when supplied."""
    with staged_output(output) as target, tempfile.TemporaryFile() as errors:
        command = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0']
        if audio:
            command += ['-i', str(audio)]
        command += ['-map', '0:v:0']
        if audio:
            command += ['-map', '1:a:0?', '-c:a', 'aac']
        command += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for frame in frames:
                if frame.shape != (height, width, 3) or str(frame.dtype) != 'uint8':
                    raise ValueError('Frame dimensions or dtype do not match the encoder')
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=1800):
                errors.seek(0)
                raise ValueError(errors.read().decode('utf-8', errors='replace')[-3000:])
        except BaseException:
            process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
