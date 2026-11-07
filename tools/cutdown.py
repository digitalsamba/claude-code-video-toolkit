#!/usr/bin/env python3
"""Social cut-downs: reframe a finished 16:9 video for Shorts, Reels, TikTok and feeds.

Takes any MP4 (a Remotion render, a NotebookLM export, a redub) and writes a
vertical (9:16), square (1:1) or portrait (4:5) copy, optionally cut down to
one or more time ranges. FFmpeg renders every mode; optional OpenCV tracks subjects locally. No API keys or project needed.

Reframe modes:
  blur  The whole frame, fitted to width, over a blurred and darkened copy of
        itself (default). Nothing is cut off, so it is safe for slides and
        screen recordings.
  crop  Fill the frame by cutting the sides off. Best for talking heads and
        anything with one point of interest. --focus picks where to cut
        (0 = left edge, 0.5 = centre, 1 = right edge), per segment if needed.
  track Follow one face or --box subject; hold position when tracking is lost.
        Requires uv sync --extra vision only for this mode.
  pad   Fitted to width over a solid colour (--pad-color, or a brand's bgDark).

Blur and pad keep the whole frame but shrink it: in 9:16 a 1920x1080 source
shows at 56% scale, so burned captions and small slide text shrink with it.
--zoom 1.3 trades a little of each side for size. Crop mode keeps full size.

Cut-downs: --start/--end keep one range; repeated --segment keeps several,
joined in order with a short audio fade at each join so cuts don't click.
A segment can carry its own crop focus: --segment 0:12-0:30@0.3.

Usage:
    # Vertical Short of the whole video, blurred fill
    uv run tools/cutdown.py --input out/video.mp4 --platform shorts

    # Vertical and square at once, 45 seconds from the middle
    uv run tools/cutdown.py --input out/video.mp4 --format 9:16 --format 1:1 \\
        --start 1:05 --end 1:50

    # Highlight reel: three ranges, cropped, each framed on its own subject
    uv run tools/cutdown.py --input out/video.mp4 --mode crop \\
        --segment 0:04-0:11@0.5 --segment 0:42-0:58@0.25 --segment 2:10-2:21@0.7

    # Add a branded frame (a 1080x1920 PNG with transparency) on top
    uv run tools/cutdown.py --input out/video.mp4 --overlay frame.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from win_encoding import ensure_utf8_output

FORMATS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
}

# Default format and longest upload that still counts as a short-form post.
# Checked 2026-10: YouTube Shorts and Instagram Reels allow up to 3 minutes,
# TikTok uploads up to 10. Longer uploads aren't rejected; they just stop being
# Shorts/Reels, so this is a warning rather than an error.
PLATFORMS = {
    "shorts": ("9:16", 180),
    "reels": ("9:16", 180),
    "tiktok": ("9:16", 600),
    "square": ("1:1", None),
    "feed": ("4:5", None),
}

# Audio fade at every cut, so joins don't click
JOIN_FADE = 0.03


@dataclass
class Segment:
    start: float
    end: float | None  # None = to the end of the input
    focus: float | None = None  # per-segment crop focus, overrides --focus


@dataclass
class Source:
    width: int
    height: int
    duration: float
    fps: str
    has_audio: bool
    audio_codec: str | None
    avg_fps: str | None = None


def parse_time(value: str) -> float:
    """Seconds from '75', '75.5', '1:15' or '01:01:15.5'."""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"not a time: {value!r} (use seconds, M:SS or H:MM:SS)")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def parse_segment(value: str) -> Segment:
    """'START-END' or 'START-END@FOCUS'; END may be empty for 'to the end'."""
    focus = None
    if "@" in value:
        value, focus_text = value.rsplit("@", 1)
        focus = parse_focus(focus_text)
    if "-" not in value:
        raise ValueError(f"segment {value!r} needs START-END, e.g. 0:12-0:30")
    start_text, end_text = value.split("-", 1)
    start = parse_time(start_text)
    end = parse_time(end_text) if end_text.strip() else None
    if end is not None and end <= start:
        raise ValueError(f"segment {value!r} ends before it starts")
    return Segment(start, end, focus)


def parse_focus(value: str) -> float:
    focus = float(value)
    if not 0 <= focus <= 1:
        raise ValueError(f"focus must be between 0 and 1, got {value}")
    return focus


def parse_hex(value: str) -> str:
    value = value.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError(f"not a #RRGGBB colour: {value!r}")
    return "0x" + value.upper()


def brand_pad_color(brand: str) -> str:
    path = Path(__file__).resolve().parent.parent / "brands" / brand / "brand.json"
    if not path.exists():
        raise ValueError(f"brand not found: {path}")
    colors = json.loads(path.read_text(encoding="utf-8")).get("colors", {})
    for key in ("bgDark", "background", "primary"):
        if isinstance(colors.get(key), str) and colors[key].startswith("#"):
            return parse_hex(colors[key])
    raise ValueError(f"{path} has no hex bgDark/background/primary colour")


def probe(path: Path) -> Source:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    video = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{path} has no video stream")
    audio = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    width, height = int(video["width"]), int(video["height"])
    # Phone footage stores rotation as metadata; ffmpeg autorotates on decode
    rotation = 0
    for side in video.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation) or 0)
    rotation = int(video.get("tags", {}).get("rotate", rotation) or 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    duration = float(info["format"].get("duration") or video.get("duration") or 0)
    return Source(
        width=width,
        height=height,
        duration=duration,
        fps=video.get("r_frame_rate", "30/1"),
        has_audio=audio is not None,
        audio_codec=audio.get("codec_name") if audio else None,
        avg_fps=video.get("avg_frame_rate"),
    )


def arg_type(fn):
    """argparse type that reports the ValueError message instead of 'invalid value'."""
    def wrapped(value: str):
        try:
            return fn(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e)) from None
    wrapped.__name__ = fn.__name__
    return wrapped


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def reframe_filter(src: Source, width: int, height: int, mode: str,
                   focus: float, zoom: float, pad_color: str) -> str:
    """Filter chain from one input video stream ([in]) to a WxH frame ([out])."""
    if mode == 'track':
        from cutdown_tracking import crop_dimensions
        cw, ch = crop_dimensions(src, width, height, zoom)
        return (f"[in]sendcmd=f='TRACKFILE',crop@TRACKLABEL=w={cw}:h={ch}:"
                f"x={round((src.width - cw) * focus)}:y={(src.height - ch) // 2}:exact=1,"
                f"scale={width}:{height},setsar=1[out]")
    if mode == "crop":
        scale = max(width / src.width, height / src.height) * zoom
        sw, sh = even(src.width * scale), even(src.height * scale)
        x = round((sw - width) * focus)
        y = (sh - height) // 2
        return f"[in]scale={sw}:{sh},crop={width}:{height}:{x}:{y},setsar=1[out]"

    # blur / pad: fit the whole frame, then zoom (cropping the overflow)
    scale = min(width / src.width, height / src.height) * zoom
    fw, fh = even(src.width * scale), even(src.height * scale)
    cw, ch = min(fw, width), min(fh, height)
    fg = f"scale={fw}:{fh}"
    if (cw, ch) != (fw, fh):
        fg += f",crop={cw}:{ch}:{round((fw - cw) * focus)}:{(fh - ch) // 2}"

    if mode == "pad":
        return (f"[in]{fg},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={pad_color},"
                f"setsar=1[out]")

    # Blur at quarter size: same look, a fraction of the cost of blurring at 1080x1920
    bw, bh = even(width / 4), even(height / 4)
    return (
        f"[in]split[bgsrc][fgsrc];"
        f"[bgsrc]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
        f"boxblur=12:3,scale={width}:{height},eq=brightness=-0.12:saturation=1.15[bg];"
        f"[fgsrc]{fg}[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[out]"
    )


def build_command(args, src: Source, segments: list[Segment], fmt: str,
                  output: Path) -> tuple[list[str], float]:
    """ffmpeg argv for one output format, and the output duration."""
    width, height = FORMATS[fmt]
    # Seek the input to the first segment so ffmpeg doesn't decode what comes before
    offset = min(s.start for s in segments)
    last_end = max(s.end if s.end is not None else src.duration for s in segments)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if offset > 0:
        cmd += ["-ss", f"{offset:.3f}"]
    cmd += ["-t", f"{last_end - offset:.3f}", "-i", str(args.input)]
    if args.overlay:
        cmd += ["-i", str(args.overlay)]

    graph: list[str] = []
    concat_inputs = ""
    total = 0.0
    single = len(segments) == 1
    for i, seg in enumerate(segments):
        start = seg.start - offset
        end = (seg.end if seg.end is not None else src.duration) - offset
        length = end - start
        total += length
        chain = reframe_filter(src, width, height, args.mode,
                               seg.focus if seg.focus is not None else args.focus,
                               args.zoom, args.pad_color)
        if args.mode == 'track':
            chain = chain.replace('TRACKFILE', f'track{i}.cmd').replace('TRACKLABEL', f'track{i}')
        chain = chain.replace("[in]", f"[v{i}in]", 1).replace("[out]", f"[v{i}]")
        for label in ("bgsrc", "fgsrc", "bg", "fg"):
            chain = chain.replace(f"[{label}]", f"[{label}{i}]")
        graph.append(f"[0:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS[v{i}in]")
        graph.append(chain)
        concat_inputs += f"[v{i}]"
        if src.has_audio:
            fade = "" if single else (
                f",afade=t=in:d={JOIN_FADE},"
                f"afade=t=out:st={max(0.0, length - JOIN_FADE):.3f}:d={JOIN_FADE}"
            )
            graph.append(
                f"[0:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS{fade}[a{i}]"
            )
            concat_inputs += f"[a{i}]"

    if single:
        graph.append("[v0]null[vcut]")
        if src.has_audio:
            graph.append("[a0]anull[acut]")
    else:
        audio_n = 1 if src.has_audio else 0
        outs = "[vcut][acut]" if src.has_audio else "[vcut]"
        graph.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a={audio_n}{outs}")

    if args.overlay:
        graph.append(f"[1:v]scale={width}:{height}[ovl]")
        graph.append("[vcut][ovl]overlay=0:0:format=auto,format=yuv420p[vout]")
    else:
        graph.append("[vcut]format=yuv420p[vout]")

    cmd += ["-filter_complex", ";".join(graph), "-map", "[vout]"]
    if src.has_audio:
        cmd += ["-map", "[acut]", "-c:a", "aac", "-b:a", "192k"]
    cmd += [
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", args.preset,
        *(["-r", src.fps] if not src.fps.startswith("0") else []), "-movflags", "+faststart", str(output),
    ]
    return cmd, total


def output_path(args, fmt: str, multiple: bool) -> Path:
    if args.output and not multiple:
        return Path(args.output)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent
    return out_dir / f"{Path(args.input).stem}-{fmt.replace(':', 'x')}.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reframe a 16:9 video to 9:16 / 1:1 / 4:5 for social, optionally cut down",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/cutdown.py --input out/video.mp4 --platform shorts
  uv run tools/cutdown.py --input out/video.mp4 --format 9:16 --format 1:1 --start 1:05 --end 1:50
  uv run tools/cutdown.py --input out/video.mp4 --mode crop --segment 0:04-0:11 --segment 0:42-0:58@0.25
  uv run tools/cutdown.py --input out/video.mp4 --mode pad --brand digital-samba

Platforms (default format, short-form length limit):
  shorts 9:16 / 3 min    reels 9:16 / 3 min    tiktok 9:16 / 10 min
  square 1:1             feed 4:5
""",
    )
    parser.add_argument("--input", "-i", required=True, help="Source video")
    parser.add_argument("--output", "-o",
                        help="Output file (single format only; default <input>-9x16.mp4 beside the input)")
    parser.add_argument("--output-dir", help="Directory for auto-named outputs (default: the input's)")
    parser.add_argument("--format", "-f", action="append", choices=sorted(FORMATS),
                        help="Output aspect ratio; repeat for several (default 9:16, or the platform's)")
    parser.add_argument("--platform", choices=sorted(PLATFORMS),
                        help="Sets the default format and warns past the platform's short-form limit")
    parser.add_argument("--mode", choices=("blur", "crop", "pad", "track"), default="blur",
                        help="blur: fit over a blurred copy (default); crop: fill, cut the sides; "
                             "pad: fit over a solid colour; track: follow a face or --box subject (vision extra)")
    parser.add_argument("--focus", type=arg_type(parse_focus), default=0.5,
                        help="Where crop cuts, 0 = keep the left edge, 1 = the right (default 0.5)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="blur/pad: enlarge the fitted frame, cropping its sides (e.g. 1.3)")
    cut = parser.add_mutually_exclusive_group()
    cut.add_argument("--segment", action="append", type=arg_type(parse_segment), metavar="START-END[@FOCUS]",
                     help="Keep a time range (seconds, M:SS or H:MM:SS); repeat to join several")
    cut.add_argument("--start", type=arg_type(parse_time), help="Keep from this time")
    parser.add_argument("--end", type=arg_type(parse_time), help="Keep until this time (with --start, or alone)")
    parser.add_argument("--pad-color", type=arg_type(parse_hex), help="pad mode background, #RRGGBB (default black)")
    parser.add_argument("--brand", help="pad mode: use brands/<name>/brand.json bgDark as the background")
    parser.add_argument("--overlay", help="PNG laid over the whole frame (scaled to it), e.g. a branded frame")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better (default 18)")
    parser.add_argument("--preset", default="medium", help="x264 preset (default medium)")
    parser.add_argument("--dry-run", action="store_true", help="Print the ffmpeg commands without running them")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    parser.add_argument('--box', help='track mode: x,y,width,height in source pixels at each segment start; default automatic single-face detection')
    parser.add_argument('--track-smoothing', type=float, default=.2, help='track mode: smoothing factor (0.001-1, default .2)')
    parser.add_argument('--max-pan-speed', type=float, default=600, help='track mode: maximum source pixels per second (default 600)')
    args = parser.parse_args(argv)
    if args.box and args.mode != 'track':
        parser.error('--box requires --mode track')
    if not math.isfinite(args.track_smoothing) or not .001 <= args.track_smoothing <= 1:
        parser.error('--track-smoothing must be between .001 and 1')
    if not math.isfinite(args.max_pan_speed) or not 1 <= args.max_pan_speed <= 10000:
        parser.error('--max-pan-speed must be between 1 and 10000')
    if args.box:
        try:
            args.box = [float(v) for v in args.box.split(',')]
            if len(args.box) != 4 or not all(math.isfinite(v) for v in args.box):
                raise ValueError()
        except ValueError:
            parser.error('--box needs four finite numbers: x,y,width,height')

    if args.segment and args.end is not None:
        parser.error("--end goes with --start; put the end inside each --segment instead")
    if not math.isfinite(args.zoom) or args.zoom < 1:
        parser.error("--zoom must be 1.0 or more")
    if args.brand:
        if args.pad_color:
            parser.error("pass --brand or --pad-color, not both")
        try:
            args.pad_color = brand_pad_color(args.brand)
        except ValueError as e:
            parser.error(str(e))
    args.pad_color = args.pad_color or "0x000000"
    if not args.format:
        args.format = [PLATFORMS[args.platform][0] if args.platform else "9:16"]
    args.format = list(dict.fromkeys(args.format))
    if args.output and len(args.format) > 1:
        parser.error("--output names one file; use --output-dir with several --format")
    return args


def main(argv=None) -> int:
    ensure_utf8_output()
    args = parse_args(argv)
    log = (lambda *a: print(*a, file=sys.stderr)) if args.json else print

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"Error: {tool} not found on PATH", file=sys.stderr)
            return 1
    source_path = Path(args.input)
    if not source_path.exists():
        print(f"Error: input not found: {source_path}", file=sys.stderr)
        return 1
    if args.overlay and not Path(args.overlay).exists():
        print(f"Error: overlay not found: {args.overlay}", file=sys.stderr)
        return 1

    try:
        src = probe(source_path)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.segment:
        segments = args.segment
    elif args.start is not None or args.end is not None:
        segments = [Segment(args.start or 0.0, args.end)]
    else:
        segments = [Segment(0.0, None)]
    for seg in segments:
        end = seg.end if seg.end is not None else src.duration
        if seg.start >= src.duration or end > src.duration + 0.05:
            print(f"Error: segment {seg.start:g}-{end:g}s is outside the "
                  f"{src.duration:.2f}s input", file=sys.stderr)
            return 1
        seg.end = min(end, src.duration)

    # Absolute inputs keep FFmpeg paths valid in the temporary render directory.
    args.input = str(source_path.resolve())
    if args.overlay:
        args.overlay = str(Path(args.overlay).resolve())
    if args.mode == 'track' and args.box is not None:
        from cutdown_tracking import rectangle
        try:
            rectangle(args.box, src.width, src.height)
        except ValueError as error:
            print(f'Error: {error}', file=sys.stderr)
            return 1
    tracks = None
    if args.mode == 'track' and not args.dry_run:
        try:
            from cutdown_tracking import track_subject
            tracks = track_subject(source_path, src, segments, args.box)
        except (ValueError, OSError) as error:
            print(f'Error: {error}', file=sys.stderr)
            return 1

    warnings: list[str] = []
    if src.width < src.height:
        warnings.append(f"source is already portrait ({src.width}x{src.height})")
    if args.mode in ("blur", "pad") and args.zoom == 1.0:
        scale = min(FORMATS[f][0] / src.width for f in args.format)
        if scale < 0.75:
            warnings.append(
                f"{args.mode} mode shows the source at {scale:.0%} scale, so burned captions "
                f"and small text shrink with it; --zoom 1.3 or --mode crop keep them larger"
            )

    multiple = len(args.format) > 1
    results = []
    for fmt in args.format:
        out = output_path(args, fmt, multiple)
        if out.resolve() in [source_path.resolve(), *([Path(args.overlay).resolve()] if args.overlay else [])]:
            print(f"Error: output would overwrite the input: {out}", file=sys.stderr)
            return 1
        cmd, duration = build_command(args, src, segments, fmt, out)
        if args.platform:
            limit = PLATFORMS[args.platform][1]
            if limit and duration > limit:
                warnings.append(
                    f"{duration:.1f}s is over the {args.platform} limit of {limit}s; it will "
                    f"post as a regular video. Cut it down with --start/--end or --segment"
                )
        width, height = FORMATS[fmt]
        result = {"format": fmt, "output": str(out), "width": width, "height": height,
                  "duration": round(duration, 3), "mode": args.mode, "segments": len(segments)}
        if args.dry_run:
            result["command"] = cmd
            if not args.json:
                print(shlex.join(cmd))
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            log(f"Rendering {fmt} ({width}x{height}, {duration:.1f}s, {args.mode}) -> {out}")
            try:
                with tempfile.TemporaryDirectory(prefix=f'.{out.stem}-', dir=out.parent.resolve()) as folder:
                    target = Path(folder) / out.name
                    cmd[-1] = str(target)
                    if tracks is not None:
                        from cutdown_tracking import write_crop_commands
                        for i, (seg, entries) in enumerate(zip(segments, tracks)):
                            write_crop_commands(Path(folder) / f'track{i}.cmd', entries, src, width, height,
                                                focus=seg.focus if seg.focus is not None else args.focus,
                                                zoom=args.zoom, smoothing=args.track_smoothing,
                                                max_speed=args.max_pan_speed, label=f'track{i}')
                        result['held_frames'] = sum(entry['held'] for entries in tracks for entry in entries)
                    proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, encoding='utf-8')
                    if proc.returncode != 0:
                        raise ValueError(f'ffmpeg failed for {fmt}:\n{proc.stderr.strip()}')
                    if not target.is_file() or target.stat().st_size == 0:
                        raise ValueError('FFmpeg produced an empty output')
                    os.replace(target, out)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                print(f'Error: {error}', file=sys.stderr)
                return 1
            result["size_mb"] = round(out.stat().st_size / 1_048_576, 2)
        results.append(result)

    warnings = list(dict.fromkeys(warnings))
    for w in warnings:
        log(f"Warning: {w}")
    if args.json:
        print(json.dumps({"source": {"path": str(source_path), "width": src.width,
                                     "height": src.height, "duration": round(src.duration, 3)},
                          "outputs": results, "warnings": warnings}, indent=2))
    elif not args.dry_run:
        for r in results:
            log(f"Done: {r['output']} ({r['size_mb']} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
