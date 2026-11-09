#!/usr/bin/env python3
"""
Add background music to existing videos using ElevenLabs or existing audio files.

Pipeline:
    1. Validate input video and get duration
    2. Get music (generate via ElevenLabs or use existing file)
    3. Mix original audio with music track (with volume controls and optional fades)
    4. Mux mixed audio back into video

Usage:
    # Add existing music file
    uv run tools/addmusic.py -i video.mp4 -m background.mp3 -o output.mp4

    # Generate and add music with ElevenLabs
    uv run tools/addmusic.py -i video.mp4 -p "Subtle corporate ambient" -o output.mp4

    # Custom volumes with fades
    uv run tools/addmusic.py -i video.mp4 -m music.mp3 --music-volume 0.2 --fade-in 2 --fade-out 3 -o output.mp4

    # Dry run to preview
    uv run tools/addmusic.py -i video.mp4 -p "Upbeat tech" --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_elevenlabs_api_key


# ElevenLabs music generation limit
MAX_MUSIC_DURATION = 300


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Add background music to videos using ElevenLabs or existing audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/addmusic.py -i video.mp4 -m background.mp3 -o output.mp4
  uv run tools/addmusic.py -i video.mp4 -p "Subtle corporate ambient" -o output.mp4
  uv run tools/addmusic.py -i video.mp4 -m music.mp3 --music-volume 0.2 --fade-in 2 --fade-out 3 -o output.mp4

Prompt tips (for ElevenLabs generation):
  Include genre, mood, instruments, tempo, and use case.
  Example: "Subtle corporate technology background music, soft synth pads, minimal beats"
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input video file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output video file path",
    )

    # Music source (mutually exclusive)
    music_source = parser.add_mutually_exclusive_group(required=True)
    music_source.add_argument(
        "--music",
        "-m",
        type=str,
        help="Existing music file to use",
    )
    music_source.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="ElevenLabs music generation prompt (auto-matches video duration)",
    )

    # Volume controls
    parser.add_argument(
        "--music-volume",
        type=float,
        default=0.15,
        help="Music volume level 0.0-1.0 (default: 0.15)",
    )
    parser.add_argument(
        "--original-volume",
        type=float,
        default=1.0,
        help="Original audio volume level 0.0-1.0 (default: 1.0)",
    )

    # Fade options
    parser.add_argument(
        "--fade-in",
        type=float,
        default=0,
        help="Music fade-in duration in seconds (default: 0, no fade)",
    )
    parser.add_argument(
        "--fade-out",
        type=float,
        default=0,
        help="Music fade-out duration in seconds (default: 0, no fade)",
    )

    # Music generation options
    parser.add_argument(
        "--vocals",
        action="store_true",
        help="Allow vocals in generated music (default: instrumental only)",
    )

    # Standard flags
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files (extracted audio, generated music)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (for machine parsing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls or writes",
    )
    parser.add_argument('--duck', action='store_true', help='Duck background music under speech using FFmpeg sidechain compression')
    parser.add_argument('--speech', help='Isolated narration replacing the original audio and controlling --duck')
    parser.add_argument('--events', type=Path, help='JSON array of {path, atSeconds, gain}; paths relative to the JSON file')
    parser.add_argument('--duck-threshold', type=float, default=.03, help='Sidechain threshold, 0.000976 to 1 (default .03)')
    parser.add_argument('--duck-ratio', type=float, default=8, help='Compression ratio, 1 to 20 (default 8)')
    parser.add_argument('--duck-attack', type=float, default=20, help='Attack in milliseconds (default 20)')
    parser.add_argument('--duck-release', type=float, default=400, help='Release in milliseconds (default 400)')
    return parser.parse_args(argv)


def get_media_duration(file_path: str) -> float | None:
    """Get media duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass
    return None


def generate_music(
    client: ElevenLabs,
    prompt: str,
    duration_seconds: int,
    output_path: str,
    force_instrumental: bool = True,
    verbose: bool = True,
) -> bool:
    """Generate music using ElevenLabs Music API."""
    if verbose:
        print(f"Generating {duration_seconds}s of music...", file=sys.stderr)

    try:
        music = client.music.compose(
            prompt=prompt,
            music_length_ms=duration_seconds * 1000,
            force_instrumental=force_instrumental,
        )

        with open(output_path, "wb") as f:
            for chunk in music:
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Music generation error: {e}", file=sys.stderr)
        return False


def has_audio(path: str) -> bool:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type', '-of', 'json', str(path)],
        capture_output=True, text=True, encoding='utf-8', check=True,
    )
    return any(s['codec_type'] == 'audio' for s in json.loads(result.stdout)['streams'])


def finite(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{label} must be a finite number between {low} and {high}')
    return value


def load_events(path, duration):
    if path is None:
        return []
    path = Path(path)
    events = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(events, list) or len(events) > 128:
        raise ValueError('events must be an array of at most 128 sound effects')
    normalized = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get('path'), str) or not event['path'].strip():
            raise ValueError('Each effect needs a path and atSeconds')
        effect = (path.parent / event['path']).resolve()
        if not effect.is_file():
            raise ValueError(f'Effect file not found: {effect}')
        if not has_audio(str(effect)):
            raise ValueError(f'Effect has no audio: {effect}')
        normalized.append({'path': str(effect),
                           'atSeconds': finite(event.get('atSeconds'), 'atSeconds', 0, duration - .001),
                           'gain': finite(event.get('gain', 1), 'effect gain', 0, 4)})
    return normalized


def build_mix_command(video_path, music_path, output_path, *, original_volume=1., music_volume=.15,
                      fade_in=0., fade_out=0., video_duration=None, duck=False, speech=None,
                      events=None, threshold=.03, ratio=8., attack=20., release=400.):
    duration = finite(video_duration or get_media_duration(str(video_path)), 'video duration', .001, 86400)
    for label, value, low, high in [('original volume', original_volume, 0, 1), ('music volume', music_volume, 0, 1),
                                   ('fade in', fade_in, 0, duration), ('fade out', fade_out, 0, duration),
                                   ('threshold', threshold, .000976, 1), ('ratio', ratio, 1, 20),
                                   ('attack', attack, .01, 2000), ('release', release, .01, 9000)]:
        finite(value, label, low, high)
    events = events or []
    if not isinstance(events, list) or len(events) > 128:
        raise ValueError('events must be an array of at most 128 sound effects')
    inputs = [video_path, music_path, speech, *[event['path'] for event in events]]
    if Path(output_path).resolve() in [Path(p).resolve() for p in inputs if p is not None]:
        raise ValueError('Output cannot replace an input')
    voice_audio = has_audio(str(speech or video_path))
    if (duck or speech) and not voice_audio:
        raise ValueError('--duck requires video audio or an audio file supplied with --speech')
    command = ['ffmpeg', '-y', '-v', 'error', '-i', str(video_path), '-stream_loop', '-1', '-i', str(music_path)]
    voice_index = 0
    if speech:
        voice_index = 2
        command += ['-i', str(speech)]
    music_filter = f'volume={music_volume},atrim=duration={duration},asetpts=PTS-STARTPTS'
    if fade_in:
        music_filter += f',afade=t=in:d={fade_in}'
    if fade_out:
        music_filter += f',afade=t=out:st={duration - fade_out}:d={fade_out}'
    filters = [f'[1:a:0]{music_filter}[music]']
    if voice_audio:
        filters.append(f'[{voice_index}:a:0]asetpts=PTS-STARTPTS,apad,atrim=duration={duration}' +
                       (',asplit=2[voice][control]' if duck else '[voice]'))
    else:
        filters.append(f'anullsrc=r=48000:cl=stereo,atrim=duration={duration}[voice]')
    filters.append(f'[voice]volume={original_volume}[spoken]')
    if duck:
        filters.append(f'[music][control]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[bed]')
    else:
        filters.append('[music]anull[bed]')
    labels = ['[spoken]', '[bed]']
    for i, event in enumerate(events):
        when = finite(event['atSeconds'], 'atSeconds', 0, duration - .001)
        gain = finite(event.get('gain', 1), 'effect gain', 0, 4)
        command += ['-i', str(event['path'])]
        index = i + (3 if speech else 2)
        filters.append(f'[{index}:a:0]asetpts=PTS-STARTPTS,volume={gain},adelay={round(when * 1000)}:all=1[fx{i}]')
        labels.append(f'[fx{i}]')
    # amix takes the first input's rate and layout, so a mono 22.05 kHz narration would fold stereo music
    # to mono. Bring every input to 48 kHz stereo before mixing.
    mixed = []
    for i, label in enumerate(labels):
        filters.append(f'{label}aformat=sample_rates=48000:channel_layouts=stereo[mix{i}]')
        mixed.append(f'[mix{i}]')
    filters.append(''.join(mixed) + f'amix=inputs={len(mixed)}:normalize=0:duration=first,alimiter=limit=0.95:latency=1[out]')
    return [*command, '-filter_complex', ';'.join(filters), '-map', '0:v:0', '-map', '[out]',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', str(duration), '-movflags', '+faststart', str(output_path)]


def mix_audio_with_video(video_path, music_path, output_path, *, verbose=True, **options) -> bool:
    """Mix at a fixed level or duck under speech, publishing only a complete result."""
    if verbose:
        print('Mixing audio tracks...', file=sys.stderr)
    temporary = None
    try:
        command = build_mix_command(video_path, music_path, output_path, **options)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f'.{output.stem}-', suffix=output.suffix, dir=output.parent)
        os.close(fd)
        temporary = Path(name)
        command[-1] = str(temporary)
        subprocess.run(command, capture_output=True, text=True, encoding='utf-8', check=True)
        if temporary.stat().st_size == 0:
            raise ValueError('FFmpeg produced an empty output')
        os.replace(temporary, output)
        return True
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'Mix failed: {getattr(error, "stderr", None) or error}', file=sys.stderr)
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    from win_encoding import ensure_utf8_output
    ensure_utf8_output()
    load_dotenv()
    args = parse_args(argv)

    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Validate music file if provided
    if args.music and not Path(args.music).exists():
        print(f"Error: Music file not found: {args.music}", file=sys.stderr)
        sys.exit(1)

    # Validate volume ranges
    if not 0.0 <= args.music_volume <= 1.0:
        print("Error: --music-volume must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.original_volume <= 1.0:
        print("Error: --original-volume must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    # Validate fade values
    if not all(math.isfinite(f) and f >= 0 for f in (args.fade_in, args.fade_out)):
        print("Error: Fade values must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Get video duration
    video_duration = get_media_duration(args.input)
    if not video_duration:
        print("Error: Could not determine video duration", file=sys.stderr)
        sys.exit(1)

    try:
        events = load_events(args.events, video_duration)
        if args.events and Path(args.output).resolve() == args.events.resolve():
            raise ValueError('Output cannot replace the events JSON')
        # Validate before generating paid music or creating output directories.
        build_mix_command(args.input, args.music or 'generated_music.mp3', args.output,
                          video_duration=video_duration, original_volume=args.original_volume,
                          music_volume=args.music_volume, fade_in=args.fade_in, fade_out=args.fade_out,
                          duck=args.duck, speech=args.speech, events=events, threshold=args.duck_threshold,
                          ratio=args.duck_ratio, attack=args.duck_attack, release=args.duck_release)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)

    # Check if we need ElevenLabs
    needs_api = args.prompt is not None
    api_key = None

    if needs_api and not args.dry_run:
        api_key = get_elevenlabs_api_key()
        if not api_key:
            print(
                "Error: No ElevenLabs API key found.\n"
                "\n"
                "To generate AI music:\n"
                "  echo \"ELEVENLABS_API_KEY=your_key\" >> .env\n"
                "\n"
                "Alternative: Use --music flag with your own audio file:\n"
                "  uv run tools/addmusic.py --input video.mp4 --music your_music.mp3 --output output.mp4",
                file=sys.stderr,
            )
            sys.exit(1)

    # Prepare output directory
    output_path = Path(args.output)

    verbose = not args.json

    # Calculate music duration for generation
    music_duration = min(int(video_duration) + 1, MAX_MUSIC_DURATION)
    duration_warning = None
    if video_duration > MAX_MUSIC_DURATION:
        duration_warning = f"Video is {video_duration:.1f}s but music capped at {MAX_MUSIC_DURATION}s (ElevenLabs limit)"

    # Dry run mode
    if args.dry_run:
        result = {
            "dry_run": True,
            "duck": args.duck,
            "events": events,
            "input": args.input,
            "output": str(output_path),
            "video_duration": round(video_duration, 2),
            "music_source": "generate" if args.prompt else "file",
            "original_volume": args.original_volume,
            "music_volume": args.music_volume,
            "fade_in": args.fade_in,
            "fade_out": args.fade_out,
        }
        if args.prompt:
            result["prompt"] = args.prompt
            result["music_duration"] = music_duration
            result["instrumental"] = not args.vocals
        else:
            result["music_file"] = args.music

        if duration_warning:
            result["warning"] = duration_warning

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would add background music:")
            print(f"  Input: {args.input}")
            print(f"  Video duration: {video_duration:.2f}s")
            if args.prompt:
                print(f"  Generate music: \"{args.prompt}\"")
                print(f"  Music duration: {music_duration}s")
            else:
                print(f"  Music file: {args.music}")
            print(f"  Original volume: {args.original_volume}")
            print(f"  Music volume: {args.music_volume}")
            if args.fade_in > 0:
                print(f"  Fade in: {args.fade_in}s")
            if args.fade_out > 0:
                print(f"  Fade out: {args.fade_out}s")
            print(f"  Output: {output_path}")
            if duration_warning:
                print(f"  Warning: {duration_warning}")
        return

    # Create temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="addmusic_")
    generated_music = Path(temp_dir) / "generated_music.mp3"

    try:
        # Get music file path
        if args.prompt:
            # Generate music via ElevenLabs
            client = ElevenLabs(api_key=api_key)

            if verbose and duration_warning:
                print(f"Warning: {duration_warning}", file=sys.stderr)

            if not generate_music(
                client,
                args.prompt,
                music_duration,
                str(generated_music),
                force_instrumental=not args.vocals,
                verbose=verbose,
            ):
                print("Error: Failed to generate music", file=sys.stderr)
                sys.exit(1)

            music_path = str(generated_music)
        else:
            music_path = args.music

        # Mix audio with video
        if not mix_audio_with_video(
            args.input,
            music_path,
            str(output_path),
            original_volume=args.original_volume,
            music_volume=args.music_volume,
            fade_in=args.fade_in,
            fade_out=args.fade_out,
            video_duration=video_duration,
            verbose=verbose,
            duck=args.duck, speech=args.speech, events=events,
            threshold=args.duck_threshold, ratio=args.duck_ratio,
            attack=args.duck_attack, release=args.duck_release,
        ):
            print("Error: Failed to mix audio", file=sys.stderr)
            sys.exit(1)

        # Get output duration
        output_duration = get_media_duration(str(output_path))

        # Build result
        result = {
            "success": True,
            "duck": args.duck,
            "events": events,
            "input": args.input,
            "output": str(output_path),
            "video_duration": round(video_duration, 2),
            "music_source": "generated" if args.prompt else "file",
            "original_volume": args.original_volume,
            "music_volume": args.music_volume,
        }

        if args.prompt:
            result["prompt"] = args.prompt
            result["music_duration"] = music_duration
        else:
            result["music_file"] = args.music

        if args.fade_in > 0:
            result["fade_in"] = args.fade_in
        if args.fade_out > 0:
            result["fade_out"] = args.fade_out

        if output_duration:
            result["output_duration"] = round(output_duration, 2)

        if duration_warning:
            result["warning"] = duration_warning

        # Keep temp files if requested
        if args.keep_temp and args.prompt:
            kept_music = output_path.parent / f"{output_path.stem}_music.mp3"
            import shutil
            shutil.copy(generated_music, kept_music)
            result["temp_files"] = {"generated_music": str(kept_music)}

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Background music added: {output_path}", file=sys.stderr)
            if output_duration:
                print(f"Duration: {output_duration:.2f}s", file=sys.stderr)
            if duration_warning:
                print(f"Warning: {duration_warning}", file=sys.stderr)

    finally:
        # Cleanup temp directory
        if not args.keep_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
