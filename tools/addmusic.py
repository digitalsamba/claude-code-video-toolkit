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
    python tools/addmusic.py -i video.mp4 -m background.mp3 -o output.mp4

    # Generate and add music with ElevenLabs
    python tools/addmusic.py -i video.mp4 -p "Subtle corporate ambient" -o output.mp4

    # Custom volumes with fades
    python tools/addmusic.py -i video.mp4 -m music.mp3 --music-volume 0.2 --fade-in 2 --fade-out 3 -o output.mp4

    # Dry run to preview
    python tools/addmusic.py -i video.mp4 -p "Upbeat tech" --dry-run
"""
from __future__ import annotations

import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add background music to videos using ElevenLabs or existing audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/addmusic.py -i video.mp4 -m background.mp3 -o output.mp4
  python tools/addmusic.py -i video.mp4 -p "Subtle corporate ambient" -o output.mp4
  python tools/addmusic.py -i video.mp4 -m music.mp3 --music-volume 0.2 --fade-in 2 --fade-out 3 -o output.mp4

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
    return parser.parse_args()


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


def mix_audio_with_video(
    video_path: str,
    music_path: str,
    output_path: str,
    original_volume: float = 1.0,
    music_volume: float = 0.15,
    fade_in: float = 0,
    fade_out: float = 0,
    video_duration: float | None = None,
    verbose: bool = True,
) -> bool:
    """Mix original video audio with background music using FFmpeg."""
    if verbose:
        print("Mixing audio tracks...", file=sys.stderr)

    # Build the filter for the music track
    music_filter_parts = [f"volume={music_volume}"]

    # Add fade in if specified
    if fade_in > 0:
        music_filter_parts.append(f"afade=t=in:d={fade_in}")

    # Add fade out if specified (need video duration for start time)
    if fade_out > 0 and video_duration:
        fade_out_start = max(0, video_duration - fade_out)
        music_filter_parts.append(f"afade=t=out:st={fade_out_start}:d={fade_out}")

    music_filter = ",".join(music_filter_parts)

    # Build filter_complex
    filter_complex = (
        f"[0:a]volume={original_volume}[a1];"
        f"[1:a]{music_filter}[a2];"
        f"[a1][a2]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", video_path,
            "-i", music_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",  # Video from first input
            "-map", "[aout]",  # Mixed audio
            "-c:v", "copy",  # Copy video codec
            "-c:a", "aac",  # Encode audio as AAC
            "-b:a", "192k",  # Audio bitrate
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}", file=sys.stderr)
        return False

    return True


def main():
    load_dotenv()
    args = parse_args()

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
    if args.fade_in < 0 or args.fade_out < 0:
        print("Error: Fade values must be non-negative", file=sys.stderr)
        sys.exit(1)

    # Get video duration
    video_duration = get_media_duration(args.input)
    if not video_duration:
        print("Error: Could not determine video duration", file=sys.stderr)
        sys.exit(1)

    # Check if we need ElevenLabs
    needs_api = args.prompt is not None
    api_key = None

    if needs_api:
        api_key = get_elevenlabs_api_key()
        if not api_key:
            print(
                "Error: No ElevenLabs API key found.\n"
                "\n"
                "To generate AI music:\n"
                "  echo \"ELEVENLABS_API_KEY=your_key\" >> .env\n"
                "\n"
                "Alternative: Use --music flag with your own audio file:\n"
                "  python3 tools/addmusic.py --input video.mp4 --music your_music.mp3 --output output.mp4",
                file=sys.stderr,
            )
            sys.exit(1)

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        ):
            print("Error: Failed to mix audio", file=sys.stderr)
            sys.exit(1)

        # Get output duration
        output_duration = get_media_duration(str(output_path))

        # Build result
        result = {
            "success": True,
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
