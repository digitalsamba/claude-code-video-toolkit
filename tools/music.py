#!/usr/bin/env python3
"""
Generate background music using ElevenLabs Music API.

Usage:
    # Generate background music
    python tools/music.py --prompt "Upbeat tech corporate" --duration 120 --output music.mp3

    # JSON output for machine parsing
    python tools/music.py --prompt "Calm ambient" --duration 60 --output bg.mp3 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_elevenlabs_api_key


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate background music using ElevenLabs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/music.py --prompt "Subtle corporate tech" --duration 150 --output public/audio/bg-music.mp3
  python tools/music.py --prompt "Upbeat indie rock" --duration 60 --output music.mp3 --json

Prompt tips:
  Include genre, mood, instruments, tempo, and use case.
  Example: "Subtle corporate technology background music, soft synth pads, minimal beats, professional presentation"
        """,
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        required=True,
        help="Music description prompt (genre, mood, instruments, tempo)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        required=True,
        help="Music duration in seconds (10-300)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output audio file path (.mp3)",
    )
    parser.add_argument(
        "--vocals",
        action="store_true",
        help="Allow vocals (default: instrumental only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (for machine parsing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making API calls",
    )
    return parser.parse_args()


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration using ffprobe if available."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
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


def main():
    load_dotenv()
    args = parse_args()

    # Validate duration
    if args.duration < 10 or args.duration > 300:
        print("Error: Duration must be between 10 and 300 seconds", file=sys.stderr)
        sys.exit(1)

    # Get API key
    api_key = get_elevenlabs_api_key()
    if not api_key:
        print(
            "Error: No ElevenLabs API key found.\n"
            "\n"
            "To generate AI music:\n"
            "  echo \"ELEVENLABS_API_KEY=your_key\" >> .env\n"
            "\n"
            "Music is optional — videos render fine without background music.\n"
            "You can also add your own music file manually with tools/addmusic.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dry run mode
    if args.dry_run:
        result = {
            "dry_run": True,
            "prompt": args.prompt,
            "duration_seconds": args.duration,
            "instrumental": not args.vocals,
            "output": str(output_path),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would generate music:")
            print(f"  Prompt: {args.prompt}")
            print(f"  Duration: {args.duration}s")
            print(f"  Instrumental: {not args.vocals}")
            print(f"  Output: {output_path}")
        return

    # Generate music
    if not args.json:
        print(f"Generating {args.duration}s of music...", file=sys.stderr)

    client = ElevenLabs(api_key=api_key)

    music = client.music.compose(
        prompt=args.prompt,
        music_length_ms=args.duration * 1000,
        force_instrumental=not args.vocals,
    )

    with open(output_path, "wb") as f:
        for chunk in music:
            f.write(chunk)

    # Get actual duration if ffprobe available
    actual_duration = get_audio_duration(str(output_path))

    # Output result
    result = {
        "success": True,
        "output": str(output_path),
        "prompt": args.prompt,
        "requested_duration": args.duration,
    }
    if actual_duration:
        result["actual_duration_seconds"] = round(actual_duration, 2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Music saved to: {output_path}", file=sys.stderr)
        if actual_duration:
            print(f"Duration: {actual_duration:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
