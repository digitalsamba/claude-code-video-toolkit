#!/usr/bin/env python3
"""
Generate sound effects using ElevenLabs Sound Effects API.

Usage:
    # Generate a single sound effect
    python tools/sfx.py --prompt "Soft whoosh transition" --duration 1.5 --output whoosh.mp3

    # Generate with stronger prompt adherence
    python tools/sfx.py --prompt "Thunder crack" --duration 3 --influence 0.5 --output thunder.mp3

    # JSON output for machine parsing
    python tools/sfx.py --prompt "UI click" --duration 0.5 --output click.mp3 --json
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


# Common SFX presets
PRESETS = {
    "whoosh": {
        "prompt": "Soft subtle whoosh, gentle air movement, UI transition sound",
        "duration": 1.5,
    },
    "click": {
        "prompt": "Soft UI click, subtle tap, gentle interface sound, minimal",
        "duration": 0.5,
    },
    "chime": {
        "prompt": "Gentle success notification chime, soft pleasant ding, achievement unlocked sound, subtle and professional",
        "duration": 2.0,
    },
    "error": {
        "prompt": "Soft error buzz, gentle warning tone, subtle negative feedback",
        "duration": 1.0,
    },
    "pop": {
        "prompt": "Soft pop sound, bubble pop, light playful UI sound",
        "duration": 0.3,
    },
    "slide": {
        "prompt": "Smooth slide sound, panel sliding, drawer opening, subtle movement",
        "duration": 0.8,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sound effects using ElevenLabs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python tools/sfx.py --prompt "Thunder rumbling" --duration 5 --output thunder.mp3
  python tools/sfx.py --preset whoosh --output public/audio/sfx-whoosh.mp3
  python tools/sfx.py --prompt "Footsteps on wood" --duration 3 --influence 0.5 --output steps.mp3

Available presets: {', '.join(PRESETS.keys())}
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="Sound effect description",
    )
    group.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Use a preset sound effect",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        help="Duration in seconds (0.5-22, required unless using preset)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output audio file path (.mp3)",
    )
    parser.add_argument(
        "--influence",
        "-i",
        type=float,
        default=0.3,
        help="Prompt influence 0-1 (default: 0.3, higher = stricter adherence)",
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
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
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

    # List presets mode
    if args.list_presets:
        print("Available presets:")
        for name, preset in PRESETS.items():
            print(f"  {name}: {preset['prompt']} ({preset['duration']}s)")
        return

    # Resolve preset or custom prompt
    if args.preset:
        preset = PRESETS[args.preset]
        prompt = preset["prompt"]
        duration = args.duration or preset["duration"]
    else:
        prompt = args.prompt
        if not args.duration:
            print("Error: --duration is required when not using a preset", file=sys.stderr)
            sys.exit(1)
        duration = args.duration

    # Validate duration
    if duration < 0.5 or duration > 22:
        print("Error: Duration must be between 0.5 and 22 seconds", file=sys.stderr)
        sys.exit(1)

    # Get API key
    api_key = get_elevenlabs_api_key()
    if not api_key:
        print(
            "Error: No ElevenLabs API key found.\n"
            "\n"
            "To generate AI sound effects:\n"
            "  echo \"ELEVENLABS_API_KEY=your_key\" >> .env\n"
            "\n"
            "Sound effects are optional — videos render fine without them.",
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
            "prompt": prompt,
            "duration_seconds": duration,
            "prompt_influence": args.influence,
            "output": str(output_path),
        }
        if args.preset:
            result["preset"] = args.preset
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would generate sound effect:")
            if args.preset:
                print(f"  Preset: {args.preset}")
            print(f"  Prompt: {prompt}")
            print(f"  Duration: {duration}s")
            print(f"  Influence: {args.influence}")
            print(f"  Output: {output_path}")
        return

    # Generate sound effect
    if not args.json:
        print(f"Generating sound effect ({duration}s)...", file=sys.stderr)

    client = ElevenLabs(api_key=api_key)

    sfx = client.text_to_sound_effects.convert(
        text=prompt,
        duration_seconds=duration,
        prompt_influence=args.influence,
    )

    with open(output_path, "wb") as f:
        for chunk in sfx:
            f.write(chunk)

    # Get actual duration if ffprobe available
    actual_duration = get_audio_duration(str(output_path))

    # Output result
    result = {
        "success": True,
        "output": str(output_path),
        "prompt": prompt,
        "requested_duration": duration,
    }
    if args.preset:
        result["preset"] = args.preset
    if actual_duration:
        result["actual_duration_seconds"] = round(actual_duration, 2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Sound effect saved to: {output_path}", file=sys.stderr)
        if actual_duration:
            print(f"Duration: {actual_duration:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
