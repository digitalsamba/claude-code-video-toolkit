#!/usr/bin/env python3
"""
Generate voiceover audio using ElevenLabs TTS.

Usage:
    # From script file
    python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3

    # From stdin (for AI piping)
    echo "Hello world" | python tools/voiceover.py --output voiceover.mp3

    # With custom voice
    python tools/voiceover.py --script script.txt --voice-id ABC123 --output out.mp3

    # JSON output for machine parsing
    python tools/voiceover.py --script script.txt --output out.mp3 --json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import VoiceSettings, save
from elevenlabs.client import ElevenLabs

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_elevenlabs_api_key, get_voice_id


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate voiceover using ElevenLabs TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3
  echo "Hello" | python tools/voiceover.py --output hello.mp3
  python tools/voiceover.py --script script.txt --voice-id ABC123 --output out.mp3 --json
        """,
    )
    parser.add_argument(
        "--script",
        "-s",
        type=str,
        help="Path to script file (reads from stdin if not provided)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output audio file path (.mp3)",
    )
    parser.add_argument(
        "--voice-id",
        "-v",
        type=str,
        help="ElevenLabs voice ID (uses default from toolkit-registry.json if not provided)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="eleven_multilingual_v2",
        choices=["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"],
        help="ElevenLabs model (default: eleven_multilingual_v2). eleven_v3 is alpha.",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=0.85,
        help="Voice stability 0-1 (default: 0.85, higher = more consistent)",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.95,
        help="Similarity boost 0-1 (default: 0.95, higher = closer to original)",
    )
    parser.add_argument(
        "--style",
        type=float,
        default=0.0,
        help="Style exaggeration 0-1 (default: 0.0, lower = more neutral)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
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


def read_script(script_path: str | None) -> str:
    """Read script from file or stdin."""
    if script_path:
        with open(script_path) as f:
            return f.read().strip()
    else:
        if sys.stdin.isatty():
            print("Reading script from stdin (Ctrl+D to end):", file=sys.stderr)
        return sys.stdin.read().strip()


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

    # Get API key
    api_key = get_elevenlabs_api_key()
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment", file=sys.stderr)
        sys.exit(1)

    # Get voice ID
    voice_id = args.voice_id or get_voice_id()
    if not voice_id:
        print(
            "Error: No voice ID provided and none found in toolkit-registry.json",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read script
    script = read_script(args.script)
    if not script:
        print("Error: Empty script", file=sys.stderr)
        sys.exit(1)

    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dry run mode
    if args.dry_run:
        result = {
            "dry_run": True,
            "voice_id": voice_id,
            "model": args.model,
            "script_length": len(script),
            "script_chars": len(script),
            "output": str(output_path),
            "settings": {
                "stability": args.stability,
                "similarity": args.similarity,
                "style": args.style,
                "speed": args.speed,
            },
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Would generate voiceover:")
            print(f"  Voice ID: {voice_id}")
            print(f"  Model: {args.model}")
            print(f"  Script: {len(script)} characters")
            print(f"  Output: {output_path}")
        return

    # Generate voiceover
    if not args.json:
        print(f"Generating voiceover ({len(script)} chars)...", file=sys.stderr)

    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        text=script,
        voice_id=voice_id,
        model_id=args.model,
        voice_settings=VoiceSettings(
            stability=args.stability,
            similarity_boost=args.similarity,
            style=args.style,
            speed=args.speed,
        ),
    )

    save(audio, str(output_path))

    # Get duration if ffprobe available
    duration = get_audio_duration(str(output_path))

    # Output result
    result = {
        "success": True,
        "output": str(output_path),
        "voice_id": voice_id,
        "model": args.model,
        "script_chars": len(script),
    }
    if duration:
        result["duration_seconds"] = round(duration, 2)
        result["duration_frames_30fps"] = int(duration * 30)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Voiceover saved to: {output_path}", file=sys.stderr)
        if duration:
            print(
                f"Duration: {duration:.2f}s ({int(duration * 30)} frames @ 30fps)",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
