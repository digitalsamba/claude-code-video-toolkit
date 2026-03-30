#!/usr/bin/env python3
"""
Generate speech using MiniMax Cloud TTS API.

MiniMax offers high-quality text-to-speech with multiple voice presets, supporting
both English and Chinese voices. No GPU required — runs entirely in the cloud.

Usage:
    # Basic usage
    python tools/minimax_tts.py --text "Hello world" --output hello.mp3

    # Choose a voice
    python tools/minimax_tts.py --text "Hello world" --voice English_Graceful_Lady --output hello.mp3

    # Choose model (hd or turbo)
    python tools/minimax_tts.py --text "Hello world" --model turbo --output fast.mp3

    # List available voices
    python tools/minimax_tts.py --list-voices

    # JSON output for machine parsing
    python tools/minimax_tts.py --text "Hello world" --output hello.mp3 --json

Setup:
    1. Get an API key from https://www.minimaxi.com/
    2. Add to .env:
       echo "MINIMAX_API_KEY=your_key_here" >> .env
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

# MiniMax TTS API endpoint
MINIMAX_TTS_API_URL = "https://api.minimax.io/v1/t2a_v2"

# speech-2.8-hd: high quality, slower
# speech-2.8-turbo: faster, slightly lower quality
MINIMAX_TTS_MODELS = {
    "hd": "speech-2.8-hd",
    "turbo": "speech-2.8-turbo",
}

# Verified voice IDs
MINIMAX_VOICES = {
    # English voices
    "English_Graceful_Lady": "English",
    "English_Insightful_Speaker": "English",
    "English_radiant_girl": "English",
    "English_Persuasive_Man": "English",
    "English_Lucky_Robot": "English",
    # Bilingual / Chinese voices
    "Wise_Woman": "Chinese",
    "cute_boy": "Chinese",
    "lovely_girl": "Chinese",
    "Friendly_Person": "Chinese",
    "Inspirational_girl": "Chinese",
    "Deep_Voice_Man": "Chinese",
    "sweet_girl": "Chinese",
}

# Default voice for English content
DEFAULT_VOICE = "English_Graceful_Lady"
DEFAULT_MODEL = "hd"


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
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


def generate_audio(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    speed: float = 1.0,
    volume: float = 1.0,
    pitch: int = 0,
    timeout: int = 60,
    verbose: bool = True,
) -> dict:
    """Generate audio using MiniMax Cloud TTS API.

    This is the main entry point, importable by voiceover.py.
    Returns dict with: success, output, duration_seconds, duration_frames_30fps

    Args:
        text: Text to synthesize.
        output_path: Path to save the output audio file (.mp3).
        voice: Voice ID (see MINIMAX_VOICES).
        model: Model shorthand — "hd" or "turbo".
        speed: Speech speed multiplier (0.5-2.0, default 1.0).
        volume: Volume level (0.1-10.0, default 1.0).
        pitch: Pitch shift in semitones (-12 to 12, default 0).
        timeout: Request timeout in seconds.
        verbose: Print progress messages.
    """
    from config import find_workspace_root

    start_time = time.time()

    # Resolve API key
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        from dotenv import load_dotenv

        load_dotenv(find_workspace_root() / ".env")
        api_key = os.getenv("MINIMAX_API_KEY")

    if not api_key:
        return {
            "success": False,
            "error": (
                "MINIMAX_API_KEY not set. Get one at https://www.minimaxi.com/ "
                "and add to .env:\n  echo \"MINIMAX_API_KEY=your_key\" >> .env"
            ),
        }

    # Resolve model name
    model_id = MINIMAX_TTS_MODELS.get(model, model)

    if verbose:
        print(f"Generating speech with MiniMax TTS ({model_id})...", file=sys.stderr)
        print(f"  Voice: {voice}", file=sys.stderr)
        print(f"  Text: {len(text)} chars", file=sys.stderr)

    # Build request payload
    payload = {
        "model": model_id,
        "text": text,
        "voice_setting": {
            "voice_id": voice,
            "speed": speed,
            "vol": volume,
            "pitch": pitch,
        },
        "audio_setting": {
            "format": "mp3",
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.post(
            MINIMAX_TTS_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Request timed out after {timeout}s"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}

    if response.status_code != 200:
        return {
            "success": False,
            "error": f"API returned HTTP {response.status_code}: {response.text[:500]}",
        }

    try:
        result = response.json()
    except json.JSONDecodeError:
        return {"success": False, "error": "Invalid JSON response from API"}

    # Check for API-level errors
    base_resp = result.get("base_resp", {})
    if base_resp.get("status_code", 0) != 0:
        return {
            "success": False,
            "error": f"API error {base_resp.get('status_code')}: {base_resp.get('status_msg', 'Unknown')}",
        }

    # Extract audio data (hex-encoded bytes)
    audio_hex = result.get("data", {}).get("audio")
    if not audio_hex:
        return {"success": False, "error": "No audio data in response"}

    # Decode hex to bytes and save
    try:
        audio_bytes = bytes.fromhex(audio_hex)
    except ValueError as e:
        return {"success": False, "error": f"Failed to decode audio data: {e}"}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(audio_bytes)

    elapsed = time.time() - start_time
    duration = get_audio_duration(output_path)

    if verbose:
        size_kb = Path(output_path).stat().st_size // 1024
        print(f"  Saved: {output_path} ({size_kb}KB)", file=sys.stderr)
        if duration:
            print(f"  Duration: {duration:.1f}s", file=sys.stderr)
        print(f"  Elapsed: {elapsed:.1f}s", file=sys.stderr)

    result_dict = {
        "success": True,
        "output": output_path,
        "script_chars": len(text),
    }
    if duration:
        result_dict["duration_seconds"] = round(duration, 2)
        result_dict["duration_frames_30fps"] = int(duration * 30)

    return result_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using MiniMax Cloud TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/minimax_tts.py --text "Hello world" --output hello.mp3

  # Choose a voice
  python tools/minimax_tts.py --text "Hello world" --voice English_Persuasive_Man --output hello.mp3

  # Fast generation with turbo model
  python tools/minimax_tts.py --text "Hello world" --model turbo --output fast.mp3

  # List voices
  python tools/minimax_tts.py --list-voices
        """,
    )

    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output audio file path (.mp3)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Voice ID (default: {DEFAULT_VOICE}). Use --list-voices to see options.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MINIMAX_TTS_MODELS.keys()),
        help=f"Model quality — hd (high quality) or turbo (faster). Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (0.5-2.0, default: 1.0)",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Volume level (0.1-10.0, default: 1.0)",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="Pitch shift in semitones (-12 to 12, default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    # Utility
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.json

    # Handle --list-voices
    if args.list_voices:
        print("MiniMax TTS Voices:")
        print()
        print(f"  {'Voice ID':<30} {'Language'}")
        print(f"  {'-' * 30} {'-' * 10}")
        for voice_id, lang in MINIMAX_VOICES.items():
            print(f"  {voice_id:<30} {lang}")
        print()
        print("Models:")
        print(f"  hd     — speech-2.8-hd (high quality, recommended)")
        print(f"  turbo  — speech-2.8-turbo (faster, good for drafts)")
        print()
        print("Usage: --voice English_Persuasive_Man --model hd")
        sys.exit(0)

    # Validate required arguments
    if not args.text:
        print("Error: --text is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    # Validate voice
    if args.voice not in MINIMAX_VOICES:
        print(
            f"Warning: '{args.voice}' is not a verified voice ID. "
            f"Use --list-voices to see options.",
            file=sys.stderr,
        )

    from dotenv import load_dotenv

    load_dotenv()

    result = generate_audio(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        model=args.model,
        speed=args.speed,
        volume=args.volume,
        pitch=args.pitch,
        timeout=args.timeout,
        verbose=verbose,
    )

    if not result.get("success"):
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        duration = result.get("duration_seconds", 0)
        print(f"Generated: {result['output']}")
        if duration:
            print(f"  Duration: {duration:.1f}s ({int(duration * 30)} frames @ 30fps)")


if __name__ == "__main__":
    main()
