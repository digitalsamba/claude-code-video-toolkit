#!/usr/bin/env python3
"""
Generate speech with ElevenLabs voices through fal.ai (https://fal.ai).

fal.ai hosts ElevenLabs TTS and bills it to the fal account, so no ElevenLabs
key is needed. This exposes a `generate_audio()` used by `voiceover.py`
(as `--provider fal`), plus a standalone CLI.

Endpoint: fal-ai/elevenlabs/tts/multilingual-v2 (synchronous, via fal.run).
`voice` accepts an ElevenLabs voice ID or a library voice name ("Rachel").

Voice settings use the toolkit's unified 0-1 scale, which is ElevenLabs'
native scale, so nothing is converted. `speed` is clamped to fal's 0.7-1.2.

Usage:
    uv run tools/fal_tts.py --text "Hello world" --voice-id dSByRdUbTGloB7TFA1qD --output hello.mp3
    uv run tools/fal_tts.py --text "Hello" --voice-id Rachel --speed 1.1 --output hello.mp3

Setup:
    echo "FAL_KEY=your_key" >> .env
    # Optional default voice:
    echo "FAL_VOICE_ID=<voice id or name>" >> .env
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from win_encoding import ensure_utf8_output

FAL_RUN_URL = "https://fal.run"
# Toolkit model id -> fal endpoint. Only endpoints verified against fal's API.
FAL_ENDPOINTS = {
    "eleven_multilingual_v2": "fal-ai/elevenlabs/tts/multilingual-v2",
}
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE = "Rachel"
SPEED_MIN, SPEED_MAX = 0.7, 1.2


def get_fal_api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("FAL_KEY")


def get_fal_voice_id() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("FAL_VOICE_ID")


def _duration(path: str) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def generate_audio(
    text: str,
    output_path: str,
    voice_id: str | None = None,
    stability: float | None = 0.5,
    similarity: float | None = 0.75,
    style: float | None = None,
    speed: float = 1.0,
    model: str = DEFAULT_MODEL,
    language_code: str | None = None,
    api_key: str | None = None,
    timeout: int = 180,
    verbose: bool = True,
) -> dict:
    """Generate one audio file via fal.ai's ElevenLabs endpoint.

    Returns {success, output, duration_seconds, duration_frames_30fps, script_chars}
    or {success: False, error: ...}, the shape the other providers use.
    """
    if not text:
        return {"success": False, "error": "text must be a non-empty string"}

    api_key = api_key or get_fal_api_key()
    if not api_key:
        return {"success": False, "error": "No fal.ai key (set FAL_KEY in .env)"}

    endpoint = FAL_ENDPOINTS.get(model)
    if not endpoint:
        return {
            "success": False,
            "error": f"Model '{model}' has no fal.ai endpoint here (supported: {', '.join(FAL_ENDPOINTS)})",
        }

    voice_id = voice_id or get_fal_voice_id() or DEFAULT_VOICE
    clamped = min(max(speed, SPEED_MIN), SPEED_MAX)
    if clamped != speed and verbose:
        print(f"fal: speed {speed} clamped to {clamped} (fal accepts {SPEED_MIN}-{SPEED_MAX})", file=sys.stderr)

    payload = {"text": text, "voice": voice_id, "speed": clamped}
    if stability is not None:
        payload["stability"] = stability
    if similarity is not None:
        payload["similarity_boost"] = similarity
    if style:
        payload["style"] = style
    if language_code:
        payload["language_code"] = language_code

    if verbose:
        print(f"fal: {endpoint} voice={voice_id} speed={clamped}", file=sys.stderr)

    try:
        resp = requests.post(
            f"{FAL_RUN_URL}/{endpoint}",
            headers={"Authorization": f"Key {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"success": False, "error": f"fal request failed: {e}"}
    if resp.status_code != 200:
        return {"success": False, "error": f"fal HTTP {resp.status_code}: {resp.text[:300]}"}

    audio_url = (resp.json().get("audio") or {}).get("url")
    if not audio_url:
        return {"success": False, "error": f"fal response has no audio url: {resp.text[:300]}"}

    try:
        audio = requests.get(audio_url, timeout=timeout)
        audio.raise_for_status()
    except requests.RequestException as e:
        return {"success": False, "error": f"downloading fal audio failed: {e}"}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio.content)

    result = {"success": True, "output": str(out), "script_chars": len(text)}
    duration = _duration(str(out))
    if duration:
        result["duration_seconds"] = round(duration, 2)
        result["duration_frames_30fps"] = int(duration * 30)
    return result


def main():
    ensure_utf8_output()
    parser = argparse.ArgumentParser(description="ElevenLabs TTS via fal.ai")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--output", "-o", required=True, help="Output .mp3 path")
    parser.add_argument("--voice-id", "-v", help="ElevenLabs voice ID or name (default: FAL_VOICE_ID or Rachel)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(FAL_ENDPOINTS))
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0, help=f"{SPEED_MIN}-{SPEED_MAX}")
    parser.add_argument("--language-code", help="ISO 639-1 code, e.g. en")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = generate_audio(
        text=args.text, output_path=args.output, voice_id=args.voice_id,
        stability=args.stability, similarity=args.similarity, style=args.style,
        speed=args.speed, model=args.model, language_code=args.language_code,
        verbose=not args.json,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("success"):
        print(f"Saved {result['output']} ({result.get('duration_seconds', '?')}s)", file=sys.stderr)
    if not result.get("success"):
        if not args.json:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
