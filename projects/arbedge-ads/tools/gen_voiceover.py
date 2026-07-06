#!/usr/bin/env python3
"""
Generate per-beat voiceover audio using ElevenLabs TTS.

Idempotent: skips beats whose audio file already exists and whose script text
hasn't changed. Updates script.json with duration_frames after generation.

Usage:
    python tools/gen_voiceover.py                # generate all missing/changed
    python tools/gen_voiceover.py --beat hook-1  # regenerate one beat
    python tools/gen_voiceover.py --force        # regenerate everything
    python tools/gen_voiceover.py --list-voices  # show available ElevenLabs voices
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# --- Paths ------------------------------------------------------------------ #

ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "script.json"
AUDIO_DIR = ROOT / "remotion/public/audio/beats"
ENV_PATH = ROOT.parent.parent / ".env"  # toolkit root .env

# --- Helpers ---------------------------------------------------------------- #

def load_env():
    """Load .env from toolkit root (fallback: process env)."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def get_audio_duration(path: Path) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0.0
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    return 0.0


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def frames_from_duration(seconds: float, fps: int = 30, padding: float = 0.4) -> int:
    """Convert audio duration to frame count with a small end-padding."""
    return max(30, int((seconds + padding) * fps))


# --- ElevenLabs ------------------------------------------------------------- #

def generate_audio(beat: dict, api_key: str, voice_id: str, force: bool) -> float:
    """Generate audio for one beat. Returns duration in seconds (0 if skipped)."""
    narration = beat.get("narration", "").strip()
    if not narration:
        print(f"  [{beat['id']}] no narration text — skipping")
        return 0.0

    audio_path = AUDIO_DIR / f"{beat['id']}.mp3"
    new_hash = text_hash(narration)
    cached_hash = beat.get("audio_hash", "")

    if not force and audio_path.exists() and cached_hash == new_hash:
        print(f"  [{beat['id']}] cached ✓")
        return get_audio_duration(audio_path)

    print(f"  [{beat['id']}] generating ({len(narration)} chars)…", end=" ", flush=True)

    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        print("\nERROR: elevenlabs not installed. Run: pip install elevenlabs")
        sys.exit(1)

    client = ElevenLabs(api_key=api_key)
    audio_chunks = client.text_to_speech.convert(
        voice_id=voice_id,
        text=narration,
        model_id="eleven_multilingual_v2",
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.80,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    )

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with open(audio_path, "wb") as f:
        for chunk in audio_chunks:
            if chunk:
                f.write(chunk)

    duration = get_audio_duration(audio_path)
    print(f"done ({duration:.2f}s)")

    beat["audio_hash"] = new_hash
    return duration


def list_voices(api_key: str):
    try:
        from elevenlabs import ElevenLabs
    except ImportError:
        print("elevenlabs not installed. Run: pip install elevenlabs")
        sys.exit(1)
    client = ElevenLabs(api_key=api_key)
    voices = client.voices.get_all()
    print(f"\n{'Name':<30} {'Voice ID':<28} Category")
    print("-" * 70)
    for v in voices.voices:
        print(f"{v.name:<30} {v.voice_id:<28} {v.category}")


# --- Main ------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beat", help="Only process this beat ID")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cached")
    parser.add_argument("--list-voices", action="store_true", help="List available ElevenLabs voices and exit")
    parser.add_argument("--voice-id", help="ElevenLabs voice ID (overrides ELEVENLABS_VOICE_ID env var)")
    parser.add_argument("--padding", type=float, default=0.4, help="End-padding in seconds added to each beat (default: 0.4)")
    args = parser.parse_args()

    load_env()

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY not set. Add it to .env in the toolkit root.")
        sys.exit(1)

    if args.list_voices:
        list_voices(api_key)
        return

    voice_id = args.voice_id or os.getenv("ELEVENLABS_VOICE_ID", "G5FFvULjfTouEEWKNmuH")

    with open(SCRIPT_PATH) as f:
        script = json.load(f)

    beats = script["beats"]
    if args.beat:
        beats = [b for b in beats if b["id"] == args.beat]
        if not beats:
            print(f"ERROR: beat '{args.beat}' not found in script.json")
            sys.exit(1)

    print(f"Processing {len(beats)} beat(s) with voice {voice_id}…\n")

    changed = False
    for beat in beats:
        duration = generate_audio(beat, api_key, voice_id, force=args.force)
        if duration > 0:
            new_frames = frames_from_duration(duration, fps=30, padding=args.padding)
            if beat.get("duration_frames") != new_frames:
                beat["duration_frames"] = new_frames
                changed = True
        elif not beat.get("duration_frames"):
            beat["duration_frames"] = 90  # 3s placeholder
            changed = True

    if changed:
        with open(SCRIPT_PATH, "w") as f:
            json.dump(script, f, indent=2)
        print(f"\n✓ Updated {SCRIPT_PATH}")

    print("\nDone! Run `npm run studio` inside remotion/ to preview.")


if __name__ == "__main__":
    main()
