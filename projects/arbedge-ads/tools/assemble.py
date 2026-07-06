#!/usr/bin/env python3
"""
Idempotent per-beat renderer + final assembly with background music at 20%.

Each beat is rendered to out/beats/{id}.mp4 only if its config has changed
(tracked by hash in out/render_state.json). The final ad is assembled by
concatenating beat videos, then mixing in background music with ffmpeg.

Usage:
    python tools/assemble.py                      # render missing/changed beats, assemble
    python tools/assemble.py --beat hook-1        # force re-render one beat, then assemble
    python tools/assemble.py --force              # re-render all beats
    python tools/assemble.py --no-assemble        # render beats only, skip final assembly
    python tools/assemble.py --assemble-only      # skip rendering, just concatenate + mix
    python tools/assemble.py --bg-music music.mp3 # custom background track path
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- Paths ------------------------------------------------------------------ #

ROOT = Path(__file__).parent.parent
SCRIPT_PATH = ROOT / "script.json"
REMOTION_DIR = ROOT / "remotion"
OUT_DIR = ROOT / "out"
BEATS_DIR = OUT_DIR / "beats"
STATE_PATH = OUT_DIR / "render_state.json"

# --- State ------------------------------------------------------------------ #

def load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def beat_hash(beat: dict) -> str:
    """Hash a beat's stable fields (type + content, not audio_hash/duration)."""
    stable = {k: v for k, v in beat.items() if k not in ("audio_hash",)}
    return hashlib.md5(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:8]


# --- Rendering -------------------------------------------------------------- #

def render_beat(beat: dict, force: bool = False) -> bool:
    """Render a single beat composition. Returns True if it was re-rendered."""
    beat_id = beat["id"]
    out_path = BEATS_DIR / f"{beat_id}.mp4"
    state = load_state()

    current_hash = beat_hash(beat)
    if not force and out_path.exists() and state.get(beat_id) == current_hash:
        print(f"  [{beat_id}] up to date — skipping")
        return False

    BEATS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [{beat_id}] rendering…")

    result = subprocess.run(
        [
            "npx", "remotion", "render",
            f"Beat-{beat_id}",
            str(out_path),
            "--props", json.dumps({"beatId": beat_id}),
        ],
        cwd=REMOTION_DIR,
    )

    if result.returncode != 0:
        print(f"  [{beat_id}] ✗ render failed")
        return False

    state[beat_id] = current_hash
    save_state(state)
    print(f"  [{beat_id}] ✓ rendered")
    return True


# --- Assembly --------------------------------------------------------------- #

def write_concat_list(beats: list) -> Path:
    concat_path = OUT_DIR / "concat.txt"
    with open(concat_path, "w") as f:
        for beat in beats:
            mp4 = (BEATS_DIR / f"{beat['id']}.mp4").resolve()
            f.write(f"file '{mp4}'\n")
    return concat_path


def assemble(beats: list, bg_music: Path | None, output: Path):
    """Concatenate beat MP4s and optionally mix background music at 20%."""
    missing = [b["id"] for b in beats if not (BEATS_DIR / f"{b['id']}.mp4").exists()]
    if missing:
        print(f"\nERROR: missing beat files: {missing}")
        print("Run without --assemble-only to render them first.")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: concatenate beats (stream copy — fast, no re-encode)
    concat_path = write_concat_list(beats)
    silent_path = OUT_DIR / "ad_concat.mp4"

    print("\nConcatenating beats…")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat_path), "-c", "copy", str(silent_path)],
        check=True,
    )

    # Step 2: mix background music at 20% volume
    if bg_music and bg_music.exists():
        print(f"Mixing background music at 20%: {bg_music.name}")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(silent_path),
                "-stream_loop", "-1", "-i", str(bg_music),
                "-filter_complex",
                "[1:a]volume=0.2[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output),
            ],
            check=True,
        )
    else:
        if bg_music:
            print(f"WARNING: background music not found at {bg_music} — skipping mix")
        print("Finalising (no background music)…")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(silent_path), "-c", "copy", str(output)],
            check=True,
        )

    # Cleanup intermediate
    silent_path.unlink(missing_ok=True)
    concat_path.unlink(missing_ok=True)
    print(f"\n✓ Final video: {output}")


# --- Main ------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--beat", help="Re-render only this beat ID (then assemble)")
    parser.add_argument("--force", action="store_true", help="Force re-render all beats")
    parser.add_argument("--no-assemble", action="store_true", help="Render beats only, skip final assembly")
    parser.add_argument("--assemble-only", action="store_true", help="Skip rendering, just assemble")
    parser.add_argument("--bg-music", default="remotion/public/audio/bg.mp3",
                        help="Background music file (relative to project root, default: remotion/public/audio/bg.mp3)")
    parser.add_argument("--output", default="out/ad.mp4", help="Final output path (default: out/ad.mp4)")
    args = parser.parse_args()

    with open(SCRIPT_PATH) as f:
        script = json.load(f)

    beats = script["beats"]
    bg_music = ROOT / args.bg_music
    output = ROOT / args.output

    # --- Render phase ---
    if not args.assemble_only:
        if args.beat:
            target = next((b for b in beats if b["id"] == args.beat), None)
            if not target:
                print(f"ERROR: beat '{args.beat}' not found in script.json")
                sys.exit(1)
            print(f"Force re-rendering beat: {args.beat}")
            render_beat(target, force=True)
        else:
            print(f"Rendering {len(beats)} beat(s)…")
            for beat in beats:
                render_beat(beat, force=args.force)

    # --- Assembly phase ---
    if not args.no_assemble:
        assemble(beats, bg_music if bg_music.exists() else None, output)


if __name__ == "__main__":
    main()
