#!/usr/bin/env python3
"""
Generate voiceover audio using ElevenLabs or Qwen3-TTS.

Usage:
    # From script file (ElevenLabs, default)
    python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3

    # From stdin (for AI piping)
    echo "Hello world" | python tools/voiceover.py --output voiceover.mp3

    # With custom voice
    python tools/voiceover.py --script script.txt --voice-id ABC123 --output out.mp3

    # JSON output for machine parsing
    python tools/voiceover.py --script script.txt --output out.mp3 --json

    # Per-scene generation (recommended)
    python tools/voiceover.py --scene-dir public/audio/scenes --json

    # With concat for SadTalker narrator
    python tools/voiceover.py --scene-dir public/audio/scenes --concat public/audio/voiceover-concat.mp3

    # Using Qwen3-TTS provider
    python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
    python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from config import get_brand_dir, get_elevenlabs_api_key, get_voice_id, load_brand_voice_config


def _get_elevenlabs_imports():
    """Lazy import ElevenLabs SDK (only when provider=elevenlabs)."""
    try:
        from elevenlabs import VoiceSettings, save
        from elevenlabs.client import ElevenLabs
        return ElevenLabs, VoiceSettings, save
    except ImportError:
        print(
            "Error: ElevenLabs Python package not installed.\n"
            "\n"
            "You have 3 options:\n"
            "\n"
            "  1. Install ElevenLabs:\n"
            "     pip install elevenlabs\n"
            "\n"
            "  2. Use Qwen3-TTS instead (free, self-hosted):\n"
            "     python3 tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json\n"
            "     (Requires RunPod account — run: python3 tools/qwen3_tts.py --setup)\n"
            "\n"
            "  3. Skip voiceover entirely:\n"
            "     Videos render fine without audio. Add voiceover later when ready.",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate voiceover using ElevenLabs or Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ElevenLabs (default)
  python tools/voiceover.py --script VOICEOVER-SCRIPT.md --output public/audio/voiceover.mp3
  python tools/voiceover.py --scene-dir public/audio/scenes --json

  # Qwen3-TTS
  python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json
  python tools/voiceover.py --provider qwen3 --instruct "Speak warmly" --script script.txt --output out.mp3
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
        help="Output audio file path (.mp3). Required for single-file mode.",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        help="Directory of .txt scripts to process (per-scene mode). Each .txt generates a .mp3.",
    )
    parser.add_argument(
        "--concat",
        type=str,
        help="Output path for concatenated audio (use with --scene-dir for SadTalker)",
    )

    # Provider selection
    parser.add_argument(
        "--provider",
        type=str,
        default="elevenlabs",
        choices=["elevenlabs", "qwen3"],
        help="TTS provider (default: elevenlabs)",
    )

    # ElevenLabs-specific options
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

    # Qwen3-TTS-specific options
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="Qwen3-TTS speaker name (default: Ryan). Use 'python tools/qwen3_tts.py --list-voices' to see options.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Auto",
        help="Qwen3-TTS language hint (default: Auto)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default="",
        help="Qwen3-TTS emotion/style instruction (e.g., 'Speak warmly'). Overrides --tone.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        help="Qwen3-TTS tone preset (e.g., 'warm', 'professional'). See 'python tools/qwen3_tts.py --list-tones'.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        help="Qwen3-TTS reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        help="Qwen3-TTS transcript of reference audio (required with --ref-audio)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Qwen3-TTS expressiveness (default: model default ~0.7, range: 0.3-1.5)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Qwen3-TTS nucleus sampling (default: model default ~0.8, range: 0.1-1.0)",
    )

    # Cloud GPU provider (for Qwen3-TTS)
    parser.add_argument(
        "--cloud",
        type=str,
        default="modal",
        choices=["runpod", "modal"],
        help="Cloud GPU provider for Qwen3-TTS (default: modal)",
    )

    # Brand integration
    parser.add_argument(
        "--brand",
        type=str,
        help="Brand name to load voice config from (e.g., 'default', 'digital-samba')",
    )

    # Common options
    parser.add_argument(
        "--max-wpm",
        type=float,
        default=None,
        help="Pace clamp (any provider): takes exceeding this words-per-minute "
             "are slowed with pitch-preserving atempo (floor 0.85x). Try 165. "
             "Results always include wpm + a pacing label for QC.",
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


def generate_single_audio(
    client,
    script: str,
    output_path: Path,
    voice_id: str,
    model: str,
    stability: float,
    similarity: float,
    style: float,
    speed: float,
    max_wpm: float | None = None,
) -> dict:
    """Generate a single audio file from script text using ElevenLabs. Returns result dict."""
    _, VoiceSettings, save = _get_elevenlabs_imports()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = client.text_to_speech.convert(
        text=script,
        voice_id=voice_id,
        model_id=model,
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=style,
            speed=speed,
        ),
    )

    save(audio, str(output_path))

    duration = get_audio_duration(str(output_path))

    result = {
        "success": True,
        "output": str(output_path),
        "script_chars": len(script),
    }
    if duration:
        result["duration_seconds"] = round(duration, 2)
        result["duration_frames_30fps"] = int(duration * 30)

    _apply_pacing_qc(result, script, max_wpm)
    return result


def _apply_pacing_qc(result: dict, script: str, max_wpm: float | None) -> None:
    """Add wpm/pacing fields to a result; clamp pace in place if max_wpm set."""
    from pacing import clamp_pace, pace_label

    if not result.get("success") or not result.get("output"):
        return
    if max_wpm:
        clamp = clamp_pace(result["output"], script, max_wpm, verbose=False)
        if clamp.get("applied"):
            new_dur = clamp["duration_seconds"]
            result["duration_seconds"] = new_dur
            result["duration_frames_30fps"] = int(new_dur * 30) if new_dur else None
            result["pace_adjusted"] = {
                "original_wpm": clamp["original_wpm"],
                "atempo": clamp["atempo"],
            }
    wpm, label = pace_label(script, result.get("duration_seconds"))
    result["wpm"] = wpm
    result["pacing"] = label


def generate_single_audio_qwen3(
    script: str,
    output_path: Path,
    speaker: str = "Ryan",
    language: str = "Auto",
    instruct: str = "",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "runpod",
    max_wpm: float | None = None,
) -> dict:
    """Generate a single audio file from script text using Qwen3-TTS. Returns result dict."""
    from qwen3_tts import generate_audio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    return generate_audio(
        text=script,
        output_path=str(output_path),
        speaker=speaker,
        language=language,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        verbose=False,
        temperature=temperature,
        top_p=top_p,
        cloud=cloud,
        max_wpm=max_wpm,
    )


def generate_batch_audio_qwen3(
    scripts: list[str],
    output_paths: list[Path],
    speaker: str = "Ryan",
    language: str = "Auto",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "runpod",
    timeout_per_scene: int = 120,
    max_wpm: float | None = None,
) -> list[dict]:
    """Generate multiple audio files in a single Qwen3-TTS call.

    For clone mode: the voice_clone_prompt is extracted ONCE from the
    reference and reused across all scripts — the Qwen-recommended pattern
    for consistent voice across long-form narration.

    Returns a list of result dicts matching the input order.
    """
    from qwen3_tts import generate_audio

    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    result = generate_audio(
        text=scripts,
        output_path=[str(p) for p in output_paths],
        speaker=speaker,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        verbose=False,
        temperature=temperature,
        top_p=top_p,
        cloud=cloud,
        timeout=timeout_per_scene,
        max_wpm=max_wpm,
    )

    if not result.get("success"):
        # Propagate error as one failure per scene so the caller can report cleanly
        err = result.get("error", "batch generation failed")
        return [{"success": False, "error": err, "output": str(p)} for p in output_paths]

    return [
        {"success": True, **item}
        for item in result.get("outputs", [])
    ]


def process_scene_directory(
    scene_dir: Path,
    dry_run: bool = False,
    json_output: bool = False,
    # Provider
    provider: str = "elevenlabs",
    # ElevenLabs params
    client=None,
    voice_id: str = "",
    model: str = "eleven_multilingual_v2",
    stability: float = 0.85,
    similarity: float = 0.95,
    style: float = 0.0,
    speed: float = 1.0,
    # Qwen3 params
    speaker: str = "Ryan",
    language: str = "Auto",
    instruct: str = "",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "runpod",
    max_wpm: float | None = None,
) -> list[dict]:
    """Process all .txt files in directory, generate .mp3 for each."""
    txt_files = sorted(scene_dir.glob("*.txt"))

    if not txt_files:
        print(f"Error: No .txt files found in {scene_dir}", file=sys.stderr)
        sys.exit(1)

    # First pass: collect scenes + their per-scene instruct overrides
    scenes = []
    for txt_file in txt_files:
        mp3_file = txt_file.with_suffix(".mp3")
        script = txt_file.read_text().strip()

        if not script:
            print(f"Warning: Empty script in {txt_file.name}, skipping", file=sys.stderr)
            continue

        scene_instruct = instruct
        if provider == "qwen3":
            import re
            first_line = script.split("\n", 1)[0].strip()
            m = re.match(r"^\[(tone|instruct):\s*(.+?)\]\s*$", first_line, re.IGNORECASE)
            if m:
                kind, value = m.group(1).lower(), m.group(2).strip()
                if kind == "tone":
                    from qwen3_tts import resolve_tone
                    scene_instruct = resolve_tone(value, "")
                else:
                    scene_instruct = value
                script = script.split("\n", 1)[1].strip() if "\n" in script else ""

        scenes.append({
            "txt_file": txt_file,
            "mp3_file": mp3_file,
            "script": script,
            "instruct": scene_instruct,
        })

    results = []
    total_duration = 0.0
    total_chars = sum(len(s["script"]) for s in scenes)

    # Dry run path — no generation
    if dry_run:
        for s in scenes:
            scene_result = {
                "dry_run": True,
                "script": str(s["txt_file"]),
                "output": str(s["mp3_file"]),
                "script_chars": len(s["script"]),
            }
            if provider == "qwen3" and s["instruct"]:
                scene_result["instruct"] = s["instruct"]
            results.append(scene_result)
            if not json_output:
                tone_note = f" [instruct: {s['instruct']}]" if s["instruct"] != instruct else ""
                print(f"  {s['txt_file'].name} → {s['mp3_file'].name} ({len(s['script'])} chars){tone_note}")
        return results, total_duration, total_chars

    # Decide batch vs serial for qwen3:
    #   - Clone mode: batch-safe (instruct is ignored by handler → all scenes share prompt)
    #   - Custom_voice: only batchable if all scenes share the same instruct
    can_batch_qwen3 = (
        provider == "qwen3"
        and len(scenes) > 1
        and (
            ref_audio is not None  # clone — instruct ignored, always batchable
            or all(s["instruct"] == scenes[0]["instruct"] for s in scenes)
        )
    )

    if can_batch_qwen3:
        if not json_output:
            mode_label = "clone (shared prompt)" if ref_audio else "custom_voice (shared instruct)"
            print(f"Batching {len(scenes)} scenes in one Qwen3 call — mode: {mode_label}", file=sys.stderr)

        batch_results = generate_batch_audio_qwen3(
            scripts=[s["script"] for s in scenes],
            output_paths=[s["mp3_file"] for s in scenes],
            speaker=speaker,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            top_p=top_p,
            cloud=cloud,
            max_wpm=max_wpm,
        )

        for s, r in zip(scenes, batch_results):
            r["script"] = str(s["txt_file"])
            results.append(r)
            if r.get("duration_seconds"):
                total_duration += r["duration_seconds"]
            if not json_output:
                if r.get("success"):
                    dur = f" ({r.get('duration_seconds', '?')}s{_pace_note(r)})"
                    print(f"  {s['mp3_file'].name}{dur}", file=sys.stderr)
                else:
                    print(f"  {s['mp3_file'].name}  [FAILED: {r.get('error')}]", file=sys.stderr)
        return results, total_duration, total_chars

    # Fallback: serial per-scene generation
    for s in scenes:
        if not json_output:
            print(f"Generating {s['mp3_file'].name}...", file=sys.stderr)

        if provider == "qwen3":
            result = generate_single_audio_qwen3(
                script=s["script"],
                output_path=s["mp3_file"],
                speaker=speaker,
                language=language,
                instruct=s["instruct"],
                ref_audio=ref_audio,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                cloud=cloud,
                max_wpm=max_wpm,
            )
        else:
            result = generate_single_audio(
                client=client,
                script=s["script"],
                output_path=s["mp3_file"],
                voice_id=voice_id,
                model=model,
                stability=stability,
                similarity=similarity,
                style=style,
                speed=speed,
                max_wpm=max_wpm,
            )
        result["script"] = str(s["txt_file"])
        results.append(result)

        if result.get("duration_seconds"):
            total_duration += result["duration_seconds"]

        if not json_output:
            duration_str = f" ({result.get('duration_seconds', '?')}s{_pace_note(result)})"
            print(f"  {s['mp3_file'].name}{duration_str}", file=sys.stderr)

    return results, total_duration, total_chars


def _pace_note(result: dict) -> str:
    """Human-mode pacing annotation for a per-scene result line."""
    wpm = result.get("wpm")
    if not wpm:
        return ""
    note = f", {wpm:.0f} wpm"
    if result.get("pace_adjusted"):
        note += f" [clamped from {result['pace_adjusted']['original_wpm']:.0f}]"
    elif result.get("pacing") in ("fast", "slow"):
        note += f" [{result['pacing'].upper()} — try --max-wpm 165]"
    return note


def concat_audio_files(mp3_files: list[Path], output_path: Path) -> dict:
    """Use ffmpeg concat demuxer to join audio files."""
    import subprocess
    import tempfile

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for mp3 in mp3_files:
            # FFmpeg concat requires specific format with escaped paths
            escaped_path = str(mp3).replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")
        concat_list = f.name

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                str(output_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error concatenating audio: {result.stderr}", file=sys.stderr)
            return {"success": False, "error": result.stderr}

        duration = get_audio_duration(str(output_path))
        return {
            "success": True,
            "output": str(output_path),
            "duration_seconds": round(duration, 2) if duration else None,
            "source_files": len(mp3_files),
        }
    finally:
        Path(concat_list).unlink(missing_ok=True)


def main():
    load_dotenv()
    args = parse_args()

    provider = args.provider

    # Validate argument combinations
    if args.scene_dir and args.script:
        print("Error: Cannot use both --scene-dir and --script", file=sys.stderr)
        sys.exit(1)

    if args.concat and not args.scene_dir:
        print("Error: --concat requires --scene-dir", file=sys.stderr)
        sys.exit(1)

    if not args.scene_dir and not args.output:
        print("Error: --output is required for single-file mode", file=sys.stderr)
        sys.exit(1)

    if args.ref_audio and not args.ref_text:
        print("Error: --ref-text is required with --ref-audio", file=sys.stderr)
        sys.exit(1)

    # Brand voice config resolution
    if args.brand:
        voice_config = load_brand_voice_config(args.brand)
        if not voice_config:
            print(f"Error: Brand '{args.brand}' not found or has no voice.json", file=sys.stderr)
            sys.exit(1)

        if provider == "qwen3":
            qwen3_cfg = voice_config.get("qwen3", {})
            clone_cfg = qwen3_cfg.get("clone", {})
            if clone_cfg and not args.ref_audio:
                brand_dir = get_brand_dir(args.brand)

                if "design" in clone_cfg:
                    # VoiceDesign mode — design a character voice from a brief,
                    # cache it, and use that cached wav as the clone reference.
                    design = clone_cfg["design"]
                    if "seedText" not in design or "instruct" not in design:
                        print("Error: clone.design requires both 'seedText' and 'instruct'", file=sys.stderr)
                        sys.exit(1)

                    cached_rel = design.get("cachedRef", "assets/voice-design-ref.wav")
                    cached_path = brand_dir / cached_rel

                    if not cached_path.exists():
                        print(
                            f"Designing brand voice for '{args.brand}' (first use)...\n"
                            f"  Instruct: {design['instruct'][:80]}...",
                            file=sys.stderr,
                        )
                        cached_path.parent.mkdir(parents=True, exist_ok=True)

                        from qwen3_tts import generate_audio as _qwen3_generate
                        design_result = _qwen3_generate(
                            text=design["seedText"],
                            output_path=str(cached_path),
                            design_instruct=design["instruct"],
                            language=(design.get("language") or "English"),
                            output_format="wav",
                            verbose=True,
                            cloud=args.cloud,
                        )
                        if not design_result.get("success"):
                            print(
                                f"Error: voice design failed: {design_result.get('error')}",
                                file=sys.stderr,
                            )
                            sys.exit(1)
                        print(f"  Cached designed voice at {cached_path}", file=sys.stderr)

                    args.ref_audio = str(cached_path)
                    args.ref_text = design["seedText"]

                elif "refAudio" in clone_cfg:
                    # Legacy: user-recorded reference audio
                    ref_audio_path = brand_dir / clone_cfg["refAudio"]
                    if ref_audio_path.exists():
                        args.ref_audio = str(ref_audio_path)
                        args.ref_text = clone_cfg.get("refText", "")
                    else:
                        print(f"Warning: Clone ref audio not found: {ref_audio_path}", file=sys.stderr)

            # Apply speaker/language/instruct defaults from brand
            if qwen3_cfg.get("speaker") and args.speaker == "Ryan":
                args.speaker = qwen3_cfg["speaker"]
            if qwen3_cfg.get("language") and args.language == "Auto":
                args.language = qwen3_cfg["language"]
            if qwen3_cfg.get("instruct") and not args.instruct:
                args.instruct = qwen3_cfg["instruct"]
            if qwen3_cfg.get("tone") and not args.tone and not args.instruct:
                args.tone = qwen3_cfg["tone"]
        elif provider == "elevenlabs":
            # Apply voice ID from brand if not explicitly provided
            if not args.voice_id and voice_config.get("voiceId") and voice_config["voiceId"] != "YOUR_VOICE_ID_HERE":
                args.voice_id = voice_config["voiceId"]

    # Resolve tone preset → instruct text for Qwen3
    if provider == "qwen3" and (args.tone or args.instruct):
        from qwen3_tts import resolve_tone
        args.instruct = resolve_tone(args.tone, args.instruct)

    # Warn if tone/instruct used with clone (clone mode ignores instruct)
    if provider == "qwen3" and args.ref_audio and args.instruct:
        print(
            "Note: --tone/--instruct is ignored when using a cloned voice.\n"
            "  The clone's tone comes from your reference recording.\n"
            "  Tip: Run /voice-clone to record a new reference with a different feel.",
            file=sys.stderr,
        )
        args.instruct = ""

    # Provider-specific setup
    client = None
    voice_id = None

    if provider == "elevenlabs":
        api_key = get_elevenlabs_api_key()
        if not api_key:
            print(
                "Error: No ElevenLabs API key found.\n"
                "\n"
                "You have 3 options:\n"
                "\n"
                "  1. Add an ElevenLabs key:\n"
                "     echo \"ELEVENLABS_API_KEY=your_key\" >> .env\n"
                "\n"
                "  2. Use Qwen3-TTS instead (free, self-hosted):\n"
                "     python3 tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json\n"
                "     (Requires RunPod account — run: python3 tools/qwen3_tts.py --setup)\n"
                "\n"
                "  3. Skip voiceover entirely:\n"
                "     Videos render fine without audio. Add voiceover later when ready.",
                file=sys.stderr,
            )
            sys.exit(1)

        voice_id = args.voice_id or get_voice_id()
        if not voice_id:
            print(
                "Error: No voice ID provided and none found in toolkit-registry.json",
                file=sys.stderr,
            )
            sys.exit(1)

        ElevenLabs, _, _ = _get_elevenlabs_imports()
        client = ElevenLabs(api_key=api_key)

    # Per-scene mode
    if args.scene_dir:
        scene_dir = Path(args.scene_dir)
        if not scene_dir.is_dir():
            print(f"Error: Scene directory not found: {scene_dir}", file=sys.stderr)
            sys.exit(1)

        if not args.json:
            txt_count = len(list(scene_dir.glob("*.txt")))
            provider_label = "Qwen3-TTS" if provider == "qwen3" else "ElevenLabs"
            print(f"Processing {txt_count} scene scripts in {scene_dir} ({provider_label})...", file=sys.stderr)

        if args.dry_run:
            if not args.json:
                print("Would generate:")
            results, total_duration, total_chars = process_scene_directory(
                scene_dir=scene_dir,
                dry_run=True,
                json_output=args.json,
                provider=provider,
                client=client,
                voice_id=voice_id or "",
                model=args.model,
                stability=args.stability,
                similarity=args.similarity,
                style=args.style,
                speed=args.speed,
                speaker=args.speaker,
                language=args.language,
                instruct=args.instruct,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                temperature=args.temperature,
                top_p=args.top_p,
                cloud=args.cloud,
            )
            result = {
                "dry_run": True,
                "mode": "per_scene",
                "provider": provider,
                "scene_dir": str(scene_dir),
                "total_chars": total_chars,
                "scenes": results,
            }
            if provider == "elevenlabs":
                result["voice_id"] = voice_id
                result["model"] = args.model
                result["settings"] = {
                    "stability": args.stability,
                    "similarity": args.similarity,
                    "style": args.style,
                    "speed": args.speed,
                }
            else:
                result["speaker"] = args.speaker
                result["language"] = args.language
                if args.instruct:
                    result["instruct"] = args.instruct
                if args.temperature is not None:
                    result["temperature"] = args.temperature
                if args.top_p is not None:
                    result["top_p"] = args.top_p
            if args.concat:
                result["concat_output"] = args.concat
            if args.json:
                print(json.dumps(result, indent=2))
            return

        # Generate per-scene audio
        results, total_duration, total_chars = process_scene_directory(
            scene_dir=scene_dir,
            dry_run=False,
            json_output=args.json,
            provider=provider,
            client=client,
            voice_id=voice_id or "",
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            max_wpm=args.max_wpm,
        )

        # Build final result
        result = {
            "success": True,
            "mode": "per_scene",
            "provider": provider,
            "scene_dir": str(scene_dir),
            "total_chars": total_chars,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_frames_30fps": int(total_duration * 30),
            "scenes": results,
        }
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model

        # Concat if requested
        if args.concat:
            mp3_files = [Path(r["output"]) for r in results if r.get("success")]
            if not args.json:
                print(f"\nConcatenating {len(mp3_files)} files...", file=sys.stderr)
            concat_result = concat_audio_files(mp3_files, Path(args.concat))
            result["concat"] = concat_result
            if not args.json and concat_result.get("success"):
                print(f"  {args.concat} ({concat_result.get('duration_seconds')}s)", file=sys.stderr)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"\nPer-scene audio generated:", file=sys.stderr)
            print(f"  Total: {total_duration:.1f}s ({int(total_duration * 30)} frames @ 30fps)", file=sys.stderr)
            print(f"  Characters: {total_chars}", file=sys.stderr)
        return

    # Single-file mode (original behavior)
    script = read_script(args.script)
    if not script:
        print("Error: Empty script", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    # Dry run mode
    if args.dry_run:
        result = {
            "dry_run": True,
            "mode": "single",
            "provider": provider,
            "script_length": len(script),
            "script_chars": len(script),
            "output": str(output_path),
        }
        if provider == "elevenlabs":
            result["voice_id"] = voice_id
            result["model"] = args.model
            result["settings"] = {
                "stability": args.stability,
                "similarity": args.similarity,
                "style": args.style,
                "speed": args.speed,
            }
        else:
            result["speaker"] = args.speaker
            result["language"] = args.language
            if args.instruct:
                result["instruct"] = args.instruct
            if args.temperature is not None:
                result["temperature"] = args.temperature
            if args.top_p is not None:
                result["top_p"] = args.top_p
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Would generate voiceover:")
            if provider == "elevenlabs":
                print(f"  Voice ID: {voice_id}")
                print(f"  Model: {args.model}")
            else:
                print(f"  Speaker: {args.speaker}")
                print(f"  Language: {args.language}")
            print(f"  Script: {len(script)} characters")
            print(f"  Output: {output_path}")
        return

    # Generate voiceover
    if not args.json:
        provider_label = "Qwen3-TTS" if provider == "qwen3" else "ElevenLabs"
        print(f"Generating voiceover ({len(script)} chars, {provider_label})...", file=sys.stderr)

    if provider == "qwen3":
        result = generate_single_audio_qwen3(
            script=script,
            output_path=output_path,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            temperature=args.temperature,
            top_p=args.top_p,
            cloud=args.cloud,
            max_wpm=args.max_wpm,
        )
    else:
        result = generate_single_audio(
            client=client,
            script=script,
            output_path=output_path,
            voice_id=voice_id,
            model=args.model,
            stability=args.stability,
            similarity=args.similarity,
            style=args.style,
            speed=args.speed,
            max_wpm=args.max_wpm,
        )

    result["mode"] = "single"
    result["provider"] = provider
    if provider == "elevenlabs":
        result["voice_id"] = voice_id
        result["model"] = args.model

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success"):
            print(f"Voiceover saved to: {output_path}", file=sys.stderr)
            duration = result.get("duration_seconds")
            if duration:
                print(
                    f"Duration: {duration:.2f}s ({int(duration * 30)} frames @ 30fps{_pace_note(result)})",
                    file=sys.stderr,
                )
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
