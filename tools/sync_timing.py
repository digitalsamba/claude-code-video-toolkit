#!/usr/bin/env python3
"""
Sync scene timing in Remotion config with actual audio durations.

After generating per-scene voiceover audio, the actual durations often differ
from the estimated durationSeconds in the config (TTS drift). This tool
automates the feedback loop: measure audio → compare → update config.

Usage:
    # Compare only (dry run, default)
    python3 tools/sync_timing.py

    # Apply changes with 1s padding (default)
    python3 tools/sync_timing.py --apply

    # Custom padding
    python3 tools/sync_timing.py --apply --padding 1.5

    # Accept voiceover.py JSON output (skip re-measuring)
    python3 tools/sync_timing.py --voiceover-json /tmp/vo.json --apply

    # Explicit paths
    python3 tools/sync_timing.py --config src/config/sprint-config.ts --audio-dir public/audio/scenes

    # JSON output for scripting
    python3 tools/sync_timing.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ─── Audio Duration ──────────────────────────────────────────

def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration using ffprobe."""
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


def scan_audio_files(audio_dir: Path) -> list[dict]:
    """Scan directory for audio files and measure durations.

    Expected naming: NN-name.mp3 (e.g., 01-title.mp3, 02-context.mp3)
    Returns list of dicts: {filename, index, name, path, duration_seconds}
    """
    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".mp3", ".wav", ".m4a")
    )

    results = []
    for af in audio_files:
        # Parse NN-name pattern
        match = re.match(r"^(\d+)-(.+)\.\w+$", af.name)
        if match:
            index = int(match.group(1)) - 1  # Convert to 0-based
            name = match.group(2)
        else:
            index = None
            name = af.stem

        duration = get_audio_duration(str(af))
        results.append({
            "filename": af.name,
            "index": index,
            "name": name,
            "path": str(af),
            "duration_seconds": round(duration, 2) if duration else None,
        })

    return results


def load_voiceover_json(json_path: str) -> list[dict]:
    """Load voiceover.py --json output and extract per-scene durations."""
    with open(json_path) as f:
        data = json.load(f)

    scenes = data.get("scenes", [])
    results = []
    for i, scene in enumerate(scenes):
        output_path = Path(scene.get("output", ""))
        match = re.match(r"^(\d+)-(.+)\.\w+$", output_path.name)
        if match:
            index = int(match.group(1)) - 1
            name = match.group(2)
        else:
            index = i
            name = output_path.stem

        results.append({
            "filename": output_path.name,
            "index": index,
            "name": name,
            "path": str(output_path),
            "duration_seconds": scene.get("duration_seconds"),
        })

    return results


# ─── Config Detection ────────────────────────────────────────

TEMPLATE_TYPES = {
    "sprint-review-v1": {
        "config_export": "sprintConfig",
        "has_scene_array": False,
        "description": "sprint-review v1 (demos only)",
    },
    "sprint-review-v2": {
        "config_export": "sprintConfig",
        "has_scene_array": True,
        "description": "sprint-review v2 (composable scenes)",
    },
    "product-demo": {
        "config_export": "demoConfig",
        "has_scene_array": True,
        "description": "product-demo",
    },
}


def detect_config_file(project_dir: Path) -> Path | None:
    """Auto-detect the config file in a project directory."""
    candidates = [
        project_dir / "src" / "config" / "sprint-config.ts",
        project_dir / "src" / "config" / "demo-config.ts",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def detect_template_type(config_text: str, config_path: Path) -> str:
    """Classify config as sprint-review-v1, sprint-review-v2, or product-demo."""
    if "demoConfig" in config_text or "demo-config" in config_path.name:
        return "product-demo"

    # v2 has scenes array with type discriminators
    if re.search(r"scenes\s*:\s*\[", config_text) and re.search(r"type\s*:\s*['\"](?:title|context|goal|highlights)", config_text):
        return "sprint-review-v2"

    return "sprint-review-v1"


# ─── Config Parsing ──────────────────────────────────────────

def parse_scenes_from_config(config_text: str, template_type: str) -> list[dict]:
    """Parse scene objects from TypeScript config text.

    Uses brace-counting to isolate each scene object in the scenes array,
    then extracts key fields with regex.

    Returns list of dicts: {type, durationSeconds, durationSeconds_line,
    durationSeconds_col, audioFile, videoFile, playbackRate}
    """
    if template_type == "sprint-review-v1":
        return _parse_v1_scenes(config_text)
    else:
        return _parse_scene_array(config_text)


def _parse_v1_scenes(config_text: str) -> list[dict]:
    """Parse sprint-review v1 which has demos with durationSeconds but no scenes array.

    v1 has: info, overview, demos[], summary, credits sections.
    Only demos have durationSeconds.
    """
    scenes = []

    # Extract fixed sections with rough timing
    # Title (5s), Overview (15s) — these don't have durationSeconds in config
    # We'll still list them but mark as non-updatable

    # Parse demos array
    demos_match = re.search(r"demos\s*:\s*\[", config_text)
    if demos_match:
        demo_blocks = _extract_array_objects(config_text, demos_match.start())
        for i, block in enumerate(demo_blocks):
            scene = _extract_scene_fields(block)
            scene["type"] = scene.get("type", "demo")
            scene["_section"] = "demos"
            scene["_block_index"] = i
            scenes.append(scene)

    return scenes


def _parse_scene_array(config_text: str) -> list[dict]:
    """Parse scenes: [...] array from v2 or product-demo config."""
    scenes_match = re.search(r"scenes\s*:\s*\[", config_text)
    if not scenes_match:
        return []

    blocks = _extract_array_objects(config_text, scenes_match.start())
    scenes = []
    for i, block in enumerate(blocks):
        scene = _extract_scene_fields(block)
        scene["_block_index"] = i
        scenes.append(scene)

    return scenes


def _extract_array_objects(config_text: str, array_start: int) -> list[dict]:
    """Extract individual objects from an array using brace counting.

    Returns list of dicts with 'text', 'start', 'end' (positions in config_text).
    """
    # Find the opening bracket
    bracket_pos = config_text.index("[", array_start)
    objects = []
    depth = 0
    obj_start = None
    in_string = False
    string_char = None
    in_line_comment = False
    in_block_comment = False
    i = bracket_pos + 1

    while i < len(config_text):
        c = config_text[i]
        prev = config_text[i - 1] if i > 0 else ""

        # Handle comments
        if not in_string and not in_block_comment and c == "/" and i + 1 < len(config_text):
            if config_text[i + 1] == "/":
                in_line_comment = True
                i += 2
                continue
            elif config_text[i + 1] == "*":
                in_block_comment = True
                i += 2
                continue

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if c == "/" and prev == "*":
                in_block_comment = False
            i += 1
            continue

        # Handle strings
        if not in_string and c in ("'", '"', '`'):
            in_string = True
            string_char = c
            i += 1
            continue
        if in_string:
            if c == string_char and prev != "\\":
                in_string = False
            i += 1
            continue

        # Track braces
        if c == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                objects.append({
                    "text": config_text[obj_start:i + 1],
                    "start": obj_start,
                    "end": i + 1,
                })
                obj_start = None
        elif c == "]" and depth == 0:
            break

        i += 1

    return objects


def _extract_scene_fields(block: dict) -> dict:
    """Extract relevant fields from a scene object block."""
    text = block["text"]
    start = block["start"]
    result = {
        "_text": text,
        "_start": start,
        "_end": block["end"],
    }

    # type: 'title'
    m = re.search(r"type\s*:\s*['\"](\w+)['\"]", text)
    if m:
        result["type"] = m.group(1)

    # durationSeconds: 15
    m = re.search(r"durationSeconds\s*:\s*(\d+(?:\.\d+)?)", text)
    if m:
        result["durationSeconds"] = float(m.group(1))
        # Calculate absolute position in file
        result["_ds_match_start"] = start + m.start()
        result["_ds_match_end"] = start + m.end()
        result["_ds_value_start"] = start + m.start(1)
        result["_ds_value_end"] = start + m.end(1)

    # audioFile: 'scenes/01-title.mp3'
    m = re.search(r"audioFile\s*:\s*['\"]([^'\"]+)['\"]", text)
    if m:
        result["audioFile"] = m.group(1)

    # videoFile: 'demo-1.mp4' or videoFile: 'demos/demo.mp4'
    m = re.search(r"videoFile\s*:\s*['\"]([^'\"]+)['\"]", text)
    if m:
        result["videoFile"] = m.group(1)

    # playbackRate: 1.5
    m = re.search(r"playbackRate\s*:\s*(\d+(?:\.\d+)?)", text)
    if m:
        result["playbackRate"] = float(m.group(1))

    return result


# ─── Matching ────────────────────────────────────────────────

def match_audio_to_scenes(
    audio_files: list[dict],
    scenes: list[dict],
) -> list[dict]:
    """Match audio files to config scenes using 3-pass matching.

    Pass 1: audioFile field match (exact filename in scene's audioFile)
    Pass 2: Index match (01-title.mp3 → scene index 0)
    Pass 3: Name match (filename contains scene type)

    Returns list of dicts: {scene_index, scene, audio, match_method}
    """
    matches = []
    matched_audio = set()
    matched_scenes = set()

    # Pass 1: audioFile field match
    for si, scene in enumerate(scenes):
        audio_file_ref = scene.get("audioFile", "")
        if not audio_file_ref:
            continue
        # Extract just the filename from the path
        ref_name = Path(audio_file_ref).name
        for ai, audio in enumerate(audio_files):
            if ai in matched_audio:
                continue
            if audio["filename"] == ref_name:
                matches.append({
                    "scene_index": si,
                    "scene": scene,
                    "audio": audio,
                    "match_method": "audioFile",
                })
                matched_audio.add(ai)
                matched_scenes.add(si)
                break

    # Pass 2: Index match
    for ai, audio in enumerate(audio_files):
        if ai in matched_audio:
            continue
        if audio["index"] is not None:
            si = audio["index"]
            if 0 <= si < len(scenes) and si not in matched_scenes:
                matches.append({
                    "scene_index": si,
                    "scene": scenes[si],
                    "audio": audio,
                    "match_method": "index",
                })
                matched_audio.add(ai)
                matched_scenes.add(si)

    # Pass 3: Name match (filename contains scene type)
    for ai, audio in enumerate(audio_files):
        if ai in matched_audio:
            continue
        audio_name = audio["name"].lower().replace("-", "").replace("_", "")
        for si, scene in enumerate(scenes):
            if si in matched_scenes:
                continue
            scene_type = scene.get("type", "").lower()
            if scene_type and scene_type in audio_name:
                matches.append({
                    "scene_index": si,
                    "scene": scene,
                    "audio": audio,
                    "match_method": "name",
                })
                matched_audio.add(ai)
                matched_scenes.add(si)
                break

    # Add unmatched scenes
    for si, scene in enumerate(scenes):
        if si not in matched_scenes:
            matches.append({
                "scene_index": si,
                "scene": scene,
                "audio": None,
                "match_method": None,
            })

    # Sort by scene index
    matches.sort(key=lambda m: m["scene_index"])
    return matches


# ─── Output ──────────────────────────────────────────────────

DELTA_THRESHOLD = 0.3  # Changes smaller than this are marked "OK"


def format_comparison_table(
    matches: list[dict],
    padding: float,
    template_type: str,
) -> str:
    """Format human-readable comparison table."""
    desc = TEMPLATE_TYPES.get(template_type, {}).get("description", template_type)
    lines = []

    # Count audio files
    audio_count = sum(1 for m in matches if m["audio"])
    lines.append("")

    # Header
    col_w = {
        "scene": 30,
        "match": 12,
        "config": 9,
        "audio": 9,
        "new": 9,
        "delta": 10,
    }
    header = (
        f"{'Scene':<{col_w['scene']}} "
        f"{'Match':<{col_w['match']}} "
        f"{'Config':>{col_w['config']}} "
        f"{'Audio':>{col_w['audio']}} "
        f"{'New':>{col_w['new']}} "
        f"{'Delta':>{col_w['delta']}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_config = 0.0
    total_new = 0.0
    update_count = 0

    for m in matches:
        scene = m["scene"]
        audio = m["audio"]
        si = m["scene_index"]
        scene_type = scene.get("type", "?")
        config_dur = scene.get("durationSeconds")

        # Scene label: "01 title"
        label = f"{si + 1:02d} {scene_type}"

        # Match indicator
        match_method = f"[{m['match_method']}]" if m["match_method"] else ""

        if config_dur is not None:
            total_config += config_dur
            config_str = f"{config_dur:.1f}s"
        else:
            config_str = "—"

        if audio and audio["duration_seconds"]:
            audio_dur = audio["duration_seconds"]
            audio_str = f"{audio_dur:.1f}s"
            proposed = math.ceil(audio_dur + padding)
            proposed_str = f"{proposed}s"

            if config_dur is not None:
                delta = proposed - config_dur
                if abs(delta) < DELTA_THRESHOLD:
                    delta_str = "OK"
                else:
                    delta_str = f"{delta:+.1f}s"
                    update_count += 1
                total_new += proposed
            else:
                delta_str = "new"
                total_new += proposed
        else:
            audio_str = "—"
            proposed_str = "—"
            delta_str = "—"
            if config_dur is not None:
                total_new += config_dur

        lines.append(
            f"{label:<{col_w['scene']}} "
            f"{match_method:<{col_w['match']}} "
            f"{config_str:>{col_w['config']}} "
            f"{audio_str:>{col_w['audio']}} "
            f"{proposed_str:>{col_w['new']}} "
            f"{delta_str:>{col_w['delta']}}"
        )

    lines.append("")
    lines.append(
        f"Totals: {total_config:.1f}s → {total_new:.1f}s "
        f"(delta: {total_new - total_config:+.1f}s)"
    )

    if update_count > 0:
        lines.append(f"\n{update_count} scene(s) need timing updates.")
    else:
        lines.append("\nAll scenes within threshold — no updates needed.")

    return "\n".join(lines)


def suggest_playback_rates(matches: list[dict], project_dir: Path) -> list[str]:
    """For demo scenes with videoFile, suggest playbackRate adjustments."""
    suggestions = []
    for m in matches:
        scene = m["scene"]
        audio = m["audio"]
        if scene.get("type") != "demo" or not scene.get("videoFile"):
            continue
        if not audio or not audio["duration_seconds"]:
            continue

        video_path = project_dir / "public" / scene["videoFile"]
        if not video_path.exists():
            video_path = project_dir / "public" / "demos" / Path(scene["videoFile"]).name
        if not video_path.exists():
            continue

        raw_duration = get_audio_duration(str(video_path))
        if not raw_duration:
            continue

        audio_dur = audio["duration_seconds"]
        ideal_rate = raw_duration / audio_dur
        current_rate = scene.get("playbackRate", 1.0)

        if abs(ideal_rate - current_rate) > 0.1:
            suggestions.append(
                f"  Scene {m['scene_index'] + 1:02d} ({scene.get('type')}): "
                f"video={raw_duration:.1f}s, audio={audio_dur:.1f}s → "
                f"playbackRate: {ideal_rate:.2f} (currently {current_rate:.1f})"
            )

    return suggestions


# ─── Apply Updates ───────────────────────────────────────────

def apply_timing_updates(
    config_path: Path,
    config_text: str,
    matches: list[dict],
    padding: float,
) -> tuple[str, int]:
    """Update durationSeconds values in config text.

    Works bottom-to-top to preserve character positions.
    Returns (updated_text, update_count).
    """
    updates = []

    for m in matches:
        scene = m["scene"]
        audio = m["audio"]

        if not audio or not audio["duration_seconds"]:
            continue
        if "_ds_value_start" not in scene:
            continue

        audio_dur = audio["duration_seconds"]
        proposed = math.ceil(audio_dur + padding)
        config_dur = scene.get("durationSeconds", 0)

        # Skip small deltas
        if abs(proposed - config_dur) < DELTA_THRESHOLD:
            continue

        updates.append({
            "start": scene["_ds_value_start"],
            "end": scene["_ds_value_end"],
            "old_value": str(int(config_dur)) if config_dur == int(config_dur) else str(config_dur),
            "new_value": str(proposed),
            "scene_index": m["scene_index"],
            "scene_type": scene.get("type", "?"),
        })

    if not updates:
        return config_text, 0

    # Create backup
    backup_path = config_path.with_suffix(".ts.bak")
    shutil.copy2(config_path, backup_path)

    # Apply bottom-to-top to preserve positions
    result = config_text
    updates.sort(key=lambda u: u["start"], reverse=True)
    for u in updates:
        result = result[:u["start"]] + u["new_value"] + result[u["end"]:]

    return result, len(updates)


# ─── JSON Output ─────────────────────────────────────────────

def build_json_output(
    matches: list[dict],
    padding: float,
    template_type: str,
    config_path: str,
    audio_dir: str | None,
) -> dict:
    """Build machine-readable output."""
    scenes = []
    total_config = 0.0
    total_new = 0.0

    for m in matches:
        scene = m["scene"]
        audio = m["audio"]
        config_dur = scene.get("durationSeconds")

        entry = {
            "index": m["scene_index"],
            "type": scene.get("type", "unknown"),
            "match_method": m["match_method"],
            "config_seconds": config_dur,
        }

        if audio and audio["duration_seconds"]:
            audio_dur = audio["duration_seconds"]
            proposed = round(audio_dur + padding, 2)
            proposed_ceil = math.ceil(audio_dur + padding)
            entry["audio_seconds"] = audio_dur
            entry["audio_file"] = audio["filename"]
            entry["proposed_seconds"] = proposed_ceil
            entry["delta"] = round(proposed_ceil - (config_dur or 0), 2)
            entry["needs_update"] = abs(entry["delta"]) >= DELTA_THRESHOLD
            if config_dur:
                total_config += config_dur
            total_new += proposed_ceil
        else:
            entry["audio_seconds"] = None
            entry["needs_update"] = False
            if config_dur:
                total_config += config_dur
                total_new += config_dur

        if scene.get("videoFile"):
            entry["video_file"] = scene["videoFile"]
        if scene.get("playbackRate"):
            entry["playback_rate"] = scene["playbackRate"]

        scenes.append(entry)

    return {
        "config_file": str(config_path),
        "audio_dir": audio_dir,
        "template_type": template_type,
        "padding_seconds": padding,
        "total_config_seconds": round(total_config, 2),
        "total_proposed_seconds": round(total_new, 2),
        "total_delta_seconds": round(total_new - total_config, 2),
        "updates_needed": sum(1 for s in scenes if s.get("needs_update")),
        "scenes": scenes,
    }


# ─── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sync scene timing with actual audio durations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/sync_timing.py                          # Dry run comparison
  python3 tools/sync_timing.py --apply                  # Update config
  python3 tools/sync_timing.py --apply --padding 1.5    # Custom padding
  python3 tools/sync_timing.py --voiceover-json vo.json # Use voiceover.py output
  python3 tools/sync_timing.py --json                   # Machine-readable output
""",
    )
    parser.add_argument(
        "--config",
        help="Path to config file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--audio-dir",
        help="Path to per-scene audio directory (default: public/audio/scenes)",
    )
    parser.add_argument(
        "--voiceover-json",
        help="Path to voiceover.py --json output (skip re-measuring audio)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Seconds of padding after audio ends (default: 1.0)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply timing updates to config file (creates .bak backup)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON for scripting",
    )

    args = parser.parse_args()

    # Determine project directory
    project_dir = Path.cwd()

    # Detect config file
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = detect_config_file(project_dir)

    if not config_path or not config_path.exists():
        print(
            "Error: Could not find config file. "
            "Run from a project directory or use --config.",
            file=sys.stderr,
        )
        sys.exit(1)

    config_text = config_path.read_text()
    template_type = detect_template_type(config_text, config_path)

    if not args.json:
        desc = TEMPLATE_TYPES.get(template_type, {}).get("description", template_type)
        print(f"Timing Sync — {config_path} ({desc})", file=sys.stderr)

    # Parse config scenes
    scenes = parse_scenes_from_config(config_text, template_type)
    if not scenes:
        print("Error: No scenes found in config.", file=sys.stderr)
        sys.exit(1)

    # Get audio durations
    if args.voiceover_json:
        audio_files = load_voiceover_json(args.voiceover_json)
        audio_dir_str = None
        if not args.json:
            print(f"Audio: from {args.voiceover_json} ({len(audio_files)} scenes)", file=sys.stderr)
    else:
        audio_dir = Path(args.audio_dir) if args.audio_dir else project_dir / "public" / "audio" / "scenes"
        if not audio_dir.exists():
            print(f"Error: Audio directory not found: {audio_dir}", file=sys.stderr)
            sys.exit(1)
        audio_files = scan_audio_files(audio_dir)
        audio_dir_str = str(audio_dir)
        if not audio_files:
            print(f"Error: No audio files found in {audio_dir}", file=sys.stderr)
            sys.exit(1)
        if not args.json:
            print(f"Audio: {audio_dir} ({len(audio_files)} files) | Padding: {args.padding}s", file=sys.stderr)

    # Match audio to scenes
    matches = match_audio_to_scenes(audio_files, scenes)

    # JSON output mode
    if args.json:
        output = build_json_output(
            matches, args.padding, template_type,
            str(config_path), audio_dir_str,
        )
        output["applied"] = False

        if args.apply:
            updated_text, count = apply_timing_updates(
                config_path, config_text, matches, args.padding,
            )
            if count > 0:
                config_path.write_text(updated_text)
                output["applied"] = True
                output["updates_applied"] = count
                output["backup"] = str(config_path.with_suffix(".ts.bak"))

        print(json.dumps(output, indent=2))
        return

    # Human-readable output
    table = format_comparison_table(matches, args.padding, template_type)
    print(table, file=sys.stderr)

    # Playback rate suggestions
    suggestions = suggest_playback_rates(matches, project_dir)
    if suggestions:
        print("\nPlayback rate suggestions:", file=sys.stderr)
        for s in suggestions:
            print(s, file=sys.stderr)

    # Apply if requested
    if args.apply:
        updated_text, count = apply_timing_updates(
            config_path, config_text, matches, args.padding,
        )
        if count > 0:
            config_path.write_text(updated_text)
            backup = config_path.with_suffix(".ts.bak")
            print(f"\nApplied {count} update(s) to {config_path}", file=sys.stderr)
            print(f"Backup saved to {backup}", file=sys.stderr)
        else:
            print("\nNo updates needed — all within threshold.", file=sys.stderr)
    else:
        has_updates = any(
            m["audio"] and m["audio"]["duration_seconds"]
            and m["scene"].get("durationSeconds") is not None
            and abs(math.ceil(m["audio"]["duration_seconds"] + args.padding) - m["scene"]["durationSeconds"]) >= DELTA_THRESHOLD
            for m in matches
        )
        if has_updates:
            print("\nRun with --apply to update config.", file=sys.stderr)


if __name__ == "__main__":
    main()
