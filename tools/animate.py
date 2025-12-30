#!/usr/bin/env python3
"""
Animate static elements in video slides using AI (SVD + SAM2).

Analyzes slide-based videos (NotebookLM, presentations), identifies animation
candidates (characters, environments), and generates subtle motion using
Stable Video Diffusion on RunPod.

Usage:
    # Full pipeline (analyze, segment, animate, composite)
    python tools/animate.py --input video.mp4 --output animated.mp4 --runpod

    # Analyze only (generate manifest for review/editing)
    python tools/animate.py --input video.mp4 --analyze --output manifest.json

    # Use edited manifest (skip auto-analysis)
    python tools/animate.py --input video.mp4 --manifest manifest.json --output animated.mp4

    # Process specific scenes only
    python tools/animate.py --input video.mp4 --scenes 2,3,5 --output animated.mp4 --runpod

    # Preview single scene (no compositing, outputs clips)
    python tools/animate.py --input video.mp4 --scene 3 --preview --output-dir ./preview/

    # Setup RunPod endpoint
    python tools/animate.py --setup

Pipeline:
    1. Scene Detection - FFmpeg scene filter identifies slide transitions
    2. Frame Extraction - Extract representative frame per scene
    3. Analysis - Claude Vision identifies animation candidates (or use manifest)
    4. Segmentation - SAM2 generates masks for each element (RunPod)
    5. Animation - SVD generates subtle motion (RunPod)
    6. Compositing - Layer animated elements over original video

Hardware:
    - RunPod: 16GB+ VRAM recommended (RTX A4000, 3090, 4090)
    - Local: NOT YET SUPPORTED (use --runpod)

Cost (RunPod):
    - ~$0.05-0.30 per video depending on elements and length
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Manifest version for compatibility tracking
MANIFEST_VERSION = "1.0"

# Default scene detection threshold (0-1, higher = fewer scenes detected)
DEFAULT_SCENE_THRESHOLD = 0.3

# Motion prompt templates by element type
MOTION_PROMPTS = {
    "character": "gentle breathing, slight weight shift, subtle eye movement, soft hair movement",
    "environment": "slow atmospheric drift, gentle parallax",
    "water": "soft rippling, organic flow",
    "fire": "gentle flickering, soft glow variation",
    "foliage": "subtle wind movement, gentle sway",
    "clouds": "slow drift, soft shape morphing",
    "default": "subtle natural movement",
}


def log(message: str, verbose: bool = True) -> None:
    """Log message to stderr."""
    if verbose:
        print(message, file=sys.stderr, flush=True)


def get_video_info(video_path: str) -> Optional[dict]:
    """Get video metadata using ffprobe.

    Returns:
        dict with keys: width, height, fps, duration, frame_count
        None if probe fails
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return None

        # Parse frame rate (may be "30/1" or "29.97")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        duration = float(data.get("format", {}).get("duration", 0))
        frame_count = int(video_stream.get("nb_frames", 0))
        if frame_count == 0 and duration > 0:
            frame_count = int(duration * fps)

        return {
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
        }
    except Exception as e:
        log(f"Error probing video: {e}")
        return None


def detect_scenes(
    video_path: str,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_duration: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """Detect scene changes in video using FFmpeg scene filter.

    Args:
        video_path: Path to input video
        threshold: Scene detection threshold (0-1, higher = fewer scenes)
        min_scene_duration: Minimum scene duration in seconds
        verbose: Print progress messages

    Returns:
        List of scene dicts with keys: scene_id, start_time, end_time, duration
    """
    log(f"Detecting scenes (threshold={threshold})...", verbose)

    video_info = get_video_info(video_path)
    if not video_info:
        log("Error: Could not read video info")
        return []

    # Use FFmpeg scene detection filter
    # The select filter outputs frame info when scene change detected
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',showinfo",
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    # Parse scene change timestamps from stderr
    # Format: [Parsed_showinfo_...] n:... pts:... pts_time:12.345 ...
    scene_times = [0.0]  # Always include start
    for line in result.stderr.split("\n"):
        if "pts_time:" in line:
            try:
                # Extract pts_time value
                parts = line.split("pts_time:")
                if len(parts) >= 2:
                    time_str = parts[1].split()[0]
                    time_val = float(time_str)
                    # Only add if significantly different from last scene
                    if time_val - scene_times[-1] >= min_scene_duration:
                        scene_times.append(time_val)
            except (ValueError, IndexError):
                continue

    # Add video end time
    scene_times.append(video_info["duration"])

    # Build scene list
    scenes = []
    for i in range(len(scene_times) - 1):
        start = scene_times[i]
        end = scene_times[i + 1]
        duration = end - start

        # Skip very short scenes
        if duration < min_scene_duration:
            continue

        scenes.append({
            "scene_id": len(scenes) + 1,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(duration, 3),
        })

    log(f"Detected {len(scenes)} scenes", verbose)
    return scenes


def extract_scene_frames(
    video_path: str,
    scenes: list[dict],
    output_dir: str,
    frame_offset: float = 0.5,
    verbose: bool = True,
) -> list[str]:
    """Extract a representative frame from each scene.

    Args:
        video_path: Path to input video
        scenes: List of scene dicts from detect_scenes()
        output_dir: Directory to save frames
        frame_offset: Relative position in scene to extract (0=start, 0.5=middle, 1=end)
        verbose: Print progress messages

    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = []

    log(f"Extracting frames from {len(scenes)} scenes...", verbose)

    for scene in scenes:
        scene_id = scene["scene_id"]
        # Calculate timestamp for frame extraction
        timestamp = scene["start_time"] + (scene["duration"] * frame_offset)

        output_path = os.path.join(output_dir, f"scene_{scene_id:03d}.png")

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite
                "-ss", str(timestamp),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",  # High quality
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and os.path.exists(output_path):
            frame_paths.append(output_path)
            log(f"  Scene {scene_id}: extracted at {timestamp:.2f}s", verbose)
        else:
            log(f"  Scene {scene_id}: FAILED to extract frame", verbose)

    return frame_paths


def create_empty_manifest(
    video_path: str,
    scenes: list[dict],
    frame_paths: list[str],
) -> dict:
    """Create a manifest structure with detected scenes but no elements.

    The manifest can be populated by Claude Vision analysis or manual editing.

    Args:
        video_path: Path to input video
        scenes: List of scene dicts from detect_scenes()
        frame_paths: List of paths to extracted frames

    Returns:
        Manifest dict ready for element analysis
    """
    manifest = {
        "version": MANIFEST_VERSION,
        "input_video": os.path.abspath(video_path),
        "analysis_date": datetime.now().isoformat(),
        "scenes": [],
    }

    for scene, frame_path in zip(scenes, frame_paths):
        manifest["scenes"].append({
            "scene_id": scene["scene_id"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
            "frame_path": os.path.abspath(frame_path) if frame_path else None,
            "elements": [],  # To be populated by analysis
        })

    return manifest


def save_manifest(manifest: dict, output_path: str) -> bool:
    """Save manifest to JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return True
    except Exception as e:
        log(f"Error saving manifest: {e}")
        return False


def load_manifest(manifest_path: str) -> Optional[dict]:
    """Load manifest from JSON file."""
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Validate version
        version = manifest.get("version", "0.0")
        if version != MANIFEST_VERSION:
            log(f"Warning: Manifest version {version} may not be compatible with {MANIFEST_VERSION}")

        return manifest
    except Exception as e:
        log(f"Error loading manifest: {e}")
        return None


def get_motion_prompt(element_type: str) -> str:
    """Get default motion prompt for element type."""
    return MOTION_PROMPTS.get(element_type, MOTION_PROMPTS["default"])


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Animate static elements in video slides using AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze video and generate manifest for review
    python tools/animate.py --input video.mp4 --analyze --output manifest.json

    # Full pipeline with RunPod
    python tools/animate.py --input video.mp4 --output animated.mp4 --runpod

    # Use edited manifest
    python tools/animate.py --input video.mp4 --manifest manifest.json --output animated.mp4

    # Setup RunPod endpoint
    python tools/animate.py --setup
        """,
    )

    # Input/output
    parser.add_argument(
        "--input", "-i",
        help="Input video file",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (video or manifest depending on mode)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for preview/intermediate files",
    )

    # Modes
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze mode: detect scenes and output manifest for review",
    )
    parser.add_argument(
        "--manifest", "-m",
        help="Use existing manifest file (skip analysis)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview mode: output animated clips without compositing",
    )

    # Scene selection
    parser.add_argument(
        "--scenes",
        help="Process specific scenes only (comma-separated, e.g., '2,3,5')",
    )
    parser.add_argument(
        "--scene",
        type=int,
        help="Process a single scene (for preview mode)",
    )

    # Scene detection settings
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        help=f"Scene detection threshold 0-1 (default: {DEFAULT_SCENE_THRESHOLD})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum scene duration in seconds (default: 1.0)",
    )

    # Processing options
    parser.add_argument(
        "--runpod",
        action="store_true",
        help="Use RunPod for GPU processing (required for animation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )

    # Setup and status
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Setup RunPod endpoint for animation",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check RunPod endpoint status",
    )

    # Output control
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = args.verbose and not args.quiet

    # Handle setup/status commands
    if args.setup:
        log("RunPod setup not yet implemented", verbose)
        log("See .ai_dev/character-animation.md for implementation plan", verbose)
        return 1

    if args.status:
        log("RunPod status check not yet implemented", verbose)
        return 1

    # Validate input
    if not args.input:
        log("Error: --input is required")
        return 1

    if not os.path.exists(args.input):
        log(f"Error: Input file not found: {args.input}")
        return 1

    # Get video info
    video_info = get_video_info(args.input)
    if not video_info:
        log("Error: Could not read video info")
        return 1

    log(f"Input: {args.input}", verbose)
    log(f"  Resolution: {video_info['width']}x{video_info['height']}", verbose)
    log(f"  Duration: {video_info['duration']:.2f}s ({video_info['frame_count']} frames)", verbose)
    log(f"  FPS: {video_info['fps']:.2f}", verbose)

    # Analyze mode: detect scenes and create manifest
    if args.analyze:
        if not args.output:
            log("Error: --output is required for analyze mode")
            return 1

        # Detect scenes
        scenes = detect_scenes(
            args.input,
            threshold=args.threshold,
            min_scene_duration=args.min_duration,
            verbose=verbose,
        )

        if not scenes:
            log("Error: No scenes detected")
            return 1

        # Create temp directory for frames
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="animate_")
        frames_dir = os.path.join(output_dir, "frames")

        # Extract frames
        frame_paths = extract_scene_frames(
            args.input,
            scenes,
            frames_dir,
            verbose=verbose,
        )

        # Create manifest
        manifest = create_empty_manifest(args.input, scenes, frame_paths)

        # Add placeholder guidance for manual editing
        manifest["_editing_guide"] = {
            "instructions": "Edit 'elements' array for each scene to define animation targets",
            "element_template": {
                "element_id": "unique_identifier",
                "type": "character|environment|water|fire|foliage|clouds",
                "description": "What this element is",
                "bbox": "[x, y, width, height] or null for auto",
                "animate": True,
                "motion_prompt": "Describe subtle motion (see examples)",
            },
            "motion_examples": MOTION_PROMPTS,
        }

        # Save manifest
        if save_manifest(manifest, args.output):
            log(f"\nManifest saved to: {args.output}", verbose)
            log(f"Frames saved to: {frames_dir}", verbose)
            log("\nNext steps:", verbose)
            log("1. Review the extracted frames in the frames directory", verbose)
            log("2. Edit the manifest to add elements you want to animate", verbose)
            log("3. Run with --manifest to process the animation", verbose)

            if args.json:
                print(json.dumps({"success": True, "manifest": args.output, "frames_dir": frames_dir}))
            return 0
        else:
            return 1

    # Full pipeline or manifest-based processing
    if args.manifest:
        manifest = load_manifest(args.manifest)
        if not manifest:
            return 1
        log(f"Loaded manifest with {len(manifest.get('scenes', []))} scenes", verbose)
    else:
        # Auto-analysis mode (requires Claude Vision - not yet implemented)
        log("Error: Auto-analysis not yet implemented", verbose)
        log("Use --analyze to create a manifest, then edit and re-run with --manifest", verbose)
        return 1

    # Check if we have elements to animate
    total_elements = sum(
        len(scene.get("elements", []))
        for scene in manifest.get("scenes", [])
    )
    if total_elements == 0:
        log("Warning: No elements defined in manifest. Nothing to animate.", verbose)
        log("Edit the manifest to add elements, or run Claude Vision analysis.", verbose)
        return 0

    log(f"Found {total_elements} elements to animate across {len(manifest['scenes'])} scenes", verbose)

    if args.dry_run:
        log("\n[DRY RUN] Would process:", verbose)
        for scene in manifest["scenes"]:
            elements = scene.get("elements", [])
            if elements:
                log(f"  Scene {scene['scene_id']}: {len(elements)} elements", verbose)
                for elem in elements:
                    log(f"    - {elem.get('type', 'unknown')}: {elem.get('description', 'no description')}", verbose)
        return 0

    # Full animation pipeline (not yet implemented)
    if not args.runpod:
        log("Error: --runpod is required for animation (local processing not yet supported)", verbose)
        return 1

    log("Animation pipeline not yet implemented", verbose)
    log("See .ai_dev/character-animation.md for implementation plan", verbose)
    return 1


if __name__ == "__main__":
    sys.exit(main())
