#!/usr/bin/env python3
"""
Locate and verify watermark positions in video files.

This tool helps identify watermark coordinates for use with dewatermark.py.
It extracts frames, overlays grids, and marks regions for visual verification.

Usage:
    # Interactive exploration - extract frames with grid overlay
    python tools/locate_watermark.py --input video.mp4 --grid --output-dir /tmp/review/

    # Verify a specific region across multiple frames
    python tools/locate_watermark.py --input video.mp4 --region 1100,650,150,50 --verify

    # Use a preset for common watermarks
    python tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

    # Quick check - mark single frame
    python tools/locate_watermark.py --input video.mp4 --region 1100,650,150,50 --mark

Presets:
    notebooklm  - Bottom-right corner (Google NotebookLM videos)
    tiktok      - Bottom-center username area
    stock-br    - Bottom-right stock footage watermark
    stock-bl    - Bottom-left stock footage watermark
    stock-center - Center watermark (common in stock footage)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Watermark presets (x, y, width, height) - will be scaled to video dimensions
PRESETS = {
    "notebooklm": {
        "description": "Google NotebookLM - bottom-right corner",
        "region_1280x720": (1100, 650, 150, 50),
        "region_1920x1080": (1650, 975, 225, 75),
    },
    "tiktok": {
        "description": "TikTok username - bottom-center",
        "region_1080x1920": (340, 1750, 400, 80),  # Portrait
        "region_1280x720": (440, 650, 400, 50),    # Landscape
    },
    "stock-br": {
        "description": "Stock footage - bottom-right",
        "region_1280x720": (1000, 620, 260, 80),
        "region_1920x1080": (1500, 930, 390, 120),
    },
    "stock-bl": {
        "description": "Stock footage - bottom-left",
        "region_1280x720": (20, 620, 260, 80),
        "region_1920x1080": (30, 930, 390, 120),
    },
    "stock-center": {
        "description": "Stock footage - center watermark",
        "region_1280x720": (440, 260, 400, 200),
        "region_1920x1080": (660, 390, 600, 300),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locate and verify watermark positions in video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames with coordinate grid for exploration
  python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

  # Verify NotebookLM watermark position
  python tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

  # Mark custom region on multiple frames
  python tools/locate_watermark.py --input video.mp4 --region 1100,650,150,50 --verify

  # Output coordinates as JSON (for scripting)
  python tools/locate_watermark.py --input video.mp4 --preset notebooklm --json
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file path",
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        help="Watermark region as x,y,width,height (e.g., 1100,650,150,50)",
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(PRESETS.keys()),
        help="Use a preset watermark position",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Directory to save marked frames (default: temp directory)",
    )

    # Actions
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Overlay coordinate grid on frames",
    )
    parser.add_argument(
        "--mark",
        action="store_true",
        help="Mark region with rectangle on frames",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Extract multiple frames and mark region for verification",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Also output cropped watermark regions",
    )

    # Sampling options
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of frames to extract (default: 5)",
    )
    parser.add_argument(
        "--timestamps",
        type=str,
        help="Specific timestamps to extract (comma-separated, e.g., '10,30,60,90')",
    )

    # Grid options
    parser.add_argument(
        "--grid-spacing",
        type=int,
        default=50,
        help="Grid line spacing in pixels (default: 50)",
    )
    parser.add_argument(
        "--grid-region",
        type=str,
        help="Only show grid in region x,y,width,height (default: bottom-right quadrant)",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available watermark presets",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open output directory in Finder after processing (macOS)",
    )

    return parser.parse_args()


def get_video_info(video_path: str) -> dict | None:
    """Get video dimensions and duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})
            return {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "duration": float(fmt.get("duration", 0)),
            }
    except Exception:
        pass
    return None


def parse_region(region_str: str) -> tuple[int, int, int, int] | None:
    """Parse region string 'x,y,width,height' into tuple."""
    try:
        parts = [int(x.strip()) for x in region_str.split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except ValueError:
        pass
    return None


def get_preset_region(preset_name: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Get region for a preset, scaled to video dimensions."""
    if preset_name not in PRESETS:
        return None

    preset = PRESETS[preset_name]
    key = f"region_{width}x{height}"

    # Try exact match first
    if key in preset:
        return preset[key]

    # Find closest match and scale
    for preset_key, region in preset.items():
        if preset_key.startswith("region_"):
            dims = preset_key.replace("region_", "").split("x")
            preset_w, preset_h = int(dims[0]), int(dims[1])

            # Scale proportionally
            scale_x = width / preset_w
            scale_y = height / preset_h

            x, y, w, h = region
            return (
                int(x * scale_x),
                int(y * scale_y),
                int(w * scale_x),
                int(h * scale_y),
            )

    return None


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame from video."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def add_grid_overlay(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    spacing: int = 50,
    region: tuple[int, int, int, int] | None = None,
) -> bool:
    """Add coordinate grid overlay to image using ImageMagick."""

    # Determine grid region (default to bottom-right quadrant)
    if region:
        grid_x, grid_y, grid_w, grid_h = region
    else:
        # Bottom-right quadrant
        grid_x = width // 2
        grid_y = height // 2
        grid_w = width // 2
        grid_h = height // 2

    draw_commands = []

    # Vertical lines
    for x in range(grid_x, grid_x + grid_w + 1, spacing):
        if x <= width:
            draw_commands.append(f"line {x},{grid_y} {x},{min(grid_y + grid_h, height)}")
            # Label at bottom
            label_y = min(grid_y + grid_h - 5, height - 5)
            draw_commands.append(f"text {x+2},{label_y} '{x}'")

    # Horizontal lines
    for y in range(grid_y, grid_y + grid_h + 1, spacing):
        if y <= height:
            draw_commands.append(f"line {grid_x},{y} {min(grid_x + grid_w, width)},{y}")
            # Label at left
            draw_commands.append(f"text {grid_x+2},{y-2} '{y}'")

    cmd = [
        "magick", input_path,
        "-stroke", "yellow",
        "-strokewidth", "1",
        "-fill", "yellow",
        "-pointsize", "12",
    ]

    for draw_cmd in draw_commands:
        cmd.extend(["-draw", draw_cmd])

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def mark_region(
    input_path: str,
    output_path: str,
    region: tuple[int, int, int, int],
    color: str = "red",
    stroke_width: int = 3,
) -> bool:
    """Mark a region with a rectangle using ImageMagick."""
    x, y, w, h = region
    x2, y2 = x + w, y + h

    result = subprocess.run(
        [
            "magick", input_path,
            "-stroke", color,
            "-strokewidth", str(stroke_width),
            "-fill", "none",
            "-draw", f"rectangle {x},{y} {x2},{y2}",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def crop_region(
    input_path: str,
    output_path: str,
    region: tuple[int, int, int, int],
) -> bool:
    """Crop image to specified region."""
    x, y, w, h = region

    result = subprocess.run(
        [
            "magick", input_path,
            "-crop", f"{w}x{h}+{x}+{y}",
            "+repage",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def calculate_timestamps(duration: float, num_samples: int, margin: float = 5.0) -> list[float]:
    """Calculate evenly spaced timestamps across video duration."""
    # Avoid very start and end of video
    start = min(margin, duration * 0.05)
    end = max(duration - margin, duration * 0.95)

    if num_samples == 1:
        return [duration / 2]

    step = (end - start) / (num_samples - 1)
    return [start + i * step for i in range(num_samples)]


def list_presets():
    """Print available presets."""
    print("Available watermark presets:")
    print("-" * 50)
    for name, preset in PRESETS.items():
        print(f"\n  {name}")
        print(f"    {preset['description']}")
        for key, value in preset.items():
            if key.startswith("region_"):
                dims = key.replace("region_", "")
                print(f"    {dims}: x={value[0]}, y={value[1]}, w={value[2]}, h={value[3]}")


def main():
    args = parse_args()

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        return

    # Check input is provided for other operations
    if not args.input:
        print("Error: --input is required", file=sys.stderr)
        sys.exit(1)

    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check for ImageMagick
    if shutil.which("magick") is None:
        print("Error: ImageMagick not found. Install with: brew install imagemagick", file=sys.stderr)
        sys.exit(1)

    # Get video info
    video_info = get_video_info(args.input)
    if not video_info:
        print("Error: Could not read video info", file=sys.stderr)
        sys.exit(1)

    width = video_info["width"]
    height = video_info["height"]
    duration = video_info["duration"]

    verbose = not args.json

    if verbose:
        print(f"Video: {args.input}")
        print(f"Dimensions: {width}x{height}")
        print(f"Duration: {duration:.1f}s")

    # Determine region
    region = None
    if args.region:
        region = parse_region(args.region)
        if not region:
            print(f"Error: Invalid region format: {args.region}", file=sys.stderr)
            print("Expected format: x,y,width,height (e.g., 1100,650,150,50)", file=sys.stderr)
            sys.exit(1)
    elif args.preset:
        region = get_preset_region(args.preset, width, height)
        if verbose:
            print(f"Preset '{args.preset}': {region[0]},{region[1]},{region[2]},{region[3]}")

    # Determine timestamps
    if args.timestamps:
        timestamps = [float(t.strip()) for t in args.timestamps.split(",")]
    else:
        timestamps = calculate_timestamps(duration, args.samples)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="locate_wm_"))
        cleanup_temp = not args.open  # Keep if opening in Finder

    if verbose:
        print(f"Output directory: {output_dir}")
        print()

    # Process frames
    results = []
    for i, ts in enumerate(timestamps):
        if verbose:
            print(f"Processing frame {i+1}/{len(timestamps)} at {ts:.1f}s...")

        # Extract frame
        frame_path = output_dir / f"frame_{ts:.0f}s.png"
        if not extract_frame(args.input, ts, str(frame_path)):
            print(f"  Warning: Failed to extract frame at {ts}s", file=sys.stderr)
            continue

        frame_result = {
            "timestamp": ts,
            "frame": str(frame_path),
        }

        # Add grid overlay
        if args.grid:
            grid_region = None
            if args.grid_region:
                grid_region = parse_region(args.grid_region)

            grid_path = output_dir / f"frame_{ts:.0f}s_grid.png"
            if add_grid_overlay(str(frame_path), str(grid_path), width, height, args.grid_spacing, grid_region):
                frame_result["grid"] = str(grid_path)
                if verbose:
                    print(f"  Created: {grid_path.name}")

        # Mark region
        if (args.mark or args.verify) and region:
            marked_path = output_dir / f"frame_{ts:.0f}s_marked.png"
            source = frame_result.get("grid", str(frame_path))
            if mark_region(source, str(marked_path), region):
                frame_result["marked"] = str(marked_path)
                if verbose:
                    print(f"  Created: {marked_path.name}")

        # Crop region
        if args.crop and region:
            crop_path = output_dir / f"frame_{ts:.0f}s_crop.png"
            if crop_region(str(frame_path), str(crop_path), region):
                frame_result["crop"] = str(crop_path)
                if verbose:
                    print(f"  Created: {crop_path.name}")

        results.append(frame_result)

    # Output
    output = {
        "input": args.input,
        "dimensions": f"{width}x{height}",
        "duration": duration,
        "region": f"{region[0]},{region[1]},{region[2]},{region[3]}" if region else None,
        "preset": args.preset,
        "output_dir": str(output_dir),
        "frames": results,
    }

    if region:
        output["dewatermark_command"] = (
            f"python tools/dewatermark.py --input \"{args.input}\" "
            f"--region {region[0]},{region[1]},{region[2]},{region[3]} "
            f"--output \"output_clean.mp4\""
        )

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print()
        print("=" * 50)
        print(f"Extracted {len(results)} frames to: {output_dir}")
        if region:
            print(f"Region: {region[0]},{region[1]},{region[2]},{region[3]}")
            print()
            print("To remove watermark, run:")
            print(f"  python tools/dewatermark.py \\")
            print(f"    --input \"{args.input}\" \\")
            print(f"    --region {region[0]},{region[1]},{region[2]},{region[3]} \\")
            print(f"    --output \"output_clean.mp4\"")
        print("=" * 50)

    # Open in Finder (macOS)
    if args.open:
        subprocess.run(["open", str(output_dir)])

    # Cleanup temp directory if not needed
    if cleanup_temp and not args.output_dir:
        # Don't cleanup - let user review
        pass


if __name__ == "__main__":
    main()
