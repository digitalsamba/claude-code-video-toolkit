#!/usr/bin/env python3
"""
Post-process NotebookLM videos with custom branding.

This tool handles the specific challenge of replacing NotebookLM's outro branding
on redubbed/synced videos where:
- The TTS audio may be longer than the safe visual trim point
- Video and audio tracks need separate handling to preserve full narration
- A freeze frame bridges the gap between video trim and audio end

Pipeline:
    1. Analyze video to find NotebookLM outro (or use specified trim point)
    2. Separate video and audio tracks
    3. Trim video to remove NotebookLM visuals
    4. Create freeze frame to cover remaining audio
    5. Generate branded outro card
    6. Reconstruct: trimmed video + freeze + outro with full audio

Usage:
    # Basic usage with logo and URL
    python tools/notebooklm_brand.py \\
        --input video_synced.mp4 \\
        --logo assets/logo.png \\
        --url "mysite.com" \\
        --output video_final.mp4

    # Specify trim point manually (seconds from start where NotebookLM outro begins)
    python tools/notebooklm_brand.py \\
        --input video_synced.mp4 \\
        --logo assets/logo.png \\
        --url "mysite.com" \\
        --trim-at 174.7 \\
        --output video_final.mp4

    # Use existing outro card image
    python tools/notebooklm_brand.py \\
        --input video_synced.mp4 \\
        --outro-card assets/outro_card.png \\
        --output video_final.mp4

    # Custom outro duration and freeze duration
    python tools/notebooklm_brand.py \\
        --input video_synced.mp4 \\
        --logo assets/logo.png \\
        --url "mysite.com" \\
        --outro-duration 5 \\
        --output video_final.mp4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process NotebookLM videos with custom branding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video file (redubbed/synced NotebookLM video)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output video file path",
    )
    parser.add_argument(
        "--logo",
        type=str,
        help="Logo image file for outro card (PNG recommended)",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL to display on outro card",
    )
    parser.add_argument(
        "--outro-card",
        type=str,
        help="Pre-made outro card image (skips logo/url generation)",
    )
    parser.add_argument(
        "--trim-at",
        type=float,
        help="Timestamp (seconds) where NotebookLM outro begins. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Use separate audio file instead of video's audio track (e.g., TTS file)",
    )
    parser.add_argument(
        "--outro-duration",
        type=float,
        default=4.0,
        help="Duration of outro card in seconds (default: 4.0)",
    )
    parser.add_argument(
        "--background-color",
        type=str,
        default="white",
        help="Background color for generated outro card (default: white)",
    )
    parser.add_argument(
        "--text-color",
        type=str,
        default="0x6B8E6B",
        help="Text color for URL (hex without #, default: 6B8E6B sage green)",
    )
    parser.add_argument(
        "--logo-scale",
        type=int,
        default=220,
        help="Logo width in pixels (default: 220)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=36,
        help="URL font size (default: 36)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files for debugging",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    return parser.parse_args()


def get_media_duration(file_path: str) -> float | None:
    """Get media duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
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


def get_audio_duration(file_path: str) -> float | None:
    """Get audio stream duration using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=duration",
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


def get_video_resolution(file_path: str) -> tuple[int, int] | None:
    """Get video resolution using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            w, h = result.stdout.strip().split(",")
            return int(w), int(h)
    except (FileNotFoundError, ValueError):
        pass
    return None


def get_frame_rate(file_path: str) -> int:
    """Get video frame rate using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            rate = result.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                return round(int(num) / int(den))
            return int(float(rate))
    except (FileNotFoundError, ValueError):
        pass
    return 24  # Default


def create_outro_card(
    output_path: str,
    logo_path: str,
    url: str,
    width: int = 1280,
    height: int = 720,
    bg_color: str = "white",
    text_color: str = "0x6B8E6B",
    logo_scale: int = 220,
    font_size: int = 36,
    verbose: bool = True,
) -> bool:
    """Generate outro card image with logo and URL."""
    if verbose:
        print(f"Creating outro card...", file=sys.stderr)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d=1",
        "-i", logo_path,
        "-filter_complex",
        f"[1:v]scale={logo_scale}:-1[logo];"
        f"[0:v][logo]overlay=(W-w)/2:(H-h)/2-40[bg];"
        f"[bg]drawtext=text={url}:fontsize={font_size}:fontcolor={text_color}:"
        f"x=(w-text_w)/2:y=(h/2)+100",
        "-frames:v", "1",
        "-update", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_frame(video_path: str, timestamp: float, output_path: str) -> bool:
    """Extract a single frame from video at timestamp."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def create_freeze_video(
    image_path: str,
    duration: float,
    output_path: str,
    fps: int = 24,
) -> bool:
    """Create a video from a static image (no audio)."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def extract_video_only(
    video_path: str,
    output_path: str,
    trim_at: float | None = None,
) -> bool:
    """Extract video track only (no audio), optionally trimmed."""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if trim_at:
        cmd.extend(["-t", str(trim_at)])
    cmd.extend(["-an", "-c:v", "copy", output_path])
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def concat_videos(input_files: list[str], output_path: str) -> bool:
    """Concatenate video files (no audio)."""
    # Create concat file
    concat_file = output_path + ".txt"
    with open(concat_file, "w") as f:
        for path in input_files:
            f.write(f"file '{path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(concat_file)
    return result.returncode == 0


def create_audio_with_silence(
    audio_path: str,
    silence_duration: float,
    output_path: str,
) -> bool:
    """Concatenate audio file with silence."""
    # Create silence
    silence_file = output_path + ".silence.m4a"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
        "-t", str(silence_duration),
        "-c:a", "aac",
        silence_file,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False

    # Concatenate
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-i", silence_file,
        "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
        "-map", "[out]",
        "-c:a", "aac",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(silence_file)
    return result.returncode == 0


def mux_video_audio(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """Mux video and audio tracks together."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    args = parse_args()
    verbose = not args.json

    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not args.outro_card and not args.logo:
        print("Error: Must provide either --outro-card or --logo", file=sys.stderr)
        sys.exit(1)

    if args.logo and not Path(args.logo).exists():
        print(f"Error: Logo file not found: {args.logo}", file=sys.stderr)
        sys.exit(1)

    if args.outro_card and not Path(args.outro_card).exists():
        print(f"Error: Outro card file not found: {args.outro_card}", file=sys.stderr)
        sys.exit(1)

    if args.audio_file and not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    # Get video properties
    video_duration = get_media_duration(args.input)
    audio_duration = get_audio_duration(args.audio_file or args.input)
    resolution = get_video_resolution(args.input)
    fps = get_frame_rate(args.input)

    if not video_duration or not audio_duration:
        print("Error: Could not determine video/audio duration", file=sys.stderr)
        sys.exit(1)

    width, height = resolution or (1280, 720)

    # Determine trim point
    # Default: assume NotebookLM outro is ~2.5s but audio extends further
    trim_at = args.trim_at
    if trim_at is None:
        # Heuristic: trim 2.5s before video ends (NotebookLM visual outro)
        # But we need to check if audio extends beyond this
        trim_at = video_duration - 2.5
        if verbose:
            print(f"Auto-detected trim point: {trim_at:.1f}s", file=sys.stderr)

    # Calculate freeze duration (audio continues after video trim)
    freeze_duration = max(0, audio_duration - trim_at)
    if freeze_duration < 0.1:
        freeze_duration = 0  # No freeze needed

    total_duration = trim_at + freeze_duration + args.outro_duration

    if verbose:
        print(f"Input video: {video_duration:.1f}s", file=sys.stderr)
        print(f"Audio duration: {audio_duration:.1f}s", file=sys.stderr)
        print(f"Trim at: {trim_at:.1f}s", file=sys.stderr)
        print(f"Freeze duration: {freeze_duration:.1f}s", file=sys.stderr)
        print(f"Outro duration: {args.outro_duration:.1f}s", file=sys.stderr)
        print(f"Final duration: {total_duration:.1f}s", file=sys.stderr)

    if args.dry_run:
        result = {
            "dry_run": True,
            "input": args.input,
            "output": args.output,
            "video_duration": video_duration,
            "audio_duration": audio_duration,
            "trim_at": trim_at,
            "freeze_duration": freeze_duration,
            "outro_duration": args.outro_duration,
            "final_duration": total_duration,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        return

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="notebooklm_brand_")

    try:
        # Step 1: Create or use outro card
        if args.outro_card:
            outro_card_path = args.outro_card
        else:
            outro_card_path = os.path.join(temp_dir, "outro_card.png")
            if not create_outro_card(
                outro_card_path,
                args.logo,
                args.url or "",
                width, height,
                args.background_color,
                args.text_color,
                args.logo_scale,
                args.font_size,
                verbose,
            ):
                print("Error: Failed to create outro card", file=sys.stderr)
                sys.exit(1)

        # Step 2: Extract video only (trimmed)
        video_only_path = os.path.join(temp_dir, "video_only.mp4")
        if verbose:
            print(f"Extracting video (trimmed at {trim_at:.1f}s)...", file=sys.stderr)
        if not extract_video_only(args.input, video_only_path, trim_at):
            print("Error: Failed to extract video", file=sys.stderr)
            sys.exit(1)

        # Step 3: Create freeze frame video (if needed)
        video_parts = [video_only_path]

        if freeze_duration > 0:
            # Extract last frame
            last_frame_path = os.path.join(temp_dir, "last_frame.png")
            if verbose:
                print(f"Creating freeze frame ({freeze_duration:.1f}s)...", file=sys.stderr)
            if not extract_frame(args.input, trim_at - 0.2, last_frame_path):
                print("Error: Failed to extract last frame", file=sys.stderr)
                sys.exit(1)

            # Create freeze video
            freeze_path = os.path.join(temp_dir, "freeze.mp4")
            if not create_freeze_video(last_frame_path, freeze_duration, freeze_path, fps):
                print("Error: Failed to create freeze frame video", file=sys.stderr)
                sys.exit(1)
            video_parts.append(freeze_path)

        # Step 4: Create outro video
        outro_video_path = os.path.join(temp_dir, "outro.mp4")
        if verbose:
            print(f"Creating outro video ({args.outro_duration}s)...", file=sys.stderr)
        if not create_freeze_video(outro_card_path, args.outro_duration, outro_video_path, fps):
            print("Error: Failed to create outro video", file=sys.stderr)
            sys.exit(1)
        video_parts.append(outro_video_path)

        # Step 5: Concatenate video parts
        combined_video_path = os.path.join(temp_dir, "combined_video.mp4")
        if verbose:
            print("Concatenating video parts...", file=sys.stderr)
        if not concat_videos(video_parts, combined_video_path):
            print("Error: Failed to concatenate videos", file=sys.stderr)
            sys.exit(1)

        # Step 6: Prepare audio (full audio + silence for outro)
        audio_source = args.audio_file or args.input
        combined_audio_path = os.path.join(temp_dir, "combined_audio.m4a")
        if verbose:
            print("Preparing audio track...", file=sys.stderr)

        # Extract audio from source if it's a video
        if args.audio_file:
            audio_for_concat = args.audio_file
        else:
            audio_for_concat = os.path.join(temp_dir, "extracted_audio.mp3")
            cmd = [
                "ffmpeg", "-y",
                "-i", args.input,
                "-vn", "-c:a", "libmp3lame", "-q:a", "2",
                audio_for_concat,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error: Failed to extract audio", file=sys.stderr)
                sys.exit(1)

        if not create_audio_with_silence(audio_for_concat, args.outro_duration, combined_audio_path):
            print("Error: Failed to prepare audio", file=sys.stderr)
            sys.exit(1)

        # Step 7: Mux video and audio
        if verbose:
            print("Muxing final video...", file=sys.stderr)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not mux_video_audio(combined_video_path, combined_audio_path, str(output_path)):
            print("Error: Failed to mux video and audio", file=sys.stderr)
            sys.exit(1)

        # Get final duration
        final_duration = get_media_duration(str(output_path))

        result = {
            "success": True,
            "input": args.input,
            "output": args.output,
            "trim_at": trim_at,
            "freeze_duration": round(freeze_duration, 2),
            "outro_duration": args.outro_duration,
            "input_duration": round(video_duration, 2),
            "audio_duration": round(audio_duration, 2),
            "final_duration": round(final_duration, 2) if final_duration else None,
        }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Done: {output_path}", file=sys.stderr)
            print(f"Final duration: {final_duration:.1f}s", file=sys.stderr)

    finally:
        # Cleanup
        if not args.keep_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            print(f"Temp files kept in: {temp_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
