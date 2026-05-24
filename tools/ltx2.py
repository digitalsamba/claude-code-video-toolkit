#!/usr/bin/env python3
"""
AI video generation using LTX-2.3 (22B DiT model).

Generates ~5 second video clips from text prompts or images via Modal cloud GPU.
Supports text-to-video and image-to-video with joint audio generation.

Examples:
  # Text-to-video
  python tools/ltx2.py --prompt "A cat playing with yarn in a sunlit room"

  # Higher resolution
  python tools/ltx2.py --prompt "Ocean waves at sunset" --width 1024 --height 576

  # Image-to-video (animate a still image)
  python tools/ltx2.py --prompt "Camera slowly pans right" --input photo.jpg

  # Fast mode (fewer steps, lower quality)
  python tools/ltx2.py --prompt "A rocket launch" --quality fast

  # Custom parameters
  python tools/ltx2.py --prompt "A timelapse of flowers blooming" \\
      --num-frames 161 --fps 24 --steps 40 --seed 42

  # Output to specific file
  python tools/ltx2.py --prompt "A dog running on the beach" --output dog_beach.mp4
"""

import argparse
import base64
import sys
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install python-dotenv")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from file_transfer import download_from_url, get_r2_payload_config

# Style LoRA presets. Keep keys in sync with AVAILABLE_LORAS in
# docker/modal-ltx2/app.py — the server only accepts keys it has baked in.
LORA_PRESETS = {
    "crt-terminal": {
        "trigger": "crtanim",
        "default_width": 1024,
        "default_height": 1024,
        "default_num_frames": 121,
        # Default neg prompt keeps on-screen text in frame; the generic one
        # excludes "text, logo" which neutralises the whole LoRA.
        "default_negative_prompt": (
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, signature, jpeg artifacts, compression artifacts"
        ),
    },
}


def log(msg: str, level: str = "info"):
    """Print formatted log message."""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "error": "\033[91m",
        "warn": "\033[93m",
        "dim": "\033[90m",
    }
    reset = "\033[0m"
    prefix = {"info": "->", "success": "OK", "error": "!!", "warn": "??", "dim": "  "}
    color = colors.get(level, "")
    print(f"{color}{prefix.get(level, '->')} {msg}{reset}")


def encode_image(path: str) -> str:
    """Encode image file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def validate_frames(num_frames: int) -> int:
    """Ensure num_frames satisfies (n-1) % 8 == 0 constraint."""
    if (num_frames - 1) % 8 != 0:
        adjusted = ((num_frames - 1 + 4) // 8) * 8 + 1
        log(f"Adjusted num_frames {num_frames} -> {adjusted} (must satisfy (n-1)%8==0)", "warn")
        return adjusted
    return num_frames


def validate_dimensions(width: int, height: int) -> tuple[int, int]:
    """Ensure dimensions are divisible by 64."""
    w = (width // 64) * 64
    h = (height // 64) * 64
    if w != width or h != height:
        log(f"Adjusted dimensions {width}x{height} -> {w}x{h} (must be divisible by 64)", "warn")
    return w, h


def generate_video(
    prompt: str,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    fps: int = 24,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    quality: str = "standard",
    lora: Optional[str] = None,
    open_result: bool = True,
    cloud: str = "modal",
    progress=None,
) -> Optional[str]:
    """
    Generate video from text prompt (and optional input image).

    Returns output path on success, None on failure.
    """
    # Validate parameters
    width, height = validate_dimensions(width, height)
    num_frames = validate_frames(num_frames)

    if output_path is None:
        slug = prompt[:40].strip().replace(" ", "_").lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        output_path = f"{slug}.mp4"

    duration = num_frames / fps

    log(f"Prompt: {prompt}", "info")
    log(f"Size: {width}x{height}, {num_frames} frames @ {fps}fps ({duration:.1f}s)", "dim")
    if input_path:
        log(f"Input image: {input_path}", "info")

    payload = {
        "input": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "quality": quality,
        }
    }

    if negative_prompt:
        payload["input"]["negative_prompt"] = negative_prompt
    if steps is not None:
        payload["input"]["num_inference_steps"] = steps
    if seed is not None:
        payload["input"]["seed"] = seed
    if lora:
        payload["input"]["lora"] = lora

    # Encode input image for I2V
    if input_path:
        if not Path(input_path).exists():
            log(f"File not found: {input_path}", "error")
            return None
        payload["input"]["image_base64"] = encode_image(input_path)

    # R2 config for large video files
    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload

    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="ltx2",
        timeout=900,
        progress_label="Generating video",
        verbose=True,
        progress=progress,
    )

    if "error" in result:
        log(f"Generation failed: {result['error']}", "error")
        return None

    # Download or decode result
    if "output_url" in result:
        log("Downloading from R2...", "dim")
        download_from_url(result["output_url"], output_path, verbose=False)
    elif "video_base64" in result:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["video_base64"]))
    else:
        log("No video data in response", "error")
        return None

    inference_ms = result.get("inference_time_ms", 0)
    out_duration = result.get("duration", duration)
    out_seed = result.get("seed", "unknown")

    log(f"Saved: {output_path}", "success")
    log(f"Duration: {out_duration}s, {result.get('num_frames', num_frames)} frames @ {result.get('fps', fps)}fps", "dim")
    log(f"Time: {elapsed:.1f}s total, {inference_ms/1000:.1f}s inference", "dim")
    log(f"Seed: {out_seed}", "dim")

    if open_result and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", output_path], check=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="AI video generation using LTX-2.3 (22B DiT model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "A cat playing with yarn"
  %(prog)s --prompt "Camera pans over a city" --width 1024 --height 576
  %(prog)s --prompt "Slow zoom in" --input photo.jpg
  %(prog)s --prompt "A rocket launch" --quality fast
  %(prog)s --prompt "Waves crashing" --num-frames 161 --seed 42
        """
    )

    # Input
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--prompt", "-p", required=True, help="Text description of the video")
    input_group.add_argument("--input", "-i", help="Input image for image-to-video")
    input_group.add_argument("--negative-prompt", help="What to avoid (has sensible default)")

    # Video parameters (defaults deferred — LoRA presets may override)
    video_group = parser.add_argument_group("Video parameters")
    video_group.add_argument("--width", "-W", type=int, default=None,
                             help="Video width, divisible by 64 (default: 768, or 1024 with a LoRA preset)")
    video_group.add_argument("--height", "-H", type=int, default=None,
                             help="Video height, divisible by 64 (default: 512, or 1024 with a LoRA preset)")
    video_group.add_argument("--num-frames", "-n", type=int, default=None,
                             help="Number of frames, (n-1)%%8==0 (default: 121, ~5s)")
    video_group.add_argument("--fps", type=int, default=24,
                             help="Frames per second (default: 24)")

    # Style LoRA
    lora_group = parser.add_argument_group("Style LoRA")
    lora_group.add_argument("--lora", choices=list(LORA_PRESETS), default=None,
                            help="Apply a style LoRA preset. First use after a LoRA change "
                                 "incurs ~60s of pipeline reload on the server.")

    # Quality
    quality_group = parser.add_argument_group("Quality")
    quality_group.add_argument("--quality", "-q", choices=["standard", "fast"], default="standard",
                               help="Quality preset (default: standard)")
    quality_group.add_argument("--steps", type=int,
                               help="Inference steps (default: 30 standard, 15 fast)")
    quality_group.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", help="Output file path (default: auto-named .mp4)")
    output_group.add_argument("--no-open", action="store_true", help="Don't open result automatically")
    output_group.add_argument("--json", action="store_true", help="Output result as JSON")
    output_group.add_argument("--progress", choices=["human", "json"], default="human",
                              help="Progress output mode: human (colored stderr, default) "
                                   "or json (JSON Lines to stderr for bots/agents)")

    # Cloud GPU
    cloud_group = parser.add_argument_group("Cloud GPU")
    cloud_group.add_argument("--cloud", type=str, default="modal", choices=["modal"],
                             help="Cloud GPU provider (default: modal)")

    args = parser.parse_args()

    from cloud_gpu import ProgressReporter
    reporter = ProgressReporter(mode=args.progress)

    # Resolve LoRA preset defaults. A preset only fills in values the user
    # didn't set explicitly, so --lora crt-terminal --width 768 still honours
    # the user's width.
    prompt = args.prompt
    width = args.width
    height = args.height
    num_frames = args.num_frames
    negative_prompt = args.negative_prompt

    if args.lora:
        preset = LORA_PRESETS[args.lora]
        trigger = preset["trigger"]
        if not prompt.lstrip().lower().startswith(trigger.lower()):
            prompt = f"{trigger}, {prompt}"
            log(f"Prepended LoRA trigger: '{trigger},'", "dim")
        if width is None:
            width = preset["default_width"]
        if height is None:
            height = preset["default_height"]
        if num_frames is None:
            num_frames = preset["default_num_frames"]
        if negative_prompt is None:
            negative_prompt = preset["default_negative_prompt"]

    # Fall back to the original non-LoRA defaults for anything still unset.
    if width is None:
        width = 768
    if height is None:
        height = 512
    if num_frames is None:
        num_frames = 121

    print()
    log("LTX-2.3 Video Generation (22B DiT)", "info")
    log("=" * 45, "dim")

    result_path = generate_video(
        prompt=prompt,
        input_path=args.input,
        output_path=args.output,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=args.fps,
        steps=args.steps,
        seed=args.seed,
        quality=args.quality,
        lora=args.lora,
        open_result=not args.no_open,
        cloud=args.cloud,
        progress=reporter,
    )

    if result_path is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
