#!/usr/bin/env python3
"""
AI-powered image editing using Qwen-Image-Edit-2511.

Capabilities:
- Background replacement (--background)
- Style transfer (--style)
- Custom edits (--prompt)
- Multi-image merge (multiple --input files)
- Viewpoint changes (--viewpoint)
- Batch processing (--input-dir)

Examples:
  # Background replacement
  python tools/image_edit.py --input photo.jpg --background "pyramids of Egypt"

  # Style transfer
  python tools/image_edit.py --input photo.jpg --style "cyberpunk neon city"

  # Custom prompt (full control)
  python tools/image_edit.py --input photo.jpg --prompt "Add warm sunset lighting"

  # Multi-image merge (group photo)
  python tools/image_edit.py --input person1.jpg person2.jpg --prompt "Both standing together in a park"

  # Viewpoint change
  python tools/image_edit.py --input photo.jpg --viewpoint "front facing"

  # Batch processing
  python tools/image_edit.py --input-dir ./photos --background "studio backdrop" --output-dir ./edited

  # With seed for reproducibility
  python tools/image_edit.py --input photo.jpg --background "office" --seed 42
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
    from PIL import Image
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install requests Pillow python-dotenv")
    sys.exit(1)

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
QWEN_EDIT_ENDPOINT = os.getenv("RUNPOD_QWEN_EDIT_ENDPOINT_ID")

# Background presets
BACKGROUND_PRESETS = {
    "office": "modern professional office with glass windows and city view",
    "studio": "clean white photography studio backdrop with soft lighting",
    "outdoors": "beautiful park with trees and natural lighting",
    "pyramids": "ancient pyramids of Giza in Egypt with desert sand",
    "beach": "tropical beach with palm trees and turquoise ocean",
    "city": "urban cityscape with skyscrapers at golden hour",
    "mountains": "majestic mountain landscape with snow peaks",
    "space": "cosmic space background with stars and nebulae",
    "forest": "lush green forest with sunlight filtering through trees",
    "cafe": "cozy cafe interior with warm ambient lighting",
}

# Style presets
STYLE_PRESETS = {
    "cyberpunk": "cyberpunk neon aesthetic with glowing lights and rain",
    "anime": "anime art style with vibrant colors",
    "oil-painting": "classic oil painting style with brush strokes",
    "watercolor": "soft watercolor painting style",
    "pixel-art": "retro pixel art style",
    "noir": "black and white film noir style with dramatic shadows",
    "pop-art": "bold pop art style like Andy Warhol",
    "sketch": "pencil sketch drawing style",
    "vintage": "vintage 1970s photograph with warm tones and grain",
    "cinematic": "cinematic movie still with dramatic lighting",
}

# Viewpoint presets
VIEWPOINT_PRESETS = {
    "front": "front facing view, looking directly at camera",
    "profile": "side profile view",
    "three-quarter": "3/4 angle view, slightly turned",
    "looking-up": "low angle shot, looking up at subject",
    "looking-down": "high angle shot, looking down at subject",
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


def decode_and_save(base64_data: str, output_path: str):
    """Decode base64 and save to file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))


def build_prompt(
    custom_prompt: Optional[str] = None,
    background: Optional[str] = None,
    style: Optional[str] = None,
    viewpoint: Optional[str] = None,
) -> str:
    """Build edit prompt from various options."""
    parts = []

    if custom_prompt:
        parts.append(custom_prompt)

    if background:
        # Check if it's a preset
        bg_desc = BACKGROUND_PRESETS.get(background.lower(), background)
        parts.append(f"Change the background to {bg_desc}")

    if style:
        # Check if it's a preset
        style_desc = STYLE_PRESETS.get(style.lower(), style)
        parts.append(f"Restyle the image as {style_desc}")

    if viewpoint:
        # Check if it's a preset
        vp_desc = VIEWPOINT_PRESETS.get(viewpoint.lower(), viewpoint)
        parts.append(f"Change the viewpoint to {vp_desc}")

    if not parts:
        raise ValueError("No edit specified. Use --prompt, --background, --style, or --viewpoint")

    # Combine parts
    prompt = ". ".join(parts)

    # Add identity preservation hint for portrait edits
    if background or viewpoint:
        prompt += ". Preserve the person's facial features and identity."

    return prompt


def call_endpoint(payload: dict, timeout: int = 600) -> tuple[dict, float]:
    """Call RunPod endpoint and return (result, elapsed_seconds)."""
    if not RUNPOD_API_KEY:
        log("RUNPOD_API_KEY not set in .env", "error")
        sys.exit(1)

    if not QWEN_EDIT_ENDPOINT:
        log("RUNPOD_QWEN_EDIT_ENDPOINT_ID not set in .env", "error")
        log("Deploy the endpoint first, then add to .env", "info")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    start = time.time()

    try:
        # Submit async job
        run_url = f"https://api.runpod.ai/v2/{QWEN_EDIT_ENDPOINT}/run"
        response = requests.post(run_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        job_id = result.get("id")
        status = result.get("status")

        if status == "COMPLETED":
            return result.get("output", result), time.time() - start

        if status == "FAILED":
            return {"error": result.get("error", "Unknown error")}, time.time() - start

        # Poll for completion
        log(f"Processing... (cold start may take 5-10 min on first run)", "warn")
        status_url = f"https://api.runpod.ai/v2/{QWEN_EDIT_ENDPOINT}/status/{job_id}"

        poll_interval = 5
        last_status = None
        while time.time() - start < timeout:
            time.sleep(poll_interval)
            elapsed = time.time() - start

            status_resp = requests.get(status_url, headers=headers, timeout=30)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status != last_status:
                if status == "IN_PROGRESS":
                    log(f"[{elapsed:.0f}s] Generating...", "dim")
                elif status == "IN_QUEUE":
                    log(f"[{elapsed:.0f}s] Waiting for GPU...", "dim")
                last_status = status

            if status == "COMPLETED":
                return status_data.get("output", status_data), time.time() - start

            if status == "FAILED":
                error = status_data.get("error", "Unknown error")
                return {"error": error}, time.time() - start

            if status in ("CANCELLED", "TIMED_OUT"):
                return {"error": f"Job {status}"}, time.time() - start

        return {"error": "polling timeout"}, time.time() - start

    except requests.exceptions.Timeout:
        return {"error": "timeout"}, time.time() - start
    except Exception as e:
        return {"error": str(e)}, time.time() - start


def edit_image(
    input_paths: list[str],
    prompt: str,
    output_path: Optional[str] = None,
    seed: Optional[int] = None,
    steps: int = 8,
    open_result: bool = True,
    verbose: bool = False,
) -> Optional[str]:
    """
    Edit image(s) with the given prompt.

    Returns output path on success, None on failure.
    """
    # Validate inputs
    for path in input_paths:
        if not Path(path).exists():
            log(f"File not found: {path}", "error")
            return None

    # Get image info
    with Image.open(input_paths[0]) as img:
        log(f"Input: {input_paths[0]} ({img.size[0]}x{img.size[1]})", "info")

    if len(input_paths) > 1:
        log(f"Additional inputs: {', '.join(input_paths[1:])}", "dim")

    log(f"Prompt: {prompt}", "info")

    # Build payload with primary image + optional reference images
    payload = {
        "input": {
            "image_base64": encode_image(input_paths[0]),
            "prompt": prompt,
            "num_inference_steps": steps,
        }
    }

    # Add additional reference images (up to 2 more for 3 total)
    if len(input_paths) > 1:
        log(f"Multi-image mode: {len(input_paths)} images", "info")
        payload["input"]["images_base64"] = [encode_image(p) for p in input_paths[1:3]]

    if seed is not None:
        payload["input"]["seed"] = seed

    # Call endpoint
    result, elapsed = call_endpoint(payload)

    if "error" in result:
        log(f"Edit failed: {result['error']}", "error")
        return None

    # Determine output path
    if output_path is None:
        input_stem = Path(input_paths[0]).stem
        output_path = f"{input_stem}_edited.png"

    # Save result
    decode_and_save(result["edited_image_base64"], output_path)

    # Report results
    inference_ms = result.get("inference_time_ms", 0)
    output_size = result.get("image_size", [0, 0])

    log(f"Saved: {output_path}", "success")
    log(f"Time: {elapsed:.1f}s total, {inference_ms/1000:.1f}s inference", "dim")
    log(f"Output: {output_size[0]}x{output_size[1]}", "dim")
    log(f"Seed: {result.get('seed', 'unknown')}", "dim")

    if verbose:
        # Estimate cost (L4 pricing)
        cost = (elapsed / 3600) * 0.34
        log(f"Est. cost: ${cost:.4f}", "dim")

    # Open result on macOS
    if open_result and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", output_path], check=False)

    return output_path


def batch_edit(
    input_dir: str,
    output_dir: str,
    prompt: str,
    seed: Optional[int] = None,
    steps: int = 8,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Batch edit all images in a directory.

    Returns (success_count, fail_count).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        log(f"Input directory not found: {input_dir}", "error")
        return 0, 0

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

    if not images:
        log(f"No images found in {input_dir}", "warn")
        return 0, 0

    log(f"Found {len(images)} images to process", "info")

    success = 0
    fail = 0

    for i, img_path in enumerate(images, 1):
        log(f"\n[{i}/{len(images)}] Processing {img_path.name}...", "info")
        out_file = output_path / f"{img_path.stem}_edited.png"

        result = edit_image(
            input_paths=[str(img_path)],
            prompt=prompt,
            output_path=str(out_file),
            seed=seed,
            steps=steps,
            open_result=False,
            verbose=verbose,
        )

        if result:
            success += 1
        else:
            fail += 1

    log(f"\nBatch complete: {success} success, {fail} failed", "success" if fail == 0 else "warn")
    return success, fail


def list_presets():
    """Print available presets."""
    print("\n\033[94mBackground Presets (--background):\033[0m")
    for name, desc in BACKGROUND_PRESETS.items():
        print(f"  {name:12} - {desc[:50]}...")

    print("\n\033[94mStyle Presets (--style):\033[0m")
    for name, desc in STYLE_PRESETS.items():
        print(f"  {name:12} - {desc[:50]}...")

    print("\n\033[94mViewpoint Presets (--viewpoint):\033[0m")
    for name, desc in VIEWPOINT_PRESETS.items():
        print(f"  {name:12} - {desc[:50]}...")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered image editing using Qwen-Image-Edit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input photo.jpg --background pyramids
  %(prog)s --input photo.jpg --style cyberpunk
  %(prog)s --input photo.jpg --prompt "Add sunglasses and a smile"
  %(prog)s --input photo.jpg --background office --style cinematic
  %(prog)s --input-dir ./photos --background studio --output-dir ./edited
  %(prog)s --list-presets
        """
    )

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--input", "-i", nargs="+", help="Input image(s)")
    input_group.add_argument("--input-dir", help="Directory of images for batch processing")

    # Edit options
    edit_group = parser.add_argument_group("Edit Options")
    edit_group.add_argument("--prompt", "-p", help="Custom edit prompt")
    edit_group.add_argument("--background", "-b", help="Background preset or description")
    edit_group.add_argument("--style", "-s", help="Style preset or description")
    edit_group.add_argument("--viewpoint", "-v", help="Viewpoint preset or description")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", help="Output file path")
    output_group.add_argument("--output-dir", help="Output directory for batch processing")
    output_group.add_argument("--no-open", action="store_true", help="Don't open result automatically")

    # Advanced options
    adv_group = parser.add_argument_group("Advanced")
    adv_group.add_argument("--seed", type=int, help="Random seed for reproducibility")
    adv_group.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8)")
    adv_group.add_argument("--verbose", action="store_true", help="Show detailed output")

    # Utility
    parser.add_argument("--list-presets", action="store_true", help="List available presets")

    args = parser.parse_args()

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        return

    # Validate inputs
    if not args.input and not args.input_dir:
        parser.print_help()
        print("\n\033[91m!! Specify --input or --input-dir\033[0m")
        sys.exit(1)

    if not any([args.prompt, args.background, args.style, args.viewpoint]):
        parser.print_help()
        print("\n\033[91m!! Specify at least one edit: --prompt, --background, --style, or --viewpoint\033[0m")
        sys.exit(1)

    # Build the prompt
    try:
        prompt = build_prompt(
            custom_prompt=args.prompt,
            background=args.background,
            style=args.style,
            viewpoint=args.viewpoint,
        )
    except ValueError as e:
        log(str(e), "error")
        sys.exit(1)

    print()
    log("Qwen Image Edit", "info")
    log("=" * 40, "dim")

    # Batch or single
    if args.input_dir:
        output_dir = args.output_dir or f"{args.input_dir}_edited"
        batch_edit(
            input_dir=args.input_dir,
            output_dir=output_dir,
            prompt=prompt,
            seed=args.seed,
            steps=args.steps,
            verbose=args.verbose,
        )
    else:
        edit_image(
            input_paths=args.input,
            prompt=prompt,
            output_path=args.output,
            seed=args.seed,
            steps=args.steps,
            open_result=not args.no_open,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
