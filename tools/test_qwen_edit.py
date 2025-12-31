#!/usr/bin/env python3
"""
Test script for Qwen-Edit RunPod endpoint (Milestone 1.2).

Usage:
  # Basic test with a test image
  python tools/test_qwen_edit.py --image test.png --prompt "Change background to office"

  # Test with timing and cost estimates
  python tools/test_qwen_edit.py --image portrait.png --prompt "Add subtle smile" --verbose

  # Generate a test pattern image if you don't have one
  python tools/test_qwen_edit.py --generate-test-image

  # Quick health check (just verifies endpoint responds)
  python tools/test_qwen_edit.py --health-check
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
QWEN_EDIT_ENDPOINT = os.getenv("RUNPOD_QWEN_EDIT_ENDPOINT_ID")

# GPU costs per hour (approximate RunPod serverless pricing)
GPU_COSTS = {
    "L4": 0.34,
    "RTX4090": 0.44,
    "A6000": 0.76,
    "A100-40GB": 1.29,
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
    prefix = {"info": "→", "success": "✓", "error": "✗", "warn": "⚠", "dim": " "}
    color = colors.get(level, "")
    print(f"{color}{prefix.get(level, '→')} {msg}{reset}")


def encode_image(path: str) -> str:
    """Encode image file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decode_and_save(base64_data: str, output_path: str):
    """Decode base64 and save to file."""
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))


def generate_test_image(output_path: str = "test_image.png"):
    """Generate a simple test image with colored rectangles."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log("PIL not installed. Run: pip install Pillow", "error")
        sys.exit(1)

    # Create a 512x512 test image
    img = Image.new("RGB", (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Add some geometric shapes
    draw.rectangle([50, 50, 200, 200], fill=(65, 105, 225))  # Blue square
    draw.ellipse([300, 50, 450, 200], fill=(220, 20, 60))    # Red circle
    draw.rectangle([50, 300, 200, 450], fill=(50, 205, 50))  # Green square
    draw.polygon([(375, 300), (300, 450), (450, 450)], fill=(255, 165, 0))  # Orange triangle

    # Add text
    draw.text((180, 230), "TEST IMAGE", fill=(50, 50, 50))

    img.save(output_path)
    log(f"Generated test image: {output_path}", "success")
    return output_path


def call_endpoint(payload: dict, timeout: int = 600) -> tuple[dict, float]:
    """
    Call RunPod endpoint and return (result, elapsed_seconds).

    Uses async run + polling for cold start support.
    """
    if not RUNPOD_API_KEY:
        log("RUNPOD_API_KEY not set in .env", "error")
        sys.exit(1)

    if not QWEN_EDIT_ENDPOINT:
        log("RUNPOD_QWEN_EDIT_ENDPOINT_ID not set in .env", "error")
        log("Deploy the endpoint first, then add to .env:", "info")
        log("  RUNPOD_QWEN_EDIT_ENDPOINT_ID=your_endpoint_id", "dim")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    log(f"Calling endpoint {QWEN_EDIT_ENDPOINT}...")
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
            elapsed = time.time() - start
            return result.get("output", result), elapsed

        if status == "FAILED":
            elapsed = time.time() - start
            return {"error": result.get("error", "Unknown error")}, elapsed

        # Poll for completion
        log(f"Job {job_id} queued, polling... (cold start may take 5-10 min)", "warn")
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
                log(f"  [{elapsed:.0f}s] Status: {status}", "dim")
                last_status = status

            if status == "COMPLETED":
                return status_data.get("output", status_data), elapsed

            if status == "FAILED":
                error = status_data.get("error", "Unknown error")
                log(f"Job failed: {error}", "error")
                return {"error": error}, elapsed

            if status in ("CANCELLED", "TIMED_OUT"):
                return {"error": f"Job {status}"}, elapsed

        # Timeout
        elapsed = time.time() - start
        log(f"Polling timed out after {elapsed:.0f}s", "error")
        return {"error": "polling timeout"}, elapsed

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        log(f"Request timed out after {timeout}s", "error")
        return {"error": "timeout"}, elapsed
    except Exception as e:
        elapsed = time.time() - start
        log(f"Request failed: {e}", "error")
        return {"error": str(e)}, elapsed


def estimate_cost(elapsed_seconds: float, gpu_type: str = "L4") -> dict:
    """Estimate cost based on elapsed time."""
    hourly_rate = GPU_COSTS.get(gpu_type, 0.34)
    cost = (elapsed_seconds / 3600) * hourly_rate

    return {
        "gpu_type": gpu_type,
        "elapsed_seconds": elapsed_seconds,
        "hourly_rate": hourly_rate,
        "estimated_cost": cost,
        "cost_per_1000": cost * 1000,
    }


def run_health_check():
    """Quick health check - verify endpoint responds."""
    log("Running health check...", "info")

    # Generate minimal test image
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except ImportError:
        log("PIL not installed for health check", "error")
        return False

    payload = {
        "input": {
            "image_base64": image_base64,
            "prompt": "test",
            "num_inference_steps": 1,  # Minimal steps for speed
        }
    }

    result, elapsed = call_endpoint(payload, timeout=180)

    if "error" in result:
        log(f"Health check failed: {result['error']}", "error")
        return False

    if result.get("success"):
        log(f"Health check passed! Response in {elapsed:.1f}s", "success")
        return True

    log(f"Unexpected response: {result}", "warn")
    return False


def run_edit_test(image_path: str, prompt: str, seed: int = None,
                  num_steps: int = 8, use_fp8: bool = False, verbose: bool = False):
    """Run a single edit test and display results."""

    if not Path(image_path).exists():
        log(f"Image not found: {image_path}", "error")
        sys.exit(1)

    # Get image info
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_size = img.size
            log(f"Input image: {image_path} ({img_size[0]}x{img_size[1]})", "info")
    except Exception:
        log(f"Input image: {image_path}", "info")

    log(f"Prompt: {prompt}", "info")
    if seed:
        log(f"Seed: {seed}", "dim")

    # Build payload
    payload = {
        "input": {
            "image_base64": encode_image(image_path),
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "use_fp8": use_fp8,
        }
    }
    log(f"Mode: {'FP8 quantized' if use_fp8 else 'BF16 full precision'}", "dim")
    if seed:
        payload["input"]["seed"] = seed

    # Call endpoint
    print()
    result, elapsed = call_endpoint(payload, timeout=300)
    print()

    if "error" in result:
        log(f"Edit failed: {result['error']}", "error")
        return None

    # Extract metrics
    inference_ms = result.get("inference_time_ms", 0)
    output_size = result.get("image_size", [0, 0])
    used_seed = result.get("seed", "unknown")

    log(f"Edit completed!", "success")
    log(f"  Total time: {elapsed:.1f}s", "dim")
    log(f"  Inference time: {inference_ms}ms ({inference_ms/1000:.1f}s)", "dim")
    log(f"  Output size: {output_size[0]}x{output_size[1]}", "dim")
    log(f"  Seed used: {used_seed}", "dim")

    # Save output
    output_path = Path(image_path).stem + "_edited.png"
    decode_and_save(result["edited_image_base64"], output_path)
    log(f"  Output saved: {output_path}", "success")

    # Cost estimate
    cost = estimate_cost(elapsed)
    print()
    log("Cost estimate (L4 24GB @ $0.34/hr):", "info")
    log(f"  This edit: ${cost['estimated_cost']:.6f}", "dim")
    log(f"  Per 1000 edits: ${cost['cost_per_1000']:.2f}", "dim")

    # For pipeline estimation
    print()
    log("Pipeline cost projection (60-sec video = 12 segments):", "info")
    edit_cost_per_segment = cost['estimated_cost']
    i2v_cost_per_segment = 0.0063  # ~30s on A6000 @ $0.76/hr (from estimates)
    total_per_segment = edit_cost_per_segment + i2v_cost_per_segment
    log(f"  Edit per segment: ${edit_cost_per_segment:.6f}", "dim")
    log(f"  I2V per segment: ${i2v_cost_per_segment:.4f} (estimated)", "dim")
    log(f"  Total per segment: ${total_per_segment:.4f}", "dim")
    log(f"  12 segments (60s): ${total_per_segment * 12:.4f}", "dim")

    if verbose:
        print()
        log("Full response:", "info")
        safe_result = {k: v for k, v in result.items() if k != "edited_image_base64"}
        print(json.dumps(safe_result, indent=2))

    # Open output
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", output_path])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen-Edit RunPod endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --image portrait.png --prompt "Change background to office"
  %(prog)s --image test.png --prompt "Add warm lighting" --seed 42 --verbose
  %(prog)s --generate-test-image
  %(prog)s --health-check
        """
    )

    parser.add_argument("--image", "-i", help="Input image path")
    parser.add_argument("--prompt", "-p", help="Edit prompt")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=8, help="Inference steps (default: 8)")
    parser.add_argument("--fp8", action="store_true", help="Use FP8 quantization (lower VRAM)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full response")
    parser.add_argument("--generate-test-image", action="store_true",
                        help="Generate a test image and exit")
    parser.add_argument("--health-check", action="store_true",
                        help="Quick endpoint health check")

    args = parser.parse_args()

    print()
    log("Qwen-Edit Endpoint Test", "info")
    log("=" * 40, "dim")
    print()

    if args.generate_test_image:
        generate_test_image()
        return

    if args.health_check:
        success = run_health_check()
        sys.exit(0 if success else 1)

    if not args.image or not args.prompt:
        parser.print_help()
        print()
        log("Required: --image and --prompt", "error")
        sys.exit(1)

    run_edit_test(
        image_path=args.image,
        prompt=args.prompt,
        seed=args.seed,
        num_steps=args.steps,
        use_fp8=args.fp8,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
