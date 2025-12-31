#!/usr/bin/env python3
"""
RunPod serverless handler for Wan2.2 Image-to-Video with LightX2V acceleration.

Supports:
- i2v: Generate video from a single image based on motion prompt

This is Worker 2 in the video generation pipeline:
  Reference Image -> [Qwen-Edit] -> Edited Frame -> [Wan I2V] -> Video Clip

Input format:
{
    "input": {
        "image_base64": str,           # Required - starting frame (base64 encoded)
        "prompt": str,                  # Required - motion description
        "negative_prompt": str,         # Optional (default: "")
        "num_frames": int,              # Optional (default: 81 = 5 sec @ 16fps)
        "num_inference_steps": int,     # Optional (default: 40)
        "guidance_scale": float,        # Optional (default: 5.0)
        "seed": int,                    # Optional (random if not set)
        "fps": int,                     # Optional (default: 16)
        "height": int,                  # Optional (default: 480)
        "width": int,                   # Optional (default: 832)
    }
}

Output format:
{
    "success": true,
    "video_base64": str,               # MP4 encoded video
    "last_frame_base64": str,          # Final frame for chaining
    "seed": int,
    "inference_time_ms": int,
    "num_frames": int,
    "fps": int
}
"""

import base64
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import runpod
import torch
from PIL import Image

# Model path (baked into Docker image)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/wan-i2v"))

# Lazy-loaded pipeline
_pipeline = None
_pipeline_config = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def get_gpu_vram_gb() -> int:
    """Detect GPU VRAM using PyTorch."""
    try:
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            vram_gb = props.total_memory // (1024 ** 3)
            log(f"Detected GPU: {props.name}, VRAM: {vram_gb}GB")
            return vram_gb
    except Exception as e:
        log(f"Warning: Could not detect GPU VRAM: {e}")
    return 48  # Default assumption


def decode_base64_image(image_base64: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        log(f"Error decoding base64 image: {e}")
        return None


def encode_image_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_video_base64(video_path: str) -> str:
    """Encode video file to base64 string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_last_frame(video_path: str) -> Optional[Image.Image]:
    """Extract the last frame from a video file."""
    try:
        import imageio.v3 as iio

        # Read all frames and get the last one
        frames = iio.imread(video_path, plugin="pyav")
        if len(frames) > 0:
            last_frame = frames[-1]
            return Image.fromarray(last_frame)
    except Exception as e:
        log(f"Error extracting last frame: {e}")

    # Fallback to ffmpeg
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        subprocess.run([
            "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
            "-update", "1", "-q:v", "1", tmp_path
        ], capture_output=True, timeout=30)

        if Path(tmp_path).exists():
            img = Image.open(tmp_path)
            os.unlink(tmp_path)
            return img
    except Exception as e:
        log(f"FFmpeg fallback failed: {e}")

    return None


def get_pipeline(vram_gb: int):
    """Get or initialize LightX2V pipeline (lazy loading)."""
    global _pipeline, _pipeline_config

    current_config = {"vram_gb": vram_gb}
    if _pipeline is not None and _pipeline_config == current_config:
        return _pipeline

    if _pipeline is not None:
        log("Pipeline config changed, reinitializing...")
        del _pipeline
        torch.cuda.empty_cache()

    log(f"Loading LightX2V Wan2.2 I2V pipeline (VRAM: {vram_gb}GB)...")
    start = time.time()

    from lightx2v import LightX2VPipeline

    _pipeline = LightX2VPipeline(
        model_path=str(MODEL_PATH),
        model_cls="wan2.2_moe",
        task="i2v",
    )

    # Enable offloading based on VRAM
    if vram_gb < 48:
        log("Enabling CPU offload for memory efficiency")
        _pipeline.enable_offload(
            cpu_offload=True,
            offload_granularity="block",
            text_encoder_offload=True,
        )

    if vram_gb < 32:
        log("Enabling aggressive offloading for low VRAM")
        # Additional memory saving for 24GB cards

    _pipeline_config = current_config
    log(f"Pipeline loaded in {time.time() - start:.1f}s")
    return _pipeline


def upload_to_r2(data: bytes, job_id: str, r2_config: dict, suffix: str, content_type: str) -> tuple[Optional[str], Optional[str]]:
    """Upload data to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config
        import uuid

        log(f"Uploading to R2 ({len(data) // 1024}KB)...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        ext = ".mp4" if "video" in content_type else ".png"
        object_key = f"wan-i2v/results/{job_id}_{uuid.uuid4().hex[:8]}{suffix}{ext}"

        client.put_object(
            Bucket=r2_config["bucket_name"],
            Key=object_key,
            Body=data,
            ContentType=content_type
        )

        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )

        log(f"  R2 upload complete: {object_key}")
        return presigned_url, object_key
    except Exception as e:
        log(f"Error uploading to R2: {e}")
        return None, None


def handle_i2v(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Handle image-to-video generation using Wan2.2.

    Required inputs:
        image_base64: Base64 encoded starting frame
        prompt: Motion description (e.g., "Person speaking naturally with subtle gestures")

    Optional inputs:
        negative_prompt: Things to avoid (default: "")
        num_frames: Number of frames to generate (default: 81 = ~5 sec @ 16fps)
        num_inference_steps: Diffusion steps (default: 40)
        guidance_scale: CFG scale (default: 5.0)
        seed: Random seed for reproducibility
        fps: Output video FPS (default: 16)
        height: Output height (default: 480)
        width: Output width (default: 832)
        r2: R2 config for result upload
    """
    start_time = time.time()

    # Extract inputs
    image_base64 = job_input.get("image_base64")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "")
    num_frames = job_input.get("num_frames", 81)
    num_inference_steps = job_input.get("num_inference_steps", 40)
    guidance_scale = job_input.get("guidance_scale", 5.0)
    seed = job_input.get("seed")
    fps = job_input.get("fps", 16)
    height = job_input.get("height", 480)
    width = job_input.get("width", 832)
    r2_config = job_input.get("r2")

    # Validate required inputs
    if not image_base64:
        return {"error": "Missing required 'image_base64' in input"}
    if not prompt:
        return {"error": "Missing required 'prompt' in input"}

    # Decode input image
    input_image = decode_base64_image(image_base64)
    if input_image is None:
        return {"error": "Failed to decode input image from base64"}

    log(f"Input image size: {input_image.size}")

    # Save input image to temp file
    input_path = str(work_dir / "input.png")
    input_image.save(input_path)

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    log(f"Using seed: {seed}")

    # Detect VRAM and get pipeline
    vram_gb = get_gpu_vram_gb()
    pipe = get_pipeline(vram_gb)

    # Select attention mode
    try:
        import flash_attn
        attn_mode = "sage_attn2"  # Optimized for video
    except ImportError:
        attn_mode = "sdpa"

    log(f"Using attention mode: {attn_mode}")

    # Configure generator
    log(f"Configuring: {num_frames} frames @ {fps}fps, {width}x{height}, steps={num_inference_steps}")
    pipe.create_generator(
        attn_mode=attn_mode,
        infer_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=[guidance_scale, guidance_scale],
    )

    # Generate video
    output_path = str(work_dir / "output.mp4")

    log(f"Generating video: '{prompt[:50]}...'")
    gen_start = time.time()

    pipe.generate(
        seed=seed,
        image_path=input_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        save_result_path=output_path,
    )

    gen_time = time.time() - gen_start
    log(f"Generation completed in {gen_time:.1f}s")

    # Verify output
    if not Path(output_path).exists():
        return {"error": "Generation failed - no output video produced"}

    video_size = Path(output_path).stat().st_size
    log(f"Output video: {video_size // 1024}KB")

    # Extract last frame for chaining
    last_frame = extract_last_frame(output_path)
    last_frame_base64 = None
    if last_frame:
        last_frame_base64 = encode_image_base64(last_frame)
        log(f"Extracted last frame: {last_frame.size}")
    else:
        log("Warning: Could not extract last frame")

    # Encode video
    video_base64 = encode_video_base64(output_path)

    elapsed_ms = int((time.time() - start_time) * 1000)

    result = {
        "success": True,
        "video_base64": video_base64,
        "last_frame_base64": last_frame_base64,
        "seed": seed,
        "inference_time_ms": elapsed_ms,
        "num_frames": num_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "video_size_bytes": video_size,
    }

    # Upload to R2 if configured
    if r2_config:
        # Upload video
        video_bytes = base64.b64decode(video_base64)
        video_url, video_key = upload_to_r2(video_bytes, job_id, r2_config, "_video", "video/mp4")
        if video_url:
            result["video_url"] = video_url
            result["video_r2_key"] = video_key

        # Upload last frame
        if last_frame_base64:
            frame_bytes = base64.b64decode(last_frame_base64)
            frame_url, frame_key = upload_to_r2(frame_bytes, job_id, r2_config, "_lastframe", "image/png")
            if frame_url:
                result["last_frame_url"] = frame_url
                result["last_frame_r2_key"] = frame_key

    return result


def handler(job: dict) -> dict:
    """
    Main RunPod handler - routes to i2v operation.
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    operation = job_input.get("operation", "i2v")
    log(f"Job {job_id}: operation={operation}")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "i2v":
            return handle_i2v(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: i2v"}
    except torch.cuda.OutOfMemoryError as e:
        log(f"CUDA OOM: {e}")
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try reducing num_frames or resolution."}
    except Exception as e:
        import traceback
        log(f"Handler exception: {e}")
        log(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}
    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            log("Cleaned up working directory")
        except Exception:
            pass


# RunPod serverless entry point
if __name__ == "__main__":
    log("Starting RunPod Wan2.2 I2V handler...")
    log(f"Model path: {MODEL_PATH}, exists: {MODEL_PATH.exists()}")

    # Check CUDA
    if torch.cuda.is_available():
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        vram_gb = get_gpu_vram_gb()
    else:
        log("WARNING: CUDA not available!")

    runpod.serverless.start({"handler": handler})
