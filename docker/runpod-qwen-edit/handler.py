#!/usr/bin/env python3
"""
RunPod serverless handler for Qwen-Image-Edit with LightX2V acceleration.

Supports:
- edit: Edit an image based on text prompt while preserving identity

This is Worker 1 in the video generation pipeline:
  Reference Image -> [Qwen-Edit] -> Edited Frame -> [Wan I2V] -> Video

Input format:
{
    "input": {
        "image_base64": str,           # Required - input image (base64 encoded)
        "prompt": str,                  # Required - edit instruction
        "negative_prompt": str,         # Optional (default: "")
        "num_inference_steps": int,     # Optional (default: 4 for Lightning, 8 for FP8)
        "guidance_scale": float,        # Optional (default: 1.0)
        "seed": int,                    # Optional (random if not set)
        "use_fp8": bool,               # Optional (default: true, uses FP8 quantization)
        "auto_resize": bool,           # Optional (default: true)
    }
}

Output format:
{
    "success": true,
    "edited_image_base64": str,
    "seed": int,
    "inference_time_ms": int,
    "image_size": [width, height]
}
"""

import base64
import io
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import runpod
import torch
from PIL import Image

# HuggingFace model IDs
MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
FP8_MODEL_ID = "lightx2v/Qwen-Image-Edit-2511-Lightning"

# Lazy-loaded pipeline
_pipeline = None
_pipeline_config = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def setup_hf_cache() -> None:
    """Set up HuggingFace cache - use baked-in cache from image."""
    # Models are baked into image at /root/.cache/huggingface
    baked_cache = Path("/root/.cache/huggingface")
    if baked_cache.exists():
        os.environ["HF_HOME"] = str(baked_cache)
        log(f"Using baked-in model cache: {baked_cache}")
    elif Path("/runpod-volume").exists() and os.access("/runpod-volume", os.W_OK):
        cache_path = Path("/runpod-volume/.cache/huggingface")
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)
        log(f"Using RunPod network volume cache: {cache_path}")
    else:
        log("WARNING: No cache found, will download models")


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
    return 24  # Default assumption


def decode_base64_image(image_base64: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    try:
        # Handle data URI prefix if present
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


def get_pipeline(use_fp8: bool = True):
    """Get or initialize diffusers pipeline (lazy loading)."""
    global _pipeline, _pipeline_config

    # Check if we need to reinitialize (different config)
    current_config = {"use_fp8": use_fp8}
    if _pipeline is not None and _pipeline_config == current_config:
        return _pipeline

    # Need to reinitialize
    if _pipeline is not None:
        log("Pipeline config changed, reinitializing...")
        del _pipeline
        torch.cuda.empty_cache()

    log(f"Loading diffusers pipeline from {MODEL_ID}...")
    start = time.time()

    # Check if models are baked into image (no HF download needed)
    baked_cache = Path("/root/.cache/huggingface")
    local_files_only = baked_cache.exists() and any(baked_cache.iterdir())
    if local_files_only:
        log("Using baked-in models (local_files_only=True)")
    else:
        log("Models not baked, will download from HuggingFace")

    try:
        # Use standard diffusers pipeline
        from diffusers import QwenImageEditPlusPipeline

        _pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        )
        _pipeline.to("cuda")
    except Exception as e:
        import traceback
        log(f"Pipeline loading error: {e}")
        log(f"Full traceback:\n{traceback.format_exc()}")
        raise

    _pipeline_config = current_config
    log(f"Pipeline loaded in {time.time() - start:.1f}s")
    return _pipeline


def upload_to_r2(image_base64: str, job_id: str, r2_config: dict) -> tuple[Optional[str], Optional[str]]:
    """Upload image to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config
        import uuid

        log("Uploading to R2...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        object_key = f"qwen-edit/results/{job_id}_{uuid.uuid4().hex[:8]}.png"

        # Decode and upload
        image_bytes = base64.b64decode(image_base64)
        client.put_object(
            Bucket=r2_config["bucket_name"],
            Key=object_key,
            Body=image_bytes,
            ContentType="image/png"
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


def handle_edit(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Handle image edit operation using Qwen-Image-Edit.

    Required inputs:
        image_base64: Base64 encoded input image
        prompt: Edit instruction (e.g., "Change the background to an office")

    Optional inputs:
        negative_prompt: Things to avoid (default: "")
        num_inference_steps: Number of diffusion steps (default: 4 for LoRA, 8 for FP8)
        guidance_scale: CFG scale (default: 1.0)
        seed: Random seed for reproducibility
        use_fp8: Use FP8 quantization (default: true)
        auto_resize: Automatically resize for optimal processing (default: true)
        r2: R2 config for result upload
    """
    start_time = time.time()

    # Extract inputs
    image_base64 = job_input.get("image_base64")
    prompt = job_input.get("prompt")
    negative_prompt = job_input.get("negative_prompt", "")
    # Default to BF16 (full quality) - requires 48GB+ GPU
    use_fp8 = job_input.get("use_fp8", False)
    num_inference_steps = job_input.get("num_inference_steps", 4)  # Lightning LoRA default
    guidance_scale = job_input.get("guidance_scale", 1.0)
    seed = job_input.get("seed")
    auto_resize = job_input.get("auto_resize", True)
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

    # Save input image to temp file (LightX2V expects file path)
    input_path = str(work_dir / "input.png")
    input_image.save(input_path)

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    log(f"Using seed: {seed}")

    # Get pipeline
    pipe = get_pipeline(use_fp8=use_fp8)

    log(f"Running edit with diffusers: steps={num_inference_steps}, guidance={guidance_scale}")
    gen_start = time.time()

    # Use diffusers API
    try:
        generator = torch.Generator(device="cuda").manual_seed(seed)

        output = pipe(
            image=[input_image],
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else " ",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=4.0,
            generator=generator,
            num_images_per_prompt=1,
        )
        output_image = output.images[0]
    except Exception as e:
        import traceback
        log(f"Generation error: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        raise

    gen_time = time.time() - gen_start
    log(f"Generation completed in {gen_time:.1f}s")
    output_base64 = encode_image_base64(output_image)

    elapsed_ms = int((time.time() - start_time) * 1000)

    result = {
        "success": True,
        "edited_image_base64": output_base64,
        "seed": seed,
        "inference_time_ms": elapsed_ms,
        "image_size": list(output_image.size),
        "num_inference_steps": num_inference_steps,
        "use_fp8": use_fp8,
    }

    # Upload to R2 if configured
    if r2_config:
        url, r2_key = upload_to_r2(output_base64, job_id, r2_config)
        if url:
            result["output_url"] = url
            result["r2_key"] = r2_key

    return result


def handler(job: dict) -> dict:
    """
    Main RunPod handler - routes to edit operation.
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    operation = job_input.get("operation", "edit")
    log(f"Job {job_id}: operation={operation}")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "edit":
            return handle_edit(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: edit"}
    except torch.cuda.OutOfMemoryError as e:
        log(f"CUDA OOM: {e}")
        torch.cuda.empty_cache()
        return {"error": "GPU out of memory. Try with use_fp8=true or smaller image."}
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
    log("Starting RunPod Qwen-Edit handler...")

    # Check CUDA first
    if torch.cuda.is_available():
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        vram_gb = get_gpu_vram_gb()
        log(f"VRAM: {vram_gb}GB")
    else:
        log("WARNING: CUDA not available!")

    # Set up HF cache for model downloads
    setup_hf_cache()

    # Pre-load the pipeline at startup to cache models
    log("Pre-loading pipeline at startup (this may take 5-10 min on first run)...")
    try:
        get_pipeline(use_fp8=False)
        log("Pipeline pre-loaded successfully")
    except Exception as e:
        log(f"Warning: Pipeline pre-load failed: {e}")
        log("Will retry on first request")

    runpod.serverless.start({"handler": handler})
