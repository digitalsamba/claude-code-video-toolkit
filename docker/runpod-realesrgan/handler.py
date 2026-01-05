#!/usr/bin/env python3
"""
RunPod serverless handler for Real-ESRGAN image upscaling.

Supports:
- upscale: Upscale images using Real-ESRGAN models

Input format:
{
    "operation": "upscale",
    "image_url": "https://...",
    "scale": 4,              # 2 or 4 (default: 4)
    "model": "general",      # general, anime, or photo (default: general)
    "face_enhance": false,   # Use GFPGAN for face enhancement (default: false)
    "output_format": "png"   # png, jpg, webp (default: png)
}

Output format:
{
    "success": true,
    "output_url": "https://...",
    "input_dimensions": "800x600",
    "output_dimensions": "3200x2400",
    "model_used": "RealESRGAN_x4plus",
    "processing_time_seconds": 2.5
}
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import runpod
import torch
from PIL import Image

# Model paths (baked into Docker image)
WEIGHTS_DIR = Path("/app/weights")
MODEL_PATHS = {
    "general": WEIGHTS_DIR / "RealESRGAN_x4plus.pth",
    "anime": WEIGHTS_DIR / "RealESRGAN_x4plus_anime_6B.pth",
    "photo": WEIGHTS_DIR / "realesr-general-x4v3.pth",
}

# Cached upscaler instances
_upscalers = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def get_upscaler(model: str = "general", scale: int = 4, face_enhance: bool = False):
    """Get or create Real-ESRGAN upscaler instance."""
    cache_key = f"{model}_{scale}_{face_enhance}"

    if cache_key in _upscalers:
        return _upscalers[cache_key]

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Select model architecture based on model type
    if model == "anime":
        model_path = str(MODEL_PATHS["anime"])
        # Anime model uses 6 blocks
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model == "photo":
        model_path = str(MODEL_PATHS["photo"])
        # General v3 model
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:  # general
        model_path = str(MODEL_PATHS["general"])
        # Standard x4plus model
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

    log(f"Loading model: {model} from {model_path}")

    # Create upscaler
    upscaler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=net,
        tile=0,  # 0 = no tiling
        tile_pad=10,
        pre_pad=0,
        half=True,  # Use fp16 for speed
        gpu_id=0,
    )

    # Add face enhancement if requested
    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upscaler
        )
        _upscalers[cache_key] = face_enhancer
        return face_enhancer

    _upscalers[cache_key] = upscaler
    return upscaler


def download_file(url: str, output_path: str, description: str = "file") -> bool:
    """Download file from URL with progress logging."""
    try:
        log(f"Downloading {description} from {url[:80]}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        log(f"  Downloaded {description}: {Path(output_path).stat().st_size // 1024}KB")
        return True
    except Exception as e:
        log(f"Error downloading {description}: {e}")
        return False


def upload_to_r2(file_path: str, job_id: str, r2_config: dict, extension: str = "png") -> tuple[Optional[str], Optional[str]]:
    """Upload file to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config
        import uuid

        log(f"Uploading result to R2 ({Path(file_path).stat().st_size // 1024}KB)...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        object_key = f"upscale/results/{job_id}_{uuid.uuid4().hex[:8]}.{extension}"

        # Set content type based on extension
        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        content_type = content_types.get(extension.lower(), "application/octet-stream")

        client.upload_file(
            file_path,
            r2_config["bucket_name"],
            object_key,
            ExtraArgs={"ContentType": content_type}
        )

        # Generate presigned URL (valid for 2 hours)
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )

        log(f"  R2 upload complete: {object_key}")
        return presigned_url, object_key
    except ImportError:
        log("Error: boto3 not available for R2 upload")
        return None, None
    except Exception as e:
        log(f"Error uploading to R2: {e}")
        return None, None


def upload_file(file_path: str, job_id: str, r2_config: Optional[dict] = None, extension: str = "png") -> dict:
    """Upload file and return upload info."""
    # Try R2 first if configured
    if r2_config:
        url, r2_key = upload_to_r2(file_path, job_id, r2_config, extension)
        if url:
            return {"output_url": url, "r2_key": r2_key}
        log("R2 upload failed, falling back to RunPod storage")

    # Fall back to RunPod storage
    try:
        log(f"Uploading result to RunPod storage ({Path(file_path).stat().st_size // 1024}KB)...")

        result_url = runpod.serverless.utils.rp_upload.upload_file_to_bucket(
            file_name=f"{job_id}_upscaled.{extension}",
            file_location=file_path
        )

        log(f"  Upload complete: {result_url[:80]}...")
        return {"output_url": result_url}
    except Exception as e:
        log(f"Error uploading file: {e}")
        return {}


def handle_upscale(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Handle upscale operation using Real-ESRGAN.

    Required inputs:
        image_url: URL to image file

    Optional inputs:
        scale: 2 or 4 (default: 4)
        model: general, anime, or photo (default: general)
        face_enhance: Use GFPGAN for face enhancement (default: false)
        output_format: png, jpg, webp (default: png)
        r2: R2 config for result upload
    """
    start_time = time.time()

    # Validate inputs
    image_url = job_input.get("image_url")
    scale = job_input.get("scale", 4)
    model = job_input.get("model", "general")
    face_enhance = job_input.get("face_enhance", False)
    output_format = job_input.get("output_format", "png").lower()
    r2_config = job_input.get("r2")

    if not image_url:
        return {"error": "Missing required 'image_url' in input"}

    if scale not in [2, 4]:
        return {"error": f"Invalid scale: {scale}. Must be 2 or 4"}

    if model not in MODEL_PATHS:
        return {"error": f"Invalid model: {model}. Must be one of: {list(MODEL_PATHS.keys())}"}

    if output_format not in ["png", "jpg", "jpeg", "webp"]:
        return {"error": f"Invalid output_format: {output_format}. Must be png, jpg, or webp"}

    log(f"Processing options: scale={scale}, model={model}, face_enhance={face_enhance}, format={output_format}")

    # Download image
    # Determine input extension from URL
    url_path = image_url.split("?")[0]  # Remove query params
    input_ext = Path(url_path).suffix.lower() or ".png"
    input_path = str(work_dir / f"input{input_ext}")

    if not download_file(image_url, input_path, "image"):
        return {"error": "Failed to download image from URL"}

    # Read image
    try:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Failed to read image file"}

        input_height, input_width = img.shape[:2]
        log(f"Input image: {input_width}x{input_height}")
    except Exception as e:
        return {"error": f"Failed to read image: {e}"}

    # Get upscaler
    try:
        upscaler = get_upscaler(model=model, scale=scale, face_enhance=face_enhance)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    # Upscale
    try:
        log("Upscaling...")
        inference_start = time.time()

        if face_enhance:
            # GFPGAN returns (cropped_faces, restored_faces, restored_img)
            _, _, output = upscaler.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
        else:
            output, _ = upscaler.enhance(img, outscale=scale)

        inference_time = time.time() - inference_start
        log(f"  Inference time: {inference_time:.2f}s")

        output_height, output_width = output.shape[:2]
        log(f"Output image: {output_width}x{output_height}")

    except Exception as e:
        import traceback
        log(f"Upscale error: {e}")
        log(traceback.format_exc())
        return {"error": f"Upscale failed: {e}"}

    # Save output
    output_ext = "jpg" if output_format == "jpeg" else output_format
    output_path = str(work_dir / f"output.{output_ext}")

    try:
        if output_format in ["jpg", "jpeg"]:
            cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif output_format == "webp":
            cv2.imwrite(output_path, output, [cv2.IMWRITE_WEBP_QUALITY, 95])
        else:  # png
            cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 6])

        log(f"Saved output: {Path(output_path).stat().st_size // 1024}KB")
    except Exception as e:
        return {"error": f"Failed to save output: {e}"}

    # Upload result
    upload_result = upload_file(output_path, job_id, r2_config, output_ext)

    if not upload_result.get("output_url"):
        return {"error": "Failed to upload result image"}

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "output_url": upload_result["output_url"],
        "input_dimensions": f"{input_width}x{input_height}",
        "output_dimensions": f"{output_width}x{output_height}",
        "scale": scale,
        "model_used": model,
        "face_enhance": face_enhance,
        "output_format": output_format,
        "inference_time_seconds": round(inference_time, 2),
        "processing_time_seconds": round(elapsed, 2),
    }

    # Include R2 key if result was uploaded to R2
    if upload_result.get("r2_key"):
        result["r2_key"] = upload_result["r2_key"]

    return result


def handler(job: dict) -> dict:
    """
    Main RunPod handler - routes to specific operations.

    Supports operations:
        - upscale: Upscale images using Real-ESRGAN
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    operation = job_input.get("operation", "upscale")
    log(f"Job {job_id}: operation={operation}")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "upscale":
            return handle_upscale(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: upscale"}
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
    log("Starting RunPod Real-ESRGAN handler...")
    log(f"Weights directory: {WEIGHTS_DIR}")
    log(f"Models available: {list(MODEL_PATHS.keys())}")

    # Verify weights exist
    for name, path in MODEL_PATHS.items():
        if path.exists():
            log(f"  {name}: OK ({path.stat().st_size // (1024*1024)}MB)")
        else:
            log(f"  {name}: MISSING")

    runpod.serverless.start({"handler": handler})
