#!/usr/bin/env python3
"""
RunPod serverless handler for character animation GPU operations.

Supports:
- segment: Generate masks using SAM2 from bboxes or points
- animate: Generate video from image using Stable Video Diffusion

Input format (segment):
{
    "operation": "segment",
    "image_url": "https://...",
    "bboxes": [[x1, y1, x2, y2], ...],  # OR
    "points": [[x, y], ...],
    "point_labels": [1, 1, ...]  # 1=foreground, 0=background
}

Input format (animate):
{
    "operation": "animate",
    "image_url": "https://...",
    "mask_url": "https://...",  # Optional: only animate masked region
    "motion_bucket_id": 127,     # Motion amount (1-255, default 127)
    "noise_aug_strength": 0.02,  # Noise augmentation (default 0.02)
    "num_frames": 25,            # Number of frames (default 25)
    "fps": 6                     # Output FPS (default 6)
}

Output format:
{
    "success": true,
    "output_url": "https://...",  # For animate: video URL
    "mask_urls": ["https://..."], # For segment: mask image URLs
    "processing_time_seconds": 12.5
}
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import runpod
import torch
from PIL import Image

# Model paths (baked into Docker image)
SAM2_CHECKPOINT = Path("/app/sam2/checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SVD model ID (cached in HF_HOME)
SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

# Lazy-loaded models (loaded on first use)
_sam2_predictor = None
_svd_pipeline = None


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
    return 16  # Default assumption


def download_file(url: str, output_path: str, description: str = "file") -> bool:
    """Download file from URL with progress logging."""
    try:
        log(f"Downloading {description} from {url[:80]}...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log(f"  Downloaded {description}: {Path(output_path).stat().st_size // 1024}KB")
        return True
    except Exception as e:
        log(f"Error downloading {description}: {e}")
        return False


def download_image(url: str) -> Optional[Image.Image]:
    """Download image from URL and return as PIL Image."""
    try:
        log(f"Downloading image from {url[:80]}...")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        log(f"  Image size: {image.size}")
        return image
    except Exception as e:
        log(f"Error downloading image: {e}")
        return None


def upload_to_r2(file_path: str, job_id: str, r2_config: dict, suffix: str = "") -> tuple[Optional[str], Optional[str]]:
    """Upload file to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config
        import uuid

        file_size = Path(file_path).stat().st_size
        log(f"Uploading to R2 ({file_size // 1024}KB)...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        ext = Path(file_path).suffix
        object_key = f"animate/results/{job_id}_{uuid.uuid4().hex[:8]}{suffix}{ext}"

        client.upload_file(file_path, r2_config["bucket_name"], object_key)

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


def upload_file(file_path: str, job_id: str, r2_config: Optional[dict] = None, suffix: str = "") -> dict:
    """Upload file and return upload info."""
    if r2_config:
        url, r2_key = upload_to_r2(file_path, job_id, r2_config, suffix)
        if url:
            return {"url": url, "r2_key": r2_key}
        log("R2 upload failed, falling back to RunPod storage")

    # Fall back to RunPod storage
    try:
        ext = Path(file_path).suffix
        result_url = runpod.serverless.utils.rp_upload.upload_file_to_bucket(
            file_name=f"{job_id}{suffix}{ext}",
            file_location=file_path
        )
        return {"url": result_url}
    except Exception as e:
        log(f"Error uploading file: {e}")
        return {}


def get_sam2_predictor():
    """Get or initialize SAM2 predictor (lazy loading)."""
    global _sam2_predictor
    if _sam2_predictor is None:
        log("Loading SAM2 model...")
        start = time.time()

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(
            SAM2_CONFIG,
            str(SAM2_CHECKPOINT),
            device="cuda"
        )
        _sam2_predictor = SAM2ImagePredictor(sam2_model)

        log(f"SAM2 loaded in {time.time() - start:.1f}s")
    return _sam2_predictor


def get_svd_pipeline():
    """Get or initialize SVD pipeline (lazy loading)."""
    global _svd_pipeline
    if _svd_pipeline is None:
        log("Loading SVD pipeline...")
        start = time.time()

        from diffusers import StableVideoDiffusionPipeline

        _svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            SVD_MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Memory optimizations
        vram_gb = get_gpu_vram_gb()
        if vram_gb < 24:
            log("Enabling CPU offload for low VRAM")
            _svd_pipeline.enable_model_cpu_offload()
        else:
            _svd_pipeline = _svd_pipeline.to("cuda")

        # Always enable forward chunking for memory efficiency
        _svd_pipeline.unet.enable_forward_chunking()

        log(f"SVD loaded in {time.time() - start:.1f}s")
    return _svd_pipeline


def handle_segment(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Segment image using SAM2.

    Required:
        image_url: URL to image

    One of:
        bboxes: List of [x1, y1, x2, y2] bounding boxes
        points: List of [x, y] points
        point_labels: Labels for points (1=foreground, 0=background)

    Optional:
        r2: R2 config for result upload
    """
    start_time = time.time()

    image_url = job_input.get("image_url")
    bboxes = job_input.get("bboxes")
    points = job_input.get("points")
    point_labels = job_input.get("point_labels")
    r2_config = job_input.get("r2")

    if not image_url:
        return {"error": "Missing required 'image_url'"}

    if not bboxes and not points:
        return {"error": "Either 'bboxes' or 'points' is required"}

    # Download image
    image = download_image(image_url)
    if image is None:
        return {"error": "Failed to download image"}

    image_np = np.array(image)

    # Get SAM2 predictor
    predictor = get_sam2_predictor()
    predictor.set_image(image_np)

    # Prepare prompts
    mask_results = []

    if bboxes:
        # Process each bbox
        for i, bbox in enumerate(bboxes):
            box_np = np.array(bbox)
            masks, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=True,
            )
            # Take best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            mask_results.append({
                "mask": mask,
                "score": float(scores[best_idx]),
                "bbox": bbox,
            })

    elif points:
        # Process points together
        points_np = np.array(points)
        labels_np = np.array(point_labels) if point_labels else np.ones(len(points))

        masks, scores, _ = predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        mask_results.append({
            "mask": masks[best_idx],
            "score": float(scores[best_idx]),
        })

    # Save masks and upload
    mask_urls = []
    for i, result in enumerate(mask_results):
        mask = result["mask"]
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))

        mask_path = str(work_dir / f"mask_{i:03d}.png")
        mask_img.save(mask_path)

        upload_result = upload_file(mask_path, job_id, r2_config, suffix=f"_mask_{i}")
        if upload_result.get("url"):
            mask_urls.append({
                "url": upload_result["url"],
                "score": result["score"],
                "bbox": result.get("bbox"),
            })

    elapsed = time.time() - start_time

    return {
        "success": True,
        "masks": mask_urls,
        "mask_count": len(mask_urls),
        "image_size": list(image.size),
        "processing_time_seconds": round(elapsed, 2),
    }


def handle_animate(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Animate image using Stable Video Diffusion.

    Required:
        image_url: URL to image (will be resized to 1024x576)

    Optional:
        mask_url: URL to mask image (white = animate, black = keep static)
        motion_bucket_id: Motion amount 1-255 (default 127)
        noise_aug_strength: Noise augmentation 0-1 (default 0.02)
        num_frames: Number of frames to generate (default 25)
        fps: Output video FPS (default 6)
        seed: Random seed for reproducibility
        r2: R2 config for result upload
    """
    start_time = time.time()

    image_url = job_input.get("image_url")
    mask_url = job_input.get("mask_url")
    motion_bucket_id = job_input.get("motion_bucket_id", 127)
    noise_aug_strength = job_input.get("noise_aug_strength", 0.02)
    num_frames = job_input.get("num_frames", 25)
    fps = job_input.get("fps", 6)
    seed = job_input.get("seed")
    r2_config = job_input.get("r2")

    if not image_url:
        return {"error": "Missing required 'image_url'"}

    # Download image
    image = download_image(image_url)
    if image is None:
        return {"error": "Failed to download image"}

    # SVD expects 1024x576 (or 576x1024 for portrait)
    original_size = image.size
    if image.width > image.height:
        target_size = (1024, 576)
    else:
        target_size = (576, 1024)

    image = image.resize(target_size, Image.Resampling.LANCZOS)
    log(f"Resized image from {original_size} to {target_size}")

    # Download mask if provided
    mask = None
    if mask_url:
        mask = download_image(mask_url)
        if mask:
            mask = mask.resize(target_size, Image.Resampling.LANCZOS)
            log("Mask loaded and resized")

    # Get pipeline
    pipe = get_svd_pipeline()

    # Set seed
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        log(f"Using seed: {seed}")

    # Generate video
    log(f"Generating {num_frames} frames (motion={motion_bucket_id}, noise={noise_aug_strength})...")
    gen_start = time.time()

    # Adjust decode_chunk_size based on VRAM
    vram_gb = get_gpu_vram_gb()
    decode_chunk_size = 8 if vram_gb >= 24 else 4 if vram_gb >= 16 else 2

    frames = pipe(
        image,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
    ).frames[0]

    log(f"Generated {len(frames)} frames in {time.time() - gen_start:.1f}s")

    # If mask provided, composite animated frames over original
    if mask is not None:
        log("Compositing with mask...")
        mask_np = np.array(mask.convert("L")) / 255.0
        original_np = np.array(image)

        composited_frames = []
        for frame in frames:
            frame_np = np.array(frame)
            # Blend: result = original * (1-mask) + animated * mask
            result = (original_np * (1 - mask_np[:, :, None]) +
                      frame_np * mask_np[:, :, None]).astype(np.uint8)
            composited_frames.append(Image.fromarray(result))
        frames = composited_frames

    # Export to video
    output_path = str(work_dir / "animated.mp4")

    from diffusers.utils import export_to_video
    export_to_video(frames, output_path, fps=fps)

    log(f"Exported video: {Path(output_path).stat().st_size // 1024}KB")

    # Upload result
    upload_result = upload_file(output_path, job_id, r2_config)

    if not upload_result.get("url"):
        return {"error": "Failed to upload result video"}

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "output_url": upload_result["url"],
        "original_size": list(original_size),
        "processed_size": list(target_size),
        "num_frames": len(frames),
        "fps": fps,
        "motion_bucket_id": motion_bucket_id,
        "processing_time_seconds": round(elapsed, 2),
    }

    if upload_result.get("r2_key"):
        result["r2_key"] = upload_result["r2_key"]

    return result


def handler(job: dict) -> dict:
    """
    Main RunPod handler - routes to specific operations.

    Supports:
        - segment: Generate masks using SAM2
        - animate: Generate video using SVD
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    operation = job_input.get("operation", "animate")
    log(f"Job {job_id}: operation={operation}")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "segment":
            return handle_segment(job_input, job_id, work_dir)
        elif operation == "animate":
            return handle_animate(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: segment, animate"}
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
    log("Starting RunPod animate handler...")
    log(f"SAM2 checkpoint exists: {SAM2_CHECKPOINT.exists()}")

    # Check CUDA
    if torch.cuda.is_available():
        log(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        log("WARNING: CUDA not available!")

    runpod.serverless.start({"handler": handler})
