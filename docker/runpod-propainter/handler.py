#!/usr/bin/env python3
"""
RunPod serverless handler for video toolkit GPU operations.

Currently supports:
- dewatermark: Remove watermarks using ProPainter AI inpainting

Extensible design for future GPU tools (upscaling, denoising, etc.).

Input format:
{
    "operation": "dewatermark",
    "video_url": "https://...",
    "region": "x,y,width,height",  # OR
    "mask_url": "https://..."      # Pre-made mask image
}

Output format:
{
    "success": true,
    "output_url": "https://...",
    "video_dimensions": "1920x1080",
    "gpu_vram_gb": 24,
    "processing_time_seconds": 120.5
}
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
import runpod

# ProPainter installation path (baked into Docker image)
PROPAINTER_PATH = Path("/app/propainter")

# Memory profiles based on GPU VRAM (GB)
# CONSERVATIVE settings to avoid OOM - ProPainter's RAFT optical flow is extremely memory hungry
# Key parameters:
#   - subvideo_length: frames per batch (lower = less memory, more batches)
#   - neighbor_length: local temporal context (lower = less memory)
#   - ref_stride: global reference sampling (higher = fewer refs = less memory)
#
# NOTE: These are deliberately conservative. ProPainter defaults are:
#   subvideo_length=80, neighbor_length=10, ref_stride=10
# But those OOM on long videos even with 80GB VRAM.
MEMORY_PROFILES = {
    8:  {"subvideo_length": 20, "neighbor_length": 3,  "ref_stride": 30},  # 8GB - minimal
    11: {"subvideo_length": 25, "neighbor_length": 4,  "ref_stride": 25},  # 12GB cards
    15: {"subvideo_length": 30, "neighbor_length": 5,  "ref_stride": 20},  # 16GB cards
    22: {"subvideo_length": 40, "neighbor_length": 5,  "ref_stride": 15},  # 24GB cards (3090, 4090)
    45: {"subvideo_length": 50, "neighbor_length": 5,  "ref_stride": 15},  # 48GB cards (A6000, A40)
    75: {"subvideo_length": 60, "neighbor_length": 5,  "ref_stride": 15},  # 80GB cards (A100, H100)
}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def get_gpu_vram_gb() -> int:
    """Detect GPU VRAM using PyTorch (respects CUDA_VISIBLE_DEVICES set by RunPod)."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get the current device (respects CUDA_VISIBLE_DEVICES)
            device_id = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_id)
            vram_bytes = props.total_memory
            vram_gb = vram_bytes // (1024 ** 3)
            log(f"Detected GPU: {props.name}, VRAM: {vram_gb}GB ({vram_bytes // (1024**2)}MB)")
            return vram_gb
        else:
            log("Warning: CUDA not available")
    except Exception as e:
        log(f"Warning: Could not detect GPU VRAM via torch: {e}")

    # Fallback to nvidia-smi if torch detection fails
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            vram_values = [int(line.strip()) for line in lines if line.strip()]
            if vram_values:
                max_vram_mb = max(vram_values)
                log(f"Fallback nvidia-smi detection: {max_vram_mb // 1024}GB")
                return max_vram_mb // 1024
    except Exception as e:
        log(f"Warning: nvidia-smi fallback also failed: {e}")

    return 16  # Default assumption


def get_memory_profile(vram_gb: int) -> dict:
    """Get optimal ProPainter settings based on available VRAM."""
    for threshold in sorted(MEMORY_PROFILES.keys(), reverse=True):
        if vram_gb >= threshold:
            return MEMORY_PROFILES[threshold].copy()
    return MEMORY_PROFILES[8].copy()


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
                if total_size > 0 and downloaded % (1024 * 1024) == 0:
                    pct = (downloaded / total_size) * 100
                    log(f"  Downloaded {downloaded // (1024*1024)}MB ({pct:.0f}%)")

        log(f"  Downloaded {description}: {Path(output_path).stat().st_size // (1024*1024)}MB")
        return True
    except Exception as e:
        log(f"Error downloading {description}: {e}")
        return False


def upload_to_r2(file_path: str, job_id: str, r2_config: dict) -> tuple[Optional[str], Optional[str]]:
    """Upload file to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config
        import uuid

        log(f"Uploading result to R2 ({Path(file_path).stat().st_size // (1024*1024)}MB)...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        object_key = f"dewatermark/results/{job_id}_{uuid.uuid4().hex[:8]}.mp4"

        client.upload_file(file_path, r2_config["bucket_name"], object_key)

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


def upload_file(file_path: str, job_id: str, r2_config: Optional[dict] = None) -> dict:
    """
    Upload file and return upload info.

    Returns dict with:
        - output_url: Presigned URL for download (always present if successful)
        - r2_key: R2 object key (only if R2 was used)
    """
    # Try R2 first if configured
    if r2_config:
        url, r2_key = upload_to_r2(file_path, job_id, r2_config)
        if url:
            return {"output_url": url, "r2_key": r2_key}
        log("R2 upload failed, falling back to RunPod storage")

    # Fall back to RunPod storage
    try:
        log(f"Uploading result to RunPod storage ({Path(file_path).stat().st_size // (1024*1024)}MB)...")

        result_url = runpod.serverless.utils.rp_upload.upload_file_to_bucket(
            file_name=f"{job_id}_dewatermarked.mp4",
            file_location=file_path
        )

        log(f"  Upload complete: {result_url[:80]}...")
        return {"output_url": result_url}
    except Exception as e:
        log(f"Error uploading file: {e}")
        return {}


def get_video_info(video_path: str) -> dict:
    """Get video dimensions, duration, fps, and frame count using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration,r_frame_rate,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})

            # Parse frame rate (can be "30/1" or "29.97")
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in str(fps_str):
                num, den = str(fps_str).split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str) if fps_str else 30.0

            duration = float(stream.get("duration") or fmt.get("duration", 0))

            # Calculate frame count
            nb_frames = stream.get("nb_frames")
            if nb_frames:
                frame_count = int(nb_frames)
            else:
                frame_count = int(duration * fps)

            return {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
            }
    except Exception as e:
        log(f"Warning: Could not get video info: {e}")
    return {"width": 0, "height": 0, "duration": 0, "fps": 30.0, "frame_count": 0}


# Memory estimation constants (empirically determined from ProPainter testing)
# ProPainter uses ~6.5MB per frame at 720p for RGB tensors, flow, masks, etc.
BYTES_PER_FRAME_720P = 6.5 * 1024 * 1024


def calculate_safe_resize_ratio(
    vram_gb: int,
    width: int,
    height: int,
    frame_count: int,
    requested_ratio: float = 1.0,
    safety_margin: float = 0.7,
) -> tuple[float, str]:
    """
    Calculate a safe resize_ratio based on available VRAM and video properties.

    Returns (resize_ratio, reason) tuple.

    The ratio is the MINIMUM of:
    - requested_ratio (what the user asked for)
    - calculated safe ratio (based on memory estimation)

    Args:
        vram_gb: Available GPU VRAM in GB
        width: Video width in pixels
        height: Video height in pixels
        frame_count: Number of frames in video
        requested_ratio: User's requested resize ratio (default 1.0 = full res)
        safety_margin: Use this fraction of VRAM (default 0.7 = 70%)

    Returns:
        (resize_ratio, reason): The ratio to use and why
    """
    # Calculate memory needed at full resolution
    pixels = width * height
    pixels_720p = 1280 * 720
    scale_factor = pixels / pixels_720p

    bytes_per_frame = BYTES_PER_FRAME_720P * scale_factor
    total_bytes_full_res = bytes_per_frame * frame_count

    # Add overhead for model weights, intermediate tensors, etc. (~2GB base)
    model_overhead_bytes = 2 * (1024 ** 3)
    total_needed_full_res = total_bytes_full_res + model_overhead_bytes

    # Available memory with safety margin
    available_bytes = vram_gb * (1024 ** 3) * safety_margin

    # If full resolution fits, use requested ratio
    if total_needed_full_res <= available_bytes:
        log(f"Memory estimate: {total_needed_full_res / (1024**3):.1f}GB needed, {available_bytes / (1024**3):.1f}GB available - full resolution OK")
        return (requested_ratio, "full_resolution_fits")

    # Calculate the resize ratio needed to fit in memory
    # Memory scales with resize_ratio^2 (both width and height reduced)
    # So: needed_memory * ratio^2 + overhead = available
    # ratio^2 = (available - overhead) / frame_memory
    # ratio = sqrt((available - overhead) / frame_memory)

    frame_memory = total_bytes_full_res - model_overhead_bytes
    if frame_memory <= 0:
        return (0.5, "fallback_conservative")

    usable_for_frames = available_bytes - model_overhead_bytes
    if usable_for_frames <= 0:
        return (0.25, "very_low_vram")

    import math
    safe_ratio = math.sqrt(usable_for_frames / frame_memory)

    # Clamp to reasonable range [0.25, 1.0]
    safe_ratio = max(0.25, min(1.0, safe_ratio))

    # Use the more conservative of user request and calculated safe ratio
    final_ratio = min(requested_ratio, safe_ratio)

    # Round to nice values for consistency
    nice_ratios = [1.0, 0.75, 0.5, 0.375, 0.25]
    for nice in nice_ratios:
        if final_ratio >= nice:
            final_ratio = nice
            break

    log(f"Memory estimate: {total_needed_full_res / (1024**3):.1f}GB needed at full res, {available_bytes / (1024**3):.1f}GB available")
    log(f"Calculated safe ratio: {safe_ratio:.2f}, using: {final_ratio}")

    reason = "auto_calculated" if final_ratio < requested_ratio else "user_requested"
    return (final_ratio, reason)


def create_mask_from_region(region: str, width: int, height: int, output_path: str) -> bool:
    """Create white-on-black mask image from x,y,w,h region string."""
    try:
        parts = [v.strip() for v in region.split(",")]
        if len(parts) != 4:
            log(f"Error: Region must be x,y,width,height - got: {region}")
            return False

        x, y, w, h = [int(p) for p in parts]

        # Validate bounds
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            log(f"Error: Invalid region values: {region}")
            return False
        if x + w > width or y + h > height:
            log(f"Error: Region {region} exceeds video dimensions {width}x{height}")
            return False

        log(f"Creating mask: {w}x{h} region at ({x},{y}) on {width}x{height} canvas")

        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=black:s={width}x{height}:d=1",
            "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:c=white:t=fill",
            "-frames:v", "1",
            output_path,
        ], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            log(f"FFmpeg error: {result.stderr}")
            return False

        return Path(output_path).exists()
    except Exception as e:
        log(f"Error creating mask: {e}")
        return False


def run_propainter(
    video_path: str,
    mask_path: str,
    output_dir: str,
    profile: dict,
    fp16: bool = True,
    resize_ratio: float = 1.0
) -> Optional[str]:
    """Run ProPainter inference and return path to output video."""

    inference_script = PROPAINTER_PATH / "inference_propainter.py"

    cmd = [
        "python3", str(inference_script),
        "-i", video_path,
        "-m", mask_path,
        "-o", output_dir,
        "--neighbor_length", str(profile["neighbor_length"]),
        "--ref_stride", str(profile["ref_stride"]),
        "--subvideo_length", str(profile["subvideo_length"]),
    ]

    if fp16:
        cmd.append("--fp16")

    if resize_ratio != 1.0:
        cmd.extend(["--resize_ratio", str(resize_ratio)])

    log(f"Running ProPainter with settings: {profile}, resize_ratio={resize_ratio}")
    log(f"Command: {' '.join(cmd)}")

    start_time = time.time()

    result = subprocess.run(
        cmd,
        cwd=PROPAINTER_PATH,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hour max
    )

    elapsed = time.time() - start_time
    log(f"ProPainter completed in {elapsed:.1f}s")

    if result.returncode != 0:
        log(f"ProPainter error (exit {result.returncode}):")
        log("=== STDOUT (last 3000 chars) ===")
        log(result.stdout[-3000:] if result.stdout else "No stdout")
        log("=== STDERR (last 3000 chars) ===")
        log(result.stderr[-3000:] if result.stderr else "No stderr")
        return None

    # Log success output for debugging
    log("=== ProPainter completed successfully ===")
    if result.stdout:
        log(f"STDOUT (last 1000 chars): {result.stdout[-1000:]}")

    # Find output file - ProPainter creates: output_dir/video_name/inpaint_out.mp4
    video_stem = Path(video_path).stem
    expected = Path(output_dir) / video_stem / "inpaint_out.mp4"

    if expected.exists():
        log(f"Output found: {expected}")
        return str(expected)

    # Fallback: search for inpaint_out.mp4 anywhere in results
    for mp4 in Path(output_dir).rglob("inpaint_out.mp4"):
        log(f"Output found (search): {mp4}")
        return str(mp4)

    # Last resort: any mp4 that's NOT masked_in.mp4
    for mp4 in Path(output_dir).rglob("*.mp4"):
        if "masked" not in mp4.name:
            log(f"Output found (fallback): {mp4}")
            return str(mp4)

    log(f"Error: No output file found in {output_dir}")
    return None


def handle_dewatermark(job_input: dict, job_id: str, work_dir: Path) -> dict:
    """
    Handle dewatermark operation using ProPainter.

    Required inputs:
        video_url: URL to video file
        region: "x,y,width,height" OR mask_url: URL to mask image

    Optional inputs:
        fp16: Use half precision (default: true, faster)
        resize_ratio: Scale factor for processing (default: "auto" - calculated based on VRAM)
                      Set to a specific value (0.25-1.0) to override auto-calculation
        r2: R2 config for result upload (endpoint_url, access_key_id, secret_access_key, bucket_name)
    """
    start_time = time.time()

    # Validate inputs
    video_url = job_input.get("video_url")
    region = job_input.get("region")
    mask_url = job_input.get("mask_url")
    fp16 = job_input.get("fp16", True)
    requested_resize_ratio = job_input.get("resize_ratio", "auto")  # Default to auto-calculation
    r2_config = job_input.get("r2")  # Optional R2 config for result upload

    if not video_url:
        return {"error": "Missing required 'video_url' in input"}
    if not region and not mask_url:
        return {"error": "Either 'region' (x,y,w,h) or 'mask_url' is required"}

    if r2_config:
        log("R2 config provided - will upload result to R2")
    log(f"Processing options: fp16={fp16}, requested_resize_ratio={requested_resize_ratio}")

    # Download video
    video_path = str(work_dir / "input_video.mp4")
    if not download_file(video_url, video_path, "video"):
        return {"error": "Failed to download video from URL"}

    # Get video info
    video_info = get_video_info(video_path)
    width, height = video_info["width"], video_info["height"]
    duration = video_info["duration"]
    frame_count = video_info["frame_count"]

    if not width or not height:
        return {"error": "Could not read video dimensions"}

    log(f"Video: {width}x{height}, {duration:.1f}s, {frame_count} frames")

    # Detect GPU and get optimal settings
    vram_gb = get_gpu_vram_gb()
    profile = get_memory_profile(vram_gb)
    log(f"GPU VRAM: {vram_gb}GB, using profile: {profile}")

    # Calculate safe resize_ratio based on VRAM and video properties
    if requested_resize_ratio == "auto":
        # Auto mode: calculate optimal ratio, aim for full resolution if possible
        resize_ratio, resize_reason = calculate_safe_resize_ratio(
            vram_gb, width, height, frame_count, requested_ratio=1.0
        )
    else:
        # User specified a ratio - use it but warn if it might OOM
        user_ratio = float(requested_resize_ratio)
        safe_ratio, _ = calculate_safe_resize_ratio(
            vram_gb, width, height, frame_count, requested_ratio=user_ratio
        )
        if safe_ratio < user_ratio:
            log(f"WARNING: Requested resize_ratio={user_ratio} may cause OOM. Safe ratio is {safe_ratio}")
        resize_ratio = user_ratio
        resize_reason = "user_specified"

    log(f"Using resize_ratio={resize_ratio} ({resize_reason})")

    # Prepare mask
    mask_path = str(work_dir / "mask.png")

    if mask_url:
        if not download_file(mask_url, mask_path, "mask"):
            return {"error": "Failed to download mask from URL"}
    else:
        if not create_mask_from_region(region, width, height, mask_path):
            return {"error": f"Failed to create mask from region: {region}"}

    # Run ProPainter
    output_dir = str(work_dir / "results")
    os.makedirs(output_dir, exist_ok=True)

    result_path = run_propainter(video_path, mask_path, output_dir, profile, fp16, resize_ratio)

    if not result_path:
        return {"error": "ProPainter processing failed - check logs for details"}

    # Upload result (to R2 if configured, otherwise RunPod storage)
    upload_result = upload_file(result_path, job_id, r2_config)

    if not upload_result.get("output_url"):
        return {"error": "Failed to upload result video"}

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "output_url": upload_result["output_url"],
        "video_dimensions": f"{width}x{height}",
        "video_duration_seconds": round(duration, 2),
        "video_frame_count": frame_count,
        "gpu_vram_gb": vram_gb,
        "profile_used": profile,
        "resize_ratio": resize_ratio,
        "resize_reason": resize_reason,
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
        - dewatermark: Remove watermarks using ProPainter
        - (future: upscale, denoise, etc.)
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    operation = job_input.get("operation", "dewatermark")
    log(f"Job {job_id}: operation={operation}")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"runpod_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if operation == "dewatermark":
            return handle_dewatermark(job_input, job_id, work_dir)
        else:
            return {"error": f"Unknown operation: {operation}. Supported: dewatermark"}
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
    log("Starting RunPod ProPainter handler...")
    log(f"ProPainter path: {PROPAINTER_PATH}")
    log(f"Weights exist: {(PROPAINTER_PATH / 'weights').exists()}")

    runpod.serverless.start({"handler": handler})
