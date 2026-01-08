#!/usr/bin/env python3
"""
RunPod serverless handler for SadTalker.
Generates talking head videos from image + audio.

Input format:
{
    "input": {
        # Required - one of:
        "image_url": str,           # URL to download image
        "image_base64": str,        # Base64 encoded image

        # Required - one of:
        "audio_url": str,           # URL to download audio
        "audio_base64": str,        # Base64 encoded audio

        # Options
        "still_mode": bool,         # Less head movement (default: false)
        "enhancer": str,            # "gfpgan" (default) or "none"
        "preprocess": str,          # "crop", "resize", "full" (default: "crop")
        "size": int,                # Output resolution: 256 or 512 (default: 256)
        "expression_scale": float,  # Expression intensity (default: 1.0)
        "pose_style": int,          # Pose variation (0-45, default: 0)

        # R2 config for result upload
        "r2": {
            "endpoint_url": str,
            "access_key_id": str,
            "secret_access_key": str,
            "bucket_name": str
        }
    }
}

Output format:
{
    "success": true,
    "video_url": str,           # Presigned R2 URL (if r2 config provided)
    "video_base64": str,        # Base64 encoded video (if no R2)
    "r2_key": str,              # R2 object key
    "duration_seconds": float,
    "chunks_processed": int,
    "processing_time_seconds": float
}

Chunking:
- Audio >45s is split into chunks to prevent drift
- Each chunk processed independently
- Results concatenated with ffmpeg
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import runpod
import requests

# SadTalker paths
SADTALKER_DIR = Path("/app/SadTalker")
CHECKPOINT_DIR = SADTALKER_DIR / "checkpoints"

# Chunk size in seconds (to prevent drift)
CHUNK_DURATION = 45


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    """Download file from URL to local path."""
    try:
        log(f"Downloading from {url[:80]}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log(f"  Downloaded: {output_path.name} ({output_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        log(f"Download error: {e}")
        return False


def decode_base64_file(data: str, output_path: Path) -> bool:
    """Decode base64 data and write to file."""
    try:
        # Handle data URI prefix if present
        if "," in data:
            data = data.split(",", 1)[1]

        decoded = base64.b64decode(data)
        output_path.write_bytes(decoded)
        log(f"Decoded base64 to {output_path.name} ({len(decoded) // 1024}KB)")
        return True
    except Exception as e:
        log(f"Base64 decode error: {e}")
        return False


def encode_file_base64(file_path: Path) -> str:
    """Encode file to base64 string."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        log(f"Error getting audio duration: {e}")
        return 0.0


def split_audio_chunks(audio_path: Path, work_dir: Path, chunk_duration: int = CHUNK_DURATION) -> list[Path]:
    """Split audio into chunks to prevent drift."""
    duration = get_audio_duration(audio_path)
    log(f"Audio duration: {duration:.1f}s")

    if duration <= chunk_duration:
        return [audio_path]

    chunks = []
    start = 0.0
    chunk_idx = 0

    while start < duration:
        chunk_path = work_dir / f"audio_chunk_{chunk_idx:03d}.wav"
        end = min(start + chunk_duration, duration)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(end - start),
                "-c:a", "pcm_s16le",
                str(chunk_path),
            ],
            capture_output=True,
            timeout=60,
        )

        if chunk_path.exists():
            chunks.append(chunk_path)
            log(f"  Chunk {chunk_idx}: {start:.1f}s - {end:.1f}s")

        start = end
        chunk_idx += 1

    log(f"Split into {len(chunks)} chunks")
    return chunks


# Global to store last error for better reporting
_last_sadtalker_error = None


def run_sadtalker(
    image_path: Path,
    audio_path: Path,
    output_dir: Path,
    still_mode: bool = False,
    enhancer: str = "gfpgan",
    preprocess: str = "crop",
    size: int = 256,
    expression_scale: float = 1.0,
    pose_style: int = 0,
) -> Optional[Path]:
    """Run SadTalker inference on a single image/audio pair."""
    global _last_sadtalker_error
    _last_sadtalker_error = None

    # Verify checkpoints exist
    if not CHECKPOINT_DIR.exists():
        _last_sadtalker_error = f"Checkpoint dir not found: {CHECKPOINT_DIR}"
        log(_last_sadtalker_error)
        return None

    bfm_dir = CHECKPOINT_DIR / "BFM_Fitting"
    if not bfm_dir.exists():
        _last_sadtalker_error = f"BFM_Fitting dir not found: {bfm_dir}"
        log(_last_sadtalker_error)
        return None

    cmd = [
        "python", str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", str(audio_path),
        "--source_image", str(image_path),
        "--result_dir", str(output_dir),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--size", str(size),
        "--expression_scale", str(expression_scale),
        "--pose_style", str(pose_style),
        "--preprocess", preprocess,
    ]

    if still_mode:
        cmd.append("--still")

    if enhancer != "none":
        cmd.extend(["--enhancer", enhancer])

    log(f"Running SadTalker: {' '.join(cmd[-6:])}")
    log(f"  Image: {image_path} exists={image_path.exists()}")
    log(f"  Audio: {audio_path} exists={audio_path.exists()}")
    log(f"  Checkpoints: {CHECKPOINT_DIR} contents={list(CHECKPOINT_DIR.iterdir())[:5]}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SADTALKER_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per chunk
        )

        # Log stdout/stderr regardless of return code
        if result.stdout:
            log(f"SadTalker stdout: {result.stdout[-500:]}")
        if result.stderr:
            log(f"SadTalker stderr: {result.stderr[-1000:]}")

        if result.returncode != 0:
            _last_sadtalker_error = f"Exit code {result.returncode}: {result.stderr[-500:]}"
            log(f"SadTalker failed: {_last_sadtalker_error}")
            return None

        # Debug: List all files in output directory
        log(f"Output dir contents: {list(output_dir.iterdir())}")
        for item in output_dir.iterdir():
            log(f"  Item: {item.name} (is_dir={item.is_dir()}, is_file={item.is_file()})")
            if item.is_dir():
                log(f"    Subdir contents: {list(item.iterdir())}")

        # Find output video - check both direct files and subdirectories
        # SadTalker outputs like: output_000/2026_01_07_00.17.22.mp4 (direct)
        # or: output_000/2026_01_07_00.17.22/input_image##input_audio_enhanced.mp4 (subdir)

        # First check for mp4 directly in output_dir
        for f in output_dir.glob("*.mp4"):
            log(f"Found video (direct): {f}")
            return f

        # Then check subdirectories
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.mp4"):
                    log(f"Found video (subdir): {f}")
                    return f

        _last_sadtalker_error = f"No output video found. All files: {list(output_dir.rglob('*'))}"
        log(_last_sadtalker_error)
        return None

    except subprocess.TimeoutExpired:
        _last_sadtalker_error = "SadTalker timed out after 600s"
        log(_last_sadtalker_error)
        return None
    except Exception as e:
        _last_sadtalker_error = f"Exception: {e}"
        log(f"SadTalker exception: {e}")
        return None


def concatenate_videos(video_paths: list[Path], output_path: Path) -> bool:
    """Concatenate multiple videos using ffmpeg."""
    if len(video_paths) == 1:
        shutil.copy(video_paths[0], output_path)
        return True

    # Create concat file
    concat_file = output_path.parent / "concat.txt"
    with open(concat_file, "w") as f:
        for vp in video_paths:
            f.write(f"file '{vp}'\n")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return output_path.exists()
    except Exception as e:
        log(f"Concatenation error: {e}")
        return False


def upload_to_r2(file_path: Path, job_id: str, r2_config: dict) -> tuple[Optional[str], Optional[str]]:
    """Upload video to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config

        log("Uploading to R2...")

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )

        object_key = f"sadtalker/results/{job_id}_{uuid.uuid4().hex[:8]}.mp4"

        client.upload_file(
            str(file_path),
            r2_config["bucket_name"],
            object_key,
            ExtraArgs={"ContentType": "video/mp4"},
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


def handler(job: dict) -> dict:
    """Main RunPod handler for SadTalker."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    log(f"Job {job_id}: Starting SadTalker")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"sadtalker_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        # Get image
        image_path = work_dir / "input_image.png"
        if job_input.get("image_url"):
            if not download_file(job_input["image_url"], image_path):
                return {"error": "Failed to download image"}
        elif job_input.get("image_base64"):
            if not decode_base64_file(job_input["image_base64"], image_path):
                return {"error": "Failed to decode image"}
        else:
            return {"error": "Missing image_url or image_base64"}

        # Get audio
        audio_path = work_dir / "input_audio.wav"
        if job_input.get("audio_url"):
            if not download_file(job_input["audio_url"], audio_path):
                return {"error": "Failed to download audio"}
        elif job_input.get("audio_base64"):
            if not decode_base64_file(job_input["audio_base64"], audio_path):
                return {"error": "Failed to decode audio"}
        else:
            return {"error": "Missing audio_url or audio_base64"}

        # Options
        still_mode = job_input.get("still_mode", False)
        enhancer = job_input.get("enhancer", "gfpgan")
        preprocess = job_input.get("preprocess", "crop")
        size = job_input.get("size", 256)
        expression_scale = job_input.get("expression_scale", 1.0)
        pose_style = job_input.get("pose_style", 0)
        r2_config = job_input.get("r2")

        # Split audio into chunks if needed
        audio_chunks = split_audio_chunks(audio_path, work_dir)
        total_duration = get_audio_duration(audio_path)

        # Process each chunk
        video_chunks = []
        for i, chunk_path in enumerate(audio_chunks):
            log(f"Processing chunk {i + 1}/{len(audio_chunks)}...")
            chunk_output_dir = work_dir / f"output_{i:03d}"
            chunk_output_dir.mkdir()

            video_path = run_sadtalker(
                image_path=image_path,
                audio_path=chunk_path,
                output_dir=chunk_output_dir,
                still_mode=still_mode,
                enhancer=enhancer,
                preprocess=preprocess,
                size=size,
                expression_scale=expression_scale,
                pose_style=pose_style,
            )

            if video_path:
                video_chunks.append(video_path)
                log(f"  Chunk {i + 1} complete: {video_path.name}")
            else:
                error_detail = _last_sadtalker_error or "Unknown error"
                return {"error": f"Failed to process chunk {i + 1}: {error_detail}"}

        # Concatenate chunks
        final_video = work_dir / "final_output.mp4"
        if not concatenate_videos(video_chunks, final_video):
            return {"error": "Failed to concatenate video chunks"}

        log(f"Final video: {final_video} ({final_video.stat().st_size // 1024}KB)")

        elapsed = time.time() - start_time

        result = {
            "success": True,
            "duration_seconds": total_duration,
            "chunks_processed": len(audio_chunks),
            "processing_time_seconds": round(elapsed, 2),
        }

        # Upload to R2 if configured
        if r2_config:
            url, r2_key = upload_to_r2(final_video, job_id, r2_config)
            if url:
                result["video_url"] = url
                result["r2_key"] = r2_key
            else:
                return {"error": "Failed to upload to R2"}
        else:
            # Return video as base64 (warning: large!)
            result["video_base64"] = encode_file_base64(final_video)
            log("Warning: Returning video as base64 (consider using R2 for large files)")

        return result

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
    log("Starting RunPod SadTalker handler...")

    # Verify SadTalker is available
    if not SADTALKER_DIR.exists():
        log(f"ERROR: SadTalker not found at {SADTALKER_DIR}")
        sys.exit(1)

    if not CHECKPOINT_DIR.exists():
        log(f"ERROR: Checkpoints not found at {CHECKPOINT_DIR}")
        sys.exit(1)

    log(f"SadTalker directory: {SADTALKER_DIR}")
    log(f"Checkpoints: {list(CHECKPOINT_DIR.glob('*.pth*'))[:3]}...")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log("WARNING: CUDA not available!")
    except ImportError:
        log("Warning: torch not imported for CUDA check")

    runpod.serverless.start({"handler": handler})
