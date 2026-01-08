#!/usr/bin/env python3
"""
RunPod serverless handler for LTX-2 video generation.
Generates AI video from text prompts or images using LTX-2 19B model.

Input format:
{
    "input": {
        # Required
        "prompt": str,              # Text prompt describing the video

        # Optional - for image-to-video
        "image_url": str,           # URL to source image
        "image_base64": str,        # Base64 encoded source image

        # Generation options
        "width": int,               # Video width (default: 768, must be divisible by 32)
        "height": int,              # Video height (default: 512, must be divisible by 32)
        "num_frames": int,          # Number of frames (default: 121, ~5s at 24fps)
        "fps": int,                 # Frame rate (default: 24)
        "num_inference_steps": int, # Denoising steps (default: 50, use 25 for fast)
        "guidance_scale": float,    # CFG scale (default: 3.0)
        "seed": int,                # Random seed (optional)
        "negative_prompt": str,     # Negative prompt (default: standard quality tags)
        "flow": str,                # "pro" or "fast" (default: "pro")

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
    "r2_key": str,              # R2 object key for cleanup
    "width": int,
    "height": int,
    "num_frames": int,
    "fps": int,
    "duration_seconds": float,
    "processing_time_seconds": float,
    "seed": int
}
"""

import base64
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import runpod
import requests
import torch


# Model configuration
MODEL_ID = os.environ.get("LTX2_MODEL_ID", "Lightricks/LTX-2")
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/.cache/huggingface")

# Default negative prompt for quality
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, low resolution, watermark, text"

# Global pipeline cache
_pipeline = None
_pipeline_config = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def get_gpu_info() -> dict:
    """Get GPU information."""
    info = {"available": False}
    try:
        if torch.cuda.is_available():
            info["available"] = True
            info["name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gb"] = props.total_memory // (1024 ** 3)
            info["compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:
        log(f"GPU info error: {e}")
    return info


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


def load_pipeline(use_fp8: bool = True):
    """Load LTX-2 pipeline with caching."""
    global _pipeline, _pipeline_config

    config = {"use_fp8": use_fp8}

    # Return cached pipeline if config matches
    if _pipeline is not None and _pipeline_config == config:
        log("Using cached pipeline")
        return _pipeline

    log(f"Loading LTX-2 pipeline (FP8={use_fp8})...")
    start_time = time.time()

    try:
        # Try diffusers first (if LTX-2 support is available)
        try:
            from diffusers import LTXPipeline, LTXImageToVideoPipeline
            from diffusers.utils import export_to_video

            log("Loading via diffusers...")
            pipe = LTXPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
            )
            pipe.to("cuda")

            _pipeline = {
                "type": "diffusers",
                "t2v": pipe,
                "i2v": None,  # Will load on demand
                "export_to_video": export_to_video,
            }
            _pipeline_config = config
            log(f"Pipeline loaded via diffusers in {time.time() - start_time:.1f}s")
            return _pipeline

        except Exception as e:
            log(f"Diffusers loading failed: {e}, trying ltx_pipelines...")

        # Fall back to official ltx_pipelines
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from huggingface_hub import hf_hub_download, snapshot_download

        # Download model files
        log("Downloading model from HuggingFace...")
        model_dir = Path(HF_HOME) / "hub" / "models--Lightricks--LTX-2"

        # Download the FP8 checkpoint
        checkpoint_name = "ltx-2-19b-dev-fp8.safetensors" if use_fp8 else "ltx-2-19b-dev.safetensors"
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=checkpoint_name,
            cache_dir=HF_HOME,
        )
        log(f"  Checkpoint: {checkpoint_path}")

        # Download spatial upsampler
        upsampler_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            cache_dir=HF_HOME,
        )
        log(f"  Upsampler: {upsampler_path}")

        # Download distilled LoRA
        lora_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="ltx-2-19b-distilled-lora-384.safetensors",
            cache_dir=HF_HOME,
        )
        log(f"  LoRA: {lora_path}")

        # Initialize pipeline
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

        distilled_lora = [
            LoraPathStrengthAndSDOps(lora_path, 0.6, LTXV_LORA_COMFY_RENAMING_MAP),
        ]

        pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=upsampler_path,
        )

        _pipeline = {
            "type": "ltx_pipelines",
            "pipe": pipe,
        }
        _pipeline_config = config
        log(f"Pipeline loaded via ltx_pipelines in {time.time() - start_time:.1f}s")
        return _pipeline

    except Exception as e:
        log(f"Pipeline loading error: {e}")
        import traceback
        log(traceback.format_exc())
        raise


def generate_video(
    pipeline: dict,
    prompt: str,
    output_path: Path,
    image_path: Optional[Path] = None,
    width: int = 768,
    height: int = 512,
    num_frames: int = 121,
    fps: int = 24,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    seed: Optional[int] = None,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict:
    """Generate video using the loaded pipeline."""
    log(f"Generating video: {width}x{height}, {num_frames} frames, {num_inference_steps} steps")

    # Set seed
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    log(f"  Seed: {seed}")

    if pipeline["type"] == "diffusers":
        # Use diffusers pipeline
        pipe = pipeline["t2v"]
        export_to_video = pipeline["export_to_video"]

        # Load image for I2V if provided
        if image_path:
            from diffusers.utils import load_image
            image = load_image(str(image_path))

            # Switch to I2V pipeline if needed
            if pipeline["i2v"] is None:
                from diffusers import LTXImageToVideoPipeline
                log("Loading I2V pipeline...")
                pipeline["i2v"] = LTXImageToVideoPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                ).to("cuda")

            result = pipeline["i2v"](
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        # Export video
        export_to_video(result.frames[0], str(output_path), fps=fps)

    elif pipeline["type"] == "ltx_pipelines":
        # Use official ltx_pipelines
        pipe = pipeline["pipe"]

        kwargs = {
            "prompt": prompt,
            "output_path": str(output_path),
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": float(fps),
            "num_inference_steps": num_inference_steps,
            "cfg_guidance_scale": guidance_scale,
        }

        # Add image for I2V
        if image_path:
            kwargs["images"] = [(str(image_path), 0, 1.0)]  # Image at frame 0, strength 1.0

        pipe(**kwargs)

    return {
        "seed": seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "duration_seconds": num_frames / fps,
    }


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

        object_key = f"ltx2/results/{job_id}_{uuid.uuid4().hex[:8]}.mp4"

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
    """Main RunPod handler for LTX-2."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    log(f"Job {job_id}: Starting LTX-2 video generation")

    # Create temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"ltx2_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        # Validate required fields
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        # Get optional image for I2V
        image_path = None
        if job_input.get("image_url"):
            image_path = work_dir / "input_image.png"
            if not download_file(job_input["image_url"], image_path):
                return {"error": "Failed to download image"}
        elif job_input.get("image_base64"):
            image_path = work_dir / "input_image.png"
            if not decode_base64_file(job_input["image_base64"], image_path):
                return {"error": "Failed to decode image"}

        # Parse options
        flow = job_input.get("flow", "pro")
        width = job_input.get("width", 768)
        height = job_input.get("height", 512)
        num_frames = job_input.get("num_frames", 121)
        fps = job_input.get("fps", 24)
        seed = job_input.get("seed")
        guidance_scale = job_input.get("guidance_scale", 3.0)
        negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        r2_config = job_input.get("r2")

        # Adjust steps based on flow mode
        default_steps = 25 if flow == "fast" else 50
        num_inference_steps = job_input.get("num_inference_steps", default_steps)

        # Validate dimensions (must be divisible by 32)
        if width % 32 != 0:
            return {"error": f"width must be divisible by 32, got {width}"}
        if height % 32 != 0:
            return {"error": f"height must be divisible by 32, got {height}"}

        # Load pipeline
        pipeline = load_pipeline(use_fp8=True)

        # Generate video
        output_path = work_dir / "output.mp4"
        gen_result = generate_video(
            pipeline=pipeline,
            prompt=prompt,
            output_path=output_path,
            image_path=image_path,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )

        if not output_path.exists():
            return {"error": "Video generation failed - no output file"}

        elapsed = time.time() - start_time
        log(f"Video generated: {output_path} ({output_path.stat().st_size // 1024}KB) in {elapsed:.1f}s")

        result = {
            "success": True,
            "processing_time_seconds": round(elapsed, 2),
            **gen_result,
        }

        # Upload to R2 if configured
        if r2_config:
            url, r2_key = upload_to_r2(output_path, job_id, r2_config)
            if url:
                result["video_url"] = url
                result["r2_key"] = r2_key
            else:
                return {"error": "Failed to upload to R2"}
        else:
            # Return video as base64 (warning: can be large!)
            result["video_base64"] = encode_file_base64(output_path)
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
    log("Starting RunPod LTX-2 handler...")

    # Log GPU info
    gpu_info = get_gpu_info()
    if gpu_info["available"]:
        log(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']}GB VRAM)")
    else:
        log("WARNING: No CUDA GPU detected!")

    # Log cache locations
    log(f"HF_HOME: {HF_HOME}")
    log(f"Model ID: {MODEL_ID}")

    # Verify cache directory exists (for network volume)
    cache_dir = Path(HF_HOME)
    if not cache_dir.exists():
        log(f"Creating cache directory: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)

    runpod.serverless.start({"handler": handler})
