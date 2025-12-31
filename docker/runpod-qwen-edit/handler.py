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

# Add LightX2V to path (pip install doesn't work correctly)
sys.path.insert(0, "/app/lightx2v")

import numpy as np
import runpod
import torch
from PIL import Image

# Model paths - set after runtime download
MODEL_PATH = None
FP8_WEIGHTS_PATH = None

# Lazy-loaded pipeline
_pipeline = None
_pipeline_config = {}


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


def ensure_models() -> None:
    """Ensure models are downloaded before first use."""
    global MODEL_PATH, FP8_WEIGHTS_PATH

    if MODEL_PATH is not None and MODEL_PATH.exists():
        return  # Already initialized

    log("Checking/downloading models...")
    from download_models import ensure_models_downloaded

    paths = ensure_models_downloaded()
    MODEL_PATH = paths["model_path"]
    FP8_WEIGHTS_PATH = paths["fp8_path"]

    log(f"Models ready: {MODEL_PATH}")


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
    """Get or initialize LightX2V pipeline (lazy loading)."""
    global _pipeline, _pipeline_config

    # Ensure models are downloaded first
    ensure_models()

    # Check if we need to reinitialize (different config)
    current_config = {"use_fp8": use_fp8}
    if _pipeline is not None and _pipeline_config == current_config:
        return _pipeline

    # Need to reinitialize
    if _pipeline is not None:
        log("Pipeline config changed, reinitializing...")
        del _pipeline
        torch.cuda.empty_cache()

    log(f"Loading LightX2V pipeline (use_fp8={use_fp8})...")
    start = time.time()

    from lightx2v import LightX2VPipeline

    # Initialize pipeline
    _pipeline = LightX2VPipeline(
        model_path=str(MODEL_PATH),
        model_cls="qwen-image-edit-2511",
        task="i2i",
    )

    # Explicitly set all component paths that LightX2V needs
    model_path_str = str(MODEL_PATH)
    _pipeline.text_encoder_path = f"{model_path_str}/text_encoder"
    _pipeline.tokenizer_path = f"{model_path_str}/tokenizer"
    _pipeline.vae_path = f"{model_path_str}/vae"
    _pipeline.transformer_path = f"{model_path_str}/transformer"
    _pipeline.processor_path = f"{model_path_str}/processor"

    # Also set on config if it exists
    if hasattr(_pipeline, 'config') and _pipeline.config:
        _pipeline.config.text_encoder_path = _pipeline.text_encoder_path
        _pipeline.config.tokenizer_path = _pipeline.tokenizer_path
        _pipeline.config.vae_path = _pipeline.vae_path
        _pipeline.config.model_path = model_path_str

    log(f"Set paths: model={model_path_str}, text_encoder={_pipeline.text_encoder_path}")

    if use_fp8:
        # Enable FP8 quantization for lower VRAM usage
        fp8_ckpt = FP8_WEIGHTS_PATH / "Qwen-quant" / "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors"
        if not fp8_ckpt.exists():
            # Try alternate path structures
            for pattern in FP8_WEIGHTS_PATH.rglob("*fp8*.safetensors"):
                fp8_ckpt = pattern
                break

        if fp8_ckpt.exists():
            log(f"Enabling FP8 quantization from {fp8_ckpt}")
            _pipeline.enable_quantize(
                dit_quantized=True,
                dit_quantized_ckpt=str(fp8_ckpt),
                quant_scheme="fp8-sgl"
            )
        else:
            log(f"Warning: FP8 weights not found, falling back to BF16")

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

    # Select attention mode based on available libraries
    # LightX2V supports: flash_attn3, flash_attn2, sage_attn2, torch_sdpa
    attn_mode = None
    try:
        import flash_attn
        attn_mode = "flash_attn3"
        log("Using flash_attn3 for attention")
    except ImportError:
        pass

    if attn_mode is None:
        try:
            import sageattention
            attn_mode = "sage_attn2"
            log("Using sage_attn2 for attention")
        except ImportError:
            pass

    if attn_mode is None:
        # Fallback to PyTorch native SDPA
        attn_mode = "torch_sdpa"
        log("Using torch_sdpa (PyTorch native) for attention")

    log(f"Attention mode: {attn_mode}")

    # Create runtime config with explicit paths
    import json
    config_path = work_dir / "lightx2v_config.json"
    runtime_config = {
        "model_path": str(MODEL_PATH),
        "text_encoder_path": str(MODEL_PATH / "text_encoder"),
        "tokenizer_path": str(MODEL_PATH / "tokenizer"),
        "vae_path": str(MODEL_PATH / "vae"),
        "transformer_path": str(MODEL_PATH / "transformer"),
        "processor_path": str(MODEL_PATH / "processor"),
        "vae_scale_factor": 8,
        "infer_steps": num_inference_steps,
        "transformer_in_channels": 64,
        "num_layers": 60,
        "attention_out_dim": 3072,
        "attention_dim_head": 128,
        "attn_type": attn_mode,
        "enable_cfg": True,
        "sample_guide_scale": guidance_scale,
        "_auto_resize": auto_resize,
    }
    with open(config_path, "w") as f:
        json.dump(runtime_config, f)
    log(f"Created runtime config: {config_path}")

    # Configure generator with explicit config
    log(f"Configuring generator: steps={num_inference_steps}, guidance={guidance_scale}")
    pipe.create_generator(
        config_json=str(config_path),
        attn_mode=attn_mode,
        auto_resize=auto_resize,
        infer_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # Generate edited image
    output_path = str(work_dir / "output.png")

    # Validate paths before generation
    log(f"Input path: {input_path} (exists: {Path(input_path).exists()})")
    log(f"Output path: {output_path}")
    log(f"Work dir: {work_dir} (exists: {work_dir.exists()})")

    # Log all parameters being passed to generate()
    log(f"=== GENERATE PARAMETERS ===")
    log(f"  seed={seed} (type={type(seed).__name__})")
    log(f"  image_path={input_path} (type={type(input_path).__name__})")
    log(f"  prompt={prompt[:50]}... (type={type(prompt).__name__})")
    log(f"  negative_prompt='{negative_prompt}' (type={type(negative_prompt).__name__})")
    log(f"  save_result_path={output_path} (type={type(output_path).__name__})")
    log(f"===========================")

    log(f"Running edit: '{prompt[:50]}...' (steps={num_inference_steps})")
    gen_start = time.time()

    try:
        pipe.generate(
            seed=seed,
            image_path=input_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_result_path=output_path,
        )
    except Exception as e:
        import traceback
        log(f"Generation error: {e}")
        log(f"Traceback:\n{traceback.format_exc()}")
        raise

    gen_time = time.time() - gen_start
    log(f"Generation completed in {gen_time:.1f}s")

    # Load and encode result
    if not Path(output_path).exists():
        return {"error": "Generation failed - no output file produced"}

    output_image = Image.open(output_path)
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

    # Download models at startup (before serverless loop)
    # This happens during container initialization, not during job execution
    log("Downloading models at startup (this may take 5-10 min on first run)...")
    ensure_models()
    log(f"Model path: {MODEL_PATH}")
    log(f"FP8 weights path: {FP8_WEIGHTS_PATH}")

    runpod.serverless.start({"handler": handler})
