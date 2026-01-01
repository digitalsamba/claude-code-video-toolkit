#!/usr/bin/env python3
"""
Model download script for Qwen-Image-Edit with LightX2V.

Downloads models to network volume for caching across cold starts.
Supports both RunPod network volumes and local fallback.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


def get_model_paths():
    """Determine model paths based on available storage."""
    # Check for RunPod network volume first
    if Path("/runpod-volume").exists() and os.access("/runpod-volume", os.W_OK):
        base_path = Path("/runpod-volume/models")
        cache_path = Path("/runpod-volume/.cache/huggingface")
        print("Using RunPod network volume for model storage")
    else:
        # Fallback to local storage (container ephemeral storage)
        base_path = Path("/models")
        cache_path = Path("/root/.cache/huggingface")
        print("WARNING: No network volume found, using ephemeral storage")
        print("Models will be re-downloaded on each cold start!")

    base_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Set HF cache location
    os.environ["HF_HOME"] = str(cache_path)

    return {
        "base_path": base_path,
        "model_path": base_path / "qwen-edit",
        "fp8_path": base_path / "qwen-edit-fp8",
    }


def check_model_exists(path: Path, min_files: int = 5) -> bool:
    """Check if model directory has enough files to be considered complete."""
    if not path.exists():
        return False
    files = list(path.glob("*"))
    return len(files) >= min_files


def download_base_model(model_path: Path) -> bool:
    """Download Qwen-Image-Edit-2511 base model."""
    if check_model_exists(model_path, min_files=10):
        print(f"Base model already exists at {model_path}")
        # Still create/update the config
        create_lightx2v_config(model_path)
        return True

    print("Downloading Qwen-Image-Edit-2511 base model (~20GB)...")
    print("This may take 5-10 minutes on first run.")

    try:
        snapshot_download(
            repo_id="Qwen/Qwen-Image-Edit-2511",
            local_dir=str(model_path),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("Base model downloaded successfully")
        # Create LightX2V-compatible config
        create_lightx2v_config(model_path)
        return True
    except Exception as e:
        print(f"ERROR downloading base model: {e}")
        return False


def create_lightx2v_config(model_path: Path) -> None:
    """Create a top-level config.json for LightX2V compatibility."""
    import json

    config = {
        # Paths
        "model_path": str(model_path),
        "text_encoder_path": str(model_path / "text_encoder"),
        "tokenizer_path": str(model_path / "tokenizer"),
        "vae_path": str(model_path / "vae"),
        "transformer_path": str(model_path / "transformer"),
        "processor_path": str(model_path / "processor"),
        # Model architecture (from Qwen-Image-Edit-2511)
        "in_channels": 64,
        "out_channels": 16,
        "transformer_in_channels": 64,
        "num_layers": 60,
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "attention_out_dim": 3072,
        "joint_attention_dim": 3584,
        # VAE settings
        "vae_scale_factor": 8,
        # Inference defaults
        "infer_steps": 40,
        "guidance_scale": 1.0,
        "sample_guide_scale": 4.0,
        "enable_cfg": True,
        "_auto_resize": True,
    }

    config_path = model_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created LightX2V config: {config_path}")


def download_fp8_weights(fp8_path: Path) -> bool:
    """Download FP8 quantized Lightning weights."""
    if check_model_exists(fp8_path, min_files=3):
        print(f"FP8 weights already exist at {fp8_path}")
        return True

    print("Downloading FP8 Lightning weights (~10GB)...")

    try:
        snapshot_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            local_dir=str(fp8_path),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("FP8 weights downloaded successfully")
        return True
    except Exception as e:
        print(f"ERROR downloading FP8 weights: {e}")
        return False


def ensure_models_downloaded() -> dict:
    """
    Ensure all required models are downloaded.

    Returns dict with model paths if successful, raises exception if not.
    """
    paths = get_model_paths()

    # Download base model
    if not download_base_model(paths["model_path"]):
        raise RuntimeError("Failed to download base model")

    # Download FP8 weights
    if not download_fp8_weights(paths["fp8_path"]):
        raise RuntimeError("Failed to download FP8 weights")

    print(f"\nAll models ready:")
    print(f"  Base model: {paths['model_path']}")
    print(f"  FP8 weights: {paths['fp8_path']}")

    return paths


if __name__ == "__main__":
    # Can be run standalone to pre-download models
    try:
        paths = ensure_models_downloaded()
        print("\nModel download complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\nModel download failed: {e}")
        sys.exit(1)
