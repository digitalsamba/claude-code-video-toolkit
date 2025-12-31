#!/usr/bin/env python3
"""
Model download script for Wan2.2-I2V with LightX2V.

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
        "model_path": base_path / "wan-i2v",
    }


def check_model_exists(path: Path, min_files: int = 5) -> bool:
    """Check if model directory has enough files to be considered complete."""
    if not path.exists():
        return False
    files = list(path.glob("*"))
    return len(files) >= min_files


def download_wan_model(model_path: Path) -> bool:
    """Download Wan2.2-I2V-A14B model."""
    if check_model_exists(model_path, min_files=10):
        print(f"Wan2.2 model already exists at {model_path}")
        return True

    print("Downloading Wan2.2-I2V-A14B model (~35GB)...")
    print("This may take 10-15 minutes on first run.")

    try:
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-I2V-A14B",
            local_dir=str(model_path),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("Wan2.2 model downloaded successfully")
        return True
    except Exception as e:
        print(f"ERROR downloading Wan2.2 model: {e}")
        return False


def ensure_models_downloaded() -> dict:
    """
    Ensure all required models are downloaded.

    Returns dict with model paths if successful, raises exception if not.
    """
    paths = get_model_paths()

    # Download Wan2.2 model
    if not download_wan_model(paths["model_path"]):
        raise RuntimeError("Failed to download Wan2.2 model")

    print(f"\nAll models ready:")
    print(f"  Wan2.2 model: {paths['model_path']}")

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
