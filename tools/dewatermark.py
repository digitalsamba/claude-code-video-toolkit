#!/usr/bin/env python3
"""
Remove watermarks from video using AI inpainting (ProPainter).

Supports both local processing (NVIDIA GPU required) and cloud processing
via RunPod serverless GPUs.

Usage:
    # Local processing (requires NVIDIA GPU + ProPainter installation)
    python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4

    # Cloud processing via RunPod (works from any machine)
    python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod

    # Use custom mask image (white = area to remove, black = keep)
    python tools/dewatermark.py --input video.mp4 --mask mask.png --output clean.mp4

    # Install ProPainter for local processing (first-time setup)
    python tools/dewatermark.py --install

    # Check installation status
    python tools/dewatermark.py --status

RunPod Setup:
    1. Create account at runpod.io
    2. Deploy the propainter Docker image (see docker/runpod-propainter/)
    3. Add to .env:
       RUNPOD_API_KEY=your_key
       RUNPOD_ENDPOINT_ID=your_endpoint

Hardware (local mode):
    - NVIDIA GPU: Recommended (uses CUDA)
    - Apple Silicon: NOT SUPPORTED (too slow, use --runpod instead)
    - CPU-only: NOT SUPPORTED (too slow, use --runpod instead)

Cost (RunPod):
    - ~$0.02-0.25 per video depending on length
    - Uses RTX 3090 (~$0.34/hr) by default
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# Default installation path
PROPAINTER_HOME = Path.home() / ".video-toolkit" / "propainter"
PROPAINTER_REPO = "https://github.com/sczhou/ProPainter.git"

# Memory-based settings profiles (subvideo_length, neighbor_length, ref_stride)
# For Apple Silicon: based on unified RAM
# For NVIDIA: based on VRAM
MEMORY_PROFILES = {
    6:  {"subvideo_length": 25, "neighbor_length": 5,  "ref_stride": 30, "description": "6GB (minimal)"},
    8:  {"subvideo_length": 30, "neighbor_length": 5,  "ref_stride": 25, "description": "8GB"},
    12: {"subvideo_length": 40, "neighbor_length": 5,  "ref_stride": 20, "description": "12GB"},
    16: {"subvideo_length": 50, "neighbor_length": 8,  "ref_stride": 15, "description": "16GB"},
    24: {"subvideo_length": 60, "neighbor_length": 10, "ref_stride": 10, "description": "24GB"},
    32: {"subvideo_length": 80, "neighbor_length": 10, "ref_stride": 10, "description": "32GB+ (fastest)"},
}

# Watermark region presets (x, y, width, height) for common video sources
# Use --preset instead of --region for convenience
WATERMARK_PRESETS = {
    "notebooklm": {
        "description": "Google NotebookLM - bottom-right corner",
        "region_1280x720": "1100,650,150,50",
        "region_1920x1080": "1650,975,225,75",
    },
    "tiktok": {
        "description": "TikTok username - bottom-center",
        "region_1080x1920": "340,1750,400,80",  # Portrait
        "region_1280x720": "440,650,400,50",    # Landscape
    },
    "stock-br": {
        "description": "Stock footage - bottom-right",
        "region_1280x720": "1000,620,260,80",
        "region_1920x1080": "1500,930,390,120",
    },
    "stock-bl": {
        "description": "Stock footage - bottom-left",
        "region_1280x720": "20,620,260,80",
        "region_1920x1080": "30,930,390,120",
    },
    "stock-center": {
        "description": "Stock footage - center watermark",
        "region_1280x720": "440,260,400,200",
        "region_1920x1080": "660,390,600,300",
    },
}


def get_system_ram_gb() -> int | None:
    """Detect system RAM in GB. Returns None if detection fails."""
    try:
        if sys.platform == "darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) // (1024 ** 3)
        elif sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // (1024 ** 2)
        elif sys.platform == "win32":
            # Windows: use wmic
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    return int(lines[1].strip()) // (1024 ** 3)
    except Exception:
        pass
    return None


def get_nvidia_vram_gb() -> int | None:
    """Detect NVIDIA GPU VRAM in GB. Returns None if no NVIDIA GPU or detection fails."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            # Returns MB, convert to GB
            vram_mb = int(result.stdout.strip().split('\n')[0])
            return vram_mb // 1024
    except Exception:
        pass
    return None


def detect_compute_device() -> dict:
    """Detect available compute device and memory."""
    result = {
        "device": "cpu",
        "memory_gb": None,
        "description": "CPU (no GPU detected)",
    }

    # Check for NVIDIA GPU first
    nvidia_vram = get_nvidia_vram_gb()
    if nvidia_vram:
        result["device"] = "cuda"
        result["memory_gb"] = nvidia_vram
        result["description"] = f"NVIDIA GPU ({nvidia_vram}GB VRAM)"
        return result

    # Check for Apple Silicon (MPS)
    if sys.platform == "darwin":
        # On macOS, check if MPS is available via the ProPainter venv
        # For now, assume MPS on Apple Silicon and use system RAM
        ram_gb = get_system_ram_gb()
        if ram_gb:
            result["device"] = "mps"
            result["memory_gb"] = ram_gb
            result["description"] = f"Apple Silicon ({ram_gb}GB unified memory)"
            return result

    # Fall back to system RAM for CPU mode
    ram_gb = get_system_ram_gb()
    if ram_gb:
        result["memory_gb"] = ram_gb
        result["description"] = f"CPU ({ram_gb}GB RAM) - expect slow processing"

    return result


def get_memory_profile(memory_gb: int | None) -> dict:
    """Get recommended settings based on available memory (RAM or VRAM)."""
    if memory_gb is None:
        memory_gb = 8  # Conservative default

    # Find the best matching profile
    for threshold in sorted(MEMORY_PROFILES.keys(), reverse=True):
        if memory_gb >= threshold:
            return MEMORY_PROFILES[threshold]

    # Fall back to most conservative
    return MEMORY_PROFILES[6]


# Memory per frame at different resolutions (empirically determined from ProPainter)
# This accounts for RGB tensors, flow tensors, masks, and PyTorch overhead
BYTES_PER_FRAME_720P = 6.5 * 1024 * 1024  # ~6.5 MB per frame at 1280x720


def estimate_frame_memory_gb(width: int, height: int, frame_count: int) -> float:
    """Estimate memory required to load all frames into ProPainter."""
    # Scale from 720p baseline
    pixels = width * height
    pixels_720p = 1280 * 720
    scale_factor = pixels / pixels_720p

    bytes_per_frame = BYTES_PER_FRAME_720P * scale_factor
    total_bytes = bytes_per_frame * frame_count

    # Add ~20% overhead for processing buffers
    total_bytes *= 1.2

    return total_bytes / (1024 ** 3)


def get_video_info(video_path: str) -> dict | None:
    """Get detailed video information using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            fmt = data.get("format", {})

            # Parse frame rate (can be "30/1" or "29.97")
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)

            duration = float(fmt.get("duration", 0))

            # nb_frames may not always be available
            nb_frames = stream.get("nb_frames")
            if nb_frames:
                frame_count = int(nb_frames)
            else:
                frame_count = int(duration * fps)

            return {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": fps,
                "duration": duration,
                "frame_count": frame_count,
            }
    except Exception:
        pass
    return None


def calculate_max_duration(memory_gb: float, width: int, height: int, fps: float, device: str = "cpu") -> float:
    """Calculate maximum video duration that fits in available memory and hardware limits.

    Constraints:
    1. Memory: On unified memory systems (Apple Silicon), only ~50% is available
    2. MPS INT_MAX: Apple Silicon MPS cannot handle tensors > 2^31 elements
    """
    # Memory constraint
    available_for_frames = memory_gb * 0.50
    pixels = width * height
    pixels_720p = 1280 * 720
    scale_factor = pixels / pixels_720p
    bytes_per_frame = BYTES_PER_FRAME_720P * scale_factor
    max_frames_memory = int((available_for_frames * 1024 ** 3) / bytes_per_frame)

    # MPS INT_MAX constraint (Apple Silicon specific)
    # MPS cannot handle tensor dimensions > INT_MAX (2^31-1)
    # Total elements = frames × width × height × 3 (RGB channels)
    INT_MAX = 2_147_483_647
    elements_per_frame = width * height * 3
    max_frames_mps = INT_MAX // elements_per_frame

    # Use stricter limit on Apple Silicon MPS
    if device.lower() == "mps":
        # Use 90% of MPS limit for safety margin
        max_frames = min(max_frames_memory, int(max_frames_mps * 0.9))
    else:
        max_frames = max_frames_memory

    return max_frames / fps


def split_video_with_overlap(
    input_path: str,
    output_dir: str,
    chunk_duration: float,
    overlap: float = 5.0,
    verbose: bool = True,
) -> list[dict]:
    """Split video into chunks with overlap for seamless processing.

    Returns list of chunk info dicts with start, end, output_path, and trim points.
    """
    info = get_video_info(input_path)
    if not info:
        return []

    duration = info["duration"]
    chunks = []

    # Calculate chunk boundaries
    effective_chunk = chunk_duration - overlap  # Usable portion per chunk
    current_start = 0.0
    chunk_idx = 0

    while current_start < duration:
        chunk_end = min(current_start + chunk_duration, duration)

        # Determine trim points (where to cut when concatenating)
        # First chunk: use from start to (end - overlap/2)
        # Middle chunks: use from (overlap/2) to (end - overlap/2)
        # Last chunk: use from (overlap/2) to end

        is_first = chunk_idx == 0
        is_last = chunk_end >= duration

        if is_first:
            trim_start = 0.0
        else:
            trim_start = overlap / 2

        if is_last:
            trim_end = chunk_end - current_start
        else:
            trim_end = chunk_duration - (overlap / 2)

        chunk_path = str(Path(output_dir) / f"chunk_{chunk_idx:03d}.mp4")

        # Extract chunk using ffmpeg
        # Must re-encode (not -c copy) to ensure fps metadata is preserved
        # torchvision.io.read_video requires video_fps in metadata
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(current_start),
            "-i", input_path,
            "-t", str(chunk_end - current_start),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            chunk_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if verbose:
                print(f"Error extracting chunk {chunk_idx}: {result.stderr}", file=sys.stderr)
            return []

        chunks.append({
            "index": chunk_idx,
            "input_path": chunk_path,
            "output_path": None,  # Will be set after processing
            "start": current_start,
            "end": chunk_end,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "duration": chunk_end - current_start,
        })

        if verbose:
            print(f"  Chunk {chunk_idx}: {current_start:.1f}s - {chunk_end:.1f}s (use {trim_start:.1f}s - {trim_end:.1f}s)")

        chunk_idx += 1
        current_start += effective_chunk

        # Avoid tiny final chunks
        if duration - current_start < overlap:
            break

    return chunks


def concatenate_chunks(
    chunks: list[dict],
    output_path: str,
    crossfade_duration: float = 0.5,
    verbose: bool = True,
) -> bool:
    """Concatenate processed chunks with crossfade at boundaries."""
    if not chunks:
        return False

    if len(chunks) == 1:
        # Single chunk, just copy
        shutil.copy(chunks[0]["output_path"], output_path)
        return True

    # Build ffmpeg filter for trimming and concatenating
    filter_parts = []
    inputs = []

    for i, chunk in enumerate(chunks):
        if not chunk["output_path"] or not Path(chunk["output_path"]).exists():
            if verbose:
                print(f"Error: Missing processed chunk {i}", file=sys.stderr)
            return False

        inputs.extend(["-i", chunk["output_path"]])

        # Trim each chunk to its usable portion
        trim_start = chunk["trim_start"]
        trim_end = chunk["trim_end"]
        filter_parts.append(
            f"[{i}:v]trim=start={trim_start}:end={trim_end},setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a]atrim=start={trim_start}:end={trim_end},asetpts=PTS-STARTPTS[a{i}]"
        )

    # Concatenate all trimmed segments
    video_inputs = "".join(f"[v{i}]" for i in range(len(chunks)))
    audio_inputs = "".join(f"[a{i}]" for i in range(len(chunks)))

    filter_complex = (
        ";".join(filter_parts) + ";"
        f"{video_inputs}concat=n={len(chunks)}:v=1:a=0[outv];"
        f"{audio_inputs}concat=n={len(chunks)}:v=0:a=1[outa]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path,
    ]

    if verbose:
        print(f"Concatenating {len(chunks)} chunks...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if verbose:
            print(f"Error concatenating: {result.stderr[-500:]}", file=sys.stderr)
        return False

    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove watermarks using AI inpainting (ProPainter)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove NotebookLM watermark (bottom-right corner)
  python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4

  # Use higher quality (slower)
  python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --fp32

  # Install ProPainter
  python tools/dewatermark.py --install
        """,
    )

    # Main arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input video file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output video file path",
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        help="Watermark region as x,y,width,height (e.g., 1100,650,150,50)",
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=list(WATERMARK_PRESETS.keys()),
        help=f"Use preset watermark region: {', '.join(WATERMARK_PRESETS.keys())}",
    )
    parser.add_argument(
        "--mask", "-m",
        type=str,
        help="Custom mask image (white=remove, black=keep)",
    )

    # Quality/performance options
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use fp32 precision (higher quality, more memory)",
    )
    parser.add_argument(
        "--neighbor-length",
        type=int,
        default=10,
        help="Local neighbor frames for propagation (default: 10, reduce for less memory)",
    )
    parser.add_argument(
        "--ref-stride",
        type=int,
        default=10,
        help="Reference frame stride (default: 10, increase for less memory)",
    )
    parser.add_argument(
        "--subvideo-length",
        type=int,
        default=80,
        help="Frames per processing batch (default: 80, reduce for less memory)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect hardware and use optimal settings (recommended)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable auto-splitting even if video exceeds memory (may fail)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=5.0,
        help="Overlap duration in seconds between chunks (default: 5.0)",
    )

    # Installation/status
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install ProPainter to ~/.video-toolkit/propainter/",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check ProPainter installation status",
    )
    parser.add_argument(
        "--propainter-path",
        type=str,
        help=f"Custom ProPainter installation path (default: {PROPAINTER_HOME})",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files (frames, masks)",
    )

    # RunPod cloud processing
    parser.add_argument(
        "--runpod",
        action="store_true",
        help="Process on RunPod serverless GPU instead of locally",
    )
    parser.add_argument(
        "--runpod-timeout",
        type=int,
        default=1800,
        help="RunPod job timeout in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up RunPod endpoint automatically (creates template + endpoint)",
    )
    parser.add_argument(
        "--setup-gpu",
        type=str,
        default="AMPERE_24",
        choices=["AMPERE_16", "AMPERE_24", "ADA_24", "AMPERE_48", "ADA_48_PRO", "AMPERE_80"],
        help="GPU type for RunPod endpoint (default: AMPERE_24 = RTX 3090)",
    )
    parser.add_argument(
        "--resize-ratio",
        type=str,
        default="auto",
        help="""Scale factor for video processing:
  'auto' - suggests based on video duration (recommended)
  1.0    - full resolution, safe for videos <30s
  0.75   - good for videos 30s-1 min
  0.5    - reliable for videos 1-5 min
  For longer videos, consider --chunk mode""",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale output to original resolution using FFmpeg lanczos (useful with resize-ratio < 1.0)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        metavar="SECONDS",
        help="Process video in chunks of N seconds (for long videos that OOM even at 0.5). Chunks are auto-joined.",
    )

    return parser.parse_args()


def get_propainter_path(custom_path: str | None = None) -> Path:
    """Get ProPainter installation path."""
    if custom_path:
        return Path(custom_path)
    return PROPAINTER_HOME


def check_propainter_installed(propainter_path: Path) -> dict:
    """Check if ProPainter is installed and ready."""
    status = {
        "installed": False,
        "path": str(propainter_path),
        "has_repo": False,
        "has_weights": False,
        "has_venv": False,
        "mps_available": False,
        "cuda_available": False,
    }

    if not propainter_path.exists():
        return status

    # Check for repo
    inference_script = propainter_path / "inference_propainter.py"
    status["has_repo"] = inference_script.exists()

    # Check for model weights
    weights_dir = propainter_path / "weights"
    if weights_dir.exists():
        expected_weights = ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]
        status["has_weights"] = all((weights_dir / w).exists() for w in expected_weights)

    # Check for venv
    venv_path = propainter_path / ".venv"
    status["has_venv"] = venv_path.exists() and (venv_path / "bin" / "python").exists()

    # Check PyTorch device availability
    if status["has_venv"]:
        python_bin = venv_path / "bin" / "python"
        try:
            result = subprocess.run(
                [str(python_bin), "-c",
                 "import torch; print('mps' if torch.backends.mps.is_available() else '', 'cuda' if torch.cuda.is_available() else '')"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                status["mps_available"] = "mps" in output
                status["cuda_available"] = "cuda" in output
        except Exception:
            pass

    status["installed"] = status["has_repo"] and status["has_weights"] and status["has_venv"]
    return status


def install_propainter(propainter_path: Path, verbose: bool = True) -> bool:
    """Install ProPainter with all dependencies."""

    if verbose:
        print("=" * 60)
        print("ProPainter Installation")
        print("=" * 60)
        print(f"Installation path: {propainter_path}")
        print("This will download ~2GB of model weights.")
        print()

    # Create parent directory
    propainter_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Clone repository
    if verbose:
        print("[1/4] Cloning ProPainter repository...")

    if propainter_path.exists():
        if verbose:
            print(f"  Directory exists, checking for updates...")
        # Try git pull if it's a git repo
        if (propainter_path / ".git").exists():
            subprocess.run(
                ["git", "pull"],
                cwd=propainter_path,
                capture_output=not verbose,
            )
    else:
        result = subprocess.run(
            ["git", "clone", PROPAINTER_REPO, str(propainter_path)],
            capture_output=not verbose,
        )
        if result.returncode != 0:
            print(f"Error: Failed to clone repository", file=sys.stderr)
            return False

    # Step 2: Create virtual environment
    if verbose:
        print("[2/4] Creating Python virtual environment...")

    venv_path = propainter_path / ".venv"
    if not venv_path.exists():
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=not verbose,
        )
        if result.returncode != 0:
            print(f"Error: Failed to create virtual environment", file=sys.stderr)
            return False

    python_bin = venv_path / "bin" / "python"
    pip_bin = venv_path / "bin" / "pip"

    # Step 3: Install dependencies
    if verbose:
        print("[3/4] Installing Python dependencies...")
        print("  This may take a few minutes...")

    # Upgrade pip first
    subprocess.run(
        [str(pip_bin), "install", "--upgrade", "pip"],
        capture_output=True,
    )

    # Install PyTorch (with MPS support for Apple Silicon)
    result = subprocess.run(
        [str(pip_bin), "install", "torch", "torchvision"],
        capture_output=not verbose,
    )
    if result.returncode != 0:
        print(f"Error: Failed to install PyTorch", file=sys.stderr)
        return False

    # Install other requirements
    requirements_file = propainter_path / "requirements.txt"
    if requirements_file.exists():
        result = subprocess.run(
            [str(pip_bin), "install", "-r", str(requirements_file)],
            capture_output=not verbose,
        )
        if result.returncode != 0:
            print(f"Warning: Some requirements may have failed to install", file=sys.stderr)

    # Step 4: Download model weights
    if verbose:
        print("[4/4] Downloading model weights (~2GB)...")

    weights_dir = propainter_path / "weights"
    weights_dir.mkdir(exist_ok=True)

    # ProPainter provides a download script
    download_script = propainter_path / "scripts" / "download_models.py"
    if download_script.exists():
        result = subprocess.run(
            [str(python_bin), str(download_script)],
            cwd=propainter_path,
            capture_output=not verbose,
        )
    else:
        # Manual download URLs
        weights_urls = {
            "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
            "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
            "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
            "i3d_rgb_imagenet.pt": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt",
        }

        for filename, url in weights_urls.items():
            weight_path = weights_dir / filename
            if not weight_path.exists():
                if verbose:
                    print(f"  Downloading {filename}...")
                result = subprocess.run(
                    ["curl", "-L", "-o", str(weight_path), url],
                    capture_output=not verbose,
                )
                if result.returncode != 0:
                    print(f"Error: Failed to download {filename}", file=sys.stderr)
                    return False

    # Verify installation
    status = check_propainter_installed(propainter_path)

    if verbose:
        print()
        print("=" * 60)
        if status["installed"]:
            print("Installation complete!")
            device = "MPS (Apple Silicon)" if status["mps_available"] else "CUDA" if status["cuda_available"] else "CPU"
            print(f"Detected device: {device}")
        else:
            print("Installation incomplete. Please check errors above.")
            print(f"  Repo: {'OK' if status['has_repo'] else 'MISSING'}")
            print(f"  Weights: {'OK' if status['has_weights'] else 'MISSING'}")
            print(f"  Venv: {'OK' if status['has_venv'] else 'MISSING'}")
        print("=" * 60)

    return status["installed"]


def get_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Get video width and height using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def create_mask_from_region(
    region: str,
    video_width: int,
    video_height: int,
    output_path: str,
) -> bool:
    """Create a mask image from x,y,w,h region specification."""
    try:
        parts = [int(x.strip()) for x in region.split(",")]
        if len(parts) != 4:
            print(f"Error: Region must be x,y,width,height (got {len(parts)} values)", file=sys.stderr)
            return False

        x, y, w, h = parts

        # Validate bounds
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            print(f"Error: Invalid region values (must be positive)", file=sys.stderr)
            return False
        if x + w > video_width or y + h > video_height:
            print(f"Error: Region exceeds video dimensions ({video_width}x{video_height})", file=sys.stderr)
            return False

        # Create mask using ffmpeg (black background, white rectangle)
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=black:s={video_width}x{video_height}:d=1",
                "-vf", f"drawbox=x={x}:y={y}:w={w}:h={h}:c=white:t=fill",
                "-frames:v", "1",
                output_path,
            ],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0

    except ValueError as e:
        print(f"Error: Invalid region format: {e}", file=sys.stderr)
        return False


def run_propainter(
    propainter_path: Path,
    video_path: str,
    mask_path: str,
    output_dir: str,
    fp16: bool = True,
    neighbor_length: int = 10,
    ref_stride: int = 10,
    subvideo_length: int = 80,
    verbose: bool = True,
) -> str | None:
    """Run ProPainter inference."""

    venv_python = propainter_path / ".venv" / "bin" / "python"
    inference_script = propainter_path / "inference_propainter.py"

    if not venv_python.exists() or not inference_script.exists():
        print("Error: ProPainter not properly installed", file=sys.stderr)
        return None

    cmd = [
        str(venv_python),
        str(inference_script),
        "-i", video_path,
        "-m", mask_path,
        "-o", output_dir,
        "--neighbor_length", str(neighbor_length),
        "--ref_stride", str(ref_stride),
        "--subvideo_length", str(subvideo_length),
    ]

    if fp16:
        cmd.append("--fp16")

    if verbose:
        print(f"Running ProPainter...")
        print(f"  Video: {video_path}")
        print(f"  Mask: {mask_path}")
        print(f"  Output: {output_dir}")
        print(f"  Precision: {'fp16' if fp16 else 'fp32'}")

    # Always capture output so we can report errors
    result = subprocess.run(
        cmd,
        cwd=propainter_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if result.stderr:
            print(f"ProPainter error: {result.stderr[-1000:]}", file=sys.stderr)
        return None

    # Find output file (ProPainter creates results/<video_name>/<video_name>_inpaint.mp4)
    video_name = Path(video_path).stem
    expected_output = Path(output_dir) / video_name / f"{video_name}_inpaint.mp4"

    if expected_output.exists():
        return str(expected_output)

    # Try to find any mp4 in output
    for mp4 in Path(output_dir).rglob("*.mp4"):
        return str(mp4)

    return None


# =============================================================================
# RunPod Cloud Processing
# =============================================================================

def get_runpod_config() -> dict:
    """Get RunPod configuration from environment."""
    # Import here to avoid circular dependency and allow graceful failure
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_runpod_api_key, get_runpod_endpoint_id
        return {
            "api_key": get_runpod_api_key(),
            "endpoint_id": get_runpod_endpoint_id(),
        }
    except ImportError:
        # Fallback to direct env var access
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "api_key": os.getenv("RUNPOD_API_KEY"),
            "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID"),
        }


def _get_r2_client():
    """Get boto3 S3 client configured for Cloudflare R2."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if not r2_config:
        return None, None

    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        return client, r2_config
    except ImportError:
        print("  boto3 not installed, skipping R2", file=sys.stderr)
        return None, None


def _upload_to_r2(file_path: str, file_name: str) -> tuple[str | None, str | None]:
    """
    Upload to Cloudflare R2 and return presigned download URL.

    Returns (url, object_key) tuple. object_key is needed for cleanup.
    """
    client, config = _get_r2_client()
    if not client:
        return None, None

    import uuid
    object_key = f"dewatermark/{uuid.uuid4().hex[:8]}_{file_name}"

    try:
        client.upload_file(file_path, config["bucket_name"], object_key)

        # Generate presigned URL (valid for 2 hours)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return url, object_key
    except Exception as e:
        print(f"  R2 upload error: {e}", file=sys.stderr)
        return None, None


def _delete_from_r2(object_key: str) -> bool:
    """Delete object from R2 after job completion."""
    client, config = _get_r2_client()
    if not client or not object_key:
        return False

    try:
        client.delete_object(Bucket=config["bucket_name"], Key=object_key)
        return True
    except Exception:
        return False


def _download_from_r2(object_key: str, output_path: str) -> bool:
    """Download object from R2 to local path."""
    client, config = _get_r2_client()
    if not client:
        return False

    try:
        client.download_file(config["bucket_name"], object_key, output_path)
        return True
    except Exception as e:
        print(f"  R2 download error: {e}", file=sys.stderr)
        return False


def upload_to_runpod_storage(file_path: str, api_key: str) -> tuple[str | None, str | None]:
    """
    Upload a file to temporary storage for job input.

    Returns (url, r2_key) tuple. r2_key is set if R2 was used (for cleanup).
    Falls back to free file hosting services if R2 not configured.
    """
    file_size = Path(file_path).stat().st_size
    file_name = Path(file_path).name

    print(f"Uploading {file_name} ({file_size // (1024*1024)}MB)...", file=sys.stderr)

    # Try R2 first if configured
    url, r2_key = _upload_to_r2(file_path, file_name)
    if url:
        print(f"  Upload complete (R2): {url[:60]}...", file=sys.stderr)
        return url, r2_key

    # Fall back to free services
    upload_services = [
        ("litterbox", _upload_to_litterbox),
        ("0x0.st", _upload_to_0x0),
        ("file.io", _upload_to_fileio),
        ("transfer.sh", _upload_to_transfersh),
    ]

    for service_name, upload_func in upload_services:
        try:
            url = upload_func(file_path, file_name)
            if url:
                print(f"  Upload complete ({service_name}): {url[:60]}...", file=sys.stderr)
                return url, None  # No R2 key for cleanup
        except Exception as e:
            print(f"  {service_name} failed: {e}", file=sys.stderr)
            continue

    print("All upload services failed", file=sys.stderr)
    return None, None


def _upload_to_litterbox(file_path: str, file_name: str) -> str | None:
    """Upload to litterbox.catbox.moe (200MB limit, 24h retention) using curl."""
    result = subprocess.run(
        [
            "curl", "-s",
            "-F", "reqtype=fileupload",
            "-F", "time=24h",
            "-F", f"fileToUpload=@{file_path}",
            "https://litterbox.catbox.moe/resources/internals/api.php",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
        else:
            raise Exception(f"Unexpected response: {url[:100]}")
    else:
        raise Exception(f"curl failed: {result.stderr[:100]}")


def _upload_to_0x0(file_path: str, file_name: str) -> str | None:
    """Upload to 0x0.st (512MB limit, 30 day retention) using curl."""
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{file_path}", "https://0x0.st"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
        else:
            raise Exception(f"Unexpected response: {url[:100]}")
    else:
        raise Exception(f"curl failed: {result.stderr[:100]}")


def _upload_to_fileio(file_path: str, file_name: str) -> str | None:
    """Upload to file.io (2GB limit, 1 download then deleted)."""
    with open(file_path, 'rb') as f:
        response = requests.post(
            "https://file.io",
            files={"file": (file_name, f)},
            timeout=600,
        )
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            return data.get("link")
    return None


def _upload_to_transfersh(file_path: str, file_name: str) -> str | None:
    """Upload to transfer.sh (10GB limit, 14 day retention)."""
    with open(file_path, 'rb') as f:
        response = requests.put(
            f"https://transfer.sh/{file_name}",
            data=f,
            headers={"Max-Downloads": "5", "Max-Days": "1"},
            timeout=600,
        )
    if response.status_code == 200:
        return response.text.strip()
    return None


def submit_runpod_job(
    endpoint_id: str,
    api_key: str,
    video_url: str,
    region: str | None = None,
    mask_url: str | None = None,
    r2_config: dict | None = None,
    resize_ratio: str | float = "auto",
) -> dict | None:
    """Submit a dewatermark job to RunPod serverless endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    # Handle resize_ratio: "auto" or numeric value
    if resize_ratio == "auto":
        ratio_value = "auto"
    else:
        ratio_value = float(resize_ratio)

    payload = {
        "input": {
            "operation": "dewatermark",
            "video_url": video_url,
            "resize_ratio": ratio_value,
        }
    }

    if region:
        payload["input"]["region"] = region
    if mask_url:
        payload["input"]["mask_url"] = mask_url

    # Pass R2 credentials for result upload (if configured)
    if r2_config:
        payload["input"]["r2"] = {
            "endpoint_url": r2_config["endpoint_url"],
            "access_key_id": r2_config["access_key_id"],
            "secret_access_key": r2_config["secret_access_key"],
            "bucket_name": r2_config["bucket_name"],
        }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Job submission failed: HTTP {response.status_code}", file=sys.stderr)
            print(f"  Response: {response.text[:500]}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Job submission error: {e}", file=sys.stderr)
        return None


def poll_runpod_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    timeout: int = 1800,
    poll_interval: int = 5,
    verbose: bool = True,
) -> dict | None:
    """Poll RunPod job until completion or timeout."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )

            if response.status_code != 200:
                print(f"Status check failed: HTTP {response.status_code}", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            data = response.json()
            status = data.get("status")

            # Log status changes
            if verbose and status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed}s] Status: {status}", file=sys.stderr)
                last_status = status

            if status == "COMPLETED":
                return data
            elif status == "FAILED":
                print(f"Job failed: {data.get('error', 'Unknown error')}", file=sys.stderr)
                return data
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                # Unknown status
                time.sleep(poll_interval)

        except Exception as e:
            print(f"Status check error: {e}", file=sys.stderr)
            time.sleep(poll_interval)

    print(f"Job timed out after {timeout}s", file=sys.stderr)
    return None


def download_from_url(url: str, output_path: str, verbose: bool = True) -> bool:
    """Download file from URL to local path."""
    try:
        if verbose:
            print(f"Downloading result...", file=sys.stderr)

        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        if verbose:
            size_mb = Path(output_path).stat().st_size // (1024 * 1024)
            print(f"  Downloaded: {output_path} ({size_mb}MB)", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        return False


def resolve_preset_region(preset: str, width: int, height: int) -> str | None:
    """
    Resolve a preset name to a region string based on video dimensions.
    Returns region as "x,y,w,h" string or None if preset not found.
    """
    if preset not in WATERMARK_PRESETS:
        return None

    preset_data = WATERMARK_PRESETS[preset]

    # Try exact match first
    key = f"region_{width}x{height}"
    if key in preset_data:
        return preset_data[key]

    # Try common resolutions
    for res_key in preset_data:
        if res_key.startswith("region_"):
            return preset_data[res_key]  # Return first available

    return None


def suggest_resize_ratio(duration_seconds: float, width: int = 1280, height: int = 720) -> tuple[float, str]:
    """
    Suggest a resize ratio based on video duration and resolution.

    Based on empirical testing with 80GB A100:
    - < 30s at 720p: 1.0 works (full resolution)
    - 30-60s at 720p: 0.75 works
    - 60-180s at 720p: 0.5 recommended
    - > 180s at 720p: 0.5 + chunking recommended

    Returns (ratio, reason) tuple.
    """
    # Adjust for resolution (720p is baseline)
    pixels = width * height
    pixels_720p = 1280 * 720
    resolution_factor = pixels / pixels_720p

    # Effective duration accounting for resolution
    effective_duration = duration_seconds * resolution_factor

    if effective_duration <= 30:
        return (1.0, f"Video is {duration_seconds:.0f}s - full resolution safe")
    elif effective_duration <= 60:
        return (0.75, f"Video is {duration_seconds:.0f}s - 0.75 should work")
    elif effective_duration <= 180:
        return (0.5, f"Video is {duration_seconds:.0f}s - 0.5 recommended for reliability")
    else:
        return (0.5, f"Video is {duration_seconds:.0f}s - consider --chunk 60 for very long videos")


def upscale_video(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    verbose: bool = True,
) -> bool:
    """Upscale video to target resolution using FFmpeg lanczos filter."""
    try:
        if verbose:
            print(f"  Upscaling to {target_width}x{target_height}...", file=sys.stderr)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            if verbose:
                size_mb = Path(output_path).stat().st_size // (1024 * 1024)
                print(f"  Upscaled: {output_path} ({size_mb}MB)", file=sys.stderr)
            return True
        else:
            if verbose:
                print(f"  Upscale failed: {result.stderr[:500]}", file=sys.stderr)
            return False
    except Exception as e:
        if verbose:
            print(f"  Upscale error: {e}", file=sys.stderr)
        return False


def mux_audio_from_original(
    video_no_audio: str,
    original_video: str,
    output_path: str,
    verbose: bool = True,
) -> bool:
    """Mux audio from original video into processed video (ProPainter strips audio)."""
    try:
        # Check if original has audio
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_name", "-of", "json", original_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False

        import json as json_module
        probe_data = json_module.loads(result.stdout)
        if not probe_data.get("streams"):
            # No audio in original, just copy video
            if verbose:
                print(f"  No audio in original, copying video only", file=sys.stderr)
            shutil.copy2(video_no_audio, output_path)
            return True

        if verbose:
            print(f"  Restoring audio from original...", file=sys.stderr)

        # Mux audio from original into processed video
        cmd = [
            "ffmpeg", "-y",
            "-i", video_no_audio,
            "-i", original_video,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        if verbose:
            print(f"  Warning: Failed to restore audio: {e}", file=sys.stderr)
        return False


def process_with_runpod(
    input_path: str,
    output_path: str,
    region: str | None = None,
    mask_path: str | None = None,
    timeout: int = 1800,
    verbose: bool = True,
    resize_ratio: str | float = "auto",
    preserve_audio: bool = True,
    upscale: bool = False,
    original_width: int | None = None,
    original_height: int | None = None,
) -> dict:
    """
    Process video using RunPod serverless endpoint.

    Args:
        upscale: If True and resize_ratio < 1.0, upscale output to original resolution
        original_width/height: Original video dimensions (for upscaling)

    Returns dict with success/error and metadata.
    """
    start_time = time.time()
    r2_keys_to_cleanup = []  # Track R2 objects for cleanup

    # Get RunPod config
    config = get_runpod_config()
    api_key = config.get("api_key")
    endpoint_id = config.get("endpoint_id")

    if not api_key:
        return {"error": "RUNPOD_API_KEY not set. Add to .env file."}
    if not endpoint_id:
        return {"error": "RUNPOD_ENDPOINT_ID not set. Add to .env file."}

    # Get R2 config (optional, for reliable file transfer)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if verbose:
        print(f"Using RunPod endpoint: {endpoint_id}", file=sys.stderr)
        if r2_config:
            print(f"Using Cloudflare R2 for file transfer", file=sys.stderr)
        else:
            print(f"R2 not configured, using free file hosting (less reliable)", file=sys.stderr)

    # Upload video
    video_url, video_r2_key = upload_to_runpod_storage(input_path, api_key)
    if not video_url:
        return {"error": "Failed to upload video"}
    if video_r2_key:
        r2_keys_to_cleanup.append(video_r2_key)

    # Upload mask if provided (instead of region)
    mask_url = None
    if mask_path:
        mask_url, mask_r2_key = upload_to_runpod_storage(mask_path, api_key)
        if not mask_url:
            return {"error": "Failed to upload mask"}
        if mask_r2_key:
            r2_keys_to_cleanup.append(mask_r2_key)

    # Submit job
    if verbose:
        print(f"Submitting job...", file=sys.stderr)

    job_response = submit_runpod_job(
        endpoint_id=endpoint_id,
        api_key=api_key,
        video_url=video_url,
        region=region,
        mask_url=mask_url,
        r2_config=r2_config,
        resize_ratio=resize_ratio,
    )

    if not job_response:
        return {"error": "Failed to submit job"}

    job_id = job_response.get("id")
    if not job_id:
        return {"error": f"No job ID in response: {job_response}"}

    if verbose:
        print(f"Job submitted: {job_id}", file=sys.stderr)
        print(f"Waiting for completion (timeout: {timeout}s)...", file=sys.stderr)

    # Poll for completion
    result = poll_runpod_job(
        endpoint_id=endpoint_id,
        api_key=api_key,
        job_id=job_id,
        timeout=timeout,
        verbose=verbose,
    )

    if not result:
        return {"error": "Job timed out or failed to get status"}

    status = result.get("status")
    if status != "COMPLETED":
        error = result.get("error") or result.get("output", {}).get("error") or "Unknown error"
        return {"error": f"Job failed: {error}"}

    # Get output from result
    output = result.get("output", {})
    if isinstance(output, dict) and output.get("error"):
        return {"error": output["error"]}

    # Download result - try R2 first if key provided, then URL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    output_r2_key = output.get("r2_key") if isinstance(output, dict) else None
    output_url = output.get("output_url") if isinstance(output, dict) else None

    if output_r2_key:
        if verbose:
            print(f"Downloading result from R2...", file=sys.stderr)
        downloaded = _download_from_r2(output_r2_key, output_path)
        if downloaded:
            r2_keys_to_cleanup.append(output_r2_key)
            if verbose:
                size_mb = Path(output_path).stat().st_size // (1024 * 1024)
                print(f"  Downloaded: {output_path} ({size_mb}MB)", file=sys.stderr)

    if not downloaded and output_url:
        downloaded = download_from_url(output_url, output_path, verbose=verbose)

    if not downloaded:
        return {"error": f"No output_url or r2_key in result: {output}"}

    # Restore audio from original (ProPainter strips audio)
    if preserve_audio:
        temp_video = output_path + ".noaudio.mp4"
        shutil.move(output_path, temp_video)
        if mux_audio_from_original(temp_video, input_path, output_path, verbose=verbose):
            Path(temp_video).unlink(missing_ok=True)
        else:
            # Fallback: keep video without audio
            shutil.move(temp_video, output_path)
            if verbose:
                print(f"  Warning: Could not restore audio, output has no audio", file=sys.stderr)

    # Upscale to original resolution if requested
    actual_ratio = resize_ratio if isinstance(resize_ratio, float) else None
    if upscale and original_width and original_height and actual_ratio and actual_ratio < 1.0:
        temp_video = output_path + ".small.mp4"
        shutil.move(output_path, temp_video)
        if upscale_video(temp_video, output_path, original_width, original_height, verbose=verbose):
            Path(temp_video).unlink(missing_ok=True)
        else:
            # Fallback: keep smaller video
            shutil.move(temp_video, output_path)
            if verbose:
                print(f"  Warning: Upscale failed, output is at reduced resolution", file=sys.stderr)

    # Cleanup R2 objects
    if r2_keys_to_cleanup:
        if verbose:
            print(f"Cleaning up {len(r2_keys_to_cleanup)} R2 objects...", file=sys.stderr)
        for key in r2_keys_to_cleanup:
            _delete_from_r2(key)

    elapsed = time.time() - start_time

    return {
        "success": True,
        "output": output_path,
        "job_id": job_id,
        "processing_time_seconds": round(elapsed, 2),
        "runpod_output": output,
    }


# =============================================================================
# RunPod Setup (GraphQL API)
# =============================================================================

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
PROPAINTER_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-propainter:latest"
PROPAINTER_TEMPLATE_NAME = "video-toolkit-propainter"
PROPAINTER_ENDPOINT_NAME = "video-toolkit-dewatermark"


def runpod_graphql_query(api_key: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against RunPod API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(
        RUNPOD_GRAPHQL_URL,
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        raise Exception(f"GraphQL request failed: HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")

    return data.get("data", {})


def list_runpod_templates(api_key: str) -> list[dict]:
    """List all user templates."""
    query = """
    query {
        myself {
            podTemplates {
                id
                name
                imageName
                isServerless
            }
        }
    }
    """
    data = runpod_graphql_query(api_key, query)
    templates = data.get("myself", {}).get("podTemplates", [])
    # Filter to serverless templates only
    return [t for t in templates if t.get("isServerless")]


def find_propainter_template(api_key: str) -> dict | None:
    """Find existing ProPainter template by name or image."""
    templates = list_runpod_templates(api_key)
    for t in templates:
        if t.get("name") == PROPAINTER_TEMPLATE_NAME:
            return t
        if t.get("imageName") == PROPAINTER_DOCKER_IMAGE:
            return t
    return None


def create_runpod_template(api_key: str, verbose: bool = True) -> dict:
    """Create a serverless template for ProPainter."""
    if verbose:
        print(f"Creating template '{PROPAINTER_TEMPLATE_NAME}'...")

    mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
            isServerless
        }
    }
    """

    variables = {
        "input": {
            "name": PROPAINTER_TEMPLATE_NAME,
            "imageName": PROPAINTER_DOCKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "dockerArgs": "",
            "env": [],
        }
    }

    data = runpod_graphql_query(api_key, mutation, variables)
    template = data.get("saveTemplate")

    if not template or not template.get("id"):
        raise Exception(f"Failed to create template: {data}")

    if verbose:
        print(f"  Template created: {template['id']}")

    return template


def list_runpod_endpoints(api_key: str) -> list[dict]:
    """List all user endpoints."""
    query = """
    query {
        myself {
            endpoints {
                id
                name
                templateId
                gpuIds
                workersMin
                workersMax
                idleTimeout
            }
        }
    }
    """
    data = runpod_graphql_query(api_key, query)
    return data.get("myself", {}).get("endpoints", [])


def find_propainter_endpoint(api_key: str, template_id: str) -> dict | None:
    """Find existing ProPainter endpoint by name or template."""
    endpoints = list_runpod_endpoints(api_key)
    for e in endpoints:
        if e.get("name") == PROPAINTER_ENDPOINT_NAME:
            return e
        if e.get("templateId") == template_id:
            return e
    return None


def create_runpod_endpoint(
    api_key: str,
    template_id: str,
    gpu_id: str = "AMPERE_24",
    verbose: bool = True,
) -> dict:
    """Create a serverless endpoint for ProPainter."""
    if verbose:
        print(f"Creating endpoint '{PROPAINTER_ENDPOINT_NAME}'...")

    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
            gpuIds
            workersMin
            workersMax
            idleTimeout
        }
    }
    """

    variables = {
        "input": {
            "name": PROPAINTER_ENDPOINT_NAME,
            "templateId": template_id,
            "gpuIds": gpu_id,
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 5,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }

    data = runpod_graphql_query(api_key, mutation, variables)
    endpoint = data.get("saveEndpoint")

    if not endpoint or not endpoint.get("id"):
        raise Exception(f"Failed to create endpoint: {data}")

    if verbose:
        print(f"  Endpoint created: {endpoint['id']}")

    return endpoint


def save_endpoint_to_env(endpoint_id: str, verbose: bool = True) -> bool:
    """Save endpoint ID to .env file."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import find_workspace_root
        env_path = find_workspace_root() / ".env"
    except ImportError:
        env_path = Path(__file__).parent.parent / ".env"

    if verbose:
        print(f"Saving endpoint ID to {env_path}...")

    # Read existing .env content
    env_content = ""
    if env_path.exists():
        env_content = env_path.read_text()

    # Check if RUNPOD_ENDPOINT_ID already exists
    lines = env_content.split("\n")
    updated = False
    new_lines = []

    for line in lines:
        if line.startswith("RUNPOD_ENDPOINT_ID="):
            new_lines.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        # Add new line
        if new_lines and new_lines[-1].strip():
            new_lines.append("")  # Ensure blank line before
        new_lines.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")

    # Write back
    env_path.write_text("\n".join(new_lines))

    if verbose:
        print(f"  Saved: RUNPOD_ENDPOINT_ID={endpoint_id}")

    return True


def setup_runpod(gpu_id: str = "AMPERE_24", verbose: bool = True) -> dict:
    """
    Set up RunPod endpoint for dewatermark tool.

    1. Check API key
    2. Find or create template
    3. Find or create endpoint
    4. Save endpoint ID to .env

    Returns dict with setup result.
    """
    result = {
        "success": False,
        "template_id": None,
        "endpoint_id": None,
        "created_template": False,
        "created_endpoint": False,
    }

    # Get API key
    config = get_runpod_config()
    api_key = config.get("api_key")

    if not api_key:
        result["error"] = "RUNPOD_API_KEY not set. Add to .env file first."
        return result

    if verbose:
        print("=" * 60)
        print("RunPod Setup")
        print("=" * 60)
        print(f"Docker Image: {PROPAINTER_DOCKER_IMAGE}")
        print(f"GPU Type: {gpu_id}")
        print()

    try:
        # Step 1: Find or create template
        if verbose:
            print("[1/3] Checking for existing template...")

        template = find_propainter_template(api_key)
        if template:
            if verbose:
                print(f"  Found existing template: {template['id']}")
            result["template_id"] = template["id"]
        else:
            template = create_runpod_template(api_key, verbose=verbose)
            result["template_id"] = template["id"]
            result["created_template"] = True

        # Step 2: Find or create endpoint
        if verbose:
            print("[2/3] Checking for existing endpoint...")

        endpoint = find_propainter_endpoint(api_key, result["template_id"])
        if endpoint:
            if verbose:
                print(f"  Found existing endpoint: {endpoint['id']}")
            result["endpoint_id"] = endpoint["id"]
        else:
            endpoint = create_runpod_endpoint(
                api_key,
                result["template_id"],
                gpu_id=gpu_id,
                verbose=verbose,
            )
            result["endpoint_id"] = endpoint["id"]
            result["created_endpoint"] = True

        # Step 3: Save to .env
        if verbose:
            print("[3/3] Saving configuration...")

        save_endpoint_to_env(result["endpoint_id"], verbose=verbose)

        result["success"] = True

        if verbose:
            print()
            print("=" * 60)
            print("Setup Complete!")
            print("=" * 60)
            print(f"Template ID:  {result['template_id']}")
            print(f"Endpoint ID:  {result['endpoint_id']}")
            print()
            print("You can now run:")
            print("  python tools/dewatermark.py --input video.mp4 --region x,y,w,h --output out.mp4 --runpod")
            print()

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


def main():
    args = parse_args()
    propainter_path = get_propainter_path(args.propainter_path)
    verbose = not args.json

    # Handle --status
    if args.status:
        status = check_propainter_installed(propainter_path)
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("ProPainter Installation Status")
            print("-" * 40)
            print(f"Path: {status['path']}")
            print(f"Installed: {'Yes' if status['installed'] else 'No'}")
            if status['installed']:
                if status["cuda_available"]:
                    print(f"Device: CUDA (supported)")
                elif status["mps_available"]:
                    print(f"Device: MPS (NOT SUPPORTED - too slow)")
                    print()
                    print("⚠️  Apple Silicon MPS is not viable for this tool.")
                    print("   Use NVIDIA GPU or cloud service (RunPod, Vast.ai)")
                else:
                    print(f"Device: CPU (NOT SUPPORTED - too slow)")
            else:
                print(f"  Repository: {'OK' if status['has_repo'] else 'Missing'}")
                print(f"  Weights: {'OK' if status['has_weights'] else 'Missing'}")
                print(f"  Venv: {'OK' if status['has_venv'] else 'Missing'}")
                print()
                print("Run with --install to set up ProPainter")
        return

    # Handle --install
    if args.install:
        success = install_propainter(propainter_path, verbose=verbose)
        sys.exit(0 if success else 1)

    # Handle --setup (RunPod endpoint setup)
    if args.setup:
        result = setup_runpod(gpu_id=args.setup_gpu, verbose=verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)
        sys.exit(0)

    # Validate required arguments for processing
    if not args.input:
        print("Error: --input is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)
    if not args.region and not args.mask and not args.preset:
        print("Error: --region, --preset, or --mask is required", file=sys.stderr)
        print(f"Available presets: {', '.join(WATERMARK_PRESETS.keys())}", file=sys.stderr)
        sys.exit(1)

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # RunPod cloud processing
    if args.runpod:
        # Get video info for smart resize ratio suggestion
        video_info = get_video_info(args.input)
        video_width = video_info.get("width", 1280)
        video_height = video_info.get("height", 720)
        video_duration = video_info.get("duration", 0)

        # Resolve preset to region if needed
        region = args.region
        if args.preset and not region:
            region = resolve_preset_region(args.preset, video_width, video_height)
            if not region:
                print(f"Error: Could not resolve preset '{args.preset}' for {video_width}x{video_height}", file=sys.stderr)
                sys.exit(1)
            if verbose:
                print(f"Using preset '{args.preset}': region {region}", file=sys.stderr)

        # Determine resize ratio
        resize_ratio = args.resize_ratio
        if resize_ratio == "auto":
            suggested_ratio, reason = suggest_resize_ratio(video_duration, video_width, video_height)
            resize_ratio = suggested_ratio
            if verbose:
                print(f"Auto resize-ratio: {resize_ratio} ({reason})", file=sys.stderr)
        else:
            try:
                resize_ratio = float(resize_ratio)
            except ValueError:
                resize_ratio = 0.5  # Safe default

        if args.dry_run:
            config = get_runpod_config()
            result = {
                "dry_run": True,
                "mode": "runpod",
                "input": args.input,
                "output": args.output,
                "preset": args.preset,
                "region": region,
                "mask": args.mask,
                "video_dimensions": f"{video_width}x{video_height}",
                "video_duration": f"{video_duration:.1f}s",
                "resize_ratio": resize_ratio,
                "upscale": args.upscale,
                "endpoint_configured": bool(config.get("endpoint_id")),
                "api_key_configured": bool(config.get("api_key")),
                "timeout": args.runpod_timeout,
            }
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("Would process with RunPod:")
                for k, v in result.items():
                    print(f"  {k}: {v}")
            return

        if verbose:
            print("Processing with RunPod cloud GPU...")

        result = process_with_runpod(
            input_path=args.input,
            output_path=args.output,
            region=region,
            mask_path=args.mask,
            timeout=args.runpod_timeout,
            verbose=verbose,
            resize_ratio=resize_ratio,
            upscale=args.upscale,
            original_width=video_width,
            original_height=video_height,
        )

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Watermark removed: {result['output']}")
            print(f"Processing time: {result.get('processing_time_seconds', 0):.1f}s")

        return

    # Check ProPainter installation
    status = check_propainter_installed(propainter_path)
    if not status["installed"]:
        print("ProPainter is not installed.", file=sys.stderr)
        print(f"Run: python tools/dewatermark.py --install", file=sys.stderr)
        print()
        print("This is an OPTIONAL COMPONENT that requires:")
        print("  - ~2GB disk space for model weights")
        print("  - PyTorch with MPS/CUDA support")

        # Don't prompt in dry-run or non-interactive mode
        if args.dry_run:
            print()
            print("(Dry-run mode - skipping installation prompt)")
            sys.exit(0)

        # Check if running interactively
        if not sys.stdin.isatty():
            print()
            print("Run interactively or use --install to set up ProPainter")
            sys.exit(1)

        print()
        response = input("Install now? [y/N] ")
        if response.lower() == 'y':
            if not install_propainter(propainter_path, verbose=True):
                sys.exit(1)
        else:
            sys.exit(1)

    # Warn about unsupported hardware
    if status.get("mps_available") and not status.get("cuda_available"):
        print()
        print("=" * 70)
        print("⚠️  WARNING: Apple Silicon (MPS) is not supported")
        print("=" * 70)
        print()
        print("ProPainter's optical flow is extremely slow on MPS:")
        print("  - 5 seconds of video takes 4+ hours")
        print("  - A 3-minute video would take 40+ hours")
        print()
        print("Recommended alternatives:")
        print("  - Use a cloud GPU service (RunPod ~$0.30/video, Vast.ai)")
        print("  - Use a machine with NVIDIA GPU")
        print()
        print("See docs/optional-components.md for details.")
        print("=" * 70)
        print()
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Get video dimensions
    dimensions = get_video_dimensions(args.input)
    if not dimensions:
        print(f"Error: Could not read video dimensions", file=sys.stderr)
        sys.exit(1)

    video_width, video_height = dimensions

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="dewatermark_")

    try:
        # Prepare mask
        if args.mask:
            mask_path = args.mask
            if not Path(mask_path).exists():
                print(f"Error: Mask file not found: {mask_path}", file=sys.stderr)
                sys.exit(1)
        else:
            # Create mask from region
            mask_path = str(Path(temp_dir) / "mask.png")
            if verbose:
                print(f"Creating mask from region: {args.region}")
            if not create_mask_from_region(args.region, video_width, video_height, mask_path):
                sys.exit(1)

        # Determine settings for dry-run display
        if args.auto:
            compute = detect_compute_device()
            profile = get_memory_profile(compute["memory_gb"])
            dry_run_settings = {
                "auto_detected": compute["description"],
                "profile": profile["description"],
                "subvideo_length": profile["subvideo_length"],
                "neighbor_length": profile["neighbor_length"],
                "ref_stride": profile["ref_stride"],
            }
        else:
            dry_run_settings = {
                "subvideo_length": args.subvideo_length,
                "neighbor_length": args.neighbor_length,
                "ref_stride": args.ref_stride,
            }

        # Get video info for dry-run display
        video_info = get_video_info(args.input)

        # Dry run
        if args.dry_run:
            # Calculate splitting info
            if args.auto and video_info:
                compute = detect_compute_device()
                available_memory = compute["memory_gb"] or 16
                estimated_memory = estimate_frame_memory_gb(
                    video_info["width"], video_info["height"], video_info["frame_count"]
                )
                max_duration = calculate_max_duration(
                    available_memory, video_info["width"], video_info["height"], video_info["fps"],
                    device=compute.get("device", "cpu")
                )
                needs_split = video_info["duration"] > max_duration
                chunk_count = max(1, int(video_info["duration"] / (max_duration * 0.9 - args.overlap)) + 1) if needs_split else 1
            else:
                estimated_memory = None
                max_duration = None
                needs_split = False
                chunk_count = 1

            result = {
                "dry_run": True,
                "input": args.input,
                "output": args.output,
                "mask": mask_path,
                "region": args.region,
                "video_dimensions": f"{video_width}x{video_height}",
                "video_duration": f"{video_info['duration']:.1f}s" if video_info else "unknown",
                "video_frames": video_info["frame_count"] if video_info else "unknown",
                "precision": "fp32" if args.fp32 else "fp16",
                "propainter_path": str(propainter_path),
                "device": "MPS" if status["mps_available"] else "CUDA" if status["cuda_available"] else "CPU",
                **dry_run_settings,
            }

            if args.auto and video_info:
                result["estimated_memory_gb"] = f"{estimated_memory:.1f}"
                result["max_chunk_duration"] = f"{max_duration:.1f}s"
                result["needs_splitting"] = needs_split
                if needs_split:
                    result["estimated_chunks"] = chunk_count

            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("Would process:")
                for k, v in result.items():
                    print(f"  {k}: {v}")
            return

        # Determine settings (auto-detect or manual)
        if args.auto:
            compute = detect_compute_device()
            profile = get_memory_profile(compute["memory_gb"])
            neighbor_length = profile["neighbor_length"]
            ref_stride = profile["ref_stride"]
            subvideo_length = profile["subvideo_length"]
            available_memory = compute["memory_gb"]
            if verbose:
                print(f"Detected: {compute['description']}")
                print(f"Using {profile['description']} profile: subvideo={subvideo_length}, neighbor={neighbor_length}, ref_stride={ref_stride}")
        else:
            neighbor_length = args.neighbor_length
            ref_stride = args.ref_stride
            subvideo_length = args.subvideo_length
            # Estimate available memory for manual mode
            compute = detect_compute_device()
            available_memory = compute["memory_gb"] or 16

        # Get video info and check if splitting is needed
        video_info = get_video_info(args.input)
        if not video_info:
            print("Error: Could not read video info", file=sys.stderr)
            sys.exit(1)

        estimated_memory = estimate_frame_memory_gb(
            video_info["width"],
            video_info["height"],
            video_info["frame_count"],
        )
        max_duration = calculate_max_duration(
            available_memory,
            video_info["width"],
            video_info["height"],
            video_info["fps"],
            device=compute.get("device", "cpu"),
        )

        needs_splitting = video_info["duration"] > max_duration and not args.no_split

        if verbose:
            print(f"Video: {video_info['duration']:.1f}s, {video_info['frame_count']} frames at {video_info['fps']:.1f}fps")
            print(f"Estimated memory for frames: {estimated_memory:.1f}GB")
            print(f"Available memory: {available_memory}GB")
            print(f"Max duration per chunk: {max_duration:.1f}s")

        if needs_splitting:
            # Auto-split mode
            chunk_duration = max_duration * 0.9  # 90% of max for safety margin
            if verbose:
                print()
                print(f"Video exceeds memory limit - auto-splitting into ~{chunk_duration:.0f}s chunks with {args.overlap}s overlap")

            # Create chunks directory
            chunks_dir = str(Path(temp_dir) / "chunks")
            Path(chunks_dir).mkdir(parents=True, exist_ok=True)

            # Split video
            if verbose:
                print("Splitting video...")
            chunks = split_video_with_overlap(
                args.input,
                chunks_dir,
                chunk_duration,
                overlap=args.overlap,
                verbose=verbose,
            )

            if not chunks:
                print("Error: Failed to split video", file=sys.stderr)
                sys.exit(1)

            if verbose:
                print(f"\nProcessing {len(chunks)} chunks...")

            # Process each chunk
            for i, chunk in enumerate(chunks):
                if verbose:
                    print(f"\n--- Chunk {i+1}/{len(chunks)} ({chunk['start']:.1f}s - {chunk['end']:.1f}s) ---")

                chunk_output_dir = str(Path(temp_dir) / f"results_{i:03d}")
                result_path = run_propainter(
                    propainter_path,
                    chunk["input_path"],
                    mask_path,
                    chunk_output_dir,
                    fp16=not args.fp32,
                    neighbor_length=neighbor_length,
                    ref_stride=ref_stride,
                    subvideo_length=subvideo_length,
                    verbose=verbose,
                )

                if not result_path:
                    print(f"Error: Failed to process chunk {i}", file=sys.stderr)
                    sys.exit(1)

                chunk["output_path"] = result_path

            # Concatenate chunks
            if verbose:
                print("\n--- Concatenating chunks ---")

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if not concatenate_chunks(chunks, str(output_path), verbose=verbose):
                print("Error: Failed to concatenate chunks", file=sys.stderr)
                sys.exit(1)

        else:
            # Single video mode (original behavior)
            if estimated_memory > available_memory and not args.no_split:
                print(f"Warning: Video may exceed memory ({estimated_memory:.1f}GB > {available_memory}GB)")
                print("Consider using --auto for auto-splitting, or --no-split to proceed anyway")

            output_dir = str(Path(temp_dir) / "results")
            result_path = run_propainter(
                propainter_path,
                args.input,
                mask_path,
                output_dir,
                fp16=not args.fp32,
                neighbor_length=neighbor_length,
                ref_stride=ref_stride,
                subvideo_length=subvideo_length,
                verbose=verbose,
            )

            if not result_path:
                print("Error: ProPainter processing failed", file=sys.stderr)
                sys.exit(1)

            # Move result to output path
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(result_path, str(output_path))

        # Output result
        result = {
            "success": True,
            "input": args.input,
            "output": str(output_path),
            "region": args.region,
            "precision": "fp32" if args.fp32 else "fp16",
        }
        if needs_splitting:
            result["chunks"] = len(chunks)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if needs_splitting:
                print(f"\nWatermark removed (processed in {len(chunks)} chunks): {output_path}")
            else:
                print(f"Watermark removed: {output_path}")

    finally:
        # Cleanup
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
