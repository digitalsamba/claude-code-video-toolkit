# RunPod Serverless handler for Wan2.2 Image-to-Video with LightX2V acceleration
#
# Build: docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-wan-i2v:latest --push .
#
# Image size: ~40GB (includes pre-baked model weights for fast cold starts)
#
# Version: 1.0.0 - CUDA 12.4, PyTorch 2.5, LightX2V + Wan2.2-I2V-A14B
#
# GPU Requirements:
#   - Minimum: 24GB VRAM (L4, RTX 4090) with aggressive offloading
#   - Recommended: 48GB VRAM (A6000, A40)
#   - Native: 80GB VRAM (A100, H100)

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    git \
    git-lfs \
    ffmpeg \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Initialize git-lfs
RUN git lfs install

WORKDIR /app

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Try to install flash-attn for optimized attention (optional - falls back to SDPA)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, will use SDPA fallback"

# Install LightX2V from source
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V.git /app/lightx2v
WORKDIR /app/lightx2v
RUN pip3 install --no-cache-dir -v .

# Install additional dependencies for video generation
RUN pip3 install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    einops>=0.7.0 \
    Pillow>=10.0.0 \
    numpy>=1.26.0 \
    imageio>=2.31.0 \
    imageio-ffmpeg>=0.4.8

# Install RunPod SDK and utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0

WORKDIR /app

# Create models directory
RUN mkdir -p /models/wan-i2v

# Download Wan2.2-I2V-A14B model (~35GB)
# MoE (Mixture of Experts) architecture - 14B active parameters
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Wan2.2-I2V-A14B model (this will take a while)...'); \
snapshot_download( \
    repo_id='Wan-AI/Wan2.2-I2V-A14B', \
    local_dir='/models/wan-i2v', \
    ignore_patterns=['*.md', '*.txt', '.gitattributes'] \
); \
print('Wan2.2 model downloaded successfully'); \
"

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV LIGHTX2V_PATH=/app/lightx2v
ENV MODEL_PATH=/models/wan-i2v

# Health check - verify imports and model paths
RUN python3 -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
from lightx2v import LightX2VPipeline; \
print('LightX2V import OK'); \
from pathlib import Path; \
print(f'Model path exists: {Path(\"/models/wan-i2v\").exists()}'); \
"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
