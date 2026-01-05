# RunPod Serverless handler for Qwen-Image-Edit with diffusers
#
# Build: docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-qwen-edit:latest --push .
#
# Image size: ~8GB (models downloaded at runtime to network volume)
#
# Version: 1.3.0 - Runtime model download (Buildjet-free)
#
# Cold start: First run ~5-10 min (downloads ~57GB model)
#             Subsequent runs ~30s (cached on network volume)
#
# GPU Requirements:
#   - Minimum: 48GB VRAM (A6000) for BF16
#   - Optimal: 80GB VRAM (A100) for best performance

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
# flash-attn compilation is notoriously fragile, so we don't fail the build if it fails
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, will use SDPA fallback"

# Install LightX2V from source (for latest fixes)
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V.git /app/lightx2v
WORKDIR /app/lightx2v
RUN pip3 install --no-cache-dir -v .

# Install diffusers from git (required for QwenImageEditPlusPipeline)
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Install additional dependencies for inference
RUN pip3 install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    einops>=0.7.0 \
    Pillow>=10.0.0 \
    numpy>=1.26.0

# Install LightX2V dependencies (from their requirements.txt)
RUN pip3 install --no-cache-dir \
    opencv-python-headless>=4.9.0 \
    imageio>=2.34.0 \
    imageio-ffmpeg>=0.4.9 \
    ftfy>=6.1.0 \
    decord>=0.6.0 \
    scipy>=1.12.0 \
    tqdm>=4.66.0 \
    tokenizers>=0.19.0

# Install sage-attention for optimized attention (required by LightX2V)
RUN pip3 install --no-cache-dir sageattention>=1.0.0 || echo "sage-attention not available"

# Install RunPod SDK and utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0 \
    loguru>=0.7.0 \
    huggingface_hub>=0.25.0 \
    prometheus_client>=0.20.0 \
    pyyaml>=6.0 \
    sentencepiece>=0.2.0 \
    tiktoken>=0.7.0 \
    qwen-vl-utils>=0.0.8 \
    gguf>=0.10.0 \
    triton>=3.0.0 \
    protobuf>=4.25.0

WORKDIR /app

# Set HF cache location - models downloaded at runtime
# On RunPod, models cache to /runpod-volume/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV LIGHTX2V_PATH=/app/lightx2v

# Health check - verify torch and diffusers
RUN python3 -c "\
import torch; \
from diffusers import QwenImageEditPlusPipeline; \
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
print('diffusers QwenImageEditPlusPipeline: available'); \
"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
