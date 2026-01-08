# LTX-2 RunPod Serverless Worker
# Generates AI video from text prompts or images using LTX-2 19B model
#
# Build:
#   docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-ltx2:latest .
#
# Image size: ~15GB (models downloaded at runtime to network volume)
#
# Cold start: First run ~10-15 min (downloads ~55GB models from HuggingFace)
#             Subsequent runs ~60-90s (cached on network volume)
#
# GPU Requirements:
#   - Minimum: 24GB VRAM (RTX 4090) for FP8
#   - Recommended: 48GB VRAM (A6000) for comfortable headroom
#
# Model: Lightricks/LTX-2 (19B parameters, FP8 quantized)

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

# Try to install flash-attn for optimized attention (optional)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn installation failed, will use SDPA fallback"

# Clone LTX-2 repository and install packages
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/ltx2
WORKDIR /app/ltx2

# Install ltx-core and ltx-pipelines from the monorepo
RUN pip3 install --no-cache-dir \
    ./packages/ltx-core \
    ./packages/ltx-pipelines

# Install diffusers from git (may have LTX-2 support)
RUN pip3 install --no-cache-dir git+https://github.com/huggingface/diffusers

# Install additional dependencies
RUN pip3 install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    einops>=0.7.0 \
    Pillow>=10.0.0 \
    numpy>=1.26.0 \
    opencv-python-headless>=4.9.0 \
    imageio>=2.34.0 \
    imageio-ffmpeg>=0.4.9 \
    scipy>=1.12.0 \
    tqdm>=4.66.0

# Install RunPod SDK and utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0 \
    huggingface_hub>=0.25.0

WORKDIR /app

# Set HF cache location - models download at runtime to network volume
# On RunPod, set Network Volume mount at /runpod-volume
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface
ENV LTX2_MODEL_ID=Lightricks/LTX-2

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/ltx2/packages/ltx-core:/app/ltx2/packages/ltx-pipelines

# Health check - verify torch and packages
RUN python3 -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
try: \
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline; \
    print('ltx_pipelines: available'); \
except ImportError as e: \
    print(f'ltx_pipelines import warning: {e}'); \
"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
