# RunPod Serverless handler for character animation (SVD + SAM2)
#
# Build: docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-animate:latest --push .
#
# Image size: ~15GB (includes pre-baked model weights for fast cold starts)
#
# Version: 1.0.0 - CUDA 12.4, PyTorch 2.5, SVD-XT + SAM2

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    git \
    ffmpeg \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Set CUDA environment for SAM2 compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install PyTorch with CUDA 12.4 support (SAM2 requires 2.5.1+)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install diffusers and dependencies for SVD
RUN pip3 install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    Pillow>=10.0.0 \
    numpy>=1.26.0

# Clone and install SAM2
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /app/sam2
WORKDIR /app/sam2
RUN pip3 install -e .

# Download SAM2 model weights (~2.5GB)
# Using sam2.1_hiera_large for best quality
RUN mkdir -p /app/sam2/checkpoints && \
    echo "Downloading SAM2 checkpoint..." && \
    curl -L -o /app/sam2/checkpoints/sam2.1_hiera_large.pt \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" && \
    echo "SAM2 checkpoint downloaded"

# Pre-download SVD-XT model weights (~5GB)
# This bakes the model into the image for ~30s cold starts vs ~3min
WORKDIR /app
RUN python3 -c "\
from diffusers import StableVideoDiffusionPipeline; \
import torch; \
print('Downloading SVD-XT model...'); \
pipe = StableVideoDiffusionPipeline.from_pretrained( \
    'stabilityai/stable-video-diffusion-img2vid-xt', \
    torch_dtype=torch.float16, \
    variant='fp16' \
); \
print('SVD-XT model downloaded successfully'); \
"

# Install RunPod SDK and additional utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0 \
    opencv-python-headless>=4.8.0

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES here - RunPod sets this dynamically

# Health check - verify CUDA and models are available
RUN python3 -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}'); \
from sam2.build_sam import build_sam2; \
print('SAM2 import OK'); \
from diffusers import StableVideoDiffusionPipeline; \
print('Diffusers import OK'); \
"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
