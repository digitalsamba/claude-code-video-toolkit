# RunPod Serverless handler for Real-ESRGAN (image upscaling)
#
# Build: docker build -t yourusername/video-toolkit-realesrgan:latest .
# Push:  docker push yourusername/video-toolkit-realesrgan:latest
#
# Image size: ~3GB (includes pre-baked model weights for fast cold starts)
#
# Version: 1.0.0 - CUDA 12.4, PyTorch 2.4, Real-ESRGAN

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1 support (older torchvision for Real-ESRGAN compatibility)
# Real-ESRGAN requires torchvision.transforms.functional_tensor which was removed in newer versions
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Real-ESRGAN and dependencies
RUN pip3 install --no-cache-dir \
    realesrgan>=0.3.0 \
    basicsr>=1.4.2 \
    facexlib>=0.3.0 \
    gfpgan>=1.3.8 \
    opencv-python>=4.8.0 \
    "numpy>=1.24.0,<2.0" \
    Pillow>=10.0.0

# Pre-download model weights (baked into image for fast cold starts)
# RealESRGAN_x4plus.pth ~64MB
# RealESRGAN_x4plus_anime_6B.pth ~17MB
# realesr-general-x4v3.pth ~65MB
RUN mkdir -p /app/weights && \
    echo "Downloading RealESRGAN_x4plus.pth..." && \
    curl -L -o /app/weights/RealESRGAN_x4plus.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" && \
    echo "Downloading RealESRGAN_x4plus_anime_6B.pth..." && \
    curl -L -o /app/weights/RealESRGAN_x4plus_anime_6B.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" && \
    echo "Downloading realesr-general-x4v3.pth..." && \
    curl -L -o /app/weights/realesr-general-x4v3.pth \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" && \
    echo "All weights downloaded successfully"

# Install RunPod SDK and additional utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1

# Health check - verify CUDA is available
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
