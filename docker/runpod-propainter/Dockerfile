# RunPod Serverless handler for ProPainter (dewatermark)
#
# Build: docker build -t yourusername/video-toolkit-propainter:latest .
# Push:  docker push yourusername/video-toolkit-propainter:latest
#
# Image size: ~4GB (includes pre-baked model weights for fast cold starts)
#
# Version: 2.0.0 - CUDA 12.4, PyTorch 2.4, fixed GPU detection

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone ProPainter repository
RUN git clone --depth 1 https://github.com/sczhou/ProPainter.git /app/propainter

# Install ProPainter requirements
# PyTorch 2.4+ has better numpy 2.x compatibility, but ProPainter may still need numpy 1.x
WORKDIR /app/propainter
RUN pip3 install --no-cache-dir "numpy>=1.26,<2" && \
    pip3 install --no-cache-dir -r requirements.txt

# Pre-download model weights (baked into image for ~30s cold starts vs ~2min)
# Weights are ~2GB total
RUN mkdir -p /app/propainter/weights && \
    echo "Downloading ProPainter.pth..." && \
    curl -L -o /app/propainter/weights/ProPainter.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth" && \
    echo "Downloading recurrent_flow_completion.pth..." && \
    curl -L -o /app/propainter/weights/recurrent_flow_completion.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth" && \
    echo "Downloading raft-things.pth..." && \
    curl -L -o /app/propainter/weights/raft-things.pth \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth" && \
    echo "Downloading i3d_rgb_imagenet.pt..." && \
    curl -L -o /app/propainter/weights/i3d_rgb_imagenet.pt \
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt" && \
    echo "All weights downloaded successfully"

# Install RunPod SDK and additional utilities
RUN pip3 install --no-cache-dir \
    runpod>=1.7.0 \
    requests>=2.31.0 \
    boto3>=1.34.0

# Copy handler
WORKDIR /app
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
# NOTE: Do NOT set CUDA_VISIBLE_DEVICES here - RunPod sets this dynamically
# to assign the correct GPU to each serverless worker

# Health check - verify CUDA is available
RUN python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Run handler
CMD ["python3", "-u", "/app/handler.py"]
