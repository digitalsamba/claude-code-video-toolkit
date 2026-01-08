# SadTalker RunPod Serverless Worker
# Generates talking head videos from image + audio
#
# Build:
#   docker build -t video-toolkit-sadtalker .
#
# Test locally:
#   docker run --gpus all -p 8000:8000 video-toolkit-sadtalker

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (cmake required for dlib, unzip for model archives)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    unzip \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create constraints file to enforce numpy<2 across ALL pip installs
# This prevents gfpgan/basicsr/facexlib from upgrading numpy to 2.0
RUN echo "numpy<2" > /app/constraints.txt

# Clone SadTalker
RUN git clone --depth 1 https://github.com/OpenTalker/SadTalker.git
WORKDIR /app/SadTalker

# Install Python dependencies with numpy constraint
# Note: Base image has PyTorch 2.1, which works with SadTalker
# Use dlib-bin (pre-compiled) to avoid building from source
RUN pip install --no-cache-dir -c /app/constraints.txt \
    dlib-bin \
    face_alignment \
    imageio \
    imageio-ffmpeg \
    kornia \
    librosa \
    pydub \
    scipy \
    tqdm \
    yacs \
    safetensors

# Install additional dependencies for face restoration (with numpy constraint)
RUN pip install --no-cache-dir -c /app/constraints.txt \
    gfpgan \
    basicsr \
    facexlib \
    realesrgan

# Install RunPod SDK and utilities
RUN pip install --no-cache-dir -c /app/constraints.txt \
    runpod \
    boto3 \
    requests

# Verify numpy version is <2
RUN python -c "import numpy; v=numpy.__version__; print(f'NumPy version: {v}'); assert v.startswith('1.'), f'Expected numpy 1.x, got {v}'"

# Patch numpy 2.0 compatibility in SadTalker AND dependencies (belt-and-suspenders)
# Replace np.float with float, np.int with int, np.VisibleDeprecationWarning with DeprecationWarning
# The \b word boundary ensures we don't replace np.float64, np.int32, etc.
RUN find /app/SadTalker -name "*.py" -exec sed -i 's/np\.float\b/float/g' {} \; && \
    find /app/SadTalker -name "*.py" -exec sed -i 's/np\.int\b/int/g' {} \; && \
    find /app/SadTalker -name "*.py" -exec sed -i 's/np\.VisibleDeprecationWarning/DeprecationWarning/g' {} \; && \
    find /usr/local/lib -name "*.py" -path "*/facexlib/*" -exec sed -i 's/np\.float\b/float/g' {} \; 2>/dev/null || true && \
    find /usr/local/lib -name "*.py" -path "*/basicsr/*" -exec sed -i 's/np\.float\b/float/g' {} \; 2>/dev/null || true && \
    find /usr/local/lib -name "*.py" -path "*/gfpgan/*" -exec sed -i 's/np\.float\b/float/g' {} \; 2>/dev/null || true

# Fix ValueError: setting an array element with a sequence (PR #859)
# Adds dtype=object to trans_params array in preprocess.py
RUN sed -i 's/trans_params = np\.array(\[w0, h0, s, t\[0\], t\[1\]\])/trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=object)/g' \
    /app/SadTalker/src/face3d/util/preprocess.py

# Download SadTalker model weights at build time (faster cold start)
# Models are ~1.5GB total
RUN mkdir -p checkpoints && \
    cd checkpoints && \
    wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar && \
    wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar && \
    wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors && \
    wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors

# Download BFM model for 3DMM (from v0.0.2 release, not v0.0.2-rc)
# Note: zip contains BFM_Fitting/ subdirectory, so unzip into checkpoints/
RUN cd checkpoints && \
    wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip && \
    unzip -q BFM_Fitting.zip && \
    rm BFM_Fitting.zip && \
    rm -rf __MACOSX

# Download hub models for face detection (face_alignment uses these)
RUN wget -q https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/hub.zip && \
    unzip -q hub.zip -d /root/.cache/torch/ && \
    rm hub.zip

# Download GFPGAN weights
RUN mkdir -p gfpgan/weights && \
    wget -q -P gfpgan/weights https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# Download facexlib detection models (used by GFPGAN)
RUN mkdir -p /root/.cache/facexlib && \
    wget -q -O /root/.cache/facexlib/detection_Resnet50_Final.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    wget -q -O /root/.cache/facexlib/parsing_parsenet.pth \
        https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

WORKDIR /app

# Copy handler
COPY handler.py /app/handler.py

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/SadTalker

# Start handler
CMD ["python", "-u", "/app/handler.py"]
