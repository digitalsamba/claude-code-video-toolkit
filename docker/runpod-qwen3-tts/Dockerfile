# Qwen3-TTS RunPod Serverless Worker (models baked in)
# Generates speech from text with voice cloning and emotion control
#
# Build (from repo root):
#   docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-qwen3-tts:latest --push docker/runpod-qwen3-tts/
#
# Image size: ~19GB (includes ~11GB model weights: CustomVoice + Base + VoiceDesign)
# Cold start: ~30s (no model download needed)
#
# Test locally:
#   docker run --gpus all -p 8000:8000 ghcr.io/conalmullan/video-toolkit-qwen3-tts:latest

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install ffmpeg for WAV→MP3 conversion
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install torchaudio (matching PyTorch 2.4.0 from base image)
RUN pip install --no-cache-dir torchaudio==2.4.0

# Install qwen-tts and dependencies
RUN pip install --no-cache-dir \
    qwen-tts \
    soundfile \
    runpod \
    boto3 \
    requests

WORKDIR /app

# Set HF cache location
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# === BAKE MODELS INTO IMAGE ===
# Download all three Qwen3-TTS models during build (~11GB total)
RUN python -c "\
from huggingface_hub import snapshot_download; \
import os; \
os.makedirs('/root/.cache/huggingface', exist_ok=True); \
print('Downloading Qwen3-TTS CustomVoice model...'); \
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', cache_dir='/root/.cache/huggingface'); \
print('Downloading Qwen3-TTS Base model...'); \
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', cache_dir='/root/.cache/huggingface'); \
print('Downloading Qwen3-TTS VoiceDesign model...'); \
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', cache_dir='/root/.cache/huggingface'); \
print('All three models downloaded successfully'); \
"

# Copy handler
COPY handler.py /app/handler.py

# Start handler
CMD ["python", "-u", "/app/handler.py"]
