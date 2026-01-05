# RunPod Qwen-Edit Docker Image

Serverless GPU handler for AI-powered image editing using Qwen-Image-Edit-2511 with LightX2V acceleration.

This is **Worker 1** in the video generation pipeline:
```
Reference Image -> [Qwen-Edit] -> Edited Frame -> [Wan I2V] -> Video Clip
```

## Quick Start

### Option A: Use Pre-built Public Image (Recommended)

```
ghcr.io/conalmullan/video-toolkit-qwen-edit:latest
```

Skip to **Step 2: Deploy on RunPod** below.

### Option B: Build Your Own Image

```bash
cd docker/runpod-qwen-edit

# Build for linux/amd64 (required for RunPod)
docker buildx build --platform linux/amd64 -t ghcr.io/yourusername/video-toolkit-qwen-edit:latest --push .
```

Build takes ~45-60 minutes (downloads ~30GB of model weights).

**Important: Make the image public**

GHCR images are private by default. RunPod cannot pull private images.

1. Go to https://github.com/users/yourusername/packages/container/video-toolkit-qwen-edit/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → Select "Public" → Confirm

### Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `ghcr.io/conalmullan/video-toolkit-qwen-edit:latest`
   - **GPU**: L4 24GB or RTX 4090 (24GB+ VRAM required)
   - **Max Workers**: 1 (scale up as needed)
   - **Idle Timeout**: 5 seconds (fast scale-down)
   - **Execution Timeout**: 120 seconds (2 min max per job)
4. Copy the **Endpoint ID** for your `.env` file

### Configure Local Tool

Add to your `.env`:

```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_QWEN_EDIT_ENDPOINT_ID=your_endpoint_id_here
```

### Use It

```bash
# Generate edited frame (via orchestration tool - coming soon)
python tools/videogen.py edit \
    --image reference.png \
    --prompt "Change background to modern office" \
    --output edited.png
```

## Image Details

| Property | Value |
|----------|-------|
| Base | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Size | ~25GB |
| Cold Start | ~60 seconds |
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.4 |

### Pre-baked Components

- **Qwen-Image-Edit-2511** base model (~20GB)
- **LightX2V Lightning FP8** weights (~10GB)
  - 4-step inference (vs 40 native)
  - FP8 quantization (50% VRAM reduction)
- **Flash Attention 3** for optimized inference
- **FFmpeg** for image processing
- **RunPod SDK**

## API Reference

### Edit Operation

Edit an image based on text prompt while preserving subject identity.

**Input:**
```json
{
    "input": {
        "image_base64": "<base64 encoded image>",
        "prompt": "Change the background to a modern office",
        "negative_prompt": "blurry, distorted",
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": 42,
        "use_fp8": true,
        "r2": {
            "endpoint_url": "https://...",
            "access_key_id": "...",
            "secret_access_key": "...",
            "bucket_name": "..."
        }
    }
}
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `image_base64` | Yes | - | Base64 encoded input image |
| `prompt` | Yes | - | Edit instruction |
| `negative_prompt` | No | "" | Things to avoid |
| `num_inference_steps` | No | 8 (FP8), 4 (LoRA) | Diffusion steps |
| `guidance_scale` | No | 1.0 | CFG scale |
| `seed` | No | random | Random seed for reproducibility |
| `use_fp8` | No | true | Use FP8 quantization (lower VRAM) |
| `auto_resize` | No | true | Auto-resize for optimal processing |
| `r2` | No | - | R2 config for result upload |

**Output:**
```json
{
    "success": true,
    "edited_image_base64": "<base64 encoded result>",
    "seed": 42,
    "inference_time_ms": 1800,
    "image_size": [1024, 1024],
    "num_inference_steps": 8,
    "use_fp8": true
}
```

If R2 config provided:
```json
{
    "success": true,
    "edited_image_base64": "<base64>",
    "output_url": "https://r2.example.com/...",
    "r2_key": "qwen-edit/results/abc123.png",
    ...
}
```

## Example Prompts

**Background changes:**
- "Change the background to a modern office"
- "Replace background with outdoor park scene"
- "Add a professional studio backdrop"

**Pose/expression adjustments:**
- "Change pose to speaking gesture"
- "Add subtle smile"
- "Subject looking slightly left"

**Lighting/style:**
- "Add professional lighting"
- "Apply warm color grading"
- "Make lighting more dramatic"

## GPU Memory Requirements

| GPU VRAM | FP8 Mode | BF16 Mode | Notes |
|----------|----------|-----------|-------|
| 24GB | Yes | Yes* | *May be tight |
| 48GB | Yes | Yes | Recommended |
| 80GB | Yes | Yes | Fastest |

**Recommendation**: Use L4, RTX 4090, or A6000 (24GB+) with FP8 mode.

## Performance

Using L4 24GB (~$0.34/hr):

| Mode | Steps | Time | Cost/Edit |
|------|-------|------|-----------|
| FP8 Lightning | 8 | ~2s | ~$0.0002 |
| LoRA Lightning | 4 | ~1.5s | ~$0.00014 |
| Native (no accel) | 40 | ~15s | ~$0.0014 |

**LightX2V acceleration provides ~10x speedup and ~50% VRAM reduction.**

## Local Testing

Test the image locally with NVIDIA GPU:

```bash
# Build
docker build -t qwen-edit-test .

# Run interactive shell
docker run --gpus all -it qwen-edit-test /bin/bash

# Inside container, test GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test LightX2V
python3 -c "from lightx2v import LightX2VPipeline; print('LightX2V OK')"

# Test model paths
python3 -c "from pathlib import Path; print(f'Model: {Path(\"/models/qwen-edit\").exists()}')"
```

## Troubleshooting

### "CUDA out of memory"

1. Ensure `use_fp8: true` is set (default)
2. Use a GPU with 24GB+ VRAM
3. Enable auto_resize to scale down large images

### "Generation failed - no output file"

Check the prompt syntax. Qwen-Image-Edit works best with:
- Clear, specific instructions
- Focus on ONE change at a time
- Avoid contradictory instructions

### Cold start is slow (~60s)

First request after idle loads the full model. Options:
- Set longer idle timeout
- Use "always on" worker for frequent usage
- Pre-warm with a test request

### Identity drift in chained edits

This is a known challenge. Mitigations:
- Use consistent seed across edits
- Keep prompts focused on single changes
- Consider IP-Adapter for explicit identity conditioning (future)

## Pipeline Integration

This worker is designed for the video generation pipeline:

```python
# Pseudo-code for pipeline orchestration
def generate_video_segment(reference_image, scenes):
    current_frame = reference_image

    for scene in scenes:
        # Worker 1: Edit frame
        edited = call_qwen_edit(current_frame, scene["edit_prompt"])

        # Worker 2: Animate (separate endpoint)
        video = call_wan_i2v(edited, scene["motion_prompt"])

        # Chain: last frame becomes next input
        current_frame = video["last_frame"]

    return concat_videos(videos)
```

See `.ai_dev/video-generation-pipeline.md` for full pipeline documentation.

## Related Workers

| Worker | Purpose | Status |
|--------|---------|--------|
| `runpod-qwen-edit` | Frame editing (this) | Phase 1 |
| `runpod-wan-i2v` | Image-to-video | Phase 2 (planned) |
| `runpod-propainter` | Watermark removal | Deployed |
| `runpod-animate` | SVD animation | Deployed |
