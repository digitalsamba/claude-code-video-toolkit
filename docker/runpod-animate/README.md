# RunPod Animate Docker Image

Serverless GPU handler for character animation using SAM2 (segmentation) and Stable Video Diffusion (image-to-video).

## Quick Start

### Option A: Use Pre-built Public Image (Recommended)

A public image will be available on GitHub Container Registry:

```
ghcr.io/conalmullan/video-toolkit-animate:latest
```

Skip to **Step 2: Deploy on RunPod** below.

### Option B: Build Your Own Image

```bash
cd docker/runpod-animate

# Build for linux/amd64 (required for RunPod)
docker buildx build --platform linux/amd64 -t ghcr.io/yourusername/video-toolkit-animate:latest --push .
```

Build takes ~25-30 minutes (downloads ~7GB of model weights).

**Important: Make the image public**

GHCR images are private by default. RunPod cannot pull private images.

1. Go to https://github.com/users/yourusername/packages/container/video-toolkit-animate/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → Select "Public" → Confirm

### Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `ghcr.io/conalmullan/video-toolkit-animate:latest`
   - **GPU**: RTX 4090 or A6000 (24GB+ VRAM recommended)
   - **Max Workers**: 1 (scale up as needed)
   - **Idle Timeout**: 5 seconds (fast scale-down)
   - **Execution Timeout**: 600 seconds (10 min max per job)
4. Copy the **Endpoint ID** for your `.env` file

### Configure Local Tool

Add to your `.env`:

```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ANIMATE_ENDPOINT_ID=your_endpoint_id_here
```

### Use It

```bash
# Analyze video and create manifest
python tools/animate.py --input video.mp4 --analyze --output manifest.json

# (Edit manifest to define elements)

# Run animation pipeline
python tools/animate.py --input video.mp4 --manifest manifest.json --output animated.mp4 --runpod
```

## Image Details

| Property | Value |
|----------|-------|
| Base | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Size | ~15GB |
| Cold Start | ~45 seconds |
| Python | 3.10 |
| PyTorch | 2.5.1 + CUDA 12.4 |

### Pre-baked Components

- SAM2 (Segment Anything 2)
  - sam2.1_hiera_large checkpoint (~2.5GB)
- Stable Video Diffusion
  - SVD-XT model (~5GB, fp16)
- FFmpeg for video processing
- RunPod SDK

## API Reference

### Segment Operation

Generate masks from bounding boxes or points using SAM2.

**Input:**
```json
{
    "input": {
        "operation": "segment",
        "image_url": "https://example.com/frame.png",
        "bboxes": [[100, 150, 400, 600]],
        "r2": {
            "endpoint_url": "https://...",
            "access_key_id": "...",
            "secret_access_key": "...",
            "bucket_name": "..."
        }
    }
}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `operation` | Yes | Must be `"segment"` |
| `image_url` | Yes | Direct URL to image |
| `bboxes` | One of | List of `[x1, y1, x2, y2]` bounding boxes |
| `points` | One of | List of `[x, y]` points |
| `point_labels` | With points | `1` = foreground, `0` = background |
| `r2` | No | R2 config for result upload |

**Output:**
```json
{
    "success": true,
    "masks": [
        {
            "url": "https://...",
            "score": 0.98,
            "bbox": [100, 150, 400, 600]
        }
    ],
    "mask_count": 1,
    "image_size": [1920, 1080],
    "processing_time_seconds": 2.5
}
```

### Animate Operation

Generate video from image using Stable Video Diffusion.

**Input:**
```json
{
    "input": {
        "operation": "animate",
        "image_url": "https://example.com/frame.png",
        "mask_url": "https://example.com/mask.png",
        "motion_bucket_id": 127,
        "noise_aug_strength": 0.02,
        "num_frames": 25,
        "fps": 6,
        "seed": 42
    }
}
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `operation` | Yes | - | Must be `"animate"` |
| `image_url` | Yes | - | Direct URL to image |
| `mask_url` | No | - | Mask image (white = animate, black = static) |
| `motion_bucket_id` | No | 127 | Motion amount (1-255) |
| `noise_aug_strength` | No | 0.02 | Noise augmentation (0-1) |
| `num_frames` | No | 25 | Frames to generate |
| `fps` | No | 6 | Output video FPS |
| `seed` | No | random | Random seed for reproducibility |
| `r2` | No | - | R2 config for result upload |

**Output:**
```json
{
    "success": true,
    "output_url": "https://...",
    "original_size": [1920, 1080],
    "processed_size": [1024, 576],
    "num_frames": 25,
    "fps": 6,
    "motion_bucket_id": 127,
    "processing_time_seconds": 45.2
}
```

## Motion Parameters

SVD uses "micro-conditioning" parameters to control motion:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `motion_bucket_id` | 1-255 | Amount of motion. Lower = subtle, Higher = dramatic |
| `noise_aug_strength` | 0.0-1.0 | Image variation. Lower = faithful, Higher = creative |

**Recommended settings for character animation:**

| Scene Type | motion_bucket_id | noise_aug_strength |
|------------|------------------|-------------------|
| Subtle breathing | 80-100 | 0.01-0.02 |
| Gentle movement | 120-150 | 0.02-0.04 |
| Active motion | 180-220 | 0.04-0.06 |

## GPU Memory Requirements

| GPU VRAM | Segment | Animate | Notes |
|----------|---------|---------|-------|
| 16GB | Yes | Yes* | *Uses CPU offload, slower |
| 24GB | Yes | Yes | Recommended |
| 48GB | Yes | Yes | Fastest |

**Recommendation**: Use RTX 4090 or A6000 (24GB+) for best performance.

## Cost Estimates

Using RTX 4090 (~$0.69/hr):

| Operation | Time | Cost |
|-----------|------|------|
| Segment (1 bbox) | ~3s | ~$0.001 |
| Segment (5 bboxes) | ~8s | ~$0.002 |
| Animate (25 frames) | ~45s | ~$0.009 |
| Animate (14 frames) | ~30s | ~$0.006 |

A typical video with 10 scenes, 2 elements each:
- 20 segment operations: ~$0.02
- 20 animate operations: ~$0.18
- **Total: ~$0.20 per video**

## Local Testing

Test the image locally with NVIDIA GPU:

```bash
# Build
docker build -t animate-test .

# Run interactive shell
docker run --gpus all -it animate-test /bin/bash

# Inside container, test GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test SAM2
python3 -c "from sam2.build_sam import build_sam2; print('SAM2 OK')"

# Test SVD
python3 -c "from diffusers import StableVideoDiffusionPipeline; print('SVD OK')"
```

## Troubleshooting

### "CUDA out of memory" during animate

The image or batch size is too large. The handler auto-adjusts decode_chunk_size, but you can also:
1. Use a GPU with more VRAM
2. Reduce `num_frames` to 14

### "Failed to download image"

- Check the image URL is publicly accessible
- URLs must be direct downloads (not web pages)
- Download timeout is 2 minutes

### Cold start is slow (~45s)

First request after idle loads both SAM2 and SVD models. Options:
- Set longer idle timeout (costs more)
- Use "always on" worker for frequent usage
- Pre-warm with a test request

### Mask quality is poor

For best SAM2 results:
- Use bounding boxes that tightly fit the subject
- Include some padding around the subject
- For complex shapes, use multiple points with foreground/background labels

## Extending

The handler supports adding new operations. See `handler.py` for the pattern.

Future operations might include:
- `stylize` - Apply style transfer to video
- `interpolate` - Frame interpolation for smoother motion
- `loop` - Create seamless loops from SVD output
