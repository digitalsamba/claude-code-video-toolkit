# RunPod ProPainter Docker Image

Serverless GPU handler for video watermark removal using ProPainter AI inpainting.

## Quick Start

### Option A: Use Pre-built Public Image (Recommended)

A public image is available on GitHub Container Registry:

```
ghcr.io/conalmullan/video-toolkit-propainter:latest
```

Skip to **Step 2: Deploy on RunPod** below.

### Option B: Build Your Own Image

```bash
cd docker/runpod-propainter

# Build for linux/amd64 (required for RunPod)
docker buildx build --platform linux/amd64 -t yourusername/video-toolkit-propainter:latest --push .
```

Build takes ~15-20 minutes (downloads ~2GB of model weights).

### Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `ghcr.io/conalmullan/video-toolkit-propainter:latest`
   - **GPU**: RTX 3090 or RTX 4090 (24GB VRAM recommended)
   - **Max Workers**: 1 (scale up as needed)
   - **Idle Timeout**: 5 seconds (fast scale-down)
   - **Execution Timeout**: 3600 seconds (1 hour max)
4. Copy the **Endpoint ID** for your `.env` file

### Configure Local Tool

Add to your `.env`:

```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

### Use It

```bash
python tools/dewatermark.py \
    --input video.mp4 \
    --region 1080,660,195,40 \
    --output clean.mp4 \
    --runpod
```

## Image Details

| Property | Value |
|----------|-------|
| Base | `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` |
| Size | ~4GB |
| Cold Start | ~30 seconds |
| Python | 3.10 |
| PyTorch | 2.1.0 + CUDA 12.1 |

### Pre-baked Components

- ProPainter repository
- Model weights (~2GB):
  - ProPainter.pth
  - recurrent_flow_completion.pth
  - raft-things.pth
  - i3d_rgb_imagenet.pt
- FFmpeg for video processing
- RunPod SDK

## API Reference

### Input Format

```json
{
    "input": {
        "operation": "dewatermark",
        "video_url": "https://example.com/video.mp4",
        "region": "1080,660,195,40"
    }
}
```

Or with a pre-made mask:

```json
{
    "input": {
        "operation": "dewatermark",
        "video_url": "https://example.com/video.mp4",
        "mask_url": "https://example.com/mask.png"
    }
}
```

### Output Format

```json
{
    "success": true,
    "output_url": "https://runpod-storage.../job123_dewatermarked.mp4",
    "video_dimensions": "1920x1080",
    "video_duration_seconds": 45.5,
    "gpu_vram_gb": 24,
    "profile_used": {
        "subvideo_length": 60,
        "neighbor_length": 10,
        "ref_stride": 10
    },
    "processing_time_seconds": 120.5
}
```

### Error Format

```json
{
    "error": "Description of what went wrong"
}
```

## GPU Memory Profiles

The handler auto-detects GPU VRAM and selects optimal settings:

| VRAM | subvideo_length | neighbor_length | ref_stride | Speed |
|------|-----------------|-----------------|------------|-------|
| 8GB  | 30 | 5 | 25 | Slow |
| 12GB | 40 | 5 | 20 | Medium |
| 16GB | 50 | 8 | 15 | Good |
| 24GB | 60 | 10 | 10 | Fast |
| 48GB | 80 | 10 | 10 | Fastest |

**Recommendation**: Use RTX 3090 or RTX 4090 (24GB) for best price/performance.

## Cost Estimates

Using RTX 3090 (~$0.34/hr):

| Video Length | Processing Time | Estimated Cost |
|--------------|-----------------|----------------|
| < 30 seconds | 2-5 minutes | ~$0.02 |
| 30s - 2 min | 5-15 minutes | ~$0.08 |
| 2 - 5 min | 15-45 minutes | ~$0.25 |
| > 5 min | 45+ minutes | ~$0.40+ |

## Local Testing

Test the image locally with NVIDIA GPU:

```bash
# Build
docker build -t propainter-test .

# Run interactive shell
docker run --gpus all -it propainter-test /bin/bash

# Inside container, test GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test handler with mock job
python3 -c "
from handler import handler
result = handler({
    'id': 'test123',
    'input': {
        'operation': 'dewatermark',
        'video_url': 'https://example.com/test.mp4',
        'region': '100,100,50,50'
    }
})
print(result)
"
```

## Troubleshooting

### "CUDA out of memory"

The video is too long for the GPU. Options:
1. Use a GPU with more VRAM
2. The local tool will auto-chunk long videos (coming soon)

### "Failed to download video"

- Check the video URL is publicly accessible
- URLs must be direct downloads (not web pages)
- Timeout is 5 minutes for download

### "No output file found"

ProPainter failed silently. Check:
- Video format is supported (MP4, MOV, AVI)
- Mask dimensions match video dimensions
- Region coordinates are valid

### Cold start is slow

First request after idle takes ~30s to load models. Subsequent requests are faster. Consider:
- Setting longer idle timeout (but costs more)
- Using "always on" worker for frequent usage

## Extending for New Operations

The handler is designed for extensibility. To add a new GPU operation:

1. Add handler function in `handler.py`:
   ```python
   def handle_upscale(job_input: dict, job_id: str, work_dir: Path) -> dict:
       # Implementation
       pass
   ```

2. Register in main handler:
   ```python
   if operation == "upscale":
       return handle_upscale(job_input, job_id, work_dir)
   ```

3. Rebuild and push image

Future operations might include:
- `upscale` - Video upscaling with Real-ESRGAN
- `denoise` - Audio/video denoising
- `stabilize` - Video stabilization
