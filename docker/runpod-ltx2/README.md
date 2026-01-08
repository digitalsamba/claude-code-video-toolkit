# LTX-2 RunPod Serverless Worker

Generate AI video from text prompts or images using LTX-2 19B model.

## Features

- Text-to-video generation from natural language prompts
- Image-to-video animation from source images
- FP8 quantization for efficient GPU memory usage
- Two flow modes: Pro (quality) and Fast (speed)
- R2 integration for result storage

## Build

```bash
# Build image (for amd64)
docker buildx build --platform linux/amd64 -t video-toolkit-ltx2 .

# Tag for GHCR
docker tag video-toolkit-ltx2 ghcr.io/conalmullan/video-toolkit-ltx2:latest

# Push to registry
docker push ghcr.io/conalmullan/video-toolkit-ltx2:latest
```

**Note:** First build downloads ~55GB of model files from HuggingFace.

## Deploy on RunPod

1. Create a new serverless template with:
   - Image: `ghcr.io/conalmullan/video-toolkit-ltx2:latest`
   - Container disk: 20GB
   - **Network Volume:** 100GB+ (for model caching)
   - GPU: RTX 4090 24GB or A6000 48GB

2. Mount network volume at `/runpod-volume`

3. Create endpoint from template

4. Note the endpoint ID

**Important:** Network volume is required for model caching. Without it, models (~55GB) download on every cold start (~10-15 minutes).

## API

### Input

```json
{
  "input": {
    "prompt": "A serene mountain landscape with snow-capped peaks",
    "width": 768,
    "height": 512,
    "num_frames": 121,
    "fps": 24,
    "flow": "pro",
    "r2": {
      "endpoint_url": "https://...",
      "access_key_id": "...",
      "secret_access_key": "...",
      "bucket_name": "..."
    }
  }
}
```

**Required:**
- `prompt` - Text description of the video to generate

**Optional image input** (for image-to-video):
- `image_url` - URL to download source image
- `image_base64` - Base64 encoded source image

**Generation options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 768 | Video width (must be divisible by 32) |
| `height` | 512 | Video height (must be divisible by 32) |
| `num_frames` | 121 | Number of frames (~5s at 24fps) |
| `fps` | 24 | Frame rate |
| `flow` | "pro" | "pro" (50 steps) or "fast" (25 steps) |
| `num_inference_steps` | 50/25 | Denoising steps (auto-set by flow) |
| `guidance_scale` | 3.0 | CFG guidance scale |
| `seed` | random | Random seed for reproducibility |
| `negative_prompt` | (default) | Negative prompt for quality |

### Output

```json
{
  "success": true,
  "video_url": "https://presigned-r2-url...",
  "r2_key": "ltx2/results/job_xxx.mp4",
  "width": 768,
  "height": 512,
  "num_frames": 121,
  "fps": 24,
  "duration_seconds": 5.04,
  "seed": 1234567890,
  "processing_time_seconds": 45.2
}
```

### Error Response

```json
{
  "error": "Error description"
}
```

## Flow Modes

| Mode | Steps | Speed | Quality | Best For |
|------|-------|-------|---------|----------|
| `pro` | 50 | Slower | Higher | Final renders, production |
| `fast` | 25 | 2x faster | Good | Prototyping, iteration |

## Resolution Presets

| Preset | Width | Height | Aspect |
|--------|-------|--------|--------|
| Default | 768 | 512 | 3:2 |
| Portrait | 512 | 768 | 2:3 |
| HD | 1280 | 720 | 16:9 |
| Square | 512 | 512 | 1:1 |

**Note:** Dimensions must be divisible by 32.

## GPU Requirements

| GPU | VRAM | Notes |
|-----|------|-------|
| RTX 4090 | 24GB | Minimum for FP8 |
| A6000 | 48GB | Recommended, comfortable headroom |
| A100 | 80GB | Best performance |

## Cost Estimates

| Duration | Frames | Pro Time | Fast Time | Cost (4090) |
|----------|--------|----------|-----------|-------------|
| 5s | 121 | ~45s | ~25s | ~$0.03 |
| 10s | 241 | ~90s | ~50s | ~$0.06 |
| 20s | 481 | ~180s | ~100s | ~$0.12 |

Cold start adds ~60-90s (with cached models) or ~10-15min (first run).

## Prompting Tips

**Good prompts:**
- Describe the scene, camera movement, and action
- Include lighting and atmosphere details
- Be specific about style (cinematic, documentary, etc.)

**Example prompts:**
```
"A majestic eagle soaring over snow-capped mountains at golden hour,
cinematic drone shot, smooth camera movement"

"A woman with long brown hair walking through autumn forest,
leaves falling around her, soft natural lighting"

"Futuristic city at night with flying cars and neon lights,
rain falling, cyberpunk aesthetic, slow pan"
```

**Negative prompt (default):**
```
worst quality, inconsistent motion, blurry, jittery, distorted,
low resolution, watermark, text
```

## Local Testing

```bash
# Run container with GPU
docker run --gpus all \
  -v /path/to/cache:/runpod-volume \
  -p 8000:8000 \
  video-toolkit-ltx2

# Test text-to-video
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cat playing with a ball of yarn",
      "width": 512,
      "height": 512,
      "num_frames": 49,
      "flow": "fast"
    }
  }'
```

## Troubleshooting

### Out of memory
- Use `fast` flow (fewer steps)
- Reduce resolution
- Reduce `num_frames`

### Slow cold start
- Ensure network volume is mounted for model caching
- First run downloads ~55GB from HuggingFace

### Blurry or low quality
- Use `pro` flow
- Increase resolution
- Improve prompt specificity

### Inconsistent motion
- Try different seeds
- Adjust guidance_scale (2.5-4.0 range)
- Simplify prompt

## Model Information

- **Model:** Lightricks/LTX-2 (19B parameters)
- **Variant:** FP8 quantized (27GB)
- **Architecture:** DiT (Diffusion Transformer)
- **License:** ltx-2-community-license-agreement

## References

- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Paper](https://arxiv.org/abs/2601.03233)
