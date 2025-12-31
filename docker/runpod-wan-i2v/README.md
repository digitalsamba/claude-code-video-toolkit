# RunPod Wan2.2 I2V Docker Image

Serverless GPU handler for AI video generation using Wan2.2 Image-to-Video with LightX2V acceleration.

This is **Worker 2** in the video generation pipeline:
```
Reference Image -> [Qwen-Edit] -> Edited Frame -> [Wan I2V] -> Video Clip
```

## Quick Start

### Option A: Use Pre-built Public Image (Recommended)

```
ghcr.io/conalmullan/video-toolkit-wan-i2v:latest
```

Skip to **Deploy on RunPod** below.

### Option B: Build Your Own Image

```bash
cd docker/runpod-wan-i2v

# Build for linux/amd64 (required for RunPod)
docker buildx build --platform linux/amd64 -t ghcr.io/yourusername/video-toolkit-wan-i2v:latest --push .
```

Build takes ~60-90 minutes (downloads ~35GB of model weights).

**Important: Make the image public**

GHCR images are private by default. RunPod cannot pull private images.

1. Go to https://github.com/users/yourusername/packages/container/video-toolkit-wan-i2v/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → Select "Public" → Confirm

### Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Docker Image**: `ghcr.io/conalmullan/video-toolkit-wan-i2v:latest`
   - **GPU**: A6000 48GB recommended (24GB works with offloading)
   - **Max Workers**: 1 (scale up as needed)
   - **Idle Timeout**: 5 seconds
   - **Execution Timeout**: 600 seconds (10 min max)
4. Copy the **Endpoint ID** for your `.env` file

### Configure Local Tool

Add to your `.env`:

```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_WAN_I2V_ENDPOINT_ID=your_endpoint_id_here
```

## Image Details

| Property | Value |
|----------|-------|
| Base | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Size | ~40GB |
| Cold Start | ~90 seconds |
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.4 |

### Pre-baked Components

- **Wan2.2-I2V-A14B** model (~35GB)
  - MoE (Mixture of Experts) architecture
  - 14B active parameters
  - 65% more training data than Wan2.1
- **LightX2V** for optimized inference
- **Flash Attention** for memory efficiency
- **FFmpeg** for video encoding
- **RunPod SDK**

## API Reference

### I2V Operation

Generate video from a starting image based on motion description.

**Input:**
```json
{
    "input": {
        "image_base64": "<base64 encoded image>",
        "prompt": "Person speaking naturally with subtle hand gestures",
        "negative_prompt": "blurry, distorted, static",
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_scale": 5.0,
        "seed": 42,
        "fps": 16,
        "height": 480,
        "width": 832,
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
| `image_base64` | Yes | - | Base64 encoded starting frame |
| `prompt` | Yes | - | Motion description |
| `negative_prompt` | No | "" | Things to avoid |
| `num_frames` | No | 81 | Frames to generate (~5 sec @ 16fps) |
| `num_inference_steps` | No | 40 | Diffusion steps |
| `guidance_scale` | No | 5.0 | CFG scale |
| `seed` | No | random | Random seed |
| `fps` | No | 16 | Output video FPS |
| `height` | No | 480 | Output height |
| `width` | No | 832 | Output width |
| `r2` | No | - | R2 config for result upload |

**Output:**
```json
{
    "success": true,
    "video_base64": "<base64 encoded MP4>",
    "last_frame_base64": "<base64 encoded PNG>",
    "seed": 42,
    "inference_time_ms": 35000,
    "num_frames": 81,
    "fps": 16,
    "resolution": "832x480",
    "video_size_bytes": 524288
}
```

If R2 config provided:
```json
{
    "success": true,
    "video_base64": "<base64>",
    "video_url": "https://r2.example.com/...",
    "video_r2_key": "wan-i2v/results/abc123_video.mp4",
    "last_frame_url": "https://r2.example.com/...",
    "last_frame_r2_key": "wan-i2v/results/abc123_lastframe.png",
    ...
}
```

## Frame Count & Duration

| num_frames | Duration @ 16fps | Use Case |
|------------|------------------|----------|
| 33 | ~2 seconds | Quick motion test |
| 49 | ~3 seconds | Short action |
| 65 | ~4 seconds | Standard clip |
| 81 | ~5 seconds | Full segment (default) |
| 97 | ~6 seconds | Extended motion |

## Example Prompts

**Natural motion:**
- "Person speaking naturally with subtle hand gestures"
- "Gentle head movement while listening attentively"
- "Nodding in agreement with a slight smile"

**Specific actions:**
- "Looking down at notes then back up at camera"
- "Gesturing with hands while explaining a concept"
- "Turning head slightly to the left"

**Ambient/subtle:**
- "Subtle breathing movement, very still"
- "Slight eye movement, minimal motion"
- "Hair gently moving, very subtle"

## GPU Memory Requirements

| GPU VRAM | Offloading | Performance | Notes |
|----------|------------|-------------|-------|
| 24GB | Aggressive | ~60s/5sec | Works but slower |
| 48GB | Light | ~35s/5sec | Recommended |
| 80GB | None | ~25s/5sec | Fastest |

**Recommendation**: Use A6000 (48GB) for best cost/performance balance.

## Performance

Using A6000 48GB (~$0.76/hr):

| Config | Frames | Time | Cost/Clip |
|--------|--------|------|-----------|
| Default | 81 (5s) | ~35s | ~$0.007 |
| Short | 49 (3s) | ~22s | ~$0.005 |
| Quick test | 33 (2s) | ~15s | ~$0.003 |

**Full 60-second video (12 segments):** ~$0.08

## Pipeline Chaining

The `last_frame_base64` output enables seamless chaining:

```python
# Pseudo-code for video generation pipeline
def generate_video_segment(reference_image, scenes):
    current_frame = reference_image

    for scene in scenes:
        # Worker 1: Edit frame for scene
        edited = call_qwen_edit(current_frame, scene["edit_prompt"])

        # Worker 2: Animate the edited frame
        result = call_wan_i2v(edited, scene["motion_prompt"])

        # Chain: last frame becomes next input
        current_frame = result["last_frame_base64"]
        segments.append(result["video_base64"])

    return stitch_segments(segments)
```

## Local Testing

```bash
# Build
docker build -t wan-i2v-test .

# Run interactive shell
docker run --gpus all -it wan-i2v-test /bin/bash

# Inside container
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "from lightx2v import LightX2VPipeline; print('LightX2V OK')"
```

## Troubleshooting

### "CUDA out of memory"

1. Use a GPU with 48GB+ VRAM
2. Reduce `num_frames` (try 49 instead of 81)
3. Reduce resolution (try 640x360)

### "Generation failed - no output video"

Check the prompt. Wan2.2 works best with:
- Clear motion descriptions
- Focus on ONE motion at a time
- Avoid contradictory instructions

### Motion looks unnatural

Try:
- Lower `guidance_scale` (3.0-4.0)
- More specific motion prompts
- Increase `num_inference_steps` (50+)

### Cold start is slow (~90s)

The 14B parameter model takes time to load. Options:
- Use longer idle timeout
- Use "always on" worker
- Pre-warm with test request

## Related Workers

| Worker | Purpose | Status |
|--------|---------|--------|
| `runpod-qwen-edit` | Frame editing | Phase 1 |
| `runpod-wan-i2v` | Image-to-video (this) | Phase 2 |
| `runpod-propainter` | Watermark removal | Deployed |
| `runpod-animate` | SVD animation | Deployed |

## References

- [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [LightX2V GitHub](https://github.com/ModelTC/LightX2V)
- [Wan2.2-I2V-A14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)
