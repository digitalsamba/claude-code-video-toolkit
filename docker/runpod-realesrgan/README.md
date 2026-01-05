# RunPod Real-ESRGAN Serverless Endpoint

Docker container for Real-ESRGAN image upscaling on RunPod serverless GPUs.

## Features

- 2x or 4x AI upscaling with Real-ESRGAN
- Multiple models: general, anime, photo
- Optional face enhancement with GFPGAN
- Pre-baked model weights for fast cold starts (~30s)
- Support for PNG, JPG, and WebP output

## Models Included

| Model | File | Size | Best For |
|-------|------|------|----------|
| general | RealESRGAN_x4plus.pth | 64MB | Most images |
| anime | RealESRGAN_x4plus_anime_6B.pth | 17MB | Anime/illustrations |
| photo | realesr-general-x4v3.pth | 65MB | Alternative general |

## Build & Deploy

```bash
# Build
docker build -t ghcr.io/YOUR_USERNAME/video-toolkit-realesrgan:latest .

# Push to registry
docker push ghcr.io/YOUR_USERNAME/video-toolkit-realesrgan:latest

# Or use automated setup from CLI
python tools/upscale.py --setup
```

## API Usage

### Input

```json
{
  "input": {
    "operation": "upscale",
    "image_url": "https://example.com/image.jpg",
    "scale": 4,
    "model": "general",
    "face_enhance": false,
    "output_format": "png",
    "r2": {
      "endpoint_url": "https://xxx.r2.cloudflarestorage.com",
      "access_key_id": "...",
      "secret_access_key": "...",
      "bucket_name": "..."
    }
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image_url | string | required | URL to input image |
| scale | int | 4 | Upscale factor (2 or 4) |
| model | string | "general" | Model: general, anime, photo |
| face_enhance | bool | false | Use GFPGAN for faces |
| output_format | string | "png" | Output: png, jpg, webp |
| r2 | object | null | R2 config for result upload |

### Output

```json
{
  "success": true,
  "output_url": "https://...",
  "input_dimensions": "800x600",
  "output_dimensions": "3200x2400",
  "scale": 4,
  "model_used": "general",
  "inference_time_seconds": 2.5,
  "processing_time_seconds": 5.2,
  "r2_key": "upscale/results/abc123.png"
}
```

## Performance

| Image Size | Scale | GPU | Time |
|------------|-------|-----|------|
| 800x600 | 4x | RTX 3090 | ~2-3s |
| 1920x1080 | 4x | RTX 3090 | ~5-8s |
| 4K | 4x | RTX 3090 | ~15-25s |

## Cost Estimate

Using RTX 3090 (~$0.34/hr):
- Small images: ~$0.01-0.02
- Large images: ~$0.02-0.05
- Very large/batch: ~$0.05-0.10

## Local Testing

```bash
# Build locally
docker build -t realesrgan-test .

# Test with GPU
docker run --gpus all -p 8000:8000 realesrgan-test

# Send test request
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"operation": "upscale", "image_url": "https://example.com/test.jpg"}}'
```

## Related

- [tools/upscale.py](../../tools/upscale.py) - CLI wrapper
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Original project
- [GFPGAN](https://github.com/TencentARC/GFPGAN) - Face enhancement
