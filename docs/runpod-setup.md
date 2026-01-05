# RunPod Cloud GPU Setup

This guide covers setting up RunPod serverless GPUs for the toolkit's Cloud GPU tools.

## Cloud GPU Tools

| Tool | Backend | Use Case | Cost/Job |
|------|---------|----------|----------|
| `image_edit` | Qwen-Image-Edit | AI image editing, style transfer | ~$0.01-0.02 |
| `upscale` | RealESRGAN | AI image upscaling (2x/4x) | ~$0.01-0.05 |
| `dewatermark` | ProPainter | AI video inpainting | ~$0.05-0.30 |

## Why RunPod?

These tools use AI models that require significant GPU power:

| Hardware | Image Edit | Upscale | Dewatermark (30s) | Viable? |
|----------|------------|---------|-------------------|---------|
| NVIDIA RTX 3090+ | ~20s | ~10s | 2-5 min | Yes |
| Apple Silicon | N/A | N/A | 4+ hours | No |
| CPU only | N/A | N/A | 10+ hours | No |

RunPod provides on-demand NVIDIA GPUs at ~$0.34/hour, making it cost-effective for occasional use.

## Quick Start (Automated)

The fastest way to set up RunPod:

```bash
# 1. Add your RunPod API key to .env
echo "RUNPOD_API_KEY=your_key_here" >> .env

# 2. Run automated setup for each tool you need
python tools/image_edit.py --setup    # AI image editing
python tools/upscale.py --setup       # AI upscaling
python tools/dewatermark.py --setup   # AI watermark removal

# 3. Done! Now use them:
python tools/image_edit.py --input photo.jpg --style cyberpunk
python tools/upscale.py --input photo.jpg --output photo_4x.png --runpod
python tools/dewatermark.py --input video.mp4 --region x,y,w,h --output out.mp4 --runpod
```

Each `--setup` command will:
- Create a serverless template using a pre-built Docker image
- Create an endpoint with appropriate GPU
- Save the endpoint ID to your `.env` file

## Pre-built Docker Images

All tools use pre-built public images (no building required):

| Tool | Image |
|------|-------|
| image_edit | `ghcr.io/conalmullan/video-toolkit-qwen-edit:latest` |
| upscale | `ghcr.io/conalmullan/video-toolkit-realesrgan:latest` |
| dewatermark | `ghcr.io/conalmullan/video-toolkit-propainter:latest` |

Use `--setup-gpu AMPERE_16` for RTX 3080 or `--setup-gpu ADA_24` for RTX 4090.

---

## Manual Setup

If you prefer to set up manually via the web console:

### 1. Create RunPod Account

1. Go to [runpod.io](https://runpod.io) and sign up
2. Add credits to your account ($10 minimum, lasts for many jobs)
3. Go to Settings > API Keys and create an API key

### 2. Create Serverless Endpoint(s)

Create one endpoint per tool you want to use:

| Tool | Docker Image | GPU | Timeout |
|------|--------------|-----|---------|
| image_edit | `ghcr.io/conalmullan/video-toolkit-qwen-edit:latest` | 48GB+ (A6000, L40S, A100) | 300s |
| upscale | `ghcr.io/conalmullan/video-toolkit-realesrgan:latest` | 24GB (RTX 3090/4090) | 300s |
| dewatermark | `ghcr.io/conalmullan/video-toolkit-propainter:latest` | 24GB (RTX 3090/4090) | 3600s |

Steps for each:

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:

| Setting | Value | Notes |
|---------|-------|-------|
| Docker Image | (see table above) | Pre-built public image |
| GPU | (see table above) | VRAM requirements vary |
| Max Workers | 1 | Scale up for batch processing |
| Idle Timeout | 5 seconds | Fast scale-down to save costs |
| Execution Timeout | (see table above) | Video processing needs longer |

4. Click **Create Endpoint**
5. Copy the **Endpoint ID** (looks like: `abc123xyz`)

### 3. Configure Local Environment

Add to your `.env` file (only the endpoints you created):

```bash
# RunPod Configuration
RUNPOD_API_KEY=your_api_key_here

# Endpoint IDs (one per tool)
RUNPOD_QWEN_EDIT_ENDPOINT_ID=abc123    # For image_edit
RUNPOD_UPSCALE_ENDPOINT_ID=def456      # For upscale
RUNPOD_ENDPOINT_ID=ghi789              # For dewatermark
```

### 4. Test It

```bash
# Image editing
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"

# Upscaling
python tools/upscale.py --input photo.jpg --output photo_4x.png --runpod

# Dewatermark (with dry run)
python tools/dewatermark.py \
    --input video.mp4 \
    --region 1080,660,195,40 \
    --output clean.mp4 \
    --runpod \
    --dry-run
```

## How It Works

```
1. Local tool uploads video to temporary storage
2. Submits job to RunPod endpoint
3. RunPod spins up GPU worker (~30s cold start)
4. Worker downloads video, runs ProPainter
5. Worker uploads result, returns URL
6. Local tool downloads result
7. Worker scales down (you stop paying)
```

## Cost Breakdown

### Per-Video Costs

| Video Length | Processing Time | Cost (RTX 3090) |
|--------------|-----------------|-----------------|
| < 30 seconds | 2-5 minutes | ~$0.02 |
| 30s - 2 min | 5-15 minutes | ~$0.08 |
| 2 - 5 min | 15-45 minutes | ~$0.25 |
| > 5 min | 45+ minutes | ~$0.40+ |

### GPU Options

| GPU | VRAM | Cost/hr | Speed | Best For |
|-----|------|---------|-------|----------|
| RTX 3090 | 24GB | $0.34 | Fast | Most videos (recommended) |
| RTX 4090 | 24GB | $0.69 | Faster | Tight deadlines |
| A100 | 80GB | $1.99 | Fastest | Very long videos |

### Tips to Minimize Costs

1. **Use 5-second idle timeout** - Workers scale down quickly
2. **Process in batches** - Submit multiple videos to same warm worker
3. **Right-size your GPU** - RTX 3090 is plenty for most videos
4. **Set max workers = 1** initially - Prevents runaway costs

## Troubleshooting

### "RUNPOD_API_KEY not set"

Add your API key to `.env`:
```bash
RUNPOD_API_KEY=your_key_here
```

### "RUNPOD_ENDPOINT_ID not set"

Add your endpoint ID to `.env`:
```bash
RUNPOD_ENDPOINT_ID=abc123xyz
```

### Job times out

Default timeout is 30 minutes. For longer videos:
```bash
python tools/dewatermark.py ... --runpod --runpod-timeout 3600
```

### "Failed to upload video"

- Check your internet connection
- Verify the video file exists and is readable
- Large files (>500MB) may take several minutes to upload

### Cold start is slow (~30-60 seconds)

This is normal for the first request after idle. The worker needs to:
1. Spin up the container
2. Load PyTorch and models into GPU memory

Subsequent requests to a warm worker are faster.

### "ProPainter processing failed"

Check the RunPod logs:
1. Go to RunPod Console > Serverless > Your Endpoint > Logs
2. Look for error messages from the handler

Common issues:
- Video format not supported (try converting to MP4)
- Region coordinates exceed video dimensions
- GPU ran out of memory (shouldn't happen with 24GB GPUs)

## File Transfer: Cloudflare R2 (Recommended)

By default, videos are uploaded via free file hosting services (litterbox.catbox.moe, etc.). These work but can be unreliable for large files.

**Cloudflare R2** provides reliable, fast file transfer with a generous free tier:
- **10 GB storage** (we clean up after each job)
- **10 million operations/month**
- **Zero egress fees** (unlike AWS S3)
- **No expiration** (unlike AWS's 12-month free tier)

### R2 Setup

1. **Create Cloudflare Account** (free): https://dash.cloudflare.com

2. **Create R2 Bucket**:
   - Go to R2 Object Storage → Create bucket
   - Name: `video-toolkit` (or any name)
   - Click Create

3. **Create API Token**:
   - R2 → Overview → Manage R2 API Tokens
   - Create API Token → Object Read & Write
   - Specify bucket: `video-toolkit`
   - Copy the **Access Key ID** and **Secret Access Key** (shown once!)

4. **Get Account ID**:
   - Visible in dashboard URL: `dash.cloudflare.com/<ACCOUNT_ID>/r2`

5. **Add to .env**:
   ```bash
   R2_ACCOUNT_ID=your_account_id
   R2_ACCESS_KEY_ID=your_access_key_id
   R2_SECRET_ACCESS_KEY=your_secret_access_key
   R2_BUCKET_NAME=video-toolkit
   ```

6. **Install boto3** (if not already):
   ```bash
   pip install boto3
   ```

That's it! The dewatermark tool will automatically use R2 for file transfer.

### Without R2

If R2 is not configured, the tool falls back to free file hosting services:
- `litterbox.catbox.moe` (200MB, 24h retention)
- `file.io` (2GB, 1 download)
- `transfer.sh` (10GB, 14 days) - often down
- `0x0.st` (512MB, 30 days) - blocks many requests

These work for testing but may fail intermittently for production use.

## Advanced Configuration

### Multiple Endpoints

You can create multiple endpoints for different use cases:

```bash
# .env
RUNPOD_ENDPOINT_ID=abc123xyz        # Default (RTX 3090)
RUNPOD_ENDPOINT_ID_FAST=def456uvw   # Fast (RTX 4090)
```

### Monitoring Usage

1. Go to RunPod Console > Usage
2. View spend by endpoint, GPU type, and time period
3. Set up billing alerts to avoid surprises

## Security Notes

- API keys grant full access to your RunPod account - keep them secret
- R2 credentials are passed to RunPod workers for result upload - ensure your bucket is private
- Without R2, videos go through public file hosting services (not recommended for sensitive content)
- R2 objects are automatically cleaned up after download
- Presigned URLs expire after 2 hours

## Future GPU Tools

The toolkit is designed for extensibility. Future tools may include:

- **video_gen** - AI video generation (Wan I2V, in development)
- **animate** - Character animation
- **denoise** - Audio/video denoising
- **stabilize** - Video stabilization

## Current Status & Known Limitations

**Working:**
- ✅ AI image editing (Qwen-Image-Edit) - style transfer, backgrounds, custom prompts
- ✅ AI upscaling (RealESRGAN) - 2x/4x with face enhancement
- ✅ AI watermark removal (ProPainter) - video inpainting
- ✅ Cloudflare R2 file transfer (reliable, fast)
- ✅ Automatic GPU detection

**Current Limitations:**

| Tool | Issue | Workaround |
|------|-------|------------|
| image_edit | Requires 48GB+ VRAM | Use A6000, L40S, or A100 GPUs |
| image_edit | Background replacement inconsistent | Use style/prompt instead |
| dewatermark | Long videos untested | Chunk videos client-side |
| dewatermark | Full resolution OOM on 24GB | Use `auto` resize_ratio |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0.0 | 2025-12-30 | **Major fix:** GPU detection (removed CUDA_VISIBLE_DEVICES), CUDA 12.4, auto resize_ratio |
| v1.2.1 | 2025-12-30 | Fix output file detection (`inpaint_out.mp4` not `masked_in.mp4`) |
| v1.2.0 | 2025-12-30 | Fix GPU detection (use max across all GPUs), improve memory profiles |
| v1.1.0 | 2025-12-30 | Add `resize_ratio` parameter, improve error logging |
| v1.0.0 | 2025-12-30 | Initial R2 integration, NumPy 1.x fix |
