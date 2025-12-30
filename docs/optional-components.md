# Optional Components

Some toolkit features require additional software that isn't included by default. These **optional components** are:

- Installed on-demand (not part of base toolkit)
- Stored in `~/.video-toolkit/` (outside the project)
- Only needed for specific use cases

## Available Optional Components

| Component | Tool | Purpose | Size |
|-----------|------|---------|------|
| ProPainter | `dewatermark.py` | AI video inpainting for watermark removal | ~2GB |

## ProPainter (Watermark Removal)

[ProPainter](https://github.com/sczhou/ProPainter) is an AI video inpainting model that can intelligently remove watermarks by reconstructing the underlying content.

### Hardware Requirements

| Hardware | Status | Notes |
|----------|--------|-------|
| NVIDIA GPU (8GB+ VRAM) | **Supported** | Recommended, ~5-15 min per minute of video |
| Cloud GPU (RunPod, etc.) | **Supported** | Good alternative, ~$0.20-0.50 per video |
| Apple Silicon (M1/M2/M3/M4) | **Not supported** | MPS is too slow (40+ hours for short videos) |
| CPU only | **Not supported** | Impractical processing times |

### Why Apple Silicon Doesn't Work

ProPainter relies on optical flow (RAFT) which performs extremely poorly on Apple's MPS backend:

1. **MPS INT_MAX Limit**: MPS cannot handle tensors > 2^31 elements, limiting chunks to ~32 seconds at 720p
2. **MPS Performance**: Optical flow on MPS is orders of magnitude slower than CUDA
3. **Real-world result**: 5 seconds of video takes 4+ hours on M1/M2/M3/M4

This is a PyTorch/MPS limitation, not something we can fix in the tool.

### Installation

```bash
# Check current status
python tools/dewatermark.py --status

# Install ProPainter
python tools/dewatermark.py --install
```

This will:
1. Clone ProPainter to `~/.video-toolkit/propainter/`
2. Create a Python virtual environment
3. Install PyTorch and dependencies
4. Download model weights (~2GB)

### Usage

**Remove watermark by specifying region:**
```bash
python tools/dewatermark.py \
    --input video.mp4 \
    --region 1080,660,195,40 \
    --output clean.mp4
```

**Use a custom mask image:**
```bash
python tools/dewatermark.py \
    --input video.mp4 \
    --mask mask.png \
    --output clean.mp4
```

### Finding Watermark Coordinates

Use the `locate_watermark.py` helper:

```bash
# Extract frames with coordinate grid
python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

# Verify a region across multiple frames
python tools/locate_watermark.py --input video.mp4 --region 1100,650,150,50 --verify
```

### Cloud GPU Alternative

For users without NVIDIA GPUs, cloud services offer affordable processing:

| Provider | GPU | Cost | Processing Time |
|----------|-----|------|-----------------|
| RunPod | RTX 4090 | ~$0.34/hr | ~15-30 min for 3-min video |
| RunPod | A100 | ~$1.99/hr | ~5-15 min for 3-min video |
| Vast.ai | RTX 3090 | ~$0.20/hr | ~20-40 min for 3-min video |

Both RunPod and Vast.ai have Python APIs for programmatic access.

### Uninstalling

```bash
rm -rf ~/.video-toolkit/propainter
```

## Future Optional Components

The optional components system is designed to support additional ML-based tools:

- **Video upscaling** (Real-ESRGAN, etc.)
- **Audio enhancement** (noise removal, etc.)
- **Scene detection** (automatic scene splitting)

These will follow the same pattern: install on first use, stored in `~/.video-toolkit/`.
