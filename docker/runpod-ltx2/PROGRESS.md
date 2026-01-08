# LTX-2 Integration Progress

**Branch:** `experiment/ltx2`
**Last updated:** 2026-01-08
**Status:** CODE COMPLETE - Awaiting build & test

## Completed
- [x] Dockerfile with runtime model download (~55GB from HuggingFace)
- [x] handler.py with T2V and I2V support, R2 integration
- [x] tools/ltx2.py CLI with --setup, presets, flow modes
- [x] README.md documentation
- [x] docs/ltx2.md user guide
- [x] CLAUDE.md updated with ltx2 tool

## Pending
- [ ] Build Docker image
- [ ] Push to ghcr.io/conalmullan/video-toolkit-ltx2:latest
- [ ] Deploy endpoint via --setup
- [ ] Test text-to-video generation
- [ ] Test image-to-video generation
- [ ] Verify FP8 model loading
- [ ] Performance benchmarks

## Technical Notes

**Model:** Lightricks/LTX-2 (19B parameters, FP8 quantized ~27GB)

**Key differences from other workers:**
- Models download at runtime (not baked in) due to size (~55GB total)
- Requires network volume at `/runpod-volume` for caching
- Uses official ltx-pipelines package from git (fallback to diffusers)
- Supports both T2V and I2V in single handler

**Flow modes:**
- `pro` - 50 inference steps, best quality
- `fast` - 25 inference steps, 2x faster

**Dimension constraints:**
- Width/height must be divisible by 32
- num_frames should be divisible by 8 + 1

## Resume Instructions

```bash
# Switch to branch
git checkout experiment/ltx2

# Build Docker image (takes ~10-15 min, downloads models)
cd docker/runpod-ltx2
docker buildx build --platform linux/amd64 -t ghcr.io/conalmullan/video-toolkit-ltx2:latest .

# Push to registry
docker push ghcr.io/conalmullan/video-toolkit-ltx2:latest

# Setup endpoint (creates template + endpoint in RunPod)
python tools/ltx2.py --setup

# IMPORTANT: In RunPod console, attach 100GB+ network volume at /runpod-volume

# Test text-to-video
python tools/ltx2.py --prompt "A cat playing with yarn" --preset fast --output test_t2v.mp4

# Test image-to-video
python tools/ltx2.py --image test_image.png --prompt "Make them wave" --output test_i2v.mp4
```

## Key Files
- `docker/runpod-ltx2/Dockerfile` - Docker build with ltx-pipelines
- `docker/runpod-ltx2/handler.py` - RunPod serverless handler
- `tools/ltx2.py` - CLI tool

## Potential Issues to Watch
1. **Cold start time** - First run downloads ~55GB, may timeout
2. **VRAM usage** - FP8 needs 24GB minimum, may need tuning
3. **diffusers vs ltx-pipelines** - Handler tries diffusers first, falls back to official packages
4. **Network volume** - Must be configured in RunPod for model caching

## Cost Estimates (projected)
| Duration | Pro Time | Fast Time | Cost (4090) |
|----------|----------|-----------|-------------|
| 5s video | ~45s | ~25s | ~$0.03 |
| 10s video | ~90s | ~50s | ~$0.06 |

## See Also
- `.ai_dev/ltx-2-evaluation.md` - Original research and model comparison
- `docs/ltx2.md` - User documentation
