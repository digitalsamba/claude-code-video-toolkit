# Video Generation Pipeline - Project Status

Last updated: 2026-01-01

## Overview

Building a video generation pipeline:
```
Reference Image → [Qwen-Edit] → Edited Frame → [Wan I2V] → Video
```

## Milestone 1.2: Qwen-Edit Endpoint ✅

### Status: WORKING

**Endpoint:** `2v28fbsk8cnrzy` on RunPod
**GPU:** A100 80GB (AMPERE_80)
**Cost:** ~$0.012/image (~$2.17/hr)

### What Works
- QwenImageEditPlusPipeline from diffusers
- BF16 precision on 80GB GPU
- Identity-preserving edits
- ~20s inference time per image

### Test Results

| Edit Type | Identity Preservation | Quality |
|-----------|----------------------|---------|
| Movie characters (Bond, Neo, etc.) | Excellent | Excellent |
| Professional (CEO, video call host) | Good-Excellent | Excellent |
| Animated (Pixar) | Good | Excellent |
| Dark scenes (Jedi) | Poor | Poor |

### Prompt Patterns
```
"{Character/Role}, {costume details}, {pose/action}, {background}, {lighting}, {aesthetic}"
```

See `docs/qwen-edit-patterns.md` for detailed prompt guidance.

## Pending: Faster Cold Starts

### Current Issue
- Cold start: ~5-10 min (model download)
- Goal: ~30s cold start

### Solution In Progress
- Bake models into Docker image (~65GB)
- Using Buildjet for builds (100GB+ disk)
- Build triggered, waiting for runner pickup

### Build Status
- Run ID: `20638078603`
- Monitor: https://github.com/digitalsamba/claude-code-video-toolkit/actions/runs/20638078603

### Files Changed
- `.github/workflows/build-videogen-workers.yml` - Switched to `buildjet-8vcpu-ubuntu-2204`
- `docker/runpod-qwen-edit/Dockerfile` - Added model download during build
- `docker/runpod-qwen-edit/handler.py` - Use baked-in cache

## GPU Options Tested

| GPU | VRAM | BF16 | Cost/hr | Status |
|-----|------|------|---------|--------|
| A100 80GB | 80GB | ✅ Works | $2.17 | Current |
| A6000 48GB | 48GB | ❌ OOM | $0.76 | Too small |
| L4 24GB | 24GB | ❓ Needs FP8 | $0.34 | Future |

**Finding:** BF16 needs ~50GB+ VRAM. For cheaper options, need FP8 quantization.

## Cost Analysis

### Per Image (A100 80GB)
- Inference: ~20s
- Cost: ~$0.012

### Per 1000 Images
- Cost: ~$12
- Time: ~5.5 hours

### Potential Savings
| Optimization | Estimated Savings |
|--------------|-------------------|
| Switch to 48GB + FP8 | ~65% |
| Batch processing | ~20% |
| Reduce steps (8→4) | ~50% time |

## Sample Outputs

Generated images saved to `/Users/conalmullan/work/video/`:
- `conor_bond.png` - James Bond
- `conor_neo.png` - The Matrix
- `conor_wick.png` - John Wick
- `conor_maverick.png` - Top Gun
- `conor_gump.png` - Forrest Gump
- `conor_indy.png` - Indiana Jones
- `conor_shelby.png` - Peaky Blinders
- `conor_pixar.png` - Pixar animated

## Next Steps

1. **Complete baked-image build** - Monitor Buildjet
2. **Update RunPod template** - Point to new image SHA
3. **Test cold start time** - Should be ~30s
4. **Milestone 1.3: Wan I2V** - Image-to-video endpoint
5. **Pipeline integration** - Chain Qwen-Edit → Wan I2V

## Files Reference

```
docker/runpod-qwen-edit/
├── Dockerfile          # Container with baked models
├── handler.py          # RunPod serverless handler
└── download_models.py  # (Legacy) Runtime download

tools/
└── test_qwen_edit.py   # Test script

docs/
├── qwen-edit-patterns.md      # Prompt patterns & learnings
└── video-generation-status.md # This file
```

## Environment Variables

```bash
RUNPOD_API_KEY=rpa_...
RUNPOD_QWEN_EDIT_ENDPOINT_ID=2v28fbsk8cnrzy
```
