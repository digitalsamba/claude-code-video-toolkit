# LTX-2 - AI Video Generation

Generate ~5 second video clips from text prompts or images using the LTX-2.3 22B model from Lightricks. Produces video with synchronized audio — both generated simultaneously by the model.

## Quick Start

```bash
# Text-to-video
uv run tools/ltx2.py --prompt "A cat playing with yarn in a sunlit room"

# Image-to-video (animate a still image)
uv run tools/ltx2.py --prompt "Camera slowly pans right" --input photo.jpg

# Higher resolution
uv run tools/ltx2.py --prompt "Ocean waves at sunset" --width 1024 --height 576

# Fast mode (fewer steps, quicker but lower quality)
uv run tools/ltx2.py --prompt "A rocket launch" --quality fast
```

## Setup

LTX-2 runs on Modal cloud GPU (A100-80GB). Setup takes about 15 minutes — most of that is baking ~62GB of model weights into the container image so future cold starts are fast.

### Prerequisites

- Modal account and CLI installed (`uv sync --extra modal && uv run modal setup`)
- HuggingFace account with a **read-access** token ([create one here](https://huggingface.co/settings/tokens) — "Read access to contents of all repos" scope is sufficient)
- Accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) (one-click "Agree" on the model page)

### Steps

1. **Create a Modal secret** with your HuggingFace token:

   ```bash
   uv run modal secret create huggingface-token HF_TOKEN=hf_your_token_here
   ```

   > **Important:** This token is used for both the LTX-2 weights (~55GB) and the Gemma text encoder (~7GB). While LTX-2 isn't a gated model, unauthenticated downloads from HuggingFace are severely rate-limited — a 46GB checkpoint that takes ~10 minutes with a token can take over an hour without one. The Gemma model is gated and will fail entirely without auth.

2. **Deploy the Modal app** (downloads and bakes all model weights — takes 10-15 min):

   ```bash
   uv run modal deploy docker/modal-ltx2/app.py
   ```

3. **Save the endpoint URL** printed by `modal deploy` to your `.env`:

   ```
   MODAL_LTX2_ENDPOINT_URL=https://yourname--video-toolkit-ltx2-ltx2-generate.modal.run
   ```

4. **Test it:**

   ```bash
   uv run tools/ltx2.py --prompt "A single lit candle flickering on a dark table, cinematic lighting"
   ```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | (required) | Text description of the video |
| `--input` | - | Input image for image-to-video |
| `--width` | 768 | Video width (must be divisible by 64) |
| `--height` | 512 | Video height (must be divisible by 64) |
| `--num-frames` | 121 | Frame count. Must satisfy `(n-1) % 8 == 0`. 121 frames = ~5s at 24fps |
| `--fps` | 24 | Frames per second |
| `--quality` | standard | `standard` (30 steps) or `fast` (15 steps) |
| `--steps` | 30 | Override inference steps directly |
| `--seed` | random | Seed for reproducible results |
| `--output` | auto | Output file path (defaults to prompt-based `.mp4`) |
| `--no-open` | - | Don't auto-open the result on macOS |
| `--negative-prompt` | sensible default | What to avoid in generation |

## Valid Frame Counts

Frame counts must satisfy `(n - 1) % 8 == 0`. Common values:

| Frames | Duration (24fps) |
|--------|-------------------|
| 25 | ~1s |
| 49 | ~2s |
| 73 | ~3s |
| 97 | ~4s |
| 121 | ~5s (default) |
| 161 | ~6.7s |
| 193 | ~8s |

If you pass an invalid count, the tool auto-adjusts to the nearest valid value.

## Dimension Constraints

Width and height must each be divisible by 64. Common presets:

| Resolution | Aspect Ratio | Notes |
|------------|--------------|-------|
| 768x512 | 3:2 | Default, good balance of quality and speed |
| 512x512 | 1:1 | Square, fastest |
| 1024x576 | 16:9 | Widescreen |
| 576x1024 | 9:16 | Portrait/vertical video |
| 1024x1536 | 2:3 | Maximum quality (slow, high VRAM) |

Higher resolutions take longer and use more VRAM. The two-stage pipeline generates at half resolution first, then upscales.

## Prompting Tips

LTX-2 responds well to cinematographic descriptions:

- **Camera motion:** "Slow dolly forward", "Aerial drone shot", "Handheld camera", "Tracking shot following..."
- **Lighting:** "Golden hour", "Cinematic lighting", "Neon-lit", "Soft diffused light"
- **Temporal:** "Timelapse of...", "Slow motion", "Gradually transitions from..."
- **Style:** "Shot on 35mm film", "Documentary style", "Studio photography"

Keep prompts under 200 words. Be specific about the scene rather than abstract.

### Example Prompts

```
# Simple scene with motion
"A single lit candle on a dark wooden table, flame gently flickering, soft bokeh background, cinematic lighting"

# Nature with camera motion
"Aerial drone shot slowly flying over turquoise ocean waves breaking on a white sand beach, golden hour sunlight"

# Urban scene
"Rain falling on a neon-lit Tokyo street at night, puddles reflecting colorful signs, people with umbrellas, cinematic"

# Product/tech (for video production)
"Close-up of hands typing on a mechanical keyboard, shallow depth of field, soft desk lamp lighting, cozy atmosphere"
```

## How It Works

LTX-2.3 is a 22B parameter diffusion transformer with a two-stage pipeline:

1. **Stage 1:** Generate video at half resolution (e.g., 384x256) over 30 denoising steps
2. **Stage 2:** Upscale to full resolution using a spatial upsampler with 4 distilled steps

The model generates **video and audio simultaneously** through bidirectional cross-attention between video and audio streams. Audio is generated as a mel spectrogram and decoded by a HiFi-GAN vocoder.

### Components

| Component | Size | Role |
|-----------|------|------|
| Transformer (22B DiT) | 46 GB | Core video+audio generation |
| Gemma 3 12B (QAT bf16) | ~24 GB | Text understanding |
| Spatial upsampler | ~1 GB | Resolution upscaling |
| Distilled LoRA | ~8 GB | Faster inference |

Total baked weight: ~55 GB. The pipeline manages memory by loading and freeing components sequentially — peak VRAM is ~44GB (the transformer alone), not the sum of all components.

## Cost & Performance

- **GPU:** A100-80GB on Modal (~$4.68/hr)
- **Cold start:** ~60-90s (loading ~55GB weights into VRAM)
- **Generation:** ~2.5 min for default 768x512, 121 frames, 30 steps
- **Estimated cost:** ~$0.23 per 5-second clip
- **Scale to zero:** Container shuts down after 60s idle (no cost when not in use)

## Known Limitations

- **Training data artifacts:** The model occasionally generates unwanted logos, text overlays, or watermark-like artifacts from its training data (~30% of generations). Re-generating with a different seed usually fixes this.
- **No visible watermarks:** LTX-2 does not intentionally add watermarks to output.
- **Max duration:** Practical limit is ~8s (193 frames). Longer clips need stitching.
- **Audio quality:** Generated audio is ambient/environmental. It won't produce speech or music — use the toolkit's voiceover and music tools for that.

## Troubleshooting

### "GPU out of memory"

Reduce dimensions or frame count:
```bash
# Try smaller resolution
uv run tools/ltx2.py --prompt "..." --width 512 --height 512

# Or fewer frames
uv run tools/ltx2.py --prompt "..." --num-frames 73
```

### "Modal endpoint is scaling up"

First request after idle triggers a cold start (~30-60s). Retry after a moment.

### Training data artifacts in output

Re-run with a different `--seed`. Adding "no text, no watermark, no logo" to the prompt can help.

### "Modal function timed out"

High-resolution or long videos can exceed the timeout. Use `--quality fast` or reduce dimensions.

## License

LTX-2 uses a [Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) from Lightricks. Key points:
- Free for individuals and companies under $10M annual revenue
- Entities over $10M revenue need a commercial license from Lightricks
- Cannot be used to build directly competing products without a commercial license
