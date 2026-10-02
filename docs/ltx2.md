# LTX-2 - AI Video Generation

Generate ~5 second video clips from text prompts or images using the LTX-2.5 22B model from Lightricks. Produces video with synchronized audio — both generated simultaneously by the model.

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

LTX-2 runs on Modal cloud GPU (A100-80GB). Setup takes about 15-20 minutes — most of that is baking ~80GB of model weights into the container image so future cold starts are fast.

### Prerequisites

- Modal account and CLI installed (`uv sync --extra modal && uv run modal setup`)
- HuggingFace account with a **read-access** token ([create one here](https://huggingface.co/settings/tokens) — a classic "Read" token works; fine-grained tokens need "Read access to contents of all public gated repos you can access")
- Accept the [LTX-2.5 terms](https://huggingface.co/Lightricks/LTX-2.5) ("Agree and access" on the model page — the repo is gated, approval is automatic)

### Steps

1. **Create a Modal secret** with your HuggingFace token:

   ```bash
   uv run modal secret create huggingface-token HF_TOKEN=hf_your_token_here
   ```

   > **Important:** LTX-2.5 is a gated repo, so the build fails with a 401/403 until the token's account has accepted the terms. The token covers all ~80GB of weights, including the bundled Gemma 4 text encoder (2.5 no longer downloads Gemma from Google's repo).

2. **Deploy the Modal app** (downloads and bakes all model weights — takes 15-20 min):

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

LTX-2.5 is a 22B parameter diffusion transformer with a two-stage pipeline:

1. **Stage 1:** Generate video at half resolution (e.g., 384x256) over 30 guided denoising steps (CFG + STG) with the full "dev" transformer
2. **Stage 2:** Upscale 2x with a latent spatial upsampler, then refine with 3 distilled steps (the distilled LoRA applied to the same transformer)
3. **Decode:** The 2.5 diffusion video decoder turns latents into frames (sharper faces and textures than the 2.3 VAE; NATTEN-accelerated in the container)

The model generates **video and audio simultaneously** through bidirectional cross-attention between video and audio streams; the audio VAE and vocoder turn the audio latents into a waveform.

The container bakes the dev transformer plus the distilled LoRA rather than the standalone distilled transformer. The distilled transformer is faster but runs a fixed 8+3 step schedule with no guidance, so `--quality`, `--steps` and `--negative-prompt` would do nothing.

### Components

LTX-2.5 ships as a split pack, one file per component, from [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5):

| Component | Size | Role |
|-----------|------|------|
| Transformer (22B DiT, dev, bf16) | 42 GB | Core video+audio generation |
| Gemma 4 12B text encoder + projections (bf16) | 26 GB | Text understanding (bundled, tokenizer included) |
| Distilled LoRA | 8.9 GB | Stage-2 refinement in 3 steps |
| Video VAE (diffusion decoder) | 1.5 GB | Latent to frames |
| Spatial upsampler (x2) | 1 GB | Resolution upscaling |
| Audio VAE + vocoder | 0.4 GB | Latent to audio |

Total baked weight: ~80 GB. The pipeline loads and frees components one at a time, so peak VRAM is roughly one transformer (~42 GB) plus activations, not the sum of all components. bf16 weights, no quantization or CPU offload, on an A100-80GB.

## Cost & Performance

- **GPU:** A100-80GB on Modal (~$2.50/hr at Modal's base rate)
- **Cold start:** ~25-30s for the container to boot and build the pipeline. Weights aren't held between requests (see next point)
- **Per-request load overhead:** ~2 min. Each request loads the Gemma 4 encoder (~26GB) and the 42GB transformer twice (once per stage, the second time with the distilled LoRA fused in), then frees them. Measured: 93s for prompt encoding + stage-1 load, 30s for the stage-2 reload
- **Measured (512x512, 25 frames, `--quality fast`):** 153s on the server, 186s end to end including cold start, ~$0.13
- **Default settings (768x512, 121 frames, 30 steps):** not benchmarked on 2.5 yet. The ~2 min load overhead comes on top of the longer denoising
- **Scale to zero:** Container shuts down after 60s idle (no cost when not in use)

## Known Limitations

- **Training data artifacts:** The model occasionally generates unwanted logos, text overlays, or watermark-like artifacts from its training data (~30% of generations). Re-generating with a different seed usually fixes this.
- **No visible watermarks:** The LTX-2 pipeline code doesn't add watermarks to output. The 2.x license forbids removing any watermark or provenance features Lightricks does ship (see License).
- **Max duration:** Practical limit is ~8s (193 frames) with this endpoint. Longer clips need stitching. (Upstream 2.5 can window long clips with `--chunked`; this endpoint doesn't expose it.)
- **Style LoRAs:** `crt-terminal` was trained on LTX-2.3. Lightricks says most 2.3 LoRAs run on 2.5 unchanged, with some exceptions, and this one hasn't been checked yet. Try a test clip before relying on it.
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

### Every call fails with "Modal HTTP 500 ... upstream request timeout"

The container is crashing on startup, which from the client looks the same as a slow cold start. Check the logs:

```bash
uv run modal app logs video-toolkit-ltx2
```

`AttributeError: module 'torch.compiler' has no attribute 'nested_compile_region'` means the app was deployed from a toolkit version that cloned LTX-2 unpinned and picked up an upstream release needing torch 2.13. Update the toolkit and redeploy — `docker/modal-ltx2/app.py` now pins an upstream release (`LTX2_REPO_REF`) together with the torch it was tested on.

### Deploy fails with 401 / 403 / `GatedRepoError`

The HuggingFace account behind the `huggingface-token` secret hasn't accepted the LTX-2.5 terms. Click "Agree and access" on [the model page](https://huggingface.co/Lightricks/LTX-2.5) with that account, then redeploy. Fine-grained tokens also need the gated-repos read permission.

## License

LTX-2.5 weights are under the [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) (dated August 11, 2026; LTX-2.3 stays under the earlier [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2)). Key points:
- Free for individuals and for entities under $10M annual revenue, commercial use included. Revenue counts across the whole entity, affiliates included
- Entities at or above $10M revenue need a paid commercial license, except for non-commercial testing and research
- You may not remove, disable or circumvent watermarking, metadata, content provenance or other transparency features included in the model or applied to its outputs, and you're responsible for AI-labelling rules that apply to you (e.g. EU AI Act)
- Commercial users may not use it to train or fine-tune a competing model, apart from permitted derivatives
- Downloading from HuggingFace means accepting these terms (the repo is gated)
