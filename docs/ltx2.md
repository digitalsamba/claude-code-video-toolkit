# LTX-2 - AI Video Generation

Generate ~5 second video clips from text prompts or images using the LTX-2.3 22B model from Lightricks. Produces video with synchronized audio — both generated simultaneously by the model.

LTX-2.3 is the default. LTX-2.5 can be deployed instead as an opt-in; see [LTX-2.5 (opt-in)](#ltx-25-opt-in).

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

LTX-2 runs on Modal cloud GPU (A100-80GB). Setup takes about 15 minutes — most of that is baking ~80GB of model weights into the container image so future cold starts are fast.

### Prerequisites

- Modal account and CLI installed (`uv sync --extra modal && uv run modal setup`)
- HuggingFace account with a **read-access** token ([create one here](https://huggingface.co/settings/tokens) — "Read access to contents of all repos" scope is sufficient)
- Access to Google's gated [Gemma 3 text encoder](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) for that account. Google approves these requests by hand, so ask well before you deploy

### Steps

1. **Create a Modal secret** with your HuggingFace token:

   ```bash
   uv run modal secret create huggingface-token HF_TOKEN=hf_your_token_here
   ```

   > **Important:** This token is used for both the LTX-2 weights (~55GB) and the Gemma text encoder (~24GB). While LTX-2.3 isn't a gated model, unauthenticated downloads from HuggingFace are severely rate-limited — a 46GB checkpoint that takes ~10 minutes with a token can take over an hour without one. The Gemma model is gated and will fail entirely without auth.

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

Total baked weight: ~80 GB, including the ~0.4 GB `crt-terminal` style LoRA. The pipeline manages memory by loading and freeing components sequentially — peak VRAM is ~44GB (the transformer alone), not the sum of all components.

## Cost & Performance

- **GPU:** A100-80GB on Modal (~$2.50/hr, modal.com/pricing, 2026-10-02)
- **Cold start:** the container boots and sets up the pipeline. The weights are not kept in VRAM between requests: at the pinned LTX-2 code, every request (warm or cold) reads the Gemma 3 encoder (~24GB) and builds the transformer once per stage, then frees them
- **Generation:** ~100s per default clip (768x512, 121 frames, 30 steps) on a warm container, ~170s from cold (measured 2026-10-12)
- **Estimated cost:** ~$0.07 per 5-second clip warm, ~$0.12 cold (GPU time only; see the benchmark table)
- **Scale to zero:** Container shuts down after 60s idle (no cost when not in use)

## Known Limitations

- **Training data artifacts:** The model occasionally generates unwanted logos, text overlays, or watermark-like artifacts from its training data (~30% of generations). Re-generating with a different seed usually fixes this.
- **No visible watermarks:** LTX-2 does not intentionally add watermarks to output.
- **Max duration:** Practical limit is ~8s (193 frames). For a longer shot, continue the clip with
  `tools/ltx_extend.py --input clip.mp4 --at <last good second> --times N --prompt "..."`: it runs image-to-video
  from the clip's own frame and joins the two with a short cross-fade. Holding the last frame reads as a pause and
  playing the tail back and forth reads as a loop. Never continue from a close-up frame (LTX invents a new scene),
  and don't say "tripod" in the prompt (it can draw one). Method: `.claude/skills/parody-short/people.md`.
- **Audio quality:** Generated audio is ambient/environmental. It won't produce speech or music — use the toolkit's voiceover and music tools for that.

## LTX-2.5 (opt-in)

The same Modal app can deploy LTX-2.5 instead of LTX-2.3. It is a deploy-time switch, like `FLUX2_MODEL` and `IMAGE_EDIT_MODEL`: the endpoint URL, the request format and `tools/ltx2.py` stay the same, and every response's `model` field (printed by the tool as `Model:`) says which version ran. LTX-2.3 stays the default: LTX-2.5 is ~25-45% slower per clip and the `crt-terminal` LoRA only works on 2.3 (see [Benchmarks](#benchmarks)).

### Deploy LTX-2.5

1. **Accept the LTX-2.5 terms** with the Hugging Face account behind the `huggingface-token` secret: open [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and click "Agree and Access". Access is granted automatically, but agreeing shares your contact information with Lightricks, and the form also asks you to consent to marketing updates. Without access the build fails with 401/403.

2. **Deploy with `LTX2_VERSION=2.5`** (Git Bash, macOS or Linux):

   ```bash
   LTX2_VERSION=2.5 uv run modal deploy --strategy recreate docker/modal-ltx2/app.py
   ```

   In PowerShell, run `$env:LTX2_VERSION='2.5'` first and then the deploy command without the prefix. The variable stays set for the rest of that PowerShell session, so clear it with `Remove-Item Env:LTX2_VERSION` before you deploy 2.3 again.

   The first 2.5 deploy builds a separate image and downloads ~80GB of weights into it. `--strategy recreate` stops the containers that were running before the deploy, so a warm 2.3 container can't serve the next request.

3. **Check the `Model:` line** of the next clip. It should read `Lightricks/LTX-2.5`.

**Switch back to 2.3:** deploy again without the variable, `uv run modal deploy --strategy recreate docker/modal-ltx2/app.py`. The 2.3 image is unchanged, so Modal reuses its cached layers and rebuilds nothing.

**Run both side by side** (for comparisons): `LTX2_VERSION=2.5 uv run modal deploy --name video-toolkit-ltx2-25 docker/modal-ltx2/app.py` deploys 2.5 as a second app with its own URL. To send a run there, set `MODAL_LTX2_ENDPOINT_URL` to that URL in the shell; a shell variable takes precedence over `.env`. Remove the second app with `uv run modal app stop video-toolkit-ltx2-25` when you're done.

### What differs from LTX-2.3

| | LTX-2.3 (default) | LTX-2.5 (`LTX2_VERSION=2.5`) |
|---|---|---|
| Weights | `Lightricks/LTX-2.3` plus Google's Gemma 3 text encoder | `Lightricks/LTX-2.5` split pack with a bundled Gemma 4 text encoder, pinned to revision `49b093a` |
| Gated access | Gemma 3 (approved by hand) | LTX-2.5 (approved automatically, shares your contact info) |
| License | LTX-2 Community License | LTX-2.x Community License (see below) |
| LTX-2 code | commit `a2c3f24` | release v1.4.2 |
| Stack | torch 2.7 / CUDA 12.6, transformers 4.57.6, flash-attn when it builds | torch 2.13 / CUDA 13.0, transformers 5.14.1, NATTEN |
| Video decoder | VAE | diffusion decoder (DiffVAE) |
| Image-to-video conditioning | CRF 0 (lossless) | CRF from the checkpoint metadata: 18 for 2.4 and later, 33 if it has no version |
| Guidance | CFG 3.0 / STG 1.0 on block 28 (audio CFG 7.0), set in the app | read from the checkpoint metadata (the same values for 2.4 and later) |
| Image size | ~80GB | ~80GB |
| GPU | A100-80GB | A100-80GB |

Both versions build every model stage per request and free it afterwards, so warm requests pay load time on either one. The two-stage pipeline, the request options (`quality`, `steps`, negative prompt, image-to-video, `lora`) and the frame and size rules are the same. The 2.5 app also rejects a width or height under 64 with a clear error, and logs the `Checkpoint declares model_version=...` lines that decide its guidance and image-to-video CRF (`uv run modal app logs video-toolkit-ltx2`).

### License (LTX-2.5)

LTX-2.5 weights are under the [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x), dated August 11, 2026. It is not a non-commercial license:

- Free for individuals and for entities under $10M annual revenue, commercial use included. Revenue counts across the whole entity, affiliates included.
- Entities with $10M or more in annual revenue need a paid commercial license, except for purely non-commercial use.
- You may not remove, disable or circumvent watermarking, content provenance or other transparency features included in the model or applied to its outputs.

### Style LoRAs on 2.5

**`crt-terminal` doesn't carry over to 2.5.** In the benchmark below, the same prompt and seed with the LoRA gave the full green-phosphor CRT terminal look on 2.3, but on 2.5 a photoreal scene with green text on top: the LoRA changes the output without reproducing its style. Use a 2.3 deployment for this preset. Lightricks says two different things about 2.3 LoRAs on 2.5:

- The [LTX-2.5 model card](https://huggingface.co/Lightricks/LTX-2.5): "Based on our testing, the large majority of LoRAs and IC-LoRAs trained on LTX-2.3 run on LTX-2.5 without changes. A small number of exceptions exist — validate your adapters before production use."
- The [LTX-2 README](https://github.com/Lightricks/LTX-2/blob/v1.4.2/README.md) at v1.4.2: "Files are not interchangeable between the two models, and a LoRA only works with the model it was trained on."

When the weights are fused, LoRA keys that don't exist in the model are skipped without an error. A LoRA that only partly applies still renders something plausible, so compare a 2.5 clip against 2.3 by eye before relying on any other 2.3 LoRA.

### Benchmarks

Measured 2026-10-12 on throwaway Modal apps (A100-80GB), the same prompt and seed (42) on both versions. Times are end to end as `ltx2.py` reports them; cost is GPU time at $2.50/h. Warm means the next request on a container that just finished one.

| Case | Settings | LTX-2.3 | LTX-2.5 |
|---|---|---|---|
| Text-to-video, cold start | 768x512, 121 frames, 30 steps | 172s, $0.12 | 236s, $0.16 |
| Text-to-video, warm (two runs) | 768x512, 121 frames, 30 steps | 100s / 100s, $0.07 | 124s / 144s, $0.09-0.10 |
| Short clip, `--quality fast`, warm | 512x512, 25 frames, 15 steps | 63s, $0.04 | 95s, $0.07 |
| Image-to-video, warm | 768x512, 121 frames, 30 steps | 99s, $0.07 | 137s, $0.10 |
| `crt-terminal` LoRA, warm | 1024x1024, 121 frames, 30 steps | 187s, $0.13 | 215s, $0.15 |
| Same prompt without the LoRA, warm | 1024x1024, 121 frames, 30 steps | 182s | 189s |

- **Speed:** 2.5 is about 25-45% slower per clip at the default size, warm or cold. Both versions rebuild their models on every request, so warm 2.3 calls aren't free either (~100s).
- **Text-to-video and image-to-video:** both versions keep the input still's identity in image-to-video, and neither showed obvious artefacts. 2.3 added drifting smoke for "gentle wind"; 2.5 was steadier, with smaller expression and gaze changes. Every clip has a 48kHz stereo audio track.
- **LoRA:** `crt-terminal` gives its CRT look on 2.3 but not on 2.5 (see above).
- **2.3 unchanged:** redeploying the default after this change built no new image layers, and a 25-frame clip from it was bit-identical (every frame's hash) to one from main's `app.py` with the same seed.

Quality was judged from contact sheets, not by an automated similarity score; audio was only checked for presence.

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

`AttributeError: module 'torch.compiler' has no attribute 'nested_compile_region'` means the app was deployed from a toolkit version that cloned LTX-2 unpinned and picked up an upstream release needing torch 2.13. Update the toolkit and redeploy — `docker/modal-ltx2/app.py` now pins the upstream commit (`LTX2_REPO_REF`).

### Deploy fails with 401, 403 or `GatedRepoError`

The Hugging Face account behind the `huggingface-token` secret can't download a gated repo the build needs: Google's Gemma 3 text encoder for the default LTX-2.3 image (Google approves access by hand), or `Lightricks/LTX-2.5` for an `LTX2_VERSION=2.5` deploy. Get access with that account, then deploy again. Fine-grained tokens also need read access to the gated repos you can access.

## License

The LTX-2.5 opt-in has its own license; see [License (LTX-2.5)](#license-ltx-25).

LTX-2 uses a [Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) from Lightricks. Key points:
- Free for individuals and companies under $10M annual revenue
- Entities over $10M revenue need a commercial license from Lightricks
- Cannot be used to build directly competing products without a commercial license
