# Modal Cloud GPU Setup

Modal is the recommended cloud GPU provider for the toolkit's AI tools. It offers $30/month free compute on the Starter plan, fast cold starts, and scale-to-zero billing.

> **Fastest path:** Run `/setup` in Claude Code — it handles Modal installation, deployment, and `.env` configuration interactively. This doc is the reference for what `/setup` does under the hood, and for manual setup.

## Create a Modal Account

1. Go to [modal.com](https://modal.com/) and sign up
2. Choose the **Starter plan** — $30/month free compute, just requires a payment method
3. Typical toolkit usage is $1-2/month, well within the free allowance
4. All apps scale to zero — no charges when idle

## Install & Authenticate

```bash
uv sync --extra modal      # Installs the Modal CLI into the toolkit's .venv
uv run modal setup         # Opens browser to authenticate, saves token to ~/.modal.toml
uv run modal app list      # Verify it works
```

## Deploy Tools

Each AI tool has its own Modal app. Deploy only what you need, or deploy all of them — idle apps cost nothing.

```bash
# Speech generation (most commonly used)
uv run modal deploy docker/modal-qwen3-tts/app.py

# Image generation & editing
uv run modal deploy docker/modal-flux2/app.py
uv run modal deploy docker/modal-image-edit/app.py
uv run modal deploy docker/modal-upscale/app.py

# Music generation
uv run modal deploy docker/modal-music-gen/app.py

# Video processing
uv run modal deploy docker/modal-sadtalker/app.py
uv run modal deploy docker/modal-propainter/app.py
```

Each deploy prints an endpoint URL like:
```
https://yourname--video-toolkit-qwen3-tts-ttsengine-generate.modal.run
```

Save each URL to your `.env` file:

```bash
# Add to .env (replace with your actual URLs from deploy output)
MODAL_QWEN3_TTS_ENDPOINT_URL=https://yourname--video-toolkit-qwen3-tts-...modal.run
MODAL_FLUX2_ENDPOINT_URL=https://yourname--video-toolkit-flux2-...modal.run
MODAL_IMAGE_EDIT_ENDPOINT_URL=https://yourname--video-toolkit-image-edit-...modal.run
MODAL_UPSCALE_ENDPOINT_URL=https://yourname--video-toolkit-upscale-...modal.run
MODAL_MUSIC_GEN_ENDPOINT_URL=https://yourname--video-toolkit-music-gen-...modal.run
MODAL_SADTALKER_ENDPOINT_URL=https://yourname--video-toolkit-sadtalker-...modal.run
MODAL_DEWATERMARK_ENDPOINT_URL=https://yourname--video-toolkit-dewatermark-...modal.run
```

> **Tip:** `/setup` automates this — it runs each deploy, parses the URL, and writes it to `.env` for you.

## Cloudflare R2 (Recommended)

R2 is free file storage that bridges your local machine and cloud GPUs. Without it, tools fall back to free upload services (slower, less reliable).

**R2 free tier:** 10GB storage, 10 million operations/month, zero egress fees.

See the R2 section in `/setup`, or configure manually:

1. Sign up at [dash.cloudflare.com](https://dash.cloudflare.com/)
2. Go to R2 Object Storage → Create bucket (name it `video-toolkit`)
3. Create an API token: R2 → Manage R2 API Tokens → Object Read & Write
4. Add to `.env`:
   ```
   R2_ACCOUNT_ID=your_account_id
   R2_ACCESS_KEY_ID=your_access_key_id
   R2_SECRET_ACCESS_KEY=your_secret_access_key
   R2_BUCKET_NAME=video-toolkit
   ```

## Use the Tools

All cloud GPU tools accept `--cloud modal`:

```bash
# AI voiceover
uv run tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3 --cloud modal

# AI image generation
uv run tools/flux2.py --prompt "A sunset over mountains" --output sunset.png --cloud modal

# AI image editing
uv run tools/image_edit.py --input photo.jpg --style cyberpunk --cloud modal

# AI upscaling
uv run tools/upscale.py --input photo.jpg --output photo_4x.png --cloud modal

# AI music generation (acemusic cloud API is now default — no Modal needed)
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
# Or use Modal: uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3 --cloud modal

# Talking head from portrait + audio
uv run tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --output talking.mp4 --cloud modal

# Watermark removal
uv run tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --cloud modal
```

## Tools & Costs

| Tool | Backend | Use Case | Est. Cost |
|------|---------|----------|-----------|
| `qwen3_tts` | Qwen3-TTS | AI speech generation | ~$0.005-0.02 |
| `flux2` | FLUX.2 Klein | AI image generation | ~$0.01-0.03 |
| `image_edit` | Qwen-Image-Edit | AI image editing, style transfer | ~$0.02-0.05 |
| `upscale` | RealESRGAN | AI image upscaling (2x/4x) | ~$0.005-0.02 |
| `music_gen` | ACE-Step 1.5 | AI music generation | Free (acemusic) / ~$0.02-0.10 (Modal) |
| `sadtalker` | SadTalker | Talking head video | ~$0.05-0.30 |
| `dewatermark` | ProPainter | AI video inpainting | ~$0.05-0.50 |

All apps use A10G GPUs (24GB VRAM) except `image_edit` which uses A100 for its 25GB model.

## Cold Starts

First request after idle triggers a cold start while Modal loads the model:

| Tool | Cold Start | Warm Request |
|------|-----------|--------------|
| `qwen3_tts` | ~60-90s | ~5-15s |
| `flux2` | ~25-30s | ~1-3s |
| `image_edit` | ~5-8min | ~15-20s |
| `upscale` | ~25-30s | ~3-5s |
| `music_gen` | ~60-90s | ~10-30s |
| `sadtalker` | ~45-60s | ~30-60s |
| `dewatermark` | ~60-70s | varies by video length |

After 60 seconds of no requests, containers scale back to zero. No charges while idle.

## Monitoring & Billing

```bash
# Check what's running (Tasks column should be 0 when idle)
uv run modal app list

# Check today's spend
uv run modal billing report --for today --json

# View container logs
uv run modal app logs video-toolkit-upscale

# Verify your setup
uv run tools/verify_setup.py
```

## Architecture

Each tool has its own Modal app (`docker/modal-*/app.py`), deployed independently:

- **One app per tool** — independent scaling, GPU assignment, and lifecycle
- **Web endpoints** — HTTP POST via `@modal.fastapi_endpoint`, no `modal` pip dependency needed on the client
- **R2 file transfer** — large results upload to Cloudflare R2 (if configured), otherwise base64
- **Scale to zero** — `scaledown_window=60` means containers shut down after 1 minute idle

The client-side abstraction lives in `tools/cloud_gpu.py`, which routes `call_cloud_endpoint()` to either `_call_runpod()` (submit + poll) or `_call_modal()` (synchronous POST).

## RunPod (Alternative)

RunPod is also supported as a fallback provider. Use `--cloud runpod` on any tool.

| Aspect | Modal | RunPod |
|--------|-------|--------|
| **Free tier** | $30/mo compute | None (pay-as-you-go) |
| **Setup** | `modal deploy` | `--setup` flag per tool |
| **Cold start** | Faster (cached layers) | Slower (Docker pull) |
| **Invocation** | Synchronous POST | Async submit + poll |
| **Auth** | Token optional for web endpoints | `RUNPOD_API_KEY` required |

See [runpod-setup.md](runpod-setup.md) for RunPod-specific instructions.
