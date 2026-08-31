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

### Windows prerequisites

Two Windows-specific issues will otherwise make the commands in this guide fail in ways
that look like toolkit bugs.

**Set `PYTHONIOENCODING=utf-8` before deploying.** Modal's progress output contains
characters the default `cp1252` console codec can't encode, so `modal deploy` aborts
mid-build with `'charmap' codec can't encode characters in position ...`. The build
itself is fine — only the printing fails. Make it permanent:

```powershell
[Environment]::SetEnvironmentVariable('PYTHONIOENCODING','utf-8','User')
```

Reopen your terminal afterwards so new processes inherit it.

**Make sure `python3` isn't the Microsoft Store stub.** A default Windows install puts
`C:\Users\<you>\AppData\Local\Microsoft\WindowsApps` ahead of your real Python on
`PATH`. That directory holds zero-byte App Execution Aliases which fail with
`Permission denied` — and since this toolkit's docs and commands all invoke
`python3 tools/...`, everything breaks. Check what resolves:

```powershell
(Get-Command python3).Source
```

If it points into `WindowsApps`, either turn off the `python.exe` / `python3.exe`
aliases in *Settings → Apps → App execution aliases*, or move your Python directory
ahead of `WindowsApps` in your user `PATH`.

**FFmpeg** is not bundled on Windows and several tools need it. If `winget install
Gyan.FFmpeg` fails with a corrupted-source error, download
`ffmpeg-release-essentials.zip` from https://www.gyan.dev/ffmpeg/builds/, extract it,
and add its `bin\` directory to `PATH`.

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

# Talking head, diffusion-based. Weights live in a Volume, so populate it FIRST
# (one-off, ~15GB, ~5 min) or the app deploys with nothing to load.
uv run modal run docker/modal-soulx/app.py::populate_weights
uv run modal deploy docker/modal-soulx/app.py

# Video generation (see the LTX-2 prerequisites note below)
uv run modal deploy docker/modal-ltx2/app.py
```

**Deploy them one at a time.** Modal rate-limits `AppCreate`; launching several
`modal deploy` calls in parallel makes most of them fail with
`App create rate limit exceeded`. Image builds happen server-side, so serial
deploys cost you nothing extra in compute — only wall-clock.

**GPU tier gates some tools.** Most apps request an `A10G`, which works on the free
tier. Two request A100-class GPUs and fail at deploy time with
`Please add a payment method to use A100-40GB GPU functions` until a card is on the
account: `image-edit` (A100) and `ltx2` (A100-80GB). The image still gets built and
cached before that error, so re-deploying after adding a payment method is fast.
`ltx2` additionally needs a Modal secret named `huggingface-token` for its gated
weights — it fails on the missing secret *before* it ever reaches the GPU check.

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
MODAL_SOULX_ENDPOINT_URL=https://yourname--video-toolkit-soulx-...modal.run
MODAL_DEWATERMARK_ENDPOINT_URL=https://yourname--video-toolkit-dewatermark-...modal.run
MODAL_LTX2_ENDPOINT_URL=https://yourname--video-toolkit-ltx2-...modal.run
```

When an endpoint label exceeds Modal's length limit, Modal truncates it and appends a
hash — `dewatermark` typically becomes `...-dewatermark-de-3e6418.modal.run`. Copy the
URL Modal actually prints; don't reconstruct it from the pattern. Note that the deploy
log wraps long URLs across lines, so check for a `(label truncated)` marker and rejoin
the pieces.

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
uv run tools/soulx.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

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
| `soulx` | SoulX-FlashHead 1.3B Pro | Talking head video, aspect-preserving | ~$0.0024 per second of output |
| `dewatermark` | ProPainter | AI video inpainting | ~$0.05-0.50 |

All apps use A10G GPUs (24GB VRAM) except `image_edit` which uses A100 for its 25GB model.

### Weight storage

Most apps **bake** their model weights into the image at build time. `soulx` is the
exception: it keeps its ~15GB in a **Modal Volume**, which is why it needs the one-off
`populate_weights` run above before its first deploy.

The split is measured, not stylistic. Cold start and generation speed are the same either
way; what differs is rebuild time after a dependency change — measured at 1.8-8.2s on a
volume against 79-385s baked, because any invalidated layer re-downloads everything below it. Apps that
still change often earn a volume; settled ones don't need one.

**Volumes are optional and free at this scale.** Modal charges $0.09/GiB/month for volume
storage with **1 TiB/month included free**, so `soulx`'s 14.7GB costs nothing — it is
~2.6% of the free allowance. There is no bill either way, which is precisely why the choice
comes down to rebuild speed versus having one self-contained artifact rather than to cost.

Every app can be built either way. `soulx` defaults to a volume; the rest bake by
default. See `docs/soulx.md` for the full
comparison.

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
| `soulx` | ~15s + ~600s first-call torch.compile | 6.4-7.9x realtime |
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
