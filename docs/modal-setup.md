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

`tools/modal_setup.py` deploys apps and writes each endpoint URL into `.env` for you:

```bash
uv run tools/modal_setup.py --list               # deployed? in .env? stale?
uv run tools/modal_setup.py qwen3_tts flux2      # deploy these, save their URLs
uv run tools/modal_setup.py --all                # every app except ltx2
uv run tools/modal_setup.py ltx2                 # ltx2 is never part of --all
uv run tools/modal_setup.py --sync-env           # no deploy: fix .env from what's live
uv run tools/modal_setup.py --all --dry-run      # print the commands instead
```

It deploys one app at a time, runs SoulX's `populate_weights` before SoulX's deploy, and
asks Modal's API for each URL rather than parsing the deploy log, so wrapped lines and
truncated labels can't garble it. Before deploying an app that needs gated Hugging Face
weights, it checks that the secret exists (see below). `--sync-env` is the quick fix if
you deployed by hand, or if `.env` went stale after a redeploy under another workspace.

The manual equivalent, and what the tool runs under the hood:

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

**On a new workspace, set `MODAL_IMAGE_BUILDER_VERSION=2025.06` before a deploy by
hand.** A Modal workspace created now (seen 2026-11-10) builds images with builder
version `2026.10` by default. The Modal client in the toolkit's `uv.lock` (1.5) doesn't
know that version and refuses every deploy before building anything:

```
This version of the modal client supports the following image builder versions:
{'2023.12', '2024.04', '2024.10', '2025.06'}. You are using '2026.10'.
Please update your client library (pip install --upgrade modal).
```

`modal_setup.py` handles it: it sees that error, retries with `2025.06`, and keeps that
version for the rest of the run. By hand, set the variable in the shell first:

```bash
export MODAL_IMAGE_BUILDER_VERSION=2025.06   # macOS, Linux, Git Bash
uv run modal deploy docker/modal-qwen3-tts/app.py
```

```powershell
$env:MODAL_IMAGE_BUILDER_VERSION = '2025.06'
uv run modal deploy docker/modal-qwen3-tts/app.py
```

Upgrading `modal` instead is not the fix: `2026.10` drops Python 3.10, which the
`sadtalker` and `propainter` images still use. A workspace older than that keeps its own
builder version and needs nothing. Setting the variable there would rebuild every image
under the new version, so leave it unset unless a deploy asks for it. `modal_setup.py`
also reads it from `.env`; `uv run modal deploy` doesn't.

**Every GPU app needs a payment method on the account.** On a new workspace even the
`A10G` apps fail at deploy time with `Please add a payment method to use A10G GPU
functions` (seen 2026-10-02), not just the bigger GPUs. The Starter plan's $30/month of
included compute still covers typical use. The image still gets built and cached before
that error, so re-deploying after adding a payment method is fast. `ltx2` additionally
needs a Modal secret holding your Hugging Face token under the key `HF_TOKEN`. The apps
look for a secret named `huggingface-token`. Modal's Hugging Face secret template fills in
`huggingface-secret` as the name instead. Either rename it, or deploy with
`HF_SECRET_NAME=huggingface-secret` set. `modal_setup.py` sets that variable for you when
`huggingface-secret` is the only one it finds, and `--hf-secret NAME` picks any other
name. The token's
account also needs access to Google's gated Gemma 3 text encoder,
https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized. Google approves those
requests by hand, so ask well before you deploy. The deploy fails on a missing secret
*before* it ever reaches the GPU check, and without Gemma access with a 401/403 mid-build.
(The LTX-2.5 opt-in below needs access to `Lightricks/LTX-2.5` instead of Gemma 3.)

**Newer non-commercial models are opt-in.** `flux2` and `image-edit` deploy the
Apache-2.0 FLUX.2-klein-4B and Qwen-Image-Edit-2511 by default. A deploy-time variable
swaps in the newer models, which are licensed for non-commercial use only:

```bash
# FLUX.2-klein-9B: FLUX Non-Commercial License (outputs need filtering or manual
# review). Gated -- needs the huggingface-token secret and the terms accepted at
# https://huggingface.co/black-forest-labs/FLUX.2-klein-9B. Runs on an L40S.
FLUX2_MODEL=klein-9b uv run modal deploy --strategy recreate docker/modal-flux2/app.py

# Qwen-Image-2.1: Qwen Research License ("research or evaluation purposes only").
# Up to 10 input images, RGBA in/out.
IMAGE_EDIT_MODEL=qwen-image-2.1 uv run modal deploy --strategy recreate docker/modal-image-edit/app.py
```

`ltx2` has an opt-in as well: LTX-2.5 in place of LTX-2.3. It is not a non-commercial
model. The LTX-2.x Community License is free under $10M annual revenue, commercial use
included, and forbids removing watermark or provenance features. It stays opt-in until it
has been benchmarked against 2.3 (see `docs/ltx2.md`). The repo is gated: accept the terms
at https://huggingface.co/Lightricks/LTX-2.5 with the `huggingface-token` account first
(approval is automatic, but you share your contact info):

```bash
# LTX-2.5 builds its own ~80GB image on the first deploy. Same A100-80GB GPU.
LTX2_VERSION=2.5 uv run modal deploy --strategy recreate docker/modal-ltx2/app.py
```

With `modal_setup.py`, pass `--recreate` in place of `--strategy recreate`, e.g.
`FLUX2_MODEL=klein-9b uv run tools/modal_setup.py flux2 --recreate`.

The endpoint URL stays the same; deploy again without the variable, also with
`--strategy recreate`, to switch back. `--strategy recreate` stops the containers that
were running before the deploy, so a warm container from the previous variant can't
serve the next request with the old model. The response's `model` field (printed by
the tools as `Model:`) says which one ran, so check it right after switching.

The inline `VAR=value command` form above is Git Bash / macOS / Linux syntax. In
PowerShell, set the variable first: `$env:FLUX2_MODEL='klein-9b'`, then run the deploy.
It stays set for the rest of that PowerShell session, so clear it with
`Remove-Item Env:FLUX2_MODEL` before you deploy the default again. The same goes for
`IMAGE_EDIT_MODEL` and `LTX2_VERSION`.

If you deploy by hand, each deploy prints an endpoint URL like:
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

> **Tip:** skip the copying: `uv run tools/modal_setup.py --sync-env` reads every deployed
> app's URL from Modal and writes it to `.env`.

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
| `flux2` | FLUX.2 klein 4B (9B opt-in, non-commercial) | AI image generation | ~$0.01-0.03 |
| `image_edit` | Qwen-Image-Edit-2511 (Qwen-Image-2.1 opt-in, non-commercial) | AI image editing, style transfer | ~$0.02-0.05 |
| `upscale` | RealESRGAN | AI image upscaling (2x/4x) | ~$0.005-0.02 |
| `music_gen` | ACE-Step 1.5 XL Turbo (4B) | AI music generation | Free (acemusic) / ~$0.02-0.10 (Modal) |
| `sadtalker` | SadTalker | Talking head video | ~$0.05-0.30 |
| `soulx` | SoulX-FlashHead 1.3B Pro + Lite | Talking head video, aspect-preserving (`--model lite` for drafts) | Pro ~$0.0024 / Lite ~$0.0004 per second of output |
| `dewatermark` | ProPainter | AI video inpainting | ~$0.05-0.50 |

Most apps use A10G GPUs (24GB VRAM). The exceptions: `flux2` moves to an L40S (48GB)
when deployed with klein-9B, which is ~35GB resident in bf16; `image_edit` and `ltx2` run
on an A100-80GB.

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
| `qwen3_tts` | ~50s (measured) | ~5-15s |
| `flux2` | ~46s (measured, klein-4B); ~50s (klein-9B on L40S) | ~6-8s |
| `image_edit` | ~2 min (measured, 2511); ~85s (Qwen-Image-2.1) | ~13s at 8 steps (2511); ~26s at 40 steps (2.1) |
| `upscale` | ~25-30s | ~3-5s |
| `music_gen` | ~50s (measured, XL Turbo) | ~4s per 10s of audio |
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

# Which apps are deployed, and does .env agree?
uv run tools/modal_setup.py --list

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
| **Setup** | `tools/modal_setup.py` | `--setup` flag per tool |
| **Cold start** | Faster (cached layers) | Slower (Docker pull) |
| **Invocation** | Synchronous POST | Async submit + poll |
| **Auth** | Token optional for web endpoints | `RUNPOD_API_KEY` required |

See [runpod-setup.md](runpod-setup.md) for RunPod-specific instructions.
