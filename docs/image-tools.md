# Image tools and cloud GPU

Cloud GPU providers, which image generator to use, image editing, upscaling and watermark removal. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Cloud GPU Providers

All cloud GPU tools support two providers via `--cloud runpod|modal`. RunPod is the default. Modal was added as a reliability fallback after RunPod outages, and offers faster cold starts.

```bash
# --- RunPod setup (automated, one-time per tool) ---
echo "RUNPOD_API_KEY=your_key_here" >> .env
uv run tools/image_edit.py --setup
uv run tools/upscale.py --setup
uv run tools/qwen3_tts.py --setup
uv run tools/music_gen.py --setup

# --- Modal setup (deploys + writes each endpoint URL to .env) ---
uv sync --extra modal && uv run modal setup
uv run tools/modal_setup.py upscale image_edit         # or --all (everything but ltx2)
uv run tools/modal_setup.py --list                     # deployed? .env in sync?
uv run tools/modal_setup.py --sync-env                 # deployed by hand? fill .env
# See docs/modal-setup.md for full guide
```

`modal_setup.py` reads URLs from Modal's API, never from the deploy log, which wraps and
truncates them. Gated apps (`ltx2`, `flux2` with `FLUX2_MODEL=klein-9b`) need a Modal
secret with `HF_TOKEN`. The apps default to the name `huggingface-token`, but Modal's
template names it `huggingface-secret`. The tool detects that and sets `HF_SECRET_NAME`;
`--hf-secret NAME` overrides it.

## AI Image Generation (FLUX.2 vs Ideogram 4)

The toolkit has **two** text-to-image generators. They barely overlap, the deciding factor is
**whether the image needs legible baked-in text**.

```bash
# FLUX.2 Klein — text-FREE backgrounds + image editing (self-hosted, free)
#   klein-4B by default (Apache-2.0). FLUX2_MODEL=klein-9b at Modal deploy time opts into
#   klein-9B (FLUX Non-Commercial, outputs need review) -- see docs/modal-setup.md
uv run tools/flux2.py --preset title-bg --brand digital-samba   # background for Remotion text overlay
uv run tools/flux2.py --prompt "Abstract tech background, no text"

# Ideogram 4 — legible IN-IMAGE text + exact color/layout (hosted API, ~$0.03-0.10/img, commercial-OK)
uv run tools/ideogram4.py --json caption.json --output title.png   # text baked into the image
uv run tools/ideogram4.py --prompt "Thumbnail: 'SHIP FASTER' bold" --output thumb.png
```

**The key distinction, baked-in text vs. background-for-overlay:**

- **FLUX.2 presets deliberately produce text-*free* backgrounds** (`title-bg`, `cta`, `thumbnail`
  all end with "no text, no words, no letters"). The intended pattern is **FLUX background →
  Remotion renders the text on top**, so the text stays editable, animatable per-letter, and
  frame-accurate. This is the right default for sprint-review **title cards / lower-thirds**.
- **Ideogram 4 bakes legible designed text *into* a flat PNG.** Pixel-perfect typography + exact
  hex colors in one shot, but static. Reach for it when **the text *is* the design**: YouTube/social
  thumbnails (static anyway), quote/stat cards, in-scene signage/logos, stylized text effects that
  Remotion overlays can't easily do. It fills a real gap, FLUX and LTX-2 both garble in-image text.

| Need | Use |
|------|-----|
| Animated/editable title text, lower-thirds, sprint-review title cards | **FLUX.2 bg + Remotion text** |
| Atmospheric/abstract backgrounds, problem/solution illustrations, presenter backdrops | **FLUX.2** presets |
| Finished graphic where typography is the design, thumbnails, quote cards, in-scene signage | **Ideogram 4** |
| Edit an existing photo (clothing, reframe, style, background) | **image_edit** (neither generator edits) |

Ideogram 4 chains with the processors like any generator: `ideogram4 → upscale.py` (crisp 4K),
`ideogram4 → ltx2.py --input` (animate the still), or drop the PNG straight into Remotion `<Img>`.
Ideogram 4 is generate-only here (no editing). See `.claude/skills/ideogram4/` for the JSON caption
format, Claude authors the caption as the "magic prompt" expander; needs `IDEOGRAM_API_KEY` in `.env`.
`--model 4.5` opts into Ideogram 4.5 on the v2 API (same flags; `--speed` maps to its quality
tiers). 4.5 has been run live: JSON captions render near-perfect text, plain prompts add
garbled filler text, so always use JSON. 4.0 stays the default because 4.5 sizes must be
multiples of 32, which rules out 1280x720 (`/thumbnail` relies on 4.0 for that exact size).

## AI Image Editing

```bash

# Image editing (Qwen-Image-Edit-2511, Apache-2.0; IMAGE_EDIT_MODEL=qwen-image-2.1 at Modal
# deploy time opts into Qwen-Image-2.1, non-commercial)
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal
uv run tools/image_edit.py --input photo.jpg --style cyberpunk
uv run tools/image_edit.py --input photo.jpg --background office
uv run tools/image_edit.py --list-presets  # Full preset list

# Upscaling (RealESRGAN)
uv run tools/upscale.py --input photo.jpg --output photo_4x.png --cloud runpod
uv run tools/upscale.py --input photo.jpg --scale 2 --model anime --face-enhance --cloud runpod
```

See `docs/qwen-edit-patterns.md` and `.claude/skills/qwen-edit/` for prompting guidance.

## Watermark Removal

```bash
# Locate watermark coordinates
uv run tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/
uv run tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

# Remove watermark (RunPod)
uv run tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod
uv run tools/dewatermark.py --setup  # One-time setup
```

**Workflow:** grid overlay → note coordinates → verify with `--region` → remove with dewatermark.

**Local mode** requires NVIDIA GPU (8GB+ VRAM). Mac users should use `--runpod`.
