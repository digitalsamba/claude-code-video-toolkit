---
name: ltx2
description: AI video generation with LTX-2.3 22B — text-to-video, image-to-video clips for video production. Use when generating video clips, animating images, creating b-roll, animated backgrounds, or motion content. Triggers include video generation, animate image, b-roll, motion, video clip, text-to-video, image-to-video.
---

# LTX-2.3 Video Generation

Generate ~5 second video clips from text prompts or images using the LTX-2.3 22B DiT model.
Runs on Modal (A100-80GB). Requires `MODAL_LTX2_ENDPOINT_URL` in `.env`.

## Quick Reference

```bash
# Text-to-video
python3 tools/ltx2.py --prompt "A sunset over the ocean, golden light on waves, cinematic" --output sunset.mp4

# Image-to-video (animate a still image)
python3 tools/ltx2.py --prompt "Gentle camera drift, soft ambient motion" --input photo.jpg --output animated.mp4

# Custom resolution and duration
python3 tools/ltx2.py --prompt "..." --width 1024 --height 576 --num-frames 161 --output wide.mp4

# Fast mode (fewer steps, quicker)
python3 tools/ltx2.py --prompt "..." --quality fast --output quick.mp4

# Reproducible output
python3 tools/ltx2.py --prompt "..." --seed 42 --output reproducible.mp4
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | (required) | Text description of the video |
| `--input` | - | Input image for image-to-video |
| `--width` | 768 | Video width (divisible by 64) |
| `--height` | 512 | Video height (divisible by 64) |
| `--num-frames` | 121 | Frame count, must satisfy `(n-1) % 8 == 0` |
| `--fps` | 24 | Frames per second |
| `--quality` | standard | `standard` (30 steps) or `fast` (15 steps) |
| `--steps` | 30 | Override inference steps directly |
| `--seed` | random | Seed for reproducibility |
| `--output` | auto | Output file path |
| `--negative-prompt` | sensible default | What to avoid |
| `--lora` | none | Style LoRA preset. Currently: `crt-terminal`. |

## Style LoRAs

Style LoRAs bias the output toward a specific visual aesthetic. They're baked into the Modal image and selected per-request; switching LoRAs forces a pipeline rebuild (~60s one-time cost per container lifetime per switch).

### `crt-terminal` — CRT / pixel-art terminals

Base: LTX-2.3 22B, trained by [@lovis93](https://huggingface.co/lovis93/crt-animation-terminal-ltx-2.3-lora) (Apache 2.0).

```bash
# Trigger word is auto-prepended — write the prompt normally
python3 tools/ltx2.py --lora crt-terminal \
  --prompt "a terminal typing out \"\\$ claude --continue\" character by character in glowing green pixel font, scanlines, phosphor glow, low choppy frame rate, hacker mood" \
  --output crt_claude.mp4
```

**What the preset changes:**
- Prepends `crtanim,` to the prompt (the LoRA's trigger word)
- Defaults to 1024×1024, 121 frames (the ratio it was trained on)
- Relaxes the default negative prompt so on-screen text isn't filtered out

**Prompt pattern:** `<CRT aesthetic> → <color palette> → <animation style> → <subject> → <literal text in quotes> → <mood>`. Keep on-screen text to 1–3 words — the model can't render long strings reliably. The LoRA prefers static framing; ask for camera moves explicitly if you want them.

## Valid Frame Counts

`(n - 1) % 8 == 0`: 25 (~1s), 49 (~2s), 73 (~3s), 97 (~4s), **121 (~5s default)**, 161 (~6.7s), 193 (~8s max practical).

## Common Resolutions

| Resolution | Ratio | Notes |
|------------|-------|-------|
| 768x512 | 3:2 | Default, good balance |
| 512x512 | 1:1 | Square, fastest |
| 1024x576 | 16:9 | Widescreen |
| 576x1024 | 9:16 | Portrait/vertical |

## Prompting Guide

LTX-2 responds well to cinematographic descriptions. Layer these dimensions:

- **Camera:** "Slow dolly forward", "Aerial drone shot", "Tracking shot", "Static wide angle"
- **Lighting:** "Golden hour", "Cinematic lighting", "Neon-lit", "Soft diffused light"
- **Motion:** "Timelapse of...", "Slow motion", "Gentle camera drift", "Gradually transitions"
- **Style:** "Shot on 35mm film", "Documentary style", "Clean minimal aesthetic"
- **Negative:** Always implicitly avoids "worst quality, blurry, jittery, watermark, text, logo"

Keep prompts under 200 words. Be specific about the scene.

### Good Prompts

```
# Atmospheric b-roll
"Aerial drone shot slowly flying over turquoise ocean waves breaking on white sand, golden hour sunlight, cinematic"

# Product/tech scene
"Close-up of hands typing on a mechanical keyboard, shallow depth of field, soft desk lamp lighting, cozy atmosphere"

# Abstract background
"Dark moody abstract background with flowing blue light streaks, subtle geometric grid, bokeh particles floating, cinematic tech atmosphere"

# Animate a portrait
"Professional headshot, subtle natural head movement, confident warm expression, studio lighting, shallow depth of field"

# Animate a slide/screenshot
"Gentle subtle particle effects floating across a presentation slide, soft ambient light shifts, very slight camera drift"
```

### Bad Prompts

```
# Too vague
"A cool video"

# Too many competing ideas
"A cat riding a skateboard while juggling fire on the moon during a thunderstorm"

# Describing text/UI (model can't render text reliably)
"A website showing the text 'Welcome to our platform'"
```

## Video Production Use Cases

### B-Roll Clips
Generate atmospheric 5s shots for cutaways between narrated scenes:
```bash
python3 tools/ltx2.py --prompt "Futuristic holographic interface, glowing data visualizations, clean workspace, cinematic" --output broll_tech.mp4
python3 tools/ltx2.py --prompt "Aerial view of European city at golden hour, modern architecture" --output broll_europe.mp4
```

### Animated Slide Backgrounds
Feed a slide screenshot and add subtle motion:
```bash
python3 tools/ltx2.py --prompt "Gentle particle effects, soft ambient light shifts, very slight camera drift" --input slide.png --output animated_slide.mp4
```

### Animated Portraits
Bring still headshots to life:
```bash
python3 tools/ltx2.py --prompt "Subtle natural head movement, warm expression, professional lighting" --input headshot.png --output animated_portrait.mp4
```

### Stylized Character Cameo (SadTalker Alternative)
For non-realistic faces — fantasy characters, masked figures, heavy beards, helmets, illustrations — SadTalker often produces uncanny or broken lip sync because it's trained on photoreal humans. LTX-2 image-to-video is frequently a better choice when **lip-sync precision isn't critical** (the viewer's brain fills in the gap as long as something is moving). Prompt for *motion + atmosphere*, not phonemes:

```bash
python3 tools/ltx2.py \
  --input character_portrait.png \
  --prompt "Ancient warrior speaks slowly with gravitas, beard shifts subtly, glowing aura pulses, embers drift past, slow head movement, cinematic close-up, mystical atmosphere" \
  --width 768 --height 768 \
  --output character_speaking.mp4
```

**When LTX-2 wins over SadTalker:**
- Stylized / illustrated / fantasy characters
- Heavy facial hair or accessories obscuring the mouth
- Masked or helmeted figures
- Short cameo lines where atmosphere matters more than precision
- Dramatic VO rather than dialogue

**When SadTalker still wins:**
- Photoreal human presenters
- Full sentences where mouth shape needs to match phonemes
- Tutorials / talking-head explainers where the viewer is effectively reading lips

### Branded Intro/Outro
Generate abstract motion backgrounds for title cards:
```bash
python3 tools/ltx2.py --prompt "Dark moody background with flowing blue and coral light streaks, bokeh particles, cinematic tech atmosphere, no text" --output intro_bg.mp4
```

### Combining with Other Tools

LTX-2 generates raw clips. Combine with the rest of the toolkit:

| Workflow | Tools |
|----------|-------|
| Generate clip → upscale | `ltx2.py` → `upscale.py` |
| Generate clip → add to Remotion | `ltx2.py` → use as `<OffthreadVideo>` in composition |
| Generate image → animate | `flux2.py` → `ltx2.py --input` |
| Generate clip → extract audio | `ltx2.py` → `ffmpeg -i clip.mp4 -vn audio.wav` |
| Generate clip → add voiceover | `ltx2.py` → mix with `qwen3_tts.py` output |

## Technical Details

- **Model:** LTX-2.3 22B DiT (Lightricks), bf16
- **GPU:** A100-80GB on Modal (~$4.68/hr)
- **Inference:** ~2.5 min per clip (768x512, 121 frames, 30 steps)
- **Cost:** ~$0.20-0.25 per 5s clip
- **Cold start:** ~60-90s (loading ~55GB weights)
- **Output:** H.264 MP4 with synchronized ambient audio (24fps)
- **Max duration:** ~8s (193 frames) per clip

### Known Limitations

- **Training data artifacts:** ~30% of generations may have unwanted logos/text from training data. Re-run with different `--seed`.
- **Text rendering:** Cannot reliably generate readable text in video. Use Remotion overlays instead.
- **Max duration:** ~8s per clip. Longer content needs stitching.
- **Audio:** Generated audio is ambient/environmental only. Use voiceover/music tools for speech and music.
- **License:** Community License — free under $10M revenue, commercial license needed above that.

## Setup

```bash
# 1. Create Modal secret for HuggingFace (one-time)
modal secret create huggingface-token HF_TOKEN=hf_your_token

# 2. Deploy (downloads ~55GB of weights, takes ~10 min)
modal deploy docker/modal-ltx2/app.py

# 3. Save endpoint URL to .env
echo "MODAL_LTX2_ENDPOINT_URL=https://yourname--video-toolkit-ltx2-ltx2-generate.modal.run" >> .env

# 4. Test
python3 tools/ltx2.py --prompt "A candle flickering on a dark table, cinematic" --output test.mp4
```

**Important:** HuggingFace token needs read-access scope. Accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) before deploying. Unauthenticated downloads are severely rate-limited.
