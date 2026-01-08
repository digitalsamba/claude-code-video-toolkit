# LTX-2 - AI Video Generation

Generate AI video from text prompts or images using the LTX-2 19B model.

## Quick Start

```bash
# Text-to-video
python tools/ltx2.py --prompt "A cat playing with yarn" --output cat.mp4

# Image-to-video
python tools/ltx2.py --image photo.jpg --prompt "Make them smile and wave" --output animated.mp4

# With preset
python tools/ltx2.py --prompt "Mountain landscape at sunset" --preset hd --output landscape.mp4
```

## Setup

1. Add your RunPod API key to `.env`:
   ```
   RUNPOD_API_KEY=your_key_here
   ```

2. Run setup to create the endpoint:
   ```bash
   python tools/ltx2.py --setup
   ```

3. **Important:** In RunPod, attach a 100GB+ network volume mounted at `/runpod-volume` for model caching. Without this, models (~55GB) download on every cold start.

## Presets

| Preset | Resolution | Flow | Best For |
|--------|------------|------|----------|
| `default` | 768x512 | pro | General use |
| `fast` | 512x384 | fast | Quick iteration |
| `hd` | 1280x720 | pro | High quality renders |
| `portrait` | 512x768 | pro | Vertical video |
| `landscape` | 768x512 | pro | Horizontal video |
| `square` | 512x512 | pro | Social media |

## Parameters

### Core Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt`, `-p` | required | Text description of the video |
| `--image`, `-i` | none | Source image for image-to-video |
| `--output`, `-o` | ltx2_output.mp4 | Output file path |
| `--preset` | none | Use preset configuration |

### Generation Options

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--width` | 768 | divisible by 32 | Video width |
| `--height` | 512 | divisible by 32 | Video height |
| `--duration` | 5s | 1-20s | Video duration in seconds |
| `--fps` | 24 | 24-50 | Frame rate |
| `--flow` | pro | pro, fast | Quality vs speed |
| `--steps` | auto | 10-100 | Inference steps |
| `--guidance` | 3.0 | 1.0-10.0 | CFG guidance scale |
| `--seed` | random | any int | Random seed |

### Flow Modes

| Mode | Steps | Speed | Quality | Use Case |
|------|-------|-------|---------|----------|
| `pro` | 50 | Baseline | Best | Final renders |
| `fast` | 25 | 2x faster | Good | Prototyping |

## Examples

### Text-to-Video (Basic)
```bash
python tools/ltx2.py \
  --prompt "A serene mountain lake at sunrise with mist rising from the water" \
  --output sunrise.mp4
```

### Text-to-Video (HD)
```bash
python tools/ltx2.py \
  --prompt "A busy Tokyo street at night with neon lights" \
  --preset hd \
  --duration 8 \
  --output tokyo.mp4
```

### Image-to-Video
```bash
python tools/ltx2.py \
  --image portrait.jpg \
  --prompt "The person smiles and turns their head slightly" \
  --output animated_portrait.mp4
```

### Fast Prototyping
```bash
python tools/ltx2.py \
  --prompt "Abstract colorful smoke swirling" \
  --preset fast \
  --output smoke_test.mp4
```

### Custom Settings
```bash
python tools/ltx2.py \
  --prompt "Ocean waves crashing on rocks" \
  --width 1280 \
  --height 720 \
  --duration 10 \
  --flow pro \
  --guidance 3.5 \
  --seed 42 \
  --output waves.mp4
```

### Reproducible Generation
```bash
# Generate with specific seed
python tools/ltx2.py \
  --prompt "A butterfly landing on a flower" \
  --seed 12345 \
  --output butterfly.mp4

# Same seed = same output
python tools/ltx2.py \
  --prompt "A butterfly landing on a flower" \
  --seed 12345 \
  --output butterfly_again.mp4
```

## Prompting Tips

### Good Prompts

**Be specific and descriptive:**
```
"A young woman with long brown hair walks through an autumn forest,
golden leaves falling around her, soft natural lighting, cinematic"
```

**Include camera movement:**
```
"Drone shot of a winding coastal road at sunset,
smooth tracking movement, waves crashing below, golden hour lighting"
```

**Describe atmosphere:**
```
"A cozy coffee shop interior, steam rising from a cup,
warm lighting, rain visible through the window, peaceful atmosphere"
```

### Prompt Structure

1. **Subject** - Who/what is in the video
2. **Action** - What's happening
3. **Setting** - Where it takes place
4. **Style** - Visual style (cinematic, documentary, etc.)
5. **Lighting** - Natural, studio, golden hour, etc.
6. **Camera** - Movement, angle, shot type

### Negative Prompt

The default negative prompt helps avoid common artifacts:
```
worst quality, inconsistent motion, blurry, jittery, distorted,
low resolution, watermark, text
```

Override with `--negative-prompt` if needed.

## Performance

| Duration | Frames | Pro Time | Fast Time | Cost (4090) |
|----------|--------|----------|-----------|-------------|
| 5s | 121 | ~45s | ~25s | ~$0.03 |
| 10s | 241 | ~90s | ~50s | ~$0.06 |
| 20s | 481 | ~180s | ~100s | ~$0.12 |

**Cold start:** Add ~60-90s (cached models) or ~10-15min (first run)

## Troubleshooting

### "RUNPOD_LTX2_ENDPOINT_ID not set"
Run `python tools/ltx2.py --setup` first.

### Slow cold start
Ensure network volume is mounted at `/runpod-volume` in RunPod. First run downloads ~55GB of models.

### Out of memory
- Use `--preset fast` or `--flow fast`
- Reduce resolution
- Reduce duration

### Poor quality
- Use `--flow pro`
- Increase resolution
- Improve prompt specificity
- Adjust guidance (try 2.5-4.0)

### Inconsistent motion
- Try different seeds
- Simplify the prompt
- Describe motion explicitly

### "width/height must be divisible by 32"
LTX-2 requires dimensions divisible by 32. Use presets or valid values like 512, 768, 1024, 1280.

## Integration with Toolkit

LTX-2 can be used alongside other toolkit tools:

```bash
# 1. Generate AI video
python tools/ltx2.py \
  --prompt "A presenter in a studio" \
  --output presenter_bg.mp4

# 2. Generate voiceover
python tools/voiceover.py \
  --script script.md \
  --output narration.mp3

# 3. Combine in Remotion project
# Use presenter_bg.mp4 as background/B-roll
# Use narration.mp3 as voiceover track
```

## Model Information

- **Model:** Lightricks/LTX-2
- **Parameters:** 19B
- **Variant:** FP8 quantized (27GB)
- **Architecture:** DiT (Diffusion Transformer)
- **Capabilities:** Text-to-video, image-to-video
- **License:** ltx-2-community-license-agreement

## References

- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 Paper](https://arxiv.org/abs/2601.03233)
