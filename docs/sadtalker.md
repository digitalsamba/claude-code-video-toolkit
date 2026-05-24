# SadTalker - Talking Head Video Generation

Generate realistic talking head videos from a portrait image and audio file.

## Quick Start

```bash
# Basic usage
python tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

# With preset
python tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --preset natural --output talking.mp4
```

## When NOT to Use SadTalker

SadTalker is trained on photoreal human faces. It struggles with:

- **Stylized / illustrated characters** — fantasy art, anime, painted portraits
- **Heavy facial hair** — mouth-region detection fails on full beards
- **Masks, helmets, face coverings** — anything occluding the lower face
- **Non-frontal angles** — anything past ~30° produces artifacts

For these cases, use **LTX-2 image-to-video** instead. It animates the whole image with motion + atmosphere rather than trying to match phonemes — for short cameos where lip sync isn't critical, the result is usually better. See the `ltx2` skill for the "Stylized Character Cameo" pattern.

## Setup

1. Add your RunPod API key to `.env`:
   ```
   RUNPOD_API_KEY=your_key_here
   ```

2. Run setup to create the endpoint:
   ```bash
   python tools/sadtalker.py --setup
   ```

## Presets

| Preset | Description | Settings |
|--------|-------------|----------|
| `default` | Balanced default settings | pose_style=0, expression_scale=1.0 |
| `natural` | Natural head movement | pose_style=45, expression_scale=1.0 |
| `expressive` | Animated, engaging | pose_style=45, expression_scale=1.3 |
| `professional` | Calm, minimal movement | still=true, expression_scale=0.8 |
| `fullbody` | Full body shots | still=true, preprocess=full |

## Parameters

### Core Settings

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--size` | 256 | 256, 512 | Face resolution. 512 is sharper but 2x slower |
| `--expression-scale` | 1.0 | 0.5-2.0 | Expression intensity. Higher = more animated |
| `--pose-style` | 0 | 0-45 | Head movement style. 45 often gives best results |
| `--still` | off | flag | Reduces head movement, good for professional content |
| `--preprocess` | crop | crop/resize/full | How to handle the input image |
| `--no-enhance` | off | flag | Skip GFPGAN face enhancement (faster) |

### Preprocess Modes

| Mode | Best For | Notes |
|------|----------|-------|
| `crop` | Close-up portraits | Default. Crops face region for animation |
| `resize` | ID-style photos | Resizes whole image. Bad for full-body shots |
| `full` | Full body images | Crops face, animates, pastes back. Use with `--still` |

## Image Guidelines

### Good Images
- Clear, front-facing portrait
- Face takes 30-70% of frame
- Good lighting, neutral expression
- High resolution (512px+ recommended)

### Avoid
- Side profiles (>30 degrees)
- Multiple faces
- Heavy occlusion (hands covering face)
- Anime/cartoon images (not supported)
- Very low resolution

## Performance

| Audio Duration | Chunks | Processing Time | Cost (approx) |
|---------------|--------|-----------------|---------------|
| 15 seconds | 1 | ~3.5 min | $0.02 |
| 1 minute | 2 | ~7 min | $0.05 |
| 3 minutes | 4 | ~14 min | $0.15 |

- Audio is automatically split into 45-second chunks to prevent drift
- Uses RTX 4090 GPU ($0.00074/sec)
- Cold start adds ~60 seconds on first request

## Examples

### Natural Talking Head
```bash
python tools/sadtalker.py \
  --image portrait.png \
  --audio narration.mp3 \
  --preset natural \
  --output talking.mp4
```

### Professional/Calm Style
```bash
python tools/sadtalker.py \
  --image headshot.png \
  --audio presentation.mp3 \
  --preset professional \
  --size 512 \
  --output presenter.mp4
```

### Expressive Animation
```bash
python tools/sadtalker.py \
  --image avatar.png \
  --audio excited_speech.mp3 \
  --preset expressive \
  --output animated.mp4
```

### Full Body Shot
```bash
python tools/sadtalker.py \
  --image fullbody.png \
  --audio voiceover.mp3 \
  --preset fullbody \
  --output fullbody_talking.mp4
```

### Fine-Tuned Settings
```bash
python tools/sadtalker.py \
  --image portrait.png \
  --audio speech.mp3 \
  --pose-style 45 \
  --expression-scale 1.2 \
  --size 512 \
  --output custom.mp4
```

## Long Audio & Job Recovery

### Auto-Timeout

Timeout is automatically calculated based on audio duration:
- ~4 minutes processing per minute of audio
- Plus 2 minute buffer for cold start/upload
- Example: 3.5 min audio → ~16 min timeout

To override: `--timeout 1800` (30 minutes)

### Recovering Timed-Out Jobs

If your client times out but the job completes on RunPod:

```bash
# Get job ID from the error output, then retrieve results
python tools/sadtalker.py --retrieve JOB_ID --output result.mp4

# Example
python tools/sadtalker.py --retrieve 7d69546a-3d31-4f74-bf12-b1c19a2f4d3c-u1 --output narrator.mp4
```

Jobs persist on RunPod for ~24 hours, so you can retrieve later.

### Checking Job Status

```bash
# Quick status check via RunPod API
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$RUNPOD_SADTALKER_ENDPOINT_ID/status/JOB_ID
```

## Output Dimensions & Background

**Important limitations:**

| Aspect | Behavior |
|--------|----------|
| Output size | Matches input image dimensions |
| Background | Preserved from source image |
| Face area | Only the face is animated |

SadTalker doesn't resize or change backgrounds. To customize:

1. **Pre-process input image:**
   ```bash
   # Resize image before SadTalker
   ffmpeg -i portrait.png -vf scale=1280:720 portrait_720p.png

   # Or use image_edit.py for background changes
   python tools/image_edit.py --input portrait.png --background studio --output portrait_studio.png
   ```

2. **Post-process output video:**
   ```bash
   # Resize after generation
   ffmpeg -i talking.mp4 -vf scale=640:360 talking_small.mp4
   ```

For **narrator PiP**, output dimensions don't matter much - Remotion scales/crops to fit the corner overlay.

## Troubleshooting

### "No face detected"
- Ensure face is clearly visible and front-facing
- Try a higher resolution image
- Face should be 30-70% of frame

### Lip sync looks off
- Audio quality matters - clear speech works best
- Try different `--expression-scale` values
- Ensure audio isn't clipped or distorted

### Output looks blurry
- Use `--size 512` for higher resolution
- Don't use `--no-enhance`
- Start with a higher resolution source image

### Processing too slow
- Use `--size 256` (default)
- Use `--no-enhance` for faster processing
- Shorter audio = faster processing

### Client timed out
- Job may still be running on RunPod
- Check RunPod dashboard or use `--retrieve JOB_ID`
- For long audio (>2 min), expect 10-20+ minutes processing
- Use `--timeout` to set explicit timeout if auto-calculation isn't enough

## Advanced: Custom Experimentation

For fine-tuning results, try these combinations:

```bash
# More animated, natural movement
--pose-style 45 --expression-scale 1.3

# Subtle, professional
--still --expression-scale 0.7

# Maximum quality (slower)
--size 512 --pose-style 45

# Quick test (faster)
--size 256 --no-enhance
```

## Integration with NarratorPiP

When creating talking head videos for the NarratorPiP component, follow these critical patterns:

### 1. Match Image Aspect Ratio to Target

**This is critical.** SadTalker preserves input dimensions. If your image doesn't match the target aspect ratio, the video won't fit properly.

| Target | Image Aspect Ratio | Example Dimensions |
|--------|-------------------|-------------------|
| NarratorPiP (standard) | **16:9** | 640×360, 1280×720 |
| TikTok/Vertical | 9:16 or 1:1 | 720×1280, 720×720 |
| Square overlay | 1:1 | 512×512, 720×720 |

### 2. Always Use `--preprocess full`

**Without this flag, SadTalker outputs a square face crop, not your original dimensions.**

| Mode | Output | Use for NarratorPiP? |
|------|--------|---------------------|
| `crop` (default) | Square face only | No |
| `full` | Original dimensions | **Yes** |

### 3. Let Users Crop Their Own Images

Humans frame faces better than automated cropping. Guide users to:
- Provide a 16:9 image with face centered
- Include head + shoulders
- Leave some headroom but not excessive sky

### 4. Recommended Command for NarratorPiP

```bash
python tools/sadtalker.py \
  --image presenter_16x9.png \
  --audio voiceover.mp3 \
  --still --expression-scale 0.8 --preprocess full \
  --output narrator.mp4
```

Key flags:
- `--preprocess full` — preserves 16:9 dimensions
- `--still` — reduces distracting head movement
- `--expression-scale 0.8` — calmer, more professional look

### 5. Full Workflow Example

```bash
# 1. Generate voiceover
python tools/voiceover.py --script script.md --output narration.mp3

# 2. Prepare 16:9 image (user crops manually for best framing)
# presenter_16x9.png should be ~640x360 or similar 16:9 ratio

# 3. Create talking head with NarratorPiP-optimized settings
python tools/sadtalker.py \
  --image presenter_16x9.png \
  --audio narration.mp3 \
  --still --expression-scale 0.8 --preprocess full \
  --output narrator.mp4

# 4. Use in Remotion sprint-config.ts:
# narrator: {
#   enabled: true,
#   videoFile: 'videos/narrator.mp4',
#   position: 'bottom-right',
#   size: 'md',
# }
```

### Output Upscaling

Note: With `--preprocess full`, SadTalker outputs at **2× input resolution**:
- Input: 658×370 → Output: 1316×740
- This is fine for NarratorPiP as Remotion scales to fit

## References

- [SadTalker GitHub](https://github.com/OpenTalker/SadTalker)
- [SadTalker Best Practices](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)
- [CVPR 2023 Paper](https://arxiv.org/abs/2211.12194)
