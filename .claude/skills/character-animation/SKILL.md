---
name: character-animation
description: Animate static elements within video slides using AI. Identifies characters and environmental elements, generates motion with subtle, physically plausible movement. Use for NotebookLM videos, presentations, or any slide-based content.
status: alpha
---

# Character Animation

Animate static illustrations and images within video slides.

> **Status:** Alpha - Phase 1 complete (scene detection, manifest format). Animation pipeline in development.

## Quick Start

```bash
# Analyze video and generate manifest
python tools/animate.py --input video.mp4 --analyze --output manifest.json

# Review manifest and frames, edit element definitions, then animate
python tools/animate.py --input video.mp4 --manifest manifest.json --output animated.mp4 --runpod
```

## Workflow

### 1. Scene Detection

The tool automatically detects slide transitions:

```bash
python tools/animate.py --input notebooklm_video.mp4 --analyze --output project/manifest.json
```

This:
- Uses FFmpeg scene detection to find slide boundaries
- Extracts a representative frame from each slide
- Creates a manifest file for editing

### 2. Element Identification

Edit the manifest to define animation targets. Each element needs:

```json
{
  "element_id": "char_001",
  "type": "character",
  "description": "Illustrated person in business attire",
  "bbox": [120, 200, 400, 600],
  "animate": true,
  "motion_prompt": "gentle breathing, slight weight shift, subtle eye movement"
}
```

**Future:** Claude Vision will auto-identify elements (Phase 4).

### 3. Animation (Coming Soon)

```bash
python tools/animate.py --input video.mp4 --manifest manifest.json --output animated.mp4 --runpod
```

## What Gets Animated

### Good Candidates

| Type | Example Elements | Default Motion |
|------|------------------|----------------|
| `character` | People, illustrated figures | Breathing, weight shift, eye movement |
| `environment` | Clouds, sky, backgrounds | Slow drift, parallax |
| `water` | Rivers, oceans, rain | Rippling, flow |
| `fire` | Flames, candles | Flickering, glow variation |
| `foliage` | Trees, plants, grass | Wind movement, gentle sway |

### Should NOT Animate

- Text and titles
- Logos and branding
- UI elements
- Charts and graphs
- Complex patterns that would distort

## Motion Prompts

Keep motion **subtle and physically plausible**:

### Good Examples

```
# Characters
"gentle breathing, slight weight shift, soft hair movement"
"subtle eye movement, relaxed posture, natural blink"

# Environment
"slow atmospheric drift, gentle parallax"
"soft rippling, organic flow"

# Foliage
"subtle wind movement, gentle sway, leaf flutter"
```

### Avoid

- Specific actions ("walk left", "turn around")
- Complex sequences ("first smile, then wave")
- Unrealistic motion ("float upward", "teleport")
- Fighting the pose ("look right" when facing left)

## Manifest Format

```json
{
  "version": "1.0",
  "input_video": "/path/to/video.mp4",
  "analysis_date": "2025-12-30T10:00:00",
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 8.5,
      "duration": 8.5,
      "frame_path": "/path/to/frames/scene_001.png",
      "elements": [
        {
          "element_id": "char_001",
          "type": "character",
          "description": "Main speaker illustration",
          "bbox": [120, 200, 400, 600],
          "animate": true,
          "motion_prompt": "gentle breathing, subtle eye movement"
        }
      ]
    }
  ]
}
```

## CLI Reference

```bash
# Analyze mode - detect scenes, extract frames, create manifest
python tools/animate.py --input video.mp4 --analyze --output manifest.json

# Options
--threshold 0.3      # Scene detection sensitivity (0-1, lower=more scenes)
--min-duration 1.0   # Minimum scene length in seconds
--output-dir ./dir   # Where to save extracted frames

# Full pipeline (not yet implemented)
python tools/animate.py --input video.mp4 --manifest manifest.json --output out.mp4 --runpod

# Preview single scene
python tools/animate.py --input video.mp4 --manifest manifest.json --scene 3 --preview

# Process specific scenes only
python tools/animate.py --input video.mp4 --manifest manifest.json --scenes 2,3,5 --runpod
```

## Implementation Status

| Phase | Status | Features |
|-------|--------|----------|
| 1. Foundation | ✅ Complete | Scene detection, frame extraction, manifest format |
| 2. RunPod SVD | 🚧 Planned | Image-to-video animation via RunPod |
| 3. SAM2 Masks | 🚧 Planned | Automatic element segmentation |
| 4. Claude Vision | 🚧 Planned | Auto-identify animation candidates |
| 5. Compositing | 🚧 Planned | Layer animated elements over video |
| 6. Polish | 🚧 Planned | Error handling, documentation |

## See Also

- `.ai_dev/character-animation.md` - Implementation tracking (local development only)
- `docker/runpod-animate/` - RunPod Docker image (coming soon)
- `reference.md` - Technical reference (coming soon)
