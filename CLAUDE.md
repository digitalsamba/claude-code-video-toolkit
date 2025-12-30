# claude-code-video-toolkit

This file provides guidance to Claude Code (claude.ai/code) when working with this video production toolkit.

## Overview

**claude-code-video-toolkit** is an AI-native video production workspace. It provides Claude Code with the skills, commands, and tools to create professional videos from concept to final render.

**Key capabilities:**
- Programmatic video creation with Remotion (React-based)
- AI voiceover generation with ElevenLabs
- Browser demo recording with Playwright
- Asset processing with FFmpeg

## Directory Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Guided workflows
├── tools/               # Python CLI automation
├── templates/           # Video templates
│   ├── sprint-review/   # Sprint review video template
│   └── product-demo/    # Marketing/product demo template
├── brands/              # Brand profiles (colors, fonts, voice)
│   ├── default/
│   └── digital-samba/
├── projects/            # Your video projects go here (gitignored)
├── examples/            # Curated showcase projects (shared)
├── assets/              # Shared assets
│   ├── voices/          # Voice samples for cloning
│   └── images/          # Shared images
├── playwright/          # Browser recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata
```

## Quick Start

**Work on a video project:**
```
/video
```

This command will:
1. Scan for existing projects (resume or create new)
2. Choose template (sprint-review, product-demo)
3. Choose brand (or create one with `/brand`)
4. Plan scenes interactively
5. Create project with VOICEOVER-SCRIPT.md

**Multi-session support:** Projects span multiple sessions. Run `/video` to resume where you left off. Each project tracks its phase, scenes, assets, and session history in `project.json`.

**Or manually:**
```bash
cp -r templates/sprint-review projects/my-video
cd projects/my-video
npm install
npm run studio   # Preview
npm run render   # Export
```

## Skills Reference

Claude Code has deep knowledge in these domains via `.claude/skills/`:

| Skill | Status | Purpose |
|-------|--------|---------|
| remotion | stable | Video compositions, animations, rendering |
| elevenlabs | stable | TTS, voice cloning, music, SFX |
| ffmpeg | beta | Asset conversion, compression |
| playwright-recording | beta | Browser demo capture |
| frontend-design | stable | Visual design refinement for slides |

## Commands

| Command | Description |
|---------|-------------|
| `/video` | Video projects - list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio (before voiceover) |
| `/design` | Focused design refinement session for a scene |
| `/brand` | Brand profiles - list, edit, or create new |
| `/template` | List available templates and their features |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements - issues, PRs, skills, templates |
| `/record-demo` | Guided Playwright browser recording |
| `/generate-voiceover` | Generate AI voiceover from script |
| `/redub` | Redub existing video with different voice |
| `/versions` | Check dependency versions and toolkit updates |

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

## Templates

Templates live in `templates/`. Each is a standalone Remotion project:

### sprint-review
Config-driven sprint review videos with:
- Theme system (colors, fonts, spacing)
- Config-driven content (`sprint-config.ts`)
- Pre-built slides: Title, Overview, Summary, Credits
- Demo components: Single video, Split-screen
- Audio integration (voiceover, music, SFX)

### product-demo
Marketing/product demo videos with dark tech aesthetic:
- Scene-based composition (title, problem, solution, demo, stats, CTA)
- Config-driven content (`demo-config.ts`)
- Animated background with floating shapes
- Narrator PiP (picture-in-picture presenter)
- Browser/terminal chrome for demo videos
- Stats cards with spring animations

## Brand Profiles

Brands live in `brands/`. Each defines visual identity:

```
brands/my-brand/
├── brand.json    # Colors, fonts, typography
├── voice.json    # ElevenLabs voice settings
└── assets/       # Logo, backgrounds
```

See `docs/creating-brands.md` for details.

## Shared Components

Reusable video components in `lib/components/`. Import in templates via:

```tsx
import { AnimatedBackground, SlideTransition, Label } from '../../../../lib/components';
```

| Component | Purpose |
|-----------|---------|
| `AnimatedBackground` | Floating shapes background (variants: subtle, tech, warm, dark) |
| `SlideTransition` | Scene transitions (fade, zoom, slide-up, blur-fade) |
| `Label` | Floating label badge with optional JIRA reference |
| `Vignette` | Cinematic edge darkening overlay |
| `LogoWatermark` | Corner logo branding |
| `SplitScreen` | Side-by-side video comparison |
| `NarratorPiP` | Picture-in-picture presenter overlay |
| `Envelope` | 3D envelope with opening flap animation |
| `PointingHand` | Animated hand emoji with slide-in and pulse |

## Python Tools

Audio and video tools in `tools/`. Config from `_internal/toolkit-registry.json`.

```bash
# Setup
pip install -r tools/requirements.txt

# Voiceover generation
python tools/voiceover.py --script SCRIPT.md --output out.mp3

# Background music
python tools/music.py --prompt "Subtle corporate" --duration 120 --output music.mp3

# Sound effects
python tools/sfx.py --preset whoosh --output sfx.mp3
python tools/sfx.py --prompt "Thunder crack" --output thunder.mp3

# Redub video with different voice (utility - no project needed)
python tools/redub.py --input video.mp4 --voice-id VOICE_ID --output dubbed.mp4

# Redub with word-level sync (recommended for different pacing)
python tools/redub.py --input video.mp4 --voice-id VOICE_ID --sync --output dubbed.mp4

# Add background music to video (utility - no project needed)
python tools/addmusic.py --input video.mp4 --prompt "Subtle corporate" --output output.mp4
python tools/addmusic.py --input video.mp4 --music bg.mp3 --music-volume 0.2 --fade-in 2 --fade-out 3 --output output.mp4
```

**SFX Presets:** whoosh, click, chime, error, pop, slide

### Utility Tools vs Project Tools

| Type | Tools | When to Use |
|------|-------|-------------|
| **Project tools** | voiceover, music, sfx | During video creation workflow |
| **Utility tools** | redub, addmusic, notebooklm_brand, locate_watermark | Quick transformations on existing videos |
| **Optional tools** | dewatermark | Requires additional installation (see below) |

Utility tools work on any video file without requiring a project structure.

### Watermark Removal (Optional Component)

The `dewatermark.py` tool uses AI inpainting (ProPainter) to remove watermarks.

**Two processing modes:**
- **RunPod (cloud)** - Works from any machine, ~$0.05-0.30/video
- **Local** - Requires NVIDIA GPU with 8GB+ VRAM

```bash
# Cloud processing via RunPod (recommended for Mac users)
# Default outputs at 50% resolution for memory safety
python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod

# Full resolution (may fail on some GPUs due to memory limits)
python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod --resize-ratio 1.0

# Local processing (requires NVIDIA GPU + ProPainter installation)
python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4

# Check local installation status
python tools/dewatermark.py --status

# Install ProPainter for local processing (~2GB download)
python tools/dewatermark.py --install
```

**RunPod setup (for cloud processing):**
```bash
# 1. Add API key to .env
echo "RUNPOD_API_KEY=your_key_here" >> .env

# 2. Run automated setup
python tools/dewatermark.py --setup

# Done! The endpoint ID is automatically saved to .env
```

For manual setup or advanced options, see `docs/runpod-setup.md`.

**Hardware requirements (local mode):**
- **Required:** NVIDIA GPU (8GB+ VRAM)
- **Not supported:** Apple Silicon, CPU-only (use `--runpod` instead)
- **Disk:** ~2GB for model weights

**Installation location:** `~/.video-toolkit/propainter/`

### Locating Watermarks

Before removing a watermark, you need to identify its exact coordinates. The `locate_watermark.py` tool helps with this.

**Requires:** ImageMagick (`brew install imagemagick`)

```bash
# Explore with coordinate grid overlay
python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

# Use a preset for common watermarks
python tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

# Verify custom region across multiple frames
python tools/locate_watermark.py --input video.mp4 --region 1100,650,150,50 --verify

# List available presets
python tools/locate_watermark.py --list-presets
```

**Workflow:**
1. Extract frames with `--grid` to identify watermark position
2. Note coordinates from the grid overlay
3. Verify with `--region x,y,w,h --verify` across multiple frames
4. Use confirmed region with `dewatermark.py`

**Presets:** notebooklm, tiktok, stock-br, stock-bl, stock-center

**Options:**
- `--samples N` - Number of frames to extract (default: 5)
- `--grid` - Overlay coordinate grid
- `--mark` - Draw rectangle on frames
- `--verify` - Mark region across multiple frames for verification
- `--crop` - Also output cropped watermark regions
- `--open` - Open output directory in Finder (macOS)

### Redub Sync Mode

The `--sync` flag enables word-level time remapping for redubbing. This is essential when the TTS voice speaks at a different pace than the original.

**Why it's needed:**
- ElevenLabs TTS pacing varies (often starts fast, ends slow)
- Linear video slowdown can't compensate for variable drift
- Without sync, audio can drift 3-4+ seconds by the end

**How it works:**
1. Scribe transcribes original audio with word-level timestamps
2. TTS generates new audio with character-level timestamps (via `convert_with_timestamps`)
3. Tool builds segment mapping (default: 15 words per segment)
4. FFmpeg applies variable speed per segment using filtergraph
5. Result: each word in video aligns with its TTS counterpart

**Options:**
```bash
--sync              # Enable word-level sync
--segment-size N    # Words per segment (default: 15)
```

**When to use:**
- Redubbing to a voice with significantly different pacing
- When simple audio replacement produces noticeable drift
- For professional results where sync matters

### NotebookLM Branding

The `notebooklm_brand.py` tool handles post-processing of NotebookLM videos with custom branding. This solves a specific challenge: when redubbing NotebookLM videos, the TTS audio may be longer than the safe visual trim point.

**The problem:**
- NotebookLM videos have a ~2s visual outro (logo + URL card)
- When you redub with a slower TTS voice, the audio extends beyond the original
- Simple trimming cuts off the final narration
- Video and audio tracks need separate handling

**The solution:**
1. Trim video to remove NotebookLM visuals
2. Keep full audio (uncut)
3. Bridge the gap with a freeze frame
4. Add custom branded outro

```bash
# Basic usage
python tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --logo assets/logo.png \
    --url "mysite.com" \
    --output video_final.mp4

# Specify exact trim point (if auto-detection is wrong)
python tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --logo assets/logo.png \
    --url "mysite.com" \
    --trim-at 174.7 \
    --output video_final.mp4

# Use existing outro card image
python tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --outro-card assets/outro_card.png \
    --output video_final.mp4

# Use separate TTS audio file (full narration)
python tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --audio-file tts_audio.mp3 \
    --outro-card assets/outro_card.png \
    --output video_final.mp4
```

**Options:**
- `--trim-at`: Seconds where NotebookLM outro begins (auto-detects if not specified)
- `--audio-file`: Use separate audio file instead of video's audio track
- `--outro-duration`: Length of branded outro (default: 4s)
- `--logo-scale`: Logo width in pixels (default: 220)
- `--text-color`: URL text color as hex (default: 6B8E6B sage green)

## Video Production Workflow

1. **Create/resume project** - Run `/video`, choose template and brand (or resume existing)
2. **Review script** - Edit `VOICEOVER-SCRIPT.md` to plan content
3. **Gather assets** - Record demos with `/record-demo` or add external videos
4. **Scene review** - Run `/scene-review` to verify visuals in Remotion Studio
5. **Design refinement** - Use `/design` or the "Refine" option in scene-review to improve slide visuals
6. **Generate audio** - Use `/generate-voiceover` for AI narration
7. **Configure** - Update config file with asset paths and timing
8. **Preview** - `npm run studio` in project directory
9. **Iterate** - Adjust timing, content, styling with Claude Code
10. **Render** - `npm run render` for final MP4

## Project Lifecycle

Projects move through phases tracked in `project.json`:

```
planning → assets → review → audio → editing → rendering → complete
```

| Phase | Description |
|-------|-------------|
| `planning` | Defining scenes, writing script |
| `assets` | Recording demos, gathering materials |
| `review` | Scene-by-scene review in Remotion Studio (`/scene-review`) |
| `audio` | Generating voiceover, music |
| `editing` | Adjusting timing, previewing |
| `rendering` | Final render in progress |
| `complete` | Done |

See `lib/project/README.md` for details on the project system.

## Video Timing

Timing is critical. Keep these guidelines in mind:

### Pacing Rules
- **Voiceover drives timing** - Narration length determines scene duration
- **Reading pace** - ~150 words/minute for comfortable narration
- **Demo pacing** - Real-time demos often need 1.5-2x speedup (`playbackRate`)
- **Transitions** - Add 1-2s padding between scenes
- **FPS** - All videos use 30fps (frames = seconds × 30)

### Scene Duration Guidelines
| Scene Type | Duration | Notes |
|------------|----------|-------|
| Title | 3-5s | Logo + headline |
| Overview | 10-20s | 3-5 bullet points |
| Demo | 10-30s | Adjust playbackRate to fit |
| Stats | 8-12s | 3-4 stat cards |
| Credits | 5-10s | Quick fade |

### Timing Calculations
```
Script words ÷ 150 = voiceover minutes
Raw demo length ÷ playbackRate = demo duration
Sum of scenes + transitions = total video
```

### When to Check Timing
- After generating VOICEOVER-SCRIPT.md (estimate per scene)
- When voiceover audio is created (compare actual vs estimated)
- Before rendering (ensure everything fits)

### Fixing Mismatches
- **Voiceover too long**: Speed up demos, trim pauses, cut content
- **Voiceover too short**: Slow demos, add scenes, expand narration
- **Demo too long**: Increase `playbackRate` (1.5x-2x typical)
- **Demo too short**: Decrease `playbackRate`, or loop/extend

## Key Patterns

### Animations (Remotion)
```tsx
const frame = useCurrentFrame();
const opacity = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: 'clamp' });
```

### Sequencing
```tsx
<Series>
  <Series.Sequence durationInFrames={150}><TitleSlide /></Series.Sequence>
  <Series.Sequence durationInFrames={900}><DemoClip /></Series.Sequence>
</Series>
```

### Media
```tsx
<OffthreadVideo src={staticFile('demo.mp4')} />
<Audio src={staticFile('voiceover.mp3')} volume={1} />
<Audio src={staticFile('music.mp3')} volume={0.15} />
```

## Scene Transitions

The toolkit includes a transitions library at `lib/transitions/` with both official Remotion transitions and custom presentations.

### Using TransitionSeries

For scene-to-scene transitions (scenes overlap during transition):

```tsx
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { glitch, lightLeak, zoomBlur } from '../../../lib/transitions';

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <TitleSlide />
  </TransitionSeries.Sequence>
  <TransitionSeries.Transition
    presentation={glitch({ intensity: 0.8 })}
    timing={linearTiming({ durationInFrames: 20 })}
  />
  <TransitionSeries.Sequence durationInFrames={120}>
    <ContentSlide />
  </TransitionSeries.Sequence>
</TransitionSeries>
```

### Available Transitions

| Transition | Description | Best For |
|------------|-------------|----------|
| `glitch()` | Digital distortion, RGB separation, scan lines | Tech demos, edgy reveals |
| `rgbSplit()` | Chromatic aberration effect | Modern tech, energetic |
| `zoomBlur()` | Radial motion blur with scale | CTAs, high-energy moments |
| `lightLeak()` | Cinematic lens flare, overexposure | Celebrations, film aesthetic |
| `clockWipe()` | Radial wipe like clock hands | Playful reveals |
| `pixelate()` | Digital mosaic dissolution | Retro/gaming themes |
| `slide()` | Push scene from direction | Standard transitions |
| `fade()` | Simple crossfade | Professional, subtle |
| `wipe()` | Edge reveal | Clean transitions |
| `flip()` | 3D card flip | Playful, dynamic |

### Transition Options Examples

```tsx
// Tech/cyberpunk feel
glitch({ intensity: 0.8, slices: 8, rgbShift: true })

// Warm celebration
lightLeak({ temperature: 'warm', direction: 'right' })

// High energy zoom
zoomBlur({ direction: 'in', blurAmount: 20 })

// Chromatic aberration
rgbSplit({ direction: 'diagonal', displacement: 30 })
```

### Timing Functions

```tsx
// Linear: constant speed
linearTiming({ durationInFrames: 30 })

// Spring: physics-based with bounce
springTiming({ config: { damping: 200 }, durationInFrames: 45 })
```

### Transition Duration Guidelines

| Type | Frames | Notes |
|------|--------|-------|
| Quick cut | 10-15 | Fast, punchy |
| Standard | 20-30 | Most common |
| Dramatic | 40-60 | Slow reveals |
| Glitch effects | 15-25 | Should feel sudden |
| Light leak | 30-45 | Needs time to sweep |

### Preview Transitions Gallery

To see all transitions in action with Scene A → Scene B demos:

```bash
cd showcase/transitions
npm install
npm run studio
```

See `lib/transitions/README.md` for full documentation.

## Design Refinement with frontend-design Skill

The `frontend-design` skill elevates slide visuals from generic to distinctive. It's integrated at multiple levels:

### During Scene Review (`/scene-review`)
When reviewing a slide scene, choose "Refine" to invoke frontend-design for visual improvements. The skill helps with:
- Color palette and mood
- Typography and text animations
- Background effects and atmosphere
- Motion and timing
- Visual coherence between scenes

### Focused Sessions (`/design`)
For deep-dive design work on a specific scene:
```
/design title    # Refine the title slide
/design cta      # Refine the CTA slide
```

### When to Use
- Slide scenes (title, problem, solution, stats, cta) that feel generic
- When something "doesn't feel right" visually
- When building visual contrast between scenes (e.g., calm title → harsh problem)
- When animations feel too basic or too busy

### Visual Narrative Arc
Consider how visual intensity builds across scenes:
- **Title**: Set the mood, plant visual seeds
- **Problem**: Create tension (harsh contrast)
- **Solution**: Relief and hope return
- **Demo**: Neutral, content-focused
- **Stats**: Build credibility
- **CTA**: Climax - maximum visual energy

The frontend-design skill understands this narrative and helps create appropriate contrast and flow.

## Toolkit vs Project Work

**Toolkit work** (evolves the toolkit itself):
- Skills, commands, templates, tools
- Tracked in `_internal/ROADMAP.md`

**Project work** (creates videos):
- Lives in `projects/`
- Each project has `project.json` (machine-readable state) and auto-generated `CLAUDE.md`

Keep these separate. Don't mix toolkit improvements with video production.

## Documentation

- `docs/getting-started.md` - First video walkthrough
- `docs/creating-templates.md` - Build new templates
- `docs/creating-brands.md` - Create brand profiles
- `docs/optional-components.md` - Setup for optional ML-based tools (ProPainter, etc.)
