# claude-code-video-toolkit

This file provides guidance to Claude Code (claude.ai/code) when working with this video production toolkit.

## Overview

**claude-code-video-toolkit** is an AI-native video production workspace. It provides Claude Code with the skills, commands, and tools to create professional videos from concept to final render.

**Key capabilities:**
- Programmatic video creation with Remotion (React-based)
- AI voiceover generation with ElevenLabs or Qwen3-TTS
- AI music generation with ACE-Step 1.5 (text-to-music, vocals, covers, stems)
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
├── projects/            # Your video projects go here (gitignored)
├── examples/            # Curated showcase projects (shared)
├── assets/              # Shared assets (voices, images)
├── playwright/          # Browser recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata & registry
```

## Registry

`_internal/toolkit-registry.json` is the canonical source for all skills, commands, tools, templates, components, transitions, and cloud endpoints — including their paths, status, options, presets, and env vars. Consult it for structured data. This file focuses on **workflow guidance, patterns, and knowledge** that the registry can't capture.

## Quick Start

**First-time setup (optional, ~5 minutes):**
```
/setup
```

Walks through cloud GPU, file transfer (R2), and voice configuration. Most features are free. Skip this if you just want to render videos with Node.js.

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

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

## Templates

Templates live in `templates/`. Each is a standalone Remotion project. See registry `templates` section for the full list.

### sprint-review
Config-driven sprint review videos with theme system, config-driven content (`sprint-config.ts`), pre-built slides (Title, Overview, Summary, Credits), demo components (single video, split-screen), and audio integration.

### product-demo
Marketing/product demo videos with dark tech aesthetic, scene-based composition (title, problem, solution, demo, stats, CTA), animated background, Narrator PiP, browser/terminal chrome, and stats cards with spring animations.

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

Reusable video components in `lib/components/`. See registry `components` section for the full list with descriptions. Import in templates via:

```tsx
import { AnimatedBackground, SlideTransition, Label } from '../../../../lib/components';
```

## Python Tools

Audio, video, and image tools in `tools/`. See registry `tools` section for the full catalog with descriptions, options, presets, and env vars. Every tool supports `--help`.

```bash
# Setup
pip install -r tools/requirements.txt
```

**Important: always invoke tools from the toolkit root directory.** When working inside a project (`projects/my-video/`), tool paths like `python3 tools/upscale.py` will fail because `tools/` is relative. Always use:
```bash
cd /path/to/claude-code-video-toolkit && python3 tools/upscale.py ...
```
This is especially critical for background commands where the working directory may not be obvious.

### Tool Categories

| Type | Tools | When to Use |
|------|-------|-------------|
| **Project tools** | voiceover, music, music_gen, sfx, sync_timing | During video creation workflow |
| **Utility tools** | redub, addmusic, notebooklm_brand, locate_watermark | Quick transformations on existing videos |
| **Cloud GPU** | image_edit, upscale, dewatermark, sadtalker, qwen3_tts, music_gen, flux2 | AI processing via RunPod or Modal (`--cloud runpod\|modal`) |

Utility tools work on any video file without requiring a project structure.

### Voiceover Generation

```bash
# Per-scene generation (recommended)
python tools/voiceover.py --scene-dir public/audio/scenes --json

# Using Qwen3-TTS (self-hosted, free alternative to ElevenLabs)
python tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json

# Single file (legacy)
python tools/voiceover.py --script SCRIPT.md --output out.mp3
```

### Timing Sync (after voiceover)

```bash
python3 tools/sync_timing.py                          # Dry run comparison
python3 tools/sync_timing.py --apply                  # Update config (1s default padding)
python3 tools/sync_timing.py --apply --padding 1.5    # Custom padding
python3 tools/sync_timing.py --voiceover-json vo.json # Use voiceover.py output
python3 tools/sync_timing.py --json                   # Machine-readable output
```

### Qwen3-TTS (Standalone)

```bash
python tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3
python tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3
python tools/qwen3_tts.py --text "Hello" --instruct "Speak enthusiastically" --output excited.mp3
python tools/qwen3_tts.py --text "Hello" --ref-audio sample.wav --ref-text "transcript" --output cloned.mp3
python tools/qwen3_tts.py --list-voices   # 9 speakers: Ryan, Aiden, Vivian, etc.
python tools/qwen3_tts.py --list-tones    # neutral, warm, professional, excited, etc.
```

Temperature controls expressiveness: `--temperature 1.2` (more expressive) or `--temperature 0.4` (more consistent).

### Cloud GPU Providers

All cloud GPU tools support two providers via `--cloud runpod|modal`. RunPod is the default. Modal was added as a reliability fallback after RunPod outages, and offers faster cold starts.

```bash
# --- RunPod setup (automated, one-time per tool) ---
echo "RUNPOD_API_KEY=your_key_here" >> .env
python tools/image_edit.py --setup
python tools/upscale.py --setup
python tools/qwen3_tts.py --setup
python tools/music_gen.py --setup

# --- Modal setup (deploy each app you need) ---
pip install modal && python3 -m modal setup
modal deploy docker/modal-upscale/app.py        # Then save URL to .env
modal deploy docker/modal-image-edit/app.py
# See docs/modal-setup.md for full guide
```

### AI Image Generation (FLUX.2 vs Ideogram 4)

The toolkit has **two** text-to-image generators. They barely overlap — the deciding factor is
**whether the image needs legible baked-in text**.

```bash
# FLUX.2 — text-FREE backgrounds + image editing (self-hosted, free, Apache-2.0/commercial-OK)
python tools/flux2.py --preset title-bg --brand digital-samba   # background for Remotion text overlay
python tools/flux2.py --prompt "Abstract tech background, no text"

# Ideogram 4 — legible IN-IMAGE text + exact color/layout (hosted API, ~$0.03-0.09/img, commercial-OK)
python3 tools/ideogram4.py --json caption.json --output title.png   # text baked into the image
python3 tools/ideogram4.py --prompt "Thumbnail: 'SHIP FASTER' bold" --output thumb.png
```

**The key distinction — baked-in text vs. background-for-overlay:**

- **FLUX.2 presets deliberately produce text-*free* backgrounds** (`title-bg`, `cta`, `thumbnail`
  all end with "no text, no words, no letters"). The intended pattern is **FLUX background →
  Remotion renders the text on top**, so the text stays editable, animatable per-letter, and
  frame-accurate. This is the right default for sprint-review **title cards / lower-thirds**.
- **Ideogram 4 bakes legible designed text *into* a flat PNG.** Pixel-perfect typography + exact
  hex colors in one shot, but static. Reach for it when **the text *is* the design**: YouTube/social
  thumbnails (static anyway), quote/stat cards, in-scene signage/logos, stylized text effects that
  Remotion overlays can't easily do. It fills a real gap — FLUX and LTX-2 both garble in-image text.

| Need | Use |
|------|-----|
| Animated/editable title text, lower-thirds, sprint-review title cards | **FLUX.2 bg + Remotion text** |
| Atmospheric/abstract backgrounds, problem/solution illustrations, presenter backdrops | **FLUX.2** presets |
| Finished graphic where typography is the design — thumbnails, quote cards, in-scene signage | **Ideogram 4** |
| Edit an existing photo (clothing, reframe, style, background) | **image_edit** (neither generator edits) |

Ideogram 4 chains with the processors like any generator: `ideogram4 → upscale.py` (crisp 4K),
`ideogram4 → ltx2.py --input` (animate the still), or drop the PNG straight into Remotion `<Img>`.
Ideogram 4 is generate-only here (no editing). See `.claude/skills/ideogram4/` for the JSON caption
format — Claude authors the caption as the "magic prompt" expander; needs `IDEOGRAM_API_KEY` in `.env`.

### AI Image Editing

```bash

# Image editing (Qwen-Image-Edit)
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal
python tools/image_edit.py --input photo.jpg --style cyberpunk
python tools/image_edit.py --input photo.jpg --background office
python tools/image_edit.py --list-presets  # Full preset list

# Upscaling (RealESRGAN)
python tools/upscale.py --input photo.jpg --output photo_4x.png --cloud runpod
python tools/upscale.py --input photo.jpg --scale 2 --model anime --face-enhance --cloud runpod
```

See `docs/qwen-edit-patterns.md` and `.claude/skills/qwen-edit/` for prompting guidance.

### AI Music Generation (ACE-Step 1.5)

Default provider is **acemusic** (official cloud API, free key from [acemusic.ai/api-key](https://acemusic.ai/api-key)). Uses XL Turbo 4B model with 5Hz LM thinking mode. Falls back to Modal/RunPod for self-hosted 2B model.

```bash
# Background music (acemusic cloud API by default)
python tools/music_gen.py --prompt "Upbeat tech corporate" --duration 60 --bpm 128 --key "G Major" --output music.mp3

# Generate 4 variations, pick the best
python tools/music_gen.py --prompt "Subtle corporate tech" --duration 60 --variations 4 --output bg.mp3

# Fast mode (disable thinking)
python tools/music_gen.py --no-thinking --prompt "Quick draft" --duration 30 --output draft.mp3

# Scene presets for video production
python tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
python tools/music_gen.py --preset tension --duration 20 --output problem.mp3
python tools/music_gen.py --preset cta --brand digital-samba --output cta.mp3

# Song with vocals and lyrics (use structure tags for sections)
python tools/music_gen.py \
  --prompt "Indie pop anthem, male vocal, bright guitar, studio polish" \
  --lyrics "[Verse]\nWalking through the morning light\nCoffee in my hand feels right\n\n[Chorus - anthemic]\nWE KEEP MOVING FORWARD\nThrough the noise and doubt\n\n[Outro - fade]\n(Moving forward...)" \
  --duration 60 --bpm 128 --key "G Major" --output song.mp3

# Cover / style transfer
python tools/music_gen.py --cover --reference theme.mp3 --prompt "Jazz piano version" --output cover.mp3

# Repaint a weak section (acemusic only)
python tools/music_gen.py --repaint --input track.mp3 --repaint-start 15 --repaint-end 25 --prompt "Guitar solo" --output fixed.mp3

# Continue from existing audio (acemusic only)
python tools/music_gen.py --continuation --input track.mp3 --prompt "Continue with jazz piano" --output extended.mp3

# Stem extraction
python tools/music_gen.py --extract vocals --input mixed.mp3 --output vocals.mp3

# Fall back to self-hosted
python tools/music_gen.py --cloud modal --prompt "Background music" --duration 60 --output bg.mp3

# List presets
python tools/music_gen.py --list-presets
```

8 scene presets: `corporate-bg`, `upbeat-tech`, `ambient`, `dramatic`, `tension`, `hopeful`, `cta`, `lofi`. See `.claude/skills/acestep/` for prompt engineering patterns and video production integration guide.

### Watermark Removal

```bash
# Locate watermark coordinates
python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/
python tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

# Remove watermark (RunPod)
python tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod
python tools/dewatermark.py --setup  # One-time setup
```

**Workflow:** grid overlay → note coordinates → verify with `--region` → remove with dewatermark.

**Local mode** requires NVIDIA GPU (8GB+ VRAM). Mac users should use `--runpod`.

### Talking Head Generation (SadTalker)

```bash
# Basic usage
python tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

# For NarratorPiP integration (recommended settings)
# CRITICAL: --preprocess full preserves image dimensions (otherwise outputs square crop)
python tools/sadtalker.py \
  --image presenter_16x9.png \
  --audio voiceover.mp3 \
  --preprocess full --still --expression-scale 0.8 \
  --output narrator.mp4
```

**Key flags for NarratorPiP:**
- `--preprocess full` — **Critical!** Preserves input dimensions (default `crop` outputs square)
- `--still` — Reduces head movement for professional look
- `--expression-scale 0.8` — Calmer expression (default 1.0)

**Image requirements:** Face 30-70% of frame, front-facing, 16:9 for NarratorPiP, 512px+ recommended.

See `docs/sadtalker.md` for detailed options and troubleshooting.

### Redub Sync Mode

```bash
python tools/redub.py --input video.mp4 --voice-id VOICE_ID --sync --output dubbed.mp4
```

The `--sync` flag enables word-level time remapping — essential when TTS voice pacing differs from original. Without it, audio can drift 3-4+ seconds by the end.

**How it works:** Scribe transcribes original → TTS generates new audio with timestamps → segment mapping (15 words/segment) → FFmpeg variable speed per segment.

### NotebookLM Branding

Post-processes NotebookLM videos with custom branding. Solves the problem where redubbed TTS audio extends beyond the safe visual trim point.

```bash
python tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --logo assets/logo.png \
    --url "mysite.com" \
    --output video_final.mp4
```

Trims NotebookLM visuals, keeps full audio, bridges with freeze frame, adds branded outro.

## Video Production Workflow

1. **Create/resume project** - Run `/video`, choose template and brand (or resume existing)
2. **Review script** - Edit `VOICEOVER-SCRIPT.md` to plan content
3. **Gather assets** - Record demos with `/record-demo` or add external videos
4. **Scene review** - Run `/scene-review` to verify visuals in Remotion Studio
5. **Design refinement** - Use `/design` or the "Refine" option in scene-review to improve slide visuals
6. **Generate audio** - Use `/generate-voiceover` for AI narration
7. **Sync timing** - Run `python3 tools/sync_timing.py --apply` to update config durations
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
- **Voiceover drives timing** — Narration length determines scene duration
- **Reading pace** — ~150 words/minute (2.5 words/second) for standard narration
- **Demo pacing** — Real-time demos often need 1.5-2x speedup (`playbackRate`)
- **Transitions** — Add 1-2s padding between scenes
- **FPS** — All videos use 30fps (frames = seconds × 30)

### Speaking Rate Tiers

| Pace | WPM | Use When |
|------|-----|----------|
| Slow | 120-130 | Technical explanations, complex concepts |
| Standard | 140-160 | General narration, demos, overviews |
| Fast | 160-180 | Energetic intros, recaps, CTAs |

### Narration Density by Scene Type

| Scene Type | Duration | Narration Density | Notes |
|------------|----------|-------------------|-------|
| Title | 3-5s | 0-10% | Logo + headline, let visuals breathe |
| Overview | 10-20s | 70-90% | 3-5 bullet points, narration-heavy |
| Demo | 10-30s | 30-50% | Let the demo speak, narrate key moments only |
| Stats | 8-12s | 70-90% | Read out highlights, skip obvious numbers |
| Credits | 5-10s | 0-20% | Quick fade, maybe a closing line |
| Problem/Solution | 10-15s | 80-90% | Narration drives the story |
| CTA | 5-10s | 60-80% | Clear call to action, leave a beat at end |

### Word Count Budgeting

Before writing scripts, budget words per scene:

```
Target duration × 2.5 = word budget (at standard pace)
Pause seconds × 2.5 = words to subtract from budget

Example: 15s scene with a 1s pause
  15 × 2.5 = 37 words budget
  1 × 2.5 = 3 words for pause
  Available: ~34 words of narration
```

Use `[pause 1.0s]` markers in scripts. Each second of pause costs ~2-3 words from the budget.

### Timing Calculations
```
Script words ÷ 150 = voiceover minutes (estimate)
Raw demo length ÷ playbackRate = demo duration
Sum of scenes + transitions = total video
```

### When to Check Timing
- **During scene planning** — Budget word counts per scene before writing
- **After writing script** — Count words per scene, compare to budget
- **After generating audio** — Run `sync_timing.py` to compare actual vs estimated
- **Before rendering** — Ensure `durationInFrames` matches actual audio for each scene

### TTS Duration Drift (The Real Timing Problem)

TTS engines do NOT consistently produce 150 WPM output. In practice:
- **ElevenLabs** tends to compress pauses and speed through short sentences. A 50s script may produce 40-45s of audio.
- **Qwen3-TTS** varies by speaker and tone preset. Ryan at "professional" tone speaks ~10% faster than "warm."
- **Short scenes drift more** — a 5-second scene might be off by 30%, while a 30-second scene is off by 10%.

**The feedback loop after TTS generation:**

1. Generate per-scene audio files
2. Run `python3 tools/sync_timing.py` to compare actual vs config durations
3. Run `python3 tools/sync_timing.py --apply` to update config automatically
4. For demo scenes: recalculate `playbackRate = rawDemoDuration / actualNarrationDuration`
5. Re-preview in Remotion Studio before rendering

**Common drift patterns and fixes:**

| Problem | Symptom | Fix |
|---------|---------|-----|
| Audio shorter than scene | Dead air / awkward silence at end | Reduce `durationInFrames` to match audio |
| Audio longer than scene | Narration cut off | Increase `durationInFrames` or trim script |
| Demo too fast for narration | Viewer can't follow | Decrease `playbackRate` or cut narration |
| Demo too slow for narration | Waiting for demo to catch up | Increase `playbackRate` (1.5-2x typical) |
| Pauses lost in TTS | Script felt spacious, audio feels rushed | Add explicit `<break time="1s"/>` in SSML or extend scene padding |

### Fixing Mismatches
- **Voiceover too long**: Speed up demos, trim pauses, cut content
- **Voiceover too short**: Slow demos, add scenes, expand narration
- **Demo too long**: Increase `playbackRate` (1.5x-2x typical)
- **Demo too short**: Decrease `playbackRate`, or loop/extend

### Audio-Anchored Timelines (the prevention approach)

`sync_timing.py` is reactive — it fixes drift after the fact. You can prevent drift entirely by **generating the audio first, then anchoring visuals to known timestamps** instead of estimating durations upfront.

**The pattern:**

1. Write the script and split into per-scene segments
2. Generate per-scene VO files: `voiceover.py --scene-dir public/audio/scenes --json`
3. Read the actual durations from the JSON output
4. Anchor every visual element to absolute timestamps in the timeline

This is especially clean for Python/moviepy builds where each clip carries its own `start=` parameter:

```python
# Audio-anchored scene timeline (25s total):
#   Scene 1 tired      0.3 → 3.74  (audio 3.44s)
#   Scene 2 worries    4.0 → 8.88  (audio 4.88s)
#   Scene 3 introduce  9.1 → 11.90 (audio 2.80s)

text_clip("TIRED OF",     start=0.5,  duration=1.2)
text_clip("THIRD-PARTY",  start=1.0,  duration=1.8)
vo_clip("01_tired.mp3",   start=0.3)
vo_clip("02_worries.mp3", start=4.0)
```

The comment block at the top is the source of truth. Every `start=` references it. Drift is impossible because durations aren't being estimated — they're being read from the rendered audio.

**Trade-off vs. `<Series>`-style auto-chaining:**

| Approach | Best for | Downside |
|----------|----------|----------|
| Audio-anchored absolute starts | Tight ad-style edits, sub-30s spots, anything with exact timing | Manual bookkeeping when re-timing a scene |
| `<Series>` / auto-chained durations | Long-form sprint reviews where adjacent scenes flex | Drift compounds across the timeline; needs `sync_timing.py` to recover |

For Remotion projects you can mix the two: use `<Sequence from={...}>` with absolute frames for tight sections and let `<Series>` handle the rest. For pure-Python builds (`build.py` + moviepy), audio-anchored is the natural default.

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

**Never use a raw HTML `<video>` tag** — Remotion requires its own component for frame-accurate rendering.

The toolkit's templates use `<OffthreadVideo>` and `<Audio>` from the `remotion` package. Stick with these for consistency with existing templates. Upstream Remotion now also documents `<Video>` and `<Audio>` from the newer `@remotion/media` package (see `.claude/skills/remotion-official/`) — those add trim props, pitch shifting via `toneFrequency`, and richer loop behavior. Migrate per-template only if you need those features; both APIs render correctly.

```tsx
<OffthreadVideo src={staticFile('demo.mp4')} />
<Audio src={staticFile('voiceover.mp3')} volume={1} />
<Audio src={staticFile('music.mp3')} volume={0.15} />
```

## Scene Transitions

The toolkit includes a transitions library at `lib/transitions/`. See registry `transitions` section for the full list with options and best-use descriptions.

### Using TransitionSeries

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

### Transition Options Examples

```tsx
glitch({ intensity: 0.8, slices: 8, rgbShift: true })      // Tech/cyberpunk
lightLeak({ temperature: 'warm', direction: 'right' })       // Warm celebration
zoomBlur({ direction: 'in', blurAmount: 20 })                // High energy
rgbSplit({ direction: 'diagonal', displacement: 30 })        // Chromatic aberration
```

### Timing Functions

```tsx
linearTiming({ durationInFrames: 30 })                                      // Constant speed
springTiming({ config: { damping: 200 }, durationInFrames: 45 })            // Physics bounce
```

### Transition Duration Guidelines

| Type | Frames | Notes |
|------|--------|-------|
| Quick cut | 10-15 | Fast, punchy |
| Standard | 20-30 | Most common |
| Dramatic | 40-60 | Slow reveals |
| Glitch effects | 15-25 | Should feel sudden |
| Light leak | 30-45 | Needs time to sweep |

Preview all transitions: `cd showcase/transitions && npm run studio`

See `lib/transitions/README.md` for full documentation.

## Design Refinement with frontend-design Skill

The `frontend-design` skill elevates slide visuals from generic to distinctive.

### Usage
- **During scene review** (`/scene-review`): Choose "Refine" for visual improvements
- **Focused sessions** (`/design`): Deep-dive on a specific scene — `/design title`, `/design cta`

### When to Use
- Slide scenes that feel generic
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
