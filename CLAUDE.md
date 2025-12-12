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

## Commands

| Command | Description |
|---------|-------------|
| `/video` | Video projects - list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio (before voiceover) |
| `/brand` | Brand profiles - list, edit, or create new |
| `/template` | List available templates and their features |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements - issues, PRs, skills, templates |
| `/record-demo` | Guided Playwright browser recording |
| `/generate-voiceover` | Generate AI voiceover from script |

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

Audio generation tools in `tools/`. Config from `_internal/toolkit-registry.json`.

```bash
# Setup
pip install -r tools/requirements.txt

# Voiceover
python tools/voiceover.py --script SCRIPT.md --output out.mp3

# Background music
python tools/music.py --prompt "Subtle corporate" --duration 120 --output music.mp3

# Sound effects
python tools/sfx.py --preset whoosh --output sfx.mp3
python tools/sfx.py --prompt "Thunder crack" --output thunder.mp3
```

**Presets:** whoosh, click, chime, error, pop, slide

## Video Production Workflow

1. **Create/resume project** - Run `/video`, choose template and brand (or resume existing)
2. **Review script** - Edit `VOICEOVER-SCRIPT.md` to plan content
3. **Gather assets** - Record demos with `/record-demo` or add external videos
4. **Generate audio** - Use `/generate-voiceover` for AI narration
5. **Configure** - Update config file with asset paths and timing
6. **Preview** - `npm run studio` in project directory
7. **Iterate** - Adjust timing, content, styling with Claude Code
8. **Render** - `npm run render` for final MP4

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
