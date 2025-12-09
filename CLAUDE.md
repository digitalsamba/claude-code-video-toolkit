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
│   └── default/
├── projects/            # Your video projects go here
├── assets/              # Shared assets
│   ├── voices/          # Voice samples for cloning
│   └── images/          # Shared images
├── playwright/          # Browser recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata
```

## Quick Start

**Create a new video:**
```
/new-sprint-video
```

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
| `/new-sprint-video` | Interactive wizard for sprint review videos |
| `/new-brand` | Create a new brand profile (colors, fonts, voice) |
| `/record-demo` | Guided Playwright browser recording |
| `/generate-voiceover` | Generate AI voiceover from script |

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

## Python Tools

Audio generation tools in `tools/`. Config from `_internal/skills-registry.json`.

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

1. **Plan** - Define video content and structure
2. **Create project** - Use `/new-sprint-video` or copy template
3. **Record assets** - Use `/record-demo` or screen recording
4. **Generate audio** - Voiceover, music, SFX with Python tools
5. **Preview** - `npm run studio` in project directory
6. **Iterate** - Adjust timing, content, styling
7. **Render** - `npm run render` for final MP4

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
- Each project has its own `PROJECT-STATUS.md`

Keep these separate. Don't mix toolkit improvements with video production.

## Documentation

- `docs/getting-started.md` - First video walkthrough
- `docs/creating-templates.md` - Build new templates
- `docs/creating-brands.md` - Create brand profiles
