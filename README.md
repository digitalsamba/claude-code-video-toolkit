# claude-code-video-toolkit

[![GitHub release](https://img.shields.io/github/v/release/digitalsamba/claude-code-video-toolkit)](https://github.com/digitalsamba/claude-code-video-toolkit/releases)

An AI-native video production workspace for [Claude Code](https://claude.ai/code). Create professional videos with AI assistance — from concept to final render.

## What is this?

This toolkit gives Claude Code the knowledge and tools to help you create videos:

- **Skills** — Domain expertise in Remotion, ElevenLabs, FFmpeg, Playwright
- **Commands** — Guided workflows like `/video`, `/record-demo`, `/contribute`
- **Templates** — Ready-to-customize video structures
- **Brands** — Visual identity profiles (colors, fonts, voice settings)
- **Tools** — Python CLI for audio generation

Clone this repo, open it in Claude Code, and start creating videos.

## Quick Start

### Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- [Node.js](https://nodejs.org/) 18+
- [Python](https://python.org/) 3.9+
- [FFmpeg](https://ffmpeg.org/)
- [ElevenLabs API key](https://elevenlabs.io/) (for AI voiceovers)
- [RunPod API key](https://runpod.io/) (optional, for AI image editing, upscaling, watermark removal, talking head generation)

### Setup

```bash
# Clone the toolkit
git clone https://github.com/digitalsamba/claude-code-video-toolkit.git
cd claude-code-video-toolkit

# Set up environment
cp .env.example .env
# Edit .env and add your ELEVENLABS_API_KEY

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt

# Start Claude Code
claude
```

### Create Your First Video

In Claude Code, run:

```
/video
```

This will:
1. Scan for existing projects (resume or create new)
2. Choose template (sprint-review, product-demo)
3. Choose brand (or create one with `/brand`)
4. Plan scenes interactively
5. Create project with VOICEOVER-SCRIPT.md

**Multi-session support:** Projects span multiple sessions. Run `/video` to resume where you left off.

Then iterate with Claude Code to record demos, refine content, and render.

## Features

### Skills

Claude Code has deep knowledge in:

| Skill | Description |
|-------|-------------|
| **remotion** | React-based video framework — compositions, animations, rendering |
| **elevenlabs** | AI audio — text-to-speech, voice cloning, music, sound effects |
| **ffmpeg** | Media processing — format conversion, compression, resizing |
| **playwright-recording** | Browser automation — record demos as video |
| **frontend-design** | Visual design refinement for distinctive, production-grade aesthetics |
| **qwen-edit** | AI image editing — prompting patterns and best practices |

### Commands

| Command | Description |
|---------|-------------|
| `/video` | Video projects — list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio |
| `/design` | Focused design refinement session for a scene |
| `/brand` | Brand profiles — list, edit, or create new |
| `/template` | List available templates or create new ones |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements — issues, PRs, examples |
| `/record-demo` | Record browser interactions with Playwright |
| `/generate-voiceover` | Generate AI voiceover from a script |
| `/redub` | Redub existing video with a different voice |
| `/versions` | Check dependency versions and toolkit updates |

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

### Templates

Pre-built video structures in `templates/`:

- **sprint-review** — Sprint review videos with demos, stats, and voiceover
- **product-demo** — Marketing videos with dark tech aesthetic, stats, CTA

See `examples/` for finished projects you can learn from (oldest first, showing toolkit evolution):

| Date | Demo | Description |
|------|------|-------------|
| 2025-12-05 | [sprint-review-cho-oyu](https://demos.digitalsamba.com/video/sprint-review.mp4) | iOS sprint review with demos |
| 2025-12-10 | [digital-samba-skill-demo](https://demos.digitalsamba.com/video/digital-samba-skill-demo.mp4) | Product demo showcasing Claude Code skill |
| 2026-01-22 | [ds-remote-mcp](https://demos.digitalsamba.com/video/ds-remote-mcp.mp4) | Remote MCP server demo *(the jazz background music is a joke)* |
| 2026-01-25 | [schlumbergera](https://demos.digitalsamba.com/video/schlumbergera.mp4) | Android sprint review video |

### Scene Transitions

The toolkit includes a transitions library for scene-to-scene effects:

| Transition | Description |
|------------|-------------|
| `glitch()` | Digital distortion with RGB shift |
| `rgbSplit()` | Chromatic aberration effect |
| `zoomBlur()` | Radial motion blur |
| `lightLeak()` | Cinematic lens flare |
| `clockWipe()` | Radial sweep reveal |
| `pixelate()` | Digital mosaic dissolution |
| `checkerboard()` | Grid-based reveal (9 patterns) |

Plus official Remotion transitions: `slide()`, `fade()`, `wipe()`, `flip()`

Preview all transitions:
```bash
cd showcase/transitions && npm install && npm run studio
```

See [lib/transitions/README.md](lib/transitions/README.md) for full documentation.

### Brand Profiles

Define visual identity in `brands/`. When you create a project with `/video`, the brand's colors, fonts, and styling are automatically applied.

```
brands/my-brand/
├── brand.json    # Colors, fonts, typography
├── voice.json    # ElevenLabs voice settings
└── assets/       # Logo, backgrounds
```

Included brands: `default`, `digital-samba`

Create your own with `/brand`.

### Project Management System

Video projects are tracked through a multi-session lifecycle:

```
planning → assets → review → audio → editing → rendering → complete
```

Each project has a `project.json` that tracks:
- **Scenes** — What to show, asset status, visual types
- **Audio** — Voiceover and music status
- **Sessions** — Work history across Claude Code sessions
- **Phase** — Current stage in the workflow

The system automatically reconciles intent (what you planned) with reality (what files exist), and generates a `CLAUDE.md` per project for instant context when resuming.

See [lib/project/README.md](lib/project/README.md) for schema details, scene status tracking, and filesystem reconciliation logic.

### Python Tools

Audio, video, and image tools in `tools/`:

```bash
# Generate voiceover
python tools/voiceover.py --script script.md --output voiceover.mp3

# Generate background music
python tools/music.py --prompt "Upbeat corporate" --duration 120 --output music.mp3

# Generate sound effects
python tools/sfx.py --preset whoosh --output sfx.mp3

# Redub video with different voice
python tools/redub.py --input video.mp4 --voice-id VOICE_ID --output dubbed.mp4

# Add background music to existing video
python tools/addmusic.py --input video.mp4 --prompt "Subtle ambient" --output output.mp4

# Rebrand NotebookLM videos (trim outro, add your logo/URL)
python tools/notebooklm_brand.py --input video.mp4 --logo logo.png --url "mysite.com" --output branded.mp4

# AI image editing (style transfer, backgrounds, custom prompts)
python tools/image_edit.py --input photo.jpg --style cyberpunk
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"

# AI image upscaling (2x/4x)
python tools/upscale.py --input photo.jpg --output photo_4x.png --runpod

# Remove watermarks (requires RunPod or NVIDIA GPU)
python tools/dewatermark.py --input video.mp4 --preset sora --output clean.mp4 --runpod

# Locate watermark coordinates
python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

# Generate talking head video from image + audio (SadTalker)
# Creates animated presenter/narrator from a static portrait + voiceover audio
# Use with NarratorPiP component for picture-in-picture presenter overlays
python tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --output talking.mp4
```

**Tool Categories:**

| Type | Tools | Purpose |
|------|-------|---------|
| **Project** | voiceover, music, sfx | Used during video creation workflow |
| **Utility** | redub, addmusic, notebooklm_brand, locate_watermark | Quick transformations, no project needed |
| **Cloud GPU** | image_edit, upscale, dewatermark, sadtalker | AI processing via RunPod (see below) |

See [docs/runpod-setup.md](docs/runpod-setup.md) for Cloud GPU tool setup.

### Pre-built Docker Images

Cloud GPU tools use pre-built Docker images deployed to RunPod serverless:

| Tool | Docker Image | GPU |
|------|--------------|-----|
| image_edit | `ghcr.io/conalmullan/video-toolkit-qwen-edit:latest` | 48GB+ (A6000, L40S) |
| upscale | `ghcr.io/conalmullan/video-toolkit-realesrgan:latest` | 24GB (RTX 3090/4090) |
| dewatermark | `ghcr.io/conalmullan/video-toolkit-propainter:latest` | 24GB (RTX 3090/4090) |
| sadtalker | `ghcr.io/conalmullan/video-toolkit-sadtalker:latest` | 24GB (RTX 4090) |

Dockerfiles and handlers are in `docker/`. Run `python tools/<tool>.py --setup` to auto-deploy.

**Cost:** RunPod is pay-per-second with no minimums. Typical usage is ~$0.01-0.15 per job. Even with heavy experimenting, expect to spend less than $10/month.

## Project Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Slash commands (/video, /brand, etc.)
├── lib/                 # Shared components, theme system, utilities
│   ├── components/      # Reusable video components (9 components)
│   ├── transitions/     # Scene transition effects (7 custom + 4 official)
│   ├── theme/           # ThemeProvider, useTheme
│   └── project/         # Multi-session project system
├── tools/               # Python CLI tools
├── templates/           # Video templates
├── brands/              # Brand profiles
├── projects/            # Your video projects (gitignored)
├── examples/            # Curated showcase projects with finished videos
├── assets/              # Shared assets
├── playwright/          # Recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata & roadmap
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [Creating Templates](docs/creating-templates.md)
- [Creating Brands](docs/creating-brands.md)
- [Project System](lib/project/README.md) — Multi-session lifecycle, schema, reconciliation
- [Optional Components](docs/optional-components.md) — GPU tools setup (dewatermark)
- [RunPod Setup](docs/runpod-setup.md) — Cloud GPU configuration
- [Toolkit Development](_internal/ROADMAP.md) — Roadmap, backlog, changelog for the toolkit itself

## Video Workflow

```
/video → Script → Assets → Scene Review → Design → Audio → Preview → Render
```

1. **Create project** — Run `/video`, choose template and brand
2. **Review script** — Edit `VOICEOVER-SCRIPT.md` to plan content and assets
3. **Gather assets** — Record demos with `/record-demo` or add external videos
4. **Scene review** — Run `/scene-review` to verify visuals in Remotion Studio
5. **Design refinement** — Use `/design` to improve slide visuals with the frontend-design skill
6. **Generate audio** — AI voiceover with `/generate-voiceover`
7. **Configure** — Update config file with asset paths and timing
8. **Preview** — `npm run studio` for live preview
9. **Iterate** — Work with Claude Code to adjust timing, styling, content
10. **Render** — `npm run render` for final MP4

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built for use with [Claude Code](https://claude.ai/code) by Anthropic.
