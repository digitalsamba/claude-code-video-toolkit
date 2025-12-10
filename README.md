# claude-code-video-toolkit

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

### Commands

| Command | Description |
|---------|-------------|
| `/video` | Video projects — list, resume, or create new |
| `/brand` | Brand profiles — list, edit, or create new |
| `/template` | List available templates and their features |
| `/skill` | List installed skills or create new ones |
| `/contribute` | Share improvements — issues, PRs, examples |
| `/record-demo` | Record browser interactions with Playwright |
| `/generate-voiceover` | Generate AI voiceover from a script |

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

### Templates

Pre-built video structures in `templates/`:

- **sprint-review** — Sprint review videos with demos, stats, and voiceover
- **product-demo** — Marketing videos with dark tech aesthetic, stats, CTA

See `examples/` for finished projects you can learn from:
- [digital-samba-skill-demo](https://demos.digitalsamba.com/video/digital-samba-skill-demo.mp4) — Product demo showcasing Claude Code skill
- [sprint-review-cho-oyu](https://demos.digitalsamba.com/video/sprint-review.mp4) — iOS sprint review with demos

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

### Python Tools

Audio generation CLI in `tools/`:

```bash
# Generate voiceover
python tools/voiceover.py --script script.md --output voiceover.mp3

# Generate background music
python tools/music.py --prompt "Upbeat corporate" --duration 120 --output music.mp3

# Generate sound effects
python tools/sfx.py --preset whoosh --output sfx.mp3
```

## Project Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Slash commands (/video, /brand, etc.)
├── lib/                 # Shared utilities and project system
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

## How It Works

1. **You describe** what video you want
2. **Claude Code uses skills** to understand the domain (Remotion, audio, etc.)
3. **Commands guide** complex workflows step-by-step
4. **Templates provide** ready-made video structures
5. **Tools automate** repetitive tasks (voiceover, music, SFX)
6. **You iterate** with live preview until it's perfect
7. **Render** to MP4

## Video Workflow

```
/video → Review script → Gather assets → Generate audio → Preview → Iterate → Render
```

1. **Create project** — Run `/video`, choose template and brand
2. **Review script** — Edit `VOICEOVER-SCRIPT.md` to plan content and assets
3. **Gather assets** — Record demos with `/record-demo` or add external videos
4. **Generate audio** — AI voiceover with `/generate-voiceover`
5. **Configure** — Update config file with asset paths and timing
6. **Preview** — `npm run studio` for live preview
7. **Iterate** — Work with Claude Code to adjust timing, styling, content
8. **Render** — `npm run render` for final MP4

## Requirements

- **Claude Code** — The AI coding assistant
- **Node.js 18+** — For Remotion
- **Python 3.9+** — For audio tools
- **FFmpeg** — For media processing
- **ElevenLabs API key** — For AI voiceovers (optional but recommended)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Built for use with [Claude Code](https://claude.ai/code) by Anthropic.
