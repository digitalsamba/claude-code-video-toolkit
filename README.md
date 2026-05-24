# claude-code-video-toolkit

<p align="center">
  <img src="assets/banner/toolkit-banner.gif" alt="claude-code-video-toolkit — NARRATE ▸ SCORE ▸ GENERATE ▸ COMPOSE ▸ RENDER" width="960" />
</p>

[![GitHub release](https://img.shields.io/github/v/release/digitalsamba/claude-code-video-toolkit)](https://github.com/digitalsamba/claude-code-video-toolkit/releases)

An AI-native video production workspace for [Claude Code](https://claude.ai/code), with experimental bridges to other agent runtimes like [Codex](#using-with-codex). Skills, commands, templates, and tools that give your AI agent everything it needs to help you create professional videos — from concept to final render.

## Quick Start

```bash
git clone https://github.com/digitalsamba/claude-code-video-toolkit.git
cd claude-code-video-toolkit
python3 -m pip install -r tools/requirements.txt   # Optional: AI voiceover, image gen, music, moviepy examples
claude                                              # Open Claude Code in the toolkit
```

Then in Claude Code:

```
/setup                    # Configure cloud GPU, storage, voice (~5 min, mostly free)
/video                    # Create your first video
```

**That's it.** `/setup` walks you through everything interactively — cloud GPU provider, file transfer, voice config. `/video` creates a project from a template and guides you through the whole workflow.

## Using with Codex

This toolkit is built around Claude Code assets in `.claude/` and `CLAUDE.md`, but it also ships an experimental migration script for Codex — contributed by [@kimhoontae-gogo](https://github.com/kimhoontae-gogo) in [#16](https://github.com/digitalsamba/claude-code-video-toolkit/pull/16).

```bash
python3 scripts/migrate_to_codex.py --force
```

This does two things:

1. Installs Codex skills into `~/.codex/skills`
2. Appends or updates a generated Codex block in the repository root `AGENTS.md` from `CLAUDE.md`

Resources created or updated by `python3 scripts/migrate_to_codex.py --force`:

1. `~/.codex/skills/acestep/`
2. `~/.codex/skills/elevenlabs/`
3. `~/.codex/skills/ffmpeg/`
4. `~/.codex/skills/frontend-design/`
5. `~/.codex/skills/ltx2/`
6. `~/.codex/skills/moviepy/`
7. `~/.codex/skills/playwright-recording/`
8. `~/.codex/skills/qwen-edit/`
9. `~/.codex/skills/remotion/`
10. `~/.codex/skills/remotion-best-practices/`
11. `~/.codex/skills/runpod/`
12. `~/.codex/skills/brand/`
13. `~/.codex/skills/contribute/`
14. `~/.codex/skills/design/`
15. `~/.codex/skills/generate-voiceover/`
16. `~/.codex/skills/record-demo/`
17. `~/.codex/skills/redub/`
18. `~/.codex/skills/scene-review/`
19. `~/.codex/skills/setup/`
20. `~/.codex/skills/skills/`
21. `~/.codex/skills/template/`
22. `~/.codex/skills/versions/`
23. `~/.codex/skills/video/`
24. `~/.codex/skills/video-toolkit/`
25. `~/.codex/skills/voice-clone/`
26. A generated Codex block inside `AGENTS.md` in the repository root

Important:

1. The migration script manages only a generated block inside the repository root `AGENTS.md`.
2. Manual `AGENTS.md` content outside that block is preserved.
3. The generated block is derived from `CLAUDE.md`, so changes to `CLAUDE.md` should be followed by `python3 scripts/migrate_to_codex.py --force`.

To remove the installed Codex skills later:

```bash
python3 scripts/migrate_to_codex.py --reset
```

`--reset` removes the toolkit skills previously installed under `~/.codex/skills` and removes the generated Codex block from `AGENTS.md`. It does not delete other user skills and it does not remove the rest of `AGENTS.md`.

**What's free:** The toolkit leans heavily on open-source AI models — voiceovers (Qwen3-TTS), image generation (FLUX.2), music (ACE-Step), and more. You deploy them to your own cloud GPU account and run them at cost. Cloudflare R2 has a generous free tier (10GB, zero egress), and Modal gives $30/month free compute on the Starter plan — more than enough for a few 5-minute videos a month.

**Requirements:** [Node.js](https://nodejs.org/) 18+ and [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Python 3.9+ recommended for AI tools. FFmpeg optional.

> **Want to skip setup and just render something?**
> ```bash
> cd examples/hello-world && npm install && npm run render
> ```
> No API keys needed — outputs an MP4 immediately.

---

## A Note from the Author *(not AI-generated)*

> I've spent months painstakingly putting this toolkit together and plan to keep iterating on it. AI makes things easier, but hard work still has huge value. Every video I create is a chance for improvement — every skill, template, tool, and workflow here has been refined through that cycle. It would be wonderful if others wanted to get involved with that: use it, refine it, and feed back into the repo via an issue or PR what you learn.
>
> My own use case is fairly specific: creating sprint review videos for the AI mobile development arm of [Digital Samba](https://www.digitalsamba.com/). But the idea behind this project is a reusable toolkit for using Claude Code to autonomously generate any kind of "explainer" style video — product demos, walkthroughs, presentations, whatever you need. Autonomous video creation is a lofty ideal for such a subjective field, but we can try :)
>
> What makes this work is that Claude Code is fantastically resourceful and flexible — give it the framing and tooling that this toolkit provides and it will adapt it to create templates and videos based on your prompting. The skills, templates, and tools here are building blocks. Claude Code is the builder. You are the director, editor, and designer.
>
> **If you're getting started**, run `/setup` then `/video` and let Claude Code guide you. Or start with `/template` to create a template for your own use case.
>
> **Cloud GPU** — I recommend [Modal](https://modal.com/) for running the toolkit's AI tools. The Starter plan gives you $30/month free compute, which is more than enough. [RunPod](https://runpod.io/) is also supported as an alternative. Run `/setup` to deploy the tools you need.
>
> My motto: **Be brave. Experiment.** And please share any videos you create or ideas you have back with the project — it helps me keep improving this toolkit for everyone.

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
| **acestep** | AI music generation — prompts, lyrics, scene presets, video integration |
| **ltx2** | AI video generation — text-to-video, image-to-video clips, prompting guide |
| **moviepy** | Python video composition — overlay text on LTX-2/SadTalker output, build.py-style projects |
| **runpod** | Cloud GPU — setup, Docker images, endpoint management, costs |

### Commands

| Command | Description |
|---------|-------------|
| `/setup` | First-time setup — cloud GPU, file transfer, voice, prerequisites |
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
| `/voice-clone` | Record, test, and save a cloned voice to a brand |
| `/versions` | Check dependency versions and toolkit updates |

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

### Templates

Pre-built video structures in `templates/`:

- **sprint-review** — Sprint review videos with demos, stats, and voiceover
- **sprint-review-v2** — Composable scene-based sprint review with modular architecture
- **product-demo** — Marketing videos with dark tech aesthetic, stats, CTA

See `examples/` for finished projects you can learn from (oldest first, showing toolkit evolution):

| Date | Demo | Description |
|------|------|-------------|
| 2025-12-05 | [sprint-review-cho-oyu](https://demos.digitalsamba.com/sprint-review-cho-oyu.mp4) | iOS sprint review with demos |
| 2025-12-10 | [digital-samba-skill-demo](https://demos.digitalsamba.com/video/digital-samba-skill-demo.mp4) | Product demo showcasing Claude Code skill |
| 2026-01-22 | [ds-remote-mcp](https://demos.digitalsamba.com/video/ds-remote-mcp.mp4) | Remote MCP server demo *(the jazz background music is a joke)* |
| 2026-01-25 | [schlumbergera](https://demos.digitalsamba.com/video/schlumbergera.mp4) | Android sprint review video |
| 2026-02-23 | [cortina](https://demos.digitalsamba.com/video/sprint-review.mp4) | Mobile platforms sprint review |
| 2026-03-15 | [the-space-between](https://demos.digitalsamba.com/video/the-space-between.mp4) | AI-generated video essay — flux2 avatar, Qwen3-TTS voice, SadTalker animation |
| 2026-04-08 | [q2-townhall-longarm-ad](https://demos.digitalsamba.com/video/q2-townhall-longarm-ad.mp4) | Super Bowl-style launch ad with dramatic Qwen3-TTS announcer and LTX-2 animated Lugh cameo |
| 2026-04-08 | [q2-townhall-stars](https://demos.digitalsamba.com/video/q2-townhall-stars.mp4) | GitHub star history time-lapse with animated chart and deadpan-to-excited commentary |

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
# Generate voiceover (ElevenLabs)
python tools/voiceover.py --script script.md --output voiceover.mp3

# Generate voiceover (Qwen3-TTS — self-hosted, cheaper alternative)
python tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
python tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3

# Generate background music (ElevenLabs)
python tools/music.py --prompt "Upbeat corporate" --duration 120 --output music.mp3

# Generate background music (ACE-Step — free cloud API, XL Turbo 4B model)
python tools/music_gen.py --preset corporate-bg --duration 120 --output music.mp3
python tools/music_gen.py --prompt "Dramatic cinematic" --duration 30 --bpm 90 --key "D Minor" --output reveal.mp3
python tools/music_gen.py --prompt "Upbeat indie rock" --duration 60 --variations 4 --output intro.mp3

# Generate sound effects
python tools/sfx.py --preset whoosh --output sfx.mp3

# Redub video with different voice
python tools/redub.py --input video.mp4 --voice-id VOICE_ID --output dubbed.mp4

# Add background music to existing video
python tools/addmusic.py --input video.mp4 --prompt "Subtle ambient" --output output.mp4

# Rebrand NotebookLM videos (trim outro, add your logo/URL)
python tools/notebooklm_brand.py --input video.mp4 --logo logo.png --url "mysite.com" --output branded.mp4

# AI image editing (style transfer, backgrounds, custom prompts)
python tools/image_edit.py --input photo.jpg --style cyberpunk --cloud modal
python tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal

# AI image upscaling (2x/4x)
python tools/upscale.py --input photo.jpg --output photo_4x.png --cloud modal

# Remove watermarks (requires cloud GPU)
python tools/dewatermark.py --input video.mp4 --preset sora --output clean.mp4 --cloud modal

# Locate watermark coordinates
python tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

# Generate talking head video from image + audio (SadTalker)
python tools/sadtalker.py --image portrait.png --audio voiceover.mp3 --output talking.mp4 --cloud modal

# AI image generation (FLUX.2 Klein 4B — text-to-image + editing)
python tools/flux2.py --prompt "A sunset over mountains" --cloud modal
python tools/flux2.py --preset title-bg --brand digital-samba --cloud modal
python tools/flux2.py --list-presets

# AI video generation (LTX-2.3 22B — text-to-video + image-to-video)
python tools/ltx2.py --prompt "A sunset over the ocean, cinematic" --cloud modal
python tools/ltx2.py --prompt "Gentle camera drift" --input photo.jpg --cloud modal
```

**Tool Categories:**

| Type | Tools | Purpose |
|------|-------|---------|
| **Project** | voiceover, music, music_gen, sfx | Used during video creation workflow |
| **Utility** | redub, addmusic, notebooklm_brand, locate_watermark | Quick transformations, no project needed |
| **Cloud GPU** | image_edit, upscale, dewatermark, sadtalker, qwen3_tts, flux2, music_gen, ltx2 | AI processing via Modal or RunPod |

### Cloud GPU (Modal + RunPod)

8 AI tools run on cloud GPUs. Use `--cloud modal` (recommended) or `--cloud runpod` on any tool.

| Tool | What It Does | Est. Cost |
|------|--------------|-----------|
| `qwen3_tts` | AI text-to-speech (9 speakers, voice cloning) | ~$0.01 |
| `flux2` | AI image generation & editing | ~$0.02 |
| `image_edit` | AI image editing & style transfer | ~$0.03 |
| `upscale` | AI image upscaling (2x/4x) | ~$0.01 |
| `music_gen` | AI music generation (8 scene presets) | Free (acemusic) / ~$0.05 (self-hosted) |
| `sadtalker` | Talking head video from portrait + audio | ~$0.10 |
| `ltx2` | AI video generation (text-to-video, image-to-video) | ~$0.23 |
| `dewatermark` | Video watermark removal | ~$0.10 |

**Modal (recommended):** Each tool deploys from `docker/modal-*/app.py` — Modal builds and hosts the containers. $30/month free compute on the Starter plan, typical usage is $1-2/month. Run `/setup` to deploy all tools automatically.

**RunPod (alternative):** Uses pre-built Docker images from `ghcr.io/conalmullan/video-toolkit-*`. Pay-per-second, no minimums. Run `python3 tools/<tool>.py --setup` to create endpoints.

See [docs/modal-setup.md](docs/modal-setup.md) and [docs/runpod-setup.md](docs/runpod-setup.md) for details.

## Project Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Slash commands (/video, /brand, etc.)
├── lib/                 # Shared components, theme system, utilities
│   ├── components/      # Reusable video components (11 components)
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
- [Modal Setup](docs/modal-setup.md) — Cloud GPU with Modal (recommended)
- [RunPod Setup](docs/runpod-setup.md) — Cloud GPU with RunPod (alternative)
- [Creating Templates](docs/creating-templates.md)
- [Creating Brands](docs/creating-brands.md)
- [Project System](lib/project/README.md) — Multi-session lifecycle, schema, reconciliation
- [Optional Components](docs/optional-components.md) — Local GPU tools setup
- [Toolkit Development](_internal/ROADMAP.md) — Roadmap, backlog, changelog

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
