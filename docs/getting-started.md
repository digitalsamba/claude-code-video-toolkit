# Getting Started

This guide will help you create your first video using the claude-code-video-toolkit.

## Prerequisites

### Minimum (renders videos immediately)

- [Node.js](https://nodejs.org/) 18+ — that's it

### Optional: AI Voiceover

| Provider | Cost | Setup |
|----------|------|-------|
| Qwen3-TTS | Free (self-hosted) | RunPod account + `python tools/qwen3_tts.py --setup` |
| ElevenLabs | Pay-per-use | API key in `.env` |

### Optional: Full Toolkit

- [Python](https://python.org/) 3.9+ — for audio tools, image editing, upscaling
- [FFmpeg](https://ffmpeg.org/) — for media conversion and compression
- [RunPod account](https://runpod.io/) — for cloud GPU processing (TTS, image editing, watermark removal, talking heads)
- [ElevenLabs API key](https://elevenlabs.io/) — for premium AI voices

## Your First Video in 2 Minutes

```bash
cd examples/hello-world
npm install
npm run studio    # Preview in browser
npm run render    # Export MP4
```

No API keys needed. Edit `src/config/sprint-config.ts` to customize content.

## Full Setup (for AI Tools)

> This is optional — you can render videos with just Node.js installed.

1. **Clone the repository**
   ```bash
   git clone https://github.com/digitalsamba/claude-code-video-toolkit.git
   cd claude-code-video-toolkit
   ```

2. **Install Python dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r tools/requirements.txt
   ```

3. **Start Claude Code and run the setup wizard**
   ```bash
   claude
   ```
   Then type `/setup` — this walks you through configuring cloud GPU, file transfer, and voice in about 5 minutes. Most features are free:
   - **Cloudflare R2**: Free (10GB storage, zero egress)
   - **Modal**: $30/month free compute on Starter plan
   - **Qwen3-TTS**: Free AI voiceovers (runs on your Modal compute)

   Or configure manually: `cp .env.example .env` and edit with your API keys.

## Optional: Codex Setup

If you use Codex instead of Claude Code, install the toolkit's Codex-compatible wrappers and regenerate `AGENTS.md` from `CLAUDE.md`:

```bash
python3 scripts/migrate_to_codex.py --force
```

This installs toolkit skills into `~/.codex/skills` and appends or updates a generated Codex block in the repository root `AGENTS.md`.

Resources created or updated by the migration script:

1. Toolkit skills under `~/.codex/skills/`
2. Command-wrapper skills under `~/.codex/skills/`
3. A generated Codex block inside repository root `AGENTS.md`

Important:

1. The script manages only a generated block inside the repository root `AGENTS.md`.
2. Manual `AGENTS.md` content outside that block is preserved.
3. The generated block is derived from `CLAUDE.md`.
4. Re-run `python3 scripts/migrate_to_codex.py --force` after updating `CLAUDE.md`.

To remove the installed toolkit skills later:

```bash
python3 scripts/migrate_to_codex.py --reset
```

`--reset` removes the generated Codex block from `AGENTS.md`, but does not remove the rest of the file.

## Your First Video

The easiest way to create a video is using the `/video` command:

```
/video
```

This unified command will:
1. Scan for existing projects (or start fresh if none found)
2. Let you choose a template (Sprint Review or Product Demo)
3. Let you choose a brand (or create one with `/brand`)
4. Gather your content (paste notes, provide URLs, or describe what you want)
5. Plan scenes interactively with your input
6. Create a project in `projects/` with all scaffolding ready

## Manual Project Creation

If you prefer manual setup:

1. **Copy a template**
   ```bash
   cp -r templates/sprint-review projects/my-video
   cd projects/my-video
   npm install
   ```

2. **Edit the config**
   Edit `src/config/sprint-config.ts` with your content.

3. **Add demo videos**
   Place `.mp4` files in `public/demos/`

4. **Preview**
   ```bash
   npm run studio
   ```

5. **Render**
   ```bash
   npm run render
   ```

## Available Commands

| Command | Description |
|---------|-------------|
| `/setup` | First-time setup - cloud GPU, file transfer, voice, prerequisites |
| `/video` | Video projects - list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio |
| `/design` | Focused design refinement session for a scene |
| `/brand` | Brand profiles - list, edit, or create new |
| `/template` | List available templates or create new ones |
| `/record-demo` | Record browser interactions with Playwright |
| `/generate-voiceover` | Generate AI voiceover from script (supports per-scene mode) |
| `/redub` | Redub existing video with a different voice |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements - issues, PRs, examples |
| `/versions` | Check dependency versions and toolkit updates |

## Project Structure

After creating a project, you'll have:

```
projects/my-video/
├── project.json           # Project state (phase, scenes, assets)
├── CLAUDE.md              # Auto-generated status for Claude Code
├── VOICEOVER-SCRIPT.md    # Narration script with asset markers
├── src/
│   ├── config/
│   │   ├── sprint-config.ts  # Your video content (or demo-config.ts)
│   │   ├── brand.ts          # Brand colors/fonts (auto-generated)
│   │   └── theme.ts
│   └── components/
├── public/
│   ├── demos/      # Your demo videos
│   ├── audio/      # Voiceovers, music, SFX
│   └── images/     # Logo, screenshots
└── package.json
```

## Multi-Session Workflow

Projects can span multiple Claude Code sessions. The `/video` command tracks progress:

```
/video
```

When you have existing projects, you'll see:

```
Found 2 video projects:

  1. **my-release-video** (sprint-review)
     Phase: assets - 2/5 demos recorded
     Last worked: 2 days ago

  2. **product-launch** (product-demo)
     Phase: audio - voiceover needed
     Last worked: 5 days ago

Which project? (or 'new' for a new project)
```

### Project Phases

| Phase | Description |
|-------|-------------|
| `planning` | Defining scenes, writing script |
| `assets` | Recording demos, gathering materials |
| `review` | Scene-by-scene review in Remotion Studio (`/scene-review`) |
| `audio` | Generating voiceover, music |
| `editing` | Adjusting timing, previewing |
| `rendering` | Final render in progress |
| `complete` | Done |

## Next Steps

- [Creating Templates](./creating-templates.md) - Build custom video structures
- [Creating Brands](./creating-brands.md) - Define visual identity
