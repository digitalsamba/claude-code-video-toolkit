# Getting Started

This guide will help you create your first video using the claude-code-video-toolkit.

## Prerequisites

- [Claude Code](https://claude.ai/code) CLI installed
- [Node.js](https://nodejs.org/) 18+
- [Python](https://python.org/) 3.9+ (for audio tools)
- [FFmpeg](https://ffmpeg.org/) installed
- [ElevenLabs API key](https://elevenlabs.io/) (for voiceovers)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/digitalsamba/claude-code-video-toolkit.git
   cd claude-code-video-toolkit
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env and add your ELEVENLABS_API_KEY
   ```

3. **Install Python dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r tools/requirements.txt
   ```

4. **Start Claude Code**
   ```bash
   claude
   ```

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
| `/video` | Video projects - list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio |
| `/design` | Focused design refinement session for a scene |
| `/brand` | Brand profiles - list, edit, or create new |
| `/template` | List available templates and their features |
| `/record-demo` | Record browser interactions with Playwright |
| `/generate-voiceover` | Generate AI voiceover from script |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements - issues, PRs, examples |

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
