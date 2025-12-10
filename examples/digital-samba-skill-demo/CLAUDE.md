# CLAUDE.md - Digital Samba Skill Demo Video

## Project Overview

Marketing/blog video demonstrating the Digital Samba Claude Code skill. Shows the journey from skill installation to a working video conferencing app.

## Related Projects

| Project | Path | Purpose |
|---------|------|---------|
| Digital Samba Skill | `~/work/digital-samba-skill` | The skill being demoed |
| Interview Room Demo | `~/work/digital-samba-interview-room-demo` | Demo app built with the skill |
| Video Workspace | `~/work/video` | Parent Remotion workspace |

## Branding Guidelines

### Digital Samba
- **Primary**: `#0066FF` (blue)
- **Accent**: `#00D4AA` (teal)
- **Logo**: `remotion/public/images/ds-logo.png`
- **App Icon**: `remotion/public/images/embedded-app-icon.png`

### Visual Style
- Dark backgrounds (`#0a0a0a`, `#111`)
- Terminal aesthetics for code sections
- Clean, modern tech feel
- Subtle animations (no flashy transitions)

## Asset Pipeline

### Existing Assets
```
assets/
├── claude-code-install.gif      # Skill installation (1008x627)
├── claude-desktop-install.gif   # Alternative install (600x296)
```

### Assets to Generate
1. **GIF → MP4 conversion**: `ffmpeg -i input.gif -movflags faststart -pix_fmt yuv420p output.mp4`
2. **Playwright recordings**: Browser automation captures app flows
3. **Asciinema conversion**: `build-app.cast` from demo repo
4. **Voiceover**: ElevenLabs TTS from VOICEOVER-SCRIPT.md
5. **Background music**: ElevenLabs generation (subtle tech ambient)

## Playwright Recording

The demo app at `~/work/digital-samba-interview-room-demo` should have Playwright tests that:
1. Record browser interactions as video
2. Capture the full user flow (create room → join → connect)
3. Output MP4 files to `remotion/public/demos/`

### Key Flows to Record
- Home page load
- Create interview room form submission
- Room code display and copy
- Candidate join flow
- Video room connection

## Remotion Composition

Target: 2:30 (4500 frames @ 30fps)

### Components Needed
- `TitleSlide` - Logos + title
- `ProblemSlide` - Setup the problem
- `SolutionIntro` - Introduce the skill
- `GifDemo` - Play converted install GIF
- `TerminalDemo` - Asciinema playback
- `BrowserDemo` - Playwright recordings
- `CodeHighlight` - Syntax-highlighted snippets
- `SummarySlide` - Stats with spring animation
- `EndCredits` - CTA + links

## Python Environment

This project uses a Python virtual environment for audio generation scripts.

```bash
# First time setup
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv elevenlabs

# Subsequent runs - always activate venv first
source .venv/bin/activate
```

**Required**: Create a `.env` file with your ElevenLabs API key:
```
ELEVENLABS_API_KEY=your_key_here
```

## Commands

```bash
# Activate Python venv (required for audio generation)
source .venv/bin/activate

# Generate audio (voiceover and/or music)
python generate_audio.py              # Both
python generate_audio.py voiceover    # Just voiceover
python generate_audio.py music        # Just background music

# Convert GIFs
ffmpeg -i assets/claude-code-install.gif -movflags faststart -pix_fmt yuv420p remotion/public/demos/skill-install.mp4

# Preview video
cd remotion && npm run studio

# Final render
cd remotion && npm run render
```

## Key Documents

- **VIDEO-SPEC.md** - Full video specification and timeline
- **VOICEOVER-SCRIPT.md** - Narration with timing marks
- **PROJECT-STATUS.md** - Phase tracking (create when starting)
