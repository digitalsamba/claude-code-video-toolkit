# Assets Needed

To run this example, you'll need to create these media assets.

## Demo Videos

| File | Duration | Description | How to Create |
|------|----------|-------------|---------------|
| `remotion/public/demos/skill-install.mp4` | ~15s | Claude Code skill installation | Convert `assets/claude-code-install.gif` with FFmpeg |
| `remotion/public/demos/app-walkthrough.mp4` | ~35s | Browser walkthrough of demo app | `/record-demo` or Playwright |
| `remotion/public/narrator.mp4` | ~2:30 | Optional narrator PiP video | Record yourself speaking the script |
| `remotion/public/demos/build-app.mp4` | ~30s | Terminal recording of the app being built | Record with asciinema, convert (below) |

### Terminal Build Recording

VIDEO-SPEC.md refers to a `build-app.cast` recorded against the demo app repo
(`digital-samba-interview-room-demo`). It is not shipped with this example — record
your own and convert it:

```bash
asciinema rec build-app.cast          # run the build, then exit
agg build-app.cast build-app.gif      # asciinema/agg
ffmpeg -i build-app.gif \
  -movflags faststart \
  -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  remotion/public/demos/build-app.mp4
```

### GIF Conversion Command

```bash
ffmpeg -i assets/claude-code-install.gif \
  -movflags faststart \
  -pix_fmt yuv420p \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  remotion/public/demos/skill-install.mp4
```

### App Walkthrough Recording

Record these flows in sequence:
1. Home page load (dual-card UI)
2. Click "Create Interview Room"
3. Fill form (title, participant names)
4. Submit → room code appears
5. Copy candidate link
6. Join as candidate with room code
7. Video room loads, "Connected" status

Recording specs: 1920x1080, 30fps

## Audio

| File | Duration | Description | How to Create |
|------|----------|-------------|---------------|
| `remotion/public/audio/voiceover.mp3` | ~2:30 | Narration from VOICEOVER-SCRIPT.md | `/generate-voiceover` |
| `remotion/public/audio/background-music.mp3` | ~3:00 | Subtle tech ambient | `uv run tools/music.py` |

### Voiceover Generation

```bash
cd /path/to/toolkit
uv run tools/voiceover.py \
  --script examples/digital-samba-skill-demo/VOICEOVER-SCRIPT.md \
  --output examples/digital-samba-skill-demo/remotion/public/audio/voiceover.mp3
```

### Background Music Generation

```bash
uv run tools/music.py \
  --prompt "subtle tech ambient, modern, clean" \
  --duration 180 \
  --output examples/digital-samba-skill-demo/remotion/public/audio/background-music.mp3
```

## Small Assets (Already Included)

These assets are tracked in the repo:

| File | Status |
|------|--------|
| `remotion/public/images/ds-logo.png` | ✅ Included |
| `remotion/public/images/embedded-app-icon.png` | ✅ Included |
| `assets/claude-code-install.gif` | ✅ Included (needs conversion) |
| `assets/claude-desktop-install.gif` | ✅ Included (alternative) |

## Directory Structure After Assets

```
remotion/public/
├── audio/
│   ├── voiceover.mp3      # Generated
│   └── background-music.mp3   # Generated
├── demos/
│   ├── skill-install.mp4  # Converted from GIF
│   └── app-walkthrough.mp4    # Playwright recording
├── images/
│   ├── ds-logo.png        # ✅ Included
│   └── embedded-app-icon.png  # ✅ Included
└── narrator.mp4           # Optional
```

## Quick Start After Creating Assets

```bash
cd remotion
npm install
npm run studio    # Preview
npm run render    # Final MP4
```
