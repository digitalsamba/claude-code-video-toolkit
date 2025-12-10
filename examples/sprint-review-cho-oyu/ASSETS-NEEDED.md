# Assets Needed

To run this example, you'll need to create these media assets.

## Demo Videos

| File | Duration | Description | How to Create |
|------|----------|-------------|---------------|
| `remotion/public/demos/sleep-lock.mp4` | ~30s | Device lock/unlock while in call | iPhone screen recording |
| `remotion/public/demos/tile-timing-phone.mp4` | ~15s | Screenshare start on phone | iPhone screen recording |
| `remotion/public/demos/tile-timing-browser.mp4` | ~15s | Remote view in browser | Browser recording |
| `remotion/public/demos/button-state.mp4` | ~30s | Screenshare button after reconnection | iPhone screen recording |
| `remotion/public/demos/recording-indicator.mp4` | ~20s | Recording indicator cleanup | iPhone screen recording |
| `remotion/public/demos/deep-links.mp4` | ~20s | Deep link opening room | iPhone screen recording |
| `remotion/public/demos/background-screenshare.mp4` | ~30s | Background screenshare flow | iPhone screen recording |

### Recording Tips

- **iPhone**: Use built-in Screen Recording (Control Center)
- **Browser**: QuickTime Player > New Screen Recording
- **Split-screen demos**: Record phone and browser separately, combine in Remotion

### Split-Screen Demos

For tile-timing demo, record both views:
```
+------------------+------------------+
|                  |                  |
|   iPhone View    |  Remote Browser  |
|   (phone.mp4)    |   (browser.mp4)  |
|                  |                  |
+------------------+------------------+
```

The `SplitScreen` component will combine them.

## Audio

| File | Duration | Description | How to Create |
|------|----------|-------------|---------------|
| `remotion/public/audio/voiceover.mp3` | ~3:00 | Narration from VOICEOVER-SCRIPT.md | `/generate-voiceover` |
| `remotion/public/audio/background-music.mp3` | ~4:00 | Subtle background track | `python tools/music.py` |
| `remotion/public/audio/sfx-whoosh.mp3` | ~1s | Transition sound | `python tools/sfx.py --preset whoosh` |
| `remotion/public/audio/sfx-chime.mp3` | ~1s | Success sound | `python tools/sfx.py --preset chime` |
| `remotion/public/audio/sfx-click.mp3` | ~0.5s | Click sound | `python tools/sfx.py --preset click` |

### Voiceover Generation

```bash
cd /path/to/toolkit
python tools/voiceover.py \
  --script examples/sprint-review-cho-oyu/VOICEOVER-SCRIPT.md \
  --output examples/sprint-review-cho-oyu/remotion/public/audio/voiceover.mp3
```

### Background Music Generation

```bash
python tools/music.py \
  --prompt "subtle corporate, professional, clean" \
  --duration 240 \
  --output examples/sprint-review-cho-oyu/remotion/public/audio/background-music.mp3
```

### Sound Effects

```bash
python tools/sfx.py --preset whoosh --output examples/sprint-review-cho-oyu/remotion/public/audio/sfx-whoosh.mp3
python tools/sfx.py --preset chime --output examples/sprint-review-cho-oyu/remotion/public/audio/sfx-chime.mp3
python tools/sfx.py --preset click --output examples/sprint-review-cho-oyu/remotion/public/audio/sfx-click.mp3
```

## Small Assets (Already Included)

| File | Status |
|------|--------|
| `remotion/public/images/ds-logo.png` | ✅ Included |

## Directory Structure After Assets

```
remotion/public/
├── audio/
│   ├── voiceover.mp3
│   ├── background-music.mp3
│   ├── sfx-whoosh.mp3
│   ├── sfx-chime.mp3
│   └── sfx-click.mp3
├── demos/
│   ├── sleep-lock.mp4
│   ├── tile-timing-phone.mp4
│   ├── tile-timing-browser.mp4
│   ├── button-state.mp4
│   ├── recording-indicator.mp4
│   ├── deep-links.mp4
│   └── background-screenshare.mp4
└── images/
    └── ds-logo.png   # ✅ Included
```

## Quick Start After Creating Assets

```bash
cd remotion
npm install
npm run studio    # Preview
npm run render    # Final MP4
```
