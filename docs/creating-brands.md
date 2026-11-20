# Creating Brand Profiles

Brand profiles let you maintain consistent visual identity across videos. Each brand defines colors, fonts, and voice settings.

> **Quick Start:** Run `/brand` in Claude Code to create a brand interactively. It can extract colors from a website URL, generate a palette from a single color, or let you specify colors manually.

## Brand Structure

```
brands/
├── default/
│   ├── brand.json     # Colors, fonts, spacing
│   ├── voice.json     # Voice settings
│   └── assets/        # Logo, backgrounds, watermarks
│       ├── logo.svg
│       └── background.mp4
└── my-company/
    ├── brand.json
    ├── voice.json
    ├── assets/
    │   ├── mark.svg           # optional: the flat mark (see "Mark and ident")
    │   ├── mark-on-dark.svg
    │   └── mark-white.svg
    └── ident/                 # optional: the open and close a template plays
        ├── ident.json
        └── *.mp4
```

## Creating a New Brand

1. **Copy the default brand**
   ```bash
   cp -r brands/default brands/my-company
   ```

2. **Edit brand.json**
   ```json
   {
     "name": "My Company",
     "description": "Corporate brand for My Company videos",
     "colors": {
       "primary": "#2563eb",
       "primaryLight": "#3b82f6",
       "textDark": "#1e293b",
       ...
     },
     "fonts": {
       "primary": "Inter, sans-serif",
       "mono": "JetBrains Mono, monospace"
     }
   }
   ```

3. **Configure voice settings** (optional)
   Edit `voice.json` to use a different ElevenLabs voice:
   ```json
   {
     "voiceId": "your-voice-id",
     "settings": {
       "stability": 0.75,
       "similarityBoost": 0.9
     }
   }
   ```

4. **Add brand assets**
   Place logo, background videos, or watermarks in `assets/`.

5. **Optional: a mark and an ident**
   See "Mark and ident" below. A brand with neither still works everywhere.

## Mark and ident

Both are optional. A template that reads them falls back to its own 2D opening and close when they are missing,
so a brand needs neither to render.

### The mark: `assets/mark.svg`

The brand's logo as a flat SVG of filled shapes, each with an `id`. Two more files give its colours on other
grounds; leave them out and the template uses its own ink there.

| File | For |
|------|-----|
| `mark.svg` | light grounds |
| `mark-on-dark.svg` | dark grounds |
| `mark-white.svg` | one colour, on a coloured ground |

Keep the three the same shapes in the same order: only the fills change. Check a mark before using it:

```bash
uv run tools/mark_sheet.py --svg brands/my-company/assets/mark.svg --on-dark brands/my-company/assets/mark-on-dark.svg \
    --mono brands/my-company/assets/mark-white.svg --brand my-company --output mark-sheet.png
```

It draws the mark at 512 to 16 px, on the brand's grounds, in a circle and as a watermark, and warns about low
contrast, a part the circle cuts and a part lost at 16 px.

### The ident: `ident/`

An ident is the brand's mark in motion with one sound: an **open** of about two seconds that ends on the mark at
rest, and a **close** that leaves it. The folder holds the clips and `ident.json`, which says what is there:

```json
{
  "default": "clap",
  "idents": {
    "clap": {
      "open":  { "seconds": 2.0, "frames": 60, "fps": 30,
                 "files": { "16x9": "clap-open-16x9.mp4", "9x16": "clap-open-9x16.mp4" },
                 "sound": "clap-open.wav", "hit_frame": 20, "title_frame": 51 },
      "close": { "seconds": 3.5, "frames": 105, "fps": 30,
                 "files": { "16x9": "clap-close-16x9.mp4", "9x16": "clap-close-9x16.mp4" },
                 "sound": "clap-close.wav" }
    }
  }
}
```

| Field | Meaning |
|-------|---------|
| `default` | The ident a template plays when it is not told which. A brand can keep several (a short one and a long one) |
| `files` | One clip per shape, with the sound in the file. A `-alpha` key is a transparent clip to lay over a scene |
| `sound` | The same sound alone, for a build that mixes its own audio |
| `frames`, `fps` | The clip's length. Templates here are 30 fps |
| `hit_frame` | The frame of the hit, where the sound lands |
| `title_frame` | The first frame from which the picture is at rest: a title may come in from here |
| `alpha_ground` | `dark` or `light`: the ground a transparent clip was lit for |

The clips are small, so they are kept in git with the brand. How they are made is up to you: any tool that writes
an MP4 will do. `examples/blender-ident` builds both from the mark's SVG in Blender (optional), and its `build.py`
writes this folder and measures `hit_frame` and `title_frame` from the rendered clips. `brands/video-toolkit/ident/`
is a filled slot to look at.

### Which templates read it

`templates/howto` does. `uv run templates/howto/scripts/ident.py` copies the brand's clips into the template and
writes `src/ident.json`; an episode then opens and closes with the ident, a Short ends on it, and the mark heads
the chapter track. `--none` goes back to the template's own 2D mark.

## Using a Brand in a Project

When creating a new project with `/video`, you'll be prompted to select a brand:

```
Which brand?

  1. default - #3B82F6 (blue)
  2. digital-samba - #3771e0 (blue)
  3. my-company - #your-color
  4. Create new brand → /brand
```

The selected brand's colors, fonts, and assets are automatically applied to your project via `src/config/brand.ts`.

### Manual Brand Loading

If you prefer manual setup, reference the brand in your project's config:

```typescript
// brand.ts (auto-generated by /video command)
import { loadBrand } from '../../../lib/brand';

export const brand = loadBrand('my-company');
```

## Brand Properties

### Colors

| Property | Description |
|----------|-------------|
| `primary` | Main accent color |
| `primaryLight` | Lighter variant for hover/highlight |
| `textDark` | Primary text color |
| `textMedium` | Secondary text color |
| `textLight` | Tertiary/muted text |
| `bgLight` | Light background |
| `bgDark` | Dark background |
| `bgOverlay` | Semi-transparent overlay |
| `divider` | Border/divider color |
| `shadow` | Shadow color (with alpha) |

### Fonts

| Property | Description |
|----------|-------------|
| `primary` | Main font for headings and body |
| `mono` | Monospace font for code |

### Typography

Define sizes and weights for each text style:
- `h1`, `h2`, `h3` - Heading styles
- `body` - Body text
- `label` - Labels and captions

## Voice Settings

### Several voices

A brand can carry more than one narrator, so different videos can sound different while sharing the
brand. Put each under `voices` and name the one to use when a video doesn't ask for one:

```json
{
  "default": "marilyn",
  "voices": {
    "marilyn": { "voiceId": "CyuaHFjYAn74u3mOqDX6" },
    "bren": { "voiceId": "RlSVB64yXMZJjq67jbB1" }
  }
}
```

`voiceover.py --brand video-toolkit` uses the default; `--voice bren` picks another for that video. Keys at
the top level (a shared `model`, `settings`, `qwen3` or `sixtydb` block) apply to every voice, and a voice's
own keys override them. A voice can also carry `maxWpm` and `minTempo`, the brand's pacing clamp. A
voice.json without `voices` works as before. ElevenLabs voice-library narrators already picked by ear are
in `assets/voices/library-shortlist.json`.

### ElevenLabs

| Property | Description |
|----------|-------------|
| `voiceId` | ElevenLabs voice ID |
| `stability` | Voice stability (0-1, higher = more consistent) |
| `similarityBoost` | Voice similarity (0-1, higher = more similar to original) |
| `style` | Style exaggeration (0-1) |
| `model` | ElevenLabs model to use (`eleven_v4` takes only stability and similarity) |

### Qwen3-TTS

Add an optional `qwen3` section to `voice.json` for self-hosted TTS:

```json
{
  "voiceId": "...",
  "settings": { ... },
  "qwen3": {
    "speaker": "Ryan",
    "language": "Auto",
    "instruct": "",
    "clone": {
      "refAudio": "assets/voice-reference.m4a",
      "refText": "Exact transcript of the reference audio."
    }
  }
}
```

| Property | Description |
|----------|-------------|
| `qwen3.speaker` | Built-in speaker name (Ryan, Aiden, Vivian, etc.) |
| `qwen3.language` | Language hint (Auto, English, Chinese, etc.) |
| `qwen3.instruct` | Emotion/style instruction (e.g., "Speak warmly") |
| `qwen3.clone.refAudio` | Path to reference audio, relative to brand directory |
| `qwen3.clone.refText` | Exact transcript of the reference audio |

Both `qwen3` and `qwen3.clone` are optional. Set `clone` to `null` or omit it to use built-in speakers only.

### Setting Up Voice Cloning

Run `/voice-clone` for a guided workflow that:
1. Records or imports reference audio
2. Tests the clone quality
3. Saves the profile to your brand's `voice.json`

Once saved, use `--brand` to load it automatically:

```bash
uv run tools/voiceover.py --provider qwen3 --brand my-company --scene-dir public/audio/scenes --json
```

## YouTube channel art

A brand that has its own channel also needs a banner, an avatar and a video watermark at YouTube's sizes,
checked where people will see them (the banner's phone crop, the avatar at 24 px, the watermark dimmed over
footage). See `docs/youtube-channel-art.md`. For a channel that runs a series, see `docs/series.md`.
