# Sprint Review Video Template

A Remotion-based template for creating professional sprint review videos with consistent theming and animations.

## Quick Start

```bash
# Install dependencies
npm install

# Open the preview studio
npm run studio

# Render final video
npm run render
```

## Configuration

### 1. Edit Sprint Config

Open `src/config/sprint-config.ts` and update:

```typescript
export const sprintConfig: SprintConfig = {
  info: {
    name: 'Everest',                    // Sprint name
    dateRange: '15th Jan – 29th Jan',   // Sprint dates
    product: 'Your Product',            // Product name
    platform: 'iOS App Update',         // Platform/section
    version: '2.0.0',                   // Version
    build: '150',                       // Build number
  },
  // ... rest of config
};
```

### 2. Add Demo Videos

Place your screen recordings in `public/demos/`:
- `demo-feature.mp4`
- `demo-bugfix.mp4`
- etc.

Then add them to the config:

```typescript
demos: [
  {
    type: 'single',
    videoFile: 'demo-feature.mp4',
    label: 'New Feature Demo',
    jiraRef: 'PROJ-123',
    durationSeconds: 15,
    playbackRate: 1.5,  // Speed up long demos
  },
  {
    type: 'split',
    leftVideo: 'mobile.mp4',
    rightVideo: 'desktop.mp4',
    leftLabel: 'Mobile',
    rightLabel: 'Desktop',
    label: 'Cross-Platform',
    durationSeconds: 12,
  },
],
```

### 3. Add Audio (Optional)

**Per-scene audio (recommended):**

Create script files in `public/audio/scenes/`:
```
public/audio/scenes/
├── 01-title.txt       → generates 01-title.mp3
├── 02-overview.txt    → generates 02-overview.mp3
├── 03-demo.txt        → generates 03-demo.mp3
└── 04-summary.txt     → generates 04-summary.mp3
```

Generate with:
```bash
python tools/voiceover.py --scene-dir public/audio/scenes --json
```

Then reference in config:
```typescript
info: {
  name: 'Sprint Name',
  // ...
  audioFile: 'scenes/01-title.mp3',
},
overview: {
  title: "What's New",
  items: [...],
  audioFile: 'scenes/02-overview.mp3',
},
demos: [
  { videoFile: 'demo.mp4', audioFile: 'scenes/03-demo.mp3', ... }
],
```

Benefits:
- Regenerate individual scenes without re-doing everything
- Scene durations match audio naturally (no offset calculations)
- Each `<Audio>` starts at frame 0 within its `Series.Sequence`

**Single voiceover (legacy):**

Place audio files in `public/audio/`:
- `voiceover.mp3` - Main narration
- `background-music.mp3` - Background track
- `sfx-chime.mp3` - Success sound effect

### 4. Customize Theme (Optional)

Edit `src/config/theme.ts` to change colors:

```typescript
export const defaultTheme: Theme = {
  colors: {
    primary: '#2563eb',      // Change accent color
    primaryLight: '#3b82f6',
    // ...
  },
};
```

## Project Structure

```
sprint-review-template/
├── src/
│   ├── config/
│   │   ├── theme.ts          # Colors, fonts, spacing
│   │   ├── sprint-config.ts  # Sprint-specific content
│   │   └── types.ts          # TypeScript definitions
│   ├── components/
│   │   ├── core/             # AnimatedBackground, SlideTransition, Label
│   │   ├── slides/           # TitleSlide, OverviewSlide, SummarySlide, EndCredits
│   │   └── demos/            # DemoSection, SplitScreen
│   ├── SprintReview.tsx      # Main composition
│   ├── Root.tsx              # Composition registration
│   └── index.ts              # Entry point
├── public/
│   ├── audio/                # Voiceover, music, SFX
│   ├── demos/                # Screen recordings
│   └── images/               # Logo, screenshots
└── package.json
```

## Commands

| Command | Description |
|---------|-------------|
| `npm run studio` | Open Remotion preview studio |
| `npm run render` | Render full quality MP4 |
| `npm run render:preview` | Render half-resolution preview |

## Available Themes

```typescript
import { defaultTheme, darkTheme, blueTheme, greenTheme } from './config/theme';

// Use in SprintReview.tsx:
<ThemeProvider theme={blueTheme}>
```

## Narrator PiP (Picture-in-Picture)

Add a narrator video overlay to your sprint review:

```typescript
// In sprint-config.ts
narrator: {
  enabled: true,
  videoFile: 'narrator.mp4',     // Place in public/
  position: 'bottom-right',       // bottom-right, bottom-left, top-right, top-left
  size: 'md',                     // sm, md, lg
  // startFrame: 120,             // Defaults to voiceoverStartFrame
},
```

**Recording tips for narrator video:**
- Record yourself speaking the voiceover script
- Frame from chest up, centered
- Use good lighting and a clean background
- Match the duration to your voiceover audio
- The component crops the bottom slightly to hide hands

## Maze Decoration (Optional)

Add an animated grid decoration to a corner of your video:

```typescript
// In sprint-config.ts
mazeDecoration: {
  enabled: true,
  corner: 'top-right',          // 'top-right' or 'top-left'
  opacity: 0.18,                // 0-1, default 0.18
  // Colors default to theme.colors.primary and theme.colors.backgroundDark
  // primaryColor: '#ea580c',   // Override primary block color
  // secondaryColor: '#475569', // Override secondary block color
},
```

The decoration creates an isometric grid of animated blocks that pulse in a wave pattern. Works well for adding subtle visual interest without distracting from content.

## Tips

- **Playback rate**: Use `playbackRate: 1.5` or `2.0` to speed up long demos
- **Timing**: Adjust `durationSeconds` per demo to match voiceover
- **Video duration**: Update `videoConfig.durationSeconds` in sprint-config.ts
- **Audio sync**: Use `voiceoverStartFrame` to delay voiceover start
- **Narrator sync**: Narrator PiP appears with voiceover and fades out before credits
