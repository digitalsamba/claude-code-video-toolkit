# Product Demo Video Template

A **Remotion-based template** for creating marketing/product demo videos with a dark tech aesthetic.

## Quick Reference

### Starting a New Product Demo

1. Copy this template folder to a new location
2. Edit `src/config/demo-config.ts` with your product details
3. Add demo videos to `public/demos/`
4. Run `npm run studio` to preview
5. Run `npm run render` to export

### Key Files to Edit

| File | Purpose |
|------|---------|
| `src/config/demo-config.ts` | Product info, scenes, timing |
| `src/config/theme.ts` | Colors, fonts, styling |

### Scene Types

| Type | Purpose |
|------|---------|
| `title` | Opening with logos and headline |
| `problem` | Pain points with code example |
| `solution` | The "better way" introduction |
| `demo` | Video playback (browser/terminal/raw) |
| `stats` | Animated stat cards |
| `cta` | Call to action with links |

### Adding a Scene

```typescript
// In demo-config.ts, add to scenes array:
{
  type: 'demo',
  durationSeconds: 30,
  content: {
    type: 'browser',
    videoFile: 'demos/my-demo.mp4',
    label: 'Feature Demo',
    caption: 'Watch how easy it is',
  },
}
```

### Commands

```bash
npm run studio          # Preview in browser
npm run render          # Export MP4
npm run render:preview  # Quick half-res preview
```

### Video Duration

Total duration is automatically calculated from scene `durationSeconds` values.

### Audio Files

Place in `public/audio/`:
- `voiceover.mp3` - Main narration
- `background-music.mp3` - Background track

### Narrator PiP

Enable picture-in-picture narrator:

```typescript
narrator: {
  enabled: true,
  videoFile: 'narrator.mp4',
  position: 'bottom-right',
  size: 'md',
  startFrame: 120,
}
```

### Theme Customization

The default theme uses a dark tech aesthetic. Customize in `theme.ts`:

```typescript
const customTheme = {
  ...defaultTheme,
  colors: {
    ...defaultTheme.colors,
    primary: '#your-brand-color',
    accent: '#your-accent-color',
  },
};
```

## Directory Structure

```
product-demo/
├── src/
│   ├── config/
│   │   ├── demo-config.ts    # Your video content
│   │   ├── theme.ts          # Visual styling
│   │   └── types.ts          # TypeScript types
│   ├── components/
│   │   ├── core/             # Shared components
│   │   └── slides/           # Scene components
│   ├── ProductDemo.tsx       # Main composition
│   └── Root.tsx              # Remotion entry
├── public/
│   ├── demos/                # Demo videos
│   ├── images/               # Logos, icons
│   └── audio/                # Voiceover, music
└── package.json
```
