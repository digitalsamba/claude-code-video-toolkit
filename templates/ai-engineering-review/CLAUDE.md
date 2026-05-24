# Sprint Review Video Template

This is a **Remotion-based video template** for creating professional sprint review videos.

## Quick Reference

### Starting a New Sprint Video

1. Copy this template folder to a new location
2. Edit `src/config/sprint-config.ts` with sprint details
3. Add demo videos to `public/demos/`
4. Run `npm run studio` to preview
5. Run `npm run render` to export

### Key Files to Edit

| File | Purpose |
|------|---------|
| `src/config/sprint-config.ts` | Sprint name, version, demos, stats |
| `src/config/theme.ts` | Colors (change `primary` for brand color) |

### Adding a Demo

```typescript
// In sprint-config.ts, add to demos array:
{
  type: 'single',
  videoFile: 'my-demo.mp4',  // Goes in public/demos/
  label: 'Feature Name',
  jiraRef: 'PROJ-123',
  durationSeconds: 15,
  playbackRate: 1.5,  // Optional: speed up
}
```

### Split Screen Demo

```typescript
{
  type: 'split',
  leftVideo: 'phone.mp4',
  rightVideo: 'browser.mp4',
  leftLabel: 'Mobile',
  rightLabel: 'Desktop',
  label: 'Cross-platform Feature',
  durationSeconds: 12,
}
```

### Commands

```bash
npm run studio          # Preview in browser
npm run render          # Export MP4
npm run render:preview  # Quick half-res preview
```

### Video Duration

Total duration is calculated from:
- Title: 5s
- Overview: 15s
- Sum of all demo `durationSeconds`
- Summary: 15s
- Credits: 30s

Update `videoConfig.durationSeconds` in sprint-config.ts to match.

### Audio Files

Place in `public/audio/`:
- `voiceover.mp3` - Main narration
- `background-music.mp3` - Background track (plays at 15% volume)
- `sfx-chime.mp3` - Success sound for summary slide

### Theme Colors

Default theme uses orange accent (#ea580c). To change:

```typescript
// In theme.ts, modify defaultTheme or use a preset:
import { blueTheme, greenTheme, darkTheme } from './config/theme';
```
