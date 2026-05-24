# Hello World

A minimal example that renders a video in under 2 minutes with zero configuration.

## Quick Start

```bash
npm install
npm run studio   # Preview in browser
npm run render   # Export to out/video.mp4
```

That's it. No API keys, no Python, no external services.

## What You Get

A 25-second video with 4 animated slides:
1. **Title** -- Project name with animated background
2. **Overview** -- Key highlights with staggered bullet animations
3. **Stats** -- Animated stat cards with spring physics
4. **Credits** -- Team credits with fade-in

## Next Steps

- **Edit content**: Open `src/config/sprint-config.ts` and change the text
- **Change colors**: Edit `src/config/brand.ts` to match your brand
- **Add demos**: Drop `.mp4` files in `public/demos/` and reference in config
- **Add voiceover**: Use `/generate-voiceover` in Claude Code
- **Full project**: Run `/video` in Claude Code to create a production project
