# banner-showcase

The animated banner shown at the top of the main `README.md`. Source for the file at `assets/banner/toolkit-banner.{gif,mp4,poster.png}`.

## Theme

The live banner uses the `amber` theme — mono phosphor CRT aesthetic on a dark grid. Three alternates (`outrun`, `dusk`, `midnight`) are kept registered in `src/Root.tsx` so we can revisit or re-render if the branding changes.

## Re-rendering

```bash
cd showcase/banner
npm install
npm run studio     # Preview / scrub in Remotion Studio
npm run render     # MP4 → out/toolkit-banner.mp4
npm run render:gif # GIF → out/toolkit-banner.gif
npm run render:poster # Static PNG poster frame
```

After rendering, copy the outputs over the top of `assets/banner/`:

```bash
cp out/toolkit-banner.{mp4,gif} out/toolkit-banner-poster.png ../../assets/banner/
```

### Shrinking the GIF

Remotion's native GIF is honest-quality but large. The published `assets/banner/toolkit-banner.gif` is a palette-optimized pass via ffmpeg:

```bash
ffmpeg -y -i out/toolkit-banner.mp4 \
  -vf "fps=20,scale=960:444:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  out/toolkit-banner.gif
```

64 colors is plenty for the amber theme. For the full-saturation variants (`outrun`, `dusk`) bump `max_colors=128` or accept a larger file.

## Structure

- `src/themes.ts` — all four color palettes in one file; swap `themes.amber` in `Root.tsx` to change the primary
- `src/SynthwaveBackground.tsx` — base color, optional banded sun, perspective grid
- `src/Wordmark.tsx` — "CLAUDE CODE VIDEO TOOLKIT" drop-in with glow + optional chromatic aberration
- `src/Pipeline.tsx` — `NARRATE ▸ SCORE ▸ GENERATE ▸ COMPOSE ▸ RENDER` sequential ignite
- `src/CRTOverlay.tsx` — scanlines, power-on sweep, vignette
