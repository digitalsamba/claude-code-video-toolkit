# banner-showcase

The animated banner shown at the top of the main `README.md`. Source for the file at `assets/banner/toolkit-banner.{gif,mp4,poster.png}`.

## The live banner

Since the rename to Agent Video Toolkit the banner is `banner.py`, a Manim scene on the brand's navy field: script,
voice, picture and video fold into the mark, then the name arrives beside it. It needs Manim (`docs/manim.md`).

```bash
uv run tools/manim_render.py render showcase/banner/banner.py Banner --brand video-toolkit \
    --size 1920x888 --out showcase/banner/out/banner.mp4
cd showcase/banner/out
ffmpeg -y -i banner.mp4 -vf "fps=20,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48[p];[b][p]paletteuse=dither=none" ../../../assets/banner/toolkit-banner.gif
ffmpeg -y -i banner.mp4 -vf scale=960:-2 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart -an ../../../assets/banner/toolkit-banner.mp4
ffmpeg -y -ss 7 -i banner.mp4 -frames:v 1 -vf scale=1080:-1 ../../../assets/banner/toolkit-banner-poster.png
```

Flat colour keeps the GIF near 0.6 MB (the glowing one was 2.9 MB).

## The earlier Remotion banner

The Remotion project here made the banner before the rename: the `amber` theme, mono phosphor on a dark grid, with
three alternates (`outrun`, `dusk`, `midnight`) registered in `src/Root.tsx`. It is kept as an example.

## Re-rendering the Remotion banner

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
- `src/Wordmark.tsx`: "AGENT VIDEO TOOLKIT" drop-in with glow + optional chromatic aberration
- `src/Pipeline.tsx` — `NARRATE ▸ SCORE ▸ GENERATE ▸ COMPOSE ▸ RENDER` sequential ignite
- `src/CRTOverlay.tsx` — scanlines, power-on sweep, vignette
