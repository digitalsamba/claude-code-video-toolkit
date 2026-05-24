# ds-crt-stinger

A 6-second brand stinger that opens with a pixel-art astronaut waving in an amber-phosphor CRT, then crossfades into the `digitalsamba` wordmark — grunged and glowing so it reads as part of the same CRT world rather than pasted over it.

![ds-crt-stinger preview](ds_stinger_astro.gif)

## What this example demonstrates

1. **LTX-2.3 with a style LoRA** — uses the `crt-terminal` LoRA (`tools/ltx2.py --lora crt-terminal`) to generate the CRT footage end-to-end from a text prompt. No reference image needed.
2. **Post-processing a raster logo** to match generated footage — `make_grungy_logo.py` tints the monochrome-white wordmark, adds a phosphor halo, applies scanlines, chromatic aberration, and grain so it visually belongs next to the LTX-2 output.
3. **Audio-anchored moviepy composite** — `build.py` crossfades the CRT clip into the grunged logo with a coordinated audio fade, ending on a 1s hold on the brand mark.

## Files

| File | Purpose |
|---|---|
| `build.py` | moviepy composite (CRT clip → crossfade → grungy logo hold) |
| `make_grungy_logo.py` | Tints the white SVG logo and applies CRT grunge effects |
| `logo-ds-white.svg` | Monochrome-white variant of the DS wordmark, derived from the brand SVG |
| `ds_boot_astro.mp4` | Pre-rendered LTX-2 output (astronaut in amber CRT). Kept so you can run `build.py` without paying for a Modal generation. |
| `ds_stinger_astro.gif` | The final stinger as a 384px GIF for README embedding |

## Run it

```bash
# From the toolkit root:
python3 -m pip install -r tools/requirements.txt   # moviepy, Pillow, numpy
brew install librsvg                               # rsvg-convert (one-time)

cd examples/ds-crt-stinger
python3 build.py                                   # → ds_stinger_astro.mp4
```

The first run auto-rasterises the white SVG and bakes the grungy logo. Subsequent runs reuse the cached intermediates.

## Regenerate the CRT footage

Skip this unless you want a different prompt, palette, or composition. The committed `ds_boot_astro.mp4` works out of the box.

The LTX-2 step requires the Modal LTX-2 endpoint to be deployed (see `docs/modal-setup.md`) and costs ~$0.30 per take.

```bash
# From the toolkit root:
python3 tools/ltx2.py \
  --lora crt-terminal --seed 42 \
  --prompt 'a CRT aesthetic with heavy scanlines and bloom, warm amber phosphor glow, chromatic aberration, a chunky pixel-art cartoon astronaut in a rounded white spacesuit with a reflective dome helmet, slowly floating in zero gravity and waving cheerfully at the camera, a few small twinkling stars in the dark background, static centered composition, low choppy frame rate, friendly retro arcade mood' \
  --output examples/ds-crt-stinger/ds_boot_astro.mp4
```

## Palette swaps

The grungy-logo builder ships three palettes. Regenerate a different phosphor color:

```bash
python3 make_grungy_logo.py amber   # warm orange — default
python3 make_grungy_logo.py coral   # deeper red, closer to DS brand coral
python3 make_grungy_logo.py green   # classic hacker phosphor
```

Then pass the palette to `build.py`:

```bash
python3 build.py astro coral        # astro clip + coral-grunged wordmark
```

## Tuning knobs

In `make_grungy_logo.py`:

| Constant | What it does | Try |
|---|---|---|
| `PALETTES[*]["tint"]` | Core glyph color | Brighter RGB for more pop |
| `PALETTES[*]["halo"]` | Bleed color around the glyphs | More saturated for neon; desaturated for subtle |
| `HALO_RADIUS` | How far the phosphor halo spreads | 15 tight / 45 dreamy |
| `SCANLINE_DARKEN` | Every-other-row darkening | 0.55 harsh / 0.85 gentle |
| `RGB_SHIFT_PX` | Chromatic aberration offset | 5–6 for more dramatic |
| `NOISE_STRENGTH` | Grain amount | 0–40 |

In `build.py`:

| Constant | What it does |
|---|---|
| `CROSSFADE` | Length of the clip→logo handoff (default 1.5s) |
| `LOGO_HOLD` | How long the logo holds after the crossfade (default 1.0s) |

## See also

- [`.claude/skills/ltx2/SKILL.md`](../../.claude/skills/ltx2/SKILL.md) — LTX-2 prompting and the `crt-terminal` style LoRA
- [`.claude/skills/moviepy/SKILL.md`](../../.claude/skills/moviepy/SKILL.md) — audio-anchored timelines and single-file `build.py` conventions
- [`examples/quick-spot/`](../quick-spot/) — the minimal moviepy example this pattern builds on
