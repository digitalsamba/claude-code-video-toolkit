"""Build a CRT-phosphor-grunged version of the white DS logo.

Starts from the monochrome-white raster, tints it with a phosphor color, adds
bloom/glow, scanlines, chromatic aberration, noise, and places it on a dark
canvas — so the handoff frame feels like it's actually being displayed ON
the CRT rather than pasted cleanly over it.

Usage:
  python3 make_grungy_logo.py amber
  python3 make_grungy_logo.py coral
  python3 make_grungy_logo.py green
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageChops

HERE = Path(__file__).parent
WHITE_SVG = HERE / "logo-ds-white.svg"
WHITE_RASTER = HERE / "logo_white_raster.png"


def ensure_raster() -> None:
    """Rasterize logo-ds-white.svg via rsvg-convert if the PNG is missing."""
    if WHITE_RASTER.exists():
        return
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        sys.exit(
            "rsvg-convert not found. Install with:\n"
            "  macOS:  brew install librsvg\n"
            "  Debian: sudo apt-get install librsvg2-bin"
        )
    subprocess.check_call([rsvg, "-w", "1600", str(WHITE_SVG), "-o", str(WHITE_RASTER)])

# Phosphor palettes. `tint` is the core color of the glyphs; `halo` is the
# softer surrounding glow (usually a brighter / warmer version of the tint);
# `bg` is the dark base that reads as "monitor off".
PALETTES = {
    "amber": {
        "tint": (255, 185,  95),
        "halo": (255, 140,  40),
        "bg":   (14, 10, 6),
    },
    "coral": {
        "tint": (255, 145, 130),
        "halo": (240,  90,  80),
        "bg":   (16, 8, 8),
    },
    "green": {
        "tint": (170, 255, 190),
        "halo": ( 60, 220, 110),
        "bg":   (4, 10, 6),
    },
}

CANVAS = 1024
LOGO_WIDTH_PX = 820           # matches the width used in the first ds_logo_1024.png
HALO_RADIUS = 28              # how far the phosphor halo spreads
HALO_GAIN = 1.15              # brighten the halo so it doesn't just sit there
SCANLINE_DARKEN = 0.75        # multiplier for every other row
RGB_SHIFT_PX = 3              # chromatic aberration offset
NOISE_STRENGTH = 9            # 0-40ish
GRAIN_RADIUS = 0.8            # small Gaussian for a slight phosphor-bleed softness


def tint_white_to(rgba_logo: Image.Image, tint: tuple[int, int, int]) -> Image.Image:
    """Recolor a white RGBA logo to the given tint, keeping alpha intact."""
    _, _, _, a = rgba_logo.split()
    solid = Image.new("RGB", rgba_logo.size, tint)
    return Image.merge("RGBA", (*solid.split(), a))


def build_halo(tinted_logo: Image.Image, halo_color: tuple[int, int, int],
               radius: int, gain: float, canvas_size: int,
               offset: tuple[int, int]) -> Image.Image:
    """Produce an RGB halo layer: the logo silhouette in halo_color, heavily
    blurred and brightened, placed on a transparent canvas at `offset`."""
    # Halo-coloured silhouette with original alpha
    halo_solid = Image.new("RGB", tinted_logo.size, halo_color)
    _, _, _, a = tinted_logo.split()
    halo_rgba = Image.merge("RGBA", (*halo_solid.split(), a))

    full = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    full.paste(halo_rgba, offset, halo_rgba)
    full = full.filter(ImageFilter.GaussianBlur(radius=radius))

    # Boost brightness so the halo isn't washed out by blur
    arr = np.asarray(full).astype(np.float32)
    arr[..., :3] *= gain
    arr[..., :3] = np.clip(arr[..., :3], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def chromatic_aberration(img: Image.Image, shift: int) -> Image.Image:
    r, g, b = img.split()
    r = ImageChops.offset(r,  shift, 0)
    b = ImageChops.offset(b, -shift, 0)
    return Image.merge("RGB", (r, g, b))


def apply_scanlines(img: Image.Image, darken: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    arr[::2] *= darken
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def add_noise(img: Image.Image, strength: int) -> Image.Image:
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.randint(-strength, strength + 1, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def dropout_lines(img: Image.Image, chance: float = 0.015, max_width: int = 4) -> Image.Image:
    """Occasional horizontal darker bands — CRT tracking wobble."""
    arr = np.asarray(img).astype(np.float32)
    h = arr.shape[0]
    rng = np.random.default_rng()
    for y in range(h):
        if rng.random() < chance:
            w = rng.integers(1, max_width + 1)
            arr[y:y + w] *= rng.uniform(0.55, 0.85)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def build(palette_name: str) -> Path:
    palette = PALETTES[palette_name]
    ensure_raster()
    logo_raw = Image.open(WHITE_RASTER).convert("RGBA")
    ratio = LOGO_WIDTH_PX / logo_raw.width
    new_size = (LOGO_WIDTH_PX, int(logo_raw.height * ratio))
    logo = logo_raw.resize(new_size, Image.LANCZOS)

    tinted = tint_white_to(logo, palette["tint"])
    # Slight softening of the core glyphs (phosphor bleed)
    tinted_soft = tinted.filter(ImageFilter.GaussianBlur(radius=GRAIN_RADIUS))

    lx = (CANVAS - tinted.width) // 2
    ly = (CANVAS - tinted.height) // 2

    # Dark phosphor background
    canvas = Image.new("RGB", (CANVAS, CANVAS), palette["bg"])

    # Halo first (under the logo). Additive blend so it brightens the bg.
    halo = build_halo(tinted, palette["halo"], HALO_RADIUS, HALO_GAIN,
                      CANVAS, (lx, ly)).convert("RGB")
    canvas = ImageChops.add(canvas, halo)

    # Crisp tinted logo on top of the halo
    canvas.paste(tinted_soft, (lx, ly), tinted_soft)

    # CRT effects applied to the combined image
    canvas = chromatic_aberration(canvas, RGB_SHIFT_PX)
    canvas = apply_scanlines(canvas, SCANLINE_DARKEN)
    canvas = dropout_lines(canvas)
    canvas = add_noise(canvas, NOISE_STRENGTH)

    out = HERE / f"ds_logo_grungy_{palette_name}.png"
    canvas.save(out)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(PALETTES)
    for name in names:
        path = build(name)
        print(f"wrote {path}")
