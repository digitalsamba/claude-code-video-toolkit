#!/usr/bin/env python3
"""A sheet that shows whether a logo holds up: every size, every ground, a circle crop, a watermark.

A mark that looks right at 512 px on white can close up at 16 px, vanish on the brand's dark ground, or lose a
corner to a round avatar. This draws one PNG with all of those side by side, and measures what it can:

    uv run tools/mark_sheet.py --svg brands/my-brand/assets/mark.svg --brand my-brand
    uv run tools/mark_sheet.py --svg mark.svg --ground "#ffffff" --ground "#101820" --frame out/still.jpg
    uv run tools/mark_sheet.py --svg mark.svg --on-dark mark-on-dark.svg --brand my-brand --json

On the sheet: the mark at 512, 150, 64, 32 and 16 px; the 16 and 32 px renders enlarged so each pixel shows;
the mark on each ground; in a circle at 150 and 40 px (an avatar, and the same beside a comment); and in one
colour at 70% in the corner of a video frame, as a watermark sits.

Measured, and printed as warnings: each part's contrast with each ground (under 3:1 it is hard to see), any
part the circle crop cuts, and any part smaller than two pixels at 16 px. `--strict` makes a warning exit 1.

Grounds come from `--brand` (its bgLight, bgAlt, darkest ground and primary colour) or from `--ground`. Give
the mark's other versions with `--on-dark` and `--mono` (one colour): each ground is shown with the first of
the three whose every part is 3:1 or better on it. The watermark uses `--frame` pictures (repeatable);
without one it sits on a plain light and a plain dark frame.

The SVG is read as flat filled shapes (lib/blender/svg.py, the reader the Blender `logo` prop uses, so the sheet
and a 3D ident agree on the shape). Strokes, text and gradients are not read; the tool says when it meets one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib" / "blender"))

import svg  # noqa: E402

SIZES = (512, 150, 64, 32, 16)
ZOOMS = ((16, 8), (32, 4))
MIN_CONTRAST = 3.0   # WCAG's floor for a graphic against what is behind it
INK, PAPER, PAGE = "#121418", "#ffffff", "#eceef2"


def rgb(hexc: str) -> tuple[int, int, int]:
    h = hexc.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def luminance(hexc: str) -> float:
    lin = [(c / 255 / 12.92) if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in rgb(hexc)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def is_dark(hexc: str) -> bool:
    return luminance(hexc) < 0.18


def masks(parts, box, size: int):
    """Each part's coverage at `size` px (an 'L' image): drawn large and averaged down, so a pixel half covered
    is half on, as a browser draws it."""
    from PIL import Image, ImageChops, ImageDraw
    x, y, w, h = box
    ss = max(4, min(16, 2048 // size))
    big = size * ss
    k = big / max(w, h)
    ox, oy = (big - w * k) / 2 - x * k, (big - h * k) / 2 - y * k
    out = []
    for p in parts:
        m = Image.new("L", (big, big), 0)
        for ring in p.rings:  # an outline inside another is a hole
            one = Image.new("L", (big, big), 0)
            # (half a pixel back: the drawing library counts a pixel as on if its centre is on or inside the edge)
            ImageDraw.Draw(one).polygon([(px * k + ox - 0.5, py * k + oy - 0.5) for px, py in ring], fill=255)
            m = ImageChops.difference(m, one)
        out.append(m.resize((size, size), Image.BOX))
    return out


def draw(parts, box, size: int, colours=None, opacity: float = 1.0):
    """The mark at `size` px on a transparent ground. `colours`: one colour for every part, or a list."""
    from PIL import Image
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for i, (p, m) in enumerate(zip(parts, masks(parts, box, size))):
        c = p.fill if colours is None else colours if isinstance(colours, str) else colours[i]
        layer = Image.new("RGBA", (size, size), (*rgb(c), 255))
        layer.putalpha(m.point(lambda v: int(v * opacity)))
        im = Image.alpha_composite(im, layer)
    return im


def brand_grounds(name: str) -> list[tuple[str, str]]:
    path = ROOT / "brands" / name / "brand.json"
    if not path.exists():
        sys.exit(f"no brand called {name!r} (looked for {path})")
    c = json.loads(path.read_text(encoding="utf-8")).get("colors", {})
    flat = {k: v for k, v in c.items() if isinstance(v, str) and v.startswith("#") and len(v) == 7}
    dark = min((v for k, v in flat.items() if k.lower().startswith(("bg", "navy", "dark", "ground"))), key=luminance,
               default="#101820")
    picks = [("light", flat.get("bgLight", PAPER)), ("alt", flat.get("bgAlt", "#f1f2f4")), ("dark", dark),
             ("primary", flat.get("primary", "#3771e0"))]
    seen, out = set(), []
    for label, v in picks:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append((label, v))
    return out


def pick(versions: dict, ground: str):
    """The version of the mark to use on a ground, as (its name, its parts): the first, in the order given, whose
    every part is 3:1 or better there; failing that, the one with the most contrast."""
    floor = {n: min(contrast(p.fill, ground) for p in parts) for n, parts in versions.items()}
    name = next((n for n in versions if floor[n] >= MIN_CONTRAST), max(floor, key=floor.get))
    return name, versions[name]


def measure(parts, box, versions, grounds) -> tuple[dict, list[str]]:
    """What can be said without looking: contrast on each ground, the circle crop, and what is left at 16 px."""
    warnings, report = [], {"contrast": {}, "circle": [], "at_16px": {}}
    for label, g in grounds:
        name, use = pick(versions, g)
        row = {p.id: round(contrast(p.fill, g), 2) for p in use}
        report["contrast"][g] = {"version": name, **row}
        for pid, c in row.items():
            if c < MIN_CONTRAST:
                fill = next(p.fill for p in use if p.id == pid)
                hint = "" if len(versions) > 2 else (" (give a version for it with --on-dark or --mono)")
                warnings.append(f"{pid} ({fill}) on the {label} ground {g}: contrast {c}:1, under {MIN_CONTRAST:g}:1{hint}")
    x, y, w, h = box
    cx, cy, r = x + w / 2, y + h / 2, min(w, h) / 2
    for p in parts:
        far = max(((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 for ring in p.rings for px, py in ring)
        if far > r:
            report["circle"].append(p.id)
            warnings.append(f"{p.id} reaches {far / r:.0%} of the way to the circle's edge and past it: a round avatar "
                            f"cuts it. Make the mark smaller in its view box")
    for p, m in zip(parts, masks(parts, box, 16)):
        px = sum(m.getdata()) / 255
        report["at_16px"][p.id] = round(px, 1)
        if px < 2:
            warnings.append(f"{p.id} covers {px:.1f} px at 16 px: it is gone at favicon size")
    return report, warnings


def sheet(parts, box, versions, grounds, frames, title: str):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, small = ImageFont.load_default(18), ImageFont.load_default(13)
    except TypeError:  # an older Pillow: one size only
        font = small = ImageFont.load_default()
    pad, W = 28, 1500
    light = grounds[0][1] if not is_dark(grounds[0][1]) else PAPER
    rows = []  # (heading, [(image, caption)])

    def on(ground, size, circle=False, use=None):
        use = use or pick(versions, ground)[1]
        tile = Image.new("RGBA", (size, size), (*rgb(ground), 255))
        tile = Image.alpha_composite(tile, draw(use, box, size))
        if circle:
            mask = Image.new("L", (size * 4, size * 4), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
            cut = Image.new("RGBA", (size, size), (*rgb(PAGE), 255))
            cut.paste(tile, (0, 0), mask.resize((size, size), Image.LANCZOS))
            tile = cut
        return tile.convert("RGB")

    rows.append((f"Sizes, on {light}", [(on(light, s), f"{s} px") for s in SIZES]
                 + [(on(light, s).resize((s * z, s * z), Image.NEAREST), f"{s} px, each pixel x{z}") for s, z in ZOOMS]))
    rows.append(("Grounds", [(on(g, 150), f"{label} {g}: {pick(versions, g)[0]}") for label, g in grounds]))
    rows.append(("In a circle: an avatar at 150 px, and at 40 px as beside a comment",
                 [(on(g, 150, circle=True), g) for _, g in grounds] + [(on(g, 40, circle=True), "40 px") for _, g in grounds]))
    marks = []
    for path, im in frames:
        im = im.convert("RGB")
        im.thumbnail((440, 440))
        im = im.copy()
        tone = sum(im.resize((1, 1)).getpixel((0, 0))) / 3
        wm = draw(parts, box, max(28, im.width // 10), PAPER if tone < 140 else INK, 0.7)
        im.paste(wm, (im.width - wm.width - 12, im.height - wm.height - 12), wm)
        marks.append((im, path))
    rows.append(("Watermark: one colour at 70%, in the corner of a frame", marks))

    height = pad + 44
    for _, items in rows:
        height += 34 + max(im.height for im, _ in items) + 26 + pad
    page = Image.new("RGB", (W, height), rgb(PAGE))
    d = ImageDraw.Draw(page)
    d.text((pad, pad), title, fill=rgb(INK), font=font)
    y = pad + 44
    for heading, items in rows:
        d.text((pad, y), heading, fill=rgb("#323e66"), font=font)
        y += 34
        x, tall = pad, max(im.height for im, _ in items)
        for im, cap in items:
            page.paste(im, (x, y + tall - im.height))
            d.text((x, y + tall + 6), str(cap)[:max(8, (max(im.width, 70) + 60) // 7)], fill=rgb("#5e667a"), font=small)
            x += max(im.width, 70) + 70
        y += tall + 26 + pad
    return page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--svg", required=True, type=Path, help="the mark, as flat filled shapes")
    ap.add_argument("--on-dark", type=Path, help="the version for dark grounds (same shapes, other colours)")
    ap.add_argument("--mono", type=Path, help="the one-colour version, for a ground the mark's own colours are lost on")
    ap.add_argument("--brand", help="take the grounds from brands/<name>/brand.json")
    ap.add_argument("--ground", action="append", default=[], help="a ground colour as #rrggbb (repeatable)")
    ap.add_argument("--frame", action="append", default=[], type=Path, help="a video frame for the watermark (repeatable)")
    ap.add_argument("--output", type=Path, help="the sheet (default: <svg name>-sheet.png beside the SVG)")
    ap.add_argument("--json", action="store_true", help="print the measurements as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything is warned about")
    a = ap.parse_args()
    try:
        parts, box, notes = svg.read(a.svg)
        versions = {"as drawn": parts}
        for name, path in (("on dark", a.on_dark), ("one colour", a.mono)):
            if path:
                versions[name] = svg.read(path)[0]
    except svg.SvgError as e:
        print(f"ERROR: {e}")
        return 2
    grounds = [(f"ground {i + 1}", g) for i, g in enumerate(a.ground)]
    for _, g in grounds:
        if svg.colour(g) != g.lower() or len(g) != 7:
            print(f"ERROR: --ground takes a colour as #rrggbb ({g!r} given)")
            return 2
    if a.brand:
        grounds = brand_grounds(a.brand) + grounds
    if not grounds:
        grounds = [("light", PAPER), ("alt", "#f1f2f4"), ("dark", "#101820")]
    from PIL import Image
    frames = []
    for f in a.frame:
        if not f.exists():
            print(f"ERROR: frame not found: {f}")
            return 2
        frames.append((f.name, Image.open(f)))
    if not frames:
        frames = [("a light frame", Image.new("RGB", (440, 248), rgb("#e3e5ea"))),
                  ("a dark frame", Image.new("RGB", (440, 248), rgb("#1b2030")))]
    report, warnings = measure(parts, box, versions, grounds)
    out = a.output or a.svg.with_name(a.svg.stem + "-sheet.png")
    tmp = out.with_name(out.name + ".tmp.png")
    sheet(parts, box, versions, grounds, frames, f"{a.svg.name}: {len(parts)} part(s), " + ", ".join(
        f"{p.id} {p.fill}" for p in parts)).save(tmp)
    tmp.replace(out)
    report.update(sheet=str(out), parts=[{"id": p.id, "fill": p.fill} for p in parts], notes=notes, warnings=warnings)
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{a.svg.name}: {len(parts)} part(s): " + ", ".join(f"{p.id} {p.fill}" for p in parts))
        for n in notes:
            print(f"  note  {n}")
        for w in warnings:
            print(f"  WARN  {w}")
        if not warnings:
            print("  nothing to warn about: every part is 3:1 or better on every ground, inside the circle, and there at 16 px")
        print(f"  sheet: {out}")
    return 1 if warnings and a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
