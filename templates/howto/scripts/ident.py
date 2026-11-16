"""Fill the template from a brand's mark and ident: writes src/ident.json and copies the clips to public/ident/.

Run it when the brand changes, or when its mark or ident does:

    uv run templates/howto/scripts/ident.py                      # from the toolkit root, or
    uv run scripts/ident.py --brand my-brand                     # from a project made from this template
    uv run scripts/ident.py --episode key --short clap           # which ident opens and closes an episode, and
                                                                 # which closes a Short (default: the manifest's)
    uv run scripts/ident.py --none                               # no mark, no ident: the template's own 2D mark

The brand is the one src/brand.ts names. Two things are read, and either can be missing:

- `brands/<name>/assets/mark.svg` (and `mark-on-dark.svg`, `mark-white.svg` for dark and coloured grounds): the
  flat mark. It takes the place of the prompt chevron at the head of the track, in every scene that shows it.
- `brands/<name>/ident/ident.json` and the clips it lists: the ident. An episode opens with the open clip and
  ends on the close clip; a Short ends on the close clip in 9:16.

With neither, src/ident.json is `{}` and the template plays its own 2D mark, so it renders with no brand work
and no Blender. Nothing here needs Blender: the clips were rendered when the ident was made
(examples/blender-ident shows how).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLKIT = ROOT.parent.parent
OUT = ROOT / "src" / "ident.json"
CLIPS = ROOT / "public" / "ident"
FPS = 30


def brand_name() -> str:
    m = re.search(r"from\s+['\"][^'\"]*brands/([^/'\"]+)/brand\.json['\"]", (ROOT / "src" / "brand.ts").read_text())
    if not m:
        sys.exit("src/brand.ts does not import a brands/<name>/brand.json, so there is no brand to read: pass --brand")
    return m.group(1)


def read_mark(assets: Path) -> dict | None:
    """The mark's shapes as SVG path data in its own coordinates, with a fill for each kind of ground."""
    if not (assets / "mark.svg").exists():
        return None
    sys.path.insert(0, str(TOOLKIT / "lib" / "blender"))
    import svg  # the toolkit's SVG reader: plain Python, no Blender

    def parts_of(name):
        f = assets / name
        if not f.exists():
            return None
        parts, _, notes = svg.read(f)
        for n in notes:
            print(f"  {name}: {n}")
        return parts

    light = parts_of("mark.svg")
    if not light:
        return None
    dark, mono = parts_of("mark-on-dark.svg"), parts_of("mark-white.svg")
    x0, y0, x1, y1 = svg.bounds(light)

    def fill(other, i):
        return other[i].fill if other and len(other) == len(light) else None

    return {
        "box": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
        "parts": [
            {
                "id": p.id,
                "d": " ".join("M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in ring) + " Z" for ring in p.rings),
                "light": p.fill,
                "dark": fill(dark, i),   # None: the field's ink
                "mono": fill(mono, i),
            }
            for i, p in enumerate(light)
        ],
    }


def clip(folder: Path, manifest: dict, name: str, part: str, shape: str) -> dict | None:
    entry = manifest.get("idents", {}).get(name, {}).get(part)
    if not entry or shape not in entry.get("files", {}):
        return None
    if entry.get("fps", FPS) != FPS:
        sys.exit(f"{folder / 'ident.json'}: {name} {part} is {entry['fps']} fps and the template is {FPS}: render the ident at {FPS}")
    src = folder / entry["files"][shape]
    if not src.exists():
        sys.exit(f"{folder / 'ident.json'} lists {src.name}, which is not in {folder}")
    CLIPS.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, CLIPS / src.name)
    out = {"file": f"ident/{src.name}", "frames": entry["frames"]}
    if "title_frame" in entry:
        out["title_frame"] = entry["title_frame"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", help="a folder in brands/ (default: the one src/brand.ts names)")
    ap.add_argument("--episode", help="the ident that opens and closes an episode (default: as last time, else the manifest's default)")
    ap.add_argument("--short", help="the ident that closes a Short (default: as last time, else the manifest's default)")
    ap.add_argument("--none", action="store_true", help="use no mark and no ident: the template's own 2D mark")
    a = ap.parse_args()

    before = json.loads(OUT.read_text()) if OUT.exists() else {}
    if CLIPS.exists():
        shutil.rmtree(CLIPS)
    if a.none:
        OUT.write_text("{}\n")
        print("src/ident.json: empty, so the template plays its own 2D mark")
        return 0

    name = a.brand or brand_name()
    brand = TOOLKIT / "brands" / name
    if not brand.is_dir():
        sys.exit(f"no brand called {name!r} in {TOOLKIT / 'brands'}")
    out: dict = {"brand": name}

    mark = read_mark(brand / "assets")
    if mark:
        out["mark"] = mark

    folder = brand / "ident"
    if (folder / "ident.json").exists():
        manifest = json.loads((folder / "ident.json").read_text())
        have = list(manifest.get("idents", {}))
        default = manifest.get("default") or (have[0] if have else None)
        same = before.get("brand") == name
        for use, shape, asked in (("episode", "16x9", a.episode), ("short", "9x16", a.short)):
            pick = asked or (before.get(use, {}).get("name") if same else None) or default
            if pick not in have:
                sys.exit(f"{folder / 'ident.json'} has no ident called {pick!r} (have: {', '.join(have) or 'none'})")
            got = {part: c for part in ("open", "close") if (c := clip(folder, manifest, pick, part, shape))}
            if use == "short":
                got.pop("open", None)  # a Short opens on its hook; only the close plays
                if (CLIPS / f"{pick}-open-9x16.mp4").exists():
                    (CLIPS / f"{pick}-open-9x16.mp4").unlink()
            if got:
                out[use] = {"name": pick, **got}

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"src/ident.json: brand {name}")
    print(f"  mark: {len(mark['parts'])} part(s) from {brand.name}/assets/mark.svg" if mark else "  mark: none, so the template's prompt chevron stays")
    for use in ("episode", "short"):
        if use in out:
            parts = ", ".join(f"{p} {out[use][p]['frames'] / FPS:.1f} s" for p in ("open", "close") if p in out[use])
            print(f"  {use}: {out[use]['name']} ({parts})")
        else:
            print(f"  {use}: no ident, so the 2D opening and close play")
    return 0


if __name__ == "__main__":
    sys.exit(main())
