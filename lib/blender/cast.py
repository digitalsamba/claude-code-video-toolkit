"""Character descriptions: what a cast member looks like, as data (plain Python, no Blender).

A character is a small JSON object an agent can write from a sentence or a photo: every field is a named choice
or a hex colour, so "a woman in her forties with a grey bob, glasses and a green shirt" maps onto it directly.

    {"id": "maya", "kind": "clay_person", "seed": 12,
     "body":    {"build": "slim", "skin": "brown"},
     "face":    {"shape": "oval", "eyes": "#3B2A1A", "brows": "thin", "nose": "small", "beard": "none"},
     "hair":    {"style": "bob", "color": "#8A8F98"},
     "clothes": {"top": "shirt", "top_color": "#2E7D5B", "pants": "#252B45", "shoes": "#F0F0EC"},
     "accessories": ["glasses"]}

`resolve()` fills what is left out. With no `seed`, the gaps take the defaults (the character the examples use).
With a `seed`, the gaps are drawn from it, so two seeds give two people and the same seed gives the same person
on every machine. A character can live in its own file and be reused across videos: `{"id": "maya", "from":
"../../kits/acme/cast/maya.json"}` reads it (path relative to the spec), and fields beside `from` override it, which
is how one video re-costumes a standing cast member.

compile.py uses this inside Blender; tests/python/test_blender_spec.py runs it on the host.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from timeline import SpecError

SKIN = {"pale": "#F3D3C2", "light": "#E2A98A", "tan": "#C98E66", "brown": "#9B6645", "dark": "#6A4530",
        "deep": "#4A2F22"}
HAIR_COLORS = {"black": "#17130F", "dark": "#2B211B", "brown": "#3A2513", "chestnut": "#6B3F22", "red": "#A4441F",
               "blonde": "#D9B26A", "grey": "#8A8F98", "white": "#E4E4E0"}
EYE_COLORS = {"brown": "#3B2412", "hazel": "#6B5326", "green": "#3F6B45", "blue": "#3C6C9E", "grey": "#69737D"}

CHOICES = {
    ("body", "build"): ["average", "slim", "broad"],
    ("face", "shape"): ["round", "oval", "square"],
    ("face", "brows"): ["medium", "thin", "thick"],
    ("face", "nose"): ["round", "small", "long"],
    ("face", "beard"): ["none", "moustache", "full"],
    ("hair", "style"): ["tufts", "short", "bob", "long", "bun", "curly", "bald"],
    ("clothes", "top"): ["hoodie", "tee", "shirt"],
}
ACCESSORIES = ["glasses", "headphones", "earrings"]
DEFAULTS = {
    "body": {"build": "average", "skin": "#E2A98A"},
    "face": {"shape": "round", "eyes": "#17110D", "brows": "medium", "nose": "round", "beard": "none"},
    "hair": {"style": "tufts", "color": "#3A2513"},
    "clothes": {"top": "hoodie", "top_color": "#F26B1F", "pants": "#252B45", "shoes": "#F0F0EC"},
    "accessories": [],
}
TOP_COLORS = ["#F26B1F", "#3771E0", "#2E7D5B", "#B23A48", "#6C4AB6", "#D9A21B", "#3A4A5C", "#E8E4DA"]
PANTS = ["#252B45", "#3B3B3F", "#5B4636", "#6F7A86"]
_HEX = re.compile(r"#[0-9a-fA-F]{6}$")


def _colour(value, names, where):
    if isinstance(value, str) and value in names:
        return names[value]
    if isinstance(value, str) and _HEX.match(value):
        return value
    raise SpecError(f"{where}: {value!r} is not a hex colour like #3A2513" +
                    (f" or one of {', '.join(names)}" if names else ""))


def _seeded(rng):
    """A whole look drawn from a seed: used for whatever the description leaves out."""
    beard = rng.choices(CHOICES[("face", "beard")], [6, 1, 2])[0]
    return {
        "body": {"build": rng.choice(CHOICES[("body", "build")]), "skin": rng.choice(list(SKIN.values()))},
        "face": {"shape": rng.choice(CHOICES[("face", "shape")]), "eyes": rng.choice(list(EYE_COLORS.values())),
                 "brows": rng.choice(CHOICES[("face", "brows")]), "nose": rng.choice(CHOICES[("face", "nose")]),
                 "beard": beard},
        "hair": {"style": rng.choice(CHOICES[("hair", "style")]), "color": rng.choice(list(HAIR_COLORS.values()))},
        "clothes": {"top": rng.choice(CHOICES[("clothes", "top")]), "top_color": rng.choice(TOP_COLORS),
                    "pants": rng.choice(PANTS), "shoes": rng.choice(["#F0F0EC", "#2A2A2E", "#8A5A3C"])},
        "accessories": [a for a, p in zip(ACCESSORIES, (0.3, 0.12, 0.15)) if rng.random() < p],
    }


def load(entry: dict, base: Path | None = None) -> dict:
    """A cast entry with its `from` file read in underneath it (the entry's own fields win, group by group)."""
    if "from" not in entry:
        return entry
    path = Path(entry["from"])
    path = path if path.is_absolute() else (base or Path(".")) / path
    if not path.exists():
        raise SpecError(f"cast member {entry.get('id')!r}: character file not found: {path}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    merged = {**stored, **{k: v for k, v in entry.items() if k != "from"}}
    for group in ("body", "face", "hair", "clothes"):
        if isinstance(stored.get(group), dict) and isinstance(entry.get(group), dict):
            merged[group] = {**stored[group], **entry[group]}
    return merged


def resolve(entry: dict, base: Path | None = None) -> dict:
    """A cast entry -> every appearance field filled and checked. Raises SpecError naming the field and its choices."""
    entry = load(entry, base)
    cid = entry.get("id", "?")
    seed = entry.get("seed")
    fill = _seeded(random.Random(int(seed))) if seed is not None else DEFAULTS
    out = {"id": cid, "seed": int(seed) if seed is not None else 7}
    for group in ("body", "face", "hair", "clothes"):
        given = entry.get(group, {})
        if not isinstance(given, dict):
            raise SpecError(f'cast member {cid!r}: "{group}" is an object, e.g. {json.dumps(DEFAULTS[group])}')
        unknown = set(given) - set(DEFAULTS[group])
        if unknown:
            raise SpecError(f'cast member {cid!r}: unknown {group} field {sorted(unknown)[0]!r} '
                            f'(have: {", ".join(DEFAULTS[group])})')
        out[group] = {**fill[group], **given}
    # the older flat form: "colors": {"top", "pants", "hair", "skin", "shoes"}
    old = entry.get("colors") or {}
    for key, (group, field) in {"top": ("clothes", "top_color"), "pants": ("clothes", "pants"),
                                "shoes": ("clothes", "shoes"), "hair": ("hair", "color"),
                                "skin": ("body", "skin")}.items():
        if key in old:
            out[group][field] = old[key]
    for (group, field), names in CHOICES.items():
        if out[group][field] not in names:
            raise SpecError(f'cast member {cid!r}: {group}.{field} is one of {", ".join(names)}, '
                            f'not {out[group][field]!r}')
    where = f"cast member {cid!r}"
    out["body"]["skin"] = _colour(out["body"]["skin"], SKIN, f"{where} body.skin")
    out["face"]["eyes"] = _colour(out["face"]["eyes"], EYE_COLORS, f"{where} face.eyes")
    out["hair"]["color"] = _colour(out["hair"]["color"], HAIR_COLORS, f"{where} hair.color")
    for field in ("top_color", "pants", "shoes"):
        out["clothes"][field] = _colour(out["clothes"][field], {}, f"{where} clothes.{field}")
    acc = entry.get("accessories", fill["accessories"])
    if not isinstance(acc, list) or any(a not in ACCESSORIES for a in acc):
        raise SpecError(f'{where}: "accessories" is a list from {", ".join(ACCESSORIES)}, not {acc!r}')
    out["accessories"] = list(acc)
    return out
