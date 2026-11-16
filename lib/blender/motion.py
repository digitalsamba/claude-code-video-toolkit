"""Beats that move a studio prop's parts, light it and place sounds: the timing, without Blender.

A studio set's beats name a part of a prop (`"part": "mark.slash"`: the prop's id, a dot, the part's id from the
SVG) or the whole prop (`"part": "mark"`). Each beat sets one thing going from where it is to `to`, between `at`
and `at + dur` (or `until`), along an ease:

    {"at": "open+0.67", "dur": 0.17, "do": "shear", "part": "mark.slash", "to": 0, "ease": "in"}

A Track is the value of one such thing at every frame; logo.py and studio.py key what the tracks say. compile.py
validates with `check_beat` before Blender builds anything, so a wrong verb, part or ease is a named error.
tests/python/test_blender_ident.py runs this file on the host.
"""
from __future__ import annotations

import math

from timeline import SpecError


def _spring(t):
    """0 to 1 with one small overshoot (about 6%), settled at 1."""
    return 1 - math.exp(-5 * t) * math.cos(t * math.pi * 1.5)


EASES = {
    "linear": lambda t: t,
    "in": lambda t: t ** 3,                      # slow start, arrives at speed: something falling, a snap
    "out": lambda t: 1 - (1 - t) ** 3,           # leaves at speed and settles
    "in_out": lambda t: t * t * (3 - 2 * t),
    "spring": _spring,                           # arrives past its mark and comes back: something knocked into place
}
ANCHORS = {"centre": (0.5, 0.5), "center": (0.5, 0.5), "left": (0.0, 0.5), "right": (1.0, 0.5), "top": (0.5, 1.0),
           "bottom": (0.5, 0.0), "top_left": (0.0, 1.0), "top_right": (1.0, 1.0), "bottom_left": (0.0, 0.0),
           "bottom_right": (1.0, 0.0)}
AXES = ("auto", "x", "y", "z")
FEET = ("bottom", "top", "left", "right")
LIGHTS = ("key", "fill", "rim")
# verb -> what it needs. "target": part (a part only), any (a part or a whole prop), prop (optional, a prop or part)
VERBS = {
    "turn":  dict(target="any", to="degrees"),
    "shear": dict(target="part", to="degrees"),
    "grow":  dict(target="any", to="share"),
    "move":  dict(target="any", to="xyz"),
    "sink":  dict(target="any", to="metres"),
    "press": dict(target="any"),
    "show":  dict(target="any"),
    "hide":  dict(target="any"),
    "blink": dict(target="any"),
    "glow":  dict(target="any", to="strength"),
    "light": dict(to="share"),
    "flash": dict(target="prop"),
    "sweep": dict(target="prop"),
    "hit":   dict(target="prop"),
    "sound": dict(),
}
GLOW_SAFE = 1.0  # emission above this clips a saturated colour toward white (the brand blue goes cyan)


class Track:
    """One number over the frames of a film: it holds `start` until a move, and its last value after."""

    def __init__(self, start=0.0):
        self.start, self.moves = float(start), []  # moves: (f0, f1, to, ease), in the order they begin

    def to(self, f0, f1, value, ease="in_out"):
        self.moves.append((int(f0), max(int(f0), int(f1)), float(value), ease))
        self.moves.sort(key=lambda m: m[0])

    def at(self, f) -> float:
        v = self.start
        for i, (f0, f1, to, ease) in enumerate(self.moves):
            if f < f0:
                break
            if f >= f1:
                v = to
                continue
            # part-way: from where the track was when this move began
            before = Track(self.start)
            before.moves = self.moves[:i]
            v0 = before.at(f0)
            v = v0 + (to - v0) * EASES[ease]((f - f0) / (f1 - f0))
            nxt = self.moves[i + 1][0] if i + 1 < len(self.moves) else None
            if nxt is None or f < nxt:
                break
        return v

    def span(self):
        return (min(m[0] for m in self.moves), max(m[1] for m in self.moves)) if self.moves else None

    def __bool__(self):
        return bool(self.moves)


def anchor(value) -> tuple[float, float]:
    """A named spot on a part's box, or [u, v] as shares of it from the bottom left."""
    if isinstance(value, str) and value in ANCHORS:
        return ANCHORS[value]
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(_num(v) for v in value):
        return float(value[0]), float(value[1])
    raise ValueError(f"{value!r} is not a spot on a part: use one of {', '.join(sorted(set(ANCHORS) - {'center'}))}, "
                     f"or [u, v] as shares of its box from the bottom left")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def blink_frames(f0, f1, fps, every=0.8, off=0.27):
    """(frame, visible) changes for a cursor that blinks from f0 to f1: on first, off for `off` s of each `every`."""
    out, t = [(f0, True)], 0.0
    while True:
        a = f0 + int(round((t + every - off) * fps))
        b = f0 + int(round((t + every) * fps))
        if a >= f1:
            break
        out.append((a, False))
        if b >= f1:
            break
        out.append((b, True))
        t += every
    return out


def split_target(ref, props):
    """'mark.slash' -> ('mark', 'slash'); 'mark' -> ('mark', None). `props` maps a prop id to its part ids
    (None for a prop with no parts: a phone, a card, a model)."""
    if not isinstance(ref, str) or not ref:
        raise ValueError(f'"part" names a prop or one of its parts, as "mark" or "mark.slash" ({ref!r} given)')
    pid, _, part = ref.partition(".")
    if pid not in props:
        raise ValueError(f'no prop called {pid!r} (have: {", ".join(props)})')
    if part:
        have = props[pid]
        if not have:
            raise ValueError(f'prop {pid!r} has no parts: only a logo is built from parts')
        if part not in have:
            raise ValueError(f'prop {pid!r} has no part called {part!r} (have: {", ".join(have)})')
    return pid, part or None


def check_beat(b, props, sounds=()):
    """Raise SpecError if a studio beat is wrong. `props` as in split_target; `sounds` the spec's sound names."""
    where = f'beat at {b.get("at")!r}'
    verb = b.get("do")
    if verb not in VERBS:
        raise SpecError(f"{where}: unknown verb {verb!r} for a studio set (have: {', '.join(sorted(VERBS))})")
    need = VERBS[verb]
    try:
        first = next(iter(props))
        target = need.get("target")
        if target:
            ref = b.get("part", first if target == "prop" else None)
            if ref is None:
                raise ValueError(f'{verb} needs "part": a prop, or one of its parts as "mark.slash"')
            pid, part = split_target(ref, props)
            if target == "part" and part is None:
                raise ValueError(f'{verb} works on one part of a logo, as "{pid}.<part>" (have: '
                                 f'{", ".join(props[pid] or ()) or "none: this prop has no parts"})')
        if b.get("ease", "in_out") not in EASES:
            raise ValueError(f'"ease" is one of {", ".join(EASES)}, not {b["ease"]!r}')
        kind = need.get("to")
        to = b.get("to")
        if kind in ("degrees", "metres", "share", "strength") and not _num(to):
            unit = {"degrees": "a number of degrees", "metres": "a distance in metres", "share": "a share (1 is whole)",
                    "strength": "a strength (0 is off, 1 is the colour at full)"}[kind]
            raise ValueError(f'{verb} needs "to": {unit}')
        if kind in ("share", "strength") and to < 0:
            raise ValueError(f'{verb}: "to" cannot be below 0')
        if kind == "xyz" and not (isinstance(to, (list, tuple)) and len(to) == 3 and all(_num(v) for v in to)):
            raise ValueError('move needs "to": [right, up, toward the viewer] in metres from where the part was made')
        if verb == "turn":
            anchor(b.get("hinge", "centre"))
            if b.get("axis", "auto") not in AXES:
                raise ValueError(f'"axis" is one of {", ".join(AXES)} (x: a flap, y: a door, z: in the logo\'s own plane)')
        elif verb == "grow":
            anchor(b.get("from", "centre"))
        elif verb == "shear" and b.get("foot", "bottom") not in FEET:
            raise ValueError(f'"foot" is the edge that stays put, one of {", ".join(FEET)}')
        elif verb == "press":
            if not _num(b.get("depth")) or b["depth"] <= 0:
                raise ValueError('press needs "depth": how far it goes down, in metres')
            if not _num(b.get("back", 0.4)) or b.get("back", 0.4) < 0:
                raise ValueError('"back" is how long it takes to come up again, in seconds (0 to stay down)')
        elif verb == "blink":
            every, off = b.get("every", 0.8), b.get("off", 0.27)
            if not (_num(every) and _num(off) and 0 < off < every):
                raise ValueError('blink takes "every" and "off" in seconds, with "off" the shorter')
            if "dur" not in b and "until" not in b:
                raise ValueError('blink needs "dur" or "until": how long it goes on')
        elif verb == "light":
            if b.get("light") not in LIGHTS:
                raise ValueError(f'light needs "light", one of {", ".join(LIGHTS)}')
        elif verb in ("flash", "sweep"):
            if not _num(b.get("strength", 1.0)) or b.get("strength", 1.0) < 0:
                raise ValueError(f'{verb} takes "strength": 1 is about the key light')
            if verb == "sweep" and b.get("from", "left") not in ("left", "right"):
                raise ValueError('sweep takes "from": "left" or "right", the side the light comes in from')
        elif verb == "hit":
            for k in ("shake", "flash"):
                if not _num(b.get(k, 0)) or b.get(k, 0) < 0:
                    raise ValueError(f'hit takes "{k}" as a number, 0 for none')
        if verb == "sound" and "sound" not in b:
            raise ValueError('a sound beat needs "sound": the name of a sound')
        if "sound" in b and (not isinstance(b["sound"], str) or (sounds and b["sound"] not in sounds)):
            raise ValueError(f'"sound" is the name of a sound in the spec\'s "sounds" ({b["sound"]!r} given; have: '
                             f'{", ".join(sounds) or "none"})')
    except ValueError as e:
        raise SpecError(f"{where}: {e}")


def glow_note(b) -> str | None:
    if b.get("do") == "glow" and _num(b.get("to")) and b["to"] > GLOW_SAFE:
        return (f'beat at {b["at"]!r}: glow {b["to"]} is past {GLOW_SAFE:g}, where a saturated colour clips toward '
                f'white: light the part with a flash, or use the look\'s "bloom", to make it glow more')
    return None
