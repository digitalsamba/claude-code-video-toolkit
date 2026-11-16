"""Read a flat SVG logo into parts: one per filled shape, each a set of closed outlines (plain Python, no Blender).

    parts, box, notes = svg.read("mark.svg")
    parts[0].id, parts[0].fill, parts[0].rings      # "slash", "#3771e0", [[(x, y), ...], ...]

A part is one `<path>`, `<polygon>`, `<rect>`, `<circle>` or `<ellipse>` with a fill. Its id is the element's
own `id`, or `p1`, `p2`... in document order. Curves are flattened to short straight runs. Coordinates are the
SVG's own (y down), after every `transform` on the element and its groups. A second outline inside the first is
a hole (the counter of an O), filled even-odd.

What it does not read is said in `notes`, not guessed: strokes (outline them first), text (convert it to
outlines), gradients (the part takes a flat grey), `<use>`, images, masks and clip paths.

logo.py builds the 3D prop from this inside Blender; tools/mark_sheet.py and the last-frame check draw the same
outlines flat, so the three never disagree about the mark's shape.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

NAMED = {"black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000", "blue": "#0000ff",
         "yellow": "#ffff00", "orange": "#ffa500", "purple": "#800080", "gray": "#808080", "grey": "#808080",
         "silver": "#c0c0c0", "navy": "#000080", "teal": "#008080", "lime": "#00ff00", "aqua": "#00ffff",
         "cyan": "#00ffff", "fuchsia": "#ff00ff", "magenta": "#ff00ff", "maroon": "#800000", "olive": "#808000",
         "pink": "#ffc0cb", "brown": "#a52a2a", "gold": "#ffd700", "currentcolor": "#000000"}
SKIPPED = {"defs", "clipPath", "mask", "symbol", "pattern", "marker", "metadata", "title", "desc", "style",
           "linearGradient", "radialGradient", "filter"}
UNREAD = {"text": "text is not read: convert it to outlines", "use": "<use> is not read: paste the shape in",
          "image": "an embedded image is not read: a logo prop takes vector shapes"}
CURVE_STEPS = 24  # straight runs per curve segment or quarter turn


class SvgError(ValueError):
    pass


class Part:
    def __init__(self, pid, fill, rings):
        self.id, self.fill, self.rings = pid, fill, rings

    def box(self):
        xs = [x for r in self.rings for x, _ in r]
        ys = [y for r in self.rings for _, y in r]
        return min(xs), min(ys), max(xs), max(ys)

    def __repr__(self):
        return f"Part({self.id!r}, {self.fill}, {len(self.rings)} outline(s))"


def colour(text) -> str | None:
    """An SVG fill as '#rrggbb', or None for no fill. A gradient or anything unknown comes back as ''."""
    t = (text or "").strip().lower()
    if t in ("none", "transparent"):
        return None
    if t in NAMED:
        return NAMED[t]
    m = re.fullmatch(r"#([0-9a-f]{3,8})", t)
    if m:
        h = m.group(1)
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h[:3])
        if len(h) in (6, 8):
            return "#" + h[:6]
    m = re.fullmatch(r"rgba?\(([^)]*)\)", t)
    if m:
        vals = [v for v in re.split(r"[\s,/]+", m.group(1).strip()) if v][:3]
        if len(vals) == 3:
            out = [float(v[:-1]) * 2.55 if v.endswith("%") else float(v) for v in vals]
            return "#" + "".join(f"{max(0, min(255, round(v))):02x}" for v in out)
    return ""


# ---------------------------------------------------------------- transforms (a, b, c, d, e, f)
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def mul(m, n):
    a, b, c, d, e, f = m
    g, h, i, j, k, l = n  # noqa: E741
    return (a * g + c * h, b * g + d * h, a * i + c * j, b * i + d * j, a * k + c * l + e, b * k + d * l + f)


def apply(m, p):
    a, b, c, d, e, f = m
    return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


def transform(text) -> tuple:
    m = IDENTITY
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            t = tuple(v)
        elif name == "translate" and v:
            t = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale" and v:
            t = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate" and v:
            c, s = math.cos(math.radians(v[0])), math.sin(math.radians(v[0]))
            t = (c, s, -s, c, 0, 0)
            if len(v) == 3:
                t = mul(mul((1, 0, 0, 1, v[1], v[2]), t), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX" and v:
            t = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY" and v:
            t = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            raise SvgError(f"transform not understood: {name}({args})")
        m = mul(m, t)
    return m


# ---------------------------------------------------------------- outlines
def _cubic(p0, p1, p2, p3):
    return [tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
                  for a, b, c, d in zip(p0, p1, p2, p3)) for t in (i / CURVE_STEPS for i in range(1, CURVE_STEPS + 1))]


def _quad(p0, p1, p2):
    return [tuple((1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c for a, b, c in zip(p0, p1, p2))
            for t in (i / CURVE_STEPS for i in range(1, CURVE_STEPS + 1))]


def _arc(p0, rx, ry, rot, large, sweep, p1):
    """An SVG elliptical arc as points (the end included), by the spec's centre form."""
    if rx == 0 or ry == 0 or p0 == p1:
        return [p1]
    rx, ry, phi = abs(rx), abs(ry), math.radians(rot)
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = c * dx + s * dy, -s * dx + c * dy
    lam = x1 * x1 / (rx * rx) + y1 * y1 / (ry * ry)
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    k = math.sqrt(max(0.0, num / (rx * rx * y1 * y1 + ry * ry * x1 * x1))) * (-1 if large == sweep else 1)
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx, cy = c * cx1 - s * cy1 + (p0[0] + p1[0]) / 2, s * cx1 + c * cy1 + (p0[1] + p1[1]) / 2
    a0 = math.atan2((y1 - cy1) / ry, (x1 - cx1) / rx)
    a1 = math.atan2((-y1 - cy1) / ry, (-x1 - cx1) / rx)
    da = a1 - a0
    if sweep and da < 0:
        da += 2 * math.pi
    elif not sweep and da > 0:
        da -= 2 * math.pi
    n = max(2, math.ceil(abs(da) / (math.pi / 2) * CURVE_STEPS))
    out = []
    for i in range(1, n + 1):
        a = a0 + da * i / n
        out.append((cx + rx * math.cos(a) * c - ry * math.sin(a) * s, cy + rx * math.cos(a) * s + ry * math.sin(a) * c))
    out[-1] = p1
    return out


def path_rings(d: str) -> list[list[tuple[float, float]]]:
    """The subpaths of a path's `d`, each as a list of points. An open subpath is closed, as a fill closes it."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d or "")
    rings, ring, i = [], [], 0
    cur = start = (0.0, 0.0)
    cmd, last_ctrl, last_cmd = None, None, ""

    def num():
        nonlocal i
        if i >= len(tokens) or re.fullmatch(r"[A-Za-z]", tokens[i]):
            raise SvgError(f"path data ends early or is malformed near token {i}: {d[:60]!r}")
        i += 1
        return float(tokens[i - 1])

    while i < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
        elif cmd is None:
            raise SvgError(f"path data must start with a command: {d[:60]!r}")
        elif cmd in "Zz":
            raise SvgError(f"path data has numbers after a close: {d[:60]!r}")
        rel = cmd.islower()
        ox, oy = cur if rel else (0.0, 0.0)
        c = cmd.upper()
        if c == "Z":
            if ring:
                rings.append(ring)
            ring, cur = [], start
        elif c == "M":
            if ring:
                rings.append(ring)
            cur = start = (ox + num(), oy + num())
            ring = [cur]
            cmd = "l" if rel else "L"  # further pairs are lines
        elif c == "L":
            cur = (ox + num(), oy + num())
            ring.append(cur)
        elif c == "H":
            cur = (ox + num(), cur[1])
            ring.append(cur)
        elif c == "V":
            cur = (cur[0], oy + num())
            ring.append(cur)
        elif c in "CS":
            if c == "C":
                p1 = (ox + num(), oy + num())
            else:  # the first control point mirrors the last curve's second
                p1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in "CS" and last_ctrl else cur
            p2 = (ox + num(), oy + num())
            end = (ox + num(), oy + num())
            ring += _cubic(cur, p1, p2, end)
            cur, last_ctrl = end, p2
        elif c in "QT":
            if c == "Q":
                p1 = (ox + num(), oy + num())
            else:
                p1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_cmd in "QT" and last_ctrl else cur
            end = (ox + num(), oy + num())
            ring += _quad(cur, p1, end)
            cur, last_ctrl = end, p1
        elif c == "A":
            rx, ry, rot, large, sweep = num(), num(), num(), num(), num()
            end = (ox + num(), oy + num())
            ring += _arc(cur, rx, ry, rot, bool(large), bool(sweep), end)
            cur = end
        last_cmd = c
    if ring:
        rings.append(ring)
    return rings


def _ellipse(cx, cy, rx, ry):
    n = CURVE_STEPS * 4
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _rect(x, y, w, h, rx, ry):
    if rx <= 0 or ry <= 0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    d = (f"M{x + rx},{y} H{x + w - rx} A{rx},{ry} 0 0 1 {x + w},{y + ry} V{y + h - ry} A{rx},{ry} 0 0 1 {x + w - rx},{y + h} "
         f"H{x + rx} A{rx},{ry} 0 0 1 {x},{y + h - ry} V{y + ry} A{rx},{ry} 0 0 1 {x + rx},{y} Z")
    return path_rings(d)[0]


def area(ring) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:] + ring[:1])) / 2


def _tidy(ring, eps=1e-7):
    out = []
    for p in ring:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _style(el) -> dict:
    return dict((k.strip(), v.strip()) for k, v in (kv.split(":", 1) for kv in (el.get("style") or "").split(";") if ":" in kv))


def _f(el, name, default=0.0):
    v = el.get(name)
    if v is None:
        return default
    m = re.match(r"\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", v)
    if not m:
        raise SvgError(f'<{el.tag.split("}")[-1]}> {name}="{v}" is not a number')
    return float(m.group(0))


def read(path) -> tuple[list[Part], tuple[float, float, float, float], list[str]]:
    """(parts in document order, the view box as x, y, width, height, notes on what was left out)."""
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as e:
        raise SvgError(f"{path.name} is not a readable SVG: {e}")
    if root.tag.split("}")[-1] != "svg":
        raise SvgError(f"{path.name} is not an SVG (its root is <{root.tag.split('}')[-1]}>)")
    parts, notes, seen = [], [], set()

    def walk(el, m, fill):
        tag = el.tag.split("}")[-1]
        if tag in SKIPPED:
            return
        if tag in UNREAD:
            notes.append(UNREAD[tag])
            return
        st = _style(el)
        f = st.get("fill", el.get("fill"))
        fill = fill if f is None else f
        m = mul(m, transform(el.get("transform")))
        if tag in ("svg", "g", "a", "switch"):
            for child in el:
                walk(child, m, fill)
            return
        if tag == "path":
            rings = path_rings(el.get("d", ""))
        elif tag in ("polygon", "polyline"):
            v = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", el.get("points", ""))]
            rings = [list(zip(v[0::2], v[1::2]))]
        elif tag == "rect":
            rx, ry = el.get("rx"), el.get("ry")
            rx, ry = (_f(el, "rx") if rx else _f(el, "ry") if ry else 0), (_f(el, "ry") if ry else _f(el, "rx") if rx else 0)
            rings = [_rect(_f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height"), rx, ry)]
        elif tag == "circle":
            rings = [_ellipse(_f(el, "cx"), _f(el, "cy"), _f(el, "r"), _f(el, "r"))]
        elif tag == "ellipse":
            rings = [_ellipse(_f(el, "cx"), _f(el, "cy"), _f(el, "rx"), _f(el, "ry"))]
        else:
            if tag == "line":
                notes.append("a <line> has no fill: strokes are not read, outline them first")
            return
        c = colour("#000000" if fill is None else fill)
        stroke = st.get("stroke", el.get("stroke"))
        if c is None:
            if stroke and stroke != "none":
                notes.append(f'<{tag}{" id=" + el.get("id") if el.get("id") else ""}> has a stroke and no fill: '
                             f'strokes are not read, outline them first')
            return
        if c == "":
            notes.append(f"fill {fill!r} is not a flat colour (a gradient?): the part is grey until the spec colours it")
            c = "#808080"
        rings = [_tidy([apply(m, p) for p in r]) for r in rings]
        rings = [r for r in rings if len(r) >= 3 and abs(area(r)) > 1e-9]
        if not rings:
            return
        pid = el.get("id") or f"p{len(parts) + 1}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            pid = re.sub(r"[^A-Za-z0-9_-]+", "_", pid).strip("_") or f"p{len(parts) + 1}"
        base, k = pid, 2
        while pid in seen:
            pid, k = f"{base}_{k}", k + 1
        seen.add(pid)
        parts.append(Part(pid, c, rings))

    walk(root, IDENTITY, None)
    if not parts:
        raise SvgError(f"{path.name} has no filled shapes to build from" + (": " + "; ".join(sorted(set(notes))) if notes else ""))
    vb = [float(x) for x in re.split(r"[\s,]+", (root.get("viewBox") or "").strip()) if x]
    if len(vb) != 4:
        x0, y0 = min(p.box()[0] for p in parts), min(p.box()[1] for p in parts)
        x1, y1 = max(p.box()[2] for p in parts), max(p.box()[3] for p in parts)
        vb = [x0, y0, x1 - x0, y1 - y0]
    return parts, tuple(vb), sorted(set(notes))


def bounds(parts) -> tuple[float, float, float, float]:
    boxes = [p.box() for p in parts]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
