"""A fitted face that keeps the model's own paint (the measuring is plain numpy; the rest runs inside Blender).

A generated model comes with a face painted on it: eyes of some size with a lid over them, brows of some shape,
lips of some colour. The rig's own face (face.py, character.py) erases the eyes and brows and puts its own
there: white ovals, a round iris, bar brows. On a cartoon model that is close to what was painted. On a model
painted nearer to life it is not: the rig's eyes are several times the size of the painted ones and wide open,
and the face stares.

So the fit reads the painted face first (`measure`): the eye's opening, the iris and how much of it the lid
covers, the lips. `check` compares the rig's face with those numbers. And with
`"paint": "keep"` in the fit's face nothing is erased: the painted face stays as it is, and the parts that have
to move are cut out of the paint itself and laid on the face as thin patches:

  * each eye: the white with the iris painted out, the painted iris as a disc that slides inside the opening
    (a glance), and a lid in the skin's colour that comes down from the painted lid line (a blink, a wink);
  * the brows: left as painted. The forehead's skin under each lifts and tilts (shape keys in the head), and
    the brow rides on it. Cutting a soft painted brow out of its skin by colour did not hold: it took the
    lid's crease with it on one face and half the brow on another;
  * the mouth: the painted lips, moved by the head's own smile and frown shapes. The rig's mouth (dark, teeth,
    lips in the painted colour) waits behind the face and comes forward only when the mouth opens.

At rest such a face is the model's own, texel for texel.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

STEP = 0.0002  # the size of one sample on the face (m): finer than a generated texture's own texels


# ------------------------------------------------------------------ small image tools (numpy only)
def _luma(p):
    return p[..., :3].mean(axis=-1)


def _sat(p):
    mx = p[..., :3].max(axis=-1)
    return (mx - p[..., :3].min(axis=-1)) / np.maximum(mx, 1e-6)


def _shift(m, dz, dx, fill=False):
    out = np.full_like(m, fill)
    h, w = m.shape[:2]
    out[max(0, dz):h + min(0, dz), max(0, dx):w + min(0, dx)] = m[max(0, -dz):h + min(0, -dz), max(0, -dx):w + min(0, -dx)]
    return out


def _dilate(m, times=1):
    for _ in range(times):
        m = m | _shift(m, 1, 0) | _shift(m, -1, 0) | _shift(m, 0, 1) | _shift(m, 0, -1)
    return m


def _erode(m, times=1):
    for _ in range(times):
        m = m & _shift(m, 1, 0) & _shift(m, -1, 0) & _shift(m, 0, 1) & _shift(m, 0, -1)
    return m


def _grow(seed, mask):
    """Everything in `mask` joined to `seed`."""
    seed = seed & mask
    while True:
        more = _dilate(seed) & mask
        if more.sum() == seed.sum():
            return seed
        seed = more


def _parts(mask, least=12):
    """The joined parts of a mask, largest first."""
    out, left = [], mask.copy()
    while left.any():
        z, x = np.argwhere(left)[0]
        seed = np.zeros_like(mask)
        seed[z, x] = True
        part = _grow(seed, left)
        left &= ~part
        if part.sum() >= least:
            out.append(part)
    return sorted(out, key=lambda p: -int(p.sum()))


def _depth(mask):
    """How many steps each pixel of a mask is from its edge."""
    d = np.zeros(mask.shape, dtype=np.int32)
    m = mask.copy()
    while m.any():
        d += m
        m = _erode(m)
    return d


def _blur(a, times=1):
    a = a.astype(np.float32)
    for _ in range(times):
        p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2), mode="edge")
        a = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 4 * p[1:-1, 1:-1]) / 8
    return a


def _skin(P, where=None):
    """The skin's colour in a patch: the middle of its brighter half, from the border unless told where."""
    h, w = P.shape[:2]
    if where is None:
        b = max(2, int(0.1 * min(h, w)))
        where = np.zeros((h, w), dtype=bool)
        where[:b] = where[-b:] = where[:, :b] = where[:, -b:] = True
    L = _luma(P)
    pick = where & (L >= np.median(L[where]))
    return np.median(P[pick][:, :3], axis=0)


def _fit_line(xs, vs, at, degree=2):
    """A smooth curve through (xs, vs), read at `at`."""
    if len(xs) < degree + 2:
        return np.full(len(at), float(np.median(vs)) if len(vs) else 0.0)
    mid = xs.mean()
    return np.polyval(np.polyfit(xs - mid, vs, degree), at - mid)


# ------------------------------------------------------------------ reading the paint
def _iris_circle(P, L, S, skin, dark, white, cz0, cx0, r_in):
    """The iris as a circle through the points where it meets the white, either side: (cx, cz, r) in samples, or
    None when too little of its edge shows against the white to say. What is neither white nor skin and joined
    to the pupil along a row is iris (or lash, or the shade under the lid: those ends are not beside white, or
    fall off the circle and are dropped)."""
    h, w = L.shape
    skin_l, skin_s = float(skin.mean()), float(_sat(skin))
    pale = ((S < 0.82 * skin_s) & (L > 0.45 * skin_l) & ~dark) | white
    warm = (L > 0.55 * skin_l) & ((P[..., 0] - P[..., 2]) > 0.45 * (skin[0] - skin[2]))
    body = ~pale & ~warm
    pts = []
    for z in range(max(0, int(cz0 - 3 * r_in)), min(h, int(cz0 + 3 * r_in) + 1)):
        if not body[z, cx0]:
            continue
        for way in (-1, 1):
            x = cx0
            while 0 <= x + way < w and body[z, x + way]:
                x += way
            beyond = pale[z, x + way:x + 4 * way:way] if way > 0 else pale[z, max(0, x - 3):x]
            if 0 < x < w - 1 and beyond.size and beyond.all():
                pts.append((x + 0.5 * way, z))
    for _ in range(3):
        if len(pts) < 10:
            return None
        a = np.array(pts, dtype=float)
        m = a.mean(axis=0)
        u, v = a[:, 0] - m[0], a[:, 1] - m[1]
        A = np.column_stack([2 * u, 2 * v, np.ones(len(a))])
        (uc, vc, c), *_ = np.linalg.lstsq(A, u * u + v * v, rcond=None)
        r = math.sqrt(max(c + uc * uc + vc * vc, 1e-9))
        off = np.abs(np.hypot(u - uc, v - vc) - r)
        keep = off <= max(1.0, 2.0 * np.median(off))
        if keep.all():
            break
        pts = [p for p, k in zip(pts, keep) if k]
    sides = (a[:, 0] < m[0] + uc).sum(), (a[:, 0] > m[0] + uc).sum()
    if min(sides) < 4 or not 0.9 * r_in <= r <= 4 * r_in or abs(m[0] + uc - cx0) > r or abs(m[1] + vc - cz0) > r:
        return None
    return float(m[0] + uc), float(m[1] + vc), float(r)


def measure_eye(P, step=STEP):
    """A painted eye, from a patch of the face round it (rows go up, columns to the character's left).
    Returns None when there is no white beside something dark (no painted eye to keep), else a dict:
    numbers in metres from the patch's corner, and the arrays the plates are made from."""
    h, w = P.shape[:2]
    L, S = _luma(P), _sat(P)
    skin = _skin(P)
    skin_l, skin_s = float(skin.mean()), float(_sat(skin))
    white = (S < 0.65 * skin_s) & (L > 0.62 * skin_l)
    dark = L < 0.55 * skin_l
    inner = np.zeros((h, w), dtype=bool)
    inner[int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = True
    near_white = _dilate(white, 3)
    best = None
    for part in _parts(dark, least=30)[:8]:  # the iris: the thickest dark shape with white beside it
        if not (part & near_white).any():
            continue
        d = _depth(part) * inner
        z, x = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[z, x] and (best is None or d[z, x] > best[0]):
            best = (int(d[z, x]), int(z), int(x), part)
    if best is None or best[0] < 0.0012 / step:
        return None
    _, cz0, cx0, part = best
    r_in = best[0]

    def rim(line, at, way):
        """Where the iris ends along a line of brightness, going `way` from `at`: the first strong step up
        between most of the dark shape's own radius and three times it (the first peak that is over 0.6 of the
        sharpest: a black liner's outer edge against skin is a sharper step than a brown iris against the
        white, and lies further out)."""
        steps = []
        for k in range(int(0.6 * r_in), int(3 * r_in)):
            i = at + way * k
            if i - 2 < 0 or i + 2 >= len(line):
                break
            steps.append((float(line[i + 2 * way] + line[i + way] - line[i - way] - line[i - 2 * way]), k))
        if not steps:
            return r_in
        top = max(steps)[0]
        for j, (v, k) in enumerate(steps):
            if v >= 0.6 * top and (j + 1 == len(steps) or steps[j + 1][0] <= v):
                return k
        return max(steps)[1]

    soft = _blur(L, 2)
    rows = [soft[z] for z in range(max(0, cz0 - 3), min(h, cz0 + 4))]
    left = float(np.median([rim(row, cx0, -1) for row in rows]))
    right = float(np.median([rim(row, cx0, 1) for row in rows]))
    r = (left + right) / 2
    cx = cx0 + (right - left) / 2
    cols_ = [soft[:, x] for x in range(max(0, int(cx) - 3), min(w, int(cx) + 4))]
    foot = float(np.median([rim(col, cz0, -1) for col in cols_]))
    cz = max(float(cz0), cz0 - foot + r)  # (a lid cuts the top of an iris, not its foot: the centre is a radius up)
    # a coloured iris has rings of its own (a dark rim, a paler band, a dark ring round the pupil), and the steps
    # between them are as sharp as its rim: where its sides can be seen against the white, the circle through
    # those points is the iris
    round_ = _iris_circle(P, L, S, skin, dark, white, cz0, cx0, r_in)
    if round_:
        cx, cz, r = round_
    zz, xx = np.mgrid[0:h, 0:w]
    rad = np.hypot(xx - cx, zz - cz)
    disc = rad <= r
    # the opening: the white joined to the iris, column by column, and a smooth lid line over the columns where
    # the iris hides it
    # (the white of a painted eye is shaded pink toward its corners: from the clean white, on into what is
    # still paler or greyer than skin and not a line)
    loose = ((S < 0.82 * skin_s) & (L > 0.45 * skin_l) & ~dark) | white
    seen = _grow(_grow(_dilate(disc | part, 3) & white, white), loose) & (rad > 0.9 * r)
    if seen.sum() < 20:
        return None
    cols = np.where(seen.any(axis=0))[0]
    x0, x1 = int(min(cols.min(), cx - r)), int(max(cols.max(), cx + r))
    xs = np.arange(x0, x1 + 1)
    top = np.full(len(xs), np.nan)
    bot = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        zs = np.where(seen[:, x])[0]
        if not len(zs):
            continue
        reach = math.sqrt(max(0.0, r * r - (x - cx) ** 2)) if abs(x - cx) < r else -1.0
        if reach < 0 or zs.max() > cz + reach - 2:
            top[i] = zs.max()
        if reach < 0 or zs.min() < cz - reach + 2:
            bot[i] = zs.min()
    for line, sign in ((top, 1), (bot, -1)):
        ok = ~np.isnan(line)
        if ok.sum() < 4:
            return None
        gone = ~ok
        if gone.any() and sign > 0:
            # over the iris the lid line runs on in a curve from the white either side (the iris goes straight
            # into the lashes there, with no edge to read), and no higher than the iris itself
            line[gone] = _fit_line(xs[ok].astype(float), line[ok], xs[gone].astype(float))
            edge = cz + np.sqrt(np.maximum(0.0, r * r - (xs[gone] - cx) ** 2))
            line[gone] = np.minimum(line[gone], edge + 1)
        elif gone.any():
            # under the iris the lower lid runs straight on between the white either side (a curve run on from
            # the corners dips below the lid, and the white then shows through the lid when the iris moves)
            line[gone] = np.interp(xs[gone], xs[ok], line[ok])
            edge = cz - np.sqrt(np.maximum(0.0, r * r - (xs[gone] - cx) ** 2))
            line[gone] = np.maximum(line[gone], edge - 1)
        m = int(0.003 / step)  # (a lash or a nick is narrower than this: the middle value steps over it)
        padded = np.pad(line, m, mode="edge")
        line[:] = [np.median(padded[i:i + 2 * m + 1]) for i in range(len(line))]
        k = np.ones(5) / 5
        line[:] = np.convolve(np.pad(line, 2, mode="edge"), k, mode="valid")
    top = np.maximum(top, bot + 1)
    opening = np.zeros((h, w), dtype=bool)
    hh = np.zeros((h, w), dtype=np.float32)  # 0 on the lid line, 1 on the lower edge, below 0 above the eye
    tops = np.interp(np.arange(w), xs, top)
    bots = np.interp(np.arange(w), xs, bot)
    span = float(np.max(top - bot))
    # (each column by its own height: a shutting lid's edge goes from the upper lid's curve to the lower lid's,
    # and its lash line then lies along the whole of the lower lid. Measured against the tallest column alone,
    # the edge kept the upper lid's curve all the way down and met the lower lid in the middle only)
    tall = np.full(w, span)
    tall[x0:x1 + 1] = np.maximum(tops[x0:x1 + 1] - bots[x0:x1 + 1], 0.25 * span)
    hh[:] = (tops[None, :] - zz) / tall[None, :]
    opening[:, x0:x1 + 1] = (zz[:, x0:x1 + 1] >= bots[None, x0:x1 + 1]) & (zz[:, x0:x1 + 1] <= tops[None, x0:x1 + 1])
    # the lash line: what is dark just above the opening
    lash_rows = []
    for x in xs[len(xs) // 5: -len(xs) // 5 or None]:
        z = int(round(tops[x])) + 1
        n = 0
        while z + n < h and dark[z + n, x]:
            n += 1
        lash_rows.append(n)
    lash = float(np.median(lash_rows)) if lash_rows else 0.0
    band = np.zeros((h, w), dtype=bool)
    reach = int(round(min(1.5 * max(lash, 0.0006 / step) + 4, 0.0045 / step)))  # (all of the lash line, and its soft edge)
    band[:, x0:x1 + 1] = (zz[:, x0:x1 + 1] > tops[None, x0:x1 + 1]) & (zz[:, x0:x1 + 1] <= tops[None, x0:x1 + 1] + reach)
    over = band & dark
    # (and a rim all round: the painted edge of the white is soft, and a shut lid has to cover the last of it)
    band |= _dilate(opening, int(0.0012 / step)) & ~opening
    lash_colour = np.median(P[over][:, :3], axis=0) if over.sum() > 10 else skin * 0.45
    above = np.zeros((h, w), dtype=bool)
    above[:, x0:x1 + 1] = (zz[:, x0:x1 + 1] > tops[None, x0:x1 + 1] + reach + 0.0008 / step) & \
                          (zz[:, x0:x1 + 1] <= tops[None, x0:x1 + 1] + reach + 0.003 / step)
    lid_colour = _skin(P, above) if above.sum() > 20 else skin
    # the iris: its colour (between the pupil and its rim), the pupil, how much of it shows
    shown = disc & opening
    ring = shown & (rad > 0.5 * r) & (rad < 0.9 * r) & ~white
    iris_colour = np.median(P[ring][:, :3], axis=0) if ring.sum() > 10 else np.median(P[shown][:, :3], axis=0)
    core = shown & (L < 0.5 * max(float(iris_colour.mean()), 0.1))
    pupil = math.sqrt(core.sum() / math.pi) if float(iris_colour.mean()) > 0.12 else r
    mid = int(np.clip(round(cx), 0, w - 1))
    covered = float(np.clip((cz + r - tops[mid]) / (2 * r), 0, 1))
    under = float(np.clip((bots[mid] - (cz - r)) / (2 * r), 0, 1))
    # the white with the iris painted out, row by row in the white's own shade
    fill = P[..., :3].copy()
    whites = opening & white & (rad > 1.12 * r)
    overall = np.median(P[whites][:, :3], axis=0) if whites.sum() > 5 else np.array([0.9, 0.9, 0.9])
    rows = np.where(whites.sum(axis=1) >= 3)[0]
    shade = np.array([np.median(P[z][whites[z]][:, :3], axis=0) for z in rows]) if len(rows) else None
    out = _dilate(disc, max(2, int(0.12 * r))) & opening
    for z in np.where(out.any(axis=1))[0]:
        c = overall if shade is None else shade[int(np.argmin(np.abs(rows - z)))]
        fill[z, out[z]] = c
    soft = _blur(fill, 3)
    edge = _dilate(out, 2)
    fill[edge] = soft[edge]
    # the iris as a disc of its own: what shows of it as painted, the rest by its rings
    half = int(math.ceil(1.15 * r)) + 2
    size = 2 * half + 1
    sprite = np.zeros((size, size, 3), dtype=np.float32)
    sz, sx = np.mgrid[0:size, 0:size]
    srad = np.hypot(sx - half, sz - half)
    rings = []
    for k in range(int(math.ceil(half * 1.5))):
        pick = shown & (np.abs(rad - k) <= 0.75)
        rings.append(np.median(P[pick][:, :3], axis=0) if pick.sum() >= 3 else (rings[-1] if rings else iris_colour))
    rings = np.array(rings)
    sprite[:] = rings[np.clip(np.round(srad).astype(int), 0, len(rings) - 1)]
    pz, px = np.round(sz - half + cz).astype(int), np.round(sx - half + cx).astype(int)
    inside = (pz >= 0) & (pz < h) & (px >= 0) & (px < w)
    real = np.zeros((size, size), dtype=bool)
    real[inside] = _erode(shown, 3)[pz[inside], px[inside]]  # (not the last of it: the lid shades the iris's edge)
    sprite[real] = P[pz[real], px[real], :3]
    return {
        "open": [(x1 - x0 + 1) * step, span * step], "corner": [x0 * step, x1 * step],
        "iris": [cx * step, cz * step], "iris_r": r * step, "pupil_r": pupil * step,
        "iris_colour": [float(v) for v in iris_colour], "covered": covered, "under": under,
        "lash_m": lash * step, "lash_colour": [float(v) for v in lash_colour],
        "lid_colour": [float(v) for v in lid_colour], "skin": [float(v) for v in skin],
        "sprite_m": size * step,
        "plates": {"fill": fill, "iris": sprite,
                   "data": np.dstack([_blur(opening, 2), np.clip((hh + 0.5) / 1.5, 0, 1), _blur(band, 1)])},
    }


def measure_lips(P, step=STEP):
    """Painted lips, from a patch round the mouth. Returns None when the mouth is a line with no lips to it."""
    h, w = P.shape[:2]
    L, S = _luma(P), _sat(P)
    skin = _skin(P)
    skin_l, skin_s = float(skin.mean()), float(_sat(skin))
    red = P[..., 0] - (P[..., 1] + P[..., 2]) / 2
    lipish = ((S > 1.2 * skin_s + 0.03) | (L < 0.82 * skin_l)) & (red > 0.8 * (skin[0] - (skin[1] + skin[2]) / 2))
    middle = np.zeros((h, w), dtype=bool)
    middle[int(0.3 * h):int(0.7 * h), int(0.35 * w):int(0.65 * w)] = True
    part = next((p for p in _parts(lipish, least=int((0.002 / step) ** 2)) if (p & middle).any()), None)
    if part is None:
        return None
    cols = np.where(part.any(axis=0))[0]
    line, upper, lower = [], [], []
    for x in cols:
        zs = np.where(part[:, x])[0]
        z = zs[int(np.argmin(L[zs, x]))]
        line.append(z)
        upper.append(zs.max() - z)
        lower.append(z - zs.min())
    line = np.array(line, dtype=float)
    mid = slice(len(cols) // 3, 2 * len(cols) // 3 + 1)
    # where the lips meet, as a curve across the mouth (from the middle of the patch, the fit's mouth mark): the
    # darkest row of each column, read from the paint a little blurred and with a lone column's jump stepped over
    soft = _blur(L, 2)
    met = np.array([zs[int(np.argmin(soft[zs, x]))] for x in cols for zs in [np.where(part[:, x])[0]]], dtype=float)
    m = max(2, len(cols) // 12)
    padded = np.pad(met, m, mode="edge")
    met = np.array([np.median(padded[i:i + 2 * m + 1]) for i in range(len(met))])
    curve = np.polyfit((cols - w / 2) * step, (met - h / 2) * step, 2)
    colour = np.median(P[part & (L > np.median(L[part]))][:, :3], axis=0)
    return {"width": (cols.max() - cols.min() + 1) * step, "upper": float(np.median(np.array(upper)[mid])) * step,
            "lower": float(np.median(np.array(lower)[mid])) * step, "colour": [float(v) for v in colour],
            "corners_up": float((line[:3].mean() + line[-3:].mean()) / 2 - np.median(line[mid])) * step,
            "line": [float(v) for v in curve], "centre": float((cols.max() + cols.min() + 1) / 2 - w / 2) * step}


# ------------------------------------------------------------------ what the check says
EYE_TALL = 1.5   # the rig's eye may stand this many times as tall as the painted eye's opening
LID_BARE = 0.12  # a painted lid that covers more of the iris than this is part of the look: an eye without it stares


def stare(seen: dict, eye_size, kept: bool) -> dict:
    """The rig's eyes against the painted ones. `seen` is what the fit read off the paint, `eye_size` the size
    the rig makes its own eyes, `kept` whether the painted face was kept. Returns numbers, and `problems` in
    words when the rig's eyes would stare where the painted ones do not."""
    eyes = [e for e in seen.get("eyes", []) if e]
    if not eyes:
        return {}
    tall = float(np.mean([e["open"][1] for e in eyes]))
    wide = float(np.mean([e["open"][0] for e in eyes]))
    covered = float(np.mean([e["covered"] for e in eyes]))
    out = {"painted_eye_m": [round(wide, 4), round(tall, 4)], "painted_lid_over_iris": round(covered, 2), "paint_kept": kept}
    if kept:
        return out
    ratio = eye_size[1] / tall
    out["eye_tall"] = round(ratio, 2)
    problems = []
    if ratio > EYE_TALL:
        problems.append(f"the rig's eyes are {ratio:.1f} times as tall as the eyes painted on the model "
                        f"({eye_size[1] * 1000:.0f} mm against {tall * 1000:.0f} mm), so the face stares")
    if covered > LID_BARE:
        problems.append(f"the painted lid rests over {covered:.0%} of the iris and the rig's eye shows all of it")
    if len(problems) == 2 or ratio > EYE_TALL:
        problems[-1] += ': keep the painted face with "paint": "keep" in the fit\'s face'
        out["problems"] = problems
    return out


def words(seen: dict) -> list:
    """What was read off the paint, a line each, for the fit to print."""
    out = []
    for side, e in zip(("right", "left"), seen.get("eyes", [])):
        if e:
            c = "#%02x%02x%02x" % tuple(int(round(255 * min(1, max(0, v)))) for v in e["iris_colour"])
            out.append(f"{side} eye opens {e['open'][0] * 1000:.1f} x {e['open'][1] * 1000:.1f} mm, iris "
                       f"{2000 * e['iris_r']:.1f} mm across ({c}), pupil {2000 * e['pupil_r']:.1f} mm, the lid over "
                       f"{e['covered']:.0%} of the iris, lash line {e['lash_m'] * 1000:.1f} mm")
        else:
            out.append(f"{side} eye: no painted eye found (no white beside an iris)")
    lips = seen.get("lips")
    if lips:
        c = "#%02x%02x%02x" % tuple(int(round(255 * min(1, max(0, v)))) for v in lips["colour"])
        out.append(f"lips {lips['width'] * 1000:.0f} mm wide, upper {lips['upper'] * 1000:.1f} mm, lower "
                   f"{lips['lower'] * 1000:.1f} mm ({c}), corners {lips['corners_up'] * 1000:+.1f} mm")
    else:
        out.append("lips: none painted (a line, or under a moustache)")
    return out


# ------------------------------------------------------------------ inside Blender: sampling the model's front
def sample_front(ob, x0, x1, z0, z1, step=STEP):
    """The model as seen from the front over a box: colour (its base colour texture, as stored) and how far
    back each sample is. Rows go up from z0, columns go from x0 to the character's left."""
    from mathutils import Vector
    me = ob.data
    img = next(n.image for n in me.materials[0].node_tree.nodes
               if n.type == "TEX_IMAGE" and n.outputs[0].links and n.outputs[0].links[0].to_socket.name == "Base Color")
    tw, th = img.size
    px = np.empty(tw * th * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(th, tw, 4)
    nx, nz = int(round((x1 - x0) / step)), int(round((z1 - z0) / step))
    P = np.zeros((nz, nx, 3), dtype=np.float32)
    Y = np.full((nz, nx), np.nan, dtype=np.float32)
    uv = me.uv_layers.active.data
    polys = me.polygons
    cache = {}
    way = Vector((0, 1, 0))
    for iz in range(nz):
        z = z0 + (iz + 0.5) * step
        for ix in range(nx):
            hit, at, _, index = ob.ray_cast(Vector((x0 + (ix + 0.5) * step, -10, z)), way)
            if not hit:
                continue
            tri = cache.get(index)
            if tri is None:
                loops = list(polys[index].loop_indices)[:3]
                a, b, c = (me.vertices[me.loops[i].vertex_index].co for i in loops)
                tri = cache[index] = (a.copy(), b - a, c - a, [np.array(uv[i].uv) for i in loops])
            a, e1, e2, uvs = tri
            p = at - a
            d11, d12, d22, p1, p2 = e1.dot(e1), e1.dot(e2), e2.dot(e2), p.dot(e1), p.dot(e2)
            det = d11 * d22 - d12 * d12 or 1e-20
            v, wgt = (d22 * p1 - d12 * p2) / det, (d11 * p2 - d12 * p1) / det
            t = uvs[0] * (1 - v - wgt) + uvs[1] * v + uvs[2] * wgt
            fx, fz = (t[0] % 1) * tw - 0.5, (t[1] % 1) * th - 0.5
            ix0, iz0 = int(math.floor(fx)), int(math.floor(fz))
            ax, az = fx - ix0, fz - iz0
            xa, xb, za, zb = ix0 % tw, (ix0 + 1) % tw, iz0 % th, (iz0 + 1) % th
            P[iz, ix] = (px[za, xa, :3] * (1 - ax) + px[za, xb, :3] * ax) * (1 - az) + \
                        (px[zb, xa, :3] * (1 - ax) + px[zb, xb, :3] * ax) * az
            Y[iz, ix] = at.y
    return P, Y


def measure(ob, face: dict, centre_x: float = 0.0, step=STEP):
    """Read the painted face off a model as loaded (before anything is erased). `face` is the fit's face.
    Returns `seen`: the eyes (character's right, then left; None where no painted eye was found), each with
    its `box` (x0, x1, z0, z1 on the model) and its arrays under `plates`, and the lips."""
    ew, eh = face["eye_size"]
    seen = {"eyes": [], "lips": None}
    for x, z in face["eyes"]:
        box = (x - centre_x - 0.8 * ew, x - centre_x + 0.8 * ew, z - 0.75 * eh, z + 0.75 * eh)
        P, _ = sample_front(ob, *box, step)
        e = measure_eye(P, step)
        if e:
            e["box"] = list(box)
        seen["eyes"].append(e)
    if not face.get("moustache"):
        mw = face["mouth_width"]
        mx, mz = face["mouth"]
        P, _ = sample_front(ob, mx - centre_x - 0.75 * mw, mx - centre_x + 0.75 * mw, mz - 0.3 * mw, mz + 0.3 * mw, step)
        seen["lips"] = measure_lips(P, step)
    return seen


# ------------------------------------------------------------------ inside Blender: the fit's side
def _linear(c):
    return tuple(float(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in c)


def store(seen: dict, placed, scale: float, gamma=None, keep=False) -> dict:
    """What goes into the fitted file. `placed` turns a point on the model as loaded into one on the model as
    saved. Returns {"seen": the numbers, "paint": where each eye's patch goes and which images it draws}
    (`paint` only when the painted face is kept: its plates are packed into the file as images)."""
    import bpy
    tone = (lambda a: np.clip(a, 0, 1) ** float(gamma)) if gamma else (lambda a: a)
    out = {"seen": {"eyes": [], "lips": None}}
    if seen.get("lips"):
        lips = seen["lips"]
        out["seen"]["lips"] = {**{k: lips[k] * scale for k in ("width", "upper", "lower", "corners_up", "centre")},
                               # (z for an x, both from the mouth mark: the square term goes by 1 / scale)
                               "line": [lips["line"][0] / scale, lips["line"][1], lips["line"][2] * scale],
                               "colour": [float(v) for v in tone(np.array(lips["colour"]))]}
    eyes = []
    for i, e in enumerate(seen["eyes"]):
        if not e:
            out["seen"]["eyes"].append(None)
            continue
        out["seen"]["eyes"].append({"open": [v * scale for v in e["open"]], "iris_r": e["iris_r"] * scale,
                                    "pupil_r": e["pupil_r"] * scale, "covered": e["covered"],
                                    "iris_colour": [float(v) for v in tone(np.array(e["iris_colour"]))]})
        if not keep:
            continue
        x0, x1, z0, z1 = e["box"]
        a, b = placed((x0, 0, z0)), placed((x1, 0, z1))
        names = {}
        for kind, arr in e["plates"].items():
            data = kind == "data"
            arr = np.asarray(arr, dtype=np.float32) if data else tone(np.asarray(arr, dtype=np.float32))
            h, w = arr.shape[:2]
            # (an image made in memory is not kept by pack(): it comes back blank from the saved file. It is
            # written out as a PNG, loaded as a file and packed from that)
            made = bpy.data.images.new("face_plate", w, h, alpha=False)
            made.colorspace_settings.name = "Non-Color" if data else "sRGB"
            made.pixels.foreach_set(np.dstack([arr[..., :3], np.ones((h, w), dtype=np.float32)]).ravel())
            made.filepath_raw = str(Path(tempfile.gettempdir()) / f"face_eye{i}_{kind}.png")
            made.file_format = "PNG"
            made.save()
            img = bpy.data.images.load(made.filepath_raw)
            bpy.data.images.remove(made)
            img.name = f"face_eye{i}_{kind}"
            img.colorspace_settings.name = "Non-Color" if data else "sRGB"
            img.use_fake_user = True
            img.pack()
            names[kind] = img.name
        at = placed((x0 + e["iris"][0], 0, z0 + e["iris"][1]))
        eyes.append({"box": [a[0], b[0], a[2], b[2]], "iris": [at[0], at[2]], "iris_r": e["iris_r"] * scale,
                     "sprite_m": e["sprite_m"] * scale, "open": [v * scale for v in e["open"]],
                     "lash": max(e["lash_m"], 0.0006) * scale,
                     "lid_colour": [float(v) for v in tone(np.array(e["lid_colour"]))],
                     "lash_colour": [float(v) for v in tone(np.array(e["lash_colour"]))], "images": names})
    if keep:
        out["paint"] = {"eyes": eyes}
    return out


# ------------------------------------------------------------------ inside Blender: the rig's side
def _patch(name, body, box, off, to_parent, parent, material, cell=0.001):
    """A thin patch lying `off` in front of the body's surface over a box of its front, shaded with the body's
    own normals so it takes the light as the skin under it does. Its UVs run 0 to 1 across the box."""
    import bmesh
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    x0, x1, z0, z1 = box
    me = body.data
    near = [pl for pl in me.polygons if x0 - 0.008 < pl.center.x < x1 + 0.008 and z0 - 0.008 < pl.center.z < z1 + 0.008]
    verts = [v.co.copy() for v in me.vertices]
    tree = BVHTree.FromPolygons(verts, [tuple(pl.vertices) for pl in near])
    corner = me.corner_normals
    nx, nz = int(round((x1 - x0) / cell)) + 1, int(round((z1 - z0) / cell)) + 1
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    grid, normals = {}, {}
    for iz in range(nz):
        for ix in range(nx):
            x, z = x0 + (x1 - x0) * ix / (nx - 1), z0 + (z1 - z0) * iz / (nz - 1)
            at, _, index, _ = tree.ray_cast(Vector((x, -10, z)), Vector((0, 1, 0)))
            if at is None:
                continue
            pl = near[index]
            loops = list(pl.loop_indices)[:3]
            a, b, c = (verts[me.loops[i].vertex_index] for i in loops)
            e1, e2, p = b - a, c - a, at - a
            d11, d12, d22, p1, p2 = e1.dot(e1), e1.dot(e2), e2.dot(e2), p.dot(e1), p.dot(e2)
            det = d11 * d22 - d12 * d12 or 1e-20
            v, w = (d22 * p1 - d12 * p2) / det, (d11 * p2 - d12 * p1) / det
            n = corner[loops[0]].vector * (1 - v - w) + corner[loops[1]].vector * v + corner[loops[2]].vector * w
            vert = bm.verts.new(to_parent(Vector((at.x, at.y - off, at.z))))
            grid[ix, iz] = (vert, (ix / (nx - 1), iz / (nz - 1)))
            normals[vert] = n.normalized()
    for iz in range(nz - 1):
        for ix in range(nx - 1):
            quad = [grid.get(k) for k in ((ix, iz), (ix + 1, iz), (ix + 1, iz + 1), (ix, iz + 1))]
            if None in quad:
                continue
            f = bm.faces.new([q[0] for q in quad])
            f.smooth = True
            for loop, q in zip(f.loops, quad):
                loop[uv].uv = q[1]
    bm.verts.index_update()
    order = [normals[v] for v in bm.verts]
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    mesh.normals_split_custom_set_from_vertices([tuple(n) for n in order])
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.visible_shadow = False
    return ob


class _Nodes:
    """A few short hands for building a material's nodes."""

    def __init__(self, tree):
        self.t = tree

    def _in(self, socket, v):
        if hasattr(v, "is_output"):
            self.t.links.new(v, socket)
        else:
            socket.default_value = v

    def value(self, name, v=0.0):
        n = self.t.nodes.new("ShaderNodeValue")
        n.name = n.label = name
        n.outputs[0].default_value = v
        return n.outputs[0]

    def math(self, op, a, b=None):
        n = self.t.nodes.new("ShaderNodeMath")
        n.operation = op
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        return n.outputs[0]

    def clamp(self, a):
        n = self.t.nodes.new("ShaderNodeMath")
        n.operation, n.use_clamp = "ADD", True
        self._in(n.inputs[0], a)
        n.inputs[1].default_value = 0.0
        return n.outputs[0]

    def tex(self, image, vector, clip=False):
        n = self.t.nodes.new("ShaderNodeTexImage")
        n.image, n.extension, n.interpolation = image, "CLIP" if clip else "EXTEND", "Linear"
        self.t.links.new(vector, n.inputs["Vector"])
        return n.outputs["Color"]

    def mix(self, fac, a, b):
        n = self.t.nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        ins = {s.identifier: s for s in n.inputs}
        self._in(ins["Factor_Float"], fac)
        self._in(ins["A_Color"], a)
        self._in(ins["B_Color"], b)
        return next(s for s in n.outputs if s.identifier == "Result_Color")

    def step(self, v, lo, hi, a, b):
        """a at `lo`, b at `hi`, eased between."""
        n = self.t.nodes.new("ShaderNodeMapRange")
        n.interpolation_type = "SMOOTHSTEP"
        for socket, x in zip(n.inputs[:5], (v, lo, hi, a, b)):
            self._in(socket, x)
        return n.outputs[0]


def _drive(tree, node, ob, kind):
    """A value node takes one of a control's own transforms, as it is keyed (no script: it also runs in a file
    opened with scripts off)."""
    d = tree.driver_add(f'nodes["{node}"].outputs[0].default_value').driver
    d.type = "SUM"
    v = d.variables.new()
    v.type = "TRANSFORMS"
    t = v.targets[0]
    t.id, t.transform_type, t.transform_space = ob, kind, "LOCAL_SPACE"


def eyes(cid, body, paint: dict, path, head, to_parent, controls: dict, lid_open: float, lid_shut: float):
    """The kept eyes: one patch over each painted eye, drawing the white, the painted iris where the pupil
    control puts it and the lid as far down as the lid control is turned. `controls` is {side: (eye, pupil,
    lid)}, the same things the rig's own eyes are moved by, so every verb works as it did."""
    import bpy
    names = [n for e in paint["eyes"] for n in e["images"].values()]
    with bpy.data.libraries.load(str(path)) as (src, dst):
        missing = [n for n in names if n not in src.images]
        if missing:
            raise RuntimeError(f"{path} has no image {missing[0]}: fit the model again")
        dst.images = list(names)  # (the list given is filled in place with what was loaded)
    images = dict(zip(names, dst.images))
    out = {}
    for s, e in zip((-1, 1), paint["eyes"]):
        m = bpy.data.materials.new(f"{cid}_eye{s}_paint")
        m.use_nodes = True
        t = m.node_tree
        N = _Nodes(t)
        shader = next(n for n in t.nodes if n.type == "BSDF_PRINCIPLED")
        # (matt: the shine of a painted eye is its painted glint. A glossy patch on the bulge of the eye picks up
        # the room as a grey haze over the pupil, which stays put when the iris moves)
        shader.inputs["Roughness"].default_value = 0.7
        if "Specular IOR Level" in shader.inputs:
            shader.inputs["Specular IOR Level"].default_value = 0.1
        coords = t.nodes.new("ShaderNodeTexCoord")
        split = t.nodes.new("ShaderNodeSeparateXYZ")
        t.links.new(coords.outputs["UV"], split.inputs[0])
        u, v = split.outputs[0], split.outputs[1]
        x0, x1, z0, z1 = e["box"]
        W, H, r, size = x1 - x0, z1 - z0, e["iris_r"], e["sprite_m"]
        fill = N.tex(images[e["images"]["fill"]], coords.outputs["UV"])
        chan = t.nodes.new("ShaderNodeSeparateColor")
        t.links.new(N.tex(images[e["images"]["data"]], coords.outputs["UV"]), chan.inputs[0])
        opening, height, band = chan.outputs[0], N.math("SUBTRACT", N.math("MULTIPLY", chan.outputs[1], 1.5), 0.5), chan.outputs[2]
        # the iris: a glance of 25 degrees slides it most of its own radius, as an eye that size would turn
        reach = 0.85 * r / 0.0211
        gx = N.math("MULTIPLY", N.value("pupil_x"), reach)
        gz = N.math("MULTIPLY", N.math("ADD", N.value("pupil_z"), 0.012), 0.6 * reach)
        grown = N.math("MULTIPLY", N.math("ADD", N.math("MULTIPLY", N.value("pupil_size", 1.0), 0.35), 0.65), size)
        ru = N.math("DIVIDE", N.math("SUBTRACT", N.math("MULTIPLY", N.math("SUBTRACT", u, (e["iris"][0] - x0) / W), W), gx), grown)
        rv = N.math("DIVIDE", N.math("SUBTRACT", N.math("MULTIPLY", N.math("SUBTRACT", v, (e["iris"][1] - z0) / H), H), gz), grown)
        at = t.nodes.new("ShaderNodeCombineXYZ")
        t.links.new(N.math("ADD", ru, 0.5), at.inputs[0])
        t.links.new(N.math("ADD", rv, 0.5), at.inputs[1])
        iris = N.tex(images[e["images"]["iris"]], at.outputs[0], clip=True)
        out_r = N.math("MULTIPLY", N.math("SQRT", N.math("ADD", N.math("MULTIPLY", ru, ru), N.math("MULTIPLY", rv, rv))), size / r)
        colour = N.mix(N.step(out_r, 0.95, 1.06, 1.0, 0.0), fill, iris)
        # the lid: shut as far as the lid control is turned, and a little for an eye an expression narrows
        turned = N.clamp(N.math("DIVIDE", N.math("SUBTRACT", N.value("lid_turn", lid_open), lid_open), lid_shut - lid_open))
        narrowed = N.math("MULTIPLY", N.clamp(N.math("SUBTRACT", 1.0, N.value("eye_tall", 1.0))), 0.5)
        shut = N.math("ADD", turned, narrowed)
        edge = N.math("MULTIPLY", shut, 1.05)  # (shut, the lid covers the opening and its lash line lies along the foot)
        moving = N.step(shut, 0.0, 0.08, 0.0, 1.0)
        lash = max(0.08, 0.6 * e["lash"] / e["open"][1])
        lid_all = N.math("MULTIPLY", N.step(height, N.math("SUBTRACT", edge, 0.05), edge, 1.0, 0.0), moving)
        lid_skin = N.math("MULTIPLY", N.step(height, N.math("SUBTRACT", edge, 0.05 + lash), N.math("SUBTRACT", edge, lash), 1.0, 0.0), moving)
        colour = N.mix(lid_all, colour, (*_linear(e["lash_colour"]), 1.0))
        colour = N.mix(lid_skin, colour, (*_linear(e["lid_colour"]), 1.0))
        # (the two masks are soft where they meet: added, not the larger of the two, or the painted lash line
        # shows through the seam as a thin arc across the lid)
        alpha = N.clamp(N.math("ADD", opening, N.math("MULTIPLY", band, N.step(shut, 0.0, 0.3, 0.0, 1.0))))
        t.links.new(colour, shader.inputs["Base Color"])
        t.links.new(alpha, shader.inputs["Alpha"])
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "DITHERED"
        else:
            m.blend_method = "HASHED"
        eye, pupil, lid = controls[s]
        _drive(t, "pupil_x", pupil, "LOC_X")
        _drive(t, "pupil_z", pupil, "LOC_Z")
        _drive(t, "pupil_size", pupil, "SCALE_X")
        _drive(t, "lid_turn", lid, "ROT_X")
        _drive(t, "eye_tall", eye, "SCALE_Z")
        # (0.5 mm cells, 0.6 mm off the skin: under a hooded lid the surface turns back sharply, and a coarser
        # patch nearer the skin let the lid's edge through it as a row of dark dashes once the eye shut)
        out[s] = _patch(f"{cid}_eye{s}_paint", body, e["box"], 0.0006, to_parent, head, m, cell=0.0005)
    return out


def brow_keys(body, marks: dict, seen: dict, brow_length: float, controls: dict, rest_z: float, unit: float):
    """Painted brows move with the skin they are painted on: two shape keys in the head for each, `brow_up`
    (the forehead over the eye lifts) and `brow_tilt` (it turns about the brow's middle), driven by the brow
    control's own height and turn. The skin moves most at the brow, not at all at the eye's lid line, and
    fades out up the forehead and to either side. The tilt is 0.6 of the control's turn: skin that turns as far as
    a separate brow would folds into a crease between the brows. `unit` is the length of one of the control's units (m)."""
    me = body.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    head = np.ones(len(co), dtype=np.float32)
    group = body.vertex_groups.get("neck")
    if group is not None:
        head[:] = 0
        for v in me.vertices:
            for g in v.groups:
                if g.group == group.index:
                    head[v.index] = g.weight
    ease = lambda t: (lambda c: c * c * (3 - 2 * c))(np.clip(t, 0.0, 1.0))  # noqa: E731
    rigid = ease((head - 0.9) / 0.1)
    if not me.shape_keys:
        body.shape_key_add(name="Basis")
    tall = float(np.mean([e["open"][1] for e in seen["eyes"] if e]))
    middle = 0.5 * (marks["eyes"][0][0] + marks["eyes"][1][0])
    ed = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ed)
    ed = ed.reshape(-1, 2)
    made = np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)
    tilted = {1: np.zeros(len(co), dtype=np.float32), -1: np.zeros(len(co), dtype=np.float32)}
    for s, brow, eye in zip((-1, 1), marks["brows"], marks["eyes"]):
        x, y, z = co[:, 0] - brow[0], co[:, 1] - brow[1], co[:, 2]
        lid = eye[2] + 0.5 * tall
        f = np.exp(-(x ** 2) / (2 * (0.6 * brow_length) ** 2)) * ease((z - lid) / max(0.002, brow[2] - 0.002 - lid)) \
            * (1 - ease((z - brow[2] - 0.006) / 0.016)) * ease((0.05 - np.abs(y)) / 0.03) * rigid
        # each brow's skin is its own half of the forehead: where the two reached across the middle they added
        # up, and the skin between the brows went further than either brow (a crease down to the nose)
        f = f * ease((s * (co[:, 0] - middle) - 0.003) / 0.012)
        side = "L" if s > 0 else "R"
        for way in (1, -1):
            tilted[way] += way * R_BROW * -0.6 * x * f
        for name, dz in ((f"brow_up.{side}", unit * f), (f"brow_tilt.{side}", -0.6 * x * f)):
            kb = body.shape_key_add(name=name)
            moved = co.copy()
            moved[:, 2] += dz
            kb.data.foreach_set("co", moved.ravel())
            kb.slider_min, kb.slider_max, kb.value = -2.0, 2.0, 0.0
            d = me.shape_keys.driver_add(f'key_blocks["{name}"].value').driver
            v = d.variables.new()
            v.name, v.type = "v", "TRANSFORMS"
            t = v.targets[0]
            t.id, t.transform_space = controls[s], "LOCAL_SPACE"
            if "up" in name:
                t.transform_type = "LOC_Z"
                d.type, d.expression = "SCRIPTED", f"v - {rest_z:.6f}"
            else:
                t.transform_type = "ROT_Y"
                d.type = "SUM"
    # how hard the brows' furthest tilt, either way, squeezes the skin: the shortest an edge of the forehead gets
    # against its length at rest (1 is untouched; near 0 the skin has folded into a crease)
    squeeze = 1.0
    for dz in tilted.values():
        moved = co.copy()
        moved[:, 2] += dz
        on = (np.abs(dz[ed[:, 0]]) + np.abs(dz[ed[:, 1]]) > 1e-5) & (made > 1e-5)
        if on.any():
            ratio = np.linalg.norm(moved[ed[on, 0]] - moved[ed[on, 1]], axis=1) / made[on]
            squeeze = min(squeeze, float(np.percentile(ratio, 0.5)))
    return squeeze


BROW_SQUEEZE_LIMIT = 0.5  # painted brows that squeeze the forehead's skin shorter than this share crease it
R_BROW = 0.42  # the furthest a brow control turns in any expression (radians: 24 degrees, `worry`)
