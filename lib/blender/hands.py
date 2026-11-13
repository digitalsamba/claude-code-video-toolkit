"""Hands with fingers for a fitted model (runs inside Blender, from fit_model.py).

Image-to-3D models come with whatever hands the picture had, and a puppet drawn with fists has mittens: one lump
on one bone, which can touch a door handle but not close on it. A second model made from the same picture with
the hands open gives fingers. This module takes only the hands from it and puts them on the fitted body:

  * a hand is found by colour and shape: the skin-coloured surface at the end of each arm, cut where the cuff
    starts (the wrist ring), on both models;
  * fingers are found by distance over the surface from the wrist: every fingertip is a peak of that distance,
    and a finger is what hangs off the palm above the web where it joins its neighbour. Each gets two bones
    (root to middle, middle to tip) and weights that bend it over a band, not a line;
  * the thumb is the finger that leaves the palm nearest the wrist; the others are named across the palm from it;
  * a hand goes on the arm it belongs to. Generators often draw the thumbs on the wrong side, which makes the
    hand at the end of the left arm a right hand: which it is shows in the order of thumb, fingers and palm,
    and it is seated on the arm of that side;
  * the hand is made as long as the body's own, moved so its wrist ring sits in the body's cuff, turned so it
    leaves the cuff the way the forearm points, and its texture is tinted to the body's own skin. The body's
    wrist joint goes to the middle of that ring, and the stub of arm inside the sleeve stays with the forearm.

What comes back is the joints of every finger and the palm's facing, for the rig (character.py) to build finger
bones on and to curl them: `open`, `relaxed`, `grip`, `point`, `fist`.
"""
from __future__ import annotations

import heapq
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

D = bpy.data
NAMES = ("index", "middle", "ring", "little")


def _colours(ob):
    """Each vertex's colour from the model's texture (sRGB as stored), and the image."""
    me = ob.data
    img = next(n.image for n in me.materials[0].node_tree.nodes
               if n.type == "TEX_IMAGE" and n.outputs[0].links and n.outputs[0].links[0].to_socket.name == "Base Color")
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    loops = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loops)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    col = np.zeros((len(me.vertices), 3), dtype=np.float32)
    col[loops] = px[np.minimum(h - 1, (uv[:, 1] % 1 * h).astype(int)), np.minimum(w - 1, (uv[:, 0] % 1 * w).astype(int)), :3]
    return col, img


def _arrays(ob):
    me = ob.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    ed = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ed)
    return co.reshape(-1, 3), ed.reshape(-1, 2)


def find_hand(ob, side: int, col=None) -> dict:
    """The hand on one side of a model standing in an A-pose (side 1 is +x, the character's left): which
    vertices it is, the middle of the ring where it meets the cuff, the ring's size, and the way the hand
    leaves the cuff."""
    co, ed = _arrays(ob)
    if col is None:
        col, _ = _colours(ob)
    x = co[:, 0] * side
    far = np.argsort(-x)[:400]  # the end of the arm is hand
    skin = np.median(col[far], axis=0)
    span = x.max()
    region = (x > 0.45 * span) & (co[:, 2] < co[far, 2].mean() + 0.45 * span)  # the forearm and hand, not the head
    # skin by hue, not brightness: the creases between fingers are dark skin, a cuff is another colour
    hue = col / np.maximum(col.sum(axis=1, keepdims=True), 1e-3)
    like = region & (np.linalg.norm(hue - skin / skin.sum(), axis=1) < 0.1) & (col.sum(axis=1) > 0.25)
    # flood out from the fingertips over skin-coloured vertices: that stops at the cuff
    nb = {}
    for a, b in ed[like[ed[:, 0]] | like[ed[:, 1]]]:
        nb.setdefault(int(a), []).append(int(b))
        nb.setdefault(int(b), []).append(int(a))
    seen = set(int(i) for i in far if like[i])
    stack = list(seen)
    while stack:
        v = stack.pop()
        for u in nb.get(v, ()):
            if like[u] and u not in seen:
                seen.add(u)
                stack.append(u)
    flood = np.zeros(len(co), dtype=bool)
    flood[list(seen)] = True
    ring = np.array([int(v) for v in seen if any(not flood[u] for u in nb.get(v, ()))])
    if len(ring) < 8:
        raise SystemExit(f"hands: no cuff found on the {'left' if side > 0 else 'right'} arm (the hand's colour "
                         "runs on up the arm): the hands are found where skin meets a sleeve")
    centre = co[ring].mean(axis=0)
    radius = float(np.median(np.linalg.norm(co[ring] - centre, axis=1)))
    # the way the hand leaves the cuff: from the middle of the sleeve's end to the middle of the ring (a plane
    # fitted to the ring itself is thrown by a ragged cuff)
    sleeve = ~flood & (np.linalg.norm(co - centre, axis=1) < 3 * radius)
    sleeve &= ((co - centre) @ (co[flood].mean(axis=0) - centre)) < 0
    normal = centre - co[sleeve].mean(axis=0)
    normal /= np.linalg.norm(normal)
    # the hand is everything past the ring's plane and near it, skin or crease, but not the cuff's own ragged
    # edge, which reaches past the ring
    reach = float(np.linalg.norm(co[flood] - centre, axis=1).max()) * 1.1
    cloth = np.median(hue[sleeve], axis=0)
    hand = ((co - centre) @ normal > -0.15 * radius) & (np.linalg.norm(co - centre, axis=1) < reach) & region
    hand &= np.linalg.norm(hue - cloth, axis=1) > 0.5 * np.linalg.norm(skin / skin.sum() - cloth)
    # and joined to the fingertips: a fleck of the sleeve that is nearer skin's colour than cloth's is not hand,
    # and taken with it leaves a hole in the cuff with the wrist showing through
    joined = {}
    for a, b in ed[hand[ed[:, 0]] & hand[ed[:, 1]]]:
        joined.setdefault(int(a), []).append(int(b))
        joined.setdefault(int(b), []).append(int(a))
    reached = set(int(i) for i in far if hand[i])
    stack = list(reached)
    while stack:
        v = stack.pop()
        for u in joined.get(v, ()):
            if u not in reached:
                reached.add(u)
                stack.append(u)
    whole = np.zeros(len(co), dtype=bool)
    whole[list(reached)] = True
    hand = whole
    length = float(((co[hand] - centre) @ normal).max())  # from the cuff to the furthest fingertip
    return {"verts": hand, "centre": Vector(centre.tolist()), "normal": Vector(normal.tolist()), "radius": radius,
            "skin": skin, "length": length}


def cut_hand(ob, found: dict, name: str, faces: int):
    """A copy of one hand as its own object, the wrist closed with a cap, thinned to about `faces` faces."""
    new = ob.copy()
    new.data = ob.data.copy()
    new.name = name
    bpy.context.scene.collection.objects.link(new)
    bm = bmesh.new()
    bm.from_mesh(new.data)
    bm.verts.ensure_lookup_table()
    keep = found["verts"]
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep[v.index]], context="VERTS")
    loose = [e for e in bm.edges if e.is_boundary]
    if loose:
        bmesh.ops.holes_fill(bm, edges=loose, sides=0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(new.data)
    bm.free()
    if len(new.data.polygons) > faces:
        mod = new.modifiers.new("Decimate", "DECIMATE")
        mod.ratio = faces / len(new.data.polygons)
        bpy.context.view_layer.objects.active = new
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for poly in new.data.polygons:
        poly.use_smooth = True
    return new


def _geodesic(co, ed, start):
    nb = [[] for _ in range(len(co))]
    for a, b in ed:
        d = float(np.linalg.norm(co[a] - co[b]))
        nb[a].append((b, d))
        nb[b].append((a, d))
    g = np.full(len(co), np.inf)
    heap = [(0.0, int(v)) for v in start]
    for _, v in heap:
        g[v] = 0.0
    while heap:
        d, v = heapq.heappop(heap)
        if d > g[v]:
            continue
        for u, w in nb[v]:
            if d + w < g[u]:
                g[u] = d + w
                heapq.heappush(heap, (d + w, u))
    g[np.isinf(g)] = 0.0
    return g, nb


def find_fingers(hand, wrist: Vector, leaves: Vector) -> dict:
    """The fingers of a cut hand, where it is: {"palm": the normal out of the palm, "axis": the way the fingers
    point from the wrist, "knuckles": the far end of the palm, "side": 1 for a left hand and -1 for a right,
    "fingers": [{"name", "joints": [root, mid, tip]}] with the thumb first, "weights": {bone: per-vertex}}."""
    co, ed = _arrays(hand)
    along = (co - np.array(wrist)) @ np.array(leaves.normalized())
    g, nb = _geodesic(co, ed, np.nonzero(along < 0.012)[0])  # from the wrist: the ring and the cap that closes it
    top = float(g.max())
    # sweep down from the furthest point: a component is born at each peak and dies where it meets an older one
    order = np.argsort(-g)
    parent, birth, members, closed, fingers = {}, {}, {}, set(), []

    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for v in order:
        v = int(v)
        parent[v], birth[v], members[v] = v, float(g[v]), [v]
        for u, _ in nb[v]:
            if u not in parent:
                continue
            a, b = root(v), root(u)
            if a == b:
                continue
            old, young = (a, b) if birth[a] >= birth[b] else (b, a)
            level = float(g[v])
            if young not in closed and birth[young] - level > 0.10 * top:  # a finger ends here, at a web
                for r in (young, old):
                    if r not in closed and birth[r] - level > 0.10 * top:
                        fingers.append({"tip": birth[r], "root": level, "verts": list(members[r])})
                closed.add(old)  # what they have joined is palm from here down
            parent[young] = old
            members[old].extend(members[young])
            if young in closed:
                closed.add(old)
    fingers = [f for f in fingers if f["tip"] > 0.45 * top]  # (a bump by the wrist is not a finger)
    fingers = sorted(fingers, key=lambda f: f["root"])[:5]
    if len(fingers) < 3:
        raise SystemExit(f"hands: found {len(fingers)} finger(s) on a hand: the hands model needs open hands with "
                         "the fingers apart")
    for f in fingers:
        v = np.array(f["verts"])
        f["tip_at"] = Vector(co[v[g[v] >= f["tip"] - 0.08 * (f["tip"] - f["root"])]].mean(axis=0).tolist())
    thumb, rest = fingers[0], fingers[1:]
    # the way the fingers point, the palm's own facing (the thumb stands on the palm's side), and across the palm
    axis = (sum((f["tip_at"] for f in rest), Vector()) / len(rest) - wrist).normalized()
    is_thumb = np.zeros(len(co), dtype=bool)
    is_thumb[thumb["verts"]] = True
    palm_v = ~is_thumb & (((co - np.array(wrist)) @ np.array(axis)) < 0.5 * max((f["tip_at"] - wrist).dot(axis) for f in rest))
    centre = Vector(co[palm_v].mean(axis=0).tolist())
    _, _, vt = np.linalg.svd(co[palm_v][::max(1, palm_v.sum() // 2000)] - np.array(centre))
    normal = Vector(vt[2].tolist())
    normal = (normal - axis * normal.dot(axis)).normalized()
    if (thumb["tip_at"] - centre).dot(normal) < 0:
        normal = -normal
    across = axis.cross(normal).normalized()
    side = 1 if (thumb["tip_at"] - centre).dot(across) < 0 else -1  # which hand it is
    # A generated hand is a mitten: its fingers are free only near the tips, and bending just the free part
    # closes nothing. So every finger bends from one knuckle line across the middle of the hand, and what lies
    # past that line belongs to the finger it is in line with (blended into its neighbours across the webs).
    h = (co - np.array(wrist)) @ np.array(axis)
    x = (co - np.array(wrist)) @ np.array(across)
    length = max((f["tip_at"] - wrist).dot(axis) for f in rest)
    knuckle = 0.5 * length
    for f in rest:
        f["x"], f["h"] = (f["tip_at"] - wrist).dot(across), (f["tip_at"] - wrist).dot(axis)
    rest.sort(key=lambda f: f["x"])
    gap = max(0.006, float(np.median(np.diff([f["x"] for f in rest])))) if len(rest) > 1 else 0.02
    for f in rest:
        near = ~is_thumb & (np.abs(x - f["x"]) < 0.5 * gap)
        slab = lambda lo, hi: Vector(co[near & (h > lo) & (h < hi)].mean(axis=0).tolist())  # noqa: E731
        half = knuckle + 0.5 * (f["h"] - knuckle)
        f["joints"] = [slab(knuckle - 0.01, knuckle + 0.01), slab(half - 0.008, half + 0.008), f["tip_at"]]
    # The thumb is all of what stands off the palm under it, not only its free end: on a mitten the web between
    # thumb and palm is thick, so the free end starts half way up, and a thumb bent from there can't lie down
    # beside the fingers. Its root is where it leaves the palm's own surface.
    off = (co - np.array(wrist)) @ np.array(normal)
    x2 = (co - np.array(wrist)) @ np.array(axis.cross(normal))
    tx = (thumb["tip_at"] - wrist).dot(axis.cross(normal))
    flat_palm = ~is_thumb & (h > 0.15 * length) & (h < knuckle) & (np.abs(x2 - tx) > 0.03)
    level = float(np.percentile(off[flat_palm], 90)) if flat_palm.any() else 0.0
    stands = (off > level + 0.006) & (np.abs(x2 - tx) < 0.045) & (h < (thumb["tip_at"] - wrist).dot(axis) + 0.01)
    seen, stack = set(thumb["verts"]), list(thumb["verts"])
    while stack:
        v = stack.pop()
        for u, _ in nb[v]:
            if stands[u] and u not in seen:
                seen.add(u)
                stack.append(u)
    tv = np.array(sorted(seen))
    is_thumb[:] = False
    is_thumb[tv] = True
    low = tv[off[tv] < off[tv].min() + 0.012]
    foot = Vector(co[low].mean(axis=0).tolist())
    span = thumb["tip_at"] - foot
    t0 = np.clip((co[tv] - np.array(foot)) @ np.array(span) / span.length_squared, 0.0, 1.0)
    part = lambda lo, hi: Vector(co[tv[(t0 >= lo) & (t0 <= hi)]].mean(axis=0).tolist())  # noqa: E731
    thumb["joints"] = [foot, part(0.45, 0.58), thumb["tip_at"]]
    thumb["name"] = "thumb"
    rest.sort(key=lambda f: abs(f["x"] - (thumb["tip_at"] - wrist).dot(across)))
    for f, name in zip(rest, NAMES):
        f["name"] = name
    rest = sorted(rest[:len(NAMES)], key=lambda f: f["x"])
    # The palm's facing, for the rig: square to the row of knuckles. A generated hand is cupped, and a plane
    # through the whole palm leans up to 30 degrees from that row, so a hand laid "palm down" by it lies tilted.
    if len(rest) >= 3:
        roots = np.array([f["joints"][0] for f in rest])
        roots = roots - roots.mean(axis=0)
        roots -= np.outer(roots @ np.array(axis), np.array(axis))
        row = Vector(np.linalg.svd(roots)[2][0].tolist())
        row = row if row.dot(across) > 0 else -row
        normal = row.cross(axis).normalized()
    smooth = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))  # noqa: E731
    weights = {"hand": np.ones(len(co), dtype=np.float32)}
    on = np.where(is_thumb, 0.0, smooth((h - (knuckle - 0.015)) / 0.03))  # past the knuckle line
    xs = np.array([f["x"] for f in rest])
    for i, f in enumerate(rest):  # across the palm: all of the finger in line, sharing the webs with its neighbours
        share = np.ones(len(co), dtype=np.float32)
        if i > 0:
            share = np.where(x < xs[i], np.clip((x - xs[i - 1]) / (xs[i] - xs[i - 1]), 0, 1), share)
        if i < len(rest) - 1:
            share = np.where(x >= xs[i], np.clip((xs[i + 1] - x) / (xs[i + 1] - xs[i]), 0, 1), share)
        far = smooth(((h - knuckle) / max(1e-4, f["h"] - knuckle) - 0.4) / 0.2)
        weights[f["name"] + ".a"] = (on * share * (1 - far)).astype(np.float32)
        weights[f["name"] + ".b"] = (on * share * far).astype(np.float32)
    weights["hand"] = (1 - on).astype(np.float32)
    lift = smooth(t0 / 0.14)
    far = smooth((t0 - 0.42) / 0.2)
    a, b_ = np.zeros(len(co), dtype=np.float32), np.zeros(len(co), dtype=np.float32)
    a[tv], b_[tv] = lift * (1 - far), lift * far
    weights["thumb.a"], weights["thumb.b"] = a, b_
    weights["hand"][tv] = 1 - lift
    # the wrist bends over a band past the cuff, and the stub of arm that sits inside the sleeve stays with the
    # forearm: turned with the hand, it comes out through the cloth when the wrist bends
    arm = (1 - smooth(along / 0.02)).astype(np.float32)
    for name in weights:
        weights[name] = weights[name] * (1 - arm)
    weights["fore"] = arm
    laid = lay_thumb(hand, co, ed, wrist, axis, normal, thumb, rest, knuckle, length, a + b_)
    return {"palm": normal, "axis": axis, "knuckles": wrist + axis * knuckle + normal * (centre - wrist).dot(normal),
            "side": side, "weights": weights, "thumb": laid,
            "fingers": [{"name": f["name"], "joints": f["joints"]} for f in [thumb] + rest]}


THUMB_DEPTH = 0.08  # how far the thumb's root lies out of the palm, as a share of the hand's length (wrist to tip)
THUMB_DEPTH_LIMIT = 0.20  # the fit fails if the root still stands further off than this share


def thumb_depth(thumb_root: Vector, rest: list, normal: Vector) -> float:
    """How far the thumb's root stands out of the palm: off the plane through the other fingers' roots, along
    the palm's facing (a hand's own thumb lies just in front of the palm, beside the first finger)."""
    plane = sum(f["joints"][0].dot(normal) for f in rest) / len(rest)
    return thumb_root.dot(normal) - plane


def lay_thumb(hand, co, ed, wrist: Vector, axis: Vector, normal: Vector, thumb: dict, rest: list, knuckle: float,
              length: float, held) -> dict:
    """Bring a generated mitt's thumb in to the palm. A generator makes the thumb stand off the palm on a thick
    wedge of flesh (its root about 40% of the hand's length out of the palm, the wrist as deep as it is wide),
    and a hand like that rests on its thumb on a table and has to tip its fingers down 45 degrees to reach a
    keyboard. So what stands in front of the palm is pressed in towards it: the thumb moves whole until its
    root lies just in front of the first finger, the wedge under it is thinned evenly (nothing folds), and the
    change fades out towards the cuff ring, so the hand still fills the sleeve, and stops at the knuckles, so
    the fingers do not move. `held` is each vertex's weight on the thumb's bones. Changes the mesh and the
    thumb's joints; returns the depth before and after and the most an edge was stretched and squashed."""
    before = thumb_depth(thumb["joints"][0], rest, normal)
    target = THUMB_DEPTH * length
    out = {"depth_before": before, "depth": before, "length": length, "stretch": 1.0, "squash": 1.0}
    if before <= target:
        return out
    w_, a_, n_ = np.array(wrist), np.array(axis), np.array(normal)
    plane = sum(f["joints"][0].dot(normal) for f in rest) / len(rest) - wrist.dot(normal)
    h = (co - w_) @ a_
    off = (co - w_) @ n_ - plane  # out of the palm, from the knuckles' plane
    smooth = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))  # noqa: E731
    # nothing moves that is no further out than the fingers' own fronts (so the fingers keep their thickness),
    # and what is further out comes in by its share of the way to the root (an even squash, nothing folds)
    front = float(np.percentile(off[(h > knuckle + 0.15 * (length - knuckle)) & (held < 0.05)], 98))
    w = np.clip((off - front) / max(1e-4, before - front), 0, 1)
    w *= smooth((h - 0.05 * knuckle) / (0.6 * knuckle))  # none at the cuff ring, all from part way up the palm
    w = np.maximum(w, held)  # the thumb moves whole
    new = co - np.outer(w * (before - target), n_)
    made = np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)
    ratio = np.linalg.norm(new[ed[:, 0]] - new[ed[:, 1]], axis=1) / np.maximum(made, 1e-6)
    hand.data.vertices.foreach_set("co", new.astype(np.float32).ravel())
    hand.data.update()
    thumb["joints"] = [j - normal * (before - target) for j in thumb["joints"]]
    thumb["tip_at"] = thumb["joints"][2]
    out.update(depth=thumb_depth(thumb["joints"][0], rest, normal),
               stretch=float(np.percentile(ratio, 99.9)), squash=float(1 / max(1e-6, np.percentile(ratio, 0.1))))
    return out


def tint(hand_img, was, want):
    """Tint the hands' texture so its skin (`was`, a colour) comes out as the body's (`want`)."""
    w, h = hand_img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    hand_img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    px[:, :3] = np.clip(px[:, :3] * (np.array(want) / np.maximum(np.array(was), 1e-3)), 0, 1)
    hand_img.pixels.foreach_set(px.ravel())
    hand_img.update()
    hand_img.pack()


def _mirrored(piece, theirs: dict, found: dict):
    """A hand, its wrist and its fingers, mirrored left to right through the wrist: the other hand of the pair."""
    new = piece.copy()
    new.data = piece.data.copy()
    bpy.context.scene.collection.objects.link(new)
    c = theirs["centre"]
    m = Matrix.Translation(c) @ Matrix.Scale(-1, 4, Vector((1, 0, 0))) @ Matrix.Translation(-c)
    new.data.transform(m)
    new.data.flip_normals()
    new.data.update()
    flip = lambda v: Vector((-v.x, v.y, v.z))  # noqa: E731
    return (new, {**theirs, "normal": flip(theirs["normal"])},
            {**found, "palm": flip(found["palm"]), "axis": flip(found["axis"]), "knuckles": m @ found["knuckles"],
             "side": -found["side"],
             "fingers": [{"name": f["name"], "joints": [m @ j for j in f["joints"]]} for f in found["fingers"]]})


def flatten(piece, centre: Vector, radius: float) -> int:
    """Smooth a lump off a hand: the vertices within `radius` of a point are drawn to the mean of their
    neighbours until the skin lies as a membrane across where the lump stood (what is further away holds still).
    Returns how many vertices moved."""
    co, ed = _arrays(piece)
    loose = np.linalg.norm(co - np.array(centre), axis=1) < radius
    if not loose.any():
        return 0
    for _ in range(400):
        total, count = np.zeros_like(co), np.zeros(len(co))
        np.add.at(total, ed[:, 0], co[ed[:, 1]])
        np.add.at(total, ed[:, 1], co[ed[:, 0]])
        np.add.at(count, ed[:, 0], 1)
        np.add.at(count, ed[:, 1], 1)
        mean = total / np.maximum(count, 1)[:, None]
        co[loose] = mean[loose]
    piece.data.vertices.foreach_set("co", co.astype(np.float32).ravel())
    piece.data.update()
    return int(loose.sum())


def graft(body, donor, src: dict, faces: int = 9000, scale: float | None = None, tuck: float = 0.012,
          lumps=()):
    """Replace the body's hands with the donor's. `src` is the body's joints (source_joints). Returns the fingers
    of each side ({"L": {...}, "R": {...}}, in the body's space) and the hand objects, to join to the body once
    it is weighted. Each `tip` joint moves to the knuckles: the hand bone is the palm, and the fingers go on
    from it. The donor's hand is made as long as the body's own (cuff to fingertip, the mean of its two
    hands, so the pair match): a second generation of the same character at the same height still comes with
    hands half as long again, and everything the rig measures off a hand (the desk, the keys, how far apart
    two hands type) follows its size. `scale` sets the donor's scale by hand instead (a ragged cuff is too
    rough a measure to size a hand by, so the cuff's width is only printed). `lumps` are places to smooth flat on
    both hands, for a nub the generator left: (along the fingers, out of the palm, across, radius) in cm on the
    fitted hand, from the middle of the knuckle row, with the back of the hand negative out of the palm. Across
    is read on the hand that was kept as it was made (the fit prints which), and mirrored for the other."""
    body_col, _ = _colours(body)
    donor_col, donor_img = _colours(donor)
    cut = []
    for side in (1, -1):
        theirs = find_hand(donor, side, donor_col)
        piece = cut_hand(donor, theirs, "hand_piece", faces)
        found = find_fingers(piece, theirs["centre"], theirs["normal"])
        # how well formed it is: fingers that point the same way as each other (a splayed hand can't close on a bar)
        ways = [(f["joints"][2] - f["joints"][0]).normalized() for f in found["fingers"][1:]]
        found["form"] = len(ways) + sum(a.dot(b) for i, a in enumerate(ways) for b in ways[i + 1:]) / max(1, len(ways) * (len(ways) - 1) / 2)
        cut.append((piece, theirs, found))
    # Generators draw hands badly: a thumb on the wrong side, fingers splayed on one hand and not the other. So
    # the better formed of the two is used for both arms: as it is on the arm it belongs to (which hand it is
    # shows in the order of thumb, fingers and palm), and mirrored on the other. The two hands then match.
    best = max(cut, key=lambda c: c[2]["form"])
    laid = best[2]["thumb"]  # (the hand's length is wrist to fingertip, so the depth reads the same at any size)
    print(f"fit: hands, thumb root {100 * laid['depth'] / laid['length']:.0f}% of the hand's length out of the palm "
          f"(made {100 * laid['depth_before'] / laid['length']:.0f}%), edges stretched {laid['stretch']:.2f}x and "
          f"squashed {laid['squash']:.2f}x at most")
    if laid["depth"] > THUMB_DEPTH_LIMIT * laid["length"]:
        print(f"fit: FAIL hands, the thumb's root stands {100 * laid['depth'] / laid['length']:.0f}% of the hand's "
              f"length off the palm (limit {100 * THUMB_DEPTH_LIMIT:.0f}%): the hand will rest on its thumb")
    for piece, _t, _f in cut:
        if piece is not best[0]:
            D.objects.remove(piece)
    cut = [best if best[2]["side"] == side else _mirrored(*best) for side in (1, -1)]
    print(f"fit: hands, both from the model's {'left' if best[1]['centre'].x > 0 else 'right'}-arm hand, which is a "
          f"{'left' if best[2]['side'] > 0 else 'right'} hand (form {best[2]['form']:.2f})")
    out, pieces, skins, old = {}, [], [], np.zeros(len(body.data.vertices), dtype=bool)
    own = {side: find_hand(body, side, body_col) for side in (1, -1)}
    measured = 0.5 * (own[1]["length"] + own[-1]["length"]) / best[1]["length"]
    print(f"fit: hands, the body's own are {50 * (own[1]['length'] + own[-1]['length']):.1f} cm from cuff to "
          f"fingertip and the donor's {100 * best[1]['length']:.1f} cm: "
          + (f"scaled {measured:.2f}x to match" if scale is None else
             f"scaled {scale:.2f}x as the fit says ({measured:.2f}x would match)"))
    scale = measured if scale is None else scale
    # the wrist joint goes where the hand meets the cuff, the same on both sides (the mean of the two rings,
    # one mirrored): read off a still by hand it lands some centimetres up the sleeve or inside the hand, and
    # then the sleeve folds where there is no joint, or the hand turns about its knuckles
    ring = (own[1]["centre"] + Vector((-own[-1]["centre"].x, own[-1]["centre"].y, own[-1]["centre"].z))) / 2
    moved = (ring - src["wrL"]).length
    for side, sd in ((1, "L"), (-1, "R")):
        src["wr" + sd] = Vector((ring.x * side, ring.y, ring.z))
    print(f"fit: hands, the wrist joint moved {100 * moved:.1f} cm from where the fit put it to the middle of the cuff")
    for (piece, theirs, found), (side, sd) in zip(cut, ((1, "L"), (-1, "R"))):
        mine = own[side]
        forearm = (src["wr" + sd] - src["el" + sd]).normalized()
        turn = theirs["normal"].rotation_difference(forearm).to_matrix().to_4x4()
        # and turned about its own length so the palm faces the leg, as a hand hangs: generators leave the
        # palms facing forward or up, which reads as hands put on the wrong way round
        way, palm = turn.to_3x3() @ found["axis"], turn.to_3x3() @ found["palm"]
        flat = lambda v: (v - way * v.dot(way)).normalized()  # noqa: E731
        now, want = flat(palm), flat(Vector((-side, 0, 0)))
        turn = Matrix.Rotation(math.atan2(now.cross(want).dot(way), now.dot(want)), 4, way) @ turn
        m = Matrix.Translation(mine["centre"] - forearm * tuck) @ turn @ Matrix.Scale(scale, 4) @ \
            Matrix.Translation(-theirs["centre"])
        piece.data.transform(m)
        piece.data.update()
        piece.name = "hand_" + sd
        for name, w in found["weights"].items():
            group = piece.vertex_groups.new(name=f"{name}.{sd}")
            for i in np.nonzero(w > 1e-3)[0]:
                group.add([int(i)], float(w[i]), "REPLACE")
        spin = turn.to_3x3()
        if lumps:
            way, palm = (spin @ found["axis"]).normalized(), (spin @ found["palm"]).normalized()
            across = way.cross(palm) * (1 if found["side"] == best[2]["side"] else -1)  # (a mirrored hand's is mirrored)
            roots = [m @ f["joints"][0] for f in found["fingers"][1:]]
            row = sum(roots, Vector()) / len(roots)
            for a_, o_, x_, r_ in lumps:
                moved = flatten(piece, row + (way * a_ + palm * o_ + across * x_) * 0.01, r_ * 0.01)
                print(f"fit: hands, {sd} a lump smoothed flat ({moved} vertices within {r_} cm)")
        src["tip" + sd] = m @ found["knuckles"]
        out[sd] = {"palm": list(spin @ found["palm"]), "axis": list(spin @ found["axis"]),
                   "fingers": [{"name": f["name"], "joints": [m @ j for j in f["joints"]]} for f in found["fingers"]]}
        skins.append((mine["skin"], theirs["skin"]))
        old |= mine["verts"]
        pieces.append(piece)
        print(f"fit: hands, {sd} {len(found['fingers'])} fingers ({', '.join(f['name'] for f in found['fingers'])}), "
              f"{len(piece.data.polygons)} faces, wrist {mine['radius'] * 100:.1f} cm against the donor's "
              f"{theirs['radius'] * 100:.1f} cm")
    tint(donor_img, np.mean([s[1] for s in skins], axis=0), np.mean([s[0] for s in skins], axis=0))
    bm = bmesh.new()  # the body loses its own hands
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if old[v.index]], context="VERTS")
    bm.to_mesh(body.data)
    bm.free()
    return out, pieces


def join(body, pieces):
    """The hands become part of the body mesh (a second material: their own texture)."""
    for o in bpy.context.view_layer.objects:
        o.select_set(o in pieces or o is body)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
