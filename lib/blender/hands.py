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


BARE_REACH = 0.30  # skin running further than this from the fingertips (m) is a bare arm, not a hand in a cuff


def _bare_wrist(co, far, flood, skin, col):
    """A hand on a bare arm (a vest top, a sleeveless dress): the skin runs on from the fingertips up to the
    shoulder, so there is no cuff to cut at. The cut is the wrist instead: the narrowest slice of the arm between
    a hand's length and a forearm's from the fingertips, across the line the hand and forearm make. Returns what
    find_hand returns, or None when the skin stops within a hand's reach (a cuff)."""
    tip = co[far].mean(axis=0)
    dist = np.linalg.norm(co - tip, axis=1)
    if dist[flood].max() < BARE_REACH:
        return None
    arm = flood & (dist < 0.40)  # the hand and the forearm
    P = co[arm]
    centre_all = P.mean(axis=0)
    axis = np.linalg.svd(P - centre_all, full_matrices=False)[2][0]
    if (centre_all - tip) @ axis < 0:
        axis = -axis  # from the fingertips toward the elbow
    t = (co - tip) @ axis
    slices = []
    for d in np.arange(0.09, 0.30, 0.005):
        slab = arm & (np.abs(t - d) < 0.006)
        if slab.sum() < 12:
            continue
        c = co[slab].mean(axis=0)
        rel = co[slab] - c
        radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
        slices.append((float(np.median(radial)), c, slab, d))
    if not slices:
        return None
    # (a forearm can be as narrow for some centimetres above the wrist as at it, and which slice of that stretch
    # measures narrowest changes with the mesh: the wrist is the end of the stretch nearest the hand)
    narrow = min(sl[0] for sl in slices)
    *best, d_best = next(sl for sl in slices if sl[0] <= 1.05 * narrow)
    radius, centre, slab = best
    # (the skin's colour is read at the wrist, where the two models meet with nothing over the join: fingertips
    # are paler or pinker than a forearm, and matched by them the hand ends in a line of another colour)
    skin = np.median(col[slab], axis=0)
    normal = -axis  # the way the hand leaves the wrist
    hand = flood & (t < d_best + 0.15 * radius)
    length = float(((co[hand] - centre) @ normal).max())
    print(f"hands: a bare arm (skin runs {100 * dist[flood].max():.0f} cm from the fingertips), cut at the wrist "
          f"{100 * d_best:.1f} cm up from them, {100 * radius:.1f} cm across")
    return {"verts": hand, "centre": Vector(centre.tolist()), "normal": Vector(normal.tolist()), "radius": radius,
            "skin": skin, "length": length, "bare": True,
            "cut": Vector((tip + axis * (d_best + 0.15 * radius)).tolist())}


def find_hand(ob, side: int, col=None, hue_tol: float = 0.1, light: float = 0.0) -> dict:
    """The hand on one side of a model standing in an A-pose (side 1 is +x, the character's left): which
    vertices it is, the middle of the ring where it meets the cuff, the ring's size, and the way the hand
    leaves the cuff. `hue_tol` is how near the fingertips' hue a vertex must be to count as skin: a sleeve of
    brown or tan wool is within 0.1 of skin (one was 0.07), and then the hand runs on up the sleeve. `light` is
    for a dark sleeve of skin's own hue (dark brown wool was 0.04 from pale skin, nearer than the skin's own
    spread): a vertex under that share of the fingertips' brightness is not skin, whatever its hue."""
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
    like = region & (np.linalg.norm(hue - skin / skin.sum(), axis=1) < hue_tol) & (col.sum(axis=1) > 0.25)
    if light > 0:
        like &= col.sum(axis=1) > light * float(skin.sum())
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
    bare = _bare_wrist(co, far, flood, skin, col)
    if bare is not None:
        return bare
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


def seam_verts(ob) -> np.ndarray:
    """The vertices of a welded mesh that sit on a seam of its texture: those whose face corners do not all
    carry the same UV."""
    me = ob.data
    vi = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", vi)
    uv = np.empty(len(me.loops) * 2, dtype=np.float64)
    me.uv_layers.active.data.foreach_get("uv", uv)
    rows = np.unique(np.column_stack([vi, np.round(uv.reshape(-1, 2) * 65536)]).astype(np.int64), axis=0)
    return np.nonzero(np.bincount(rows[:, 0], minlength=len(me.vertices)) > 1)[0]


def reduce(ob, faces: int) -> int:
    """Thin a welded mesh to about `faces` faces without touching the seams of its texture. A generated model's
    texture is an atlas of hundreds of small islands, and an edge collapsed across the border of two leaves a
    face with corners in both: it is painted with whatever lies between them in the atlas, a shard of the wrong
    colour (by an eye, down the side of a finger). So the vertices on a seam are held out of the reduction and
    the faces come off the insides of the islands. Returns how many vertices were held."""
    if len(ob.data.polygons) <= faces:
        return 0
    held = seam_verts(ob)
    group = ob.vertex_groups.new(name="reduce")
    group.add(list(range(len(ob.data.vertices))), 1.0, "REPLACE")
    group.add([int(i) for i in held], 0.0, "REPLACE")
    mod = ob.modifiers.new("Decimate", "DECIMATE")
    mod.ratio = faces / len(ob.data.polygons)
    mod.vertex_group = group.name
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.vertex_groups.remove(ob.vertex_groups["reduce"])
    return len(held)


def paint_lost(ob, was: dict) -> float:
    """The share of a reduced mesh's faces that are painted from the wrong part of the texture: the middle of
    each face is looked up in the texture where its own corners put it, and where the nearest point of the mesh
    as it was (`was`, from `surface`) had it. A face inside one island of the atlas agrees with the model to
    within its own size; a face left with corners in two islands reads from between them."""
    from mathutils.bvhtree import BVHTree
    me = ob.data
    n = len(me.polygons)
    if not n or any(p.loop_total != 3 for p in me.polygons):
        return 0.0
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    vi = np.empty(n * 3, dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    uv = np.empty(n * 6, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    P = co.reshape(-1, 3)[vi].reshape(n, 3, 3)
    mid, now = P.mean(axis=1), uv.reshape(n, 3, 2).mean(axis=1)
    size = np.linalg.norm(P - np.roll(P, 1, axis=1), axis=2).max(axis=1)  # each face's longest edge (m)
    tree = BVHTree.FromPolygons([tuple(v) for v in was["co"]], [tuple(t) for t in was["tris"]])
    hit = [tree.find_nearest(tuple(m)) for m in mid]
    at = np.array([h[0] for h in hit], dtype=np.float64)
    tri = np.array([h[2] for h in hit])
    a, b, c = (was["co"][was["tris"][tri, k]].astype(np.float64) for k in range(3))
    e1, e2, p = b - a, c - a, at - a
    d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    p1, p2 = (p * e1).sum(1), (p * e2).sum(1)
    det = np.where(np.abs(d11 * d22 - d12 * d12) < 1e-20, 1e-20, d11 * d22 - d12 * d12)
    v, w = (d22 * p1 - d12 * p2) / det, (d11 * p2 - d12 * p1) / det
    then = was["uv"][tri, 0] * (1 - v - w)[:, None] + was["uv"][tri, 1] * v[:, None] + was["uv"][tri, 2] * w[:, None]
    # how much texture a metre of the model's surface covers (the middle value over its faces)
    A, U = was["co"][was["tris"]], was["uv"]
    per_m = float(np.median(np.linalg.norm(U[:, 1] - U[:, 0], axis=1)
                            / np.maximum(np.linalg.norm(A[:, 1] - A[:, 0], axis=1), 1e-7)))
    return float((np.linalg.norm(now - then, axis=1) > 2 * per_m * size + 2e-3).mean())


def surface(ob) -> dict:
    """A mesh's surface and paint as they are now (triangles, their UVs, the texture), to hold a reduced copy
    against with `paint_lost`."""
    me = ob.data
    n = len(me.polygons)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    start = np.empty(n, dtype=np.int32)
    me.polygons.foreach_get("loop_start", start)
    corners = start[:, None] + np.arange(3)[None, :]  # (a face with more corners is read by its first three)
    img = _colours(ob)[1]
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return {"co": co.reshape(-1, 3), "tris": vi[corners], "uv": uv.reshape(-1, 2)[corners], "px": px.reshape(h, w, 4)}


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
    kept = {v for v in bm.verts if keep[v.index]}
    if found.get("bare"):
        # (a hand on a bare arm is cut off with a plane: taken vertex by vertex, its wrist is the ragged edge of
        # whatever triangles ended there, and with no cuff over it that edge shows as shards round the wrist)
        at, normal, radius = found["cut"], found["normal"], found["radius"]
        near = [v for v in bm.verts if (v.co - at).length < 4 * radius]
        across = {f for v in near for f in v.link_faces}
        edges = {e for f in across for e in f.edges}
        cut = bmesh.ops.bisect_plane(bm, geom=[*near, *edges, *across], dist=1e-6, plane_co=at, plane_no=normal)
        kept |= {v for v in cut["geom_cut"] if isinstance(v, bmesh.types.BMVert)
                 and any(e.other_vert(v) in kept for e in v.link_edges)}
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v not in kept], context="VERTS")
    loose = [e for e in bm.edges if e.is_boundary]
    if loose:
        bmesh.ops.holes_fill(bm, edges=loose, sides=0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(new.data)
    bm.free()
    reduce(new, faces)
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
    shares = np.ones((len(rest), len(co)), dtype=np.float32)
    for i, f in enumerate(rest):  # across the palm: all of the finger in line, sharing the webs with its neighbours
        share = shares[i]
        if i > 0:
            share = np.where(x < xs[i], np.clip((x - xs[i - 1]) / (xs[i] - xs[i - 1]), 0, 1), share)
        if i < len(rest) - 1:
            share = np.where(x >= xs[i], np.clip((xs[i + 1] - x) / (xs[i + 1] - xs[i]), 0, 1), share)
        shares[i] = share
    # A finger's free part is its own, all of it: shared by where it lies across the hand, the side of a free
    # finger that faces its neighbour went half with the neighbour, and a pointing finger left a sheet of its
    # skin behind with the curled ones. Where the fingers are free from the knuckles that is most of the finger;
    # on a mitten it is the tips. The step at each web is spread over a few edges.
    for i, f in enumerate(rest):
        shares[:, f["verts"]] = 0.0
        shares[i, f["verts"]] = 1.0
    free = np.zeros(len(co), dtype=bool)
    for f in rest:
        free[f["verts"]] = True
    for _ in range(3):
        total = np.zeros_like(shares)
        count = np.zeros(len(co), dtype=np.float32)
        for a_, b_ in ((0, 1), (1, 0)):
            np.add.at(total, (slice(None), ed[:, a_]), shares[:, ed[:, b_]])
            np.add.at(count, ed[:, a_], 1.0)
        mean = total / np.maximum(count, 1.0)
        shares = np.where(free[None, :], shares, mean)  # (the free part stays whole; the web under it takes the step)
    for i, f in enumerate(rest):
        share = shares[i]
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
            # (thumb first, then index to little whichever hand it is: across the palm a right hand runs the other way)
            "fingers": [{"name": f["name"], "joints": f["joints"]}
                        for f in [thumb] + sorted(rest, key=lambda f: NAMES.index(f["name"]))]}


THUMB_STRETCH_LIMIT = 3.0  # an edge of the thumb's web stretched more than this by laying the thumb in has torn
THUMB_SPREAD_TO = 2.0  # laying the thumb in is spread over more of the web until no edge is stretched past this
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
    out = {"depth_before": before, "depth": before, "length": length, "stretch": 1.0, "squash": 1.0, "spread": 0}
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
    made = np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1)

    def pressed(w):
        new = co - np.outer(w * (before - target), n_)
        return new, np.linalg.norm(new[ed[:, 0]] - new[ed[:, 1]], axis=1) / np.maximum(made, 1e-6)

    new, ratio = pressed(w)
    # A thumb that was made near the palm has a thin web and no wedge to thin: the thumb moves whole, the skin
    # beside its root hardly at all, and the edges between them are pulled to several times their length (the
    # web folds over itself, a torn patch at the thumb's root). So the move is spread out over the skin round
    # the thumb, each vertex taking the mean of its neighbours, until no edge is stretched past THUMB_SPREAD_TO.
    # The cuff ring and the fingers past the knuckles still do not move, and the thumb still moves whole.
    stays = smooth((h - 0.05 * knuckle) / (0.6 * knuckle)) * (1 - smooth((h - knuckle) / (0.25 * (length - knuckle))))
    count = np.zeros(len(co))
    np.add.at(count, ed[:, 0], 1)
    np.add.at(count, ed[:, 1], 1)
    rounds = 0
    while np.percentile(ratio, 99.9) > THUMB_SPREAD_TO and rounds < 60:
        for _ in range(5):
            total = np.zeros(len(co))
            np.add.at(total, ed[:, 0], w[ed[:, 1]])
            np.add.at(total, ed[:, 1], w[ed[:, 0]])
            w = np.maximum(held, np.minimum(stays, total / np.maximum(count, 1)))
        new, ratio = pressed(w)
        rounds += 1
    out["spread"] = rounds
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


TINT_LIMIT = 1.5  # the most the hands' skin is brightened or darkened, per channel, to match the body's
NOT_SKIN = 0.45  # a texel this far from skin's hue toward grey (0 skin, 1 no colour at all) is not skin
UNPAINTED = 0.3  # a texel under this share of skin's brightness is not skin either: the atlas's own ground


def skin_only(pieces, hand_img, skin) -> float:
    """Paint out what is not skin in the texture the hands use. A model generated from a picture of open hands
    has the picture's background baked into the sides of the fingers and the webbing between them, where the
    picture showed background: grey streaks down every finger once the hand is on an arm. Every texel inside a
    face of the hands whose hue is nearer no colour than skin's becomes skin, as light or dark as it was.
    So does a texel far darker than any crease (the cap that closes a wrist and the odd sliver of a finger are
    painted from the atlas's unpainted ground, near black: flecks at the cuff and a dark nick on a finger).
    Returns the share of the hands' texels that were painted."""
    w, h = hand_img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    hand_img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    skin = np.array(skin, dtype=np.float32)
    hue0, grey = skin / max(float(skin.sum()), 1e-3), np.full(3, 1 / 3, dtype=np.float32)
    line = grey - hue0
    if float(line @ line) < 1e-4:  # (skin with no colour of its own: nothing to tell it from grey by)
        return 0.0
    seen = painted = moved = 0
    for piece in pieces:
        me = piece.data
        uv = np.empty(len(me.loops) * 2, dtype=np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2) % 1 * (w, h)
        across = []
        for poly in me.polygons:
            if poly.loop_total != 3:
                continue
            a, b, c = uv[poly.loop_start:poly.loop_start + 3]
            x0, x1 = int(min(a[0], b[0], c[0])), int(max(a[0], b[0], c[0])) + 1
            y0, y1 = int(min(a[1], b[1], c[1])), int(max(a[1], b[1], c[1])) + 1
            if x1 - x0 > 96 or y1 - y0 > 96:  # (a face across two islands of the atlas: a cap, not skin)
                across.append(poly.index)
                continue
            ys, xs = np.mgrid[y0:min(y1 + 1, h), x0:min(x1 + 1, w)]
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(det) < 1e-9:  # (a face painted from one point of the atlas: that point, if it is unpainted)
                yy, xx = min(h - 1, int(a[1])), min(w - 1, int(a[0]))
                if px[yy, xx, :3].sum() < UNPAINTED * float(skin.sum()):
                    px[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2, :3] = skin * 0.6
                continue
            l1 = ((b[1] - c[1]) * (xs + 0.5 - c[0]) + (c[0] - b[0]) * (ys + 0.5 - c[1])) / det
            l2 = ((c[1] - a[1]) * (xs + 0.5 - c[0]) + (a[0] - c[0]) * (ys + 0.5 - c[1])) / det
            inside = (l1 > -0.35) & (l2 > -0.35) & (l1 + l2 < 1.35)  # (and a texel's width round it: filtering)
            if not inside.any():
                continue
            tile = px[y0:y0 + inside.shape[0], x0:x0 + inside.shape[1], :3]
            total = tile.sum(axis=2)
            t = ((tile / np.maximum(total[..., None], 1e-3) - hue0) @ line) / float(line @ line)
            off = inside & (((t > NOT_SKIN) & (total > 0.1)) | (total < UNPAINTED * float(skin.sum())))
            seen += int(inside.sum())
            painted += int(off.sum())
            if off.any():
                tile[off] = skin * np.clip(total[off] / float(skin.sum()), 0.6, 1.05)[:, None]
        # A face across two islands shows whatever lies between them in the atlas, often its unpainted ground:
        # the cap that closes the wrist (dark flecks in the cuff) and the odd sliver of a finger (a dark nick).
        # Each is painted from one texel of plain skin instead, a little shaded.
        if across:
            at = np.minimum((h - 1, w - 1), uv[:, ::-1].astype(int))
            plain = int(np.argmin(np.abs(px[at[:, 0], at[:, 1], :3] - 0.85 * skin).sum(axis=1)))
            flat = np.empty(len(me.loops) * 2, dtype=np.float32)
            me.uv_layers.active.data.foreach_get("uv", flat)
            flat = flat.reshape(-1, 2)
            for i in across:
                poly = me.polygons[i]
                flat[poly.loop_start:poly.loop_start + 3] = flat[plain]
            me.uv_layers.active.data.foreach_set("uv", flat.ravel())
            moved += len(across)
    if moved:
        print(f"fit: hands, {moved} faces were painted from between two patches of the texture (the wrist's cap, "
              "slivers): now plain skin")
    hand_img.pixels.foreach_set(px.ravel())
    hand_img.update()
    return painted / max(1, seen)


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
          lumps=(), hue_tol: float = 0.1, donor_hue_tol: float | None = None, light: float = 0.0,
          donor_light: float | None = None):
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
        theirs = find_hand(donor, side, donor_col, hue_tol if donor_hue_tol is None else donor_hue_tol,
                           light if donor_light is None else donor_light)
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
          f"squashed {laid['squash']:.2f}x at most"
          + (f" (the move spread over the web in {laid['spread']} rounds)" if laid["spread"] else ""))
    if laid["depth"] > THUMB_DEPTH_LIMIT * laid["length"]:
        print(f"fit: FAIL hands, the thumb's root stands {100 * laid['depth'] / laid['length']:.0f}% of the hand's "
              f"length off the palm (limit {100 * THUMB_DEPTH_LIMIT:.0f}%): the hand will rest on its thumb")
    if laid["stretch"] > THUMB_STRETCH_LIMIT:
        print(f"fit: FAIL hands, laying the thumb in stretched an edge of its web {laid['stretch']:.1f}x (limit "
              f"{THUMB_STRETCH_LIMIT:.0f}x): the skin there folds over itself and shows as a torn patch at the "
              "thumb's root in a close shot. Make the open-hands model again with the thumb nearer the palm")
    for piece, _t, _f in cut:
        if piece is not best[0]:
            D.objects.remove(piece)
    cut = [best if best[2]["side"] == side else _mirrored(*best) for side in (1, -1)]
    print(f"fit: hands, both from the model's {'left' if best[1]['centre'].x > 0 else 'right'}-arm hand, which is a "
          f"{'left' if best[2]['side'] > 0 else 'right'} hand (form {best[2]['form']:.2f})")
    out, pieces, skins, old, cuts = {}, [], [], np.zeros(len(body.data.vertices), dtype=bool), []
    own = {side: find_hand(body, side, body_col, hue_tol, light) for side in (1, -1)}
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
        if mine.get("bare"):  # (no cuff over the join: `seal` closes the arm's end onto the hand, once it has weights)
            piece["bare_cut"] = [*mine["cut"], *mine["normal"], mine["radius"]]
            cuts.append((mine["cut"], mine["normal"], mine["radius"]))
        old |= mine["verts"]
        pieces.append(piece)
        print(f"fit: hands, {sd} {len(found['fingers'])} fingers ({', '.join(f['name'] for f in found['fingers'])}), "
              f"{len(piece.data.polygons)} faces, wrist {mine['radius'] * 100:.1f} cm against the donor's "
              f"{theirs['radius'] * 100:.1f} cm")
    grey = skin_only(pieces, donor_img, np.mean([s[1] for s in skins], axis=0))
    print(f"fit: hands, {100 * grey:.2f}% of their texture was not skin's colour (background baked between the "
          "fingers) and was painted skin")
    was, want = np.mean([s[1] for s in skins], axis=0), np.mean([s[0] for s in skins], axis=0)
    gain = want / np.maximum(was, 1e-3)
    if gain.max() > TINT_LIMIT or gain.min() < 1 / TINT_LIMIT:
        # (two generations of one character have near the same skin: a gain this big means the donor's skin was
        # read off something else, a sleeve of a colour near skin's, and applied it turns the hands white or black)
        print(f"fit: WARN hands, the donor's skin read as {np.round(was, 3).tolist()} against the body's "
              f"{np.round(want, 3).tolist()}: a tint of {np.round(gain, 2).tolist()} would wash the hands out, so they "
              "keep their own colour")
    else:
        tint(donor_img, was, want)
    bm = bmesh.new()  # the body loses its own hands
    bm.from_mesh(body.data)
    bm.verts.ensure_lookup_table()
    gone = {v for v in bm.verts if old[v.index]}
    for at, normal, radius in cuts:
        # a bare arm is cut across with a plane, so its end is a clean ring: taking the hand's vertices away
        # leaves the edge of whatever triangles straddled the wrist, a ragged ring with nothing over it
        near = [v for v in bm.verts if (v.co - at).length < 4 * radius]
        faces = {f for v in near for f in v.link_faces}
        edges = {e for f in faces for e in f.edges}
        bmesh.ops.bisect_plane(bm, geom=[*near, *edges, *faces], dist=1e-6, plane_co=at, plane_no=normal)
        past = lambda v: (v.co - at).dot(normal) > 1e-5 and (v.co - at).length < 0.3  # noqa: E731
        stack = [v for v in gone if past(v)]
        while stack:  # (and the creases and flecks beyond the plane that were not skin's colour)
            v = stack.pop()
            for e in v.link_edges:
                u = e.other_vert(v)
                if u not in gone and past(u):
                    gone.add(u)
                    stack.append(u)
    bmesh.ops.delete(bm, geom=list(gone), context="VERTS")
    bm.to_mesh(body.data)
    bm.free()
    return out, pieces


SEAL_BAND = 0.03  # how far up a bare arm from its cut the skin is drawn in to meet the hand (m)


def seal(body, pieces) -> list:
    """Close the end of a bare arm onto the hand that was put in it (after the body has its weights, before the
    hands are joined to it). A hand in a cuff is tucked up the sleeve and the cloth hides the join; a bare arm
    ends in an open ring a few millimetres wider than the donor's wrist, with the hand's stub loose inside it.
    Each vertex of that ring goes to the nearest point of the hand's skin and takes the weights the hand has
    there, so the join stays shut when the wrist bends; the arm above it follows, less and less, over
    SEAL_BAND. Returns, for each bare arm, the widest gap between ring and hand before and after (m)."""
    from mathutils.kdtree import KDTree
    names = {g.index: g.name for g in body.vertex_groups}
    out = []
    for piece in pieces:
        if "bare_cut" not in piece.keys():
            continue
        c = list(piece["bare_cut"])
        at, normal, radius = Vector(c[:3]), Vector(c[3:6]), c[6]
        to_piece = piece.matrix_world.inverted() @ body.matrix_world
        back = to_piece.inverted()
        up = lambda v: -(v.co - at).dot(normal)  # noqa: E731  (how far up the arm from the cut)
        band = [v for v in body.data.vertices if -1e-4 < up(v) < SEAL_BAND and (v.co - at).length < 4 * radius]
        ring = [v for v in band if up(v) < 1e-4]
        if len(ring) < 6:
            continue
        move, takes, before = {}, {}, 0.0
        tree, ring_at = KDTree(len(ring)), [v.index for v in ring]
        # the hand is cut across on the arm's own plane, so it has a ring of vertices where the arm ends, and
        # its stub above that is drawn in to half its width: a donor's wrist is not the arm's shape, and where
        # it was the wider its end stood over the arm like the cuff of a glove
        bm = bmesh.new()
        bm.from_mesh(piece.data)
        p_at, p_no = to_piece @ at, (to_piece.to_3x3() @ normal).normalized()
        bmesh.ops.bisect_plane(bm, geom=[*bm.verts, *bm.edges, *bm.faces], dist=1e-6, plane_co=p_at, plane_no=p_no)
        flat = lambda v: abs((v.co - p_at).dot(p_no)) < 1e-5  # noqa: E731
        on = [v.co.copy() for v in bm.verts if flat(v)]
        middle = sum(on, Vector()) / max(1, len(on))
        dv = bm.verts.layers.deform.verify()
        gname = {g.index: g.name for g in piece.vertex_groups}
        rim = [(back @ e.verts[0].co, back @ e.verts[1].co, *({gname[k]: w for k, w in v[dv].items()} for v in e.verts))
               for e in bm.edges if flat(e.verts[0]) and flat(e.verts[1])]
        for v in bm.verts:
            rise = -(v.co - p_at).dot(p_no)
            if rise > 1e-5:
                out_ = (v.co - middle) - p_no * (v.co - middle).dot(p_no)
                v.co = v.co - out_ * 0.5
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        bm.to_mesh(piece.data)
        bm.free()
        piece.data.update()
        for i, v in enumerate(ring):  # (onto the hand's own ring: the nearest point of its rim, edge by edge)
            best = None
            for a, b, wa, wb in rim:
                t = max(0.0, min(1.0, (v.co - a).dot(b - a) / max((b - a).length_squared, 1e-12)))
                d = (a.lerp(b, t) - v.co).length
                if best is None or d < best[0]:
                    best = (d, a.lerp(b, t), {k: wa.get(k, 0.0) * (1 - t) + wb.get(k, 0.0) * t for k in set(wa) | set(wb)})
            before = max(before, best[0])
            move[i], takes[i] = best[1] - v.co, best[2]
            tree.insert(v.co, i)
        tree.balance()
        plan = []
        for v in band:
            _co, i, _d = tree.find(v.co)
            k = 1 - up(v) / SEAL_BAND
            plan.append((v.index, i, max(0.0, k) ** 2 * (3 - 2 * max(0.0, k))))
        for index, i, k in plan:
            v = body.data.vertices[index]
            v.co = v.co + move[i] * k
            own = {names[g.group]: g.weight for g in v.groups}
            for name in set(own) | set(takes[i]):
                if name not in body.vertex_groups:
                    body.vertex_groups.new(name=name)
                    names = {g.index: g.name for g in body.vertex_groups}
                body.vertex_groups[name].add([index], own.get(name, 0.0) * (1 - k) + takes[i].get(name, 0.0) * k, "REPLACE")
        after = max(min((a.lerp(b, max(0.0, min(1.0, (body.data.vertices[k].co - a).dot(b - a) / max((b - a).length_squared, 1e-12))))
                         - body.data.vertices[k].co).length for a, b, _wa, _wb in rim) for k in ring_at)
        out.append((before, after))
    body.data.update()
    return out


def join(body, pieces) -> int:
    """The hands become part of the body mesh (a second material: their own texture). On a bare arm the two
    meet with nothing over the join, and they are still two surfaces there: each shades by its own normals, so
    the light breaks along the ring and the join shows as a line however well the colours match. The vertices
    of both on the ring are given one normal between them (the mean of those within a few millimetres), and
    the mesh keeps those normals. Returns how many vertices were given a shared normal."""
    from mathutils.kdtree import KDTree
    cuts = [(Vector(list(p["bare_cut"])[:3]), Vector(list(p["bare_cut"])[3:6]), list(p["bare_cut"])[6])
            for p in pieces if "bare_cut" in p.keys()]
    for o in bpy.context.view_layer.objects:
        o.select_set(o in pieces or o is body)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    if not cuts:
        return 0
    me = body.data
    normals = [v.normal.copy() for v in me.vertices]
    ring = [v.index for v in me.vertices
            if any(abs((v.co - at).dot(no)) < RING_NEAR and (v.co - at).length < 4 * radius for at, no, radius in cuts)]
    tree = KDTree(len(ring))
    for i, index in enumerate(ring):
        tree.insert(me.vertices[index].co, i)
    tree.balance()
    shared = {}
    for index in ring:
        total = Vector()
        for _co, i, _d in tree.find_range(me.vertices[index].co, RING_SHARE):
            total += me.vertices[ring[i]].normal
        if total.length > 1e-6:
            shared[index] = total.normalized()
    for index, n in shared.items():
        normals[index] = n
    me.normals_split_custom_set_from_vertices([tuple(n) for n in normals])
    me.update()
    return len(shared)


RING_NEAR = 0.0015   # a vertex this near a bare arm's cut plane (m) is on the ring where arm and hand meet
RING_SHARE = 0.004   # and shares its normal with the ring's vertices within this (m)
