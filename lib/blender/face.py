"""A face for a fitted model (runs inside Blender).

The clay cast's mouth is a thin arc on a round head. A generated model has a face of its own: lips where it was
sculpted, often under a moustache, so an arc stuck on in front of it reads as a sticker. This module makes the
mouth from the model's own marks instead:

  * shape keys on the body, in the head only: `smile` lifts and spreads the corners of the mouth and the cheeks
    (a moustache goes with them, since it is the same surface), `frown` drops the corners, `jaw` lowers the chin
    and whatever beard is on it. A bearded face shows its mood in these more than in the lips.
  * one mouth object lying on the face under the lip line: a lower lip, the dark of an open mouth and a row of
    teeth, with its own keys (`smile`, `frown`, `open`, `round`). Closed, the lip covers the rest. With a
    moustache the upper edge tucks under the hair; without one an upper lip is made too.
  * eyelids: a shell over each eye that turns down to shut it, in the model's own skin colour, with a lash line.
    A shut eye is then a lid, not the white squashed to a dash.

Marks come from the fit (`face` in fit.json, see docs/blender.md): "mouth" is where the lips meet (the lower edge
of a moustache), "mouth_width" is corner to corner, "moustache" says the upper lip is under hair, "chin" is the
height of the chin's underside.
"""
from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from kit import R, arc_tube, empty, mat, mesh_obj

# how much of each key a face takes: smile, frown, open, round. `round` is a whole shape of its own, open already
# (keys add up, and a round mouth is not an open smile made narrower); the body's jaw key follows open + round
MOUTHS = {
    "neutral":    (0.25, 0.0, 0.0, 0.0),
    "happy":      (0.85, 0.0, 0.4, 0.0),
    "content":    (0.6, 0.0, 0.0, 0.0),
    "shocked":    (0.0, 0.0, 0.0, 1.0),
    "worried":    (0.0, 1.0, 0.15, 0.0),
    "determined": (0.0, 0.55, 0.0, 0.0),
    "relieved":   (0.5, 0.0, 0.2, 0.0),
    "grin":       (1.0, 0.0, 0.7, 0.0),
}
KEYS = ("smile", "frown", "open", "round")
JAW = 0.032        # how far the chin drops with the mouth fully open (m, on a head 0.42 m wide)
LID_OPEN, LID_SHUT = -52.0, 62.0  # degrees the lid shell is turned: resting above the eye, and down over it


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def body_keys(body, marks: dict) -> None:
    """Add `smile`, `frown` and `jaw` shape keys to a fitted body. `marks` is the fit's face, in the mesh's own
    space. Only the front of the head moves (the collar under the chin stays), and every field is smooth, so no
    triangle tears."""
    me = body.data
    if me.shape_keys and "smile" in me.shape_keys.key_blocks:
        return
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    cx, cy, cz = marks["mouth"]
    k = marks["half_width"] / 0.208  # the fields were made on a head this wide
    half = marks["mouth_width"] / 2
    chin = marks.get("chin", cz - 0.115 * k)
    head = np.zeros(len(co), dtype=np.float32)  # 1 where the head bone alone moves the vertex
    group = body.vertex_groups.get("neck")
    if group is not None:
        gi = group.index
        for v in me.vertices:
            for g in v.groups:
                if g.group == gi:
                    head[v.index] = g.weight
    else:
        head[:] = 1.0
    x, y, z = co[:, 0] - cx, co[:, 1] - cy, co[:, 2] - cz
    front = _smooth((0.13 * k - y) / (0.07 * k)) * _smooth((head - 0.9) / 0.1)  # the face, not the back of the head

    def bump(px, pz, sigma):
        return np.exp(-((x - px) ** 2 + (z - pz) ** 2) / (2 * (sigma * k) ** 2)) * front

    def field(corner, cheek=None):
        d = np.zeros_like(co)
        for s in (1, -1):
            g = bump(s * (half + 0.01 * k), 0.0, 0.032)
            d += g[:, None] * np.array([s * corner[0], corner[1], corner[2]], dtype=np.float32) * k
            if cheek:
                g = bump(s * (half + 0.05 * k), 0.045 * k, 0.04)
                d += g[:, None] * np.array([s * cheek[0], cheek[1], cheek[2]], dtype=np.float32) * k
        return d

    # the jaw: nothing above the lips, all of it from just under them to the chin, fading out below the chin
    down = _smooth(-z / (0.025 * k)) * _smooth((z - (chin - cz) + 0.04 * k) / (0.04 * k))
    wide = np.exp(-(x ** 2) / (2 * (0.075 * k) ** 2))
    jaw = np.zeros_like(co)
    jaw[:, 2] = -JAW * k * down * wide * front
    jaw[:, 1] = 0.25 * JAW * k * down * wide * front  # (a jaw swings: the chin goes back a little as it drops)
    body.shape_key_add(name="Basis")
    for name, d in (("smile", field((0.008, 0.003, 0.014), (0.003, -0.005, 0.008))),
                    ("frown", field((-0.004, 0.0, -0.012))), ("jaw", jaw)):
        kb = body.shape_key_add(name=name)
        kb.data.foreach_set("co", (co + d).ravel())
        kb.value = 0.0


def _surface(body, marks):
    """How far forward the face is under the lip line, across the mouth: y for an x (the mesh's own space)."""
    me = body.data
    cx, cy, cz = marks["mouth"]
    reach = marks["half_width"]
    verts = [v.co.copy() for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons
             if abs(verts[p.vertices[0]].z - cz) < reach * 0.6 and verts[p.vertices[0]].y < cy + reach * 0.6]
    tree = BVHTree.FromPolygons(verts, polys)
    below = cz - 0.012 * marks["half_width"] / 0.208
    xs = np.linspace(cx - reach, cx + reach, 81)
    ys = []
    for x in xs:
        hit = tree.ray_cast(Vector((x, cy - 1.0, below)), Vector((0, 1, 0)))
        ys.append(hit[0].y if hit[0] else float("nan"))
    ys = np.array(ys)
    good = ~np.isnan(ys)
    return lambda x: float(np.interp(x, xs[good], ys[good]))


def mouth(name, body, marks: dict, mats: dict, parent, to_parent):
    """The mouth object for a fitted face, with its keys. `to_parent` turns a point in the mesh's space into the
    parent's. Returns the object."""
    cx, cy, cz = marks["mouth"]
    k = marks["half_width"] / 0.208
    half = marks["mouth_width"] / 2
    surf = _surface(body, marks)
    cols, ring = 33, 8
    ts = [-1 + 2 * i / (cols - 1) for i in range(cols)]
    upper = not marks.get("moustache")

    def shape(smile=0.0, frown=0.0, open_=0.0, round_=0.0):
        """Every vertex, in one fixed order: lip rings, then the dark, the teeth, and an upper lip if there is one."""
        hw = half * (1 + 0.14 * smile - 0.10 * frown - 0.42 * round_)
        corner = cz + (-0.003 + 0.014 * smile - 0.014 * frown + 0.002 * round_) * k
        mid = cz + (-0.006 - 0.004 * smile + 0.005 * frown) * k - JAW * k * open_
        top = cz + 0.006 * k
        r0 = (0.0075 + 0.001 * open_) * k
        bend = 2 + 6 * round_  # a round mouth has a flat floor and steep sides: a U, where a smile is a bowl
        out = []
        line = []
        for t in ts:
            x = cx + hw * t
            a = abs(t)
            zl = mid + (corner - mid) * a ** bend
            r = r0 * math.sqrt(max(0.05, 1 - t * t))
            y = surf(x)
            line.append((x, y, zl, r))
            for j in range(ring):  # the lower lip: a tube, fuller in the middle, flatter front to back
                u = 2 * math.pi * j / ring
                out.append((x, y - 0.0035 * k + 0.7 * r * math.sin(u), zl + r * math.cos(u)))
        for (x, y, zl, r), t in zip(line, ts):  # the dark of the mouth, from under the upper lip to the lower
            zu = top + (corner - top) * abs(t) ** 3
            out += [(x, y - 0.0012 * k, max(zu, zl)), (x, y - 0.0012 * k, zl)]
        for (x, y, zl, r), t in zip(line, ts):  # upper teeth: a band under the upper lip, hidden until the lip drops
            a = min(1.0, abs(t) / 0.82)
            zu = top + (corner - top) * abs(t) ** 3
            zb = min(zu, cz - 0.011 * k * (1 - a ** 3))
            zb += (zu - zb) * min(1.0, round_)  # (a round mouth shows no teeth)
            out += [(x, y - 0.002 * k, zu), (x, y - 0.002 * k, max(zb, zl) if a < 1 else zu)]
        if upper:
            for (x, y, zl, r), t in zip(line, ts):
                zu = cz + 0.004 * k + (corner - cz - 0.004 * k) * abs(t) ** 2 + 0.002 * k * round_ * (1 - t * t)
                for j in range(ring):
                    u = 2 * math.pi * j / ring
                    out.append((x, y - 0.0035 * k + 0.6 * r * math.sin(u), zu + 0.8 * r * math.cos(u)))
        return [to_parent(Vector(p)) for p in out]

    base = shape()
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in base]
    lip0, dark0, teeth0, up0 = 0, cols * ring, cols * ring + cols * 2, cols * ring + cols * 4
    slots = []  # (face, material slot)
    for i in range(cols - 1):
        for j in range(ring):
            a, b = lip0 + i * ring, lip0 + (i + 1) * ring
            slots.append((bm.faces.new((vs[a + j], vs[a + (j + 1) % ring], vs[b + (j + 1) % ring], vs[b + j])), 0))
            if upper:
                a, b = up0 + i * ring, up0 + (i + 1) * ring
                slots.append((bm.faces.new((vs[a + j], vs[a + (j + 1) % ring], vs[b + (j + 1) % ring], vs[b + j])), 0))
        for start, slot in ((dark0, 1), (teeth0, 2)):
            a, b = start + i * 2, start + (i + 1) * 2
            slots.append((bm.faces.new((vs[a], vs[a + 1], vs[b + 1], vs[b])), slot))
    for face, slot in slots:
        face.material_index = slot
    bmesh.ops.recalc_face_normals(bm, faces=[f for f, s in slots if s == 0])
    ob = mesh_obj(name, bm, None, parent)
    for m in (mats["lip"], mats["mouthin"], mats["teeth"]):
        ob.data.materials.append(m)
    ob.visible_shadow = False
    ob.shape_key_add(name="Basis")
    for name_, args in (("smile", (1, 0, 0, 0)), ("frown", (0, 1, 0, 0)), ("open", (0, 0, 1, 0)), ("round", (0, 0, 0.9, 1))):
        kb = ob.shape_key_add(name=name_)
        for v, p in zip(kb.data, shape(*args)):
            v.co = p
        kb.value = 0.0
    return ob


def skin_colour(body, point) -> tuple:
    """The model's own colour at a point on it (linear RGB): its texture, at the vertex nearest the point."""
    me = body.data
    img = next(n.image for n in me.materials[0].node_tree.nodes
               if n.type == "TEX_IMAGE" and n.outputs[0].links and n.outputs[0].links[0].to_socket.name == "Base Color")
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    near = set(np.argsort(((co - np.array(point, dtype=np.float32)) ** 2).sum(axis=1))[:40].tolist())
    uv = me.uv_layers.active.data
    got = []
    for loop in me.loops:
        if loop.vertex_index in near:
            u, v = uv[loop.index].uv
            got.append(px[min(h - 1, int(v % 1 * h)), min(w - 1, int(u % 1 * w)), :3])
    c = np.median(np.array(got), axis=0)
    if img.colorspace_settings.name == "sRGB":
        c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return tuple(float(x) for x in c)


def eyelid(name, eye, skin, lash):
    """A lid over an eye (an empty the eye white is built on, 0.075 in radius, flattened front to back): a half
    shell that turns about the eye's own axis, from resting above it to down over it. Returns the pivot to turn."""
    squash = empty(f"{name}_lid_at", eye)
    squash.scale = (1.0, 1.27, 1.0)  # (deep enough to close over the pupil and its glint, which stand proud of the white)
    pivot = empty(f"{name}_lid", squash, rot=(R(LID_OPEN), 0, 0))
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=0.0795)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -1e-6], context="VERTS")
    shell = mesh_obj(f"{name}_lid_shell", bm, skin, pivot)
    shell.visible_shadow = False
    line = arc_tube(f"{name}_lash", 0.0795, 0.0045, R(22), R(158), lash, pivot, (0, 0, 0), seg=20, ring=6)
    line.rotation_euler = (R(90), 0, 0)  # the arc lies along the shell's front edge
    line.visible_shadow = False
    return pivot


def materials(cid, body, marks) -> dict:
    # the model's skin, for the lids: from the cheek under an eye (the eye sockets themselves were painted over)
    eye = marks["eyes"][0]
    between = [eye[0], eye[1] + 0.01, eye[2] - 1.6 * marks["eye_size"][1]]
    return {"lip": mat(f"{cid}_lip", (0.36, 0.13, 0.11), 0.45), "teeth": mat(f"{cid}_teeth", (0.86, 0.84, 0.78), 0.3),
            "lid": mat(f"{cid}_lid", skin_colour(body, between), 0.55), "lash": mat(f"{cid}_lash", (0.05, 0.035, 0.03), 0.6)}


MOUTH_SHARE = 0.18  # a mouth narrower than this share of the face's width is too small to read in a close-up
MOUTH_OFF = 0.02    # and one whose lip stands further than this off the face (m, on a head 0.42 m wide) floats


def report(body, mouth_ob, marks: dict, to_mesh) -> dict:
    """What `check` says about a fitted face: how wide the mouth is against the face, how far its lip stands off
    the model's surface, and how far its corners travel between a frown and a smile."""
    k = marks["half_width"] / 0.208
    share = marks["mouth_width"] / (2 * marks["half_width"])
    lip = [to_mesh(v.co) for v in mouth_ob.data.vertices[:33 * 8]]
    me = body.data  # (the mesh as fitted, whatever pose the scene is in)
    near = [i for i, v in enumerate(me.vertices) if (v.co - Vector(marks["mouth"])).length < 2 * marks["mouth_width"]]
    index = {v: i for i, v in enumerate(near)}
    tree = BVHTree.FromPolygons([me.vertices[i].co for i in near],
                                [[index[v] for v in pl.vertices] for pl in me.polygons
                                 if all(v in index for v in pl.vertices)])
    off = max((tree.find_nearest(p)[0] - p).length for p in lip)
    keys = mouth_ob.data.shape_keys.key_blocks
    travel = max((to_mesh(a.co) - to_mesh(b.co)).length for a, b in zip(keys["smile"].data, keys["frown"].data))
    out = {"mouth_share": round(share, 3), "mouth_off_m": round(off, 4), "mouth_travel_m": round(travel, 4)}
    problems = []
    if share < MOUTH_SHARE:
        problems.append(f"the mouth is {share:.0%} of the face's width, too small to read: give the fit a "
                        f"mouth_width from corner to corner (at least {MOUTH_SHARE:.0%} of the face)")
    if off > MOUTH_OFF * k:
        problems.append(f"the mouth stands {off * 100:.1f} cm off the face: put the fit's mouth mark where the lips meet")
    if problems:
        out["problems"] = problems
    return out
