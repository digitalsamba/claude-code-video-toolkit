"""A logo as a studio prop, and the beats that move a prop's parts (runs inside Blender).

    {"id": "mark", "kind": "logo", "file": "assets/mark.svg", "size": 0.3, "material": "satin",
     "extras": ["caret"], "parts": {"play": {"color": "#ffffff"}}}

Every filled shape of the SVG is its own part: extruded, bevelled, in its own fill colour (or the spec's), as its
own object, so a beat can move one of them (`"part": "mark.slash"`). The logo stands on the floor facing the
camera, its longest side `size` metres; `"lay": true` lays it on its back, and `"base": {"kind": "key"}` puts it
on a keycap. `extras` are parts the film needs and the finished mark does not have (a cursor that becomes
something): they are left out when the logo is measured and when its last frame is compared with the flat mark.

Colour: a part's face is its colour lit by the stage plus a little of its own light (`glow`), under the Standard
view, so a brand colour comes out as the brand's. Past a glow of 1 a saturated colour clips toward white, so
more light comes from a `flash` or the look's bloom, never from emission.

`perform()` reads the spec's beats into tracks (motion.py) and keys what they say: parts and whole props turn,
shear, grow, move, sink and are pressed; lights change, flash and cross the face; a hit shakes the camera.
"""
from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector, geometry

import svg
from kit import R, color, empty, hex_rgb, link, mesh_obj
from motion import GLOW_SAFE, Track, anchor, blink_frames, split_target

D = bpy.data
# what a part is made of: roughness, metal, clear coat, and how much of its colour it gives off by itself
MATERIALS = {
    "satin": dict(rough=0.35, glow=0.3),
    "matte": dict(rough=0.8, glow=0.3),
    "gloss": dict(rough=0.12, glow=0.25, coat=1.0),
    "metal": dict(rough=0.25, metal=1.0, glow=0.0),
    "flat":  dict(rough=1.0, glow=1.0, unlit=True),  # the colour exactly, whatever the lights do
}
BASES = ("key",)
HIDDEN = 0.02  # a part grown to less than this share is not drawn


def bsdf(m):
    return next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def part_material(name, hexc, preset, strength, glow=None):
    p = MATERIALS[preset]
    m = D.materials.new(name)
    m.use_nodes = True
    b = bsdf(m)
    rgb = (*hex_rgb(hexc), 1)
    b.inputs["Base Color"].default_value = (0, 0, 0, 1) if p.get("unlit") else rgb
    b.inputs["Roughness"].default_value = p["rough"]
    b.inputs["Metallic"].default_value = p.get("metal", 0.0)
    if p.get("unlit") and "Specular IOR Level" in b.inputs:
        b.inputs["Specular IOR Level"].default_value = 0.0
    if p.get("coat") and "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = p["coat"]
    b.inputs["Emission Color"].default_value = rgb
    m["glow"] = p["glow"] if glow is None else float(glow)
    m["strength"] = strength  # the film's exposure, cancelled, so a glow of 1 is the colour itself
    b.inputs["Emission Strength"].default_value = m["glow"] * strength
    return m


def extrude(name, rings, depth, m, parent, bevel):
    """Closed outlines in the XZ plane (the first the outside, any inside it holes) as a solid `depth` deep along
    +Y: its face looks at -Y, toward a camera in front of the stage."""
    flat = [p for r in rings for p in r]
    tris = geometry.tessellate_polygon([[Vector((x, z, 0.0)) for x, z in r] for r in rings])
    bm = bmesh.new()
    front = [bm.verts.new((x, 0.0, z)) for x, z in flat]
    back = [bm.verts.new((x, depth, z)) for x, z in flat]
    caps = []
    for a, b, c in tris:
        for vs in ((front[a], front[b], front[c]), (back[c], back[b], back[a])):
            try:
                caps.append(bm.faces.new(vs))
            except ValueError:  # a sliver the fill made twice
                pass
    i = 0
    for r in rings:
        n = len(r)
        for k in range(n):
            a, b = i + k, i + (k + 1) % n
            try:
                bm.faces.new((front[a], front[b], back[b], back[a]))
            except ValueError:
                pass
        i += n
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # the fill's triangles become one flat face where they can, so the bevel runs round the outline only
    inner = [e for e in bm.edges if len(e.link_faces) == 2 and all(f in caps for f in e.link_faces)]
    bmesh.ops.dissolve_limit(bm, angle_limit=0.002, edges=inner, verts=[])
    ob = mesh_obj(name, bm, m, parent, smooth=True)
    if hasattr(ob.data, "set_sharp_from_angle"):
        ob.data.set_sharp_from_angle(angle=R(40))
    if bevel > 0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method, mod.angle_limit = bevel, 3, "ANGLE", R(40)
        if hasattr(mod, "harden_normals"):
            mod.harden_normals = True
    return ob


class Mover:
    """One object a beat can move (a part, or a whole prop): its tracks, and the box its hinges are named on."""

    def __init__(self, ob, box, objects=None):
        self.ob, self.box = ob, box  # box: (x0, z0, x1, z1) in the object's parent space
        self.objects = objects or [ob]  # what a show or a hide covers
        self.turn = None    # (Track, (x, z) hinge, axis)
        self.grow = None    # (Track, (x, z) corner)
        self.move = [Track(), Track(), Track()]  # right, up, toward the viewer (m)
        self.sink = Track()
        self.shear = {}     # foot -> Track (tan of the angle)
        self.vis = []       # (frame, shown)

    def spot(self, name):
        u, v = anchor(name)
        x0, z0, x1, z1 = self.box
        return (x0 + (x1 - x0) * u, z0 + (z1 - z0) * v)

    def matrix(self, f) -> Matrix:
        m = Matrix.Translation((self.move[0].at(f), -self.move[2].at(f) + self.sink.at(f), self.move[1].at(f)))
        if self.turn:
            tr, (hx, hz), axis = self.turn
            a = R(tr.at(f))
            rot = {"x": Matrix.Rotation(a, 4, "X"), "y": Matrix.Rotation(-a, 4, "Z"), "z": Matrix.Rotation(-a, 4, "Y")}[axis]
            m = m @ Matrix.Translation((hx, 0, hz)) @ rot @ Matrix.Translation((-hx, 0, -hz))
        if self.grow:
            tr, (cx, cz) = self.grow
            g = max(tr.at(f), 0.001)
            m = m @ Matrix.Translation((cx, 0, cz)) @ Matrix.Diagonal((g, 1, g, 1)) @ Matrix.Translation((-cx, 0, -cz))
        return m

    def shown(self, f) -> bool:
        on = True
        for at, v in self.vis:
            if at <= f:
                on = v
        return on and not (self.grow and self.grow[0].at(f) < HIDDEN)

    def spans(self):
        tracks = [t for t in (*self.move, self.sink, *(self.shear.values()),
                              self.turn[0] if self.turn else None, self.grow[0] if self.grow else None) if t]
        return [t.span() for t in tracks]

    def keys(self, last):
        """Write the tracks as keyframes: a key on every frame of the stretch that moves."""
        spans = self.spans()
        ob = self.ob
        if spans:
            a, b = max(1, min(s[0] for s in spans)), min(last, max(s[1] for s in spans))
            rest = ob.matrix_basis.copy()
            for f in range(a, b + 1):
                ob.matrix_basis = rest @ self.matrix(f)
                for path in ("location", "rotation_euler", "scale"):
                    ob.keyframe_insert(path, frame=f)
            for foot, tr in self.shear.items():
                kb = shear_key(ob, foot)
                for f in range(a, b + 1):
                    kb.value = tr.at(f)
                    kb.keyframe_insert("value", frame=f)
        if self.vis or self.grow:
            was = None
            for f in range(1, last + 1):
                on = self.shown(f)
                if on != was:
                    for o in self.objects:
                        o.hide_render = o.hide_viewport = not on
                        o.keyframe_insert("hide_render", frame=f)
                        o.keyframe_insert("hide_viewport", frame=f)
                    was = on


def shear_key(ob, foot):
    """A shape key that leans the part by a tangent of 1 per unit of its value, with the `foot` edge staying put."""
    me = ob.data
    if not me.shape_keys:
        ob.shape_key_add(name="Basis")
    name = f"shear_{foot}"
    kb = me.shape_keys.key_blocks.get(name)
    if kb:
        return kb
    kb = ob.shape_key_add(name=name)
    kb.slider_min, kb.slider_max = -10.0, 10.0
    xs, zs = [v.co.x for v in me.vertices], [v.co.z for v in me.vertices]
    for v, k in zip(me.vertices, kb.data):
        x, y, z = v.co
        if foot == "bottom":
            k.co = (x + (z - min(zs)), y, z)
        elif foot == "top":
            k.co = (x + (z - max(zs)), y, z)
        elif foot == "left":
            k.co = (x, y, z + (x - min(xs)))
        else:
            k.co = (x, y, z + (x - max(xs)))
    return kb


def _world_box(objects):
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objects:
        if o.type == "MESH":
            ev = o.evaluated_get(dg)
            pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]


def build_logo(Prop, spec, base: Path, strength: float, accent: str):
    """The prop for a `logo` spec. `Prop` is studio.Prop (passed in: studio imports this module)."""
    pid = spec["id"]
    path = Path(spec["file"])
    path = path if path.is_absolute() else base / path
    if not path.exists():
        raise ValueError(f"prop {pid!r}: logo file not found: {path}")
    try:
        parts, _vb, notes = svg.read(path)
    except svg.SvgError as e:
        raise ValueError(f"prop {pid!r}: {e}")
    ids = [p.id for p in parts]
    extras = list(spec.get("extras") or [])
    over = spec.get("parts") or {}
    for name in [*extras, *over]:
        if name not in ids:
            raise ValueError(f'prop {pid!r}: {path.name} has no part called {name!r} (have: {", ".join(ids)})')
    mark = [p for p in parts if p.id not in extras]
    if not mark:
        raise ValueError(f"prop {pid!r}: every part is an extra, so there is no mark left")
    x0, y0, x1, y1 = svg.bounds(mark)
    size = float(spec.get("size", 0.3))
    k = size / max(x1 - x0, y1 - y0)
    w, h = (x1 - x0) * k, (y1 - y0) * k
    depth = float(spec.get("depth", size * 0.1))
    bevel = float(spec.get("bevel", depth * 0.08))
    preset = spec.get("material", "satin")
    lay = bool(spec.get("lay", False))
    plinth = spec.get("base")

    holder = empty(f"prop_{pid}_at")
    stand = empty(f"prop_{pid}_stand", holder)      # how it is put on the stage: standing, or on its back
    body = empty(f"prop_{pid}_body", stand)         # what a beat on the whole prop moves
    face = empty(f"prop_{pid}_face", body)          # the mark's own plane: x right, z up, the viewer at -y
    prop = Prop(pid, holder)
    prop.kind, prop.body, prop.face, prop.notes = "logo", body, face, notes
    prop.extras, prop.size, prop.depth, prop.parts, prop.materials = extras, size, depth, {}, {}
    prop.box = (-w / 2, 0.0, w / 2, h)
    prop.movers = {}

    def local(p):
        return ((p[0] - (x0 + x1) / 2) * k, (y1 - p[1]) * k)

    prop.rings, prop.colors = {}, {}
    for p in parts:
        o = over.get(p.id) or {}
        mat_name = o.get("material", preset)
        if mat_name not in MATERIALS:
            raise ValueError(f'prop {pid!r}: no material called {mat_name!r} (have: {", ".join(MATERIALS)})')
        col = o.get("color", p.fill)
        m = part_material(f"{pid}_{p.id}", col, mat_name, strength, o.get("glow"))
        rings = [[local(q) for q in r] for r in p.rings]
        ob = extrude(f"{pid}_{p.id}", rings, float(o.get("depth", depth)), m, face, bevel)
        xs, zs = [q[0] for r in rings for q in r], [q[1] for r in rings for q in r]
        prop.parts[p.id], prop.materials[p.id] = ob, m
        prop.rings[p.id], prop.colors[p.id] = rings, col
        prop.movers[p.id] = Mover(ob, (min(xs), min(zs), max(xs), max(zs)))

    lift = float(spec.get("lift", 0.0))
    solid = [prop.parts[p.id] for p in mark]
    if plinth is not None:
        kind = plinth.get("kind", "key")
        if kind not in BASES:
            raise ValueError(f'prop {pid!r}: no base called {kind!r} (have: {", ".join(BASES)})')
        t = max(w, h) * float(plinth.get("pad", 1.48)) / 2      # half the top of the cap
        b, tall, well = t * 1.12, t * 0.6, t * 0.2
        cz = h / 2
        dark = color(plinth.get("color", "#1b2440"))
        cap_m = D.materials.new(f"{pid}_cap")
        cap_m.use_nodes = True
        bs = bsdf(cap_m)
        bs.inputs["Base Color"].default_value = (*dark, 1)
        bs.inputs["Roughness"].default_value = float(plinth.get("rough", 0.32))
        y_top = depth * 0.98  # the parts stand a hair into the cap, so nothing shares a plane with it
        bm = bmesh.new()
        top = [bm.verts.new((sx * t, y_top, cz + sz * t)) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        bot = [bm.verts.new((sx * b, y_top + tall, cz + sz * b)) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        bm.faces.new(top)
        bm.faces.new(bot[::-1])
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((top[j], top[i], bot[i], bot[j]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        cap = mesh_obj(f"{pid}_cap", bm, cap_m, face, smooth=True)
        if hasattr(cap.data, "set_sharp_from_angle"):
            cap.data.set_sharp_from_angle(angle=R(40))
        mod = cap.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = t * 0.114, 6, "ANGLE"
        solid.append(cap)
        prop.cap, prop.travel = cap, well
        # the plate the key sits in: a frame round a hole the skirt goes down into when it is pressed
        gap, rim = b * 0.03, b * 0.45
        inner, outer = b + gap, b + gap + rim
        y0p, y1p = y_top + tall - well, y_top + tall + well * 0.2
        bm = bmesh.new()
        ring = lambda half, y: [bm.verts.new((sx * half, y, cz + sz * half))  # noqa: E731
                                for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        o_top, i_top, o_bot, i_bot = ring(outer, y0p), ring(inner, y0p), ring(outer, y1p), ring(inner, y1p)
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((o_top[i], o_top[j], i_top[j], i_top[i]))
            bm.faces.new((i_top[i], i_top[j], i_bot[j], i_bot[i]))
            bm.faces.new((o_top[j], o_top[i], o_bot[i], o_bot[j]))
        bm.faces.new(i_bot)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        plate_m = D.materials.new(f"{pid}_plate")
        plate_m.use_nodes = True
        bs = bsdf(plate_m)
        bs.inputs["Base Color"].default_value = (*[c * 0.55 for c in dark], 1)
        bs.inputs["Roughness"].default_value = 0.45
        plate = mesh_obj(f"{pid}_plate", bm, plate_m, stand, smooth=False)  # it stays put while the key goes down
        pm = plate.modifiers.new("bevel", "BEVEL")
        pm.width, pm.segments, pm.limit_method = b * 0.02, 2, "ANGLE"
        prop.plate = plate
        back = y_top + tall + well * 0.2
        if lay:
            stand.rotation_euler = (R(-90), 0, 0)
            stand.location = (0, -cz, back + lift)
        else:
            stand.location = (0, 0, outer - cz + lift)
    elif lay:
        stand.rotation_euler = (R(-90), 0, 0)
        stand.location = (0, -h / 2, depth + lift)
    else:
        stand.location = (0, 0, lift)
    bpy.context.view_layer.update()
    prop.rest = _world_box(solid + ([prop.plate] if plinth is not None else []))
    prop.mark_objects = [prop.parts[p.id] for p in mark]
    prop.rest_verts = {p.id: _verts(prop.parts[p.id]) for p in mark}
    prop.movers[None] = Mover(body, prop.box, [o for o in prop.objects() if o.type == "MESH" and o is not getattr(prop, "plate", None)])
    return prop


def _verts(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    return [ev.matrix_world @ v.co for v in ev.data.vertices]


def frame_of(prop):
    """A prop's own directions in the room, as it stands at rest: (centre, right, up, toward the viewer, width)."""
    ob = getattr(prop, "face", None) or prop.root
    rot = ob.matrix_world.to_3x3()
    right, up, out = ((rot @ Vector(v)).normalized() for v in ((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    corners = getattr(prop, "rest", None) or prop.corners()
    centre = (corners[0] + corners[7]) / 2
    if getattr(prop, "kind", "") == "logo":
        x0, z0, x1, z1 = prop.box
        centre = ob.matrix_world @ Vector(((x0 + x1) / 2, 0, (z0 + z1) / 2))
        return centre, right, up, out, x1 - x0
    width = max(abs((c - centre).dot(right)) for c in corners) * 2
    return centre, right, up, out, width


def mover_for(prop, part):
    """The Mover for a part of a logo, or for a whole prop (made on first use for a phone, a card or a model)."""
    movers = getattr(prop, "movers", None)
    if movers is None:
        movers = prop.movers = {}
    if part not in movers:
        if part is not None:
            raise ValueError(f"prop {prop.id!r} has no parts")
        body = empty(f"prop_{prop.id}_body", prop.root)
        for child in [c for c in prop.root.children if c is not body]:
            keep = child.matrix_world.copy()
            child.parent = body
            child.matrix_world = keep
        bpy.context.view_layer.update()
        inv = body.matrix_world.inverted()
        pts = [inv @ c for c in prop.corners()]
        box = (min(p.x for p in pts), min(p.z for p in pts), max(p.x for p in pts), max(p.z for p in pts))
        movers[None] = Mover(body, box, [o for o in prop.objects() if o.type == "MESH"])
    return movers[part]


def perform(room, beats, tl):
    """Turn a studio set's beats into keyframes. Returns (sound events, camera shakes as (frame, metres),
    notes for the person reading the check)."""
    fps, last = tl.fps, tl.frames - 1
    F = tl.frame
    events, shakes, notes = [], [], []
    glows, lights, extra_lights = {}, {}, []
    props = {pid: (list(p.parts) if getattr(p, "kind", "") == "logo" else None) for pid, p in room.props.items()}
    first = next(iter(room.props))
    key_energy = room.lights["key"][1]
    for b in sorted(beats, key=lambda b: tl.time(b["at"])):
        verb = b["do"]
        t0 = tl.time(b["at"])
        f0 = F(t0)
        timed = "until" in b or "dur" in b
        f1 = F(tl.time(b["until"])) if "until" in b else F(t0 + float(b.get("dur", 0.0)))
        ease = b.get("ease", "in_out")
        pid = part = prop = mv = None
        if "part" in b or verb in ("flash", "sweep", "hit"):
            pid, part = split_target(b.get("part", first), props)
            prop = room.props[pid]
        if verb in ("turn", "shear", "grow", "move", "sink", "press", "show", "hide", "blink"):
            mv = mover_for(prop, part)
        if verb == "turn":
            hinge = b.get("hinge", "centre")
            axis = b.get("axis", "auto")
            if axis == "auto":
                axis = "y" if hinge in ("left", "right") else "x" if hinge in ("top", "bottom") else "z"
            if mv.turn is None:
                mv.turn = (Track(), mv.spot(hinge), axis)
            mv.turn[0].to(f0, f1, b["to"], ease)
        elif verb == "shear":
            mv.shear.setdefault(b.get("foot", "bottom"), Track()).to(f0, f1, math.tan(R(b["to"])), ease)
        elif verb == "grow":
            if mv.grow is None:
                mv.grow = (Track(1.0), mv.spot(b.get("from", "centre")))
            mv.grow[0].to(f0, f1, b["to"], ease)
        elif verb == "move":
            for tr, v in zip(mv.move, b["to"]):
                tr.to(f0, f1, v, ease)
        elif verb == "sink":
            mv.sink.to(f0, f1, b["to"], ease)
        elif verb == "press":  # down at a steady rate, then (unless it stays) back up with a little spring
            mv.sink.to(f0, f1, b["depth"], b.get("ease", "linear"))
            back = float(b.get("back", 0.4))
            if back > 0:
                mv.sink.to(f1, F(tl.time(b["until"]) + back) if "until" in b else F(t0 + float(b.get("dur", 0.0)) + back),
                           0.0, "spring")
        elif verb in ("show", "hide"):
            mv.vis.append((f0, verb == "show"))
        elif verb == "blink":
            mv.vis += blink_frames(f0, f1, fps, float(b.get("every", 0.8)), float(b.get("off", 0.27)))
            mv.vis.append((f1, True))
        elif verb == "glow":
            if part is None and getattr(prop, "kind", "") != "logo":
                raise ValueError(f"glow: prop {pid!r} is not a logo, so it has no parts that give off light")
            for name in ([part] if part else [n for n in prop.parts if n not in prop.extras]):
                m = prop.materials[name]
                glows.setdefault(m.name, (m, Track(m["glow"])))[1].to(f0, f1, b["to"], ease)
        elif verb == "light":
            ob, energy = room.lights[b["light"]]
            lights.setdefault(b["light"], (ob, energy, Track(1.0)))[2].to(f0, f1, b["to"], ease)
        elif verb in ("flash", "sweep", "hit"):
            centre, right, up, out, width = frame_of(prop)
            if part:
                mvp = prop.movers[part]
                x0, z0, x1, z1 = mvp.box
                centre = prop.face.matrix_world @ Vector(((x0 + x1) / 2, 0, (z0 + z1) / 2))
            col = color(b["color"]) if "color" in b else (color(prop.colors[part]) if part else (1, 1, 1))
            if verb == "sweep":
                dur = (f1 - f0) if timed else max(2, round(0.73 * fps))
                sign = 1 if b.get("from", "left") == "left" else -1
                extra_lights.append(dict(kind="sweep", f0=f0, f1=f0 + dur, centre=centre, right=right * sign, up=up, out=out,
                                         width=width, energy=0.3 * key_energy * float(b.get("strength", 1.0)), color=col))
            else:
                strength = float(b.get("strength", 1.0)) if verb == "flash" else float(b.get("flash", 1.0))
                dur = (f1 - f0) if timed else max(2, round(0.5 * fps))
                if strength > 0:
                    extra_lights.append(dict(kind="flash", f0=f0, f1=f0 + dur, at=centre + out * width * 0.68,
                                             energy=0.45 * key_energy * strength, color=col, soft=width * 0.1))
                if verb == "hit" and float(b.get("shake", 0.014)) > 0:
                    shakes.append((f0, float(b.get("shake", 0.014)) * width, right, up))
        if verb == "hit" or "sound" in b:
            fs = F(tl.time(b["sound_at"])) if "sound_at" in b else (f1 if verb == "press" else f0)
            e = {"frame": fs, "kind": b.get("sound", "hit")}
            if verb == "hit":
                e["hit"] = True  # the render is checked: the picture's biggest change should be on this frame
            if verb == "sound" and timed:
                e["until"] = round((f1 - 1) / fps, 4)
            events.append(e)

    for prop in room.props.values():
        for mv in getattr(prop, "movers", {}).values():
            mv.keys(last)
    for m, tr in glows.values():
        sock = bsdf(m).inputs["Emission Strength"]
        a, b_ = tr.span()
        for f in range(max(1, a), min(last, b_) + 1):
            sock.default_value = tr.at(f) * m["strength"]
            sock.keyframe_insert("default_value", frame=f)
        if max(tr.at(f) for f in range(max(1, a), min(last, b_) + 1)) > GLOW_SAFE:
            notes.append(f"{m.name} glows past {GLOW_SAFE:g}: a saturated colour clips toward white there")
    for ob, energy, tr in lights.values():
        a, b_ = tr.span()
        for f in range(max(1, a), min(last, b_) + 1):
            ob.data.energy = energy * tr.at(f)
            ob.data.keyframe_insert("energy", frame=f)
    for i, x in enumerate(extra_lights):
        f0, f1 = x["f0"], min(last, x["f1"])
        if x["kind"] == "flash":
            d = D.lights.new(f"flash_{i}", "POINT")
            d.color = x["color"]
            d.shadow_soft_size = x["soft"]
            d.specular_factor = 0.0  # it lights the faces; a point light's own reflection in the floor is a blob
            ob = link(D.objects.new(f"flash_{i}", d))
            ob.location = x["at"]
            tau = max(1.0, (x["f1"] - x["f0"]) / 5)
            for f in range(max(1, f0 - 1), f1 + 2):
                d.energy = x["energy"] * math.exp(-(f - f0) / tau) if f0 <= f <= f1 else 0.0
                d.keyframe_insert("energy", frame=f)
        else:
            # a spot, not an area light: an area light is dark behind its own plane, which draws a hard line
            # across the floor as it travels
            d = D.lights.new(f"sweep_{i}", "SPOT")
            d.color, d.spot_size, d.spot_blend = x["color"], R(38), 0.9
            d.shadow_soft_size = x["width"] * 0.15
            ob = link(D.objects.new(f"sweep_{i}", d))
            aim = empty(f"sweep_{i}_aim", None, x["centre"])
            ob.constraints.new("TRACK_TO").target = aim
            w = x["width"]
            for f in range(max(1, f0 - 1), f1 + 2):
                t = min(1.0, max(0.0, (f - f0) / max(1, f1 - f0)))
                ob.location = x["centre"] + x["right"] * w * (-2.45 + 4.9 * t) + x["out"] * w * 1.36 + x["up"] * w * 0.4
                ob.keyframe_insert("location", frame=f)
                d.energy = x["energy"] * math.sin(t * math.pi) if f0 <= f <= f1 else 0.0
                d.keyframe_insert("energy", frame=f)
    return events, shakes, notes


def mark_report(prop, cam, frame):
    """How a logo stands at one frame against the flat mark: is each part where it was made, is anything that
    should be gone still there, does the lens face it, and where each part's outline falls in the picture (for
    the colour test on the rendered still)."""
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    sc.frame_set(frame)
    sc.camera = cam
    bpy.context.view_layer.update()
    centre, right, up, out, width = frame_of(prop)
    looking = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    facing = math.degrees(math.acos(max(-1.0, min(1.0, -looking.dot(out)))))
    seen_up = cam.matrix_world.to_3x3().inverted() @ up  # the mark's own up, as the picture has it
    roll = abs(math.degrees(math.atan2(seen_up.x, seen_up.y)))
    rep = {"frame": frame, "facing_deg": round(facing, 1), "roll_deg": round(roll, 1), "parts": [], "issues": []}
    if facing > 8:
        rep["issues"].append(f"the lens is {facing:.0f} degrees off square to the mark")
    if roll > 3:
        rep["issues"].append(f"the mark is rolled {roll:.0f} degrees in the frame")
    fm = prop.face.matrix_world
    for name, ob in prop.parts.items():
        shown = not ob.hide_render and min(abs(s) for s in ob.matrix_world.to_scale()) > HIDDEN * 0.5
        if name in prop.extras:
            if shown:
                rep["issues"].append(f"{name} is still on screen: the finished mark does not have it")
            continue
        now = _verts(ob)
        off = max((a - b).length for a, b in zip(now, prop.rest_verts[name])) / prop.size if shown else None
        outline = [[[round(v.x, 5), round(1 - v.y, 5)] for v in (world_to_camera_view(sc, cam, fm @ Vector((x, 0, z)))
                                                                 for x, z in ring)] for ring in prop.rings[name]]
        rep["parts"].append({"id": name, "color": prop.colors[name], "outline": outline,
                             "off": None if off is None else round(off, 4)})
        if not shown:
            rep["issues"].append(f"{name} is not on screen")
        elif off > 0.01:
            rep["issues"].append(f"{name} is {off:.0%} of the mark's size from where the flat mark has it")
    return rep
