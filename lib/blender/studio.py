"""Studio: a product stage for shot specs (runs inside Blender). Set kind `studio`.

No room and no character needed: a backdrop curving from floor to wall, soft lights sized to what is on the stage, and one or
more props a camera can orbit:

  * `phone`: a phone standing on the stage with a picture or a video on its screen (a Remotion render),
  * `card`:  a thin floating panel showing a picture or a video: a UI screenshot or a scene with depth,
  * `model`: any glTF/GLB, OBJ, FBX or STL file, scaled to `size` metres and set down on the floor.

    "set": {"kind": "studio", "palette": {"backdrop": "#EEF1F6"},
            "props": [{"id": "app", "kind": "phone", "screen": {"video": "../out/app.mp4"}},
                      {"id": "mug", "kind": "model", "file": "assets/mug.glb", "size": 0.12, "at": [0.14, 0.02]}]}

A prop is a camera subject by its id (`"subject": "app"`); the shot types `product` and `orbit` frame it from
its size, so a 15 cm phone and a 2 m imported chair both fill the frame the same way. Everything is unlit-safe:
screens are emission, so their colours come out as made under the Standard view.
"""
from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

from kit import R, area_light, box, empty, mat, media, mesh_obj, quad

D = bpy.data
IMPORTERS = {".glb": "gltf", ".gltf": "gltf", ".obj": "obj", ".fbx": "fbx", ".stl": "stl"}
PROP_KINDS = ("phone", "card", "model")


class Prop:
    """Something on the stage: a root empty, the meshes under it, and (for a phone or a card) a screen."""

    def __init__(self, pid, root, screen=None):
        self.id, self.root, self.screen = pid, root, screen  # screen: (quad object, width, height)

    def objects(self) -> set:
        out, stack = set(), [self.root]
        while stack:
            o = stack.pop()
            out.add(o)
            stack.extend(o.children)
        return out

    def corners(self) -> list:
        """The eight corners of the box round every mesh of the prop, in world space, as it is posed now."""
        dg = bpy.context.evaluated_depsgraph_get()
        pts = []
        for o in self.objects():
            if o.type == "MESH":
                ev = o.evaluated_get(dg)
                pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        if not pts:
            raise ValueError(f"prop {self.id!r} has no geometry")
        lo = Vector([min(p[i] for p in pts) for i in range(3)])
        hi = Vector([max(p[i] for p in pts) for i in range(3)])
        return [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

    def bounds(self):
        """Centre and radius of the sphere round the prop."""
        c = self.corners()
        centre = (c[0] + c[7]) / 2
        return centre, (c[7] - c[0]).length / 2

    def screen_points(self) -> list:
        ob, w, h = self.screen
        return [ob.matrix_world @ Vector(p) for p in ((0, 0, 0), (w * 0.42, 0, h * 0.42), (-w * 0.42, 0, h * 0.42),
                                                      (w * 0.42, 0, -h * 0.42), (-w * 0.42, 0, -h * 0.42))]

    def screen_normal(self) -> Vector:
        return (self.screen[0].matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()


def media_aspect(path) -> float:
    img = D.images.load(str(path), check_existing=True)
    w, h = img.size
    return w / h if w and h else 16 / 9


def import_model(path: Path):
    """Bring a model file in and return its top-level objects."""
    kind = IMPORTERS.get(path.suffix.lower())
    if not kind:
        raise ValueError(f"cannot import {path.name}: use one of {', '.join(IMPORTERS)}")
    before = set(D.objects)
    if kind == "gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif kind == "obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        bpy.ops.wm.stl_import(filepath=str(path))
    new = [o for o in D.objects if o not in before]
    for o in new:  # imported lights and cameras would fight the stage's
        if o.type in ("LIGHT", "CAMERA"):
            D.objects.remove(o)
    new = [o for o in new if o.name in D.objects]
    if not any(o.type == "MESH" for o in new):
        raise ValueError(f"{path.name} has no mesh in it")
    return [o for o in new if o.parent not in new]


class Studio:
    def __init__(self, cfg: dict, base: Path, fps: int, start_frame, strength: float, backdrop: bool = True):
        p = {"backdrop": "#EEF1F6", "accent": "#3771E0", "device": "#1B1D22", **(cfg.get("palette") or {})}
        self.palette, self.events, self.props, self.fps = p, [], {}, fps
        self.m = {"backdrop": mat("backdrop", p["backdrop"], 0.9), "device": mat("device", p["device"], 0.35, 0.6),
                  "glass": mat("glass", (0.01, 0.01, 0.012), 0.08)}
        x = 0.0
        for i, spec in enumerate(cfg.get("props") or []):
            prop = self._prop(spec, base, start_frame, strength)
            at = spec.get("at")
            prop.root.location.x, prop.root.location.y = at if at else (x, 0.0)
            if "turn" in spec:
                prop.root.rotation_euler.z += R(float(spec["turn"]))
            bpy.context.view_layer.update()
            self.props[spec["id"]] = prop
            if not at:  # the next one stands beside it
                x += prop.bounds()[1] * 1.6
        if not self.props:
            raise ValueError('a studio set needs "props"')
        pts = [c for pr in self.props.values() for c in pr.corners()]
        lo = Vector([min(q[i] for q in pts) for i in range(3)])
        hi = Vector([max(q[i] for q in pts) for i in range(3)])
        self.centre, self.radius = (lo + hi) / 2, max((hi - lo).length / 2, 0.01)
        if backdrop:
            self._backdrop()
        self._lights()

    # ------------------------------------------------------------ props
    def _prop(self, spec, base, start_frame, strength) -> Prop:
        pid, kind = spec["id"], spec["kind"]
        root = empty(f"prop_{pid}")
        if kind == "model":
            path = Path(spec["file"])
            path = path if path.is_absolute() else base / path
            if not path.exists():
                raise ValueError(f"prop {pid!r}: model file not found: {path}")
            for o in import_model(path):
                o.parent = root
            prop = Prop(pid, root)
            bpy.context.view_layer.update()
            c = prop.corners()
            size = max((c[7] - c[0])[i] for i in range(3))
            if size <= 0:
                raise ValueError(f"prop {pid!r}: {path.name} has no geometry to show (it measures nothing)")
            root.scale = (float(spec.get("size", 0.3)) / size,) * 3  # its longest side becomes `size` metres
            bpy.context.view_layer.update()
            c = prop.corners()
            centre = (c[0] + c[7]) / 2
            root.location = (-centre.x, -centre.y, -c[0].z)  # centred, standing on the floor
            holder = empty(f"prop_{pid}_at")
            root.parent = holder
            return Prop(pid, holder)
        scr = spec.get("screen") or {}
        src = scr.get("video") or scr.get("image")
        path = None
        if src:
            path = Path(src) if Path(src).is_absolute() else base / src
            if not path.exists():
                raise ValueError(f"prop {pid!r}: screen media not found: {path}")
            if scr.get("video"):  # one frame of the video per scene frame, as on the laptop
                clip = D.movieclips.load(str(path))
                rate = clip.fps
                D.movieclips.remove(clip)
                if rate and abs(rate - self.fps) > 0.01:
                    raise ValueError(f'prop {pid!r}: screen video {path.name} is {rate:g} fps and the spec is '
                                     f'{self.fps} fps: render it at {self.fps} fps, or set "fps": {rate:g}')
        if kind == "phone":
            w = 0.068
            aspect = media_aspect(path) if path else 9 / 19.5
            if aspect > 1:  # a landscape picture: the phone lies on its side
                w = 0.148
            h, bezel, depth = w / aspect, 0.0035, 0.0085
            tilt, lift = R(-9), 0.0
        else:  # card
            w = float(spec.get("width", 0.6))
            h = w / (media_aspect(path) if path else 16 / 9)
            bezel, depth = 0.0, 0.012
            tilt, lift = R(-4), h * 0.12
        body = box(f"{pid}_body", (w + 2 * bezel, depth, h + 2 * bezel), self.m["device"], root,
                   (0, 0, (h + 2 * bezel) / 2 + lift), bevel=min(0.009, depth * 0.45))
        screen_mat = media(f"{pid}_screen", path, strength * float(scr.get("brightness", 1.0)),
                           start_frame(scr.get("at", 0)), int(scr.get("skip_frames", 0))) if path \
            else mat(f"{pid}_screen", self.palette["accent"], 0.4, emit=1.0)
        # the picture faces -Y, toward a camera in front of the stage (quad() faces +Y, so it is turned round)
        face = empty(f"{pid}_face", body, (0, -depth / 2 - 0.0006, 0), (0, 0, R(180)))
        panel = quad(f"{pid}_panel", -w / 2, w / 2, -h / 2, h / 2, screen_mat, face)
        root.rotation_euler = (tilt, 0, 0)  # leaning back a little, as a phone on a stand does
        return Prop(pid, root, (panel, w, h))

    # ------------------------------------------------------------ stage
    def _backdrop(self):
        """A sweep with no corner: floor, a curve, then a wall, wide enough that no shot sees its edge."""
        r = self.radius
        half, depth, curve, height = r * 40, r * 9, r * 6, r * 30
        prof = [(-r * 40, 0.0), (depth, 0.0)]
        prof += [(depth + curve * math.sin(a), curve - curve * math.cos(a))
                 for a in (math.pi / 2 * i / 16 for i in range(1, 17))]
        prof.append((depth + curve, height))
        bm = bmesh.new()
        rows = [[bm.verts.new((sx * half, y, z)) for sx in (-1, 1)] for y, z in prof]
        for a, b in zip(rows, rows[1:]):
            bm.faces.new((a[0], a[1], b[1], b[0]))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh_obj("backdrop", bm, self.m["backdrop"])

    def _lights(self):
        r = self.radius
        aim = empty("aim", None, self.centre)
        # an area light's reach falls with distance squared, so the energy grows with the stage's size squared
        k = (r / 0.1) ** 2
        area_light("key", self.centre + Vector((-4 * r, -6 * r, 6 * r)), 6 * k, 5 * r, (1, 0.96, 0.9), aim)
        area_light("fill", self.centre + Vector((6 * r, -5 * r, 3 * r)), 2.5 * k, 7 * r, (0.85, 0.9, 1.0), aim)
        area_light("rim", self.centre + Vector((2 * r, 6 * r, 5 * r)), 5 * k, 3 * r, (1, 1, 1), aim)

    def visible_objects(self) -> set:
        return {o for p in self.props.values() for o in p.objects()}

    def first_screen(self):
        return next((p for p in self.props.values() if p.screen), None)
