"""Studio: a product stage for shot specs (runs inside Blender). Set kind `studio`.

No room and no character needed: a backdrop curving from floor to wall, soft lights sized to what is on the stage, and one or
more props a camera can orbit:

  * `phone`: a phone standing on the stage with a picture or a video on its screen (a Remotion render),
  * `card`:  a thin floating panel showing a picture or a video: a UI screenshot or a scene with depth,
  * `model`: any glTF/GLB, OBJ, FBX or STL file, scaled to `size` metres and set down on the floor,
  * `logo`:  an SVG, each filled shape its own extruded part that beats can move (logo.py).

`"stage": "dark"` swaps the sweep for a dark, glossy floor with no wall: the stage an ident plays on.

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

from kit import R, area_light, box, color, empty, mat, media, mesh_obj, quad

D = bpy.data
IMPORTERS = {".glb": "gltf", ".gltf": "gltf", ".obj": "obj", ".fbx": "fbx", ".stl": "stl"}
WORLD = 0.55  # the world is lit by the backdrop's colour at this strength (compile.py sets it)
PROP_KINDS = ("phone", "card", "model", "logo")
STAGES = ("sweep", "dark")


class Prop:
    """Something on the stage: a root empty, the meshes under it, and (for a phone or a card) a screen."""

    kind = "prop"

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
        self.stage = cfg.get("stage", "sweep")
        # how matt the dark stage's floor is under the prop: 0.22 holds a reflection of something standing on it;
        # nearer 0.6 for a prop seen from above, where a glossy floor shows the lights themselves
        self.floor_rough = float((cfg.get("floor") or {}).get("rough", 0.22))
        self.light_share = {"key": 1.0, "fill": 1.0, "rim": 1.0, **(cfg.get("lights") or {})}
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
            self._floor() if self.stage == "dark" else self._backdrop()
        self._lights()

    # ------------------------------------------------------------ props
    def _prop(self, spec, base, start_frame, strength) -> Prop:
        pid, kind = spec["id"], spec["kind"]
        if kind == "logo":
            from logo import build_logo
            return build_logo(Prop, spec, base, strength, self.palette["accent"])
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

    def _floor(self):
        """The dark stage: one glossy floor in the backdrop's colour and nothing behind it, so a prop stands on
        its own reflection and the world's colour is the wall."""
        r = self.radius
        # glossy under the prop and matt far off: a far floor seen edge-on picks up stray reflections of the prop
        # (screen-space tracing), and a matt floor is not traced
        nt = self.m["backdrop"].node_tree
        coord, length, ramp = nt.nodes.new("ShaderNodeTexCoord"), nt.nodes.new("ShaderNodeVectorMath"), nt.nodes.new("ShaderNodeMapRange")
        length.operation = "LENGTH"
        ramp.inputs["From Min"].default_value, ramp.inputs["From Max"].default_value = 5 * r, 14 * r
        ramp.inputs["To Min"].default_value, ramp.inputs["To Max"].default_value = self.floor_rough, 0.9
        nt.links.new(coord.outputs["Object"], length.inputs[0])
        nt.links.new(length.outputs["Value"], ramp.inputs["Value"])
        nt.links.new(ramp.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Roughness"])
        # and far off it goes into the world's own colour, so there is no line where the floor ends
        fade = nt.nodes.new("ShaderNodeMapRange")
        fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = 5 * r, 14 * r
        nt.links.new(length.outputs["Value"], fade.inputs["Value"])
        nt.links.new(fade.outputs["Result"], nt.nodes["Principled BSDF"].inputs["Emission Strength"])
        nt.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (*[c * WORLD for c in color(self.palette["backdrop"])], 1)
        bm = bmesh.new()
        bm.faces.new([bm.verts.new((sx * r * 60, sy * r * 60, 0.0)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        floor = mesh_obj("floor", bm, self.m["backdrop"], smooth=False)
        floor.location = (self.centre.x, self.centre.y, 0.0)
        opts = getattr(bpy.context.scene.eevee, "ray_tracing_options", None)
        if opts is not None:  # how thick the tracer takes things to be: its default is for a room, not a 30 cm logo
            opts.screen_trace_thickness = max(0.002, r * 0.03)
            opts.screen_trace_quality = 0.5

    def _lights(self):
        r = self.radius
        aim = empty("aim", None, self.centre)
        # an area light's reach falls with distance squared, so the energy grows with the stage's size squared
        k = (r / 0.1) ** 2
        dark = self.stage == "dark"  # white lights on a dark stage: a tint would shift a brand colour
        # (the dark stage's key is set so a satin part's face reads as its own colour: the mark test measures it)
        made = {"key": (self.centre + Vector((-4 * r, -6 * r, 6 * r)), (10 if dark else 6) * k, 5 * r,
                        (1, 1, 1) if dark else (1, 0.96, 0.9)),
                "fill": (self.centre + Vector((6 * r, -5 * r, 3 * r)), (0.8 if dark else 2.5) * k, 7 * r,
                         (1, 1, 1) if dark else (0.85, 0.9, 1.0)),
                "rim": (self.centre + Vector((2 * r, 6 * r, 5 * r)), (0.0 if dark else 5) * k, 3 * r, (1, 1, 1))}
        self.lights = {}  # name -> (the light, its energy as built): a `light` beat scales it
        for name, (at, energy, size, tint) in made.items():
            energy *= float(self.light_share[name])
            self.lights[name] = (area_light(name, at, energy, size, tint, aim), energy)

    def visible_objects(self) -> set:
        return {o for p in self.props.values() for o in p.objects()}

    def first_screen(self):
        return next((p for p in self.props.values() if p.screen), None)
