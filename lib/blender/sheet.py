"""Review stills of a character in even light: blender -b scene.blend -P sheet.py -- views.json out_dir

`views.json` is a list of views: {"name", "frame", "on": "head" | "body", "az": degrees round the character
(0 in front, + toward their left), "elev": degrees, "dist": in heights of what is framed, "res": [w, h],
"alone": true to leave the set out}. Each is rendered as <name>.png, and marks.json gets, for each view, where
the character's own landmarks fall in the picture (pixels from the top left): the eyes, the box round the head,
the glasses, the body and the joints. Those are the model's side of a comparison with a photo; the photo's side
is read off the photo."""
import json
import math
import os
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
views, out = json.load(open(args[0], encoding="utf-8")), args[1]
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.timeline_markers.clear()
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
who = rig.name[:-len("_rig")]
root, head, body = (bpy.data.objects[f"{who}_{n}"] for n in ("root", "head", "body"))
own = {o for o in bpy.data.objects if o.name.startswith(who + "_")}
rest = [o for o in bpy.data.objects if o not in own and o.type in ("MESH", "FONT", "CURVE")]

# even light, no grading: colours are compared with a photo, and a room's mood light would be compared instead
for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
    bpy.data.objects.remove(o)
vs = sc.view_settings
vs.view_transform, vs.exposure, vs.gamma = "Standard", 0.0, 1.0
try:
    vs.look = "None"
except TypeError:
    pass
world = sc.world or bpy.data.worlds.new("sheet")
sc.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value, bg.inputs[1].default_value = (0.62, 0.64, 0.68, 1), 0.55
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
sc.render.resolution_percentage = 100
lights = []
for name, energy in (("key", 2.4), ("fill", 1.1), ("back", 1.2)):
    d = bpy.data.lights.new("sheet_" + name, "SUN")
    d.energy, d.angle = energy, math.radians(25)
    o = bpy.data.objects.new("sheet_" + name, d)
    sc.collection.objects.link(o)
    lights.append(o)
cd = bpy.data.cameras.new("sheet")
cd.clip_start = 0.01
cam = bpy.data.objects.new("sheet", cd)
sc.collection.objects.link(cam)
sc.camera = cam
groups = {g.index: g.name for g in body.vertex_groups}
head_group = next(i for i, n in groups.items() if n == "neck")
# the head is what goes with the neck bone and lies above its joint: the collar below it goes with the neck too
is_head = [any(g.group == head_group and g.weight > 0.6 for g in v.groups) for v in body.data.vertices]


def posed():
    """The body's vertices where the rig has put them on this frame, in the room, and those of the head."""
    dg = bpy.context.evaluated_depsgraph_get()
    ob = body.evaluated_get(dg)
    mw = ob.matrix_world
    pts = [mw @ v.co for v in ob.data.vertices]
    neck = rig.matrix_world @ rig.pose.bones["neck"].head
    up = (head.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
    return pts, [p for p, h in zip(pts, is_head) if h and (p - neck).dot(up) > 0.02]


def box(pts):
    return [Vector(c) for c in zip(*((min(p[i] for p in pts), max(p[i] for p in pts)) for i in range(3)))]


def pixel(p, w, h):
    v = world_to_camera_view(sc, cam, p)
    return [round(v.x * w, 1), round((1 - v.y) * h, 1)]


def span(pts, w, h):
    px = [pixel(p, w, h) for p in pts]
    return {"left": min(p[0] for p in px), "right": max(p[0] for p in px),
            "top": min(p[1] for p in px), "bottom": max(p[1] for p in px)}


marks = {}
for v in views:
    sc.frame_set(v["frame"])
    bpy.context.view_layer.update()
    for o in rest:
        o.hide_render = bool(v.get("alone", True))
    pts, head_pts = posed()
    lo, hi = box(head_pts if v["on"] == "head" else pts)
    mid, tall = (lo + hi) / 2, (hi - lo).z
    turn = root.matrix_world.to_3x3() if v["on"] == "body" else head.matrix_world.to_3x3()
    az, el = math.radians(v.get("az", 0)), math.radians(v.get("elev", 0))
    # a direction in the character's own frame: -y is ahead of them, +x their left
    way = (turn @ Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))).normalized()
    w, h = v.get("res", [720, 720])
    sc.render.resolution_x, sc.render.resolution_y = w, h
    dist = v.get("dist", 3.0) * tall
    cd.sensor_fit, cd.sensor_height = "VERTICAL", 24.0
    # the lens that makes what is framed `fill` of the picture's height from that distance
    cd.lens = 24.0 * dist * v.get("fill", 0.8) / tall
    cam.location = mid + way * dist
    cam.rotation_euler = (-way).to_track_quat("-Z", "Y").to_euler()
    for o, (side, lift) in zip(lights, ((-0.6, 0.5), (0.8, 0.25), (0.3, 0.9))):
        aim = (-way + cam.matrix_world.to_3x3() @ Vector((-side, -lift, 0))).normalized()
        o.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    lights[2].rotation_euler = way.to_track_quat("-Z", "Y").to_euler()  # from behind: an edge on the hair
    bpy.context.view_layer.update()
    sc.render.filepath = os.path.join(out, v["name"] + ".png")
    bpy.ops.render.render(write_still=True)
    bones = rig.pose.bones
    joint = {n: pixel(rig.matrix_world @ (bones[b].tail if end else bones[b].head), w, h)
             for n, b, end in (("shoulder_l", "upper.L", 0), ("shoulder_r", "upper.R", 0), ("elbow_l", "fore.L", 0),
                               ("elbow_r", "fore.R", 0), ("wrist_l", "hand.L", 0), ("wrist_r", "hand.R", 0),
                               ("hip_l", "thigh.L", 0), ("hip_r", "thigh.R", 0), ("knee_l", "shin.L", 0),
                               ("knee_r", "shin.R", 0), ("ankle_l", "foot.L", 0), ("ankle_r", "foot.R", 0),
                               ("neck", "neck", 0))}
    m = {"res": [w, h], "frame": v["frame"], "lens_mm": round(cd.lens, 1), "dist_m": round(dist, 3),
         "head": span(head_pts, w, h), "body": span(pts, w, h), "joints": joint,
         "head_m": [round(x, 4) for x in (hi - lo)] if v["on"] == "head" else None}
    eyes = [bpy.data.objects.get(f"{who}_eye{s}") for s in (-1, 1)]
    if all(eyes):
        m["eyes"] = [pixel(e.matrix_world.translation, w, h) for e in eyes]
    if body.get("face") and len(m.get("eyes", ())) == 2:
        # the chin's underside, as the fit has it, below the middle of the eyes (the head is level at rest)
        a, b = (e.matrix_world.translation for e in eyes)
        m["chin"] = pixel(Vector(((a.x + b.x) / 2, (a.y + b.y) / 2, json.loads(body["face"])["chin"])), w, h)
    frames = bpy.data.objects.get(f"{who}_frames")
    if frames:
        fm = frames.matrix_world
        m["glasses"] = span([fm @ Vector(c) for c in frames.bound_box], w, h)
    marks[v["name"]] = m
    print("SHEET", v["name"], flush=True)
json.dump(marks, open(os.path.join(out, "marks.json"), "w", encoding="utf-8"), indent=1)
print("SHEET_DONE")
