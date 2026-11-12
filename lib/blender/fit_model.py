"""Fit a generated character model to the ClayPerson rig (runs inside Blender). Experimental.

    uv run tools/blender_render.py fit --model puppet.glb --fit fit.json --out fitted.blend --preview dir

The input is one textured mesh of a standing character in an A-pose, facing -Y, feet on the ground (what an
image-to-3D service returns from a front view). The output is a .blend holding `model_body`: the same mesh, still
standing in its A-pose, with a vertex group per rig bone, the head on the neck bone, where its joints are and where
its face is. A cast entry names that file with "model"; the rig is built on those joints and its bones pose the
model (sitting included), so every verb of the rig drives it.

How: a clean clay body is baked on the model's own joints and weighted by bone heat (which fails on generated
meshes: they are triangle soup with open seams); its weights are copied to the model by nearest surface.

The model keeps its own shape and proportions: it is only scaled as a whole (`scale`) and stood on the floor.
The head is everything that is not clothing above `clothes.below`: clothing is told by colour, so a beard that
hangs over a collar stays with the head. The model's painted eyes and brows are erased (texture filled with the
skin round them, surface smoothed), because the rig's own eyes and brows are placed there and move.

fit.json (metres, in the model's own space; side joints once, for the character's left, measured from centre_x):
    {"centre_x": 0.015, "scale": 1.3, "faces": 150000, "gamma": 0.8,
     "joints": {"pelvis": [x, y, z], "neckb", "necktop", "sh", "el", "wr", "tip", "hip", "kn", "an", "toe"},
     "clothes": {"below": 1.30, "above": 1.45, "sample": [x0, x1, z0, z1], "tolerance": 0.12, "dark": 0.05},
     "face": {"eyes": [[x, z], [x, z]], "eye_size": [w, h], "brows": [[x, z], [x, z]], "brow_size": [w, h],
              "mouth": [x, z], "mouth_width": w, "half_width": w, "moustache": true, "chin": z}}
Eyes and brows are the character's right, then left, as seen from the front.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import hands as hand_graft  # noqa: E402
from character import skeleton  # noqa: E402

D = bpy.data
BONES = [("spine", "pelvis", "neckb"), ("neck", "neckb", "necktop")]
for _sd in "LR":
    BONES += [("upper." + _sd, "sh" + _sd, "el" + _sd), ("fore." + _sd, "el" + _sd, "wr" + _sd),
              ("hand." + _sd, "wr" + _sd, "tip" + _sd), ("thigh." + _sd, "hip" + _sd, "kn" + _sd),
              ("shin." + _sd, "kn" + _sd, "an" + _sd), ("foot." + _sd, "an" + _sd, "toe" + _sd)]
PARENT = {"neck": "spine", "upper": "spine", "fore": "upper", "hand": "fore", "thigh": "hips", "shin": "thigh",
          "foot": "shin"}


def only(*obs):
    bpy.context.view_layer.update()  # (an object removed since the last update is still listed, as None)
    for o in bpy.context.view_layer.objects:
        if o:
            o.select_set(False)
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[-1]


def images_of(ob):
    return [n.image for m in ob.data.materials for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]


def base_colour(ob):
    return next(n.image for n in ob.data.materials[0].node_tree.nodes
                if n.type == "TEX_IMAGE" and n.outputs[0].links and n.outputs[0].links[0].to_socket.name == "Base Color")


def pixels_of(img):
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def front_y(ob, x, z):
    """Where a ray from the front, at (x, z), meets the model."""
    hit, at, _, _ = ob.ray_cast(Vector((x, -10, z)), Vector((0, 1, 0)))
    if not hit:
        raise SystemExit(f"fit: nothing on the model at x {x}, z {z}")
    return at.y


def erase(ob, shapes):
    """Take painted features off the model: `shapes` are ellipses (x, z, rx, rz) on its front. Their texels take
    the median of the ring round each, in every texture, and the surface under them is smoothed flat."""
    me = ob.data
    uv = me.uv_layers.active.data
    imgs = [(img, pixels_of(img)) for img in images_of(ob)]
    bm_verts = set()
    for x, z, rx, rz in shapes:
        y0 = front_y(ob, x, z)
        inside, ring = [], []
        for poly in me.polygons:
            c = poly.center
            if c.y > y0 + 0.04:
                continue
            d = ((c.x - x) / rx) ** 2 + ((c.z - z) / rz) ** 2
            if d <= 1:
                inside.append(poly)
            elif d <= 2.2:
                ring.append(poly)
        for img, px in imgs:
            h, w = px.shape[:2]
            at = np.array([[uv[li].uv[0] * w, uv[li].uv[1] * h] for poly in ring for li in poly.loop_indices])
            fill = np.median(px[np.clip(at[:, 1].astype(int), 0, h - 1), np.clip(at[:, 0].astype(int), 0, w - 1)], axis=0)
            for poly in inside:
                t = np.array([[uv[li].uv[0] * w, uv[li].uv[1] * h] for li in poly.loop_indices])
                x0, x1 = int(max(0, t[:, 0].min() - 2)), int(min(w - 1, t[:, 0].max() + 2))
                z0, z1 = int(max(0, t[:, 1].min() - 2)), int(min(h - 1, t[:, 1].max() + 2))
                px[z0:z1 + 1, x0:x1 + 1] = fill  # (the box round a small triangle: it pads the seams too)
        bm_verts.update(v for poly in inside for v in poly.vertices)
    for img, px in imgs:
        img.pixels.foreach_set(px.ravel())
        img.update()
        img.pack()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in bm_verts]
    for _ in range(30):
        bmesh.ops.smooth_vert(bm, verts=verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(me)
    bm.free()


def load_donor(path: Path, centre_x: float | None):
    """A second model of the same character, as made (for its hands): one mesh, centred the same way."""
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"{path} has no mesh")
    only(*meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in new:
        if o is not ob:
            D.objects.remove(o)
    xs = [v.co.x for v in ob.data.vertices]
    ob.data.transform(Matrix.Translation((-(centre_x if centre_x is not None else (min(xs) + max(xs)) / 2), 0, 0)))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


def face_shapes(face):
    m = 1.35  # a margin: the painted edge is soft
    out = [(x, z, face["eye_size"][0] / 2 * m, face["eye_size"][1] / 2 * m) for x, z in face["eyes"]]
    return out + [(x, z, face["brow_size"][0] / 2 * 1.15, face["brow_size"][1] / 2 * 1.9) for x, z in face["brows"]]


def load_model(path: Path, centre_x: float, faces: int, face: dict):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"{path} has no mesh")
    only(*meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.parent = None
    ob.location.x -= centre_x
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o is not ob:
            D.objects.remove(o)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)  # the importer splits vertices at UV seams
    bpy.ops.object.mode_set(mode="OBJECT")
    marks = {"eyes": [(x - centre_x, front_y(ob, x - centre_x, z), z) for x, z in face["eyes"]],
             "brows": [(x - centre_x, front_y(ob, x - centre_x, z), z) for x, z in face["brows"]],
             "mouth": (face["mouth"][0] - centre_x, front_y(ob, face["mouth"][0] - centre_x, face["mouth"][1]),
                       face["mouth"][1])}
    erase(ob, [(x - centre_x, z, rx, rz) for x, z, rx, rz in face_shapes(face)])
    if len(ob.data.polygons) > faces:
        dec = ob.modifiers.new("Decimate", "DECIMATE")
        dec.ratio = faces / len(ob.data.polygons)
        bpy.ops.object.modifier_apply(modifier=dec.name)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    ob.name = "model_body"
    return ob, marks


def head_mask(body, clothes):
    """1 for a vertex of the head, 0 for the body. Above `above` everything is head and below `below` nothing is;
    between them a vertex in the clothes' colour (sampled on the chest) or near black is body."""
    me = body.data
    px = pixels_of(base_colour(body))
    h, w = px.shape[:2]
    uv = me.uv_layers.active.data
    colour = np.zeros((len(me.vertices), 3), dtype=np.float32)
    for loop in me.loops:
        u, v = uv[loop.index].uv
        colour[loop.vertex_index] = px[min(h - 1, int(v % 1 * h)), min(w - 1, int(u % 1 * w)), :3]
    co = np.array([v.co for v in me.vertices])
    x0, x1, z0, z1 = clothes["sample"]
    chest = (co[:, 0] > x0) & (co[:, 0] < x1) & (co[:, 2] > z0) & (co[:, 2] < z1) & (co[:, 1] < 0)
    cloth = np.median(colour[chest], axis=0)
    is_cloth = (np.linalg.norm(colour - cloth, axis=1) < clothes.get("tolerance", 0.12)) | \
               (colour.max(axis=1) < clothes.get("dark", 0.05))
    mask = np.where(co[:, 2] > clothes["above"], 1.0, np.where(co[:, 2] < clothes["below"], 0.0, 1.0 - is_cloth))
    edges = np.array([e.vertices for e in me.edges])
    for _ in range(3):  # soften the line a little, so the join stretches over a few triangles and not one
        total, count = mask.copy(), np.ones(len(mask))
        np.add.at(total, edges[:, 0], mask[edges[:, 1]])
        np.add.at(total, edges[:, 1], mask[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1)
        np.add.at(count, edges[:, 1], 1)
        mask = total / count
    print("fit: clothes colour", [round(float(c), 3) for c in cloth], "head vertices", int((mask > 0.5).sum()))
    return mask


def source_joints(fit: dict) -> dict:
    """The model's joints in its A-pose, named as the rig's skeleton names them."""
    j = {k: Vector(v) for k, v in fit["joints"].items()}
    out = {"pelvis": j["pelvis"], "neckb": j["neckb"], "necktop": j["necktop"]}
    for name, t in (("waist", 0.28), ("chest", 0.59), ("upch", 0.82)):
        out[name] = j["pelvis"].lerp(j["neckb"], t)
    for s, sd in ((1, "L"), (-1, "R")):
        for k in ("sh", "el", "wr", "tip", "hip", "kn", "an", "toe"):
            out[k + sd] = Vector((j[k].x * s, j[k].y, j[k].z))
        out["th" + sd] = out["wr" + sd].lerp(out["tip" + sd], 0.5) + Vector((0, -0.03, 0))
    return out


def clay_proxy(src: dict):
    """A clean body on the model's joints, weighted by bone heat: the source of the model's weights."""
    SK, edges = skeleton()
    names = list(SK)
    idx = {n: i for i, n in enumerate(names)}
    scale = (Vector(SK["neckb"][0]) - Vector(SK["pelvis"][0])).length / (src["neckb"] - src["pelvis"]).length
    me = D.meshes.new("proxy_skel")
    me.from_pydata([src[n] for n in names], [(idx[a], idx[b]) for a, b in edges], [])
    skel = D.objects.new("proxy_skel", me)
    bpy.context.scene.collection.objects.link(skel)
    skm = skel.modifiers.new("Skin", "SKIN")
    skm.branch_smoothing = 0.4
    for name in names:
        rx, ry = SK[name][1]
        me.skin_vertices[0].data[idx[name]].radius = (rx / scale, ry / scale)
    me.skin_vertices[0].data[idx["pelvis"]].use_root = True
    skel.modifiers.new("Pre", "SUBSURF").levels = 2
    proxy_me = D.meshes.new_from_object(skel.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    D.objects.remove(skel)
    proxy = D.objects.new("proxy", proxy_me)
    bpy.context.scene.collection.objects.link(proxy)

    arm = D.armatures.new("proxy_rig")
    rig = D.objects.new("proxy_rig", arm)
    bpy.context.scene.collection.objects.link(rig)
    only(rig)
    bpy.ops.object.mode_set(mode="EDIT")
    b = arm.edit_bones.new("hips")
    b.head, b.tail = src["pelvis"], src["pelvis"] + Vector((0, 0, -0.12))
    for name, h, t in BONES:
        b = arm.edit_bones.new(name)
        b.head, b.tail = src[h], src[t]
        parent = PARENT.get(name.split(".")[0])
        if parent:
            b.parent = arm.edit_bones[parent + name[-2:] if parent not in ("spine", "hips") else parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    only(proxy, rig)
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    return proxy, rig


def copy_weights(proxy, body, src, reach=0.25, smooth=8):
    """The proxy's weights, onto the model. Each model vertex takes them from the nearest proxy face that looks
    the same way: the inside of a sleeve and the side of the chest are close together in an A-pose but face each
    other, and plain nearest-surface copying swaps them (which tears the armpit when the arm lifts). The result
    An arm then only keeps what lies inside its own sleeve (a tube round shoulder, elbow and wrist, as thick as
    the model's arm measures): a bulky chest is nearer the proxy's arm than its thin chest, and would be pulled
    about by it. The result is smoothed along the model's own edges, so a joint bends over a band of triangles
    and not a line."""
    from mathutils.bvhtree import BVHTree
    pm, me = proxy.data, body.data
    names = [g.name for g in proxy.vertex_groups]
    pw = np.zeros((len(pm.vertices), len(names)), dtype=np.float32)
    for v in pm.vertices:
        for g in v.groups:
            pw[v.index, g.group] = g.weight
    polys = [list(poly.vertices) for poly in pm.polygons]
    tree = BVHTree.FromPolygons([v.co for v in pm.vertices], polys)
    w = np.zeros((len(me.vertices), len(names)), dtype=np.float32)
    turned = 0
    for v in me.vertices:
        near = sorted(tree.find_nearest_range(v.co, reach), key=lambda hit: hit[3])
        hit = next((h for h in near if h[1].dot(v.normal) > 0.2), None)
        if hit is None:
            hit = near[0] if near else tree.find_nearest(v.co)
        elif hit is not near[0]:
            turned += 1
        w[v.index] = pw[polys[hit[2]]].mean(axis=0)
    co = np.array([v.co for v in me.vertices], dtype=np.float32)

    def to_segment(a, b):
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        t = np.clip((co - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        return np.linalg.norm(co - (a + t[:, None] * (b - a)), axis=1)

    top = w.argmax(axis=1)
    for sd in "LR":
        upper, fore = names.index("upper." + sd), names.index("fore." + sd)
        d = np.minimum(to_segment(src["sh" + sd], src["el" + sd]), to_segment(src["el" + sd], src["wr" + sd]))
        r = float(np.percentile(d[top == fore], 90))  # the forearm is all arm: its thickness is the sleeve's
        keep = np.clip((1.7 * r - d) / (0.5 * r), 0, 1)
        w[:, upper] *= keep
        w[:, fore] *= keep
    # A leg ends at the hip. The copy gives a thigh whatever is nearest it, and under a bulky top that reaches
    # the hem of a jumper and the waistband: when the thigh bends to sit, it pulls the hem into a fringe of
    # spikes. So a thigh's pull fades out over a band just above the hip joint, on every vertex alike.
    for sd in "LR":
        hip, knee = src["hip" + sd], src["kn" + sd]
        leg = (hip - knee).length
        keep = np.clip((hip.z + 0.4 * leg - co[:, 2]) / (0.3 * leg), 0, 1)
        for bone in ("thigh." + sd, "shin." + sd):
            w[:, names.index(bone)] *= keep
    lost = w.sum(axis=1) < 1e-4
    w[lost, names.index("spine")] = 1
    w /= w.sum(axis=1)[:, None]
    edges = np.array([e.vertices for e in me.edges])
    for _ in range(smooth):
        total, count = w.copy(), np.ones(len(w), dtype=np.float32)
        np.add.at(total, edges[:, 0], w[edges[:, 1]])
        np.add.at(total, edges[:, 1], w[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1)
        np.add.at(count, edges[:, 1], 1)
        w = total / count[:, None]
    for sd in "LR":  # a shoe is stiff: everything below the ankle goes with the foot, blending in just above it
        foot, ankle = names.index("foot." + sd), src["an" + sd]
        side = (co[:, 0] > 0) == (ankle.x > 0)
        stiff = np.clip((ankle.z * 1.5 - co[:, 2]) / (ankle.z * 0.5), 0, 1) * side
        w *= (1 - stiff)[:, None]
        w[:, foot] += stiff
    for i, name in enumerate(names):
        g = body.vertex_groups.new(name=name)
        for vi in np.nonzero(w[:, i] > 1e-3)[0]:
            g.add([int(vi)], float(w[vi, i]), "REPLACE")
    print(f"fit: weights copied, {turned} vertices took a facing surface over the nearest one")


def skin(body, head):
    """Tidy the copied weights: every vertex's weights sum to 1, the head (`head` is 1 there) is rigid on the neck
    bone and nothing else is on it. Returns how many body vertices had no weight (they go on the spine)."""
    groups = {g.index: g.name for g in body.vertex_groups}
    neck_i, spine_i = body.vertex_groups["neck"].index, body.vertex_groups["spine"].index
    unweighted = 0
    for v in body.data.vertices:
        t = float(head[v.index])
        w = {g.group: g.weight for g in v.groups if g.weight > 1e-4 and g.group in groups and g.group != neck_i}
        total = sum(w.values())
        if not total:
            unweighted += t < 1
            w, total = {spine_i: 1.0}, 1.0
        w = {i: x / total * (1 - t) for i, x in w.items()}
        if t:
            w[neck_i] = t
        for g in list(v.groups):
            body.vertex_groups[g.group].remove([v.index])
        for i, x in w.items():
            if x > 1e-4:
                body.vertex_groups[i].add([v.index], x, "REPLACE")
    return unweighted


def preview(body, folder: Path):
    """Three stills of the fitted body (Workbench, textured), to look at before a spec is run against it."""
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light, sc.display.shading.color_type = "STUDIO", "TEXTURE"
    sc.render.resolution_x = sc.render.resolution_y = 800
    cam = D.objects.new("cam", D.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    top = max(v.co.z for v in body.data.vertices)
    cam.data.type, cam.data.ortho_scale = "ORTHO", top * 1.15
    for name, at, rot in (("front", (0, -6, top / 2), (1.5708, 0, 0)), ("side", (-6, 0, top / 2), (1.5708, 0, -1.5708)),
                          ("three_quarter", (-4, -4, top / 2 + 1.5), (1.3, 0, -0.7854))):
        cam.location, cam.rotation_euler = at, rot
        sc.render.filepath = str(folder / f"fit-{name}.png")
        bpy.ops.render.render(write_still=True)
    D.objects.remove(cam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    fit = json.loads(Path(args.fit).read_text(encoding="utf-8"))

    cx, scale = fit.get("centre_x", 0.0), float(fit.get("scale", 1.0))
    body, marks = load_model(Path(args.model), cx, int(fit.get("faces", 150000)), fit["face"])
    src = source_joints(fit)
    if "arm_stretch" in fit:
        print("fit: arm_stretch is no longer used: the model keeps its own arms")
    fingers, pieces = None, []
    if fit.get("hands"):  # hands with fingers, from a second model of the character made with its hands open
        donor_path = Path(fit["hands"]["model"])
        donor = load_donor(donor_path if donor_path.is_absolute() else Path(args.fit).resolve().parent / donor_path,
                           fit["hands"].get("centre_x"))
        fingers, pieces = hand_graft.graft(body, donor, src, int(fit["hands"].get("faces", 9000)),
                                           float(fit["hands"].get("scale", 1.0)))
        D.objects.remove(donor)
    proxy, proxy_rig = clay_proxy(src)
    copy_weights(proxy, body, src)
    D.objects.remove(proxy)
    D.objects.remove(proxy_rig)
    unweighted = skin(body, head_mask(body, fit["clothes"]))
    if pieces:
        hand_graft.join(body, pieces)
        body["hand_material"] = len(body.data.materials) - 1  # (the rig measures a hand's whole skin by it)
    body.data.transform(Matrix.Scale(scale, 4))  # the whole model, evenly: nothing is stretched or re-posed
    lift = -min(v.co.z for v in body.data.vertices)  # soles on the floor
    body.data.transform(Matrix.Translation((0, 0, lift)))
    body.data.update()
    placed = lambda p: Vector(p) * scale + Vector((0, 0, lift))  # noqa: E731  a point on the model, as saved
    if fit.get("gamma"):  # generated textures come back darker than the picture they were made from
        for img in {n.image for m in body.data.materials for n in m.node_tree.nodes if n.type == "TEX_IMAGE"
                    and n.outputs[0].links and n.outputs[0].links[0].to_socket.name == "Base Color"}:
            px = pixels_of(img)
            px[:, :, :3] = np.clip(px[:, :, :3], 0, 1) ** float(fit["gamma"])
            img.pixels.foreach_set(px.ravel())
            img.update()
            img.pack()

    on_head = lambda p: list(placed(p))  # noqa: E731
    face = fit["face"]
    body["rest"] = "a_pose"
    body["joints"] = json.dumps({k: list(placed(v)) for k, v in src.items()})
    if fingers:  # each finger's root, middle and tip and the palm's facing, for the rig to build and curl them
        body["fingers"] = json.dumps({sd: {"palm": h["palm"], "axis": h["axis"], "fingers": [
            {"name": f["name"], "joints": [list(placed(j)) for j in f["joints"]]} for f in h["fingers"]]}
            for sd, h in fingers.items()})
    body["face"] = json.dumps({
        "eyes": [on_head(p) for p in marks["eyes"]], "brows": [on_head(p) for p in marks["brows"]],
        "mouth": on_head(marks["mouth"]), "eye_size": [x * scale for x in face["eye_size"]],
        "brow_size": [x * scale for x in face["brow_size"]], "mouth_width": face["mouth_width"] * scale,
        "half_width": face["half_width"] * scale, "moustache": bool(face.get("moustache")),
        # the chin's underside (the jaw key moves what is between it and the lips): given, or a mouth's width down
        "chin": placed((0, 0, face["chin"])).z if "chin" in face
        else on_head(marks["mouth"])[2] - 1.2 * face["mouth_width"] * scale})
    if args.preview:
        preview(body, Path(args.preview))
    bpy.ops.file.pack_all()
    out = Path(args.out).resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    print(f"FIT scale {scale:.3f}, {len(body.data.polygons)} faces, {unweighted} vertices had no weight -> {out}")


main()
