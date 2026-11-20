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
Optional: "joints_right": {"el": [x, y, z], ...}, a right-side joint where the model is not symmetrical, written
as the left is (x positive). And "hands": {"model": "open_hands.glb", "scale": 0.7, "smooth": [[along, out_of_palm, across, radius]],
"hue": 0.05, "donor_hue": 0.1} (hue: how near the fingertips' hue counts as skin, 0.1 unless given; lower it for
brown or tan sleeves; donor_hue: the same for the hands model, `hue` unless given; "light": 0.6 and "donor_light"
leave out what is under that share of the fingertips' brightness, for a dark sleeve of skin's own hue; the hands
model may be the body's own model when its fingers are apart)
(hands.py: the donor's hands are sized to the body's own unless `scale` says, the wrist joint moves to the cuff
whatever `wr` says, and `smooth` flattens a lump on both hands, in cm from the middle of the knuckle row), and in
"face", "brows_hidden": [0] for a brow under a fringe. "lid": 0.3 rests the upper lids over that share of the eye
(hooded eyes; without it the rig's eyes are wide open at rest).
"skirt": 0.15 says the model wears a long skirt or dress with its hem at that height: the cloth above the hem
shares the two legs' weights and is not cut between them (without it the fit parts the legs, and a skirt shows a
slit up the front and the back).
"face", "paint": "keep" leaves the painted face on the model and cuts the eyes that move from it (facepaint.py):
for a face painted nearer to life than a cartoon, where the rig's own eyes would be twice the size and stare.
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

import facepaint  # noqa: E402
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


def _in_triangle(pts, a, b, c, pad):
    """Which of `pts` (n x 2, texels) lie inside triangle abc or within `pad` texels of its edges."""
    inside = np.ones(len(pts), dtype=bool)
    area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(area) < 1e-9:
        return np.zeros(len(pts), dtype=bool)
    for p, q in ((a, b), (b, c), (c, a)):
        e = q - p
        n = np.hypot(*e)
        if n < 1e-9:
            continue
        side = ((e[0] * (pts[:, 1] - p[1]) - e[1] * (pts[:, 0] - p[0])) / n) * np.sign(area)
        inside &= side > -pad
    return inside


def erase(ob, shapes):
    """Take painted features off the model: `shapes` are ellipses (x, z, rx, rz) on its front. Their texels take
    the median of the ring round each, in every texture, and the surface under them is smoothed flat."""
    me = ob.data
    uv = me.uv_layers.active.data
    imgs = [(img, pixels_of(img)) for img in images_of(ob)]
    base = pixels_of(base_colour(ob))
    bh, bw = base.shape[:2]

    def colour(poly):  # a face's own colour: the base colour under its corners and its middle
        at = np.array([uv[li].uv for li in poly.loop_indices])
        at = np.vstack([at, at.mean(axis=0)])
        return base[np.clip((at[:, 1] * bh).astype(int), 0, bh - 1), np.clip((at[:, 0] * bw).astype(int), 0, bw - 1), :3].mean(axis=0)

    bm_verts = set()
    for x, z, rx, rz in shapes:
        y0 = front_y(ob, x, z)
        inside, ring = [], []
        for poly in me.polygons:
            c = poly.center
            if c.y > y0 + 0.04 or c.y < y0 - 0.03:  # (not the back of the head, nor a fringe hanging in front)
                continue
            d = ((c.x - x) / rx) ** 2 + ((c.z - z) / rz) ** 2
            if d <= 1:
                inside.append(poly)
            elif d <= 2.2:
                ring.append(poly)
        # painted lines that run on past the shape (lashes at the corner of an eye) go with it: faces of the ring
        # that are dark beside its skin, and dark in every channel (hair at a temple is dark too, but it is a
        # colour: its strongest channel is as bright as skin's)
        shade = np.array([colour(poly) for poly in ring]) if ring else np.zeros((0, 3))
        if len(shade):
            skin = np.median(shade[shade.sum(axis=1) >= np.median(shade.sum(axis=1))], axis=0)
            lines = (shade.sum(axis=1) < 0.55 * skin.sum()) & (shade.max(axis=1) < 0.55 * skin.max())
            inside += [poly for poly, dark in zip(ring, lines) if dark]
            ring = [poly for poly, dark in zip(ring, lines) if not dark]
        for img, px in imgs:
            h, w = px.shape[:2]
            at = np.array([[uv[li].uv[0] * w, uv[li].uv[1] * h] for poly in ring for li in poly.loop_indices])
            ring_px = px[np.clip(at[:, 1].astype(int), 0, h - 1), np.clip(at[:, 0].astype(int), 0, w - 1)]
            # the skin round it: the brighter half of the ring (blush, lashes and a fringe are darker than skin)
            light = ring_px[:, :3].sum(axis=1)
            fill = np.median(ring_px[light >= np.median(light)], axis=0)
            for poly in inside:
                t = np.array([[uv[li].uv[0] * w, uv[li].uv[1] * h] for li in poly.loop_indices])
                x0, x1 = int(max(0, t[:, 0].min() - 2)), int(min(w - 1, t[:, 0].max() + 2))
                z0, z1 = int(max(0, t[:, 1].min() - 2)), int(min(h - 1, t[:, 1].max() + 2))
                # the texels in the face, padded by a texel or two for the seams (not its whole box: a large
                # triangle's box reaches into other parts of the texture, such as an eye beside a fringe)
                gx, gz = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(z0, z1 + 1) + 0.5)
                pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
                keep = np.zeros(len(pts), dtype=bool)
                for i in range(1, len(t) - 1):  # (a fan of triangles over the face)
                    keep |= _in_triangle(pts, t[0], t[i], t[i + 1], 2.0)
                sub = px[z0:z1 + 1, x0:x1 + 1]
                sub[keep.reshape(sub.shape[:2])] = fill
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


def mend(ob) -> dict:
    """Take the slivers off a welded model. A generator leaves a few stray triangles standing on the surface
    like fins (an edge shared by three faces, the third with an open edge) and pinholes a millimetre across.
    Under cloth nobody sees them; on a face they render as a dark triangle at a temple or a shard by an eye,
    whichever way the light falls on the fin. The fins are deleted and the pinholes closed. Returns the counts,
    and how many open edges are left (a real opening, such as a sleeve's end, is left alone)."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    fins = 0
    for _ in range(4):
        loose = {f for e in bm.edges if len(e.link_faces) > 2 for f in e.link_faces
                 if any(x.is_boundary for x in f.edges)}
        if not loose:
            break
        fins += len(loose)
        bmesh.ops.delete(bm, geom=list(loose), context="FACES")
    seen, holes = set(), []
    for e in [e for e in bm.edges if e.is_boundary]:
        if e in seen:
            continue
        ring, stack = [], [e]
        seen.add(e)
        while stack:
            x = stack.pop()
            ring.append(x)
            for v in x.verts:
                for y in v.link_edges:
                    if y.is_boundary and y not in seen:
                        seen.add(y)
                        stack.append(y)
        holes.append(ring)
    small = [r for r in holes if len(r) <= MEND_EDGES and
             max((a.verts[0].co - b.verts[0].co).length for a in r for b in r) < MEND_SIZE]
    if small:
        bmesh.ops.holes_fill(bm, edges=[e for r in small for e in r], sides=MEND_EDGES)
    stray = [v for v in bm.verts if not v.link_faces]
    if stray:
        bmesh.ops.delete(bm, geom=stray, context="VERTS")
    left = sum(1 for e in bm.edges if e.is_boundary)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return {"fins": fins, "pinholes": len(small), "open_edges": left}


PAINT_LOST_LIMIT = 0.002  # the share of faces that may be painted from the wrong part of the texture after the reduction
MEND_EDGES = 8      # a hole with no more edges than this round it, and
MEND_SIZE = 0.006   # no wider than this (m), is a pinhole the generator left, and is closed


def face_shapes(face):
    m = 1.35  # a margin: the painted edge is soft
    out = [(x, z, face["eye_size"][0] / 2 * m, face["eye_size"][1] / 2 * m) for x, z in face["eyes"]]
    hidden = set(face.get("brows_hidden", []))  # (a brow under a fringe: nothing painted there to take off)
    return out + [(x, z, face["brow_size"][0] / 2 * 1.15, face["brow_size"][1] / 2 * 1.9)
                  for i, (x, z) in enumerate(face["brows"]) if i not in hidden]


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
    mended = mend(ob)
    print(f"fit: mesh, {mended['fins']} stray faces standing on the surface removed, {mended['pinholes']} pinholes "
          f"closed, {mended['open_edges']} open edges left")
    marks = {"eyes": [(x - centre_x, front_y(ob, x - centre_x, z), z) for x, z in face["eyes"]],
             "brows": [(x - centre_x, front_y(ob, x - centre_x, z), z) for x, z in face["brows"]],
             "mouth": (face["mouth"][0] - centre_x, front_y(ob, face["mouth"][0] - centre_x, face["mouth"][1]),
                       face["mouth"][1])}
    # the painted face is read before anything is taken off it: `check` holds the rig's face against it, and
    # with "paint": "keep" the paint stays and the eyes that move are cut from it (facepaint.py)
    bpy.context.view_layer.update()
    seen = facepaint.measure(ob, face, centre_x)
    for line in facepaint.words(seen):
        print(f"fit: painted face, {line}")
    keep = face.get("paint") == "keep"
    if keep and not all(seen["eyes"]):
        raise SystemExit('fit: "paint": "keep" needs a painted eye on each side (white beside an iris) and found '
                         "none on one: check the fit's eyes are on the model's eyes, or leave paint out")
    if not keep:
        erase(ob, [(x - centre_x, z, rx, rz) for x, z, rx, rz in face_shapes(face)])
    if len(ob.data.polygons) > faces:
        # (thinned inside the islands of its texture, never across their seams: hands.reduce says why)
        was, made = hand_graft.surface(ob), len(ob.data.polygons)
        held = hand_graft.reduce(ob, faces)
        lost = hand_graft.paint_lost(ob, was)
        print(f"fit: mesh, reduced from {made} to {len(ob.data.polygons)} faces with {held} vertices on the "
              f"texture's seams left alone; {100 * lost:.2f}% of the faces are painted from the wrong place")
        if lost > PAINT_LOST_LIMIT:
            print(f"fit: FAIL mesh, {100 * lost:.1f}% of the faces lost their paint in the reduction (limit "
                  f"{100 * PAINT_LOST_LIMIT:.1f}%): shards of the wrong colour will show, worst on the face")
    for poly in ob.data.polygons:
        poly.use_smooth = True
    ob.name = "model_body"
    return ob, marks, seen


def recolour(body, rules, gamma=None) -> list:
    """Shift parts of the model's own paint to a colour read off a photo. Each rule is {"name", "sample":
    [x0, x1, z0, z1] (a patch on the model's front where the part is, in the model's metres), "to": a hex colour,
    "tolerance", "above", "below" (heights the part lies between), "keep": 0 to 1, "fill": texels}. Every texel between those
    heights whose colour is within `tolerance` of the patch's takes the new colour, as much lighter or darker
    than it as the texel was than the patch (so strands and stubble keep their shading), fading out to nothing at
    1.6 times the tolerance. `keep` leaves that share of the old colour in. Rules
    paint in the order given and none repaints what an earlier one moved, so put the small part first (the grey
    of a chin before the beard round it). `fill` closes gaps up to that many
    texels across in what was moved: stubble painted as dots of hair colour on skin goes dark as dots, with the
    skin left pale between them, and reads as pepper. Returns what each rule changed."""
    me = body.data
    uv = me.uv_layers.active.data
    co = np.array([v.co for v in me.vertices])
    images = {}
    for i, m in enumerate(me.materials):
        img = next((n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image and n.outputs[0].links
                    and n.outputs[0].links[0].to_socket.name == "Base Color"), None)
        if img:
            images[i] = (img, pixels_of(img))
    lum = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    done = []
    def patch_colour(rule):
        x0, x1, z0, z1 = rule["sample"]
        img, px = images[0]
        h, w = px.shape[:2]
        patch = []
        for poly in me.polygons:
            c = poly.center
            if poly.material_index == 0 and x0 < c.x < x1 and z0 < c.z < z1 and c.y < 0 and poly.normal.y < -0.3:
                at = np.array([uv[li].uv for li in poly.loop_indices]).mean(axis=0)
                patch.append(px[min(h - 1, int(at[1] % 1 * h)), min(w - 1, int(at[0] % 1 * w)), :3].copy())
        if not patch:
            raise SystemExit(f"fit: recolour {rule.get('name', '')}: nothing on the model's front in its sample {rule['sample']}")
        return np.median(np.array(patch), axis=0)

    # every rule's patch is read before any rule paints: a rule's patch may lie in what an earlier one moves
    samples = [patch_colour(rule) for rule in rules]
    touched = {i: np.zeros(px.shape[:2], dtype=bool) for i, (_, px) in images.items()}
    for rule, was in zip(rules, samples):
        lo, hi = rule.get("above", -1e9), rule.get("below", 1e9)
        tol = float(rule.get("tolerance", 0.12))
        to = np.array([int(rule["to"].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=np.float32)
        if gamma:  # the whole texture is lightened afterwards: aim at what will come out as `to`
            to = to ** (1 / float(gamma))
        zs = co[:, 2]
        changed = total = 0
        moved = {i: np.zeros(px.shape[:2], dtype=np.float32) for i, (_, px) in images.items()}
        within = {i: np.zeros(px.shape[:2], dtype=bool) for i, (_, px) in images.items()}
        for poly in me.polygons:
            if poly.material_index not in images:
                continue
            z = zs[list(poly.vertices)]
            if z.max() < lo or z.min() > hi:
                continue
            img, px = images[poly.material_index]
            h, w = px.shape[:2]
            t = np.array([[uv[li].uv[0] * w, uv[li].uv[1] * h] for li in poly.loop_indices])
            a0, a1 = int(max(0, t[:, 0].min() - 1)), int(min(w - 1, t[:, 0].max() + 1))
            b0, b1 = int(max(0, t[:, 1].min() - 1)), int(min(h - 1, t[:, 1].max() + 1))
            gx, gz = np.meshgrid(np.arange(a0, a1 + 1) + 0.5, np.arange(b0, b1 + 1) + 0.5)
            pts = np.stack([gx.ravel(), gz.ravel()], axis=1)
            inside = np.zeros(len(pts), dtype=bool)
            for i in range(1, len(t) - 1):
                inside |= _in_triangle(pts, t[0], t[i], t[i + 1], 1.0)
            sub = px[b0:b1 + 1, a0:a1 + 1]
            rgb = sub[:, :, :3]
            # a texel moved by an earlier face of this rule is no longer near the patch's colour, so it is left
            near = np.linalg.norm(rgb - was, axis=2)
            free = ~touched[poly.material_index][b0:b1 + 1, a0:a1 + 1]  # (what an earlier rule moved stays as it left it)
            wgt = np.clip((1.6 * tol - near) / (0.6 * tol), 0, 1) * inside.reshape(near.shape) * free * (1 - float(rule.get("keep", 0)))
            shade = (rgb @ lum) / max(1e-6, float(was @ lum))
            rgb += wgt[:, :, None] * (np.clip(to[None, None, :] * shade[:, :, None], 0, 1) - rgb)
            changed += int((wgt > 0.5).sum())
            total += int(inside.sum())
            np.maximum(moved[poly.material_index][b0:b1 + 1, a0:a1 + 1], wgt, out=moved[poly.material_index][b0:b1 + 1, a0:a1 + 1])
            within[poly.material_index][b0:b1 + 1, a0:a1 + 1] |= inside.reshape(near.shape)
        filled = 0
        r = int(rule.get("fill", 0))
        if r:
            def spread(a, pick):  # the largest (or smallest) value within r texels, along one axis then the other
                for axis in (0, 1):
                    a = pick.reduce([np.roll(a, k, axis=axis) for k in range(-r, r + 1)])
                return a
            for i, (img, px) in images.items():
                closed = spread(spread(moved[i], np.maximum), np.minimum)
                gap = np.clip(closed - moved[i], 0, 1) * within[i] * ~touched[i] * (1 - float(rule.get("keep", 0)))
                px[:, :, :3] += gap[:, :, None] * (to[None, None, :] - px[:, :, :3])
                filled += int((gap > 0.5).sum())
                moved[i] = np.maximum(moved[i], gap)
        changed += filled
        for i in touched:
            touched[i] |= moved[i] > 0.05
        done.append({"name": rule.get("name", ""), "was": [round(float(c), 3) for c in was], "texels": changed,
                     "share": round(changed / max(1, total), 3)})
    for img, px in images.values():
        img.pixels.foreach_set(px.ravel())
        img.update()
        img.pack()
    return done


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
    # (the joints are written for the character's left and mirrored; `joints_right` gives a joint of the right
    # side where the model is not symmetrical, written the same way, x positive. A right elbow mirrored 3 cm
    # outside the arm puts the inside of the arm out of its sleeve, and that skin then hangs from the hip)
    right = {k: Vector(v) for k, v in fit.get("joints_right", {}).items()}
    for s, sd in ((1, "L"), (-1, "R")):
        for k in ("sh", "el", "wr", "tip", "hip", "kn", "an", "toe"):
            at = right.get(k, j[k]) if sd == "R" else j[k]
            out[k + sd] = Vector((at.x * s, at.y, at.z))
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


def copy_weights(proxy, body, src, reach=0.25, smooth=14):
    """The proxy's weights, onto the model. Each model vertex takes them from the nearest proxy face that looks
    the same way: the inside of a sleeve and the side of the chest are close together in an A-pose but face each
    other, and plain nearest-surface copying swaps them (which tears the armpit when the arm lifts).
    An arm then only keeps what lies inside its own sleeve (a tube round shoulder, elbow and wrist, as thick as
    the model's upper arm and forearm each measure): a bulky chest is nearer the proxy's arm than its thin chest, and would be pulled
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

    def sleeve(a, b, side):
        """How thick the model's arm is round the bone from a to b (m): its vertices on the side away from the
        body, where there is only arm (the inner side of an arm in an A-pose runs into the chest)."""
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        axis = (b - a) / np.linalg.norm(b - a)
        t = (co - a) @ axis / np.linalg.norm(b - a)
        off = co - a - np.outer((co - a) @ axis, axis)
        out = np.array([side, 0, 0], dtype=np.float32)
        out -= axis * (out @ axis)
        d = np.linalg.norm(off, axis=1)
        mine = (t > 0.2) & (t < 0.9) & (off @ out > 0) & (d < 0.3)
        return float(np.percentile(d[mine], 95)) if mine.any() else 0.08

    # (each part of the arm by its own thickness, and the two arms alike: a puppet's upper sleeve is twice as
    # thick as its forearm, and a tube sized by the forearm lets the back of the upper arm go to the chest)
    thick = {part: 0.5 * sum(sleeve(src[a + sd], src[b + sd], 1.0 if sd == "L" else -1.0) for sd in "LR")
             for part, a, b in (("upper", "sh", "el"), ("fore", "el", "wr"))}
    print(f"fit: sleeves {100 * thick['upper']:.1f} cm thick round the upper arm and {100 * thick['fore']:.1f} cm "
          "round the forearm")
    for sd in "LR":
        upper, fore = names.index("upper." + sd), names.index("fore." + sd)
        keep = np.maximum(
            np.clip((1.7 * thick["upper"] - to_segment(src["sh" + sd], src["el" + sd])) / (0.5 * thick["upper"]), 0, 1),
            np.clip((1.7 * thick["fore"] - to_segment(src["el" + sd], src["wr" + sd])) / (0.5 * thick["fore"]), 0, 1))
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
    # The other way round from the sleeve: a forearm that hangs beside the hips in the A-pose takes some of the
    # hip's weights (the hip's surface faces the same way as the forearm's, and smoothing spreads it along the
    # arm). Such a vertex is held back toward the hip when the arm comes up, so the forearm's underside sags, and
    # on a desk it goes through the top. So the arm from half way down the upper arm to the wrist is arm only,
    # where a vertex is half arm or more already and far nearer the arm than the middle of the body (a chest or hip
    # vertex near the arm is not half arm, and keeps its weights; the shoulder end of the upper arm keeps the
    # spine's share a shoulder needs). Not by a tube the forearm's thickness, which is measured on the outer side:
    # the inside of an arm can stand further off its bone (5.6 cm on a forearm measured 3.2 cm thick).
    arm = np.array([name.split(".")[0] in ("upper", "fore", "hand") for name in names])
    middle = np.hypot(co[:, 0], co[:, 1] - src["pelvis"].y)  # how far out from the body's middle
    freed = 0
    for sd in "LR":
        half = (src["sh" + sd] + src["el" + sd]) / 2
        d = np.minimum(to_segment(src["el" + sd], src["wr" + sd]), to_segment(half, src["el" + sd]))
        share = w[:, arm].sum(axis=1) / np.maximum(w.sum(axis=1), 1e-6)
        mine = np.clip((0.6 - d / np.maximum(middle, 1e-6)) / 0.25, 0, 1) * np.clip((share - 0.3) / 0.2, 0, 1)
        freed += int(((mine > 0) & (share < 0.999)).sum())
        w[:, ~arm] *= (1 - mine)[:, None]
    w /= np.maximum(w.sum(axis=1), 1e-6)[:, None]
    print(f"fit: arms, {freed} vertices from the elbow down had some of the body's weight, now the arm's")
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


def skirt_weights(body, hem: float, half: float) -> int:
    """A long skirt or dress is one cloth round both legs, and the proxy the weights come from has two legs: each
    side of the cloth took its own leg's weight, with a hard line down the middle, front and back. Above the hem
    the two legs' weights are shared across the cloth instead, all of a leg's at that leg (`half` out from the
    middle, the hip joint) and half each on the middle line, so the cloth between the legs spans them as they
    part. Returns how many vertices were changed."""
    index = {g.name: g.index for g in body.vertex_groups}
    pairs = [(index[f"{part}.L"], index[f"{part}.R"]) for part in ("thigh", "shin", "foot")
             if f"{part}.L" in index and f"{part}.R" in index]
    changed = 0
    for v in body.data.vertices:
        if v.co.z <= hem:
            continue
        w = {g.group: g.weight for g in v.groups}
        t = min(1.0, max(0.0, (v.co.x / half + 1) / 2))
        t = t * t * (3 - 2 * t)
        moved = False
        for left, right in pairs:
            total = w.get(left, 0.0) + w.get(right, 0.0)
            if total < 1e-4:
                continue
            body.vertex_groups[left].add([v.index], total * t, "REPLACE")
            body.vertex_groups[right].add([v.index], total * (1 - t), "REPLACE")
            moved = True
        changed += moved
    return changed


def part_legs(body, above: float | None = None) -> int:
    """Cut the faces that join one leg to the other. A generator often makes the feet (or knees) touch, and the
    mesh then runs from one shoe into the other: weighted, those faces stretch across the gap the moment a foot
    moves (30x on a stride). A face is cut when one of its corners moves with the left leg and another with the
    right. Under a skirt (`above`, its hem) only what is below the hem is cut: the cloth is meant to join the
    legs, and cut it showed a slit up the front and the back. Returns how many faces were cut."""
    legs = {}
    for g in body.vertex_groups:
        part, _, sd = g.name.rpartition(".")
        if part in ("thigh", "shin", "foot") and sd in ("L", "R"):
            legs[g.index] = sd
    side = {}
    for v in body.data.vertices:
        w = {"L": 0.0, "R": 0.0}
        for g in v.groups:
            if g.group in legs:
                w[legs[g.group]] += g.weight
        if max(w.values()) > 0.5:
            side[v.index] = "L" if w["L"] > w["R"] else "R"
    bm = bmesh.new()
    bm.from_mesh(body.data)
    cut = [f for f in bm.faces if len({side.get(v.index) for v in f.verts} - {None}) > 1
           and (above is None or min(v.co.z for v in f.verts) <= above)]
    if cut:
        bmesh.ops.delete(bm, geom=cut, context="FACES")  # (and the edges and vertices left loose)
        bm.to_mesh(body.data)
    bm.free()
    return len(cut)


ARM_JOINED_LIMIT = 5  # edges running straight from a forearm or hand to the body before the fit fails it


def arm_joined(body) -> dict:
    """How many edges of the mesh run straight from each forearm or hand to the body (spine, hips or a thigh),
    each end by the bone that moves it most. On a free arm there are none: between them is the upper arm. Where
    the inside of a forearm is left moving with the body, a sheet of skin hangs from the arm to the hip and bags
    whenever the arm moves. Seen on a model whose right arm hung 3 cm nearer the body than its left: the elbow,
    mirrored from the left, stood outside the arm and its inside went to the spine (29 edges; every other arm
    measured has none). The fix is the joint: `joints_right` in the fit file."""
    names = [g.name for g in body.vertex_groups]
    part = []
    for v in body.data.vertices:
        if not v.groups:
            part.append(None)
            continue
        top = names[max(v.groups, key=lambda g: g.weight).group]
        kind = top.split(".")[0]
        part.append(top[-1] if kind in ("fore", "hand") else ("body" if kind in ("spine", "hips", "thigh") else None))
    out = {"L": 0, "R": 0}
    for e in body.data.edges:
        a, b = part[e.vertices[0]], part[e.vertices[1]]
        if {a, b} in ({"L", "body"}, {"R", "body"}):
            out[a if a != "body" else b] += 1
    return out


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
    body, marks, seen = load_model(Path(args.model), cx, int(fit.get("faces", 150000)), fit["face"])
    src = source_joints(fit)
    if "arm_stretch" in fit:
        print("fit: arm_stretch is no longer used: the model keeps its own arms")
    fingers, pieces = None, []
    if fit.get("hands"):  # hands with fingers, from a second model of the character made with its hands open
        donor_path = Path(fit["hands"]["model"])
        donor = load_donor(donor_path if donor_path.is_absolute() else Path(args.fit).resolve().parent / donor_path,
                           fit["hands"].get("centre_x"))
        # (a hands model made from a closer picture, hips up so the hands are bigger in it, comes back sized as
        # if it were the whole figure: the hands are looked for by lengths in metres, so it is brought to the
        # body's arm span first)
        span = max(abs(v.co.x) for v in body.data.vertices) / max(abs(v.co.x) for v in donor.data.vertices)
        if abs(span - 1) > 0.15:
            donor.data.transform(Matrix.Scale(span, 4))
            donor.data.update()
            print(f"fit: hands, the hands model is {1 / span:.2f}x the body's arm span: scaled to match before "
                  "its hands are looked for")
        fingers, pieces = hand_graft.graft(body, donor, src, int(fit["hands"].get("faces", 9000)),
                                           float(fit["hands"]["scale"]) if "scale" in fit["hands"] else None,
                                           lumps=fit["hands"].get("smooth", ()),
                                           hue_tol=float(fit["hands"].get("hue", 0.1)),
                                           light=float(fit["hands"].get("light", 0.0)),
                                           donor_light=(float(fit["hands"]["donor_light"])
                                                        if "donor_light" in fit["hands"] else None),
                                           donor_hue_tol=(float(fit["hands"]["donor_hue"]) if "donor_hue" in fit["hands"]
                                                          else None))
        D.objects.remove(donor)
    proxy, proxy_rig = clay_proxy(src)
    copy_weights(proxy, body, src)
    D.objects.remove(proxy)
    D.objects.remove(proxy_rig)
    unweighted = skin(body, head_mask(body, fit["clothes"]))
    hem = fit.get("skirt")
    if hem is not None:
        shared = skirt_weights(body, float(hem), abs(src["hipL"].x) or 0.09)
        print(f"fit: skirt, {shared} vertices above its hem at {float(hem):.2f} m share the two legs' weights")
    parted = part_legs(body, None if hem is None else float(hem))
    if parted:
        print(f"fit: the legs touched in the model: {parted} faces joining one to the other were cut")
    for sd, joined in arm_joined(body).items():
        if joined > ARM_JOINED_LIMIT:
            print(f"fit: FAIL weights, the inside of the {'left' if sd == 'L' else 'right'} forearm moves with the body: "
                  f"{joined} edges run from it straight to the body (limit {ARM_JOINED_LIMIT}), and the skin will hang "
                  "from the arm to the hip as it moves. The elbow or wrist is likely off the arm: measure that side's "
                  "arm by slices and give its joints in `joints_right` (or correct `joints`)")
    if pieces:
        for before, after in hand_graft.seal(body, pieces):
            print(f"fit: hands, a bare arm's end drawn in onto the hand: the ring stood {1000 * before:.1f} mm off "
                  f"the hand's skin, now {1000 * after:.1f} mm")
        ring = hand_graft.join(body, pieces)
        if ring:
            print(f"fit: hands, {ring} vertices on the ring where a bare arm meets its hand share their normals")
        body["hand_material"] = len(body.data.materials) - 1  # (the rig measures a hand's whole skin by it)
    for r in recolour(body, fit.get("recolour", ()), fit.get("gamma")):
        print(f"fit: recolour {r['name']}: {r['texels']} texels ({100 * r['share']:.0f}% of the paint between its "
              f"heights) moved from {r['was']}")
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
        "lid": float(face.get("lid", 0)),
        # the chin's underside (the jaw key moves what is between it and the lips): given, or a mouth's width down
        "chin": placed((0, 0, face["chin"])).z if "chin" in face
        else on_head(marks["mouth"])[2] - 1.2 * face["mouth_width"] * scale,
        **facepaint.store(seen, placed, scale, fit.get("gamma"), keep=face.get("paint") == "keep")})
    if args.preview:
        preview(body, Path(args.preview))
    bpy.ops.file.pack_all()
    out = Path(args.out).resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    print(f"FIT scale {scale:.3f}, {len(body.data.polygons)} faces, {unweighted} vertices had no weight -> {out}")


main()
