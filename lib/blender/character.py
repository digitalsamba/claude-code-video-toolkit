"""ClayPerson: a seated, rigged cartoon character with an acting vocabulary (runs inside Blender).

Body: one mesh baked from a skeleton with the Skin modifier, weighted to an armature by bone heat. Arms are IK
(wrist target + elbow pole), the spine and neck follow control empties through Child Of. The head and face are
rigid parts on the neck control: eyes and pupils, brows, three mouth shapes, a sweat drop.

Everything is driven through verbs that take frames, never raw keyframes, so a shot spec can say
"expression shocked at 5.6 s" and the timing, easing and channels are handled here:

    p = ClayPerson("dev", cast.resolve({"id": "dev", "hair": {"style": "bob"}}))
    p.typing(1, 84, "calm"); p.expression(136, "shocked"); p.pose(218, 232, "hands_on_head")

What a character looks like (build, face, hair, clothes, accessories, a seed) is a description resolved by
cast.py; this module turns it into geometry. The rig is the same for everyone, so every verb works on every
character. It is built at the origin, sitting, facing -Y (one character per set for now).
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Vector

from kit import (R, arc_tube, box, cyl, empty, key, link, mat, n, sphere, ticks, tween)

D = bpy.data

BUILD = {"slim": (0.86, 0.9), "average": (1.0, 1.0), "broad": (1.16, 1.1)}       # torso radius, limb radius
HEAD = {"round": (1, 1, 1), "oval": (0.9, 0.96, 1.08), "square": (1.06, 1.0, 0.94)}  # head scale
NOSE = {"round": (0.045, (1, 1, 1)), "small": (0.034, (1, 1, 1)), "long": (0.04, (0.9, 1.5, 1.25))}
BROW = {"medium": (0.1, 0.028), "thin": (0.095, 0.016), "thick": (0.11, 0.042)}      # length, height

# wrist targets (spine space, +x = character's left) and elbow pole directions
POSES = {
    "type":              ((0.13, -0.47, 0.34), (1, 0.15, -1)),
    "rest":              ((0.13, -0.47, 0.34), (1, 0.15, -1)),
    "hands_on_head":     ((0.36, -0.05, 0.97), (1, -0.4, -0.2)),
    "arms_up":           ((0.45, -0.10, 1.18), (1, 0.2, -0.2)),
    "hands_behind_head": ((0.24, 0.20, 1.08), (1, -0.6, 0.3)),
}
# eyes scale, pupil scale, brows, mouth, mouth size
FACES = {
    "neutral":    ((1, 1, 1), 1.0, "neutral", "smile", 0.8),
    "happy":      ((1, 1, 1), 1.0, "neutral", "smile", 1.0),
    "content":    ((1, 1, 1), 1.0, "happy", "smile", 1.2),
    "shocked":    ((1.25, 1.25, 1.35), 0.6, "worry", "o", 1.0),
    "worried":    ((1.1, 1.1, 1.1), 0.8, "worry", "frown", 1.0),
    "determined": ((1.1, 1.1, 0.75), 1.0, "angry", "frown", 0.9),
    "relieved":   ((1, 1, 1), 1.0, "happy", "smile", 1.0),
    "grin":       ((1, 1, 0.55), 1.0, "happy", "smile", 1.5),
}
BROWS = {"neutral": (0, 0.16), "worry": (24, 0.175), "angry": (-18, 0.145), "happy": (-8, 0.175)}
LOOKS = {"screen": 7, "down": 12, "ahead": -2, "camera": 0, "up": -12}  # neck pitch, degrees
LEANS = {"forward": 10, "none": 0, "back": -6, "far_back": -12, "recoil": -4}
TYPING = {"calm": (4, 0.03), "frantic": (2, 0.035)}  # frames per key at 24 fps, lift (m)
# Every length below written as n(x) is frames at 24 fps, converted to the scene's rate.


class ClayPerson:
    def __init__(self, cid: str, look: dict):
        """`look` is a description from cast.resolve(): body, face, hair, clothes, accessories, seed."""
        self.id, self.desc = cid, look
        self.rng = random.Random(look["seed"])
        self.shape_rng = random.Random(look["seed"] * 7919 + 1)  # geometry draws: never disturbs the acting's
        cl = look["clothes"]
        self.m = {
            "skin": mat(f"{cid}_skin", look["body"]["skin"], 0.55), "top": mat(f"{cid}_top", cl["top_color"], 0.7),
            "pants": mat(f"{cid}_pants", cl["pants"], 0.8), "shoes": mat(f"{cid}_shoes", cl["shoes"], 0.5),
            "hair": mat(f"{cid}_hair", look["hair"]["color"], 0.6), "eyew": mat(f"{cid}_eyew", "#F3F3F3", 0.15),
            "iris": mat(f"{cid}_iris", look["face"]["eyes"], 0.2),
            "frame": mat(f"{cid}_frame", (0.03, 0.03, 0.035), 0.3),
            "gold": mat(f"{cid}_gold", (0.9, 0.62, 0.2), 0.25, 1.0),
            "pupil": mat(f"{cid}_pupil", (0.02, 0.02, 0.025), 0.05),
            "glint": mat(f"{cid}_glint", (1, 1, 1), 0.1, emit=6),
            "mouth": mat(f"{cid}_mouth", (0.35, 0.07, 0.07), 0.4),
            "mouthin": mat(f"{cid}_mouthin", (0.12, 0.02, 0.02), 0.5),
            "sweat": mat(f"{cid}_sweat", (0.55, 0.82, 1.0), 0.03, emit=0.4),
        }
        self.events: list[dict] = []
        self._build_body()
        self._build_head()
        self.neck_pitch, self.spine_pitch = 7.0, 0.0
        key(self.spine, "rotation_euler", 1, (0, 0, 0))
        key(self.neck, "rotation_euler", 1, (R(self.neck_pitch), 0, 0))
        for s in (1, -1):
            self._arm_key(s, 1, self._mirror(POSES["type"][0], s), self._mirror(POSES["type"][1], s))

    # ------------------------------------------------------------ build
    def _build_body(self):
        cid, M = self.id, self.m
        self.hips = empty(f"{cid}_hips", None, (0, 0, 0.62))
        self.breath = empty(f"{cid}_breath", self.hips)  # a slow idle under every lean, so a hold is never a freeze
        self.spine = empty(f"{cid}_spine", self.breath)
        self.neck = empty(f"{cid}_neck", self.spine, (0, 0, 0.8))
        top = self.desc["clothes"]["top"]
        torso, limb = BUILD[self.desc["body"]["build"]]
        if top == "hoodie":
            sphere(f"{cid}_hood", 1, M["top"], self.spine, (0, 0.17, 0.66), (0.2, 0.12, 0.11))
            for s in (1, -1):
                # drawstrings lie on the chest, which slopes back toward the neck: upright they stand off it
                cyl(f"{cid}_string", 0.008, 0.13, M["shoes"], self.spine,
                    (0.05 * s, -0.208 * torso - 0.004, 0.46), (R(-17), 0, 0), smooth=True)
        elif top == "shirt":
            for s in (1, -1):  # a collar: two flaps either side of the neck
                box(f"{cid}_collar", (0.1, 0.02, 0.06), M["top"], self.spine, (0.055 * s, -0.1, 0.705),
                    (R(-38), R(-24 * s), R(-28 * s)), bevel=0.006)

        SK = {"pelvis": ((0, 0, 0.62), (0.22, 0.2)), "waist": ((0, 0, 0.82), (0.27, 0.22)),
              "chest": ((0, 0, 1.04), (0.29, 0.23)), "upch": ((0, 0, 1.2), (0.25, 0.19)),
              "neckb": ((0, 0, 1.33), (0.09, 0.09)), "necktop": ((0, 0, 1.47), (0.08, 0.08))}
        for s, sd in ((1, "L"), (-1, "R")):
            SK["sh" + sd] = ((0.3 * s, 0, 1.22), (0.085, 0.085))
            SK["el" + sd] = ((0.4 * s, -0.08, 0.88), (0.066, 0.066))
            SK["wr" + sd] = ((0.4 * s, -0.42, 0.8), (0.055, 0.05))
            SK["tip" + sd] = ((0.4 * s, -0.55, 0.79), (0.068, 0.045))
            SK["th" + sd] = ((0.35 * s, -0.47, 0.83), (0.03, 0.03))
            SK["hip" + sd] = ((0.13 * s, 0, 0.6), (0.11, 0.11))
            SK["kn" + sd] = ((0.13 * s, -0.47, 0.6), (0.095, 0.095))
            SK["an" + sd] = ((0.13 * s, -0.47, 0.1), (0.075, 0.075))
            SK["toe" + sd] = ((0.13 * s, -0.64, 0.06), (0.078, 0.06))
        edges = [("pelvis", "waist"), ("waist", "chest"), ("chest", "upch"), ("upch", "neckb"), ("neckb", "necktop")]
        for sd in "LR":
            edges += [("upch", "sh" + sd), ("sh" + sd, "el" + sd), ("el" + sd, "wr" + sd), ("wr" + sd, "tip" + sd),
                      ("wr" + sd, "th" + sd), ("pelvis", "hip" + sd), ("hip" + sd, "kn" + sd),
                      ("kn" + sd, "an" + sd), ("an" + sd, "toe" + sd)]
        for name, (at, (rx, ry)) in list(SK.items()):  # the build: a thicker or thinner torso and limbs
            k = torso if name in ("waist", "chest", "upch") else limb if name[:2] in ("sh", "el", "hi", "kn") else 1
            SK[name] = (at, (rx * k, ry * k))
        names = list(SK)
        idx = {n: i for i, n in enumerate(names)}
        P = lambda n: Vector(SK[n][0])  # noqa: E731

        # 1) bake one smooth body mesh from the skeleton (Skin + Subsurf), then drop the skeleton object
        sk_me = D.meshes.new(f"{cid}_skel")
        sk_me.from_pydata([SK[n][0] for n in names], [(idx[a], idx[b]) for a, b in edges], [])
        skel = link(D.objects.new(f"{cid}_skel", sk_me))
        skm = skel.modifiers.new("Skin", "SKIN")
        skm.use_smooth_shade, skm.branch_smoothing = True, 0.4
        for name in names:  # not `n`: that is kit's frame converter
            sk_me.skin_vertices[0].data[idx[name]].radius = SK[name][1]
        sk_me.skin_vertices[0].data[idx["pelvis"]].use_root = True
        skel.modifiers.new("Pre", "SUBSURF").levels = 2
        body_me = D.meshes.new_from_object(skel.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        D.objects.remove(skel)
        self.body = body = link(D.objects.new(f"{cid}_body", body_me))

        # 2) clothing regions by nearest skeleton segment
        def seg_dist(p, a, b):
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
            return (p - (a + ab * t)).length

        # cut the mesh where the clothes end (wrists, neck, ankles), so a cuff is a clean line and not the
        # zigzag of whichever faces happen to straddle it
        bm = bmesh.new()
        bm.from_mesh(body_me)
        cuts = [("neckb", "upch", "necktop", 0, 1)]  # joint, the bone before, the bone after, material each side
        short = top == "tee"  # sleeves end above the elbow: the forearm is skin
        for sd in "LR":
            cuts += [("an" + sd, "kn" + sd, "toe" + sd, 2, 3),
                     ("el" + sd, "sh" + sd, "wr" + sd, 0, 1) if short else ("wr" + sd, "el" + sd, "tip" + sd, 0, 1)]
        planes = []
        for joint, before, after, m_before, m_after in cuts:
            j = P(joint)
            normal = ((j - P(before)).normalized() + (P(after) - j).normalized()).normalized()
            planes.append((j, normal, m_before, m_after, {joint, before, after}))
            near = [f for f in bm.faces if (f.calc_center_median() - j).length < 0.16]
            geom = list({v for f in near for v in f.verts}) + list({e for f in near for e in f.edges}) + near
            bmesh.ops.bisect_plane(bm, geom=geom, plane_co=j, plane_no=normal, dist=1e-5)
        bm.to_mesh(body_me)
        bm.free()

        for m in ("top", "skin", "pants", "shoes"):
            body_me.materials.append(M[m])
        for poly in body_me.polygons:
            c = poly.center
            a, b = min(edges, key=lambda e: seg_dist(c, P(e[0]), P(e[1])))
            if (a, b) == ("neckb", "necktop") or a.startswith("wr") or \
                    (short and a.startswith("el") and abs(c.x) > 0.33):  # (a broad torso meets the elbows)
                poly.material_index = 1
            elif a.startswith(("hip", "kn")) or (a == "pelvis" and b.startswith("hip")) or \
                    (a == "pelvis" and b == "waist" and c.z < 0.7):
                poly.material_index = 2
            elif a.startswith("an"):
                poly.material_index = 3
            for j, normal, m_before, m_after, limb_joints in planes:  # near a cuff, on that limb, the cut decides
                if (c - j).length < 0.2 and a in limb_joints and (b in limb_joints or b.startswith("th")):
                    poly.material_index = m_after if (c - j).dot(normal) > 0 else m_before
            poly.use_smooth = True

        # 3) armature in the same seated rest pose
        rig_data = D.armatures.new(f"{cid}_rig")
        self.rig = rig = link(D.objects.new(f"{cid}_rig", rig_data))
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")

        def bone(name, h, t, parent=None, connect=False):
            b = rig_data.edit_bones.new(name)
            b.head, b.tail, b.roll = h, t, 0
            if parent:
                b.parent, b.use_connect = rig_data.edit_bones[parent], connect

        bone("hips", P("pelvis"), P("pelvis") + Vector((0, 0, -0.12)))
        bone("spine", P("pelvis"), P("neckb"))
        bone("neck", P("neckb"), P("necktop"))
        for sd in "LR":
            bone("upper." + sd, P("sh" + sd), P("el" + sd), "spine")
            bone("fore." + sd, P("el" + sd), P("wr" + sd), "upper." + sd, True)
            bone("hand." + sd, P("wr" + sd), P("tip" + sd), "fore." + sd, True)
            bone("thigh." + sd, P("hip" + sd), P("kn" + sd), "hips")
            bone("shin." + sd, P("kn" + sd), P("an" + sd), "thigh." + sd, True)
            bone("foot." + sd, P("an" + sd), P("toe" + sd), "shin." + sd, True)
        bpy.ops.object.mode_set(mode="OBJECT")

        # 4) bone-heat weights (distance weights bled the chest into the arms when they were raised)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        body.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        sub = body.modifiers.new("Subsurf", "SUBSURF")
        sub.levels, sub.render_levels = 0, 1

        # 5) controls: spine/neck empties -> Child Of; wrist + pole empties -> IK (pole angle +90 for roll-0 bones)
        self.SH = {s: Vector((0.3 * s, 0, 0.6)) for s in (1, -1)}
        self.hand_t = {s: empty(f"{cid}_hand_t{s}", self.spine) for s in (1, -1)}
        self.pole_t = {s: empty(f"{cid}_pole_t{s}", self.spine) for s in (1, -1)}
        bpy.context.view_layer.update()
        pb = rig.pose.bones
        for bn, ctl in (("spine", self.spine), ("neck", self.neck)):
            c = pb[bn].constraints.new("CHILD_OF")
            c.target, c.inverse_matrix = ctl, ctl.matrix_world.inverted()
        for s, sd in ((1, "L"), (-1, "R")):
            c = pb["fore." + sd].constraints.new("IK")
            c.target, c.pole_target, c.chain_count, c.pole_angle = self.hand_t[s], self.pole_t[s], 2, R(90)

    def _build_head(self):
        cid, M = self.id, self.m
        face = self.desc["face"]
        self.head = head = sphere(f"{cid}_head", 0.3, M["skin"], self.neck, (0, 0, 0.3), seg=48)
        head.scale = HEAD[face["shape"]]  # everything on the head is its child, so the face follows the shape
        for s in (1, -1):
            sphere(f"{cid}_ear", 0.06, M["skin"], head, (0.29 * s, 0, 0), (0.45, 1, 1), seg=16)
        nose_r, nose_scale = NOSE[face["nose"]]
        sphere(f"{cid}_nose", nose_r, M["skin"], head, (0, -0.3, -0.02), nose_scale, seg=16)
        self._build_hair(head)
        self._build_extras(head)
        self.brow_len, brow_h = BROW[face["brows"]]
        self.eyes, self.pupils, self.brows = {}, {}, {}
        for s in (1, -1):
            e = empty(f"{cid}_eye{s}", head, (0.1 * s, -0.235, 0.05))
            self.eyes[s] = e
            sphere(f"{cid}_eyewhite", 0.075, M["eyew"], e, (0, 0, 0), (1, 0.75, 1))
            p = sphere(f"{cid}_pupil", 0.036, M["pupil"], e, (0, -0.052, -0.012))
            self.pupils[s] = p
            if face["eyes"].upper() != "#17110D":  # the default eye is all pupil; a colour gets an iris round it
                sphere(f"{cid}_iris", 0.052, M["iris"], p, (0, 0.016, 0), (1, 0.5, 1), seg=24)
            sphere(f"{cid}_glint", 0.011, M["glint"], p, (0.012, -0.03, 0.014), seg=12)
            self.brows[s] = box(f"{cid}_brow", (self.brow_len, 0.025, brow_h), M["hair"], head,
                                (0.1 * s, -0.275, 0.16), bevel=min(0.01, brow_h * 0.4))
        self.mouths = {
            "smile": arc_tube(f"{cid}_smile", 0.07, 0.014, math.pi * 1.08, math.pi * 1.92, M["mouth"], head,
                              (0, -0.288, -0.09), wrap=0.3),
            "frown": arc_tube(f"{cid}_frown", 0.06, 0.014, math.pi * 0.1, math.pi * 0.9, M["mouth"], head,
                              (0, -0.29, -0.16), wrap=0.3),
            "o": sphere(f"{cid}_mouth_o", 0.045, M["mouthin"], head, (0, -0.282, -0.12), (0.8, 0.35, 1.0)),
        }
        for ob in self.mouths.values():  # a mouth is drawn on the face: it shouldn't shadow the chin
            ob.visible_shadow = False
        self.mouths["smile"].scale = (0.8,) * 3
        self.mouths["frown"].scale = self.mouths["o"].scale = (0, 0, 0)
        self.sweat_drop = sphere(f"{cid}_sweat", 0.03, M["sweat"], head, (0.27, -0.12, 0.15), (0, 0, 0), seg=16)

    def _build_hair(self, head):
        """Hair is a cap and a few volumes, all spheres: the style decides which."""
        cid, m, rng = self.id, self.m["hair"], self.shape_rng
        style = self.desc["hair"]["style"]
        if style == "bald":
            return
        blob = lambda name, r, at, scale=(1, 1, 1): sphere(f"{cid}_{name}", r, m, head, at, scale, seg=20)  # noqa: E731
        sphere(f"{cid}_hair", 0.31, m, head, (0, 0.08, 0.11), (1.03, 0.93, 0.8))  # the cap
        if style == "tufts":
            for i, (x, y, z, r) in enumerate([(-0.09, -0.12, 0.3, 0.08), (0.06, -0.15, 0.29, 0.075),
                                              (0.0, -0.05, 0.35, 0.08), (0.14, -0.06, 0.29, 0.06)]):
                blob(f"tuft{i}", r, (x, y, z))
        elif style == "short":
            blob("fringe", 0.16, (0.03, -0.13, 0.25), (1.3, 0.8, 0.45))
        elif style in ("bob", "long"):
            drop, reach = (0.0, 1.5) if style == "bob" else (-0.16, 2.3)
            blob("fringe", 0.17, (0.0, -0.12, 0.25), (1.35, 0.8, 0.5))
            for s in (1, -1):
                blob("side", 0.15, (0.235 * s, 0.07, -0.06 + drop), (0.62, 1.15, reach))
            blob("back", 0.25, (0, 0.2, -0.05 + drop), (1.12, 0.62, reach * 0.82))
        elif style == "bun":
            blob("fringe", 0.16, (0.0, -0.13, 0.25), (1.3, 0.8, 0.45))
            blob("bun", 0.13, (0, 0.13, 0.38))
        elif style == "curly":
            made = 0
            while made < 26:
                x, y, z = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.0, 1)
                length = (x * x + y * y + z * z) ** 0.5
                if not 0.3 < length <= 1 or (y / length < -0.45 and z / length < 0.62):
                    continue  # keep the face clear
                blob(f"curl{made}", rng.uniform(0.07, 0.1),
                     (x / length * 0.3, y / length * 0.3 + 0.03, z / length * 0.3 + 0.04))
                made += 1

    def _build_extras(self, head):
        """Beard, glasses, headphones, earrings: rigid parts on the head."""
        cid, M = self.id, self.m
        beard = self.desc["face"]["beard"]
        if beard == "full":  # a second sphere, low and back: it shows along the jaw and leaves the mouth clear
            sphere(f"{cid}_beard", 0.3, M["hair"], head, (0, 0.045, -0.085), (1.01, 1, 1), seg=32)
        if beard in ("full", "moustache"):
            for s in (1, -1):
                sphere(f"{cid}_tash", 0.05, M["hair"], head, (0.045 * s, -0.292, -0.062), (1.25, 0.4, 0.42), seg=16)
        acc = self.desc["accessories"]
        if "glasses" in acc:
            for s in (1, -1):
                arc_tube(f"{cid}_lens", 0.084, 0.006, 0, 2 * math.pi, M["frame"], head, (0.1 * s, -0.318, 0.05))
                box(f"{cid}_temple", (0.012, 0.3, 0.012), M["frame"], head, (0.195 * s, -0.17, 0.06))
            box(f"{cid}_bridge", (0.03, 0.012, 0.012), M["frame"], head, (0, -0.318, 0.07))
        if "headphones" in acc:
            arc_tube(f"{cid}_band", 0.345, 0.02, 0.06, math.pi - 0.06, M["frame"], head, (0, 0.03, 0))
            for s in (1, -1):
                cyl(f"{cid}_cup", 0.095, 0.07, M["frame"], head, (0.325 * s, 0.03, 0), (0, R(90), 0), smooth=True)
        if "earrings" in acc:
            for s in (1, -1):
                sphere(f"{cid}_earring", 0.022, M["gold"], head, (0.305 * s, 0.0, -0.075), seg=12)

    # ------------------------------------------------------------ anchors (for cameras and checks)
    def anchors(self) -> dict:
        return {"head": (0, 0, 1.72), "face": (0, -0.3, 1.72), "torso": (0, -0.1, 1.25), "body": (0, -0.3, 1.2)}

    def visible_objects(self) -> set:
        """Objects that count as 'seeing the character' when a ray from the camera hits them."""
        out = {self.body}
        stack = [self.head]
        while stack:
            o = stack.pop()
            out.add(o)
            stack.extend(o.children)
        return out

    def face_points(self) -> list:
        """World points that must be on screen and unblocked for a face shot."""
        mw = self.head.matrix_world
        return [mw @ Vector(p) for p in ((0, -0.3, 0.05), (0.1, -0.26, 0.05), (-0.1, -0.26, 0.05), (0, -0.28, -0.1))]

    def frame_points(self) -> list:
        mw = self.head.matrix_world
        # close-up framing: eyes, mouth and both ears in frame; cropping the top of the hair is fine
        return [mw @ Vector(p) for p in ((0.1, -0.26, 0.08), (-0.1, -0.26, 0.08), (0, -0.28, -0.14),
                                         (0.29, 0, 0), (-0.29, 0, 0))]

    # ------------------------------------------------------------ arms
    @staticmethod
    def _mirror(v, s):
        return Vector((v[0] * s, v[1], v[2]))

    def _pole_pos(self, s, hand, pole):
        sh = self.SH[s]
        ax = (hand - sh).normalized()
        n = (pole - ax * pole.dot(ax)).normalized()
        return (sh + hand) / 2 + n * 0.6

    def _arm_key(self, s, f, hand, pole):
        hand = Vector(hand)
        key(self.hand_t[s], "location", f, hand)
        key(self.pole_t[s], "location", f, self._pole_pos(s, hand, Vector(pole)))

    def pose(self, f0: int, f1: int, name: str):
        hand, pole = POSES[name]
        for s in (1, -1):
            self.hand_t[s].keyframe_insert("location", frame=f0)
            self.pole_t[s].keyframe_insert("location", frame=f0)
            self._arm_key(s, f1, self._mirror(hand, s), self._mirror(pole, s))

    def hold_arms(self, f: int):
        for s in (1, -1):
            self.hand_t[s].keyframe_insert("location", frame=f)
            self.pole_t[s].keyframe_insert("location", frame=f)

    def typing(self, f0: int, f1: int, speed: str = "calm"):
        step, amp = TYPING[speed]
        hand, pole = POSES["type"]
        for i, f in enumerate(ticks(f0, f1 + 1, step)):
            for s in (1, -1):
                up = amp if (i + (s > 0)) % 2 else 0.0
                h = self._mirror(hand, s) + Vector((self.rng.uniform(-.012, .012), self.rng.uniform(-.02, .02), up))
                self._arm_key(s, f, h, self._mirror(pole, s))
            if i:
                self.events.append({"frame": f, "kind": "key_click"})

    def tap(self, f: int):
        """Both hands lift and land: 'done!'"""
        hand, pole = POSES["type"]
        for s in (1, -1):
            self._arm_key(s, f, self._mirror(hand, s) + Vector((0, 0, 0.06)), self._mirror(pole, s))
            self._arm_key(s, f + n(6), self._mirror(hand, s), self._mirror(pole, s))
        self.events.append({"frame": f + n(6), "kind": "tap"})

    def celebrate(self, f0: int, f1: int):
        self.pose(f0, f0 + n(12), "arms_up")
        hand, pole = POSES["arms_up"]
        for i, f in enumerate(ticks(f0 + n(16), f1, 6)):
            w = 0.06 if i % 2 else -0.06
            for s in (1, -1):
                self._arm_key(s, f, self._mirror(hand, s) + Vector((w, 0, 0.03 if i % 2 else 0)), self._mirror(pole, s))
        self.events.append({"frame": f0 + n(6), "kind": "celebrate"})

    # ------------------------------------------------------------ body + head
    def _set_spine(self, f0, f1, pitch=None, roll=0.0):
        if pitch is not None:
            self.spine_pitch = pitch
        tween(self.spine, "rotation_euler", f0, f1, (R(self.spine_pitch), R(roll), 0))

    def _set_neck(self, f0, f1, pitch=None, yaw=0.0):
        if pitch is not None:
            self.neck_pitch = pitch
        tween(self.neck, "rotation_euler", f0, f1, (R(self.neck_pitch), 0, R(yaw)))

    def lean(self, f0: int, f1: int, amount):
        self._set_spine(f0, f1, LEANS.get(amount, amount) if isinstance(amount, str) else amount)

    def look(self, f0: int, f1: int, at):
        self._set_neck(f0, f1, LOOKS.get(at, at) if isinstance(at, str) else at)

    def head_bob(self, f0: int, f1: int):
        base = self.neck_pitch
        for i, f in enumerate(ticks(f0 + n(12), f1, 12)):
            key(self.neck, "rotation_euler", f, (R(base + 3 * (i % 2)), R(2 * ((i % 3) - 1)), 0))

    def shake_head(self, f0: int, f1: int, every: int = 7):
        for i, f in enumerate(ticks(f0, f1, every)):
            side = 1 if i % 2 else -1
            key(self.neck, "rotation_euler", f, (R(self.neck_pitch), 0, R(10 * side)))
            key(self.spine, "rotation_euler", f, (R(self.spine_pitch), R(4 * side), 0))
            self.events.append({"frame": f, "kind": "whoosh"})
        key(self.neck, "rotation_euler", f1, (R(self.neck_pitch), 0, 0))
        key(self.spine, "rotation_euler", f1, (R(self.spine_pitch), 0, 0))

    def gulp(self, f: int):
        p = self.neck_pitch
        tween(self.neck, "rotation_euler", f - n(6), f, (R(p + 6), 0, 0))
        tween(self.neck, "rotation_euler", f, f + n(8), (R(p), 0, 0))
        self.events.append({"frame": f, "kind": "gulp"})

    def expression(self, f: int, face: str, dur: int = 6):
        eyes, pupil, brows, mouth, size = FACES[face]
        for s in (1, -1):
            tween(self.eyes[s], "scale", f, f + dur, eyes)
            tween(self.pupils[s], "scale", f, f + dur, (pupil,) * 3)
        ang, z = BROWS[brows]
        for s in (1, -1):
            b = self.brows[s]
            b.keyframe_insert("rotation_euler", frame=f + n(2))
            b.keyframe_insert("location", frame=f + n(2))
            b.rotation_euler, b.location = (0, R(ang * s), 0), (0.1 * s, -0.275, z)  # (the length is in the mesh)
            b.keyframe_insert("rotation_euler", frame=f + dur + n(2))
            b.keyframe_insert("location", frame=f + dur + n(2))
        for name, ob in self.mouths.items():
            full = (0.8, 0.35, 1.0) if name == "o" else (size,) * 3
            tween(ob, "scale", f + max(0, dur - n(3)), f + dur, full if name == mouth else (0, 0, 0))

    def breathe(self, f0: int, f1: int, period: float = 3.4, pitch: float = 0.7):
        """Idle from f0 to f1: the torso rocks a little, once every `period` seconds."""
        half = n(period * 12)  # half a period, in frames
        for i, f in enumerate(range(f0, f1 + half, half)):
            up = i % 2
            key(self.breath, "rotation_euler", f, (R(pitch if up else -pitch * 0.4), 0, 0))

    def blink(self, f: int, dur: int = 3):
        for s in (1, -1):
            sc = tuple(self.eyes[s].scale)
            key(self.eyes[s], "scale", f - dur, sc)
            key(self.eyes[s], "scale", f, (sc[0], sc[1], 0.08))
            key(self.eyes[s], "scale", f + dur, sc)

    def sweat(self, f0: int, f1: int):
        d = self.sweat_drop
        key(d, "scale", f0 - n(4), (0, 0, 0))
        key(d, "scale", f0, (1, 1, 1.5))
        key(d, "location", f0, (0.27, -0.12, 0.15))
        key(d, "location", f1, (0.27, -0.17, -0.1))
        key(d, "scale", f1, (1, 1, 1.5))
        key(d, "scale", f1 + n(3), (0, 0, 0))
        self.events.append({"frame": f0, "kind": "plip"})

    def tremble(self, f0: int, f1: int):
        for f in ticks(f0, f1, 3):
            for s in (1, -1):
                key(self.pupils[s], "location", f,
                    (self.rng.uniform(-.005, .005), -0.052, -0.012 + self.rng.uniform(-.004, .004)))
        for s in (1, -1):
            key(self.pupils[s], "location", f1, (0, -0.052, -0.012))

    # ------------------------------------------------------------ checks
    def ik_error(self, frames) -> float:
        """Largest wrist-to-target distance (m) over the given frames: a pose the arm can't reach shows here."""
        sc = bpy.context.scene
        worst = 0.0
        for f in frames:
            sc.frame_set(f)
            for s, sd in ((1, "L"), (-1, "R")):
                wrist = self.rig.matrix_world @ self.rig.pose.bones["fore." + sd].tail
                worst = max(worst, (wrist - self.hand_t[s].matrix_world.translation).length)
        return worst
