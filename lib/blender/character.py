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
character. It is built at the origin, sitting, facing -Y (one character per set for now). The clay cast's mesh is
made seated; a fitted model ("model" in the description, from fit_model.py) is bound in its own A-pose and seated by
the rig.
"""
from __future__ import annotations

import json
import math
import random

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

import face as fitted_face
import kit
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
    "down":              ((0.38, -0.05, -0.07), (0.4, 1, 0.1)),   # arms at the sides: standing and walking
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
POSTURES = ("sit", "stand")
SEAT = 0.62  # how high the clay cast's pelvis sits on its chair (m); a fitted model's comes from its legs
# Every length below written as n(x) is frames at 24 fps, converted to the scene's rate.


def skeleton():
    """The seated rest pose every character shares: joints as (position, Skin radii), and the edges between them."""
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
    return SK, edges


class ClayPerson:
    def __init__(self, cid: str, look: dict):
        """`look` is a description from cast.resolve(): body, face, hair, clothes, accessories, seed."""
        self.id, self.desc = cid, look
        self.rng = random.Random(look["seed"])
        self.shape_rng = random.Random(look["seed"] * 7919 + 1)  # geometry draws: never disturbs the acting's
        self.move_rng = random.Random(look["seed"] * 104729 + 7)  # how a posture verb varies: its own draws too
        cl = look["clothes"]
        self.m = {
            "skin": mat(f"{cid}_skin", look["body"]["skin"], 0.55), "top": mat(f"{cid}_top", cl["top_color"], 0.7),
            "pants": mat(f"{cid}_pants", cl["pants"], 0.8), "shoes": mat(f"{cid}_shoes", cl["shoes"], 0.5),
            "hair": mat(f"{cid}_hair", look["hair"]["color"], 0.6), "eyew": mat(f"{cid}_eyew", "#F3F3F3", 0.15),
            "iris": mat(f"{cid}_iris", look["face"]["eyes"], 0.2),
            "frame": mat(f"{cid}_frame", (0.03, 0.03, 0.035), 0.3),
            "rivet": mat(f"{cid}_rivet", (0.75, 0.76, 0.78), 0.3, 1.0),
            "gold": mat(f"{cid}_gold", (0.9, 0.62, 0.2), 0.25, 1.0),
            "pupil": mat(f"{cid}_pupil", (0.02, 0.02, 0.025), 0.05),
            "glint": mat(f"{cid}_glint", (1, 1, 1), 0.1, emit=6),
            "mouth": mat(f"{cid}_mouth", (0.35, 0.07, 0.07), 0.4),
            "mouthin": mat(f"{cid}_mouthin", (0.12, 0.02, 0.02), 0.5),
            "sweat": mat(f"{cid}_sweat", (0.55, 0.82, 1.0), 0.03, emit=0.4),
        }
        self.events: list[dict] = []
        self.foot_t, self.knee_t, self.foot_at, self.foot_yaw = {}, {}, {}, {}
        self.touches: list[dict] = []  # each time a hand was put on something: how far off it was
        self.moves: list[tuple] = []   # each posture verb's stretch (verb, first frame, last), for how_it_moves
        self.walks: list[tuple] = []   # each walk (first frame, last, curved or not), for walk_report
        self.grips: list[dict] = []    # each time a hand with fingers held something: how open it still was
        self.posture, self.at, self.yaw, self.arm_pose = "sit", Vector((0, 0)), 0.0, "type"
        self._build_body()
        self._build_head()
        self.under = {}     # each hand's underside below its wrist, by shape (_hand_under)
        self.hand_pts = {}  # each hand's surface in its own frame, by shape (_hand_points)
        self.flat_pitch = {}  # how far each hand slopes laid flat (_pitch)
        self.hand_skin = None  # which vertices are the hands' own skin (a fitted model's grafted hands)
        self.hang_at = {}   # where a hanging or swinging arm's wrist goes, by side, swing and elbow (_hang)
        self.keys = None    # the laptop's keys, if the set has them (compile sets it): hands are put on them
        self.typed = []     # each stretch of typing with fingers (first frame, last), for typing_report
        self.swung = {}     # each arm's last swing key (frame, swing, elbow, out), for the keys between (_swing)
        self.wrist_q = {}   # each hand's last turn on the forearm (so the next key goes the short way round)
        self.palm_off = []  # each time a hand was laid on something: how far its palm was from facing it (degrees)
        self.hold_at, self.holding = {}, {1: False, -1: False}  # where a hand holds a bar (the hand bone's space)
        self._finger_setup()
        self.neck_pitch, self.spine_pitch, self.height = 7.0, 0.0, self.seat_h
        key(self.spine, "rotation_euler", 1, (0, 0, 0))
        key(self.neck, "rotation_euler", 1, (R(self.neck_pitch), 0, 0))
        for s in (1, -1):
            self._arm_key(s, 1, self._mirror(self._pose("type")[0], s), self._mirror(self._pose("type")[1], s))
        if look.get("model"):  # (its legs rest standing: the rig seats it)
            self.place("sit")

    # ------------------------------------------------------------ build
    def _build_body(self):
        cid, M = self.id, self.m
        model = self.desc.get("model")
        SK, edges = skeleton()
        # `drop` is how much lower than the clay cast this character's pelvis is when sitting (0 for the clay cast)
        self.seat, self.drop, self.belly = Vector((0, 0, 0)), 0.0, None
        if model:
            # a fitted model keeps its own proportions and its own A-pose: the rig is built on its joints, standing,
            # and the bones (not the mesh) put it in the chair. Its seat is as high as its own knees: thighs
            # level, feet flat on the floor.
            self.body = self._model_body(model)
            joints = {k: Vector(v) for k, v in json.loads(self.body["joints"]).items()}
            seat_h = joints["anL"].z + (joints["knL"] - joints["anL"]).length + joints["pelvis"].z - joints["hipL"].z
            self.drop = SEAT - seat_h
            self.seat = Vector((0, 0, seat_h)) - joints["pelvis"]
            self.body.location = self.seat
            for name, at in joints.items():
                SK[name] = (tuple(at + self.seat), SK[name][1])
        self.seat_h = SEAT - self.drop
        self.root = empty(f"{cid}_root")  # the spot on the floor under the pelvis, and which way they face
        self.hips = empty(f"{cid}_hips", self.root, (0, 0, self.seat_h))
        self.breath = empty(f"{cid}_breath", self.hips)  # a slow idle under every lean, so a hold is never a freeze
        self.spine = empty(f"{cid}_spine", self.breath)
        top = None if model else self.desc["clothes"]["top"]
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

        # how far this character's neck and head sit above the pelvis, against the clay cast's (0 for the clay cast)
        self.rise = SK["neckb"][0][2] - 1.33 + self.drop
        self.neck = empty(f"{cid}_neck", self.spine, (0, 0, 0.8 + self.rise))
        for name, (at, (rx, ry)) in list(SK.items()):  # the build: a thicker or thinner torso and limbs
            k = torso if name in ("waist", "chest", "upch") else limb if name[:2] in ("sh", "el", "hi", "kn") else 1
            SK[name] = (at, (rx * k, ry * k))
        if not model:
            self.body = self._bake_body(SK, edges, top)
        self._build_rig(SK, self.body, a_pose=bool(model))

    def _model_body(self, path):
        """A body fitted to this skeleton by fit_model.py: one textured mesh, head included, already weighted."""
        with D.libraries.load(path) as (src, dst):
            if "model_body" not in src.objects:
                raise RuntimeError(f"{path} has no object named model_body: make it with lib/blender/fit_model.py")
            dst.objects = ["model_body"]
        body = link(dst.objects[0])
        if body.get("rest") != "a_pose":
            raise RuntimeError(f"{path} was fitted in the seated pose, which the rig no longer reads: "
                               "run lib/blender/fit_model.py on the model again")
        body.name = f"{self.id}_body"
        return body

    def _bake_body(self, SK, edges, top):
        cid, M = self.id, self.m
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
        body = link(D.objects.new(f"{cid}_body", body_me))

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

        return body

    def _build_rig(self, SK, body, a_pose=False):
        """The armature, the body bound to it, and the controls. The clay cast's rest pose is seated (its mesh is
        baked seated). A fitted model's rest pose is its own A-pose (`a_pose`): its mesh is bound as it was made,
        and the leg and arm IK seat it, so nothing in the mesh is stretched or pre-bent."""
        cid = self.id
        J = {name: Vector(at) for name, (at, _) in SK.items()}
        if a_pose:
            # a straight limb gives IK no side to bend to: put the elbows a little back and the knees a little
            # forward of the line through the limb (the bones only: the mesh is not touched)
            for sd in "LR":
                for a, mid, b, way in (("sh", "el", "wr", 1), ("hip", "kn", "an", -1)):
                    J[mid + sd].y = (J[a + sd].y + J[b + sd].y) / 2 + way * 0.04 * (J[b + sd] - J[a + sd]).length
        P = lambda n: J[n]  # noqa: E731

        # 3) armature in the rest pose
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
        # fingers, if the fit found any (hands.py): two bones each, root to middle and middle to tip
        found = json.loads(body["fingers"]) if a_pose and "fingers" in body else {}
        self.fingers, self.palm, self.hand_way = {1: [], -1: []}, {}, {}
        for sd, hand in found.items():
            s = 1 if sd == "L" else -1
            self.palm[s] = Vector(hand["palm"]).normalized()
            self.hand_way[s] = Vector(hand["axis"]).normalized()  # the way the fingers point (not quite the bone's)
            for f in hand["fingers"]:
                a, m, t = (Vector(j) + self.seat for j in f["joints"])
                bone(f"{f['name']}.a.{sd}", a, m, "hand." + sd)
                bone(f"{f['name']}.b.{sd}", m, t, f"{f['name']}.a.{sd}", True)
                self.fingers[s].append({"name": f["name"], "way": (t - a).normalized(), "length": (m - a).length + (t - m).length})
        bpy.ops.object.mode_set(mode="OBJECT")

        # 4) bone-heat weights (distance weights bled the chest into the arms when they were raised)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        body.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        if not a_pose:
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
            sub = body.modifiers.new("Subsurf", "SUBSURF")
            sub.levels, sub.render_levels = 0, 1
        else:  # a fitted model carries its own weights, in vertex groups named for these bones
            bpy.ops.object.parent_set(type="ARMATURE")
            for mod in body.modifiers:
                if mod.type == "ARMATURE":  # a bent elbow or knee keeps its thickness (dual quaternion skinning)
                    mod.use_deform_preserve_volume = True

        # 5) controls: spine/neck empties -> Child Of; wrist + pole empties -> IK (pole angle +90 for roll-0 bones)
        self.SH = {s: Vector((P("shL").x * s, 0, P("shL").z - 0.62 + self.drop)) for s in (1, -1)}
        # arm length against the clay cast's (1 for the clay cast)
        self.arm = ((P("elL") - P("shL")).length + (P("wrL") - P("elL")).length) / 0.7122 if a_pose else 1
        self.hand_len = (P("tipL") - P("wrL")).length
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
            if a_pose:  # (the targets, at the resting wrist and behind the resting elbow, are keyed after this)
                root, mid, end = (self.rig.data.bones[b].head_local for b in ("upper." + sd, "fore." + sd, "hand." + sd))
                self.hand_t[s].location = end - Vector((0, 0, self.seat_h))
                self.pole_t[s].location = mid + (mid - (root + end) / 2).normalized() - Vector((0, 0, self.seat_h))
                self._fit_pole(c, "fore." + sd)

    # ------------------------------------------------------------ hands
    # degrees each finger's two bones curl towards the palm, then the thumb's two, whether the index stays out,
    # and how far the thumb lies out to the side (0: as it was made, standing off the palm to close on
    # something; 1: flat beside the first finger, in the palm's plane)
    HAND_SHAPES = {
        "open":    ((0, 0), (0, 0), False, 0.5),
        "relaxed": ((24, 20), (6, 10), False, 0.5),
        "grip":    ((58, 64), (30, 36), False, 0.0),
        "flat":    ((5, 4), (0, 6), False, 0.6),    # laid on a table top
        "type":    ((20, 15), (10, 14), False, 0.6),  # on a keyboard (lay "keys"): the fingertips come down onto the keys
        "fist":    ((84, 86), (40, 50), False, 0.3),
        "point":   ((84, 86), (40, 50), True, 0.3),
    }

    def _finger_setup(self):
        """Work out which way each finger curls (about an axis across it, towards the palm; the thumb towards
        the fingers), and rest the hands relaxed. A fitted model without fingers has none of this."""
        pb = self.rig.pose.bones
        self.hand_shape_now = {1: "relaxed", -1: "relaxed"}
        me = self.body.data
        at = self.body.get("hand_material", len(me.materials) - 1 if len(me.materials) > 1 else None)
        if at is not None:  # (the grafted hands keep their own material: their skin, heel to fingertips)
            idx = np.empty(len(me.polygons), dtype=np.int32)
            me.polygons.foreach_get("material_index", idx)
            skin = np.zeros(len(me.vertices), dtype=bool)
            for poly in np.nonzero(idx == int(at))[0]:
                skin[list(me.polygons[int(poly)].vertices)] = True
            self.hand_skin = skin if skin.any() else None
        for s, sd in ((1, "L"), (-1, "R")):
            hand = pb.get("hand." + sd)
            if hand:
                hand.rotation_mode = "QUATERNION"
            fingers = self.fingers.get(s, [])
            if not fingers:
                continue
            along = self.hand_way[s]
            # where a bar is held: past the knuckles in the crook of the fingers, off the palm by a finger's thickness
            bone = self.rig.data.bones["hand." + sd]
            size = sum(f["length"] for f in fingers[1:]) / max(1, len(fingers) - 1)
            self.hold_at[s] = bone.matrix_local.inverted() @ (bone.tail_local + along * 0.3 * size + self.palm[s] * 0.35 * size)
            roots = [self.rig.data.bones[f"{f['name']}.a.{sd}"].head_local for f in fingers]
            for f, at in zip(fingers, roots):
                towards = self.palm[s]
                if f["name"] == "thumb":  # a thumb closes across the palm, towards the first finger
                    towards = (0.6 * self.palm[s] + 0.8 * (roots[1] - at).normalized()).normalized()
                axis = f["way"].cross(towards).normalized()
                if f["name"] == "thumb":
                    # generators make the thumb standing straight off the palm: how far round the hand's own
                    # length it has to swing to lie beside the first finger, pointing away from the others
                    flat = lambda v: (v - along * v.dot(along)).normalized()  # noqa: E731
                    names = [g["name"] for g in fingers]
                    first, last = roots[names.index("index")], roots[names.index(names[1] if names[1] != "index" else names[-1])]
                    now, want = flat(f["way"]), flat(first - last)
                    f["out"] = (self.rig.data.bones[f"thumb.a.{sd}"].matrix_local.to_3x3().transposed() @ along,
                                math.atan2(now.cross(want).dot(along), now.dot(want)))
                f["axes"] = []
                for part in "ab":
                    b = pb[f"{f['name']}.{part}.{sd}"]
                    b.rotation_mode = "QUATERNION"
                    f["axes"].append((b, self.rig.data.bones[b.name].matrix_local.to_3x3().transposed() @ axis))
            self._fingers_key(s, 1, "relaxed")

    def _fingers_set(self, s, shape, lift=None):
        """Pose hand `s`'s fingers in a shape; `lift` is {finger: degrees} that finger is raised from it (typing)."""
        (fa, fb), (ta, tb), point, out = self.HAND_SHAPES[shape]
        bones = []
        for finger in self.fingers.get(s, []):
            if finger["name"] == "thumb":
                a, b = ta, tb
            elif point and finger["name"] == "index":
                a, b = 4, 4
            else:
                a, b = fa, fb
            up = (lift or {}).get(finger["name"], 0.0)
            a, b = a - up, b - 0.5 * up
            for i, ((bone, axis), deg) in enumerate(zip(finger["axes"], (a, b))):
                q = Quaternion(axis, R(deg))
                if i == 0 and "out" in finger:
                    q = Quaternion(finger["out"][0], finger["out"][1] * out) @ q
                bone.rotation_quaternion = q
                bones.append(bone)
        return bones

    def _fingers_key(self, s, f, shape, lift=None):
        for bone in self._fingers_set(s, shape, lift):
            bone.keyframe_insert("rotation_quaternion", frame=f)
        self.hand_shape_now[s] = shape

    # degrees a typing hand slopes down from the wrist (lay "keys"), as a typist's does. It needs the thumb to lie
    # in to the palm (hands.lay_thumb does that in the fit): a generated mitt's thumb stands 40% of the hand's length
    # under the palm, and its fingertips only come lower than it at about 45 degrees, which reads as a paw
    TYPE_PITCH = 20.0

    def _hand_under(self, s, shape="flat"):
        """How far the underside of hand `s` is below its wrist when it lies palm down in a shape, and how far
        below the wrist its knuckles are (m), measured on the mesh with the fingers posed: a generated hand is
        cupped and thick, and its thumb stands off the palm, so neither can be told from its bones. This is
        what sets how high a wrist is held over a desk or a leg for the hand to lie on it and not in it."""
        got = self.under.get((s, shape))
        if got:
            return got
        pts, knuckle = self._hand_points(s, shape)
        pitch = self._pitch(s, shape)
        if pitch:
            # (a hand on a surface slopes down from the wrist: the depth below the wrist is along the fingers as
            # well as off the palm: the fingertips reach the keys, and a flat hand rests on its heel and its
            # fingers both, not on the heel alone)
            down = np.array([math.sin(R(pitch)), math.cos(R(pitch)), 0.0])
            self.under[(s, shape)] = (float((pts @ down).max()), float(knuckle @ down))
        else:
            self.under[(s, shape)] = (float(pts[:, 1].max()), float(knuckle[1]))
        return self.under[(s, shape)]

    # shapes a hand takes lying on something: measured whole, the heel of the hand too
    ON_SURFACE = ("flat", "type")

    def _pitch(self, s, shape):
        """Degrees hand `s` slopes down from the wrist lying on a surface in a shape: a typist's slope on the
        keys, and laid flat, whatever puts the heel of the hand and the fingers at the same depth (a generated
        hand's heel is deep: held level, it rests on the heel and the fingers stand off)."""
        if shape == "type":
            return self.TYPE_PITCH
        if shape != "flat" or not self.fingers.get(s):
            return 0.0
        got = self.flat_pitch.get(s)
        if got is None:
            pts = self._hand_points(s, shape)[0]
            front, heel = pts[pts[:, 0] > 0.02], pts[pts[:, 0] <= 0.0]
            lo, hi = -10.0, 30.0
            if len(front) and len(heel):
                for _ in range(30):  # (the fingers come down as the hand tips: deeper in front past the heel)
                    mid = 0.5 * (lo + hi)
                    d = np.array([math.sin(R(mid)), math.cos(R(mid)), 0.0])
                    lo, hi = (mid, hi) if (front @ d).max() < (heel @ d).max() else (lo, mid)
            got = self.flat_pitch[s] = 0.5 * (lo + hi) if len(front) and len(heel) else 0.0
        return got

    def _hand_points(self, s, shape):
        """Hand `s`'s surface with its fingers in a shape, in the hand's own frame: along the fingers, off the
        palm, and across (way x palm), from the wrist; and its knuckles in the same frame. Measured on the mesh
        with the fingers posed (and cached), so a hand can be put anywhere and tested against the body."""
        got = self.hand_pts.get((s, shape))
        if got:
            return got
        sd = "L" if s > 0 else "R"
        pb = self.rig.pose.bones
        mine = [pb[f"{f['name']}.{part}.{sd}"] for f in self.fingers[s] for part in "ab"]
        was = [b.rotation_quaternion.copy() for b in mine]
        self._fingers_set(s, shape)
        bpy.context.view_layer.update()
        ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        mw = np.array(ev.matrix_world)
        W = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        names = {"hand." + sd} | {b.name for b in mine}
        groups = {g.index for g in self.body.vertex_groups if g.name in names}
        mask = np.array([sum(g.weight for g in v.groups if g.group in groups) > 0.5 for v in self.body.data.vertices])
        whole = shape in self.ON_SURFACE and self.hand_skin is not None  # (the heel too: it is what a hand rests on)
        if whole:
            mask &= self.hand_skin
        bone, rest = pb["hand." + sd], self.rig.data.bones["hand." + sd]
        spin = (self.rig.matrix_world @ bone.matrix).to_3x3() @ rest.matrix_local.to_3x3().inverted()
        palm, way = np.array(spin @ self.palm[s]), np.array(spin @ self.hand_way[s])
        wrist, knuckle = self.rig.matrix_world @ bone.head, self.rig.matrix_world @ bone.tail
        rel = W[mask] - np.array(wrist)
        if not whole:
            rel = rel[rel @ way > 0.0]  # (not the stub that sits inside the cuff)
        for b, q in zip(mine, was):
            b.rotation_quaternion = q
        across = np.cross(way, palm)
        frame = np.stack([way, palm, across], axis=1)
        self.hand_pts[(s, shape)] = (rel @ frame, np.array(knuckle - wrist) @ frame)
        return self.hand_pts[(s, shape)]

    def _desk_lift(self, s, shape="type"):
        """How much higher (m, 0 or less) this hand's wrist is held over the keys than a pose puts it, for the hand
        in a shape to lie on them: the desk is set for the deeper of the two hands typing (the fingertips curled
        down onto the keys), so a hand laid flat comes down by what its fingertips no longer reach."""
        if not (self.fingers.get(1) and self.fingers.get(-1)):
            return 0.0
        return self._hand_under(s, shape)[0] - max(self._hand_under(1, "type")[0], self._hand_under(-1, "type")[0])

    def hand_shape(self, f0: int, f1: int, shape: str, hand: str | None = None):
        """The fingers take a shape by f1: open, relaxed, grip, point or fist. `hand` is "left" or "right"
        (the character's own); with none, both. A character without fingers (the clay cast) is left as it is."""
        if shape not in self.HAND_SHAPES:
            raise ValueError(f"unknown hand shape {shape!r}: one of {', '.join(self.HAND_SHAPES)}")
        for s in ((1, -1) if hand is None else (1 if hand == "left" else -1,)):
            if self.fingers.get(s):
                self._fingers_key(s, f0, self.hand_shape_now[s])
                self._fingers_key(s, f1, shape)

    WRIST_BEND = 65.0   # degrees a hand can bend at the wrist from the forearm's line
    WRIST_SIDE = 20.0   # and how far it turns sideways toward straight ahead when it lies on a desk

    def _wrist_key(self, s, f, hand, lay=None, was=None, mix=1.0):
        """Turn the hand on the forearm for frame f, from the arm as it is posed on that frame (the forearm's
        own twist comes out of the IK, so a turn worked out in the rest pose points the palm somewhere else).
        With no `lay` the hand follows the forearm and only rolls: hanging at the side the palm faces the leg,
        and as the hand comes forward it turns palm down, as a hand does to take a handle. `lay` is how it lies
        on something: "flat" (palm down, the hand level, on a desk), "side" (palm to the leg all through a
        swing) or a function of the frame giving the way
        the palm faces and the way the fingers point, in the room (on a thigh). `was` and `mix` blend from one
        lay to another on the way between two things. Only a hand with fingers shows which way it faces, so
        the clay cast is left alone."""
        if not self.fingers.get(s):
            return
        sc = bpy.context.scene
        if sc.frame_current != f:
            sc.frame_set(f)
        bpy.context.view_layer.update()
        sd = "L" if s > 0 else "R"
        bone, rest = self.rig.pose.bones["hand." + sd], self.rig.data.bones["hand." + sd]
        plain = bone.matrix.to_3x3() @ bone.rotation_quaternion.to_matrix().inverted()  # with no turn of its own
        posed = plain @ rest.matrix_local.to_3x3().inverted()
        way0, palm0 = (posed @ self.hand_way[s]).normalized(), (posed @ self.palm[s]).normalized()
        to_rig = self.rig.matrix_world.to_3x3().inverted()
        own = self.root.matrix_world.to_3x3()
        ahead = self.SH[s].y - hand.y  # how far in front of the shoulder
        up = hand.z - self.SH[s].z
        t = max(0.0, min(1.0, (ahead - 0.12 * self.arm) / (0.2 * self.arm)))
        t = t * t * (3 - 2 * t) * (1.0 if up < 0.1 else max(0.0, 1 - (up - 0.1) / 0.2))

        def turn(how):
            if how is None:
                palm, way = own @ Vector((-s, 0, 0)).lerp(Vector((0, 0, -1)), t), None
            elif how == "side":  # an arm swinging in a walk: the palm stays to the leg, or the hand flaps
                palm, way = own @ Vector((-s, 0, 0)), None
            elif how in ("flat", "keys"):
                palm, way = Vector((0, 0, -1)), how
            else:
                palm, way = how(f)
            palm = to_rig @ Vector(palm)
            if way is None:
                way = way0
            else:
                if isinstance(way, str):  # level (or sloping), and turned a little toward straight ahead
                    flat = Vector((way0.x, way0.y, 0.0))
                    flat = flat.normalized() if flat.length > 1e-3 else to_rig @ (own @ Vector((0, -1, 0)))
                    fwd = to_rig @ (own @ Vector((0, -1, 0)))
                    side = flat.angle(fwd, 0.0)
                    way = flat.slerp(fwd, min(1.0, R(self.WRIST_SIDE) / side)) if side > 1e-3 else flat
                    pitch = self._pitch(s, "type" if how == "keys" else "flat")
                    if pitch:  # (sloping down from the wrist onto the keys or the desk, the palm square to it)
                        down = to_rig @ Vector((0, 0, -1))
                        way = (way * math.cos(R(pitch)) + down * math.sin(R(pitch))).normalized()
                        palm = (down - way * down.dot(way)).normalized()
                else:
                    way = (to_rig @ Vector(way)).normalized()
                bend = way0.angle(way, 0.0)
                if bend > R(self.WRIST_BEND):
                    way = way0.slerp(way, R(self.WRIST_BEND) / bend).normalized()
            palm = palm - way * palm.dot(way)
            palm = palm.normalized() if palm.length > 1e-3 else palm0
            a = Matrix((way0, palm0, way0.cross(palm0))).transposed()
            b = Matrix((way, palm, way.cross(palm))).transposed()
            return (plain.inverted() @ (b @ a.inverted()) @ plain).to_quaternion()

        q = turn(lay)
        if mix < 1.0:
            q = turn(was).slerp(q, mix)
        last = self.wrist_q.get(s)
        if last is not None and last.dot(q) < 0:
            q = -q
        self.wrist_q[s] = q.copy()
        bone.rotation_quaternion = q
        bone.keyframe_insert("rotation_quaternion", frame=f)

    def _hold_point(self, s) -> Vector:
        """The point of hand `s` that goes on a thing, in the world, now: the knuckles, or for a hand with
        fingers that is taking hold of a bar, the crook of its fingers."""
        bone = self.rig.pose.bones["hand." + ("L" if s > 0 else "R")]
        if self.holding[s] and s in self.hold_at:
            return self.rig.matrix_world @ (bone.matrix @ self.hold_at[s])
        return self.rig.matrix_world @ bone.tail

    def grip_open(self, s, point) -> float:
        """How open hand `s` is round a point it holds, now: the mean distance of its fingertips from the point
        over the fingers' own length. About 1 is a flat hand laid on it, under 0.7 is closed round it."""
        sd = "L" if s > 0 else "R"
        tips = [self.rig.matrix_world @ self.rig.pose.bones[f"{f['name']}.b.{sd}"].tail for f in self.fingers[s][1:]]
        length = sum(f["length"] for f in self.fingers[s][1:]) / len(tips)
        return sum((t - Vector(point)).length for t in tips) / len(tips) / length

    def _fit_pole(self, con, tip):
        """Set an IK constraint's pole angle by trying it: with the target and pole where they are, the angle
        that leaves the chain's middle joint (the head of bone `tip`) nearest to where it rests. Worked out
        from the bones it depends on their roll and on Blender's conventions; measured, it does not."""
        rest = self.rig.data.bones[tip].head_local
        bone = self.rig.pose.bones[tip]

        def miss(angle):
            con.pole_angle = angle
            bpy.context.view_layer.update()
            return (bone.head - rest).length

        best = min((R(a) for a in range(-180, 180, 15)), key=miss)
        step = R(7.5)
        while step > R(0.05):
            best = min((best - step, best, best + step), key=miss)
            step /= 2
        con.pole_angle = best
        bpy.context.view_layer.update()

    def _leg_rig(self):
        """Legs and pelvis under control, made the first time a posture needs them (a clay character that only
        sits at its desk never gets them, and renders as it always has). The pelvis bone follows the hips control.
        Each foot has one control, free in the world, so a planted foot stays put while the body moves: the shin
        reaches for it (IK, the knee pointing where the foot points) and the foot takes its turn and tilt, staying flat on the floor however the knee bends."""
        if self.foot_t:
            return
        cid, bones, pb = self.id, self.rig.data.bones, self.rig.pose.bones
        c = pb["hips"].constraints.new("CHILD_OF")
        c.target, c.inverse_matrix = self.hips, Matrix.Translation((0, 0, -self.seat_h))
        self.thigh, self.shin = bones["thigh.L"].length, bones["shin.L"].length
        self.hip_at = bones["thigh.L"].head_local - Vector((0, 0, self.seat_h))  # the left hip joint, from the pelvis
        self.ankle_h = bones["foot.L"].head_local.z - self.seat.z         # the ankle above the sole
        toe = bones["foot.L"].tail_local - bones["foot.L"].head_local
        self.foot_len = math.hypot(toe.x, toe.y) * 1.25  # ankle to the tip of the shoe, along the floor
        self.toe_h = bones["foot.L"].tail_local.z - self.seat.z           # the foot bone's far end above the sole
        self.stand_h = self.ankle_h + 0.985 * (self.thigh + self.shin) - self.hip_at.z
        for s, sd in ((1, "L"), (-1, "R")):
            rest = bones["foot." + sd].head_local.copy()
            self.foot_t[s] = t = empty(f"{cid}_foot_t{s}", None, rest)
            self.knee_t[s] = empty(f"{cid}_knee_t{s}", t, (0, -0.8, 0.5))
            c = pb["shin." + sd].constraints.new("IK")
            c.target, c.pole_target, c.chain_count = t, self.knee_t[s], 2
            self._fit_pole(c, "shin." + sd)
            # the foot keeps the control's turn and tilt whatever the shin above it does (as a child of the
            # shin it would tip its toes into the floor whenever the knee bends): it copies a helper that
            # holds the foot bone's own resting direction, carried round by the control
            way = empty(f"{cid}_foot_way{s}", t)
            way.rotation_euler = bones["foot." + sd].matrix_local.to_euler()
            c = pb["foot." + sd].constraints.new("COPY_ROTATION")
            c.target, c.target_space, c.owner_space = way, "WORLD", "WORLD"
            self.foot_at[s], self.foot_yaw[s] = rest, 0.0

    def _build_head(self):
        cid, M = self.id, self.m
        face = self.desc["face"]
        fit = json.loads(self.body["face"]) if self.desc.get("model") else None
        self.head_r, self.brow_at, self.sweat_x, eye_parent = 0.3, (0.1, -0.275, 0.0), 0.27, {}
        self.mouth, self.lids = None, {}
        marks = json.loads(self.body["face"]) if fit else None  # (as fitted: the mesh's own space)
        if fit:
            # the model's head is part of the body mesh, on the neck bone. The face parts are made for a head
            # 0.3 m in radius, so they go on an empty scaled to the model's face: verbs and checks are unchanged
            for k in ("eyes", "brows"):  # (the fit is in the model's standing space)
                fit[k] = [list(Vector(p) + self.seat) for p in fit[k]]
            fit["mouth"] = list(Vector(fit["mouth"]) + self.seat)
            eyes = [Vector(e) for e in fit["eyes"]]  # character's right, then left: points on the face
            mid = (eyes[0] + eyes[1]) / 2
            k = (eyes[1].x - eyes[0].x) / 0.2
            at = Vector((mid.x, mid.y + 0.278 * k, mid.z - 0.05 * k))
            self.head = head = empty(f"{cid}_head", self.neck, at - Vector((0, 0, 1.42 + self.rise - self.drop)))
            head.scale = (k, k, k)
            self.head_at, self.head_r = tuple(at), fit["half_width"]
            local = lambda p: (Vector(p) - at) / k  # noqa: E731  a point on the model, in the head's own units
            ew, eh = fit["eye_size"][0] / k / 0.15, fit["eye_size"][1] / k / 0.15
            for s, e in ((-1, eyes[0]), (1, eyes[1])):
                eye_parent[s] = empty(f"{cid}_socket{s}", head, local(e) + Vector((0, 0.3 * 0.056 * ew, 0)))
                eye_parent[s].scale = (ew, ew * 0.5, eh)
            brow = local((Vector(fit["brows"][0]) + Vector(fit["brows"][1])) / 2)
            self.brow_at = ((fit["brows"][1][0] - fit["brows"][0][0]) / 2 / k, brow.y - 0.004, brow.z - 0.16)
            self.sweat_x = self.head_r / k + 0.02
            if "glasses" in self.desc["accessories"]:
                self._build_frames(head, (local(eyes[0]).y + local(eyes[1]).y) / 2)
        else:
            self.head = head = sphere(f"{cid}_head", 0.3, M["skin"], self.neck, (0, 0, 0.3), seg=48)
            head.scale = HEAD[face["shape"]]  # everything on the head is its child, so the face follows the shape
            for s in (1, -1):
                sphere(f"{cid}_ear", 0.06, M["skin"], head, (0.29 * s, 0, 0), (0.45, 1, 1), seg=16)
            nose_r, nose_scale = NOSE[face["nose"]]
            sphere(f"{cid}_nose", nose_r, M["skin"], head, (0, -0.3, -0.02), nose_scale, seg=16)
            self._build_hair(head)
            self._build_extras(head)
        self.brow_len, brow_h = BROW[face["brows"]]
        if fit:
            self.brow_len, brow_h = fit["brow_size"][0] / head.scale[0], fit["brow_size"][1] / head.scale[0]
        self.eyes, self.pupils, self.brows = {}, {}, {}
        for s in (1, -1):
            e = empty(f"{cid}_eye{s}", eye_parent[s]) if fit else empty(f"{cid}_eye{s}", head, (0.1 * s, -0.235, 0.05))
            self.eyes[s] = e
            sphere(f"{cid}_eyewhite", 0.075, M["eyew"], e, (0, 0, 0), (1, 0.75, 1))
            p = sphere(f"{cid}_pupil", 0.036, M["pupil"], e, (0, -0.052, -0.012))
            self.pupils[s] = p
            if face["eyes"].upper() != "#17110D":  # the default eye is all pupil; a colour gets an iris round it
                sphere(f"{cid}_iris", 0.052, M["iris"], p, (0, 0.016, 0), (1, 0.5, 1), seg=24)
            sphere(f"{cid}_glint", 0.011, M["glint"], p, (0.012, -0.03, 0.014), seg=12)
            self.brows[s] = box(f"{cid}_brow", (self.brow_len, 0.025, brow_h), M["hair"], head,
                                (self.brow_at[0] * s, self.brow_at[1], 0.16 + self.brow_at[2]),
                                bevel=min(0.01, brow_h * 0.4))
        if fit:
            # a fitted face has its own mouth: made on the model's lip line, with keys on the head for the corners,
            # cheeks and jaw (face.py). The clay cast keeps its three drawn shapes.
            FM = fitted_face.materials(cid, self.body, marks)
            fitted_face.body_keys(self.body, marks)
            self.mouths = {}
            self.mouth = fitted_face.mouth(f"{cid}_mouth", self.body, marks, {**M, **FM}, head,
                                           lambda p: (p + self.seat - Vector(self.head_at)) / head.scale[0])
            self.face_marks = marks
            self.lids = {s: fitted_face.eyelid(f"{cid}_eye{s}", self.eyes[s], FM["lid"], FM["lash"]) for s in (1, -1)}
            self._mouth_now = (0.0, 0.0, 0.0, 0.0)
            self._face_keys(1, 1, "neutral")
        else:
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
        self.sweat_drop = sphere(f"{cid}_sweat", 0.03, M["sweat"], head, (self.sweat_x, -0.12, 0.15), (0, 0, 0), seg=16)

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

    def _build_frames(self, head, face_y=-0.29):
        """Thick rectangular glasses, a separate item on the head: one front cut from a slab (two lens openings
        and a notch for the nose), a rivet at each top corner, and the arms."""
        cid, m = self.id, self.m["frame"]
        y, z, wide, tall, deep = face_y - 0.03, 0.035, 0.5, 0.155, 0.035
        front = box(f"{cid}_frames", (wide, deep, tall), m, head, (0, y, z), bevel=0.018)
        cutters = [box(f"{cid}_cut", (0.175, 0.2, 0.095), m, head, (0.124 * s, y, z - 0.012), bevel=0.028) for s in (1, -1)]
        cutters.append(box(f"{cid}_cut", (0.05, 0.2, 0.09), m, head, (0, y, z - tall / 2), bevel=0.02))
        solvers = bpy.types.BooleanModifier.bl_rna.properties["solver"].enum_items.keys()
        for c in cutters:
            mod = front.modifiers.new("cut", "BOOLEAN")
            mod.object, mod.operation = c, "DIFFERENCE"
            # (the exact solver returned nothing for the nose notch on 5.3; the names differ before 4.5)
            mod.solver = next(x for x in ("MANIFOLD", "FLOAT", "FAST") if x in solvers)
        bpy.context.view_layer.update()
        front.data = D.meshes.new_from_object(front.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        front.modifiers.clear()
        for c in cutters:
            D.objects.remove(c)
        for poly in front.data.polygons:
            poly.use_smooth = True
        for s in (1, -1):
            cyl(f"{cid}_rivet", 0.011, 0.008, self.m["rivet"], head, (0.22 * s, y - deep / 2, z + 0.047), (R(90), 0, 0))
            box(f"{cid}_temple", (0.024, 0.36, 0.04), m, head, (0.239 * s, y + 0.18, z + 0.04), bevel=0.006)

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

    def furniture(self) -> dict:
        """Where this character needs its furniture, against the clay cast's (all zero for the clay cast): `chair`
        is how much higher the seat is, `desk` how much further forward (-y is forward) and higher the desk and
        what stands on it are. The seat is at the character's own knee height; the desk puts the keyboard under
        its hands as it types with its elbows bent, and stands clear of its belly."""
        if not self.desc.get("model"):
            return {"chair": 0.0, "desk": (0.0, 0.0)}
        (x, y, z), _ = POSES["type"]
        hand = Vector(self.SH[1]) + (Vector((x, y, z)) - Vector((0.3, 0, 0.6))) * self.arm
        # (under the hand, not under the wrist: a thicker hand than the clay cast's needs the desk that much lower)
        dy, dz = hand.y - y, hand.z + self.seat_h - (z + SEAT) - max(0.0, self._girth()["hand"] - 0.05)
        if self.fingers.get(1) and self.fingers.get(-1):
            # a hand with fingers is measured as it types: the keys (0.925 up for the clay cast) come to its
            # fingertips, with a few millimetres of air
            under = max(self._hand_under(1, "type")[0], self._hand_under(-1, "type")[0])
            dz = hand.z + self.seat_h - under - 0.004 - 0.925
        if self.belly is None:  # the body's front at desk height (the mesh rests standing: the torso is the same)
            co = np.empty(len(self.body.data.vertices) * 3, dtype=np.float32)
            self.body.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) + np.array(self.seat, dtype=np.float32)
            top = 0.9 + dz
            band = co[(np.abs(co[:, 0]) < 0.2) & (co[:, 2] > top - 0.12) & (co[:, 2] < top + 0.05)]
            self.belly = float(band[:, 1].min()) if len(band) else 0.0
        dy = min(dy, self.belly - 0.05 + 0.35)  # the desk's near edge is at y -0.35 for the clay cast
        return {"chair": -self.drop, "desk": (dy, dz)}

    # ------------------------------------------------------------ anchors (for cameras and checks)
    def anchors(self) -> dict:
        x, y, z = getattr(self, "head_at", (0, 0, 1.72))
        reach = 0.3 * self.head.scale[1]
        return {"head": (x, y, z), "face": (x, y - reach, z), "torso": (0, -0.1, 1.25 + self.rise / 2 - self.drop),
                "body": (0, -0.3, 1.2 + self.rise / 2 - self.drop)}

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

    def feet_points(self) -> list:
        """World points that must be on screen and unblocked for a shot of the feet: each ankle and each toe."""
        bpy.context.view_layer.update()
        bones = self.rig.pose.bones
        return [self.rig.matrix_world @ p for sd in "LR" for p in (bones["foot." + sd].head, bones["foot." + sd].tail)]

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

    def _arm_key(self, s, f, hand, pole, lay=None, was=None, mix=1.0):
        hand = Vector(hand)
        key(self.hand_t[s], "location", f, hand)
        key(self.pole_t[s], "location", f, self._pole_pos(s, hand, Vector(pole)))
        self._wrist_key(s, f, hand, lay, was, mix)

    def _lay(self, name=None):
        """How the hands lie in a pose: on the keys to type (sloping down from the wrist, for hands with fingers),
        flat on the desk to rest, otherwise as the forearm leaves them."""
        name = name or self.arm_pose
        if name == "type" and self.fingers.get(1):
            return "keys"
        return "flat" if name in ("type", "rest") else None

    def _pose(self, name):
        """A pose's wrist target and elbow pole. The poses that reach for the head follow this character's head:
        how high it sits and how wide it is (the clay cast's head is 0.3 m in radius, centred 1.72 m up)."""
        (x, y, z), pole = POSES[name]
        if self.arm != 1 and name in ("type", "rest", "arms_up", "down"):
            # a fitted model has its own arms: the pose is the clay cast's, measured from the shoulder in arm
            # lengths, so a shorter arm bends as much and reaches less far (the set comes to the hands)
            at = Vector(self.SH[1]) + (Vector((x, y, z)) - Vector((0.3, 0, 0.6))) * self.arm
            if name in ("type", "rest"):  # (and out to the keyboard, if the desk had to stand off a round belly)
                at.y = y + self.furniture()["desk"][0]
                if self.fingers.get(1):  # (broad hands side by side, not into each other: checked as hands_apart)
                    at.x += 0.035
            if name == "down":  # as near the body as the hips and the hand allow, not a set distance out
                at.x = self._side_clear()
            return tuple(at), pole
        if name in ("hands_on_head", "hands_behind_head"):
            up = getattr(self, "head_at", (0, 0, 1.72))[2] - 1.72 + self.drop
            wide = self.head_r / 0.3
            x, y, z = (x - 0.3 + self.head_r, y, z + up) if name == "hands_on_head" else (x * wide, y * wide, z + up)
        elif name == "arms_up":
            z += self.rise
        return (x, y, z), pole

    def pose(self, f0: int, f1: int, name: str):
        self.arm_pose = name
        self.swung = {}
        hand, pole = self._pose(name)
        # (hands with fingers lie flat on a desk and hang or lift relaxed)
        shape = {"type": "type", "rest": "flat", "arms_up": "open"}.get(name, "relaxed")
        self.hand_shape(f0, f1, shape)
        for s in (1, -1):  # (both held before either moves: keying a hand with fingers looks at the frame it is for)
            self.hand_t[s].keyframe_insert("location", frame=f0)
            self.pole_t[s].keyframe_insert("location", frame=f0)
            if self.fingers.get(s):
                self.rig.pose.bones["hand." + ("L" if s > 0 else "R")].keyframe_insert("rotation_quaternion", frame=f0)
        for s in (1, -1):
            lift = Vector((0, 0, self._desk_lift(s, shape))) if self._lay(name) else Vector()
            if self._lay(name) and self.fingers.get(s) and f1 - f0 >= 4:
                # (lifted on the way: a hand slid along a desk goes through whatever lies on it)
                mid = self.hand_t[s].location.lerp(self._mirror(hand, s) + lift, 0.5) + Vector((0, 0, 0.035))
                self._arm_key(s, (f0 + f1) // 2, mid, self._mirror(pole, s), self._lay(name))
            end = self._mirror(hand, s) + lift
            if self._lay(name):
                end = self._on_keys(s, f1, end, shape)
            self._arm_key(s, f1, end, self._mirror(pole, s), self._lay(name))

    def hold_arms(self, f: int):
        for s in (1, -1):
            self.hand_t[s].keyframe_insert("location", frame=f)
            self.pole_t[s].keyframe_insert("location", frame=f)

    KEY_AIR = 0.002  # how far a hand resting on the keys is held off them (m)

    def _on_keys(self, s, f, at, shape):
        """Wrist target `at` (the chest's space) moved up or down for hand `s` in a shape to rest on the keys on
        frame f: the keys' height in the room and the hand's own depth below its wrist, measured, and not the
        pose's height (the body's lean and how it sits put a pose's wrist a couple of centimetres off)."""
        if self.keys is None or not self.fingers.get(s):
            return at
        top = getattr(self, "keys_top", None)
        if top is None:
            top = self.keys_top = max((self.keys.matrix_world @ v.co).z for v in self.keys.data.vertices)
        sc = bpy.context.scene
        if sc.frame_current != f:
            sc.frame_set(f)
        mw = self.spine.matrix_world
        world = mw @ Vector(at)
        world.z = top + self.KEY_AIR + self._hand_under(s, shape)[0]
        return mw.inverted() @ world

    TYPE_LIFT = 24.0  # degrees a finger lifts off the keys before it comes down on the next one

    def typing(self, f0: int, f1: int, speed: str = "calm"):
        step, amp = TYPING[speed]
        hand, pole = self._pose("type")
        if self.fingers.get(1) and self.fingers.get(-1):
            # hands with fingers rest on the keys and the fingers type: on each beat one finger of one hand lifts,
            # and comes down on the next; the wrists hardly move (measured as `typing` in checks.json)
            self.typed.append((f0, f1))
            self.hand_shape(f0, f0 + n(3), "type")
            names = {s: [g["name"] for g in self.fingers[s] if g["name"] != "thumb"] for s in (1, -1)}
            for i, f in enumerate(ticks(f0 + n(3), f1 + 1, step)):
                for s in (1, -1):
                    h = self._mirror(hand, s) + Vector((self.rng.uniform(-.006, .006), self.rng.uniform(-.01, .01),
                                                        self._desk_lift(s, "type")))
                    h = self._on_keys(s, f, h, "type") + Vector((0, 0, self.rng.uniform(0, 0.002)))
                    self._arm_key(s, f, h, self._mirror(pole, s), "keys")
                    up = None
                    if (i + (s > 0)) % 2:  # (index and middle fingers do most of the work)
                        up = self.rng.choices(names[s], [1 if g in ("little", "ring") else 3 for g in names[s]])[0]
                    self._fingers_key(s, f, "type", {up: self.TYPE_LIFT} if up else None)
                if i:
                    self.events.append({"frame": f, "kind": "key_click"})
            return
        for i, f in enumerate(ticks(f0, f1 + 1, step)):
            for s in (1, -1):
                up = amp if (i + (s > 0)) % 2 else 0.0
                h = self._mirror(hand, s) + Vector((self.rng.uniform(-.012, .012), self.rng.uniform(-.02, .02),
                                                    up + self._desk_lift(s)))
                self._arm_key(s, f, h, self._mirror(pole, s), "flat")
            if i:
                self.events.append({"frame": f, "kind": "key_click"})

    def tap(self, f: int):
        """Both hands lift and land: 'done!'"""
        hand, pole = self._pose("type")
        for s in (1, -1):
            lift = Vector((0, 0, self._desk_lift(s, self.hand_shape_now.get(s, "flat"))))
            self._arm_key(s, f, self._mirror(hand, s) + lift + Vector((0, 0, 0.06)), self._mirror(pole, s), "flat")
            self._arm_key(s, f + n(6), self._mirror(hand, s) + lift, self._mirror(pole, s), "flat")
        self.events.append({"frame": f + n(6), "kind": "tap"})

    def celebrate(self, f0: int, f1: int):
        self.pose(f0, f0 + n(12), "arms_up")
        hand, pole = self._pose("arms_up")
        for i, f in enumerate(ticks(f0 + n(16), f1, 6)):
            w = 0.06 if i % 2 else -0.06
            for s in (1, -1):
                self._arm_key(s, f, self._mirror(hand, s) + Vector((w, 0, 0.03 if i % 2 else 0)), self._mirror(pole, s))
        self.events.append({"frame": f0 + n(6), "kind": "celebrate"})

    # ------------------------------------------------------------ legs: postures and moving
    def _world(self, local, at=None, yaw=None):
        """A point given from the character's floor spot (x to its left, -y ahead of it), in the world."""
        at, yaw = self.at if at is None else at, self.yaw if yaw is None else yaw
        c, sn = math.cos(yaw), math.sin(yaw)
        return Vector((at[0] + c * local[0] - sn * local[1], at[1] + sn * local[0] + c * local[1], local[2]))

    def _foot_spot(self, s, posture):
        """Where a foot rests, from the floor spot: under the hip when standing, under the knee when sitting
        (on the floor if the shin reaches it, hanging if the chair is too high for these legs)."""
        x, y = self.hip_at.x * s, self.hip_at.y
        if posture == "sit":
            return Vector((x, y - self.thigh, max(self.ankle_h, self.seat_h + self.hip_at.z - self.shin)))
        return Vector((x, y, self.ankle_h))

    def _foot_key(self, s, f, at=None, yaw=None, pitch=0.0):
        """Key a foot (its ankle) at a world point, turned `yaw` and tipped onto its toes by `pitch` degrees.
        With nothing given it holds where it is: two equal keys either side of a stretch keep a foot planted."""
        at = self.foot_at[s] if at is None else Vector(at)
        yaw = self.foot_yaw[s] if yaw is None else yaw
        key(self.foot_t[s], "location", f, at)
        key(self.foot_t[s], "rotation_euler", f, (R(pitch), 0, yaw))
        self.foot_at[s], self.foot_yaw[s] = at, yaw

    def _body_key(self, f, at=None, yaw=None, height=None):
        """Key the floor spot, the facing and the pelvis height."""
        if at is not None:
            self.at = Vector(at[:2])
        if yaw is not None:
            self.yaw = yaw
        if height is not None:
            self.height = height
        key(self.root, "location", f, (self.at.x, self.at.y, 0))
        key(self.root, "rotation_euler", f, (0, 0, self.yaw))
        key(self.hips, "location", f, (0, 0, self.height))
        self.hips.keyframe_insert("rotation_euler", frame=f)

    def _near(self, angle, ref):
        """`angle` (radians), as the turn nearest to `ref`: 350 degrees from 10 is 20 back, not 340 on."""
        return ref + (angle - ref + math.pi) % (2 * math.pi) - math.pi

    def _need(self, posture, verb):
        self._leg_rig()
        if self.posture != posture:
            raise ValueError(f"{self.id} is {'sitting' if self.posture == 'sit' else 'standing'}: {verb} needs them "
                             f"{'sitting' if posture == 'sit' else 'standing'} "
                             f"({'stand_up' if posture == 'stand' else 'sit_down'} first)")

    def place(self, posture: str = "sit", at=(0, 0), facing: float = 0.0, f: int = 1):
        """Where the character is at the start: sitting or standing on a spot, facing `facing` degrees (0 is -Y,
        positive turns to their own left). A standing character's spot is under them; a sitting one's is under
        the seat."""
        self._leg_rig()
        self.posture = posture
        self._body_key(f, at, R(facing), self.seat_h if posture == "sit" else self.stand_h)
        for s in (1, -1):
            self._foot_key(s, f, self._world(self._foot_spot(s, posture)), self.yaw)
        if posture == "stand":
            self.arm_pose = "down"
            for s in (1, -1):
                hand, pole = self._down(s)
                self._arm_key(s, f, hand, self._mirror(pole, s))

    # ------------------------------------------------------------ how a body moves
    # A move looks like a machine's when every part starts and stops on one frame, the left side does what the
    # right does, nothing leads and nothing follows, and it arrives without landing. The posture verbs (sit_down,
    # stand_up, turn, pull_in) are all built from the pieces below, so none of them is tuned on its own:
    #   lead      the eyes, then the head, go where the body is about to go (_look_first)
    #   shift     the weight goes over the foot that stays before the other one lifts (_hips; _carry in a walk)
    #   overlap   the pelvis moves, the chest follows a couple of frames later, the head after that and the
    #             hands last; the head keeps some of its own line of sight while the chest leans (_torso)
    #   settle    a part arrives a little past its mark and comes back (_settle)
    #   sides     one side leads, the other is a few frames late (_sides)
    #   contact   a hand is put on something in the room or on the body and stays on it (_hands_path)
    LAG = {"chest": 2, "head": 4, "hands": 5}  # frames at 24 fps behind the pelvis

    def _sides(self):
        """The side that leads this move (1 is their left) and how many frames the other is behind it."""
        return self.move_rng.choice((1, -1)), n(self.move_rng.choice((2, 3, 3, 4)))

    def _torso(self, f, pitch=0.0, roll=0.0, yaw=0.0, head=(0.0, 0.0), level=0.55):
        """The chest's lean at f (degrees on top of the lean the character holds: forward, to their right, turned
        to their left), keyed a little after f, and the head a little after that. The head gives back `level` of
        the chest's lean, so the face stays up while the body tips; `head` is a look of its own on top (down, left)."""
        key(self.spine, "rotation_euler", f + n(self.LAG["chest"]), (R(self.spine_pitch + pitch), R(roll), R(yaw)))
        key(self.neck, "rotation_euler", f + n(self.LAG["head"]),
            (R(self.neck_pitch - level * pitch + head[0]), R(-level * roll), R(head[1] - level * yaw)))

    def _settle(self, ob, attr, f, rest, over, beat=5.0):
        """Arrive past the mark and come back: at f the channel is `over` beyond `rest`, swings back a third as
        far the other way, and is at rest two and a half beats on."""
        rest, over = Vector(rest), Vector(over)
        key(ob, attr, f, rest + over)
        key(ob, attr, f + n(beat), rest - over * 0.33)
        key(ob, attr, f + n(beat * 2.5), rest)

    def _look_first(self, f, f_back, yaw=0.0, pitch=0.0):
        """The eyes go to where the body is about to go, by f, and come back by f_back (the head's share of the
        look is keyed with the chest, in _torso)."""
        self.glance(f - n(3), f, yaw, pitch)
        self.glance(f_back - n(3), f_back, 0, 0)

    def _arm_mask(self):
        """Which of the body's vertices belong to an arm or a hand (the rest is torso, head and legs)."""
        if getattr(self, "arm_mask", None) is None:
            arm = ("upper", "fore", "hand", "thumb", "index", "middle", "ring", "little")
            groups = {g.index for g in self.body.vertex_groups if g.name.split(".")[0] in arm}
            self.arm_mask = np.zeros(len(self.body.data.vertices), dtype=bool)
            for v in self.body.data.vertices:
                self.arm_mask[v.index] = sum(g.weight for g in v.groups if g.group in groups) > 0.2
        return self.arm_mask

    def _side_clear(self):
        """How far out from the middle (m) a wrist hangs for the hand to clear the hip and thigh: the body's
        half width where the hand hangs, the hand's own thickness and a finger's width. Measured on the mesh
        as it was made, so a broad character's arms hang wider than a slim one's and no wider than they need."""
        if getattr(self, "side_clear", None) is None:
            co = np.empty(len(self.body.data.vertices) * 3, dtype=np.float32)
            self.body.data.vertices.foreach_get("co", co)
            co = (co.reshape(-1, 3) + np.array(self.body.location, dtype=np.float32))[~self._arm_mask()]
            z = co[:, 2] - self.rig.data.bones["spine"].head_local.z
            lo = self.SH[1].z - 0.7122 * self.arm - self.hand_len   # from the knuckles of a hanging hand
            band = co[(z > lo) & (z < lo + self.hand_len * 1.6) & (np.abs(co[:, 1]) < 0.12)]
            half = float(np.percentile(np.abs(band[:, 0]), 99)) if len(band) else self.SH[1].x
            self.side_clear = half + self._girth()["hand"] + 0.012
            if self.fingers.get(1) and self.fingers.get(-1):
                # (hands with fingers hang relaxed, palm to the leg: curled fingers and a thumb reach further
                # toward it than the palm does)
                self.side_clear = half + max(self._hand_under(1, "relaxed")[0], self._hand_under(-1, "relaxed")[0]) + 0.012
        return self.side_clear

    ARM_HANG = 12.0   # degrees a relaxed arm is bent at the elbow, hanging at the side
    HAND_GAP = 0.012  # how far a hanging hand stays off the hip and thigh (m)

    def _hang(self, s, swing=0.0, elbow=None):
        """Where wrist `s` goes (in the chest's space) for an arm that hangs from its shoulder, and the way its
        elbow points (the pose's convention: as for the left arm): the upper arm `swing` degrees forward of
        straight down (back if negative), the elbow bent `elbow` degrees, and the whole arm tipped out at the
        shoulder only as far as the hand needs to clear the hip and thigh by HAND_GAP. The clearance is measured:
        the relaxed hand, palm to the leg, put where the arm leaves it, against the body as it was made. Built
        from the arm's own bones, so the elbow bends by exactly that much and a relaxed arm reads straight (an
        IK target put a set distance under the shoulder bends the elbow by whatever is left over)."""
        elbow = self.ARM_HANG if elbow is None else elbow
        tag = (s, round(swing, 1), round(elbow, 1))
        if tag in self.hang_at:
            return self.hang_at[tag]
        sd = "L" if s > 0 else "R"
        bones = self.rig.data.bones
        upper, fore = bones["upper." + sd].length, bones["fore." + sd].length
        sh = bones["upper." + sd].head_local - Vector((0, 0, self.seat_h))
        a1, a2 = R(swing), R(swing + elbow)

        def arm(tip):  # (elbow, wrist) for the arm tipped out by `tip` about the forward axis through the shoulder
            out = lambda p: Vector((-s * p.z * math.sin(tip), p.y, p.z * math.cos(tip)))  # noqa: E731
            el = out(Vector((0, -upper * math.sin(a1), -upper * math.cos(a1))))
            return sh + el, sh + el + out(Vector((0, -fore * math.sin(a2), -fore * math.cos(a2))))

        if self.fingers.get(s):
            pts = self._hand_points(s, "relaxed")[0][::4]
            co = np.empty(len(self.body.data.vertices) * 3, dtype=np.float32)
            self.body.data.vertices.foreach_get("co", co)
            body = (co.reshape(-1, 3) + np.array(self.body.location - Vector((0, 0, self.seat_h)), dtype=np.float32))
            body = body[~self._arm_mask()]
            body = body[body[:, 0] * s > 0]

            def gap(tip):  # how far the hand is out from the body's side beside it (m; under 0 is inside it)
                el, wr = arm(tip)
                way = (wr - el).normalized()
                palm = Vector((-s, 0, 0))
                palm = (palm - way * palm.dot(way)).normalized()
                frame = np.array([way, palm, way.cross(palm)])
                P = pts @ frame + np.array(wr)
                near = body[(body[:, 2] > P[:, 2].min() - 0.03) & (body[:, 2] < P[:, 2].max() + 0.03)
                            & (body[:, 1] > P[:, 1].min() - 0.03) & (body[:, 1] < P[:, 1].max() + 0.03)]
                if not len(near):
                    return 1.0
                worst = 1.0
                for p in P:
                    by = near[(np.abs(near[:, 1] - p[1]) < 0.025) & (np.abs(near[:, 2] - p[2]) < 0.025)]
                    if len(by):
                        worst = min(worst, float(p[0] * s - (by[:, 0] * s).max()))
                return worst

            lo, hi = R(-8), R(40)  # (a little in, for shoulders wider than the hips)
            if gap(lo) >= self.HAND_GAP:
                hi = lo
            for _ in range(14):
                if hi - lo < R(0.1):
                    break
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if gap(mid) >= self.HAND_GAP else (mid, hi)
            tip = hi
        else:  # (a fitted model without fingers: as far out as the hips allow, as _side_clear measures it)
            drop = -arm(0.0)[1].z + sh.z
            tip = math.asin(max(0.0, min(1.0, (self._side_clear() - abs(sh.x)) / max(drop, 1e-3))))
        el, wr = arm(tip)
        pole = (el - (sh + wr) / 2).normalized()
        self.hang_at[tag] = (wr, self._mirror(pole, s))
        return self.hang_at[tag]

    def _down(self, s):
        """Arm `s` at the side: its wrist (in the chest's space, this side) and its elbow's way (as for the left
        arm). A fitted model's arm hangs from its shoulder (_hang); the clay cast keeps its pose."""
        if self.desc.get("model"):
            return self._hang(s)
        hand, pole = self._pose("down")
        return self._mirror(hand, s), Vector(pole)

    def _girth(self):
        """How thick this character's thigh and hand are (m), measured on the mesh as it was made: what a hand
        resting on a leg has to stand off by."""
        if getattr(self, "girth", None):
            return self.girth
        co = np.empty(len(self.body.data.vertices) * 3, dtype=np.float32)
        self.body.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) + np.array(self.body.location, dtype=np.float32)
        bones, self.girth = self.rig.data.bones, {}
        for name, bone, u, reach in (("thigh", "thigh.L", 0.6, 0.22), ("hand", "hand.L", 0.5, 0.11)):
            a, b = np.array(bones[bone].head_local), np.array(bones[bone].tail_local)
            axis = (b - a) / np.linalg.norm(b - a)
            rel = co - (a + (b - a) * u)
            along = rel @ axis
            radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
            ring = radial[(np.abs(along) < 0.03) & (radial < reach)]
            self.girth[name] = float(np.percentile(ring, 75)) if len(ring) else reach / 2
            if name == "hand" and self.palm.get(1):  # an open hand is flat: its thickness, not half its width
                near = (np.abs(along) < 0.03) & (radial < reach)
                if near.any():
                    self.girth[name] = float(np.percentile(np.abs(rel[near] @ np.array(self.palm[1])), 90))
        return self.girth

    def _lap(self):
        """How far along the thigh (0 hip, 1 knee) a hand can rest clear of the belly: a round one pushes the
        hands out toward the knees. Measured on the mesh as it was made."""
        if getattr(self, "lap_u", None) is None:
            co = np.empty(len(self.body.data.vertices) * 3, dtype=np.float32)
            self.body.data.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3) + np.array(self.body.location, dtype=np.float32)
            z = co[:, 2] - self.rig.data.bones["spine"].head_local.z
            band = co[(np.abs(co[:, 0]) < 0.2) & (z > 0.13) & (z < 0.35)]
            front = -float(band[:, 1].min()) if len(band) else 0.25
            clear = front + self._girth()["hand"] + 0.02 + self.hip_at.y
            self.lap_u = max(0.5, min(0.92, clear / self.thigh))
        return self.lap_u

    def _on_thigh(self, s, u=0.55, round_=0.0):
        """The spot on thigh `s` for its hand to rest on, `u` of the way from hip to knee and `round_` degrees
        round from the top toward the outside (where a belly leaning over the leg is not): a function of the
        frame, read from the leg as it is on that frame, so the hand rides the leg as it bends."""
        # (the leg's surface, and the hand's own thickness off it: the knuckles' middle, not the skin)
        sd, r = "L" if s > 0 else "R", self._girth()["thigh"] + 0.8 * self._girth()["hand"]
        if self.fingers.get(s):  # (a hand with fingers, measured as it lies relaxed: knuckles that far off the leg)
            under, knuckles = self._hand_under(s, "relaxed")
            r = self._girth()["thigh"] + under - knuckles

        def at(f):
            bone, mw = self.rig.pose.bones["thigh." + sd], self.rig.matrix_world
            hip, knee = mw @ bone.head, mw @ bone.tail
            left = self.root.matrix_world.to_3x3() @ Vector((1, 0, 0))
            top = (knee - hip).normalized().cross(left).normalized()  # the leg's front standing, its top sitting
            return hip.lerp(knee, u) + (top * math.cos(R(round_)) + left * s * math.sin(R(round_))) * r

        def lay(f):  # the palm on the leg, the fingers along it toward the knee
            bone, mw = self.rig.pose.bones["thigh." + sd], self.rig.matrix_world
            hip, knee = mw @ bone.head, mw @ bone.tail
            left = self.root.matrix_world.to_3x3() @ Vector((1, 0, 0))
            top = (knee - hip).normalized().cross(left).normalized()
            return -(top * math.cos(R(round_)) + left * s * math.sin(R(round_))), (knee - hip).normalized()
        at.lay = lay
        return at

    def _wrist_for(self, s, point):
        """Where the wrist goes for hand `s` to touch a point in the room: back along the line to the shoulder
        by the hand's own length, so it is the knuckles that arrive and not the wrist (in the chest's space)."""
        inv = self.spine.matrix_world.inverted()
        local = inv @ Vector(point)
        wrist = local + (self.SH[s] - local).normalized() * self.hand_len
        arm = wrist - self.SH[s]  # (never further than the arm is long: a hand that can't reach stops short)
        return self.SH[s] + arm * min(1.0, 0.985 * 0.7122 * self.arm / max(arm.length, 1e-6))

    def _reaches(self, s, point, f, part=0.92):
        """Whether hand `s` can touch a point in the room on frame f (with the elbow still a little bent)."""
        bpy.context.scene.frame_set(f)
        local = self.spine.matrix_world.inverted() @ Vector(point)
        return (local - self.SH[s]).length < part * (0.7122 * self.arm + self.hand_len)

    def _hips(self, f, sway=0.0, roll=0.0, yaw=0.0):
        """The pelvis's own shift at f: `sway` metres to their left (over that foot), tipped and turned."""
        key(self.hips, "location", f, (sway, 0, self.height))
        key(self.hips, "rotation_euler", f, (0, R(roll), R(yaw)))

    def _pole_dir(self, s):
        """The elbow direction hand `s` has now, as a pose gives it (their left arm's, mirrored for the right)."""
        mid = (self.SH[s] + self.hand_t[s].location) / 2
        d = (self.pole_t[s].location - mid).normalized()
        return (d.x * s, d.y, d.z)

    def _hands_path(self, stages: dict):
        """Move hands through stages, keyed on every frame so a hand stays on what it touches while the body
        moves under it. `stages` is {side: [(frame, target, pole, arc), ...]}: a target is a wrist position in the
        chest's space (a pose) or a function of the frame giving a point in the room; `pole` is the elbow's
        direction; `arc` lifts the hand's path into a stage (m). Between stages the hand eases from one to the next."""
        sc = bpy.context.scene
        self.swung = {}
        for f in range(min(st[0][0] for st in stages.values()), max(st[-1][0] for st in stages.values()) + 1):
            sc.frame_set(f)
            for s, st in stages.items():
                spot = lambda t: self._wrist_for(s, t(f)) if callable(t) else Vector(t)  # noqa: E731
                if not st[0][0] <= f <= st[-1][0]:
                    continue
                i = max(k for k in range(len(st)) if st[k][0] <= f)
                (fa, ta, pa, _), (fb, tb, pb, arc) = st[i], st[min(i + 1, len(st) - 1)]
                u = 0.0 if fb <= fa else (f - fa) / (fb - fa)
                u = u * u * (3 - 2 * u)
                at = spot(ta).lerp(spot(tb), u) + Vector((0, -0.4, 0.9)) * arc * math.sin(math.pi * u)
                pole = self._mirror(Vector(pa).lerp(Vector(pb), u), s)
                # how the hand lies on what it is leaving and on what it is going to
                was, lay = (getattr(t, "lay", None) if callable(t) else self._lay() if t is st[0][1] else None
                            for t in (ta, tb))
                self._arm_key(s, f, at, pole, lay, was, u)
                on = ta if callable(ta) and (u == 0 or callable(tb)) else tb if callable(tb) and u >= 1 else None
                if on:
                    # on the thing: the hand points where the forearm does, so see where the knuckles came out
                    # and move the wrist by what they missed (a few times: moving it turns the forearm)
                    inv = self.spine.matrix_world.inverted().to_3x3()
                    for _ in range(4):
                        bpy.context.view_layer.update()
                        miss = inv @ (on(f) - self._hold_point(s))
                        if miss.length < 0.004:
                            break
                        arm = at + miss - self.SH[s]
                        at = self.SH[s] + arm * min(1.0, 0.985 * 0.7122 * self.arm / max(arm.length, 1e-6))
                        self._arm_key(s, f, at, pole, lay, was, u)

    def _touched(self, s, f, point, what, held=False):
        """Note how far hand `s` is from what it is meant to be touching on frame f (checked by contacts());
        `held` also notes how open its fingers are round it, for a hand that has fingers."""
        bpy.context.scene.frame_set(f)
        bpy.context.view_layer.update()
        self.holding[s] = held
        tip = self._hold_point(s)
        self.holding[s] = False
        self.touches.append({"frame": f, "m": (tip - Vector(point(f))).length, "hand": "left" if s > 0 else "right",
                             "what": what})
        if held and self.fingers.get(s):
            self.grips.append({"frame": f, "open": round(self.grip_open(s, point(f)), 2),
                               "hand": "left" if s > 0 else "right", "what": what})

    def push(self, f0: int, f1: int, where, hand: str | None = None, what: str = "handle"):
        """A hand takes hold of something that is about to move (a door's handle) and goes with it for as long
        as the arm reaches, then comes away to the side. `where(f)` is the thing's place on frame f; it starts
        to move at f0, so the hand sets out before that. They look at it first and lean in a little as they
        push. Raises if the hand can't reach it from where they stand."""
        sc = bpy.context.scene
        sc.frame_set(f0)
        local = self.root.matrix_world.inverted() @ Vector(where(f0))
        s = (1 if local.x > 0 else -1) if hand is None else (1 if hand == "left" else -1)
        if not self._reaches(s, where(f0), f0):
            gap = (self.spine.matrix_world.inverted() @ Vector(where(f0)) - self.SH[s]).length \
                - 0.92 * (0.7122 * self.arm + self.hand_len)
            raise ValueError(f"{self.id}'s hand is {gap:.2f} m short of the {what}: stand them closer to it")
        out = max(1, f0 - n(9))
        let_go = max((f for f in range(f0, f1 + 1) if self._reaches(s, where(f), f)), default=f0)
        let_go = min(let_go, f0 + max(2, (f1 - f0) * 2 // 3))
        start, pole = self._hands_now(out)[s]
        side = math.degrees(math.atan2(local.x, -local.y))
        self._look_first(out + n(2), let_go, yaw=max(-20, min(20, side)), pitch=16)
        self._torso(out)
        self._torso(f0 - n(2), pitch=3, yaw=-4 * s, head=(8, 0.4 * side))
        self._torso((f0 + let_go) // 2, pitch=7, yaw=-7 * s, head=(4, 0.2 * side))
        self._torso(let_go + n(6), pitch=-1)
        self._torso(let_go + n(12))
        hang, elbow = self._down(s) if self.posture == "stand" else (start, Vector(self.HANG))
        self.holding[s] = True  # (a hand with fingers takes the thing in the crook of them)
        self._hands_path({s: [(out, start, pole, 0), (f0, where, (1, 0.5, -0.6), 0.05), (let_go, where, (1, 0.5, -0.6), 0),
                              (let_go + n(8), hang + Vector((0, -0.05, 0.02)), elbow, 0.04),
                              (let_go + n(14), hang, elbow, 0)]})
        self.holding[s] = False
        # (not opened flat on its way: straight fingers arrive at the door before the hand does, and go through it)
        self.hand_shape(out, f0 - n(4), "relaxed", "left" if s > 0 else "right")
        self.hand_shape(f0 - n(3), f0, "grip", "left" if s > 0 else "right")   # closes on the thing
        self.hand_shape(let_go, let_go + n(6), "relaxed", "left" if s > 0 else "right")  # and lets go
        self._touched(s, f0, where, what, held=True)
        self._touched(s, let_go, where, what, held=True)
        self.events.append({"frame": f0, "kind": "touch"})

    def _hands_now(self, f):
        """Each hand's wrist target and elbow direction as they are on frame f: where a path starts from."""
        bpy.context.scene.frame_set(f)
        return {s: (self.hand_t[s].location.copy(), self._pole_dir(s)) for s in (1, -1)}

    LAP = (1, 0.55, -0.35)   # elbows out and a little back: hands on the thighs
    HANG = (0.4, 1, 0.1)     # elbows back: arms at the sides

    def stand_up(self, f0: int, f1: int):
        """From the chair to their feet. The head comes up first, the feet square up one after the other, the
        hands go to the knees and the chest comes forward over the feet before the pelvis leaves the seat;
        then up, a little past standing and back, the chest straightening after the legs and the arms
        dropping to the sides last."""
        self._need("sit", "stand_up")
        self.moves.append(("stand_up", f0, f1))
        T = f1 - f0
        lead, late = self._sides()
        ready = f0 + round(T * 0.26)   # feet under them, chest forward, hands on the knees
        up = f0 + round(T * 0.76)      # on their feet
        half = ready + round((up - ready) * 0.5)
        seat, feet = self.at.copy(), self._world((0, -self.thigh, 0)).xy
        rise = self.stand_h - self.seat_h
        hands, lap = self._hands_now(f0), self._lap()

        # feet: each one comes back flat under its knee, the leading one first
        for s in (1, -1):
            d = 0 if s == lead else late
            spot = self._foot_spot(s, "sit")
            to = self._world((spot.x, spot.y, self.ankle_h))
            self._foot_key(s, f0 + d)
            if (to - self.foot_at[s]).length > 0.01 or abs(self.foot_yaw[s] - self.yaw) > 0.02:
                self._foot_key(s, f0 + d + n(3), self.foot_at[s].lerp(to, 0.5) + Vector((0, 0, 0.025)), pitch=6)
                self._foot_key(s, f0 + d + n(6), to, self.yaw)
            self._foot_key(s, f1)

        # pelvis: forward on the seat, then forward and up (forward first), past standing and back
        self._body_key(f0)
        self._body_key(ready, seat.lerp(feet, 0.14), height=self.seat_h)
        self._body_key(half, seat.lerp(feet, 0.62), height=self.seat_h + rise * 0.45)
        self._body_key(up, feet, height=self.stand_h + 0.006)
        self._body_key(up + n(5), height=self.stand_h - 0.004)
        self._body_key(f1, height=self.stand_h)
        self._hips(up, 0)
        self._hips(up + n(4), sway=-0.012 * lead, roll=1.5 * lead)  # the weight finds a foot, then both
        self._hips(f1)

        # chest and head: the head lifts first, the chest comes over the knees, and straightens after the legs
        self._look_first(f0 + n(3), ready, pitch=-8)
        self._torso(f0)
        self._torso(f0 + round(T * 0.1), pitch=4, head=(-7, 0))
        self._torso(ready, pitch=20, head=(-4, 0))
        self._torso(ready + round((up - ready) * 0.3), pitch=27)
        self._torso(half + n(2), pitch=17)
        self._torso(up, pitch=3)
        self._torso(up + round((f1 - up) * 0.5), pitch=-3, head=(2, 0))
        self._torso(f1 - n(self.LAG["head"]) - 1)

        # hands: push on the knees, ride them up, then fall to the sides, one after the other, and swing once
        stages = {}
        for s in (1, -1):
            d = 0 if s == lead else late
            start, pole = hands[s]
            hang, elbow = self._down(s)
            stages[s] = [(f0 + d, start, pole, 0),
                         (ready - n(2) + d, self._on_thigh(s, min(0.97, lap + 0.12), 55), self.LAP, 0.06),
                         (half + d, self._on_thigh(s, min(0.97, lap + 0.05), 60), self.LAP, 0),
                         (up + d, hang + Vector((0, 0.05, 0.01)), elbow, 0.03),   # (swung a little past)
                         (up + d + n(5), hang + Vector((0, -0.02, 0)), elbow, 0),
                         (f1, hang, elbow, 0)]
        self._hands_path(stages)
        self.arm_pose = "down"
        self.posture = "stand"
        self.events.append({"frame": ready + n(3), "kind": "rustle"})

    def sit_down(self, f0: int, f1: int):
        """Onto the chair behind them. A look down at it, the pelvis goes back before it goes down with the
        chest tipping forward to balance, the hands come to the thighs, the seat takes the weight with a small
        dip, and the chest, the head and then the hands settle, one foot shifting last."""
        self._need("stand", "sit_down")
        self.moves.append(("sit_down", f0, f1))
        T = f1 - f0
        lead, late = self._sides()
        go = f0 + round(T * 0.14)       # the look is done, the pelvis starts back
        land = f0 + round(T * 0.68)     # the seat takes the weight
        half = go + round((land - go) * 0.5)
        feet, seat = self.at.copy(), self._world((0, self.thigh, 0)).xy
        rise = self.stand_h - self.seat_h
        hands, lap = self._hands_now(f0), self._lap()

        # pelvis: knees unlock, back first, then down into the seat and a small rebound
        self._body_key(f0)
        self._body_key(go, feet.lerp(seat, 0.05), height=self.stand_h - rise * 0.04)
        self._body_key(half, feet.lerp(seat, 0.72), height=self.seat_h + rise * 0.6)
        self._body_key(land, seat, height=self.seat_h - 0.014)
        self._body_key(land + n(4), height=self.seat_h + 0.004)
        self._body_key(land + n(9), height=self.seat_h)

        # chest and head: a look down to the leading side, forward over the knees on the way down, back past
        # upright once the seat has them, then upright; the head arrives after the chest
        look = 26 * lead
        self._look_first(f0 + n(2), half, yaw=20 * lead, pitch=18)
        self._torso(f0)
        self._torso(go, pitch=7, yaw=0.3 * look, head=(16, look))
        self._torso(half, pitch=28, yaw=0.1 * look, head=(6, 0.3 * look))
        self._torso((half + land) // 2, pitch=19)     # (the lean comes off as the thighs come level: a body
        self._torso(land, pitch=8, roll=-1.5 * lead)  # does not fold in half over its own legs)
        self._torso(land + round((f1 - land) * 0.6), pitch=-4, head=(3, 0))
        self._torso(f1 - n(self.LAG["head"]) - 1)

        # feet stay where they are; once sitting, the other foot shifts a little to get comfortable
        for s in (1, -1):
            self._foot_key(s, f0)
            self._foot_key(s, land, self._world(self._foot_spot(s, "sit")))
        s = -lead
        spot = self._foot_spot(s, "sit")
        to = self._world(spot + Vector((0.025 * s, -0.06, 0)))
        self._foot_key(s, land + n(3))
        self._foot_key(s, land + n(6), self.foot_at[s].lerp(to, 0.5) + Vector((0, 0, 0.03)), self.yaw + R(5 * s), pitch=8)
        self._foot_key(s, land + n(9), to, self.yaw + R(9 * s))
        self._foot_key(lead, f1)

        # hands: to the thighs on the way down (the leading hand first), on them as the weight lands, then
        # they slide back to rest, last of all
        stages = {}
        for s in (1, -1):
            d = 0 if s == lead else late
            start, pole = hands[s]
            # (down the outside of the thigh to the knee: straight there would go through the leg as it comes up)
            stages[s] = [(f0 + d, start, pole, 0), (go + d, start, pole, 0),
                         ((go + half) // 2 + d, self._on_thigh(s, 0.45, 95), self.HANG, 0.02),
                         (half + n(3) + d, self._on_thigh(s, min(0.97, lap + 0.1), 70), self.LAP, 0.03),
                         (land + d, self._on_thigh(s, min(0.97, lap + 0.12), 55), self.LAP, 0),
                         (f1 - (late - d), self._on_thigh(s, lap + (0 if s == lead else 0.05), 20), self.LAP, 0.012)]
            stages[s].append((f1, stages[s][-1][1], self.LAP, 0))
        self._hands_path(stages)
        self.arm_pose = "lap"
        self.posture = "sit"
        self.events.append({"frame": land, "kind": "rustle"})

    def scoot(self, f0: int, f1: int, to, rides=(), grip=None):
        """Sitting, shift chair and all to a spot (pulling in to the desk). They look where they are going and
        lean into it, each foot reaches ahead, plants and pulls, so the body comes in surges and not at one
        speed; the chest rocks back with each pull and forward as the chair stops, then settles. `rides` are
        objects that go along (the chair). `grip` gives a point in the room for each hand to take hold of once
        it is in reach (the desk's edge): the last of the way they pull themselves in by it."""
        self._need("sit", "pull_in")
        self.moves.append(("pull_in", f0, f1))
        T = f1 - f0
        lead, late = self._sides()
        start, by = self.at.copy(), Vector(to[:2]) - self.at
        # each foot can only be a little ahead of its knee or behind it, and stays down while the other one
        # steps: so the feet walk in short steps, and the chair rolls on between them
        steps = max(2, math.ceil(by.length / (0.2 * (self.thigh + self.shin))))
        go, stop = f0 + round(T * 0.16), f0 + round(T * 0.8)
        share = [self.move_rng.uniform(0.8, 1.2) for _ in range(steps)]
        share[0] *= 1.3   # setting off and
        share[-1] *= 1.3  # stopping take longer
        edge = [go + round((stop - go) * sum(share[:k]) / sum(share)) for k in range(steps + 1)]
        done = [k / steps for k in range(steps + 1)]
        hands, lap = self._hands_now(f0), self._lap()
        rest = {ob: ob.location.copy() for ob in rides}

        def body(f, part, height=None):
            self._body_key(f, start + by * part, height=height)
            for ob in rides:
                key(ob, "location", f, rest[ob] + Vector((by.x * part, by.y * part, 0)))

        body(f0, 0)
        body(go, 0)
        self._look_first(f0 + n(2), stop, pitch=14)
        self._torso(f0)
        self._torso(go, pitch=9, head=(7, 0))
        self._hips(go)
        for s in (1, -1):
            self._foot_key(s, go)
        for k in range(1, steps + 1):
            s = lead if k % 2 else -lead
            a, b = edge[k - 1], edge[k]
            to_spot = self._world(self._foot_spot(s, "sit"), at=start + by * done[min(k + 1, steps)])
            self._foot_key(s, a)
            self._foot_key(-s, a)
            self._foot_key(s, (a + b) // 2, self.foot_at[s].lerp(to_spot, 0.5) + Vector((0, 0, 0.025)), pitch=-7)
            self._foot_key(s, b, to_spot)   # heel first, a little ahead of where the knee will be
            self._foot_key(-s, b)
            if k < steps:
                body(b, done[k])
                self._hips(b, sway=0.008 * s, roll=-1.2 * s)
        body(stop + n(3), 1)                # the chair rolls the last of the way after the feet have stopped
        self._hips(stop + n(3))
        self._torso((go + stop) // 2, pitch=6, head=(5, 0))
        self._torso(stop, pitch=9, head=(3, 0))  # as the chair stops the chest carries on
        for s in (1, -1):  # both feet come to rest under the knees, the one left behind last
            end = self._world(self._foot_spot(s, "sit"))
            if (end - self.foot_at[s]).length > 0.01:
                self._foot_key(s, stop + n(2))
                self._foot_key(s, stop + n(5), self.foot_at[s].lerp(end, 0.5) + Vector((0, 0, 0.02)), pitch=5)
                self._foot_key(s, stop + n(8), end)
        self._torso(stop + round((f1 - stop) * 0.55), pitch=-2, head=(-2, 0))
        self._torso(f1 - n(self.LAG["head"]) - 1)
        self._hips(f1)

        # hands: on the knees while the feet do the work; once the desk is in reach they take its edge and
        # pull, and stay on it
        stages, grips = {}, []
        for s in (1, -1):
            d = 0 if s == lead else late
            begin, pole = hands[s]
            on_knee = self._on_thigh(s, min(0.97, lap + 0.08), 40)
            stages[s] = [(f0 + d, begin, pole, 0), (go + d, on_knee, self.LAP, 0.03)]
            took = None
            if grip:
                # (the knuckles' own height over the desk top with the hand laid flat on it)
                over = 0.6 * self._girth()["hand"]
                if self.fingers.get(s):
                    under, knuckles = self._hand_under(s)
                    over = under - knuckles + 0.004
                point = Vector(grip(s)) + Vector((0, 0, over))
                if self.fingers.get(s):  # (wide of a laptop: a hand laid flat reaches a hand's length onto the desk)
                    point.x = s * max(abs(point.x), 0.42)
                took = next((f for f in edge[2:-1:2] + [stop - n(2)] if self._reaches(s, point, f + d)), None)
            if took is None:
                stages[s] += [(f1 - (late - d), self._on_thigh(s, lap, 20), self.LAP, 0.01), (f1, self._on_thigh(s, lap, 20), self.LAP, 0)]
            else:
                edge_at = lambda f, p=point: p  # noqa: E731
                edge_at.lay = "flat"  # (laid on the desk top, not hooked on it side on)
                # (up past the desk's edge before forward onto it: fingers lead the hand, and would go through it)
                over_edge = lambda f, p=point + Vector((0, 0.13, 0.07)): p  # noqa: E731
                over_edge.lay = "flat"
                rise = [(took - n(7) + d, on_knee, self.LAP, 0), (took - n(2) + d, over_edge, (1, 0.3, -0.8), 0.03),
                        (took + d, edge_at, (1, 0.3, -0.8), 0)]
                if not self.fingers.get(s):  # (a mitten on one bone goes straight to the edge)
                    rise = [(took - n(5) + d, on_knee, self.LAP, 0), (took + d, edge_at, (1, 0.3, -0.8), 0.07)]
                stages[s] += rise + [
                              (f1, edge_at, (1, 0.3, -0.8), 0)]
                self.events.append({"frame": took + d, "kind": "touch"})
                grips.append((s, took + d, point))
        self._hands_path(stages)
        for s, f, point in grips:
            side = "left" if s > 0 else "right"
            self.hand_shape(f - n(8), f - n(3), "open", side)  # open over the desk's edge, then laid flat on it,
            self.hand_shape(f - n(2), f + n(2), "flat", side)  # the thumb under
            self._touched(s, f + n(2), lambda f, p=point: p, "desk")
            self._touched(s, f1, lambda f, p=point: p, "desk")
        self.arm_pose = "lap"

    def door_fit(self) -> dict:
        """The door this character needs: clear of their head, the handle where their hand falls."""
        self._leg_rig()
        top = self.stand_h + (getattr(self, "head_at", (0, 0, 1.72))[2] - self.seat_h) + self.head_r * 1.1
        wide = 2 * (self._side_clear() + self._girth()["hand"]) + 0.4  # hands at the sides, elbows, and room to spare
        fit = {"height": round(top * 1.12, 3), "width": round(max(0.9, top * 0.45, wide), 3),
               "handle": round(self.stand_h + self.SH[1].z - 0.62 * 0.7122 * self.arm, 3)}
        if self.fingers.get(1):  # fingers close behind the bar: it stands off the leaf by their thickness and more
            fit["handle_off"] = round(max(0.07, 2.2 * self._girth()["hand"]), 3)
        return fit

    def _step(self, s, f0, f1, to, yaw, roll=True):
        """One step: foot `s` peels off the floor heel first, swings through and lands on its heel at `to` by
        f1, then settles flat; the other stays planted. `roll` off is a flat-footed shuffle (turning on the spot)."""
        start, start_yaw = self.foot_at[s].copy(), self.foot_yaw[s]
        span = f1 - f0
        self._foot_key(s, f0)
        self._foot_key(-s, f0)
        if roll and span >= 6:  # heel up on the toes before the foot leaves the floor
            # (the foot turns about its ankle, so the ankle rises by what the toes would have sunk)
            self._foot_key(s, f0 + span // 6, start + Vector((0, 0, self.foot_len * math.sin(R(22)))), start_yaw, pitch=22)
        mid = start.lerp(Vector(to), 0.55) + Vector((0, 0, 0.11 * (self.thigh + self.shin) + self.foot_len * 0.15))
        self._foot_key(s, f0 + span * 3 // 5, mid, (start_yaw + yaw) / 2, pitch=6 if roll else 10)
        self._foot_key(s, f1, Vector(to) + Vector((0, 0, self.foot_len * 0.13 if roll else 0)), yaw, pitch=-12 if roll else 0)
        if roll:  # the toes come down after the heel
            self._foot_key(s, f1 + max(1, span // 5), to, yaw, pitch=0)
        self._foot_key(-s, f1)
        self.events.append({"frame": f1, "kind": "step"})

    def _carry(self, f, s=0, lean=0.0, amount=1.0, bank=None, look=0.0):
        """How the body rides a step, keyed at f: the pelvis shifts over the planted foot, drops a little on
        the swinging side and turns with the leading leg, and the chest turns the other way. `s` is the foot
        in the air or just landed (0 squares everything up again). On a curved path `bank` leans the whole
        body into the bend (degrees to their left, shared by the pelvis and the chest) and `look` turns the
        head to where the path goes; the head stays nearer level than the chest."""
        sway = -s * abs(self.hip_at.x) * 0.22 * amount
        lean_in = bank or 0.0
        key(self.hips, "rotation_euler", f, (0, R(-3.0 * s * amount + 0.4 * lean_in), R(-6.0 * s * amount)))
        key(self.hips, "location", f, (sway, 0, self.hips.location.z))
        key(self.spine, "rotation_euler", f, (R(self.spine_pitch + lean), R(2.0 * s * amount + 0.6 * lean_in),
                                              R(10.0 * s * amount)))
        if bank is not None:
            key(self.neck, "rotation_euler", f, (R(self.neck_pitch), R(-0.5 * lean_in), R(look)))

    def _swing(self, f, s=0, reach=0.0, out=0.0, late=True):
        """Arms at the sides, the one opposite foot `s` forward by `reach` (m), the other back by less; the
        elbow bends as the hand comes forward. Keyed a few frames after f: the arms follow the legs. Only when
        the arms are down."""
        if self.arm_pose != "down":
            return
        hand, pole = self._pose("down")
        f += n(self.LAG["hands"] - 2) if s and late else 0
        if self.desc.get("model"):
            # a fitted model's arm swings from the shoulder like a pendulum, nearly straight behind and bending
            # a little more in front; the forearm never comes up toward level (measured as the walk's `arms`)
            bones = self.rig.data.bones
            full = bones["upper.L"].length + bones["fore.L"].length
            for side in (1, -1):
                fwd = reach * (1.0 if side == -s else -0.7) if s else 0.0
                k = fwd / (0.5 * full)  # about 1 at the front of a full stride
                swing = max(-16.0, min(20.0, 18.0 * k))
                elbow = self.ARM_HANG + 12.0 * max(0.0, min(1.2, k)) - 4.0 * max(0.0, min(1.0, -k / 0.7))
                side_out = out * (side == s)
                was = self.swung.get(side)
                if was and 2 < f - was[0] <= n(24):
                    # (keys on the arc between: a target eased in a straight line from front to back cuts
                    # inside the arc, nearer the shoulder, and the elbow folds to make up the difference)
                    for part in (1 / 3, 2 / 3):
                        mid = lambda a, b: a + (b - a) * part  # noqa: E731
                        wrist, way = self._hang(side, mid(was[1], swing), mid(was[2], elbow))
                        self._arm_key(side, round(mid(was[0], f)), wrist + Vector((side * mid(was[3], side_out), 0, 0)),
                                      self._mirror(way, side), lay="side")
                wrist, way = self._hang(side, swing, elbow)
                self._arm_key(side, f, wrist + Vector((side * side_out, 0, 0)), self._mirror(way, side), lay="side")
                self.swung[side] = (f, swing, elbow, side_out)
            return
        for side in (1, -1):
            fwd = reach * (1.0 if side == -s else -0.7) if s else 0.0
            lift = fwd * 0.5 if fwd > 0 else -fwd * 0.15  # the hand rises more in front than behind
            # (`out`: the hand on the stepping side stands off, for a leg that steps out sideways in a turn)
            self._arm_key(side, f, self._mirror(hand, side) + Vector((-side * max(0.0, fwd) * 0.15 + side * out * (side == s),
                                                                      -fwd, lift)),
                          self._mirror(pole, side), lay="side")

    def turn(self, f0: int, f1: int, facing: float):
        """Turn on the spot to face `facing` degrees, stepping round. The eyes and the head go first, the
        chest starts round after them, the weight shifts off the foot that is about to step, the steps are not
        equal, and at the end the chest swings a little past and comes back."""
        self._need("stand", "turn")
        self.moves.append(("turn", f0, f1))
        yaw0, yaw1 = self.yaw, self._near(R(facing), self.yaw)
        by = math.degrees(yaw1 - yaw0)
        steps = 2 * max(1, math.ceil(abs(by) / 100))
        first = 1 if yaw1 > yaw0 else -1  # the foot on the side they turn to leads
        T = f1 - f0
        go = f0 + max(1, round(T * 0.16))
        stop = f1 - max(1, round(T * 0.14))
        share = [1.2 if k % 2 == 0 else 0.9 for k in range(steps)]  # the leading foot takes longer than the one that follows
        edge = [go + round((stop - go) * sum(share[:k]) / sum(share)) for k in range(steps + 1)]
        ahead = max(-38.0, min(38.0, by * 0.45))  # how far the head runs ahead of the body
        self._body_key(f0)
        self._look_first(f0 + n(2), stop, yaw=max(-20, min(20, by * 0.3)))
        self._torso(f0)
        self._torso(go, yaw=0.3 * ahead, head=(0, ahead))
        self._hips(f0)
        self._hips(go, sway=-0.2 * abs(self.hip_at.x) * first, roll=1.5 * first)  # off the foot that steps first
        self._body_key(go)
        self._swing(f0)
        for k in range(1, steps + 1):
            s = first if k % 2 else -first
            a, b = edge[k - 1], edge[k]
            yaw = yaw0 + (yaw1 - yaw0) * math.ceil(k / 2) / (steps // 2)
            self._step(s, a, b, self._world(self._foot_spot(s, "stand"), yaw=yaw), yaw, roll=False)
            self._body_key(b, yaw=yaw0 + (yaw1 - yaw0) * k / steps)
            left = 1 - k / steps
            self._hips(b, sway=0.2 * abs(self.hip_at.x) * s * (1 if k < steps else 0), roll=-1.5 * s * (1 if k < steps else 0))
            self._torso(b, yaw=0.3 * ahead * left - (0.12 * by / steps if k == steps else 0), head=(0, ahead * left))
            slow = b - a >= n(7)  # (a quick pivot inside a walk has no room for the arms to trail)
            if slow:
                self._swing((a + b) // 2, s, 0.03, out=0.07, late=False)
            self._swing(b, s if k < steps else 0, 0.07, out=0.03 if slow else 0.06, late=False)
        self._torso(stop + max(1, (f1 - stop) // 2), yaw=0.04 * by / steps)  # the chest swings past, and back
        self._torso(f1 - n(self.LAG["head"]) - 1)
        self._hips(f1)
        self._swing(f1)
        return f1

    def _curve(self, p0, p1, yaw0, heading):
        """A curved way from p0 to p1: it leaves the way they face (`yaw0`) and arrives travelling `heading`
        (radians; None is straight at the spot from wherever the bend has brought them). Returns its length and
        three functions of the distance along it: the point, the direction of travel (unwrapped from yaw0) and
        the bend (radians of turn per metre, left positive)."""
        chord = (p1 - p0).length
        ahead = lambda a: Vector((math.sin(a), -math.cos(a)))  # noqa: E731
        m0 = ahead(yaw0) * chord
        m1 = (ahead(heading) if heading is not None else (p1 - p0).normalized() * 0.6 + ahead(yaw0) * -0.1) * chord
        us = np.linspace(0.0, 1.0, 161)
        h = [2 * us ** 3 - 3 * us ** 2 + 1, us ** 3 - 2 * us ** 2 + us, -2 * us ** 3 + 3 * us ** 2, us ** 3 - us ** 2]
        pts = np.outer(h[0], p0) + np.outer(h[1], m0) + np.outer(h[2], p1) + np.outer(h[3], m1)
        seg = np.diff(pts, axis=0)
        along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])
        yaw = np.unwrap(np.arctan2(seg[:, 0], -seg[:, 1]))
        yaw += round((yaw0 - yaw[0]) / (2 * math.pi)) * 2 * math.pi
        mid = (along[:-1] + along[1:]) / 2
        bend = np.gradient(yaw, mid)
        soft = np.convolve(np.pad(bend, 6, mode="edge"), np.ones(13) / 13, mode="valid")  # (no jerk in the lean)
        return (float(along[-1]),
                lambda d: Vector((float(np.interp(d, along, pts[:, 0])), float(np.interp(d, along, pts[:, 1])))),
                lambda d: float(np.interp(d, mid, yaw)), lambda d: float(np.interp(d, mid, soft)))

    def walk(self, f0: int, f1: int | None, to, facing: float | None = None, curve: bool = False,
             heading: float | None = None):
        """Walk to a point on the floor and stop there, facing `facing` degrees (or the way they came). A small
        change of direction happens on the first step; for a bigger one they turn on the spot first. They turn
        to `facing` on the last step. With no f1 they take the time the steps need at a stroll; returns the
        frame they arrive on.

        With `curve` the way there is a bend and not a line: they set off the way they are facing and come round
        to the spot, arriving in the direction `heading` (degrees) if one is given. The feet land along the
        bend, the body turns with it and leans into it, as a body does to stay over its feet (more when the
        bend is tighter or the walk faster), and the head looks along the path ahead."""
        self._need("stand", "walk")
        p0, p1 = self.at.copy(), Vector(to[:2])
        dist = (p1 - p0).length
        if dist < 0.05:
            return self.turn(f0, f1 or f0 + n(18), math.degrees(self.yaw) if facing is None else facing)
        way = (p1 - p0) / dist
        yaw_w = self._near(math.atan2(way.x, -way.y), self.yaw)
        f_start = f0
        # a leg can't swing round that far in one step (on a curve the path itself turns them, up to a point)
        if abs(yaw_w - self.yaw) > (R(110) if curve else R(30)):
            pivots = 2 * max(1, math.ceil(abs(yaw_w - self.yaw) / R(100)))
            turned = f0 + pivots * n(8) if f1 is None else f0 + round((f1 - f0) * 0.3)
            self.turn(f0, turned, math.degrees(yaw_w))
            f0 = turned
        yaw0 = self.yaw
        point, travel, bend = (lambda d: p0 + way * d), (lambda d: yaw_w), (lambda d: 0.0)
        if curve:
            dist, point, travel, bend = self._curve(p0, p1, yaw0, None if heading is None else
                                                    self._near(R(heading), yaw0))
            yaw_w = travel(dist)  # the way they are going as they arrive
        yaw1 = yaw_w if facing is None else self._near(R(facing), yaw_w)
        leg = self.thigh + self.shin
        after = None  # a big turn at the end is made on the spot too, once they have arrived
        if abs(yaw1 - yaw_w) > R(30):
            pivots = 2 * max(1, math.ceil(abs(yaw1 - yaw_w) / R(100)))
            after = (pivots * n(8) if f1 is None else round((f1 - f0) * 0.3), math.degrees(yaw1))
            f1 = None if f1 is None else f1 - after[0]
            yaw1 = yaw_w
        steps = max(2, math.ceil(dist / (0.5 * leg)) + 1)  # the last step brings the feet together
        half = dist / (steps - 1)                          # how far each landing is past the one before
        f1 = f0 + steps * n(8) if f1 is None else f1
        # no two steps quite alike: each takes a little more or less time and lands a little long or short
        beat = [1 + self.rng.uniform(-0.09, 0.09) for _ in range(steps)]
        beat[0] *= 1.15   # setting off and
        beat[-1] *= 1.2   # pulling up are slower
        frames = [f0 + round((f1 - f0) * sum(beat[:k]) / sum(beat)) for k in range(steps + 1)]
        off = [0.0] + [self.rng.uniform(-0.07, 0.07) * half for _ in range(max(0, steps - 2))] + [0.0, 0.0]
        land = lambda k: min(k, steps - 1) * half + off[min(k, steps)]  # noqa: E731
        first = 1 if (bend(0.2 * dist) if curve else yaw_w - yaw0) > 1e-3 else -1
        speed = dist / max(1e-3, (f1 - f0) / kit.FPS)

        def lean_in(d):
            """Degrees to lean into the bend at d along the path: what keeps a body over its feet at this speed
            (speed squared over the bend's radius, against gravity), a little more than life, and none on a line."""
            if not curve:
                return None, 0.0
            bank = max(-9.0, min(9.0, 1.4 * math.degrees(math.atan(speed * speed * bend(d) / 9.81))))
            look = max(-28.0, min(28.0, 0.6 * math.degrees(travel(min(dist, d + 0.9)) - travel(d))))
            return bank, look

        self._body_key(f0)
        self._swing(f0)
        self._carry(f0, bank=0.0 if curve else None)
        for k in range(1, steps + 1):
            s = first if k % 2 else -first
            a, b = frames[k - 1], frames[k]
            last = k == steps
            yaw = yaw1 if last else travel(land(k))
            spot = self._foot_spot(s, "stand")
            spot.x *= 0.8  # feet fall nearer the line of travel than they stand
            self._step(s, a, b, self._world(spot if not last else self._foot_spot(s, "stand"),
                                            at=point(land(k)), yaw=yaw), yaw)
            apart = land(k) - land(k - 1)
            low = self.ankle_h - self.hip_at.z + math.sqrt((0.985 * leg) ** 2 - (apart / 2) ** 2) - 0.012
            mid = (a + b) // 2
            under = (land(k) + land(k - 1)) / 2  # where the body is as the foot lands: between its feet
            key(self.hips, "location", mid, (0, 0, self.stand_h - 0.006))  # tallest as one leg passes the other
            bank, look = lean_in((under + (land(k - 1) + land(max(0, k - 2))) / 2) / 2)
            self._carry(mid, s, lean=3.5, amount=0.6, bank=bank, look=look)
            self._body_key(b, point(under), yaw1 if last else travel(under), low)
            bank, look = (0.0 if curve else None, 0.0) if last else lean_in(under)
            self._carry(b, 0 if last else s, lean=0.0 if last else 3.5, bank=bank, look=look)
            self._swing(b, s if not last else 0, apart * 0.85)
        self.walks.append((f_start, f1, bool(curve)))
        if after:
            f1 = self.turn(f1, f1 + after[0], after[1])
        return f1

    def walk_report(self, f0: int, f1: int, curve: bool) -> dict:
        """What a walk did, read back from the animation: how far its path bows from the straight line between
        its ends (`bow_m`), how far it turns on the way (`turn_deg`), the most the chest leans sideways
        (`lean_deg`) and how much of that lean is into the bend (`into_bend`: 1 is all of it, -1 is leaning out)."""
        sc = bpy.context.scene
        rows = []
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            up = self.spine.matrix_world.to_3x3() @ Vector((0, 0, 1))
            mw = self.root.matrix_world
            left = mw.to_3x3() @ Vector((1, 0, 0))
            rows.append((mw.translation.x, mw.translation.y, mw.to_euler().z, math.degrees(math.asin(max(-1, min(1, up.dot(left)))))))
        xy = np.array([r[:2] for r in rows])
        yaw = np.unwrap(np.array([r[2] for r in rows]))
        lean = np.array([r[3] for r in rows])
        chord = xy[-1] - xy[0]
        span = float(np.linalg.norm(chord)) or 1.0
        bow = np.abs((xy[:, 0] - xy[0, 0]) * chord[1] - (xy[:, 1] - xy[0, 1]) * chord[0]) / span
        rate = np.gradient(yaw)
        turning = np.abs(rate) > math.radians(0.4)
        into = float(np.sum(lean[turning] * np.sign(rate[turning])) / max(1e-6, np.sum(np.abs(lean[turning])))) if turning.any() else 0.0
        out = {"curve": curve, "bow_m": round(float(bow.max()), 3), "turn_deg": round(math.degrees(float(yaw[-1] - yaw[0])), 1),
               "lean_deg": round(float(np.abs(lean).max()), 1), "into_bend": round(into, 2)}
        if self.desc.get("model"):
            out["arms"] = self.arms_report(f0, f1)
        return out

    def typing_report(self, keys) -> dict | None:
        """How hands with fingers typed, read back from the animation against the keys (an object): how far the
        lowest fingertip of each hand is over the keys, as a rule (`rest_m`, the median over the frames: a hand
        resting on the keys is within a few millimetres of them) and at its lowest (`into_m`, under 0 is into
        them), and how far each finger moves up and down on its own, against its wrist (`finger_travel_m`, the
        busiest finger of the hand that moves its fingers least: 0 is a hand that types without moving them)."""
        if not self.typed or keys is None:
            return None
        top = max((keys.matrix_world @ v.co).z for v in keys.data.vertices)
        sc, pb, mw = bpy.context.scene, self.rig.pose.bones, self.rig.matrix_world
        tips = {}
        for s, sd in ((1, "L"), (-1, "R")):
            names = [g["name"] for g in self.fingers[s] if g["name"] != "thumb"]
            groups = {g.index for g in self.body.vertex_groups if g.name.split(".")[0] in names and g.name.endswith("." + sd)}
            tips[s] = (names, np.array([sum(g.weight for g in v.groups if g.group in groups) > 0.5 for v in self.body.data.vertices]))
        lows, rise = {1: [], -1: []}, {1: [], -1: []}
        for f0, f1 in self.typed:
            for f in range(f0 + n(4), f1 + 1):
                sc.frame_set(f)
                ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
                me = ev.to_mesh()
                co = np.empty(len(me.vertices) * 3, dtype=np.float32)
                me.vertices.foreach_get("co", co)
                ev.to_mesh_clear()
                z = co.reshape(-1, 3) @ np.array(ev.matrix_world)[2, :3] + ev.matrix_world[2][3]
                for s, sd in ((1, "L"), (-1, "R")):
                    names, mask = tips[s]
                    lows[s].append(float(z[mask].min()) - top)
                    wrist = (mw @ pb["hand." + sd].head).z
                    rise[s].append([(mw @ pb[f"{g}.b.{sd}"].tail).z - wrist for g in names])
        travel = min(max(float(np.ptp(np.array(rise[s])[:, i])) for i in range(len(tips[s][0]))) for s in (1, -1))
        return {"rest_m": round(max(float(np.median(lows[s])) for s in (1, -1)), 4),
                "into_m": round(min(min(lows[s]) for s in (1, -1)), 4), "finger_travel_m": round(travel, 4)}

    def arms_report(self, f0: int, f1: int) -> dict:
        """How the arms hang and swing from f0 to f1, read back from the animation: the most the elbow bends
        (`elbow_deg`, 0 straight; a relaxed arm hangs at about 12 and bends a little more in front), the
        forearm's least angle below level (`forearm_deg`, 90 hanging; a forearm that comes up toward level reads
        as marching) and the furthest the hand stands out from the body's side beside it (`hand_off_m`, measured
        on the mesh; a hand held off the hip reads as a gunslinger's)."""
        sc, pb = bpy.context.scene, self.rig.pose.bones
        mask = self._arm_mask()
        hands = {}
        for s, sd in ((1, "L"), (-1, "R")):
            names = {n for n in ("hand", "thumb", "index", "middle", "ring", "little")}
            groups = {g.index for g in self.body.vertex_groups if g.name.split(".")[0] in names and g.name.endswith("." + sd)}
            hands[s] = np.array([sum(g.weight for g in v.groups if g.group in groups) > 0.5 for v in self.body.data.vertices])
        elbow, fore, off = 0.0, 90.0, 0.0
        worst = {"elbow": f0, "fore": f0, "off": f0}
        for f in range(f0, f1 + 1, 2):
            sc.frame_set(f)
            mw = self.rig.matrix_world
            for sd in "LR":
                sh, el, wr = (mw @ pb[b + sd].head for b in ("upper.", "fore.", "hand."))
                bend = math.degrees((el - sh).angle(wr - el, 0.0))
                below = math.degrees(math.asin(max(-1.0, min(1.0, -(wr - el).normalized().z))))
                if bend > elbow:
                    elbow, worst["elbow"] = bend, f
                if below < fore:
                    fore, worst["fore"] = below, f
            if (f - f0) % 6:
                continue
            ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            inv = np.array(self.root.matrix_world.inverted() @ ev.matrix_world)
            L = co.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]  # in the body's own frame, on the floor
            trunk = L[~mask]
            for s in (1, -1):
                H = L[hands[s]][::3]
                near = trunk[(trunk[:, 0] * s > 0) & (trunk[:, 2] > H[:, 2].min() - 0.03) & (trunk[:, 2] < H[:, 2].max() + 0.03)]
                gaps = []
                for p in H:
                    by = near[(np.abs(near[:, 1] - p[1]) < 0.025) & (np.abs(near[:, 2] - p[2]) < 0.025)]
                    if len(by):
                        gaps.append(p[0] * s - (by[:, 0] * s).max())
                if gaps and min(gaps) > off:
                    off, worst["off"] = float(min(gaps)), f
        return {"elbow_deg": round(elbow, 1), "forearm_deg": round(fore, 1), "hand_off_m": round(off, 3),
                "frames": worst}

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
            b.rotation_euler = (0, R(ang * s), 0)  # (the length is in the mesh)
            b.location = (self.brow_at[0] * s, self.brow_at[1], z + self.brow_at[2])
            b.keyframe_insert("rotation_euler", frame=f + dur + n(2))
            b.keyframe_insert("location", frame=f + dur + n(2))
        for name, ob in self.mouths.items():
            full = (0.8, 0.35, 1.0) if name == "o" else (size,) * 3
            tween(ob, "scale", f + max(0, dur - n(3)), f + dur, full if name == mouth else (0, 0, 0))
        if self.mouth:
            self._face_keys(f, f + dur, face)

    def _face_keys(self, f0: int, f1: int, face: str):
        """A fitted face takes an expression in its keys: the mouth's own (smile, frown, open, round) and the
        head's (the corners and cheeks with smile and frown, the jaw with open)."""
        want = fitted_face.MOUTHS[face]
        blocks = self.mouth.data.shape_keys.key_blocks
        head = self.body.data.shape_keys.key_blocks
        was = dict(zip(fitted_face.KEYS, self._mouth_now))
        to = dict(zip(fitted_face.KEYS, want))
        moves = [(blocks[name], was[name], to[name]) for name in fitted_face.KEYS]
        moves += [(head[name], was[name], to[name]) for name in ("smile", "frown")]
        moves.append((head["jaw"], min(1.0, was["open"] + was["round"]), min(1.0, to["open"] + to["round"])))
        for kb, a, b in moves:
            kb.value = a
            kb.keyframe_insert("value", frame=f0)
            kb.value = b
            kb.keyframe_insert("value", frame=f1)
        self._mouth_now = want

    def face_report(self) -> dict | None:
        """A fitted face's numbers for the check (None for the clay cast, whose face is drawn to size)."""
        if not self.mouth:
            return None
        k, at = self.head.scale[0], Vector(self.head_at)
        return fitted_face.report(self.body, self.mouth, self.face_marks, lambda p: Vector(p) * k + at - self.seat)

    def skin_report(self, frames) -> dict | None:
        """How a fitted body's skin holds up (None for the clay cast, whose mesh is made for the rig): how many
        vertices well above the hips still move with a leg (a hem or a waistband tears into spikes when the
        thighs bend), and the worst stretch of its edges over `frames` (posed length over rest length, at the
        99.9th percentile: a bent knee gives 2 to 3, weights that pull vertices apart give 5 and up)."""
        if not self.desc.get("model"):
            return None
        me, sc = self.body.data, bpy.context.scene
        joints = json.loads(self.body["joints"])
        hip, knee = Vector(joints["hipL"]), Vector(joints["knL"])
        legs = {self.body.vertex_groups[b].index for b in ("thigh.L", "thigh.R", "shin.L", "shin.R")}
        line = hip.z + 0.6 * (hip - knee).length
        high = sum(1 for v in me.vertices
                   if v.co.z > line and sum(g.weight for g in v.groups if g.group in legs) > 0.15)
        count = len(me.vertices)
        rest = np.empty(count * 3, dtype=np.float32)
        me.vertices.foreach_get("co", rest)
        rest = rest.reshape(-1, 3)
        ed = np.empty(len(me.edges) * 2, dtype=np.int32)
        me.edges.foreach_get("vertices", ed)
        ed = ed.reshape(-1, 2)
        length = np.linalg.norm(rest[ed[:, 0]] - rest[ed[:, 1]], axis=1)
        # (not the head or its join to the collar: the head is rigid, and the join is a band inside the collar
        # that has to stretch when the head turns)
        neck = self.body.vertex_groups["neck"].index
        on_head = np.array([any(g.group == neck and g.weight > 0.01 for g in v.groups) for v in me.vertices])
        keep = (length > 0.002) & ~on_head[ed[:, 0]] & ~on_head[ed[:, 1]]
        ed, length = ed[keep], length[keep]
        worst, at = 0.0, None
        for f in frames:
            sc.frame_set(f)
            ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = ev.to_mesh()
            co = np.empty(count * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            co = co.reshape(-1, 3)
            ratio = float(np.percentile(np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1) / length, 99.9))
            if ratio > worst:
                worst, at = ratio, f
        return {"leg_above_hips": high, "stretch": round(worst, 2), "frame": at}

    def _lid(self, s, f, shut: bool):
        key(self.lids[s], "rotation_euler", f, (R(fitted_face.LID_SHUT if shut else fitted_face.LID_OPEN), 0, 0))

    def breathe(self, f0: int, f1: int, period: float = 3.4, pitch: float = 0.7):
        """Idle from f0 to f1: the torso rocks a little, once every `period` seconds."""
        half = n(period * 12)  # half a period, in frames
        for i, f in enumerate(range(f0, f1 + half, half)):
            up = i % 2
            key(self.breath, "rotation_euler", f, (R(pitch if up else -pitch * 0.4), 0, 0))

    def blink(self, f: int, dur: int = 3):
        if self.lids:  # a fitted face shuts its lids; the clay cast squashes its eyes
            for s in (1, -1):
                self._lid(s, f - dur, False)
                self._lid(s, f, True)
                self._lid(s, f + dur, False)
            return
        for s in (1, -1):
            sc = tuple(self.eyes[s].scale)
            key(self.eyes[s], "scale", f - dur, sc)
            key(self.eyes[s], "scale", f, (sc[0], sc[1], 0.08))
            key(self.eyes[s], "scale", f + dur, sc)

    def wink(self, f: int, side: str = "left", dur: int = 3, hold: int = 3):
        """One eye shuts and opens (the character's own left or right), its brow dipping with it."""
        s = 1 if side == "left" else -1
        e, b = self.eyes[s], self.brows[s]
        sc = tuple(e.scale)
        if self.lids:
            for at, shut in ((f - dur, False), (f, True), (f + hold, True), (f + hold + dur, False)):
                self._lid(s, at, shut)
        else:
            key(e, "scale", f - dur, sc)
            key(e, "scale", f, (sc[0], sc[1], 0.08))
            key(e, "scale", f + hold, (sc[0], sc[1], 0.08))
            key(e, "scale", f + hold + dur, sc)
        at = tuple(b.location)
        key(b, "location", f - dur, at)
        key(b, "location", f, (at[0], at[1], at[2] - 0.022))
        key(b, "location", f + hold, (at[0], at[1], at[2] - 0.022))
        key(b, "location", f + hold + dur, at)

    def glance(self, f0: int, f1: int, yaw: float = 0.0, pitch: float = 0.0):
        """The eyes alone: pupils turned `yaw` degrees to the character's left and `pitch` degrees down (0, 0 is
        straight ahead). About 25 degrees is as far as they go."""
        x = 0.05 * math.sin(R(max(-25, min(25, yaw))))
        z = -0.05 * math.sin(R(max(-25, min(25, pitch))))
        for s in (1, -1):
            tween(self.pupils[s], "location", f0, f1, (x, -0.052, -0.012 + z))

    def look_at(self, f0: int, f1: int, point):
        """Head and eyes to a point in the room: the neck turns most of the way, the eyes do the rest."""
        sc = bpy.context.scene
        sc.frame_set(f1)
        bpy.context.view_layer.update()
        head = self.head.matrix_world.translation
        d = Vector(point) - head
        yaw_now = self.spine.matrix_world.to_euler().z  # the way the chest faces
        c, sn = math.cos(-yaw_now), math.sin(-yaw_now)
        lx, ly = c * d.x - sn * d.y, sn * d.x + c * d.y
        yaw = math.degrees(math.atan2(lx, -ly))
        pitch = -math.degrees(math.atan2(d.z, math.hypot(lx, ly)))
        # the head turns most of the way to what it looks at and bows to look down, but it barely tips back to
        # look up: the eyes do that (a chin lifted at the lens reads as looking down the nose)
        neck_yaw, neck_pitch = max(-60, min(60, yaw * 0.8)), max(-6, min(30, pitch * (0.8 if pitch > 0 else 0.25)))
        self._set_neck(f0, f1, neck_pitch, neck_yaw)
        self.glance(f0, f1, yaw - neck_yaw, pitch - neck_pitch)

    def reach(self, f0: int, f1: int, point, hand: str = "right"):
        """A hand to a point in the room (a door handle, a mug). Raises if it is further than the arm is long."""
        s = 1 if hand == "left" else -1
        sc = bpy.context.scene
        sc.frame_set(f1)
        bpy.context.view_layer.update()
        local = self.spine.matrix_world.inverted() @ Vector(point)
        gap = (local - self.SH[s]).length - 0.99 * 0.7122 * self.arm
        if gap > 0:
            raise ValueError(f"{self.id}'s {hand} hand is {gap:.2f} m short of {tuple(round(v, 2) for v in point)}: "
                             f"walk them closer first")
        self.arm_pose = "reach"
        for side in (1, -1):
            self.hand_t[side].keyframe_insert("location", frame=f0)
            self.pole_t[side].keyframe_insert("location", frame=f0)
        self._arm_key(s, f1, local, self._mirror((1, 0.6, -0.8), s))
        self.events.append({"frame": f1, "kind": "touch"})

    def hand_follows(self, f0: int, f1: int, ob, hand: str = "right"):
        """Keep a hand on something that moves (a door handle as the door swings), from f0 to f1."""
        s = 1 if hand == "left" else -1
        sc = bpy.context.scene
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            bpy.context.view_layer.update()
            local = self.spine.matrix_world.inverted() @ ob.matrix_world.translation
            self._arm_key(s, f, local, self._mirror((1, 0.6, -0.8), s))

    def sweat(self, f0: int, f1: int):
        d = self.sweat_drop
        key(d, "scale", f0 - n(4), (0, 0, 0))
        key(d, "scale", f0, (1, 1, 1.5))
        key(d, "location", f0, (self.sweat_x, -0.12, 0.15))
        key(d, "location", f1, (self.sweat_x, -0.17, -0.1))
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
    def under_floor(self, frames) -> float:
        """How far below the floor (m) the body goes at its worst over the given frames, and when: a foot
        tipped into the ground, a pelvis sat too low. 0 if it never does. `floor_worst` holds the frame."""
        sc = bpy.context.scene
        worst, self.floor_worst = 0.0, None
        for f in frames:
            sc.frame_set(f)
            ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            mw = np.array(ev.matrix_world)
            low = float((co.reshape(-1, 3) @ mw[2, :3] + mw[2, 3]).min())
            if -low > worst:
                worst, self.floor_worst = -low, f
        return worst

    def contacts(self, frames, solids=(), limit=0.03) -> dict:
        """What the body touches that it should not, and what it should hold that it does not, over the given
        frames, each as the worst case in metres with the frame it happens on:
          under_floor   how far any of the body is below the floor
          hand_in_body  how deep a hand is inside the torso, head or legs
          through       how deep the body is inside a piece of furniture; `solids` is [(object, give)], `give`
                        being how far that thing may be pressed into (a seat cushion)
          foot_slide    how far a foot travels along the floor in one stretch of standing flat on it
          touch_miss    how far a hand was from the thing it was put on (a door handle, the desk's edge)
        and for a character whose hands have fingers:
          hands_apart   the nearest the two hands come to each other (m; under half a centimetre they are
                        into each other)
          palm_tilt     degrees a hand resting over a table top is rolled from palm down onto its side, at its worst
          palm_turn     degrees a hanging hand's palm turns away from the leg during a walk, at its worst (a
                        palm that turns down and back on every swing reads as a hand flapping)
        `through` also says how long it lasted (`first` and `last` frame over `limit`)."""
        sc = bpy.context.scene
        arm = self._arm_mask()
        hand_r = 0.8 * self._girth()["hand"]
        boxes = []
        for ob, give in solids:
            corners = np.array([tuple(c) for c in ob.bound_box])
            boxes.append((ob, give, corners.min(0), corners.max(0)))
        out = {k: {"m": 0.0, "frame": None} for k in ("under_floor", "hand_in_body", "through", "foot_slide")}
        out["touch_miss"] = max(self.touches, key=lambda t: t["m"], default={"m": 0.0, "frame": None})
        run = {sd: [0.0, None] for sd in "LR"}  # this stretch's travel, and where the foot was last frame
        fingered = bool(self.fingers.get(1) and self.fingers.get(-1))
        hands, was_at, deep = {}, {}, {}
        if fingered:  # each hand's own vertices (one in four), and each fingertip
            out["hands_apart"], out["palm_tilt"] = {"m": 9.0, "frame": None}, {"deg": 0.0, "frame": None}
            out["palm_turn"] = {"deg": 0.0, "frame": None}
            names = {g.index: g.name for g in self.body.vertex_groups}
            for sd in "LR":
                mine = {i for i, nm in names.items() if nm.endswith("." + sd)
                        and nm.split(".")[0] in ("hand", "thumb", "index", "middle", "ring", "little")}
                hands[sd] = np.array([v.index for v in self.body.data.vertices
                                      if sum(g.weight for g in v.groups if g.group in mine) > 0.5])[::4]

        self.contact_trace = trace = []  # every frame's numbers, for looking into a failure
        tris = None

        def worst(kind, m, f, **more):
            trace.append((f, kind, round(float(m), 4), *more.values()))
            if m > out[kind]["m"]:
                out[kind] = {"m": float(m), "frame": f, **more}

        for f in frames:
            sc.frame_set(f)
            ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            ev.to_mesh_clear()
            mw = np.array(ev.matrix_world)
            W = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
            worst("under_floor", -float(W[:, 2].min()), f)
            for ob, give, lo, hi in boxes:
                inv = np.array(ob.matrix_world.inverted())
                L = W @ inv[:3, :3].T + inv[:3, 3]
                inside = np.all((L > lo) & (L < hi), axis=1)
                if inside.any():
                    scale = np.array(ob.matrix_world.to_scale())
                    depth = (np.minimum(L[inside] - lo, hi - L[inside]) * scale).min(axis=1).max()
                    # (a thin thing is pierced by the time anything is half way into it: a finger through a
                    # door leaf is never more than 2.5 cm from one of its faces)
                    depth *= max(1.0, 0.065 / float(((hi - lo) * scale).min()))
                    worst("through", depth - give, f, what=ob.name)
                    if depth - give > limit:
                        deep.setdefault(ob.name, []).append(f)
            if len(W) == len(arm):
                B = W[~arm]
                if tris is None:  # the triangles of everything that is not an arm (the same on every frame)
                    tris = self._body_tris(arm)
                pb, rw = self.rig.pose.bones, self.rig.matrix_world
                for sd in "LR":
                    h, t = rw @ pb["hand." + sd].head, rw @ pb["hand." + sd].tail
                    # the palm (two spots, as thick as the hand) and each fingertip (as thick as a finger)
                    probes = [(h.lerp(t, 0.35), hand_r), (h.lerp(t, 0.8), hand_r)]
                    probes += [(rw @ pb[f"{g['name']}.b.{sd}"].tail, 0.012) for g in self.fingers.get(1 if sd == "L" else -1, [])]
                    for probe, thick in probes:
                        d = np.array(probe) - B
                        near = float(np.sqrt(np.einsum("ij,ij->i", d, d).min()))
                        if near < 0.2:
                            # inside or out is asked of the surface itself: a line from the hand out to the
                            # side crosses it an odd number of times only from inside (the nearest vertex's
                            # facing is fooled by a fold of cloth or the edge of a raised knee)
                            inside = self._crossings(np.array(probe), 1.0 if sd == "L" else -1.0, W, tris) % 2 == 1
                            worst("hand_in_body", thick + near if inside else thick - near, f,
                                  hand="left" if sd == "L" else "right")
            if fingered and len(W) == len(arm):
                L, Rt = W[hands["L"]], W[hands["R"]]
                gap = float(np.sqrt(((L[:, None, :] - Rt[None, :, :]) ** 2).sum(axis=2).min()))
                trace.append((f, "hands_apart", round(gap, 4)))
                if gap < out["hands_apart"]["m"]:
                    out["hands_apart"] = {"m": gap, "frame": f}
                pb, rw = self.rig.pose.bones, self.rig.matrix_world
                for s, sd in ((1, "L"), (-1, "R")):
                    bone = pb["hand." + sd]
                    wrist = rw @ bone.head
                    elbow = rw @ pb["fore." + sd].head
                    if any(a <= f <= b for a, b, _curve in self.walks) and elbow.z - wrist.z > 0.6 * pb["fore." + sd].length:
                        spin = (rw @ bone.matrix).to_3x3() @ self.rig.data.bones["hand." + sd].matrix_local.to_3x3().inverted()
                        turn = math.degrees((spin @ self.palm[s]).angle(self.root.matrix_world.to_3x3() @ Vector((-s, 0, 0))))
                        trace.append((f, "palm_turn", round(turn, 1), "left" if s > 0 else "right"))
                        if turn > out["palm_turn"]["deg"]:
                            out["palm_turn"] = {"deg": turn, "frame": f, "hand": "left" if s > 0 else "right"}
                    still = sd in was_at and (wrist - was_at[sd]).length < 0.002
                    was_at[sd] = wrist.copy()
                    if not still:  # (a hand on its way somewhere is still turning)
                        continue
                    H = W[hands[sd]]
                    for ob, _give, lo, hi in boxes:
                        mo = ob.matrix_world
                        size = (hi - lo) * np.array(mo.to_scale())
                        if mo.to_3x3().col[2].normalized().z < 0.9 or min(size[0], size[1]) < 0.15:
                            continue  # (not something with a top to lay a hand on)
                        inv = np.array(mo.inverted())
                        Lh = H @ inv[:3, :3].T + inv[:3, 3]
                        over = np.all((Lh[:, :2] > lo[:2]) & (Lh[:, :2] < hi[:2]), axis=1)
                        if over.sum() < len(H) // 3 or (Lh[over, 2].min() - hi[2]) * mo.to_scale().z > 0.05:
                            continue
                        spin = (rw @ bone.matrix).to_3x3() @ self.rig.data.bones["hand." + sd].matrix_local.to_3x3().inverted()
                        # (rolled onto its side, about the fingers: a hand on the keys slopes down from the wrist
                        # on purpose, so the slope is not counted; a palm turned up counts as all the way over)
                        palm = spin @ self.palm[s]
                        across = (spin @ self.hand_way[s]).cross(palm).normalized()
                        tilt = math.degrees(math.asin(min(1.0, abs(across.z))))
                        tilt = tilt if palm.z <= 0 else 180.0 - tilt
                        trace.append((f, "palm_tilt", round(tilt, 1), "left" if s > 0 else "right", ob.name))
                        if tilt > out["palm_tilt"]["deg"]:
                            out["palm_tilt"] = {"deg": tilt, "frame": f, "hand": "left" if s > 0 else "right", "what": ob.name}
            if self.foot_t:
                pb, rw = self.rig.pose.bones, self.rig.matrix_world
                for sd in "LR":
                    ankle, toe = rw @ pb["foot." + sd].head, rw @ pb["foot." + sd].tail
                    flat = ankle.z < self.ankle_h + 0.004 and toe.z < self.toe_h + 0.004
                    if flat and run[sd][1] is not None:
                        run[sd][0] += (ankle.xy - run[sd][1]).length
                        worst("foot_slide", run[sd][0], f, foot="left" if sd == "L" else "right")
                    elif not flat:
                        run[sd][0] = 0.0
                    run[sd][1] = ankle.xy.copy() if flat else None
        self.floor_worst = out["under_floor"]["frame"]
        run_of = deep.get(out["through"].get("what"), [])
        if run_of:  # the stretch it lasted, round the worst frame
            at = run_of.index(out["through"]["frame"])
            a = b = at
            while a > 0 and run_of[a - 1] == run_of[a] - 1:
                a -= 1
            while b < len(run_of) - 1 and run_of[b + 1] == run_of[b] + 1:
                b += 1
            out["through"].update(first=run_of[a], last=run_of[b])
        return out

    def how_it_moves(self, f0: int, f1: int) -> dict:
        """What makes a move look like a machine's, measured over a stretch: the pelvis, chest, head, hands and
        feet are followed, and the answer is how far apart in time they start and stop (0 s: all on one frame)
        and how different the left side is from the right mirrored (0 m: the same). Posture verbs note their
        stretch in `moves`, and the check reports each."""
        sc = bpy.context.scene
        track = {k: [] for k in ("pelvis", "chest", "head", "handL", "handR", "footL", "footR")}
        own = {k: [] for k in ("handL", "handR", "footL", "footR")}  # in the body's own space, to compare sides
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            pb, mw = self.rig.pose.bones, self.rig.matrix_world
            track["pelvis"].append(tuple(mw @ pb["spine"].head))
            track["chest"].append(tuple(mw @ pb["spine"].tail))
            track["head"].append(tuple(self.head.matrix_world.translation))
            inv = self.root.matrix_world.inverted()
            for sd in "LR":
                for part in ("hand", "foot"):
                    at = mw @ pb[part + "." + sd].head
                    track[part + sd].append(tuple(at))
                    own[part + sd].append(tuple(inv @ at))
        fps = sc.render.fps
        starts, stops = [], []
        for pts in track.values():
            speed = np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1) * fps
            moving = np.where(speed > 0.03)[0]
            if len(moving):
                starts.append(int(moving[0]))
                stops.append(int(moving[-1]))
        differ = 0.0
        for part in ("hand", "foot"):
            left, right = np.array(own[part + "L"]), np.array(own[part + "R"])
            right[:, 0] *= -1
            differ = max(differ, float(np.linalg.norm(left - right, axis=1).max()))
        spread = lambda v: round((max(v) - min(v)) / fps, 3) if v else 0.0  # noqa: E731
        return {"starts_spread_s": spread(starts), "stops_spread_s": spread(stops), "sides_differ_m": round(differ, 3)}

    def _body_tris(self, arm):
        """The body's triangles (vertex indices) that have no arm vertex in them."""
        me = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tri)
        tri = tri.reshape(-1, 3)
        return tri[~arm[tri].any(axis=1)]

    @staticmethod
    def _crossings(p, way, W, tris):
        """How many of the triangles a line from point p along the x axis (`way` 1 or -1) passes through."""
        A, B, C = W[tris[:, 0]], W[tris[:, 1]], W[tris[:, 2]]
        lo, hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
        m = (lo[:, 1] <= p[1]) & (hi[:, 1] >= p[1]) & (lo[:, 2] <= p[2]) & (hi[:, 2] >= p[2]) \
            & ((hi[:, 0] >= p[0]) if way > 0 else (lo[:, 0] <= p[0]))
        A, B, C = A[m, 1:], B[m, 1:], C[m, 1:]  # seen along x: is p inside each triangle, and is the hit ahead
        q = p[1:]
        d1 = (q[0] - B[:, 0]) * (A[:, 1] - B[:, 1]) - (A[:, 0] - B[:, 0]) * (q[1] - B[:, 1])
        d2 = (q[0] - C[:, 0]) * (B[:, 1] - C[:, 1]) - (B[:, 0] - C[:, 0]) * (q[1] - C[:, 1])
        d3 = (q[0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (q[1] - A[:, 1])
        hit = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
        if not hit.any():
            return 0
        T = np.stack([W[tris[m][hit][:, k]] for k in range(3)], axis=1)       # (n, 3 corners, xyz)
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ok = np.abs(nrm[:, 0]) > 1e-12
        x = T[ok, 0, 0] - ((p[1] - T[ok, 0, 1]) * nrm[ok, 1] + (p[2] - T[ok, 0, 2]) * nrm[ok, 2]) / nrm[ok, 0]
        return int(((x - p[0]) * way > 0).sum())

    def ik_error(self, frames) -> float:
        """Largest wrist-to-target (and, with legs under control, ankle-to-target) distance (m) over the given
        frames: a pose an arm can't reach, or a step a leg can't make, shows here. `ik_worst` says where."""
        sc = bpy.context.scene
        worst, self.ik_worst = 0.0, None
        for f in frames:
            sc.frame_set(f)
            for s, sd in ((1, "L"), (-1, "R")):
                limbs = [("hand", "fore." + sd, self.hand_t[s])]
                if self.foot_t:
                    limbs.append(("foot", "shin." + sd, self.foot_t[s]))
                for name, bone, target in limbs:
                    end = self.rig.matrix_world @ self.rig.pose.bones[bone].tail
                    miss = (end - target.matrix_world.translation).length
                    if miss > worst:
                        worst, self.ik_worst = miss, {"frame": f, "limb": f"{'left' if s > 0 else 'right'} {name}"}
        return worst
