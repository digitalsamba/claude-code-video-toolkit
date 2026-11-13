"""Verbs for a presenter on their feet (runs inside Blender): talking on the narration's words, blinking, a
couple of talking gestures, dusting the hands, and a cup and a stick of chalk carried, put down and picked up.

Each takes the character (a ClayPerson) and frames. They key their own channels where they can (the mouth's
`talk` keys, the head's delta rotation), so an expression or a look set by another beat is left as it is.
"""
from __future__ import annotations

import math
import random
import re

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from kit import R, key, n, ticks

GESTURES = ("open", "both", "shrug", "count")


# ---------------------------------------------------------------- talking
def _syllables(word: str) -> list[str]:
    """A word's vowel groups, one per syllable, near enough: "production" -> o, u, io."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:  # a number written in digits: one beat per digit, up to three
        return ["e"] * max(1, min(3, len(word)))
    groups = re.findall(r"[aeiouy]+", w)
    if len(groups) > 1 and w.endswith("e") and not w.endswith(("le", "ee")):
        groups = groups[:-1]  # a silent e
    return groups or ["e"]


# how far a vowel opens the mouth and how round it is
VOWELS = {"a": (0.85, 0.0), "o": (0.6, 0.55), "u": (0.35, 0.6), "e": (0.5, 0.0), "i": (0.42, 0.0), "y": (0.42, 0.0)}


def _talk_keys(p):
    """The mouth's own keys for speech, beside the ones expressions use: `talk` (open), `talk_round` on the mouth
    and `talk_jaw` on the head, each a copy of the shape an expression would use. Shape keys add up, so a smile
    stays a smile while the mouth talks."""
    if getattr(p, "talk_blocks", None):
        return p.talk_blocks
    mouth, head = p.mouth.data.shape_keys.key_blocks, p.body.data.shape_keys.key_blocks

    def copy(ob, blocks, src, name):
        kb = ob.shape_key_add(name=name)
        co = np.empty(len(kb.data) * 3, dtype=np.float32)
        blocks[src].data.foreach_get("co", co)
        kb.data.foreach_set("co", co)
        kb.slider_max = 1.6  # (speech opens further than a smile does: a mouth that barely moves reads as a mumble)
        kb.value = 0.0
        return kb

    p.talk_blocks = (copy(p.mouth, mouth, "open", "talk"), copy(p.mouth, mouth, "round", "talk_round"),
                     copy(p.body, head, "jaw", "talk_jaw"))
    return p.talk_blocks


def talk(p, words, frame, fps, amount: float = 1.5, seed: int = 0):
    """The mouth opens and closes on the narration: `words` is [(word, start s, end s)] on the film's timeline
    and `frame` turns seconds into a frame. Each syllable is one opening, wider for an open vowel and rounder
    for an o or a u, never quite alike; the mouth shuts between words that have a gap between them and on a
    b, m or p. The head dips a little on the longer words and drifts between sentences. Only a fitted face has
    a mouth that opens: the clay cast is left as it is."""
    if not getattr(p, "mouth", None) or not words:
        return
    rng = random.Random(1000 + seed)
    knots = []  # (seconds, open, round)
    for i, (word, t0, t1) in enumerate(words):
        t1 = max(t1, t0 + 0.06)
        syl = _syllables(word)
        syl = syl[:max(1, round((t1 - t0) / 0.1))]
        gap_before = t0 - words[i - 1][2] if i else 1.0
        gap_after = words[i + 1][1] - t1 if i + 1 < len(words) else 1.0
        if gap_before > 0.1 or word[:1] in "bmp":
            knots.append((t0 - 0.02, 0.0, 0.0))
        k = len(syl)
        for j, v in enumerate(syl):
            op, rd = VOWELS.get(v[0], (0.5, 0.0))
            op *= rng.uniform(0.8, 1.12) * (1.0 if j == 0 else 0.85)
            c = t0 + (j + 0.5) / k * (t1 - t0)
            knots.append((c, op, rd))
            if j + 1 < k:
                knots.append((t0 + (j + 1) / k * (t1 - t0), 0.3 * op, 0.5 * rd))
        knots.append((t1 + 0.03, 0.0, 0.0) if gap_after > 0.1 else (t1, 0.12, 0.0))
    knots.sort()
    ts = np.array([k[0] for k in knots])
    op = np.array([k[1] for k in knots]) * amount
    rd = np.array([k[2] for k in knots]) * amount
    f0, f1 = frame(words[0][1]) - 2, frame(words[-1][2]) + 3
    opens, rounds, jaw = _talk_keys(p)
    last = None
    for f in range(f0, f1 + 1):
        t = (f - 1) / fps
        a, b = float(np.interp(t, ts, op, left=0.0, right=0.0)), float(np.interp(t, ts, rd, left=0.0, right=0.0))
        # (a held shut mouth is keyed at each end of the hold only)
        if a < 1e-3 and b < 1e-3 and last == 0 and f not in (f0, f1) and \
                float(np.interp(f / fps, ts, op, left=0.0, right=0.0)) < 1e-3:
            continue
        last = 0 if a < 1e-3 and b < 1e-3 else 1
        for kb, v in ((opens, min(1.5, a)), (rounds, min(1.2, b)), (jaw, min(1.4, 0.8 * a + 0.5 * b))):
            kb.value = v
            kb.keyframe_insert("value", frame=f)
    # the head: a small dip on a word worth one, a slow drift to one side and back between them
    neck = p.neck
    neck.delta_rotation_euler = (0, 0, 0)
    neck.keyframe_insert("delta_rotation_euler", frame=f0 - n(4))
    yaw = roll = 0.0
    next_drift = words[0][1]
    for word, t0, t1 in words:
        if t0 >= next_drift:
            yaw, roll = rng.uniform(-3.0, 3.0), rng.uniform(-1.6, 1.6)
            next_drift = t0 + rng.uniform(1.4, 2.8)
            key(neck, "delta_rotation_euler", frame(t0 + 0.35), (0, R(roll), R(yaw)))
        if len(word) >= 6 and rng.random() < 0.45:
            dip = rng.uniform(1.5, 3.2)
            key(neck, "delta_rotation_euler", frame(t0 + 0.08), (R(dip), R(roll), R(yaw)))
            key(neck, "delta_rotation_euler", frame(t0 + 0.34), (0, R(roll), R(yaw)))
    key(neck, "delta_rotation_euler", f1 + n(10), (0, 0, 0))


def blinks(p, f0: int, f1: int, fps, seed: int = 0):
    """Blink from f0 to f1 as people do: every two to five seconds, now and then twice."""
    rng = random.Random(2000 + seed)
    f = f0 + round(rng.uniform(0.6, 2.5) * fps)
    while f < f1 - n(4):
        p.blink(f, n(3))
        if rng.random() < 0.15:
            p.blink(f + n(8), n(3))
        f += round(rng.uniform(2.2, 5.2) * fps)


# ---------------------------------------------------------------- hands
def _side(hand):
    return 1 if hand == "left" else -1


def _hand_state(p, s, f):
    """Hold hand `s` where it is at f, and return what puts it back there."""
    sd = "L" if s > 0 else "R"
    bone = p.rig.pose.bones.get("hand." + sd)
    p.hand_t[s].keyframe_insert("location", frame=f)
    p.pole_t[s].keyframe_insert("location", frame=f)
    q = None
    if bone and p.fingers.get(s):
        bone.keyframe_insert("rotation_quaternion", frame=f)
        q = bone.rotation_quaternion.copy()
    return p.hand_t[s].location.copy(), p.pole_t[s].location.copy(), q, p.hand_shape_now.get(s, "relaxed")


def _hand_home(p, s, f, state):
    """Put hand `s` back where `state` says by frame f."""
    hand, pole, q, shape = state
    key(p.hand_t[s], "location", f, hand)
    key(p.pole_t[s], "location", f, pole)
    if q is not None:
        bone = p.rig.pose.bones["hand." + ("L" if s > 0 else "R")]
        q = q if bone.rotation_quaternion.dot(q) >= 0 else -q
        bone.rotation_quaternion = q
        bone.keyframe_insert("rotation_quaternion", frame=f)
        p.wrist_q[s] = q
    if p.fingers.get(s):
        p._fingers_key(s, f, shape)


def _own(p):
    return p.root.matrix_world.to_3x3()


def gesture(p, f0: int, f1: int, kind: str = "open", hand: str | None = None):
    """A talking gesture, held from f0 to f1 and then back to where the hands were. "open": one forearm lifts,
    the hand open and palm up, with a small beat down in the middle (a point being made). "both": the two hands
    open out in front. "shrug": both hands turn out and up, wider, the head tipping. "count": one hand lifts,
    the index out. With no `hand`, the left (the right one usually has the chalk)."""
    if kind not in GESTURES:
        raise ValueError(f"unknown gesture {kind!r}: one of {', '.join(GESTURES)}")
    sc = bpy.context.scene
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    sides = (1, -1) if kind in ("both", "shrug") else (_side(hand or "left"),)
    up = f0 + max(2, min(n(10), (f1 - f0) // 3))
    L = 0.7122 * p.arm
    wide = {"open": 0.04, "both": -0.08, "shrug": 0.02, "count": 0.02}[kind]
    lift = {"open": -0.4, "both": -0.44, "shrug": -0.5, "count": -0.2}[kind]
    ahead = {"open": -0.5, "both": -0.5, "shrug": -0.4, "count": -0.5}[kind]
    states = {s: _hand_state(p, s, f0) for s in sides}
    own = _own(p)
    for s in sides:
        wrist = Vector(p.SH[s]) + Vector((s * wide * L, ahead * L, lift * L))
        if kind == "count":
            lay = lambda f, s=s: (own @ Vector((-s, 0, 0)), own @ Vector((0, -0.45, 0.9)))  # noqa: E731
        else:
            # (the palm faces in and a little up, the thumb on top: an open hand offered. Turned palm up the
            # hand twists a quarter turn and more on the forearm, and the cuff wrings like a cloth)
            tilt = 0.55 if kind == "shrug" else 0.2
            lay = lambda f, s=s, tilt=tilt: (own @ Vector((-s, 0, tilt)),  # noqa: E731
                                             own @ Vector((s * (0.15 + tilt * 0.5), -1, 0.1)))
        pole = p._mirror((0.3, 1, -0.7) if kind == "shrug" else (0.5, 1, -0.3), s)
        name = "left" if s > 0 else "right"
        p.hand_shape(f0, up, "point" if kind == "count" else "open", name)
        late = n(2) if s != sides[0] else 0  # (the second hand is a little behind the first)
        p._arm_key(s, up + late, wrist, pole, lay)
        if f1 - up >= n(10):
            mid = (up + f1) // 2
            p._arm_key(s, mid - n(2), wrist, pole, lay)
            p._arm_key(s, mid + n(2), wrist + Vector((0, -0.02, -0.035)), pole, lay)
        p._arm_key(s, f1, wrist, pole, lay)
        _hand_home(p, s, f1 + n(11) + late, states[s])
    if kind == "shrug":  # the head tips with it
        d = p.neck.delta_rotation_euler.copy()
        p.neck.keyframe_insert("delta_rotation_euler", frame=f0)
        key(p.neck, "delta_rotation_euler", up, (d.x, d.y + R(5), d.z))
        key(p.neck, "delta_rotation_euler", f1, (d.x, d.y + R(5), d.z))
        key(p.neck, "delta_rotation_euler", f1 + n(10), d)
    p.events.append({"frame": up, "kind": "rustle"})


def present(p, f0: int, f1: int, target, hand: str | None = None, turn: float = 30.0):
    """Show something that is beside or behind them (a board): the chest turns part of the way to it, the head
    and eyes go to it, and the arm on that side points at it, held to f1; then the arm and the chest come back.
    `turn` is the most the chest turns, in degrees. Follow it with a look_at for where the eyes go next."""
    sc = bpy.context.scene
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    local = p.root.matrix_world.inverted() @ Vector(target)
    side = math.degrees(math.atan2(local.x, -local.y))  # degrees to their left of straight ahead
    yaw = max(-turn, min(turn, 0.4 * side))
    lag = n(p.LAG["chest"])
    p._torso(f0 - lag)
    p._torso(f0 + n(7) - lag, yaw=yaw, roll=-0.08 * yaw)
    p.neck.keyframe_delete("rotation_euler", frame=f0 + n(7) - lag + n(p.LAG["head"]))  # (the look below is the head's)
    p._torso(f1 - lag, yaw=yaw, roll=-0.08 * yaw)
    p._torso(f1 + n(12) - lag)
    # (the head goes to it after the chest has started round, stays on it for the first half of the hold and
    # comes back with the chest's last key: looked at before the chest's keys, the head was taken back to the
    # hall by them within a few frames, a flick to the board and away)
    p.look_at(f0 + n(4), f0 + n(11), target)
    stay = f0 + n(11) + max(1, (f1 - f0 - n(11)) // 2)
    p.neck.keyframe_insert("rotation_euler", frame=stay)
    for s in (1, -1):
        p.pupils[s].keyframe_insert("location", frame=stay)
    p.point(f0 + n(3), f1, target, hand, True)


def dust_hands(p, f0: int, f1: int):
    """The hands come up in front of the chest, palms together, and brush past each other up and down (chalk
    dust), then go back where they were."""
    sc = bpy.context.scene
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    L = 0.7122 * p.arm
    up = f0 + n(9)
    own = _own(p)
    states = {s: _hand_state(p, s, f0) for s in (1, -1)}
    thick = max(p._hand_under(s, "flat")[0] for s in (1, -1)) if p.fingers.get(1) and p.fingers.get(-1) else 0.03
    half = thick + 0.006  # each wrist this far from the middle: the palms nearly meet
    p.hand_shape(f0, up, "flat")
    beats = list(ticks(up, f1, 4))
    for s in (1, -1):
        lay = lambda f, s=s: (own @ Vector((-s, 0, 0)), own @ Vector((0, -0.8, 0.6)))  # noqa: E731
        base = Vector((s * half, -0.52 * L, p.SH[s].z - 0.42 * L))
        pole = p._mirror((0.7, 0.8, -0.5), s)
        for i, f in enumerate(beats):
            way = s if i % 2 == 0 else -s
            p._arm_key(s, f, base + Vector((0, -0.012 * way, 0.04 * way)), pole, lay)
        # (out round the belly on the way back to the side, not through it)
        p._arm_key(s, f1 + n(6), base + Vector((s * 0.17, -0.02, -0.12)), pole, lay)
        _hand_home(p, s, f1 + n(15), states[s])
    for f in beats[1:]:
        p.events.append({"frame": f, "kind": "dust"})


# ---------------------------------------------------------------- things in the hand
def _carry_lay(p, s):
    own = _own(p)
    return lambda f: (own @ Vector((-s, 0, 0)), own @ Vector((0, -1, 0.12)))


def _carry_wrist(p, s):
    L = 0.7122 * p.arm
    # (in front of the body, between the shoulder and the middle, a little under the chest)
    return Vector((p.SH[s].x * 0.5, -0.62 * L, p.SH[s].z - 0.36 * L)), p._mirror((0.6, 1, -0.5), s)


def _bone_parent(ob, p, bone):
    ob.parent, ob.parent_type, ob.parent_bone = p.rig, "BONE", bone
    bpy.context.view_layer.update()


def _palm_now(p, s):
    """The way hand `s`'s palm faces and its fingers point, in the room, now."""
    if s not in p.palm:  # (a mitten: as the body faces)
        return _own(p) @ Vector((-s, 0, 0)), _own(p) @ Vector((0, -1, 0))
    sd = "L" if s > 0 else "R"
    bone, rest = p.rig.pose.bones["hand." + sd], p.rig.data.bones["hand." + sd]
    spin = (p.rig.matrix_world @ bone.matrix).to_3x3() @ rest.matrix_local.to_3x3().inverted()
    return (spin @ p.palm[s]).normalized(), (spin @ p.hand_way[s]).normalized()


def carry(p, ob, f: int, hand: str = "left", radius: float = 0.04):
    """From frame f the character carries `ob` (a cup, upright) in a hand held in front of them, and that arm
    stays out of a walk's swing. The cup rides the hand bone."""
    s = _side(hand)
    sc = bpy.context.scene
    wrist, pole = _carry_wrist(p, s)
    p.hand_shape(f, f, "hold", hand)
    p.busy.add(s)
    p._arm_key(s, f, wrist, pole, _carry_lay(p, s))
    sc.frame_set(f)
    bpy.context.view_layer.update()
    p.holding[s] = True
    at = p._hold_point(s)
    p.holding[s] = False
    palm, way = _palm_now(p, s)
    _bone_parent(ob, p, "hand." + ("L" if s > 0 else "R"))
    yaw = p.root.matrix_world.to_euler().z
    # (held by its upper half, so the hand is clear of whatever the cup is put down on)
    ob.matrix_world = Matrix.Translation(at + palm * (radius - 0.012) + way * 0.005 - Vector((0, 0, 0.028))) \
        @ Matrix.Rotation(yaw, 4, "Z")
    key(ob, "scale", f, (1, 1, 1))
    p.carry_spans.setdefault(s, []).append([f, 10 ** 9])


def put_down(p, ob_hand, ob_table, spot, f0: int, f1: int, hand: str = "left", lean: float = 9.0):
    """The hand that carries `ob_hand` takes it to `spot` (a point in the room: the middle of the thing as it
    stands there) by f1 and lets go: from then `ob_table` stands there and the hand goes back to the side. The
    chest leans to it a little. Raises if the arm is too short from where they stand."""
    s = _side(hand)
    sc = bpy.context.scene
    sd = "L" if s > 0 else "R"
    lay = _carry_lay(p, s)
    spot = Vector(spot)
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    local = p.root.matrix_world.inverted() @ spot
    side = math.degrees(math.atan2(local.x, -local.y))
    start, pole = p.hand_t[s].location.copy(), p._mirror((0.6, 1, -0.5), s)
    for side_ in (1, -1):
        p.hand_t[side_].keyframe_insert("location", frame=f0)
        p.pole_t[side_].keyframe_insert("location", frame=f0)
    lag = n(p.LAG["chest"])
    p._look_first(f0 + n(2), f1 + n(6), yaw=max(-20, min(20, side)), pitch=18)
    p._torso(f0 - lag)
    p._torso(f1 - lag, pitch=lean, yaw=0.35 * side, head=(14, 0.3 * side))
    sc.frame_set(f1)
    bpy.context.view_layer.update()
    inv = p.spine.matrix_world.inverted().to_3x3()
    at = start.copy()
    reach = 0.985 * 0.7122 * p.arm
    for _ in range(12):
        p._arm_key(s, f1, at, pole, lay)
        bpy.context.view_layer.update()
        miss = inv @ (spot - ob_hand.matrix_world.translation)
        if miss.length < 0.0015:
            break
        at = at + miss
        arm = at - Vector(p.SH[s])
        if arm.length > reach:
            raise ValueError(f"{p.id}'s {hand} hand is {arm.length - reach:.2f} m short of putting it down at "
                             f"{tuple(round(v, 2) for v in spot)}: stand them closer, or move the spot")
    # (lifted a little on the way, so it comes down onto the table and not along it)
    mid = (f0 + f1) // 2
    p._arm_key(s, mid, start.lerp(at, 0.5) + Vector((0, 0, 0.03)), pole, lay)
    p._arm_key(s, f1, at, pole, lay)
    # the swap: the one in the hand goes, the one on the table is there
    key(ob_hand, "scale", f1, (1, 1, 1))
    key(ob_hand, "scale", f1 + 1, (0, 0, 0))
    ob_table.location = spot
    ob_table.rotation_euler = (ob_table.rotation_euler.x, ob_table.rotation_euler.y, p.root.matrix_world.to_euler().z)
    key(ob_table, "scale", f1, (0, 0, 0))
    key(ob_table, "scale", f1 + 1, (1, 1, 1))
    p.events.append({"frame": f1, "kind": "put_down"})
    # let go, and back to the side
    p.hand_shape(f1 + n(1), f1 + n(5), "relaxed", hand)
    p._arm_key(s, f1 + n(3), at + Vector((0, 0.02, 0.01)), pole, lay)
    p.busy.discard(s)
    if p.carry_spans.get(s):
        p.carry_spans[s][-1][1] = f1
    down, elbow = p._down(s)
    # (out round the belly on the way back to the side, not through it)
    p._arm_key(s, f1 + n(10), at.lerp(Vector(down), 0.5) + Vector((s * 0.09, -0.06, 0.03)), pole, None)
    p._arm_key(s, f1 + n(18), down, p._mirror(elbow, s), "side")
    p._torso(f1 + n(4), pitch=lean * 0.5, yaw=0.2 * side, head=(6, 0.15 * side))
    p._torso(f1 + n(14))
    p.swung = {}


def hold_small(p, ob, f: int, hand: str = "right"):
    """From frame f a small thing (a stick of chalk) is in the hand: it lies along the index finger, the thumb
    on it, and the hand keeps that shape through whatever else it does."""
    s = _side(hand)
    sd = "L" if s > 0 else "R"
    p.keeps[s] = "chalk"
    sc = bpy.context.scene
    if p.fingers.get(s):
        p._fingers_key(s, f, "chalk")
    sc.frame_set(f)
    bpy.context.view_layer.update()
    tip_bone = p.rig.pose.bones.get(f"index.b.{sd}") or p.rig.pose.bones["hand." + sd]
    mw = p.rig.matrix_world
    head, tip = mw @ tip_bone.head, mw @ tip_bone.tail
    along = (tip - head).normalized()
    palm, _ = _palm_now(p, s)
    _bone_parent(ob, p, tip_bone.name)
    turn = along.to_track_quat("Z", "Y").to_matrix().to_4x4()
    ob.matrix_world = Matrix.Translation(tip - along * 0.012 + palm * 0.011) @ turn
    key(ob, "scale", f - 1, (0, 0, 0))
    key(ob, "scale", f, (1, 1, 1))


def pick_up(p, ob_hand, ob_table, f0: int, f1: int, hand: str = "right"):
    """A hand goes to a small thing on the table (`ob_table`), takes it by f1 (from then `ob_hand` is in the
    hand) and goes back to the side."""
    s = _side(hand)
    sc = bpy.context.scene
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    spot = ob_table.matrix_world.translation.copy()
    state = _hand_state(p, s, f0)
    _hand_state(p, -s, f0)
    local = p.root.matrix_world.inverted() @ spot
    side = math.degrees(math.atan2(local.x, -local.y))
    lag = n(p.LAG["chest"])
    p._look_first(f0 + n(2), f1 + n(6), yaw=max(-20, min(20, side)), pitch=18)
    p._torso(f0 - lag)
    p._torso(f1 - lag, pitch=12, yaw=0.5 * side, head=(14, 0.3 * side))
    sc.frame_set(f1)
    bpy.context.view_layer.update()
    wrist = p._wrist_for(s, spot + Vector((0, 0, 0.05)))
    gap = (p.spine.matrix_world.inverted() @ spot - Vector(p.SH[s])).length - 0.98 * (0.7122 * p.arm + p.hand_len)
    if gap > 0:
        raise ValueError(f"{p.id}'s {hand} hand is {gap:.2f} m short of the thing on the table: stand them closer")
    pole = p._mirror((0.6, 1, -0.5), s)
    p.hand_shape(f0, f1 - n(3), "open", hand)
    # (up and over the table's edge on the way, and out round the hip on the way back: a hand taken straight
    # from the side to the table top goes through the table, and straight back goes through the thigh)
    over = state[0].lerp(wrist, 0.6) + Vector((s * 0.03, -0.03, 0.1))
    p._arm_key(s, (f0 + f1) // 2, over, pole, "flat")
    p._arm_key(s, f1, wrist, pole, "flat")
    key(ob_table, "scale", f1, tuple(ob_table.scale))
    key(ob_table, "scale", f1 + 1, (0, 0, 0))
    hold_small(p, ob_hand, f1 + 1, hand)
    p._arm_key(s, f1 + n(3), wrist + Vector((0, 0.01, 0.02)), pole, "flat")
    p.events.append({"frame": f1, "kind": "pick_up"})
    hand_at, pole_at, q, _ = state
    p._arm_key(s, f1 + n(9), hand_at.lerp(wrist, 0.45) + Vector((s * 0.07, -0.05, 0.05)), pole, None)
    _hand_home(p, s, f1 + n(18), (hand_at, pole_at, q, "chalk"))
    p._torso(f1 + n(4), pitch=3, yaw=0.2 * side, head=(6, 0.15 * side))
    p._torso(f1 + n(14))
    p.swung = {}


def leave_small(p, ob_hand, ob_table, spot, f0: int, f1: int, hand: str = "right"):
    """The hand takes the small thing it holds to `spot` on the table by f1 and leaves it there."""
    s = _side(hand)
    sc = bpy.context.scene
    sc.frame_set(f0)
    bpy.context.view_layer.update()
    spot = Vector(spot)
    state = _hand_state(p, s, f0)
    _hand_state(p, -s, f0)
    local = p.root.matrix_world.inverted() @ spot
    side = math.degrees(math.atan2(local.x, -local.y))
    lag = n(p.LAG["chest"])
    p._look_first(f0 + n(2), f1 + n(6), yaw=max(-20, min(20, side)), pitch=18)
    p._torso(f0 - lag)
    p._torso(f1 - lag, pitch=13, yaw=0.6 * side, head=(14, 0.3 * side))
    sc.frame_set(f1)
    bpy.context.view_layer.update()
    gap = (p.spine.matrix_world.inverted() @ spot - Vector(p.SH[s])).length - 0.98 * (0.7122 * p.arm + p.hand_len)
    if gap > 0:
        raise ValueError(f"{p.id}'s {hand} hand is {gap:.2f} m short of {tuple(round(v, 2) for v in spot)}: "
                         f"stand them closer, or move the spot")
    wrist = p._wrist_for(s, spot + Vector((0, 0, 0.05)))
    pole = p._mirror((0.6, 1, -0.5), s)
    p._arm_key(s, (f0 + f1) // 2, state[0].lerp(wrist, 0.6) + Vector((s * 0.03, -0.03, 0.1)), pole, "flat")
    p._arm_key(s, f1, wrist, pole, "flat")
    key(ob_hand, "scale", f1, (1, 1, 1))
    key(ob_hand, "scale", f1 + 1, (0, 0, 0))
    ob_table.location = spot
    key(ob_table, "scale", f1, (0, 0, 0))
    key(ob_table, "scale", f1 + 1, (1, 1, 1))
    p.keeps.pop(s, None)
    p.events.append({"frame": f1, "kind": "put_down"})
    p._arm_key(s, f1 + n(3), wrist + Vector((0, 0.01, 0.02)), pole, "flat")
    hand_at, pole_at, q, _ = state
    p._arm_key(s, f1 + n(9), hand_at.lerp(wrist, 0.45) + Vector((s * 0.07, -0.05, 0.05)), pole, None)
    _hand_home(p, s, f1 + n(18), (hand_at, pole_at, q, "relaxed"))
    p._torso(f1 + n(4), pitch=4, yaw=0.3 * side, head=(6, 0.15 * side))
    p._torso(f1 + n(14))
    p.swung = {}
