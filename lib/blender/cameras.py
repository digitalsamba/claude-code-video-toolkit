"""Shot cameras placed by type and verified by geometry (runs inside Blender).

A shot names a *type* and a *subject*, not coordinates:

    {"id": "S2", "dur": 2.5, "type": "over_shoulder", "subject": "screen"}

Each type is a placement relative to the subject's anchor (distance, angle round the subject, height, lens).
Before a camera is kept, `solve()` checks it at the start, middle and end of the shot:

  * in frame: the subject's key points project inside the picture (with a margin),
  * unblocked: a ray from the lens to each key point hits the subject first, not a head or a lid in the way.

If the nominal placement fails it tries nearby ones (swinging round the subject, then rising) and keeps the first
that passes, recording what it changed and why. That is the "his head blocks the screen" fix, done by geometry.
Angles: 0 is straight in front of the subject; positive swings toward the subject's own left.

Three screen shots are moves, not placements. `screen_fill` sits square on the screen at the distance where the
picture covers the frame edge to edge, so a flat Remotion scene can cut to the same scene on the laptop.
`pull_out` starts there, holds, then travels back over the shoulder to a wide view of the desk; `push_in` is the
same move reversed, ending on the screen, ready to cut back to the flat scene.
"""
from __future__ import annotations

import math
import random

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from kit import empty, key, link, n, ticks

D = bpy.data

SHOT_TYPES = {
    #               target offset (subject space)   dist  az   height lens  push  focus
    "wide":          dict(target=(0, -0.35, 1.25), dist=3.7, az=-34, h=0.5, lens=35, push=0.1, check="face"),
    "front_wide":    dict(target=(0, -0.3, 1.6), dist=3.5, az=0, h=0.0, lens=30, push=0.1, check="face"),
    "medium":        dict(target=(0, 0, 1.45), dist=2.55, az=-21, h=0.15, lens=38, push=0.03, check="face"),
    "closeup":       dict(target=(0, 0, 1.7), dist=2.1, az=-4, h=0.1, lens=50, push=0.14, check="head", dof=2.8),
    "side":          dict(target=(0, -0.4, 1.25), dist=2.32, az=-88, h=0.2, lens=40, push=0.04, check="face"),
    "over_shoulder": dict(target=(0, 0, 0), dist=1.58, az=-52, h=0.82, lens=45, push=0.08, check="screen", dof=3.5),
    "insert":        dict(target=(0, 0, 0), dist=1.27, az=-51, h=0.5, lens=50, push=0.1, check="screen", dof=3.5),
    # moves from/to the screen filling the frame: az/elev (degrees) and dist (m) are the far end, hold in seconds
    "screen_fill":   dict(move="fill", lens=28, check="screen"),
    "pull_out":      dict(move="out", lens=28, az=-78, elev=20, dist=2.6, hold=0.4, aim=(0.55, 0.3), check="screen"),
    "push_in":       dict(move="in", lens=28, az=-78, elev=20, dist=2.6, hold=0.4, aim=(0.55, 0.3), check="screen"),
    # a prop on a studio set: the distance comes from the prop's size (`fill` is the share of the frame it takes)
    "product":       dict(prop=True, az=-28, elev=14, lens=60, fill=0.7, push=0.06, check="prop"),
    "orbit":         dict(prop=True, az=-55, az_to=55, elev=12, lens=60, fill=0.7, push=0.0, check="prop"),
    # starts on a prop's screen edge to edge (as screen_fill), holds, then pulls back to the `product` view
    "reveal":        dict(move="out", lens=50, az=-28, elev=14, hold=0.4, fill=0.85, check="screen"),
}
READABLE_DEG = 60  # a screen turned further than this from the lens is not being read
# tried in order until one passes: (swing degrees, extra height m, distance factor)
CANDIDATES = [(0, 0, 1), (0, 0, 1.15), (12, 0, 1), (-12, 0, 1), (24, 0, 1), (-24, 0, 1), (0, 0.3, 1), (12, 0.3, 1),
              (-12, 0.3, 1), (0, 0, 1.3), (36, 0, 1), (-36, 0, 1), (24, 0, 1.15), (-24, 0, 1.15), (0, 0.6, 1)]
BLOCKED_LIMIT = 2  # key points hidden behind something else before a view counts as blocked
MARGIN = 0.03
NEAR_SCREEN = 0.78  # a move is not checked closer to the screen than this (m)
CLEARANCE = 0.3    # a moving lens stays this far (m) from the cast, or a head fills the frame as it passes


class Subject:
    """What a camera points at: a character (faces -Y) or the laptop screen (faces +Y)."""

    def __init__(self, kind, origin, front, points_fn, frame_fn, visible, focus, normal=None, size=None):
        self.kind, self.origin, self.front = kind, Vector(origin), Vector(front)
        self.points_fn, self.frame_fn, self.visible, self.focus = points_fn, frame_fn, visible, focus
        self.normal, self.size = normal, size  # a screen's facing direction and (width, height) in metres


def character_subject(person):
    return Subject("character", (0, 0, 0), (0, -1, 0), person.face_points, person.frame_points,
                   person.visible_objects(), person.head)


def screen_subject(room):
    return Subject("screen", room.screen_center(), (0, 1, 0), room.screen_points, room.screen_points,
                   room.visible_objects(), room.lid, room.screen_normal(), (room.sw, room.sh))


def prop_subject(prop):
    centre, radius = prop.bounds()
    inner = lambda: [centre + (c - centre) * 0.9 for c in prop.corners()]  # noqa: E731  (rays aim inside the box)
    sub = Subject("prop", centre, (0, -1, 0), inner, prop.corners, prop.objects(), prop.root, size=radius)
    sub.prop = prop
    return sub


def prop_screen_subject(prop):
    """A phone's or a card's screen, for the moves that start on it edge to edge."""
    ob, w, h = prop.screen
    n = prop.screen_normal()
    flat = Vector((n.x, n.y, 0)).normalized()
    sub = Subject("screen", ob.matrix_world.translation, flat, prop.screen_points, prop.screen_points,
                  prop.objects(), ob, n, (w, h))
    sub.prop = prop
    return sub


def fit_distance(radius, lens, aspect, fill, sensor=36.0):
    """How far back a lens has to be for a sphere of this radius to take `fill` of the frame's short side."""
    half = math.atan(sensor / 2 / lens * min(aspect, 1 / aspect) if aspect != 1 else sensor / 2 / lens)
    return radius / (fill * math.tan(half))


def _offset(front, az_deg, dist):
    a = math.radians(az_deg)
    x, y = front.x, front.y
    return Vector((x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a), 0)).normalized() * dist


def _nearest(dg, point, objects):
    """Distance from a point to the nearest surface of the given objects, and that object's name."""
    best = (1e9, "")
    for ob in objects:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        if abs(ev.matrix_world.determinant()) < 1e-12:  # scaled to nothing: a hidden mouth shape, a sweat drop
            continue
        ok, loc, _n, _i = ev.closest_point_on_mesh(ev.matrix_world.inverted() @ point)
        if ok:
            d = (ev.matrix_world @ loc - point).length
            if d < best[0]:
                best = (d, ob.name)
    return best


def _check(cam, sub, frames, place, kind, in_frame=None, avoid=()):
    sc = bpy.context.scene
    issues = []
    for f in frames:
        sc.frame_set(f)
        cam.location = place(f)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        if avoid and (in_frame is None or in_frame(f)):
            near, who = _nearest(dg, cam.matrix_world.translation, avoid)
            if near < CLEARANCE:
                issues.append(f"f{f}: lens {near:.2f} m from {who}")
        pts = sub.points_fn()
        frame_pts = sub.frame_fn() if kind in ("head", "prop") else pts
        for p in frame_pts if in_frame is None or in_frame(f) else ():
            v = world_to_camera_view(sc, cam, p)
            if v.z <= 0 or not (MARGIN <= v.x <= 1 - MARGIN and MARGIN <= v.y <= 1 - MARGIN):
                issues.append(f"f{f}: subject point off frame ({v.x:.2f}, {v.y:.2f})")
                break
        origin = cam.matrix_world.translation
        blocked = 0
        for p in pts:
            d = p - origin
            # a prop stands on the floor, so its rays stop short of their point; other subjects reach just past it
            reach = d.length * 0.98 if kind == "prop" else d.length + 0.05
            hit, _loc, _n, _i, ob, _m = sc.ray_cast(dg, origin, d.normalized(), distance=reach)
            if hit and ob.original not in sub.visible:
                blocked += 1
                culprit = ob.original.name
        if blocked >= BLOCKED_LIMIT:
            issues.append(f"f{f}: view blocked by {culprit}")
    return issues


def fill_distance(size, lens, aspect, sensor=36.0):
    """Distance at which a w x h panel, seen square on, covers a frame of the given aspect edge to edge."""
    w, h = size
    return (min(w, h * aspect) if aspect >= 1 else min(h, w / aspect)) * lens / sensor


def _ease(t):
    return t * t * (3 - 2 * t)


def _solve_move(shot, spec, sub, f0, f1, fps, avoid=()):
    """screen_fill / pull_out / push_in: a camera that starts (or ends) with the screen covering the frame."""
    if sub.kind != "screen":
        raise ValueError(f'shot {shot["id"]}: {shot["type"]} needs "subject": "screen"')
    sc = bpy.context.scene
    target = Vector(sub.origin)
    n = Vector(sub.normal)
    d0 = fill_distance(sub.size, spec["lens"], sc.render.resolution_x / sc.render.resolution_y)
    elev0 = math.degrees(math.asin(max(-1, min(1, n.z))))
    cd = D.cameras.new(shot["id"])
    cd.lens, cd.clip_start = spec["lens"], 0.02
    cam = link(D.objects.new(shot["id"], cd))
    tgt = empty(shot["id"] + "_tgt", None, target)
    cam.constraints.new("TRACK_TO").target = tgt
    move = spec["move"]
    hold = int(round(spec.get("hold", 0) * fps))
    span = max(1, f1 - f0 - hold)

    def progress(f):
        t = (f - f0 - hold) / span if move == "out" else (f1 - hold - f) / span
        return _ease(max(0.0, min(1.0, t)))

    # at the far end the camera looks past the screen, toward whoever sits at it (aim: metres forward, metres up)
    fwd, up = spec.get("aim", (0, 0))
    drift = Vector(spec["aim_at"]) - target if "aim_at" in spec else Vector(sub.front) * fwd + Vector((0, 0, up))
    if move != "fill" and drift.length > 1e-6:
        for f in range(f0, f1 + 1):
            key(tgt, "location", f, target + drift * progress(f) ** 2)

    def placer(daz, dh, back):
        az1, elev1, d1 = spec.get("az", 0) + daz, spec.get("elev", elev0) + dh * 30, spec.get("dist", d0) * back

        def place(f):
            if move == "fill":
                return target + n * d0
            s = progress(f)
            # swing out early and back away late, so the lens clears the head it starts in front of
            az, elev, d = az1 * s ** 0.6, elev0 + (elev1 - elev0) * s ** 0.8, d0 + (d1 - d0) * s ** 1.7
            flat = _offset(sub.front, az, 1.0)
            e = math.radians(elev)
            return target + Vector((flat.x * math.cos(e), flat.y * math.cos(e), math.sin(e))) * d
        return place

    frames = [f0, f1] if move == "fill" else [f0 + round((f1 - f0) * i / 12) for i in range(13)]
    # the checks start once the camera has left the screen: before that the picture covers the frame and the lens
    # is over the keyboard by design. In metres, not multiples of d0, which is small in a portrait frame.
    near = spec.get("near", max(d0 * 2.2, NEAR_SCREEN))
    wide = lambda f: (placer(0, 0, 1)(f) - target).length > near  # noqa: E731
    report = {"id": shot["id"], "type": shot["type"], "frames": [f0, f1], "fill_distance_m": round(d0, 3)}
    best = None
    for daz, dh, back in ([(0, 0, 1)] if move == "fill" else CANDIDATES):
        issues = _check(cam, sub, frames, placer(daz, dh, back), "screen", in_frame=wide, avoid=avoid)
        if best is None or len(issues) < len(best[3]):
            best = (daz, dh, back, issues)
        if not issues:
            break
    daz, dh, back, issues = best
    adjusted = (daz, dh, back) != (0, 0, 1)
    report.update(passed=not issues, issues=issues, adjusted=adjusted, swing=daz, rise=dh, back=back)
    if adjusted:
        report["nominal_issues"] = _check(cam, sub, frames, placer(0, 0, 1), "screen", in_frame=wide, avoid=avoid)
    place = placer(daz, dh, back)
    for f in range(f0, f1 + 1):
        key(cam, "location", f, place(f))
    return cam, report


def _facing(sub, cam_at):
    """Degrees between a prop's screen and the line to the lens (0 = square on), or None if it has no screen."""
    prop = getattr(sub, "prop", None)
    if not prop or not prop.screen:
        return None
    to_cam = (cam_at - prop.screen[0].matrix_world.translation).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, prop.screen_normal().dot(to_cam)))))


def _solve_prop(shot, spec, sub, f0, f1):
    """product / orbit: a camera round a prop, at the distance its size asks for."""
    sc = bpy.context.scene
    aspect = sc.render.resolution_x / sc.render.resolution_y
    centre, radius = Vector(sub.origin), sub.size
    dist = spec.get("dist") or fit_distance(radius, spec["lens"], aspect, spec["fill"])
    cd = D.cameras.new(shot["id"])
    cd.lens, cd.clip_start, cd.clip_end = spec["lens"], max(0.01, dist * 0.02), max(100.0, dist * 50)
    cam = link(D.objects.new(shot["id"], cd))
    cam.constraints.new("TRACK_TO").target = empty(shot["id"] + "_tgt", None, centre)
    az0, az1 = spec["az"], spec.get("az_to", spec["az"])
    span = max(1, f1 - f0)

    def placer(daz, dh, back):
        def place(f):
            t = (f - f0) / span
            az = az0 + (az1 - az0) * t + daz
            e = math.radians(spec["elev"] + dh * 30)
            flat = _offset(sub.front, az, 1.0)
            return centre + Vector((flat.x * math.cos(e), flat.y * math.cos(e), math.sin(e))) * dist * back * \
                (1 - spec["push"] * t)
        return place

    frames = sorted({f0 + round(span * i / 8) for i in range(9)})
    if "dist" not in spec:
        # the sphere is a safe first guess; now measure what the prop really covers along the move and close in
        # until its widest moment takes `fill` of the frame (a flat card is much smaller than its sphere)
        for _ in range(3):
            extent = 0.0
            for f in frames:
                cam.location = placer(0, 0, 1)(f)
                bpy.context.view_layer.update()
                for p in sub.frame_fn():
                    v = world_to_camera_view(sc, cam, p)
                    extent = max(extent, abs(v.x - 0.5) * 2, abs(v.y - 0.5) * 2)
            if extent > 0:
                dist *= extent / spec["fill"]
    report = {"id": shot["id"], "type": shot["type"], "frames": [f0, f1], "distance_m": round(dist, 3)}
    best = None
    for daz, dh, back in CANDIDATES:
        issues = _check(cam, sub, frames, placer(daz, dh, back), "prop")
        if best is None or len(issues) < len(best[3]):
            best = (daz, dh, back, issues)
        if not issues:
            break
    daz, dh, back, issues = best
    place = placer(daz, dh, back)
    angles = [_facing(sub, place(f)) for f in range(f0, f1 + 1)]
    if angles[0] is not None:  # a screen is only worth showing while it can be read
        readable = sum(a <= READABLE_DEG for a in angles) / len(angles)
        report["screen_readable"] = round(readable, 2)
        report["screen_angle_deg"] = [round(min(angles)), round(max(angles))]
        if readable < 0.5:
            issues = issues + [f"the screen faces the lens for {readable:.0%} of the shot (turned up to "
                               f"{max(angles):.0f} deg): narrow the orbit"]
    adjusted = (daz, dh, back) != (0, 0, 1)
    report.update(passed=not issues, issues=issues, adjusted=adjusted, swing=daz, rise=dh, back=back)
    if adjusted:
        report["nominal_issues"] = _check(cam, sub, frames, placer(0, 0, 1), "prop")
    for f in range(f0, f1 + 1) if (az0 != az1 or spec["push"]) else (f0,):
        key(cam, "location", f, place(f))
    return cam, report


def solve(shot, sub, f0, f1, seed=0, fps=24, avoid=()):
    """Create the shot's camera, choose a placement that passes the checks, key its move. Returns (cam, report)."""
    spec = {**SHOT_TYPES[shot["type"]],
            **{k: shot[k] for k in ("dist", "az", "az_to", "h", "lens", "push", "elev", "hold", "fill") if k in shot}}
    if spec.get("prop"):
        return _solve_prop(shot, spec, sub, f0, f1)
    if shot["type"] == "reveal":  # the far end is the product view of the prop the screen belongs to
        prop = getattr(sub, "prop", None)
        if sub.kind != "screen" or prop is None:
            raise ValueError(f'shot {shot["id"]}: reveal needs a studio prop with a screen as its subject')
        centre, radius = prop.bounds()
        sc = bpy.context.scene
        spec.setdefault("dist", fit_distance(radius, spec["lens"], sc.render.resolution_x / sc.render.resolution_y,
                                             spec["fill"]))
        spec["aim_at"] = centre
        spec["near"] = spec["dist"] * 0.6
    if spec.get("move"):
        return _solve_move(shot, spec, sub, f0, f1, fps, avoid)
    if shot.get("side") == "right":
        spec["az"] = -abs(spec["az"])
    elif shot.get("side") == "left":
        spec["az"] = abs(spec["az"])
    target = sub.origin + Vector(spec["target"])
    cd = D.cameras.new(shot["id"])
    cd.lens, cd.clip_start = spec["lens"], 0.05
    if spec.get("dof"):
        cd.dof.use_dof, cd.dof.focus_object, cd.dof.aperture_fstop = True, sub.focus, spec["dof"]
    cam = link(D.objects.new(shot["id"], cd))
    cam.constraints.new("TRACK_TO").target = empty(shot["id"] + "_tgt", None, target)
    frames = [f0 + 1, (f0 + f1) // 2, f1 - 1]
    push = spec["push"]

    def placer(daz, dh, back=1.0):
        off = _offset(sub.front, spec["az"] + daz, spec["dist"] * back)
        z = spec["h"] + dh

        def place(f):
            t = (f - f0) / max(1, f1 - f0)
            o = off * (1 - push * t)
            return target + Vector((o.x, o.y, z * (1 - push * t)))
        return place

    report = {"id": shot["id"], "type": shot["type"], "frames": [f0, f1]}
    best = None
    for daz, dh, back in CANDIDATES:
        issues = _check(cam, sub, frames, placer(daz, dh, back), spec["check"])
        if best is None or len(issues) < len(best[3]):
            best = (daz, dh, back, issues)
        if not issues:
            break
    daz, dh, back, issues = best
    adjusted = (daz, dh, back) != (0, 0, 1)
    report.update(passed=not issues, issues=issues, adjusted=adjusted, swing=daz, rise=dh, back=back)
    if adjusted:
        report["nominal_issues"] = _check(cam, sub, frames, placer(0, 0, 1), spec["check"])
    place = placer(daz, dh, back)
    key(cam, "location", f0, place(f0))
    key(cam, "location", f1, place(f1))
    shake = shot.get("shake", 0.0)
    if shake:
        rng = random.Random(seed)
        for f in ticks(f0 + n(3), f1, 3):
            key(cam, "location", f, place(f) + Vector([rng.uniform(-shake, shake) for _ in range(3)]))
    return cam, report
