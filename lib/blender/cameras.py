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
import numpy as np
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
    # a character on their feet. `body` aims at that share of their height; `floor` measures the target from the
    # floor and not from the seated chest; `follow` keeps the camera with them as they walk
    "full":          dict(target=(0, 0, 0), body=0.5, dist=4.7, az=-24, h=0.25, lens=28, push=0.04, check="body"),
    "feet":          dict(target=(0, -0.1, 0.14), floor=True, dist=1.9, az=-38, h=0.22, lens=50, push=0.0,
                          check="feet", follow=True),
    # starts on the feet and tilts up to the face: hold (seconds) at each end
    "tilt_up":       dict(target=(0, -0.05, 0.14), floor=True, tilt=True, dist=3.4, az=-14, h=0.75, lens=40, push=0.0,
                          hold=0.5, check="face"),
    # moves from/to the screen filling the frame: az/elev (degrees) and dist (m) are the far end, hold in seconds
    "screen_fill":   dict(move="fill", lens=28, check="screen"),
    "pull_out":      dict(move="out", lens=28, az=-78, elev=20, dist=2.6, hold=0.4, aim=(0.55, 0.3), check="screen"),
    "push_in":       dict(move="in", lens=28, az=-78, elev=20, dist=2.6, hold=0.4, aim=(0.55, 0.3), check="screen"),
    # a camera put where the spec says: "cam" and "aim" are points in the room, "lens" is for the whole frame.
    # "push" moves it that share of the way to "aim" over the shot.
    # "window": [x0, y0, x1, y1] renders only that share of the frame (from the left and from the bottom): set
    # the spec's resolution to the window's shape
    "fixed":         dict(fixed=True, lens=50, check="face"),
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
# someone on their feet can face any way, anywhere in a room: further round, for when a wall is in the way
CANDIDATES_ROUND = CANDIDATES + [(a * sg, h, 1) for h in (0, 0.4) for a in (48, 72, 96, 120, 150) for sg in (1, -1)] \
    + [(180, 0, 1), (60, 0, 0.8), (-60, 0, 0.8), (120, 0, 0.8), (-120, 0, 0.8), (180, 0, 0.8)]
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
    # shot types are placed for the clay cast sitting: someone who sits lower is filmed that much lower
    sub = Subject("character", (0, 0, -person.drop), (0, -1, 0), person.face_points, person.frame_points,
                  person.visible_objects(), person.head)
    sub.person = person
    return sub


CLAY_HEAD = 1.72  # how high the clay cast's head is, sitting: the shots' targets are set from it


def _standing_at(person, f):
    """Where a character whose legs are under control is at a frame: the spot on the floor under them, lifted by
    as much as they are above their chair (cameras are placed for someone sitting), and the way they face."""
    bpy.context.scene.frame_set(f)
    bpy.context.view_layer.update()
    m = person.root.matrix_world
    yaw = m.to_euler().z
    rise = person.hips.location.z - person.seat_h
    return (Vector((m.translation.x, m.translation.y, rise - person.drop)),
            Vector((math.sin(yaw), -math.cos(yaw), 0)))


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


def _solve_fixed(shot, spec, f0, f1):
    """A camera where the spec puts it, looking at "aim". With a "window" it sees only that share of the frame
    the lens would give at `frame` (16:9 unless said): the same lens from the same place, shifted and narrowed,
    so what it renders lies exactly where it would in the whole frame."""
    if not all(isinstance(shot.get(k), (list, tuple)) and len(shot[k]) == 3 for k in ("cam", "aim")):
        raise ValueError(f'shot {shot["id"]}: a fixed shot needs "cam" and "aim", each a point as [x, y, z] in metres')
    sc = bpy.context.scene
    cd = D.cameras.new(shot["id"])
    cd.clip_start, cd.clip_end, cd.sensor_fit = 0.05, 200.0, "HORIZONTAL"
    cam = link(D.objects.new(shot["id"], cd))
    cam.location = shot["cam"]
    aim = empty(shot["id"] + "_tgt", None, shot["aim"])
    cam.constraints.new("TRACK_TO").target = aim
    lens = spec["lens"]
    report = {"id": shot["id"], "type": shot["type"], "frames": [f0, f1], "passed": True, "issues": [],
              "adjusted": False, "swing": 0, "rise": 0, "back": 1.0}
    win = shot.get("window")
    if win:
        x0, y0, x1, y1 = win
        fw, fh = (float(v) for v in str(shot.get("frame", "16:9")).split(":"))
        w, h = x1 - x0, y1 - y0
        want = w * fw / (h * fh)
        got = sc.render.resolution_x / sc.render.resolution_y
        if not (0 < w <= 1 and 0 < h <= 1) or abs(got / want - 1) > 0.01:
            raise ValueError(f'shot {shot["id"]}: the window is {want:.3f} wide for its height and the spec\'s '
                             f'resolution is {got:.3f}: set "resolution" to the window\'s own shape')
        cd.lens = lens / w
        cd.shift_x = ((x0 + x1) / 2 - 0.5) / w
        cd.shift_y = ((y0 + y1) / 2 - 0.5) * (fh / fw) / w
        report["window"] = list(win)
    else:
        cd.lens = lens
    push = float(shot.get("push", 0.0))
    if push:  # a slow move toward what it looks at, by this share of the distance over the shot
        key(cam, "location", f0, Vector(shot["cam"]))
        key(cam, "location", f1, Vector(shot["cam"]).lerp(Vector(shot["aim"]), push))
    return cam, report


def solve(shot, sub, f0, f1, seed=0, fps=24, avoid=()):
    """Create the shot's camera, choose a placement that passes the checks, key its move. Returns (cam, report)."""
    spec = {**SHOT_TYPES[shot["type"]],
            **{k: shot[k] for k in ("dist", "az", "az_to", "h", "lens", "push", "elev", "hold", "fill") if k in shot}}
    if spec.get("prop"):
        return _solve_prop(shot, spec, sub, f0, f1)
    if spec.get("fixed"):
        return _solve_fixed(shot, spec, f0, f1)
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
    person = getattr(sub, "person", None)
    moving = person is not None and bool(person.foot_t)  # legs under control: they may be anywhere, facing any way
    follow = moving and bool(shot.get("follow", spec.get("follow")))
    if spec["check"] in ("body", "feet") or spec.get("floor") or spec.get("body"):
        if not moving:
            raise ValueError(f'shot {shot["id"]}: {shot["type"]} films someone on their feet: give the cast member '
                             f'"posture": "stand", or a stand_up beat')
    front, shift = sub.front, lambda f: Vector((0, 0, 0))  # noqa: E731
    if moving:
        mid = (f0 + f1) // 2
        origin, front = _standing_at(person, mid)
        tx, ty, tz = spec["target"]
        left = Vector((-front.y, front.x, 0))
        target = origin + left * tx - front * ty + Vector((0, 0, tz))
        if spec.get("floor"):
            target.z = tz
        elif person.desc.get("model") and spec["check"] in ("face", "head"):
            # (a fitted body has its own proportions: its head is not where the clay cast's would be on legs as
            # long, so the shot aims as far from its own head as the clay cast's shots aim from the clay head)
            target.z = person.head.matrix_world.translation.z + tz - CLAY_HEAD
        elif spec.get("body"):
            target.z = spec["body"] * (person.head.matrix_world.translation.z + person.head_r)
        if spec["check"] in ("body", "feet"):
            pts = person.feet_points if spec["check"] == "feet" else lambda: person.feet_points() + person.face_points()
            sub = Subject(sub.kind, origin, front, pts, pts, sub.visible, sub.focus)
        if follow:  # the camera and its aim travel with the spot under them (not with the bob of the walk)
            # ... and not with each step either: the spot's path is smoothed over about a second, so the camera
            # glides and the walker moves a little in the frame, as one does in front of an operator
            spots = {f: _standing_at(person, f)[0] for f in range(f0, f1 + 1)}
            for _ in range(2):
                w = max(1, round(fps * 0.3))
                spots = {f: sum((spots[min(f1, max(f0, g))] for g in range(f - w, f + w + 1)), Vector()) / (2 * w + 1)
                         for f in spots}
            shift = lambda f: Vector((spots[f].x - origin.x, spots[f].y - origin.y, 0))  # noqa: E731
    else:
        target = sub.origin + Vector(spec["target"])
    cd = D.cameras.new(shot["id"])
    cd.lens, cd.clip_start = spec["lens"], 0.05
    if spec.get("dof"):
        cd.dof.use_dof, cd.dof.focus_object, cd.dof.aperture_fstop = True, sub.focus, spec["dof"]
    cam = link(D.objects.new(shot["id"], cd))
    aim = empty(shot["id"] + "_tgt", None, target)
    cam.constraints.new("TRACK_TO").target = aim
    frames = [f0 + 1, (f0 + f1) // 2, f1 - 1]
    push = spec["push"]
    if follow:
        for f in range(f0, f1 + 1):
            key(aim, "location", f, target + shift(f))
    elif spec.get("tilt"):  # the aim climbs from the feet to the face; the picture is judged where it ends
        hold = min(round(spec["hold"] * fps), (f1 - f0) // 3)
        bpy.context.scene.frame_set(f1)
        bpy.context.view_layer.update()
        face = person.head.matrix_world.translation.copy()
        key(aim, "location", f0, target)
        key(aim, "location", f0 + hold, target)
        key(aim, "location", f1 - hold, face)
        key(aim, "location", f1, face)
        frames = [f1 - 1]
    if shot.get("reveal"):  # the subject comes into view during the shot (through a door): judged where it ends
        frames = [f1 - 1]

    def placer(daz, dh, back=1.0):
        off = _offset(front, spec["az"] + daz, spec["dist"] * back)
        z = spec["h"] + dh

        def place(f):
            t = (f - f0) / max(1, f1 - f0)
            o = off * (1 - push * t)
            return target + Vector((o.x, o.y, z * (1 - push * t))) + shift(f)
        return place

    report = {"id": shot["id"], "type": shot["type"], "frames": [f0, f1]}
    best = None
    for daz, dh, back in CANDIDATES_ROUND if moving and spec["check"] != "screen" else CANDIDATES:
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
    if follow:
        for f in range(f0 + 1, f1):
            key(cam, "location", f, place(f))
    shake = shot.get("shake", 0.0)
    if shake:
        rng = random.Random(seed)
        for f in ticks(f0 + n(3), f1, 3):
            key(cam, "location", f, place(f) + Vector([rng.uniform(-shake, shake) for _ in range(3)]))
    return cam, report


# ---------------------------------------------------------------- surfaces
def _flat_faces(o, cache={}):
    """An object's flat faces, in its own space: (normal, corners) for each face of a square centimetre or more."""
    if o.data.name not in cache:
        cache[o.data.name] = [(pl.normal.copy(), [o.data.vertices[i].co.copy() for i in pl.vertices])
                              for pl in o.data.polygons if pl.area > 1e-4]
    return cache[o.data.name]


def _inside(p, corners, normal, slack=1e-4):
    k = len(corners)
    return all((corners[(i + 1) % k] - corners[i]).cross(p - corners[i]).dot(normal) > slack for i in range(k))


def fighting(cams, every: int = 4, gap: float = 0.0002) -> list:
    """Pairs of objects with faces in one plane that a camera sees: the renderer draws one or the other from
    frame to frame, so the strip flickers as the camera or the object moves (z-fighting). Faces are grouped by
    the plane they lie in; where two objects' faces face the same way within `gap` of each other and overlap,
    and a shot's lens sees the overlap, the two fight (`gap` is under the 0.4 mm that screen graphics are layered
    by, which render clean). Returns [{"a", "b", "shot", "frame", "points"}]."""
    sc = bpy.context.scene
    solid = [o for o in sc.objects if o.type == "MESH" and not o.hide_render
             and not any(m.type == "ARMATURE" for m in o.modifiers)]
    found = {}
    for f0, f1, cam in cams:
        for f in range(f0, f1 + 1, every):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            origin = cam.matrix_world.translation
            planes = {}
            for o in solid:
                mw = o.matrix_world
                if abs(mw.determinant()) < 1e-9:  # scaled to nothing: a hidden mouth shape
                    continue
                rot = mw.to_3x3().inverted().transposed()
                for nor, corners in _flat_faces(o):
                    wn = (rot @ nor).normalized()
                    wc = [mw @ c for c in corners]
                    d = wn.dot(wc[0])
                    key = (round(wn.x, 2), round(wn.y, 2), round(wn.z, 2))
                    planes.setdefault(key, []).append((d, o, wn, wc))
            for faces in planes.values():
                faces.sort(key=lambda r: r[0])
                for i, (d, o, wn, wc) in enumerate(faces):
                    # (a graphic that pops up is layered by a set share of its size, and stays layered while it
                    # is scaled small: the gap that counts shrinks with the face)
                    near = min(gap, 0.0005 * max((a - b).length for a in wc for b in wc))
                    for d2, o2, wn2, wc2 in faces[i + 1:]:
                        if d2 - d > near:
                            break
                        if o2 is o or wn.dot(wn2) < 0.9995:
                            continue
                        if [m.material for m in o.material_slots] == [m.material for m in o2.material_slots]:
                            continue  # (two bars of one window frame: the same surface either way)
                        mid, mid2 = sum(wc, Vector()) / len(wc), sum(wc2, Vector()) / len(wc2)
                        shared = [p for c, m, other, on in ((wc, mid, wc2, wn2), (wc2, mid2, wc, wn))
                                  for p in [m] + [m + (v - m) * t for v in c for t in (0.5, 0.9)]
                                  if _inside(p, other, on)]
                        seen = 0
                        for p in shared:
                            way = p - origin
                            v = world_to_camera_view(sc, cam, p)
                            if v.z <= 0 or not (0 <= v.x <= 1 and 0 <= v.y <= 1):
                                continue
                            hit, at, _n, _i, _ob, _m = sc.ray_cast(dg, origin, way.normalized(), distance=way.length + 0.01)
                            seen += bool(hit and (at - p).length < 0.002)
                        if seen:
                            pair = tuple(sorted((o.name, o2.name)))
                            rec = found.setdefault(pair, {"a": pair[0], "b": pair[1], "shot": cam.name, "frame": f,
                                                          "points": 0})
                            rec["points"] += seen
    return sorted(found.values(), key=lambda r: -r["points"])


LIMB_NEAR = 0.3  # a moving part of a character this near a still point puts its shadow on it (m)


def still_points(cams, step: float = 0.12, most: int = 8, flat: float = 0.02, pixels: float = 0.002) -> dict:
    """Points on flat surfaces, and where each is in the picture on every frame it is seen and standing still
    (the frame before and after too). A render's pixels at such a point change smoothly or not at all: a jump
    for one frame is flicker. `blender_render.py` reads this beside the clips it has rendered.

    A point counts only where the surface is flat for `flat` metres round it, and on frames where that patch is
    in clear view and at least `pixels` of the picture's width across: the reader averages a few pixels, and an
    edge, a thin leg or something passing in front would look like a jump. A point is also left out on frames
    when part of a character is moving within `LIMB_NEAR` of it: fingers passing over a desk put a shadow on a
    point for one frame and take it off the next, which is the picture changing and not flicker."""
    sc = bpy.context.scene
    pts = []
    for o in sc.objects:
        if o.type != "MESH" or o.hide_render or any(m.type == "ARMATURE" for m in o.modifiers):
            continue
        lo = Vector(tuple(min(c[i] for c in o.bound_box) for i in range(3)))
        hi = Vector(tuple(max(c[i] for c in o.bound_box) for i in range(3)))
        size = hi - lo
        if max(size) < 0.05:
            continue
        for ax in range(3):
            u, v = (i for i in range(3) if i != ax)
            if size[u] < 3 * flat or size[v] < 3 * flat:
                continue
            nu, nv = (max(1, min(most, int(size[k] / step))) for k in (u, v))
            for side in (0, 1):
                for i in range(nu):
                    for j in range(nv):
                        p = Vector((0, 0, 0))
                        p[ax] = hi[ax] if side else lo[ax]
                        p[u], p[v] = lo[u] + size[u] * (i + 0.5) / nu, lo[v] + size[v] * (j + 0.5) / nv
                        round_it = []
                        for k, way in ((u, 1), (u, -1), (v, 1), (v, -1)):
                            q = p.copy()
                            q[k] += way * flat
                            round_it.append(q)
                        if all((o.closest_point_on_mesh(q)[1] - q).length < 5e-4 for q in [p] + round_it):
                            pts.append((o, p, round_it))
    out = {"objects": [o.name for o, _p, _r in pts], "shots": {}}
    rigs = [o for o in sc.objects if o.type == "ARMATURE"]

    def limbs():  # where every bone of every character is, in the room, on this frame
        return np.array([tuple(r.matrix_world @ b.head) for r in rigs for b in r.pose.bones]
                        + [tuple(r.matrix_world @ b.tail) for r in rigs for b in r.pose.bones]).reshape(-1, 3)

    for f0, f1, cam in cams:
        seen, where = {}, {}
        sc.frame_set(max(1, f0 - 1))
        before = limbs()
        for f in range(f0, f1 + 1):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            origin = cam.matrix_world.translation
            now = limbs()
            moving = now[np.linalg.norm(now - before, axis=1) > 0.002] if len(now) == len(before) else now
            before = now

            def clear(w):
                d = w - origin
                hit, at, _n, _i, _ob, _m = sc.ray_cast(dg, origin, d.normalized(), distance=d.length + 0.01)
                return hit and (at - w).length < 0.004

            for k, (o, p, round_it) in enumerate(pts):
                w = o.matrix_world @ p
                x, y, z = world_to_camera_view(sc, cam, w)
                if z <= 0 or not (0.02 < x < 0.98 and 0.02 < y < 0.98):
                    continue
                if len(moving) and float(np.linalg.norm(moving - np.array(w), axis=1).min()) < LIMB_NEAR:
                    continue
                edge = [o.matrix_world @ q for q in round_it]
                views = [world_to_camera_view(sc, cam, q) for q in edge]
                if min(math.hypot(q.x - x, (q.y - y) * 0.5625) for q in views) < pixels:  # too small or too edge-on
                    continue
                if clear(w) and all(clear(q) for q in edge):
                    seen.setdefault(k, {})[f] = (round(x, 5), round(1 - y, 5))
                    where.setdefault(k, {})[f] = w.copy()
        rows = []
        for k, at in seen.items():
            for f, xy in at.items():
                near = [where[k].get(f + s) for s in (-1, 1)]
                if all(n is not None and (n - where[k][f]).length < 1e-5 for n in near):
                    rows.append([f, k, xy[0], xy[1], *at[f - 1], *at[f + 1]])
        out["shots"][cam.name] = {"frames": [f0, f1], "points": sorted(rows)}
    return out
