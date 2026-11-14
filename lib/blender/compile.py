"""Compile a shot spec (JSON) into a Blender scene, check it, and render (runs inside Blender).

    blender -b -P lib/blender/compile.py -- --spec spec.json --out OUT --mode check|render [--res 960x540]

Called by tools/blender_render.py, which is the entry point people and agents use. Writes to OUT:
    checks.json   per-shot camera verdicts (in frame, unblocked, what the solver changed) + IK reach
    events.json   every sound-worthy moment in seconds (key clicks, alert, gulp...) + the spec's score
    scene.blend   the built scene, to open and scrub in Blender
    stills/       one frame per shot (check mode)       silent.mp4  the picture (render mode)
    frames/<shot>/f_0001.png ...   with --clips: a PNG sequence per shot (RGBA with --alpha), encoded on the host

Spec format: see .claude/skills/blender/SKILL.md (worked example: examples/blender-disk-panic/spec.json).
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy  # noqa: E402

import acting  # noqa: E402
import cameras  # noqa: E402
import cast as casting  # noqa: E402
import kit  # noqa: E402
from character import FACES, LEANS, LOOKS, POSES, POSTURES, TYPING, ClayPerson  # noqa: E402

HAND_SHAPES = tuple(ClayPerson.HAND_SHAPES)
from hall import LectureHall  # noqa: E402
from sets import DeskRoom, confetti, title3d  # noqa: E402
from studio import PROP_KINDS, Studio  # noqa: E402
from timeline import SpecError, Timeline  # noqa: E402

SET_KINDS = ("desk_room", "desk", "studio", "lecture_hall")
PROPS = ("cup", "chalk")  # what a lecture hall has to carry and put down
SCORE_MOODS = ("calm", "tension", "resolve")
CHAR_VERBS = {"type", "pose", "expression", "look", "lean", "head_bob", "shake_head", "gulp", "blink", "sweat",
              "tremble", "tap", "celebrate", "hold_arms", "stand_up", "sit_down", "walk", "turn", "wink", "glance",
              "look_at", "reach", "hand", "point", "talk", "blinks", "gesture", "dust_hands", "carry", "put_down",
              "pick_up", "hold", "present", "whistle"}
# how far (m) each kind of contact may be out before the check fails and says where (see ClayPerson.contacts)
CONTACT_LIMITS = {"hand_in_body": 0.03, "through": 0.03, "foot_slide": 0.03, "touch_miss": 0.03, "snap": 0.02}
HANDS_APART = 0.005  # nearer than this the two hands are into each other (m)
PALM_TILT = 35.0     # a hand resting over a table top is palm down to within this, about its fingers (degrees)
PALM_TURN = 35.0     # a hanging hand keeps its palm to the leg through a walk to within this (degrees)
ARM_LIMITS = {"elbow_deg": 35.0, "forearm_deg": 40.0, "hand_off_m": 0.04}  # a fitted model's arms in a walk (arms_report)
TYPING_LIMITS = {"rest_m": 0.008, "into_m": -0.004, "finger_travel_m": 0.006}  # hands with fingers on the keys
GRIP_OPEN = 0.75  # a hand holding something has its fingertips within this share of a finger's length of it
SKIN_LIMITS = {"leg_above_hips": 50, "stretch": 4.0}  # a fitted model's weights (character.skin_report)
SET_VERBS = {"alert", "work", "done", "open_door", "close_door", "pull_in"}
LOOK_AT = ("camera", "screen")  # things a look_at or a reach can name in place of a point
FX_VERBS = {"confetti", "title"}


def _one_of(beat, field, table, numbers=False, default=None):
    v = beat.get(field, default)
    if numbers and isinstance(v, (int, float)) and not isinstance(v, bool):
        return
    if not isinstance(v, str) or v not in table:
        raise SpecError(f'beat at {beat["at"]!r}: {beat["do"]} needs "{field}" to be one of {", ".join(table)}'
                        + (" or a number of degrees" if numbers else "") + f", not {v!r}")


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _point(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(_number(x) for x in v)


def validate(spec):
    """Everything about a spec that can be wrong without Blender: a clear SpecError now, not a traceback later."""
    fps = spec.get("fps", 24)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps != int(fps) or fps < 1:
        raise SpecError(f'"fps" is a whole number ({fps!r} given): use 24 or 30, and render any screen video at it')
    if not re.fullmatch(r"\d+x\d+", str(spec.get("resolution", "1920x1080"))):
        raise SpecError(f'"resolution" is WIDTHxHEIGHT, as in "1920x1080" ({spec["resolution"]!r} given)')
    st = spec.get("set") or {}
    kind = st.get("kind", "desk_room")
    if kind not in SET_KINDS:
        raise SpecError(f"unknown set kind {kind!r} (have: {', '.join(SET_KINDS)})")
    studio = kind == "studio"
    if st.get("door") is not None and (not isinstance(st["door"], dict) or kind not in ("desk_room", "lecture_hall")):
        raise SpecError('"door" is an object (it may be empty: {}), and only a desk_room has a wall to put it in')
    if not _number(st.get("chair_out", 0)):
        raise SpecError('"chair_out" is how far the chair starts pulled back from the desk, in metres')
    spec.setdefault("cast", [])
    if not isinstance(spec.get("shots"), list) or not spec["shots"]:
        raise SpecError('spec needs a "shots" list')
    if not isinstance(spec["cast"], list) or (not spec["cast"] and not studio):
        raise SpecError('spec needs a "cast" list (only a studio set can go without one)')
    if studio and spec["cast"]:
        raise SpecError("a studio set has props and no characters yet: leave out the cast")
    ids = [c.get("id") for c in spec["cast"]]
    for c in spec["cast"]:
        if c.get("posture", "sit") not in POSTURES:
            raise SpecError(f'cast member {c.get("id")!r}: "posture" is one of {", ".join(POSTURES)}, '
                            f'not {c["posture"]!r}')
        if not _point(c.get("at", [0, 0])) or not _number(c.get("facing", 0)):
            raise SpecError(f'cast member {c.get("id")!r}: "at" is a spot on the floor as [x, y] in metres and '
                            f'"facing" a number of degrees (0 faces -Y, positive turns to their left)')
    props = st.get("props") or [] if studio else []
    for p in props:
        if not isinstance(p.get("id"), str) or p.get("kind") not in PROP_KINDS:
            raise SpecError(f'a prop needs an "id" and a "kind" from {", ".join(PROP_KINDS)}: {p}')
        if p["kind"] == "model" and not p.get("file"):
            raise SpecError(f'prop {p["id"]!r}: a model needs "file"')
    if studio and not props:
        raise SpecError('a studio set needs "props"')
    ids += [p["id"] for p in props]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids) or "screen" in ids:
        raise SpecError(f'each cast member and prop needs its own "id" (and not "screen"): {ids}')
    if not ids:
        raise SpecError("nothing to film: the spec has no cast and no props")
    for s in spec["shots"]:
        if s.get("type") not in cameras.SHOT_TYPES:
            raise SpecError(f"unknown shot type {s.get('type')!r} (have: {', '.join(cameras.SHOT_TYPES)})")
        subject = s.get("subject", ids[0])
        on_prop = cameras.SHOT_TYPES[s["type"]].get("prop") or s["type"] == "reveal"
        if on_prop != studio:
            raise SpecError(f'shot {s.get("id")!r}: {s["type"]} is for ' + (
                'a studio set (props)' if on_prop else 'a desk set (a character or its laptop)') + f', not {kind}')
        if subject != "screen" and subject not in ids:
            raise SpecError(f'shot {s.get("id")!r}: unknown subject {subject!r} (have: screen, {", ".join(ids)})')
        if s.get("side") not in (None, "left", "right"):
            raise SpecError(f'shot {s.get("id")!r}: "side" is "left" or "right"')
    for b in spec.get("beats", []):
        if "at" not in b or "do" not in b:
            raise SpecError(f'a beat needs "at" and "do": {b}')
        verb = b["do"]
        if "dur" in b and (isinstance(b["dur"], bool) or not isinstance(b["dur"], (int, float))):
            raise SpecError(f'beat at {b["at"]!r}: "dur" is a number of seconds ({b["dur"]!r} given)')
        if verb not in CHAR_VERBS | SET_VERBS | FX_VERBS:
            raise SpecError(f"unknown beat verb {verb!r} at {b['at']} (have: "
                            f"{', '.join(sorted(CHAR_VERBS | SET_VERBS | FX_VERBS))})")
        if studio:
            raise SpecError(f"beat at {b['at']!r}: a studio set has no verbs yet (the camera does the moving)")
        if verb in ("open_door", "close_door") and b.get("by") is not None and b["by"] not in ids:
            raise SpecError(f'beat at {b["at"]!r}: "by" is who opens the door, and nobody in the cast is called '
                            f'{b["by"]!r} (have: {", ".join(ids)})')
        if verb in CHAR_VERBS and b.get("who", ids[0]) not in ids:
            raise SpecError(f'beat at {b["at"]!r}: nobody in the cast is called {b["who"]!r} (have: {", ".join(ids)})')
        if verb in ("carry", "put_down", "pick_up", "hold", "dust_hands", "talk", "blinks", "gesture"):
            if kind != "lecture_hall" and verb in ("carry", "put_down", "pick_up", "hold"):
                raise SpecError(f'beat at {b["at"]!r}: {verb} needs a set with things to hold (set kind lecture_hall)')
            if verb in ("carry", "put_down", "pick_up", "hold") and b.get("prop") not in PROPS:
                raise SpecError(f'beat at {b["at"]!r}: {verb} needs "prop", one of {", ".join(PROPS)}')
            if b.get("hand", "left") not in ("left", "right"):
                raise SpecError(f'beat at {b["at"]!r}: {verb} takes "hand": "left" or "right" (the character\'s own)')
            if verb == "gesture":
                _one_of(b, "kind", acting.GESTURES, default="open")
        elif verb == "type":
            _one_of(b, "speed", TYPING, default="calm")
        elif verb == "pose":
            _one_of(b, "pose", POSES)
        elif verb == "expression":
            _one_of(b, "face", FACES)
        elif verb == "look":
            _one_of(b, "target", LOOKS, numbers=True, default="screen")
        elif verb == "lean":
            _one_of(b, "amount", LEANS, numbers=True, default="none")
        elif verb == "walk" and (not _point(b.get("to")) or not _number(b.get("facing", 0))
                                 or not _number(b.get("heading", 0)) or not isinstance(b.get("curve", False), bool)
                                 or not (_number(b.get("stride", 0.5)) and 0.3 <= b.get("stride", 0.5) <= 0.85)):
            raise SpecError(f'beat at {b["at"]!r}: walk needs "to", a spot on the floor as [x, y] in metres (and '
                            f'takes "facing", degrees, for the way they end up looking; "curve": true for a bend '
                            f'in place of a line, with "heading", degrees, for the way they are going as they arrive; "stride", 0.3 to '
                            f'0.85 of a leg, for the length of a step)')
        elif verb == "hand" and (b.get("shape") not in HAND_SHAPES or b.get("hand", "left") not in ("left", "right")):
            raise SpecError(f'beat at {b["at"]!r}: hand needs "shape", one of {", ".join(HAND_SHAPES)} (and takes '
                            f'"hand": "left" or "right", the character\'s own; both without it)')
        elif verb == "wink" and b.get("side", "left") not in ("left", "right"):
            raise SpecError(f'beat at {b["at"]!r}: wink takes "side": "left" or "right" (the character\'s own)')
        elif verb in ("reach", "point") and b.get("hand", "right") not in ("left", "right"):
            raise SpecError(f'beat at {b["at"]!r}: {verb} takes "hand": "left" or "right" (the character\'s own)')
        elif verb == "point" and not isinstance(b.get("back", True), bool):
            raise SpecError(f'beat at {b["at"]!r}: point takes "back": true or false (the arm goes back after)')
        elif verb in ("look_at", "reach", "point", "present") and not (b.get("target") in LOOK_AT or (
                isinstance(b.get("target"), list) and len(b["target"]) == 3 and all(_number(x) for x in b["target"]))):
            raise SpecError(f'beat at {b["at"]!r}: {verb} needs "target": a point in the room as [x, y, z] in metres'
                            f', or one of {", ".join(LOOK_AT)}')
        elif verb == "glance" and not (_number(b.get("yaw", 0)) and _number(b.get("pitch", 0))):
            raise SpecError(f'beat at {b["at"]!r}: glance takes "yaw" and "pitch" in degrees')
        elif verb == "turn" and not _number(b.get("facing")):
            raise SpecError(f'beat at {b["at"]!r}: turn needs "facing", a number of degrees '
                            f'(0 faces -Y, positive turns to their left)')
        elif verb == "title" and not isinstance(b.get("text"), str):
            raise SpecError(f'beat at {b["at"]!r}: title needs "text"')


def build_studio(spec, st, tl, base, fps, exposure, sc):
    """A studio set: props, no cast, cameras that frame a prop from its size."""
    try:
        room = Studio(st, base, fps, lambda ref: tl.frame(tl.time(ref)), 2 ** -exposure,
                      backdrop=not sc.render.film_transparent)
    except ValueError as e:
        raise SpecError(str(e))
    # the stage is lit from all round by its own backdrop colour, as a light tent is
    tint = kit.color(room.palette["backdrop"])
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (*[c * 0.55 for c in tint], 1)
    first = next(iter(room.props))
    reports, cams = [], []
    for i, s in enumerate(spec["shots"]):
        name = s.get("subject", first)
        prop = room.first_screen() if name == "screen" else room.props.get(name)
        if prop is None:
            raise SpecError(f'shot {s["id"]!r}: no prop called {name!r} (have: {", ".join(room.props)})')
        if s["type"] == "reveal" and not prop.screen:
            raise SpecError(f'shot {s["id"]!r}: reveal needs a prop with a screen (a phone or a card), not {name!r}')
        sub = cameras.prop_screen_subject(prop) if s["type"] == "reveal" else cameras.prop_subject(prop)
        f0, f1 = tl.frame(tl.shots[s["id"]][0]), tl.frame(tl.shots[s["id"]][1]) - 1
        try:
            cam, rep = cameras.solve(s, sub, f0, f1, seed=i, fps=fps)
        except ValueError as e:
            raise SpecError(str(e))
        sc.timeline_markers.new(s["id"], frame=f0).camera = cam
        cams.append((f0, f1, cam))
        reports.append(rep)
    sc.camera = cams[0][2]
    return tl, cams, reports, 0.0, []


def build(spec, args):
    validate(spec)
    fps = int(spec.get("fps", 24))
    res = tuple(int(x) for x in (args.res or spec.get("resolution", "1920x1080")).split("x"))
    look = spec.get("look", {})
    base = Path(args.spec).resolve().parent
    tl = Timeline(spec["shots"], fps, spec.get("voice"), base)  # before Blender work: a bad spec fails fast
    bpy.ops.wm.read_factory_settings(use_empty=True)
    exposure = look.get("exposure", -0.9)
    kit.setup_render(fps, res, engine=args.engine or spec.get("engine", "eevee"),
                     exposure=exposure, look=look.get("look", "AgX - Punchy"), view=look.get("view", "AgX"),
                     samples=16 if args.mode == "check" else int(spec.get("samples", 32)),
                     transparent=bool(args.alpha or spec.get("transparent")))
    sc = kit.scene()
    sc.frame_start, sc.frame_end = 1, tl.frames - 1

    st = spec.get("set", {"kind": "desk_room"})
    kind = st.get("kind", "desk_room")
    if kind == "studio":
        return build_studio(spec, st, tl, base, fps, exposure, sc)
    scr = st.get("screen") or {}
    media_path = scr.get("video") or scr.get("image")
    if media_path:
        media_path = Path(media_path) if Path(media_path).is_absolute() else base / media_path
        if not media_path.exists():
            raise SpecError(f"screen media not found: {media_path}")
        if scr.get("video"):
            # the picture advances one of its frames per scene frame: at another rate it plays fast or slow
            clip = bpy.data.movieclips.load(str(media_path))
            rate = clip.fps
            bpy.data.movieclips.remove(clip)
            if rate and abs(rate - fps) > 0.01:
                raise SpecError(f'screen video {media_path.name} is {rate:g} fps and the spec is {fps} fps: render '
                                f'it at {fps} fps, or set "fps": {rate:g}')
    cast = {}
    for c in spec.get("cast", []):
        if c.get("kind", "clay_person") != "clay_person":
            raise SpecError(f"unknown cast kind {c.get('kind')!r} (prototype has: clay_person)")
        if cast:
            raise SpecError("prototype supports one character per set")
        cast[c["id"]] = ClayPerson(c["id"], casting.resolve(c, base))
        if any(k in c for k in ("posture", "at", "facing")):
            cast[c["id"]].place(c.get("posture", "sit"), c.get("at", (0, 0)), float(c.get("facing", 0)))
    if not cast:
        raise SpecError("spec needs a cast")
    # the picture's emission cancels the film's exposure, so under the Standard view its colours come out as made
    door = ({**next(iter(cast.values())).door_fit(), **st["door"]} if isinstance(st.get("door"), dict) else None)
    if kind == "lecture_hall":
        picture = (st.get("board") or {}).get("image")
        if picture:
            picture = Path(picture) if Path(picture).is_absolute() else base / picture
            if not picture.exists():
                raise SpecError(f"board image not found: {picture}")
        room = LectureHall(st.get("palette"), walls=bool(st.get("walls", True)), door=door, board=st.get("board"),
                           image=picture, strength=float((st.get("board") or {}).get("brightness", 1.0)) * 2 ** -exposure,
                           table=st.get("table"))
    else:
        room = DeskRoom(st.get("palette"), scr, walls=kind == "desk_room", media_path=media_path,
                        media_start=tl.frame(tl.time(scr.get("at", 0))), media_offset=int(scr.get("skip_frames", 0)),
                        media_strength=float(scr.get("brightness", 1.0)) * 2 ** -exposure,
                        fit=next(iter(cast.values())).furniture(), door=door,
                        chair_out=float(st.get("chair_out", 0.0)))
    bpy.context.view_layer.update()  # the cameras read where the screen is: it has to have been worked out
    for p in cast.values():  # (hands with fingers are put on the keys where they are in the room)
        p.keys = bpy.data.objects.get("lap_keys")
    default_who = next(iter(cast))

    fx_events, at_camera, point_camera = [], [], []
    F = tl.frame
    for b in sorted(spec.get("beats", []), key=lambda b: tl.time(b["at"])):
        verb = b.get("do")
        t0 = tl.time(b["at"])
        f0 = F(t0)
        f1 = F(tl.time(b["until"])) if "until" in b else F(t0 + float(b.get("dur", 0.3)))
        if verb in CHAR_VERBS:
            p = cast[b.get("who", default_who)]
            if verb == "type":
                p.typing(f0, f1, b.get("speed", "calm"))
            elif verb == "pose":
                p.pose(f0, f1, b["pose"])
            elif verb == "expression":
                p.expression(f0, b["face"], max(2, f1 - f0))
            elif verb == "look":
                p.look(f0, f1, b.get("target", "screen"))
            elif verb == "lean":
                p.lean(f0, f1, b.get("amount", "none"))
            elif verb == "head_bob":
                p.head_bob(f0, f1)
            elif verb == "shake_head":
                p.shake_head(f0, f1)
            elif verb == "gulp":
                p.gulp(f0)
            elif verb == "blink":
                p.blink(f0, max(1, f1 - f0) if "dur" in b or "until" in b else kit.n(b.get("frames", 3)))
            elif verb == "sweat":
                p.sweat(f0, f1)
            elif verb == "tremble":
                p.tremble(f0, f1)
            elif verb == "tap":
                p.tap(f0)
            elif verb == "celebrate":
                p.celebrate(f0, f1)
            elif verb == "hold_arms":
                p.hold_arms(f0)
            elif verb == "wink":
                p.wink(f0, b.get("side", "left"))
            elif verb == "hand":
                p.hand_shape(f0, f1, b["shape"], b.get("hand"))
            elif verb == "glance":
                p.glance(f0, f1, float(b.get("yaw", 0)), float(b.get("pitch", 0)))
            elif verb == "talk":  # the narration's words between the beat's start and its end
                t1 = tl.time(b["until"]) if "until" in b else tl.duration
                said = [w for w in tl.words if t0 - 0.01 <= w[1] < t1]
                acting.talk(p, said, F, fps, float(b.get("amount", 1.5)), seed=f0)
            elif verb == "whistle":
                acting.whistle(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(60), seed=f0)
            elif verb == "blinks":
                acting.blinks(p, f0, f1 if "until" in b or "dur" in b else tl.frames - 1, fps, seed=f0)
            elif verb == "present":
                try:
                    point = room.screen_center() if b["target"] == "screen" else b["target"]
                    acting.present(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(36), point, b.get("hand"))
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
            elif verb in ("gesture", "dust_hands", "carry", "put_down", "pick_up", "hold"):
                try:
                    hand = b.get("hand")
                    if verb == "gesture":
                        acting.gesture(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(30),
                                       b.get("kind", "open"), hand)
                    elif verb == "dust_hands":
                        acting.dust_hands(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(36))
                    else:
                        in_hand, on_table, spot = room.prop(b["prop"])
                        small = b["prop"] == "chalk"
                        end = f1 if "until" in b or "dur" in b else f0 + kit.n(18)
                        if verb == "carry":
                            acting.carry(p, in_hand, f0, hand or "left")
                        elif verb == "hold":
                            on_table.scale = (0, 0, 0)
                            acting.hold_small(p, in_hand, f0, hand or "right")
                        elif verb == "pick_up":
                            acting.pick_up(p, in_hand, on_table, f0, end, hand or "right")
                        elif small:
                            acting.leave_small(p, in_hand, on_table, b.get("to", list(spot)), f0, end, hand or "right")
                        else:
                            acting.put_down(p, in_hand, on_table, b.get("to", list(spot)), f0, end, hand or "left")
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
            elif verb == "look_at" and b["target"] == "camera":
                at_camera.append((p, f0, f1))  # (the cameras are placed after the acting)
            elif verb == "point" and b["target"] == "camera":
                point_camera.append((p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(36), b.get("hand"),
                                     b.get("back", True)))
            else:  # the verbs that move the whole body: a second or so unless the beat says
                timed = "until" in b or "dur" in b
                try:
                    if verb == "point":
                        point = room.screen_center() if b["target"] == "screen" else b["target"]
                        p.point(f0, f1 if timed else f0 + kit.n(36), point, b.get("hand"), b.get("back", True))
                    elif verb in ("look_at", "reach"):
                        point = room.screen_center() if b["target"] == "screen" else b["target"]
                        if verb == "look_at":
                            p.look_at(f0, f1, point)
                        else:
                            p.reach(f0, f1, point, b.get("hand", "right"))
                    elif verb == "walk":  # (untimed, a walk takes what its steps need)
                        p.walk(f0, f1 if timed else None, b["to"], b.get("facing"), bool(b.get("curve")), b.get("heading"),
                               float(b.get("stride", 0.5)))
                    elif verb == "turn":
                        p.turn(f0, f1 if timed else f0 + kit.n(20), b["facing"])
                    elif verb == "stand_up":
                        p.stand_up(f0, f1 if timed else f0 + kit.n(40))
                    elif verb == "sit_down":
                        p.sit_down(f0, f1 if timed else f0 + kit.n(44))
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
        elif verb in SET_VERBS:
            if verb == "alert":
                room.alert(f0)
            elif verb == "work":
                room.work(f0, f1)
            elif verb == "done":
                room.done(f0)
            else:
                timed = "until" in b or "dur" in b
                try:
                    if verb == "pull_in":  # whoever sits on the chair comes in to the desk with it
                        end = f1 if timed else f0 + kit.n(34)
                        room.events.append({"frame": f0, "kind": "chair"})
                        cast[b.get("who", default_who)].scoot(f0, end, (0, 0), rides=[room.chair], grip=room.desk_edge)
                    else:
                        end = f1 if timed else f0 + kit.n(18)
                        room.open_door(f0, end, float(b.get("angle", 100)) if verb == "open_door" else 0.0)
                        if b.get("by"):  # someone's hand is on the handle as it goes
                            who = cast[b["by"]]
                            bpy.context.scene.frame_set(f0)
                            bpy.context.view_layer.update()
                            handle = room.handle_for(who._world((0, 0, 1)))  # the one on their side of the door
                            who.push(f0, end, lambda f, h=handle: room.bar_of(h), b.get("hand"))
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
        elif verb == "confetti":
            confetti(f0, int(b.get("count", 60)), colors=b.get("colors"), desk=(room.desk_y, room.desk_z))
            fx_events.append({"frame": f0, "kind": "confetti"})
        elif verb == "title":
            title3d(f0, b["text"], tl.frames - 1, c=b.get("color", "#FFD24A"))
            fx_events.append({"frame": f0, "kind": "title"})

    for p in cast.values():
        # ("clock": where this spec starts on a longer film's timeline, in seconds: clips made from several
        # specs then breathe in step, and join without a jump)
        period = 2 * kit.n(3.4 * 12)
        p.breathe(1 - int(round(float(spec.get("clock", 0.0)) * fps)) % period, tl.frames)

    # cameras last, so the solver checks them against the finished acting
    reports, cams = [], []
    people = [o for c in cast.values() for o in c.visible_objects()]
    for i, s in enumerate(spec["shots"]):
        subj = s.get("subject", default_who)
        sub = cameras.screen_subject(room) if subj == "screen" else cameras.character_subject(cast[subj])
        f0, f1 = F(tl.shots[s["id"]][0]), F(tl.shots[s["id"]][1]) - 1
        try:
            cam, rep = cameras.solve(s, sub, f0, f1, seed=i, fps=fps, avoid=people)
        except ValueError as e:
            raise SpecError(str(e))
        sc.timeline_markers.new(s["id"], frame=f0).camera = cam
        cams.append((f0, f1, cam))
        reports.append(rep)
    sc.camera = cams[0][2]
    for person, a, b in at_camera:  # a look into the lens of whichever shot it happens in
        cam = next((c for c0, c1, c in cams if c0 <= b <= c1), cams[-1][2])
        sc.frame_set(b)
        bpy.context.view_layer.update()
        person.look_at(a, b, cam.matrix_world.translation)
    for person, a, b, hand, back in point_camera:  # (at the lens of the shot it happens in, the viewer)
        cam = next((c for c0, c1, c in cams if c0 <= b <= c1), cams[-1][2])
        sc.frame_set(a)
        bpy.context.view_layer.update()
        person.point(a, b, cam.matrix_world.translation.copy(), hand, back)

    p = cast[default_who]
    ik = p.ik_error(range(1, tl.frames, 6 if not p.foot_t else 2))
    kit.ik_worst = p.ik_worst
    # someone on their feet is checked against the floor too (the clay cast at its desk has its feet where they
    # were made, and is left alone)
    kit.contacts = p.contacts(range(1, tl.frames), room.solids()) if p.foot_t else None
    kit.under_floor = (round(kit.contacts["under_floor"]["m"], 3), p.floor_worst) if p.foot_t else None
    kit.contact_trace = getattr(p, "contact_trace", [])
    kit.moves = [{"do": verb, "t": round((a - 1) / fps, 2), **p.how_it_moves(a, b)} for verb, a, b in p.moves]
    events = [{"t": round((e["frame"] - 1) / fps, 3), "shot": tl.shot_at((e["frame"] - 1) / fps), **e}
              for e in sorted(p.events + room.events + fx_events, key=lambda e: e["frame"])]
    kit.grips = p.grips
    kit.walks = [{"t": round((a - 1) / fps, 2), **p.walk_report(a, b, curved)} for a, b, curved in p.walks]
    kit.typing = p.typing_report(p.keys)
    kit.face = p.face_report()
    kit.skin = p.skin_report(range(1, tl.frames, max(1, tl.frames // 14)))
    kit.fighting = cameras.fighting(cams)  # faces in one plane that a camera sees: they would flicker
    return tl, cams, reports, ik, events


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["check", "render"], default="check")
    ap.add_argument("--res")
    ap.add_argument("--engine", choices=["eevee", "cycles"])
    ap.add_argument("--alpha", action="store_true", help="transparent background")
    ap.add_argument("--clips", action="store_true", help="render mode: a PNG sequence per shot instead of one film")
    ap.add_argument("--shots", help="comma-separated shot ids to check or render (default: all)")
    ap.add_argument("--bench", type=int, default=0, help="render this many frames at full quality and time them")
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        tl, cams, reports, ik, events = build(spec, args)
        score = []
        for s in spec.get("score", []):
            if s.get("mood") not in SCORE_MOODS or "from" not in s or "to" not in s:
                raise SpecError(f'a score section needs "from", "to" and a "mood" from {", ".join(SCORE_MOODS)}: {s}')
            score.append({**s, "from": tl.time(s["from"]), "to": tl.time(s["to"])})
    except SpecError as e:
        (out / "checks.json").write_text(json.dumps({"passed": False, "spec_error": str(e)}, indent=2))
        print("SPEC ERROR:", e)
        sys.exit(2)

    sc = kit.scene()
    alpha = bool(sc.render.film_transparent)
    wanted = [x.strip() for x in args.shots.split(",")] if args.shots else list(tl.shots)
    missing = [x for x in wanted if x not in tl.shots]
    if missing:
        (out / "checks.json").write_text(json.dumps({"passed": False, "spec_error": f"unknown shot {missing[0]!r}"}))
        print("SPEC ERROR: unknown shot", missing[0])
        sys.exit(2)
    for r in reports:
        r["start"], r["dur"] = round(tl.shots[r["id"]][0], 4), round(tl.shots[r["id"]][1] - tl.shots[r["id"]][0], 4)
    floor = getattr(kit, "under_floor", None)
    sunk, sunk_at = floor or (0.0, None)
    contacts = {k: c for k, c in (getattr(kit, "contacts", None) or {}).items() if k != "under_floor"}
    apart, tilt = contacts.pop("hands_apart", None), contacts.pop("palm_tilt", None)
    turned = contacts.pop("palm_turn", None)
    touching = all(c["m"] <= CONTACT_LIMITS[k] for k, c in contacts.items())
    checks = {"passed": all(r["passed"] for r in reports) and ik < 0.01 and sunk <= 0.015 and touching,
              "duration": tl.duration, "fps": tl.fps,
              "resolution": [sc.render.resolution_x, sc.render.resolution_y], "alpha": alpha,
              "engine": sc.render.engine, "blender": bpy.app.version_string,
              "frames": tl.frames - 1, "ik_max_error_m": round(ik, 4), "shots": reports}
    worst = getattr(kit, "ik_worst", None)
    if floor is not None:  # someone on their feet: how deep a foot went through the floor, and when
        checks["under_floor_m"] = sunk
        if sunk > 0.015:
            checks["under_floor_at"] = {"frame": sunk_at, "t": round((sunk_at - 1) / tl.fps, 2)}
    said = {"hand_in_body": "{hand} hand {cm} cm inside the body", "through": "body {cm} cm into the {what}",
            "foot_slide": "{foot} foot slides {cm} cm along the floor while planted",
            "touch_miss": "{hand} hand {cm} cm off the {what}",
            "snap": "the {what} jumps {cm} cm in one frame (it hardly moves in the frames either side)"}
    problems = [f"a foot {sunk * 100:.0f} cm under the floor at {(sunk_at - 1) / tl.fps:.2f} s"] if sunk > 0.015 else []
    lasted = lambda c: (f' (from {(c["first"] - 1) / tl.fps:.2f} s to {(c["last"] - 1) / tl.fps:.2f} s)'  # noqa: E731
                        if c.get("last", 0) > c.get("first", 0) else "")
    problems += [said[k].format(cm=round(c["m"] * 100), **c) + f' at {(c["frame"] - 1) / tl.fps:.2f} s' + lasted(c)
                 for k, c in contacts.items() if c["m"] > CONTACT_LIMITS[k]]
    if apart and apart["frame"] is not None:  # hands with fingers: do they go into each other, do they lie palm down
        checks["hands_apart_m"] = round(apart["m"], 3)
        if apart["m"] < HANDS_APART:
            checks["passed"] = False
            problems.append(f'the hands go into each other at {(apart["frame"] - 1) / tl.fps:.2f} s')
    if tilt and tilt["frame"] is not None:
        checks["palm_tilt_deg"] = round(tilt["deg"], 1)
        if tilt["deg"] > PALM_TILT:
            checks["passed"] = False
            problems.append(f'{tilt["hand"]} hand lies on the {tilt["what"]} rolled {tilt["deg"]:.0f} degrees from palm '
                            f'down onto its side at {(tilt["frame"] - 1) / tl.fps:.2f} s')
    if turned and turned["frame"] is not None:
        checks["palm_turn_deg"] = round(turned["deg"], 1)
        if turned["deg"] > PALM_TURN:
            checks["passed"] = False
            problems.append(f'{turned["hand"]} palm turns {turned["deg"]:.0f} degrees away from the leg as the arm swings '
                            f'in a walk at {(turned["frame"] - 1) / tl.fps:.2f} s')
    if getattr(kit, "moves", None):  # each posture verb: do its parts start and stop together, are the sides the same
        checks["moves"] = kit.moves
    if problems:  # in words, for the person or agent reading the check
        checks["body_problems"] = problems
    face = getattr(kit, "face", None)
    if face:  # a fitted model's face: is its mouth big enough to read, and on the face
        checks["face"] = {k: v for k, v in face.items() if k != "problems"}
        if face.get("problems"):
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + face["problems"]
    grips = getattr(kit, "grips", None)
    if grips:  # a hand with fingers that took hold of something: did it close on it
        worst = max(grips, key=lambda g: g["open"])
        checks["grip_open"] = worst["open"]
        if worst["open"] > GRIP_OPEN:
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + [
                f'{worst["hand"]} hand is open on the {worst["what"]} at {(worst["frame"] - 1) / tl.fps:.2f} s (fingertips '
                f'{worst["open"]:.2f} of a finger\'s length from it): it should close on what it holds']
    walks = getattr(kit, "walks", None)
    if walks:  # each walk's path: how far it bows, how far it turns, how the body leans as it does
        checks["walks"] = walks
        for w in walks:
            if w["curve"] and abs(w["turn_deg"]) > 20 and (w["into_bend"] < 0.5 or w["lean_deg"] < 1.0):
                checks["passed"] = False
                checks["body_problems"] = checks.get("body_problems", []) + [
                    f'the walk at {w["t"]} s turns {abs(w["turn_deg"]):.0f} degrees and the body does not lean into '
                    f'the bend (lean {w["lean_deg"]} degrees, {w["into_bend"]:+.2f} of it inward)']
            arms = w.get("arms")
            if arms:  # a fitted model's arms: straight enough, low enough in front, near enough the body
                said_arms = []
                if arms["elbow_deg"] > ARM_LIMITS["elbow_deg"]:
                    said_arms.append(f'the elbows bend {arms["elbow_deg"]:.0f} degrees in the walk at {w["t"]} s '
                                     f'(frame {arms["frames"]["elbow"]}): arms that swing from the shoulder stay nearly straight')
                if arms["forearm_deg"] < ARM_LIMITS["forearm_deg"]:
                    said_arms.append(f'a forearm comes up to {arms["forearm_deg"]:.0f} degrees below level in the walk at '
                                     f'{w["t"]} s (frame {arms["frames"]["fore"]}): it reads as marching')
                if arms["hand_off_m"] > ARM_LIMITS["hand_off_m"]:
                    said_arms.append(f'the hands stand {arms["hand_off_m"] * 100:.0f} cm off the body through the walk at {w["t"]} s '
                                     f'(most at frame {arms["frames"]["off"]}): arms held out from the sides')
                if said_arms:
                    checks["passed"] = False
                    checks["body_problems"] = checks.get("body_problems", []) + said_arms
    typing = getattr(kit, "typing", None)
    if typing:  # hands with fingers typing: do they rest on the keys, and do the fingers move
        checks["typing"] = typing
        said_typing = []
        if typing["rest_m"] > TYPING_LIMITS["rest_m"]:
            said_typing.append(f'the hands type {typing["rest_m"] * 100:.1f} cm over the keys: they should rest on them')
        if typing["into_m"] < TYPING_LIMITS["into_m"]:
            said_typing.append(f'a fingertip goes {-typing["into_m"] * 100:.1f} cm into the keys')
        if typing["finger_travel_m"] < TYPING_LIMITS["finger_travel_m"]:
            said_typing.append(f'a finger moves only {typing["finger_travel_m"] * 1000:.0f} mm while typing: the fingers '
                               f'should press the keys, not the hands')
        if said_typing:
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + said_typing
    skin = getattr(kit, "skin", None)
    if skin:  # a fitted model's weights: does a leg pull on the top's hem, is anything stretched apart
        checks["skin"] = skin
        said_skin = []
        if skin["leg_above_hips"] > SKIN_LIMITS["leg_above_hips"]:
            said_skin.append(f'{skin["leg_above_hips"]} vertices above the hips move with a leg, so a hem or waistband '
                             f'tears into spikes when the thighs bend: fit the model again (blender_render.py fit)')
        if skin["stretch"] > SKIN_LIMITS["stretch"]:
            said_skin.append(f'the skin stretches {skin["stretch"]:.1f}x at {(skin["frame"] - 1) / tl.fps:.2f} s: '
                             f'the fit\'s weights pull vertices apart')
        if said_skin:
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + said_skin
    fights = getattr(kit, "fighting", None) or []
    if fights:  # two faces in one plane, seen by a camera: the strip flickers between them as things move
        checks["passed"] = False
        checks["surfaces_fight"] = fights
        checks["set_problems"] = [f'{r["a"]} and {r["b"]} have faces in one plane and will flicker in {r["shot"]} '
                                  f'(frame {r["frame"]}): move one, or end it inside the other' for r in fights]
    for kind, c in contacts.items():  # a hand inside the body or off what it holds, the body through furniture,
        checks[kind + "_m"] = round(c["m"], 3)  # a planted foot sliding: how far, and where if it is too far
        if c["m"] > CONTACT_LIMITS[kind]:
            checks[kind + "_at"] = {k: v for k, v in c.items() if k != "m"} | {"t": round((c["frame"] - 1) / tl.fps, 2)}
    if worst and ik >= 0.01:  # which limb could not reach, and when
        checks["ik_worst"] = {**worst, "t": round((worst["frame"] - 1) / tl.fps, 2)}
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    (out / "events.json").write_text(json.dumps({"fps": tl.fps, "duration": tl.duration, "events": events,
                                                 "score": score}, indent=2))
    if os.environ.get("BLENDER_CONTACT_TRACE"):  # every frame's contact numbers, to look into a failure
        (out / "contacts.json").write_text(json.dumps(getattr(kit, "contact_trace", [])))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if args.mode == "render" and not args.bench:  # where still surfaces are in each frame: read against the render
        wanted_cams = [c for c in cams if not args.shots or c[2].name in [x.strip() for x in args.shots.split(",")]]
        (out / "still_points.json").write_text(json.dumps(cameras.still_points(wanted_cams)))

    if args.bench:
        import time
        (out / "bench").mkdir(exist_ok=True)
        spans = [(f0, f1, cam) for f0, f1, cam in cams if cam.name in wanted]
        times = []
        for i in range(args.bench + 1):  # the first frame also compiles shaders, so it is not counted
            f0, f1, cam = spans[i % len(spans)]
            sc.frame_set((f0 + f1) // 2 + i)
            sc.camera = cam
            kit.set_output(out / "bench" / f"b{i}.png", video=False, alpha=alpha)
            t = time.time()
            bpy.ops.render.render(write_still=True)
            times.append(time.time() - t)
        print("BENCH", json.dumps({"first_frame_s": round(times[0], 2),
                                   "seconds_per_frame": round(sum(times[1:]) / len(times[1:]), 3)}))
    elif args.mode == "check":
        (out / "stills").mkdir(exist_ok=True)
        for f0, f1, cam in cams:
            if cam.name not in wanted:
                continue
            # a move is judged at its far end (where it shows the room); everything else mid-shot
            move = cameras.SHOT_TYPES[next(s["type"] for s in spec["shots"] if s["id"] == cam.name)].get("move")
            tilt = cameras.SHOT_TYPES[next(s["type"] for s in spec["shots"] if s["id"] == cam.name)].get("tilt")
            f = f1 if move == "out" or tilt else f0 if move == "in" else (f0 + f1) // 2
            sc.frame_set(f)
            sc.camera = cam
            kit.set_output(out / "stills" / f"{cam.name}.png", video=False, alpha=alpha)
            bpy.ops.render.render(write_still=True)
    elif args.clips:
        for f0, f1, cam in cams:
            if cam.name not in wanted:
                continue
            (out / "frames" / cam.name).mkdir(parents=True, exist_ok=True)
            sc.frame_start, sc.frame_end = f0, f1
            kit.set_output(out / "frames" / cam.name / "f_", video=False, alpha=alpha)
            bpy.ops.render.render(animation=True)
    else:
        kit.set_output(out / "silent.mp4", video=True)
        bpy.ops.render.render(animation=True)
    print("COMPILE_DONE", json.dumps({"passed": checks["passed"], "out": str(out)}))


if __name__ == "__main__":
    main()
