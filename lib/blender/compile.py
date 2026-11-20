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
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import acting  # noqa: E402
import cameras  # noqa: E402
import cast as casting  # noqa: E402
import kit  # noqa: E402
import logo  # noqa: E402
import motion  # noqa: E402
import svg  # noqa: E402
from character import FACES, LEANS, LOOKS, POSES, POSTURES, TYPING, ClayPerson  # noqa: E402

HAND_SHAPES = tuple(ClayPerson.HAND_SHAPES)
from hall import STATURE as HALL_STATURE, LectureHall, board_floor, board_rect, door_floor  # noqa: E402
from kitchen import Kitchen  # noqa: E402
from lounge import LivingRoom  # noqa: E402
from pub import Pub  # noqa: E402
from sets import DeskRoom, confetti, title3d  # noqa: E402
import studio as studio_set  # noqa: E402
from studio import PROP_KINDS, STAGES, Studio  # noqa: E402
from timeline import SpecError, Timeline  # noqa: E402

SET_KINDS = ("desk_room", "desk", "studio", "lecture_hall", "kitchen", "living_room", "pub")
PROPS = ("cup", "chalk")  # what a lecture hall has to carry and put down
SCORE_MOODS = ("calm", "tension", "resolve")
CHAR_VERBS = {"type", "pose", "expression", "look", "lean", "head_bob", "shake_head", "gulp", "blink", "sweat",
              "tremble", "tap", "celebrate", "hold_arms", "settle", "stand_up", "sit_down", "walk", "turn", "wink", "glance",
              "look_at", "reach", "hand", "point", "talk", "blinks", "gesture", "dust_hands", "carry", "put_down",
              "pick_up", "hold", "present", "whistle", "sip", "phone", "wag", "lap", "let_go"}
# how far (m) each kind of contact may be out before the check fails and says where (see ClayPerson.contacts)
CONTACT_LIMITS = {"hand_in_body": 0.03, "through": 0.03, "foot_slide": 0.03, "touch_miss": 0.03, "snap": 0.02}
HANDS_APART = 0.005  # nearer than this the two hands are into each other (m)
PALM_TILT = 35.0     # a hand resting over a table top is palm down to within this, about its fingers (degrees)
PALM_TURN = 35.0     # a hanging hand keeps its palm to the leg through a walk to within this (degrees)
ARM_LIMITS = {"elbow_deg": 35.0, "forearm_deg": 40.0, "hand_off_m": 0.04}  # a fitted model's arms in a walk (arms_report)
TYPING_LIMITS = {"rest_m": 0.008, "into_m": -0.004, "finger_travel_m": 0.006}  # hands with fingers on the keys
CARRIED_TILT = 12.0     # a mug in the hand is level to within this (degrees; the tip of a sip is its own part)
CARRIED_THROUGH = 0.006  # and no deeper than this in a table or a door (m)
GRIP_OPEN = 0.75  # a hand holding something has its fingertips within this share of a finger's length of it
SKIN_LIMITS = {"leg_above_hips": 50, "stretch": 4.0}  # a fitted model's weights (character.skin_report)
BOARD_COVER = 0.10  # the most of a lecture hall's board its presenter may hide from the board camera
SET_VERBS = {"alert", "work", "done", "open_door", "close_door", "pull_in", "open_lid", "close_lid", "claude_type",
             "claude_turn", "lights"}
KITCHEN_VERBS = ("open_lid", "close_lid", "claude_type", "claude_turn")  # verbs only a kitchen has
LOOK_AT = ("camera", "screen", "phone", "pint")  # things a look_at or a reach can name in place of a point
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


def _board_ref(v):
    """A place named by the lecture hall's board: {"board": [u, v]} in board pixels (a spot on the floor takes
    [u, metres out from the wall])."""
    return isinstance(v, dict) and set(v) == {"board"} and _point(v["board"])


def tl_time_ok(b, spec):
    """The most seconds a latch can come before its door swings: back to the spec's start (a beat timed by a cue
    is checked when it is placed)."""
    return float(b["at"]) if _number(b.get("at")) else 99.0


def _door_ref(v):
    """A spot named by a lecture hall's door: {"door": out}, on the door's line, `out` metres (the hall's measure)
    into the room from it, or behind it (in the corridor) when negative."""
    return isinstance(v, dict) and set(v) == {"door"} and _number(v["door"])


def _switch_ref(v):
    """A spot named by a dark lecture hall's light switch: {"switch": out}, on the floor in front of the plate,
    `out` metres (the hall's measure) into the room from its wall, or {"switch": "reach"}: as far out as the
    walker, side-on to the wall, presses it with a straight arm (their own arm's length, so any body)."""
    return isinstance(v, dict) and set(v) == {"switch"} and ((_number(v["switch"]) and v["switch"] > 0)
                                                              or v["switch"] == "reach")


def _hall_ref(v):
    """A point named in a lecture hall's own measure: {"hall": [x, out, z]}, x from the board's middle, out from
    its wall, z up from the floor."""
    return isinstance(v, dict) and set(v) == {"hall"} and isinstance(v["hall"], (list, tuple)) and len(v["hall"]) == 3 \
        and all(_number(x) for x in v["hall"])


def _framing(v, end=False):
    """One end of a move: {"cam", "aim", "lens"}, points in metres or {"hall": [...]}; the far end may be
    {"cam": "board"} (and a lens)."""
    if not isinstance(v, dict):
        return False
    if end and v.get("cam") == "board":
        return set(v) <= {"cam", "lens"} and _number(v.get("lens", 108))
    pt = lambda x: (isinstance(x, list) and len(x) == 3 and all(_number(c) for c in x)) or _hall_ref(x)  # noqa: E731
    return set(v) == {"cam", "aim", "lens"} and pt(v["cam"]) and pt(v["aim"]) and _number(v["lens"]) and v["lens"] > 0


def _board_cover(room, p, spec, tl, cams):
    """How much of the board the presenter hides from each "board" camera, every half second: the share of the
    board's pixels behind them (their mesh cast onto the board from the lens, on a grid of 10 board pixels), the
    box they ever reach in board pixels, where their head is at the start, and whether they stay inside the
    shot's window (a clip of part of the frame cuts off a hand that reaches out of it)."""
    shots = [s for s in spec.get("shots", []) if s.get("board_cam") and not s.get("built_only")]
    if not shots:
        return None
    from hall import WALL_Y
    sc = kit.scene()
    b = room.board
    gw, gh = 192, 108
    objs = [o for o in p.visible_objects() if o.type == "MESH"]
    out = {"share": 0.0, "frame": None, "box_px": None, "head_px": None, "outside_window": 0}
    for s in shots:
        cam = next(c for f0, f1, c in cams if c.name == s["id"])
        f0, f1 = tl.frame(tl.shots[s["id"]][0]), tl.frame(tl.shots[s["id"]][1]) - 1
        win = s.get("window")
        for f in sorted(set(range(f0, f1 + 1, max(1, round(tl.fps / 2)))) | {f1}):
            sc.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            pts = []
            for o in objs:
                ev = o.evaluated_get(dg)
                mesh = ev.to_mesh()
                co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
                mesh.vertices.foreach_get("co", co)
                ev.to_mesh_clear()
                mw = np.array(o.matrix_world)
                co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
                pts.append(co)
            P = np.concatenate(pts)
            c = np.array(cam.matrix_world.translation)
            t = (WALL_Y - 0.03 - c[1]) / (P[:, 1] - c[1])
            on = c + (P - c) * t[:, None]
            u = (on[:, 0] - b["x0"]) / (b["x1"] - b["x0"]) * 1920
            v = (b["z1"] - on[:, 2]) / (b["z1"] - b["z0"]) * 1080
            if f == f0 and out["head_px"] is None:
                hu, hv = room.board_px(c + (np.array(p.head.matrix_world.translation) - c)
                                       * ((WALL_Y - 0.03 - c[1]) / (p.head.matrix_world.translation.y - c[1])))
                out["head_px"] = [round(hu), round(hv)]
            box = [float(u.min()), float(v.min()), float(u.max()), float(v.max())]
            out["box_px"] = box if out["box_px"] is None else [min(out["box_px"][0], box[0]), min(out["box_px"][1], box[1]),
                                                               max(out["box_px"][2], box[2]), max(out["box_px"][3], box[3])]
            if win:  # (the window is shares of the frame from the left and the bottom; what is off the frame
                # is cut off by the frame anyway, legs below its bottom edge included)
                wx0, wy0, wx1, wy1 = win
                seen = [max(0.0, box[0]), max(0.0, box[1]), min(1920.0, box[2]), min(1080.0, box[3])]
                if (seen[0] < wx0 * 1920 - 1 or seen[2] > wx1 * 1920 + 1 or seen[1] < (1 - wy1) * 1080 - 1
                        or seen[3] > (1 - wy0) * 1080 + 1):
                    out["outside_window"] += 1
            inside = (u >= 0) & (u < 1920) & (v >= 0) & (v < 1080)
            grid = np.zeros((gh, gw), bool)
            grid[(v[inside] / 1080 * gh).astype(int), (u[inside] / 1920 * gw).astype(int)] = True

            def grow(g):
                h = g.copy()
                h[1:] |= g[:-1]
                h[:-1] |= g[1:]
                h[:, 1:] |= g[:, :-1]
                h[:, :-1] |= g[:, 1:]
                return h
            grid = ~grow(~grow(grid))  # (closed: the gaps between projected vertices filled, the outline kept)
            share = float(grid.mean())
            if share > out["share"]:
                out["share"], out["frame"] = round(share, 4), f
    out["box_px"] = [round(x) for x in out["box_px"]]
    return out


def _resolve_board(spec, room, cast):
    """Turn the places a lecture spec names by the board into the hall's metres, now that the hall is sized: a
    cast member's spot, a walk's end, a beat's target, and a "board" camera (square on, edge to edge with the
    shot's lens, 108 mm unless it says)."""
    for b in spec.get("beats", []):
        if _board_ref(b.get("target")):
            b["target"] = room.board_point(*b["target"]["board"])
        if b.get("do") == "walk" and _board_ref(b.get("to")):
            b["to"] = room.board_floor(*b["to"]["board"])
        if b.get("do") == "walk" and _door_ref(b.get("to")):
            b["to"] = room.door_floor(b["to"]["door"])
        if b.get("do") == "walk" and _switch_ref(b.get("to")):
            out, along = b["to"]["switch"], 0.0
            if out == "reach":  # (with the hand their lights beat presses it with, a step toward the door from it)
                who = b.get("who", spec["cast"][0]["id"])
                hand = next((x.get("hand", "right") for x in spec.get("beats", [])
                             if x.get("do") == "lights" and x.get("by") == who), "right")
                along = room.SWITCH_ALONG
                out = cast[who].press_reach(hand, room.switch_point(), along * room.scale) / room.scale
            b["to"] = room.switch_floor(out, along)
    for s in spec.get("shots", []):
        if s.get("type") == "move":  # (a move's two framings, in the hall's own measure or on the board)
            for end in ("from", "to"):
                fr = dict(s[end])
                if fr.get("cam") == "board":
                    fr["lens"] = float(fr.get("lens", 108))
                    fr["cam"], fr["aim"] = room.board_camera(fr["lens"])
                    fr["board"] = True
                for k in ("cam", "aim"):
                    if _hall_ref(fr.get(k)):
                        fr[k] = room.hall_point(*fr[k]["hall"])
                s[end] = fr
        if s.get("cam") == "board":
            s["lens"] = float(s.get("lens", 108))
            s["cam"], s["aim"] = room.board_camera(s["lens"])
            s["board_cam"] = True
            # (a long lens stands the camera well back, behind the room's front wall: the wall still lights the
            # room and is not drawn for the camera)
            front = bpy.data.objects.get("wall_front")
            if front is not None:
                front.visible_camera = False


def validate(spec, base=None):
    """Everything about a spec that can be wrong without Blender: a clear SpecError now, not a traceback later."""
    base = Path(base or ".")
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
    if st.get("door") is not None and (not isinstance(st["door"], dict) or kind not in ("desk_room", "lecture_hall", "kitchen")):
        raise SpecError('"door" is an object (it may be empty: {}), and only a desk_room has a wall to put it in')
    if st.get("hall") is not None and (kind != "lecture_hall" or not isinstance(st["hall"], dict)):
        raise SpecError('"hall" dresses a lecture_hall (rows, windows, lights): an object, which may be empty: {}')
    if isinstance(st.get("hall"), dict):
        from hall import HALL
        unknown = set(st["hall"]) - set(HALL)
        if unknown:
            raise SpecError(f'"hall" takes {", ".join(HALL)} (not {", ".join(sorted(unknown))})')
    # (a studio stage's "lights" is its own: shares of its key, fill and rim, checked with the stage below)
    if kind == "studio":
        pass
    elif st.get("lights", "on") not in ("on", "off") or (st.get("lights", "on") == "off" and kind != "lecture_hall"):
        raise SpecError('"lights" is "on" (the default) or "off": a lecture_hall that starts dark, lit by its windows '
                        'until a "lights" beat switches it on')
    if kind != "studio" and any(b.get("do") == "lights" for b in spec.get("beats", [])) and st.get("lights") != "off":
        raise SpecError('a "lights" beat switches on a hall that starts dark: give the set "lights": "off" '
                        '(a lecture_hall\'s, for now)')
    if st.get("table") is False and kind == "lecture_hall" and any(b.get("do") in ("put_down", "pick_up")
                                                                   for b in spec.get("beats", [])):
        raise SpecError('"table": false leaves the table out, and put_down and pick_up need it')
    tab = (st.get("table") or {}).get("at") if isinstance(st.get("table"), dict) else None
    if tab is not None and not (_point(tab) or (kind == "lecture_hall" and _board_ref(tab))):
        raise SpecError('"table": "at" is a spot on the floor as [x, y] in metres, or {"board": [u, out]}')
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
        at_ok = _point(c.get("at", [0, 0])) or (kind == "lecture_hall" and (_board_ref(c.get("at")) or (
            _door_ref(c.get("at")) and st.get("door") is not None)))
        if not at_ok or not _number(c.get("facing", 0)):
            raise SpecError(f'cast member {c.get("id")!r}: "at" is a spot on the floor as [x, y] in metres (in a '
                            f'lecture_hall also {{"board": [u, out]}}: in front of board pixel column u, out metres '
                            f'from the wall) and "facing" a number of degrees (0 faces -Y, positive turns to their left)')
    props = st.get("props") or [] if studio else []
    for p in props:
        if not isinstance(p.get("id"), str) or p.get("kind") not in PROP_KINDS:
            raise SpecError(f'a prop needs an "id" and a "kind" from {", ".join(PROP_KINDS)}: {p}')
        if p["kind"] == "model" and not p.get("file"):
            raise SpecError(f'prop {p["id"]!r}: a model needs "file"')
        if p["kind"] == "logo":
            if not p.get("file"):
                raise SpecError(f'prop {p["id"]!r}: a logo needs "file", an SVG')
            for k in ("size", "depth", "bevel", "lift"):
                if k in p and (not _number(p[k]) or p[k] < 0 or (k in ("size", "depth") and p[k] == 0)):
                    raise SpecError(f'prop {p["id"]!r}: "{k}" is a length in metres ({p[k]!r} given)')
            mats = [p.get("material", "satin")] + [o.get("material", "satin") for o in (p.get("parts") or {}).values()
                                                    if isinstance(o, dict)]
            for m in mats:
                if m not in logo.MATERIALS:
                    raise SpecError(f'prop {p["id"]!r}: no material called {m!r} (have: {", ".join(logo.MATERIALS)})')
            if p.get("base") is not None and (not isinstance(p["base"], dict) or p["base"].get("kind", "key") not in logo.BASES):
                raise SpecError(f'prop {p["id"]!r}: "base" is an object with a "kind" from {", ".join(logo.BASES)}')
    if studio and not props:
        raise SpecError('a studio set needs "props"')
    if studio and st.get("stage", "sweep") not in STAGES:
        raise SpecError(f'"stage" is one of {", ".join(STAGES)}, not {st["stage"]!r}')
    if studio and st.get("floor") is not None and not (isinstance(st["floor"], dict) and _number(st["floor"].get("rough", 0.22))
                                                      and 0 <= st["floor"].get("rough", 0.22) <= 1):
        raise SpecError('"floor" is how matt the dark stage\'s floor is: {"rough": 0.6}, from 0 (a mirror) to 1')
    for name, share in (st.get("lights") or {}).items() if studio else ():
        if name not in motion.LIGHTS or not _number(share) or share < 0:
            raise SpecError(f'"lights" scales the stage\'s own lights: {{"key": 1, "fill": 0.5}} with names from '
                            f'{", ".join(motion.LIGHTS)}')
    sounds = spec.get("sounds") or {}
    if not isinstance(sounds, dict):
        raise SpecError('"sounds" maps a name to the layers that make it: {"clap": [{"file": "sfx/slate.mp3"}]}')
    for name, layers in sounds.items():
        if not isinstance(layers, list) or not layers:
            raise SpecError(f'sound {name!r} is a list of layers, each a recording ({{"file": ...}}) or tones ({{"tones": [hz, ...]}})')
        for layer in layers:
            if not isinstance(layer, dict) or ("file" in layer) == ("tones" in layer):
                raise SpecError(f'sound {name!r}: a layer has "file" (a recording) or "tones" (frequencies in Hz), one of them')
            if "file" in layer:
                f = Path(layer["file"])
                if not (f if f.is_absolute() else base / f).exists():
                    raise SpecError(f'sound {name!r}: recording not found: {layer["file"]}')
            elif not (isinstance(layer["tones"], list) and layer["tones"] and all(_number(x) and x > 0 for x in layer["tones"])):
                raise SpecError(f'sound {name!r}: "tones" is a list of frequencies in Hz')
            for k in ("gain", "lowpass", "attack", "decay", "hold", "beat"):
                if k in layer and (not _number(layer[k]) or layer[k] < 0):
                    raise SpecError(f'sound {name!r}: "{k}" is a number, 0 or more')
    parts_of = {}
    for p in props:
        parts_of[p["id"]] = None
        if p["kind"] == "logo":
            f = Path(p["file"])
            f = f if f.is_absolute() else base / f
            if not f.exists():
                raise SpecError(f'prop {p["id"]!r}: logo file not found: {f}')
            try:
                parts_of[p["id"]] = [q.id for q in svg.read(f)[0]]
            except svg.SvgError as e:
                raise SpecError(f'prop {p["id"]!r}: {e}')
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
        if not isinstance(s.get("built_only", False), bool):
            raise SpecError(f'shot {s.get("id")!r}: "built_only" is true for a shot that is built and not rendered '
                            f'(the acting runs up to the next shot, or on after it)')
        if s.get("side") not in (None, "left", "right"):
            raise SpecError(f'shot {s.get("id")!r}: "side" is "left" or "right"')
        if s.get("cam") == "board" and (kind != "lecture_hall" or s["type"] != "fixed"):
            raise SpecError(f'shot {s.get("id")!r}: "cam": "board" is a fixed shot square on to a lecture_hall\'s board')
        if s["type"] == "move":
            if not (_framing(s.get("from")) and _framing(s.get("to"), end=True)):
                raise SpecError(f'shot {s.get("id")!r}: a move needs "from" and "to", each {{"cam": [x, y, z], "aim": '
                                f'[x, y, z], "lens": mm}} (points may be {{"hall": [x, out, z]}} in a lecture_hall, and '
                                f'"to" may be {{"cam": "board"}}, the board camera, with a "lens")')
            if kind != "lecture_hall" and any(_hall_ref(s[e].get(k)) for e in ("from", "to") for k in ("cam", "aim")) \
                    or (s["to"].get("cam") == "board" and kind != "lecture_hall"):
                raise SpecError(f'shot {s.get("id")!r}: "hall" points and "cam": "board" are a lecture_hall\'s')
            for k in ("start", "arrive"):
                v = s.get(k)
                if v is not None and v not in ("walk", "last_walk") and not (_number(v) and 0 <= v <= float(s.get("dur", 0))):
                    raise SpecError(f'shot {s.get("id")!r}: "{k}" is seconds into the shot (0 to its "dur"), "walk" '
                                    f'or (to set off) "last_walk"')
        if "mark" in s or s["type"] == "ident":
            if s.get("mark", "end") not in ("start", "end", "none"):
                raise SpecError(f'shot {s.get("id")!r}: "mark" is where the finished logo is checked against its SVG: '
                                f'"start", "end" or "none"')
            if s.get("mark", "end") != "none" and parts_of.get(subject) is None:
                if "mark" in s:
                    raise SpecError(f'shot {s.get("id")!r}: "mark" needs a logo as the subject, and {subject!r} is not one')
            for k in ("from", "to"):
                if k in s and not (isinstance(s[k], dict) and all(_number(v) for v in s[k].values())
                                   and set(s[k]) <= {"az", "elev", "back"}):
                    raise SpecError(f'shot {s.get("id")!r}: "{k}" is where the camera is at that end of the move: '
                                    f'{{"az": degrees, "elev": degrees, "back": 1.2}}')
            if "offset" in s and not _point(s["offset"]):
                raise SpecError(f'shot {s.get("id")!r}: "offset" is [right, up] as shares of the frame')
    for b in spec.get("beats", []):
        if "at" not in b or "do" not in b:
            raise SpecError(f'a beat needs "at" and "do": {b}')
        verb = b["do"]
        if "dur" in b and (isinstance(b["dur"], bool) or not isinstance(b["dur"], (int, float))):
            raise SpecError(f'beat at {b["at"]!r}: "dur" is a number of seconds ({b["dur"]!r} given)')
        if studio:  # a stage has no cast: its beats move props and their parts, light them and place sounds
            motion.check_beat(b, parts_of, tuple(sounds))
            continue
        if verb not in CHAR_VERBS | SET_VERBS | FX_VERBS:
            raise SpecError(f"unknown beat verb {verb!r} at {b['at']} (have: "
                            f"{', '.join(sorted(CHAR_VERBS | SET_VERBS | FX_VERBS))})")
        if verb in KITCHEN_VERBS and kind != "kitchen":
            raise SpecError(f'beat at {b["at"]!r}: {verb} is a kitchen verb (set kind kitchen)')
        if verb == "claude_type" and b.get("speed", "frantic") not in TYPING:
            raise SpecError(f'beat at {b["at"]!r}: claude_type takes "speed", one of {", ".join(TYPING)}')
        if verb == "claude_turn" and not _number(b.get("facing")):
            raise SpecError(f'beat at {b["at"]!r}: claude_turn needs "facing", a number of degrees (0 faces -Y)')
        if verb in ("claude_type", "claude_turn") and st.get("claude") is None:
            raise SpecError(f'beat at {b["at"]!r}: {verb} needs Claude in the kitchen: give the set "claude": {{}}')
        if verb == "open_door" and b.get("latch") is not None and not (
                _number(b["latch"]) and 0.2 <= b["latch"] <= tl_time_ok(b, spec)):
            raise SpecError(f'beat at {b["at"]!r}: "latch" is how many seconds before the door swings its handle goes '
                            f'down (0.2 or more, and not before the spec starts)')
        if verb == "lights":
            if not (_number(b.get("banks", 3)) and b.get("banks", 3) >= 1 and isinstance(b.get("stutter", True), bool)
                    and _number(b.get("gap", 0.45)) and b.get("gap", 0.45) >= 0):
                raise SpecError(f'beat at {b["at"]!r}: lights takes "banks" (how many groups come on, front to back: 3), '
                                f'"stutter" (true: each flashes before it holds), "gap" (seconds between groups: 0.45)')
            if b.get("by") is not None and st.get("door") is None:
                raise SpecError(f'beat at {b["at"]!r}: the light switch is beside the door: give the set "door": {{}}')
            if b.get("hand", "right") not in ("left", "right"):
                raise SpecError(f'beat at {b["at"]!r}: "hand" is "left" or "right"')
        if verb in ("open_door", "close_door", "open_lid", "close_lid", "lights") and b.get("by") is not None and b["by"] not in ids:
            raise SpecError(f'beat at {b["at"]!r}: "by" is who opens the door, and nobody in the cast is called '
                            f'{b["by"]!r} (have: {", ".join(ids)})')
        if verb in CHAR_VERBS and b.get("who", ids[0]) not in ids:
            raise SpecError(f'beat at {b["at"]!r}: nobody in the cast is called {b["who"]!r} (have: {", ".join(ids)})')
        if verb in ("carry", "put_down", "pick_up", "hold", "dust_hands", "talk", "blinks", "gesture"):
            if kind not in ("lecture_hall", "kitchen", "pub") and verb in ("carry", "put_down", "pick_up", "hold"):
                raise SpecError(f'beat at {b["at"]!r}: {verb} needs a set with things to hold (set kind lecture_hall, '
                                f'kitchen or pub)')
            have = ("cup",) if kind == "kitchen" else ("pint", "cup") if kind == "pub" else PROPS
            if verb in ("carry", "put_down", "pick_up", "hold") and b.get("prop") not in have:
                raise SpecError(f'beat at {b["at"]!r}: {verb} needs "prop", one of {", ".join(have)}')
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
        elif verb == "walk" and (not (_point(b.get("to")) or (kind == "lecture_hall" and (_board_ref(b.get("to")) or (
                _door_ref(b.get("to")) and st.get("door") is not None) or (
                _switch_ref(b.get("to")) and st.get("door") is not None and st.get("lights") == "off")))) or not _number(b.get("facing", 0))
                                 or not _number(b.get("heading", 0)) or not isinstance(b.get("curve", False), bool)
                                 or not (_number(b.get("stride", 0.5)) and 0.3 <= b.get("stride", 0.5) <= 0.85)):
            raise SpecError(f'beat at {b["at"]!r}: walk needs "to", a spot on the floor as [x, y] in metres (in a '
                            f'lecture_hall also {{"board": [u, out]}}, {{"door": out}}, or {{"switch": out}} in one '
                            f'that starts dark) (and '
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
                isinstance(b.get("target"), list) and len(b["target"]) == 3 and all(_number(x) for x in b["target"]))
                or (kind == "lecture_hall" and _board_ref(b.get("target")))):
            raise SpecError(f'beat at {b["at"]!r}: {verb} needs "target": a point in the room as [x, y, z] in metres'
                            f', or one of {", ".join(LOOK_AT)}' + (' (or {"board": [u, v]}, a board pixel)'
                                                                   if kind == "lecture_hall" else ''))
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
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (*[c * studio_set.WORLD for c in tint], 1)
    first = next(iter(room.props))
    try:
        heard, shakes, notes = logo.perform(room, spec.get("beats", []), tl)
    except ValueError as e:
        raise SpecError(str(e))
    notes += [f"{p.id}: {n}" for p in room.props.values() for n in getattr(p, "notes", [])]
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
            cam, rep = cameras.solve(s, sub, f0, f1, seed=i, fps=fps, shakes=[x for x in shakes if f0 <= x[0] <= f1])
        except ValueError as e:
            raise SpecError(str(e))
        sc.timeline_markers.new(s["id"], frame=f0).camera = cam
        cams.append((f0, f1, cam))
        if s.get("built_only"):
            rep["built_only"] = True
        where = s.get("mark", "end" if s["type"] == "ident" else "none")
        if where != "none" and getattr(prop, "kind", "") == "logo":
            # the frame a viewer is left with is the avatar: is it the flat mark?
            rep["mark"] = logo.mark_report(prop, cam, f0 if where == "start" else f1)
            rep["issues"] = rep["issues"] + rep["mark"].pop("issues")
            rep["passed"] = not rep["issues"]
        reports.append(rep)
    sc.camera = cams[0][2]
    kit.notes = getattr(kit, "notes", []) + notes
    events = [{"t": round((e["frame"] - 1) / fps, 3), "shot": tl.shot_at((e["frame"] - 1) / fps), **e}
              for e in sorted(heard, key=lambda e: e["frame"])]
    return tl, cams, reports, 0.0, events


def build(spec, args):
    base = Path(args.spec).resolve().parent
    validate(spec, base)
    fps = int(spec.get("fps", 24))
    res = tuple(int(x) for x in (args.res or spec.get("resolution", "1920x1080")).split("x"))
    look = spec.get("look", {})
    if any(p.get("kind") == "logo" for p in (spec.get("set") or {}).get("props") or []):
        # a logo's colours are the brand's: the Standard view at no exposure shows them as they were given
        look = {"view": "Standard", "exposure": 0.0, **look}
    tl = Timeline(spec["shots"], fps, spec.get("voice"), base)  # before Blender work: a bad spec fails fast
    bpy.ops.wm.read_factory_settings(use_empty=True)
    exposure = look.get("exposure", -0.9)
    kit.setup_render(fps, res, engine=args.engine or spec.get("engine", "eevee"),
                     exposure=exposure, look=look.get("look", "AgX - Punchy"), view=look.get("view", "AgX"),
                     samples=16 if args.mode == "check" else int(spec.get("samples", 32)),
                     transparent=bool(args.alpha or spec.get("transparent")),
                     raytracing=look.get("raytracing", True))
    sc = kit.scene()
    kit.notes = []
    if look.get("bloom"):
        said = kit.bloom(look["bloom"])
        if said:
            kit.notes.append(said)
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
        if any(k in c for k in ("posture", "at", "facing")) and not (_board_ref(c.get("at")) or _door_ref(c.get("at"))):
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
        # (the hall is sized by its presenter, as a desk is by whoever sits at it: hall.py)
        who = next(iter(cast.values()))
        scale = round(who.stature() / HALL_STATURE, 4) if who.desc.get("model") else 1.0
        rect = board_rect(st.get("board"), scale)
        for c in spec.get("cast", []):  # (a spot named by the board or its door is a place once the board is sized)
            if _board_ref(c.get("at")):
                cast[c["id"]].place(c.get("posture", "stand"), board_floor(rect, scale, *c["at"]["board"]),
                                    float(c.get("facing", 0)))
            elif _door_ref(c.get("at")):
                cast[c["id"]].place(c.get("posture", "stand"), door_floor(rect, scale, door, c["at"]["door"]),
                                    float(c.get("facing", 0)))
        room = LectureHall(st.get("palette"), walls=bool(st.get("walls", True)), door=door, board=st.get("board"),
                           image=picture, strength=float((st.get("board") or {}).get("brightness", 1.0)) * 2 ** -exposure,
                           table=st.get("table"), scale=scale, hall=st.get("hall"), lights=st.get("lights", "on"))
        _resolve_board(spec, room, cast)
        kit.hall = {"scale": scale, "stature_m": round(who.stature(), 3),
                    "board_m": {k: round(v, 3) for k, v in room.board.items()}}
    elif kind == "pub":
        who = next(iter(cast.values()))
        room = Pub(st.get("palette"), fit=who.furniture(), pint=st.get("pint"),
                   hand=who._breadth() if who.fingers.get(1) else 0.085)
    elif kind == "living_room":
        room = LivingRoom(st.get("palette"), fit=next(iter(cast.values())).furniture(), sofa=st.get("sofa"))
    elif kind == "kitchen":
        cl = st.get("claude")
        cl_media = cl_start = None
        if cl is not None:
            if not isinstance(cl, dict):
                raise SpecError('"claude" is an object (it may be empty: {})')
            cs = cl.get("screen") or {}
            cl_media = cs.get("video") or cs.get("image")
            if cl_media:
                cl_media = Path(cl_media) if Path(cl_media).is_absolute() else base / cl_media
                if not cl_media.exists():
                    raise SpecError(f"claude screen media not found: {cl_media}")
                if cs.get("video"):
                    clip = bpy.data.movieclips.load(str(cl_media))
                    rate = clip.fps
                    bpy.data.movieclips.remove(clip)
                    if rate and abs(rate - fps) > 0.01:
                        raise SpecError(f"claude's screen video {cl_media.name} is {rate:g} fps and the spec is {fps}")
            cl_start = tl.frame(tl.time(cs.get("at", 0)))
        room = Kitchen(st.get("palette"), scr, media_path=media_path,
                       media_start=tl.frame(tl.time(scr.get("at", 0))), media_offset=int(scr.get("skip_frames", 0)),
                       media_strength=float(scr.get("brightness", 1.0)) * 2 ** -exposure,
                       fit=next(iter(cast.values())).furniture(), door=door,
                       chair_out=float(st.get("chair_out", 0.0)), lid=st.get("lid", "open"), mug=st.get("mug"),
                       claude=cl, claude_media=cl_media, claude_start=cl_start or 1,
                       claude_strength=float(((cl or {}).get("screen") or {}).get("brightness", 1.0)) * 2 ** -exposure)
    else:
        room = DeskRoom(st.get("palette"), scr, walls=kind == "desk_room", media_path=media_path,
                        media_start=tl.frame(tl.time(scr.get("at", 0))), media_offset=int(scr.get("skip_frames", 0)),
                        media_strength=float(scr.get("brightness", 1.0)) * 2 ** -exposure,
                        fit=next(iter(cast.values())).furniture(), door=door,
                        chair_out=float(st.get("chair_out", 0.0)))
    bpy.context.view_layer.update()  # the cameras read where the screen is: it has to have been worked out
    for p in cast.values():  # (hands with fingers are put on the keys where they are in the room)
        p.keys = bpy.data.objects.get("lap_keys")
        p.table = getattr(room, "desk_top", None)  # (a lean stops at its edge; a hand on it stays on it)
        if getattr(room, "rest_pose", None) == "lap" and p.posture == "sit" and p.foot_t:
            p.lap(1)  # (a set with no desk: the hands start on the thighs)
    default_who = next(iter(cast))

    fx_events, at_camera, point_camera = [], [], []
    walked = {}  # each character's walks, (first frame, the frame they stand still at the end)
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
            elif verb == "settle":
                try:
                    p.settle(f0, f1 if "until" in b or "dur" in b else f0 + kit.n(10))
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
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
            elif verb in ("phone", "wag", "lap", "let_go"):
                try:
                    if verb == "phone":
                        img = b.get("image")
                        if img:
                            img = Path(img) if Path(img).is_absolute() else base / img
                            if not img.exists():
                                raise SpecError(f"phone image not found: {img}")
                        acting.phone(p, f0, f1 if "until" in b or "dur" in b else f0, b.get("hand", "left"),
                                     float(b.get("dist", 0.36)), float(b.get("drop", 0.07)), float(b.get("side", 0.06)),
                                     img, float(b.get("brightness", 1.0)) * 2 ** -exposure, b.get("scale"))
                    elif verb == "wag":
                        toward = b.get("toward")
                        if toward == "phone":
                            toward = acting.phone_point(p, f0)
                        acting.wag(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(36), b.get("hand", "right"),
                                   float(b.get("rate", 3.0)), float(b.get("swing", 16.0)), toward, b.get("seen_from"),
                                   b.get("shape", "point"), float(b.get("lift", -0.08)), float(b.get("out", -0.04)))
                    elif verb == "let_go":
                        acting.let_go(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(14), b.get("hand", "left"),
                                      getattr(room, "beside_laptop", None), float(b.get("away", 0.0)))
                    else:
                        p.lap(f0, f1)
                except ValueError as e:
                    raise SpecError(f'beat at {b["at"]!r}: {e}')
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
            elif verb in ("gesture", "dust_hands", "carry", "put_down", "pick_up", "hold", "sip"):
                try:
                    hand = b.get("hand")
                    if verb == "sip":
                        acting.sip(p, f0, f1 if "until" in b or "dur" in b else f0 + kit.n(40), hand or "left",
                                   part=float(b.get("part", 1.0)), stay=bool(b.get("stay", False)))
                    elif verb == "gesture":
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
                        elif verb == "pick_up" and not small and kind == "pub":
                            acting.take_up(p, in_hand, on_table, f0, end, hand or "left", lift=bool(b.get("lift", True)))
                        elif verb == "pick_up":
                            acting.pick_up(p, in_hand, on_table, f0, end, hand or ("right" if small else "left"))
                        elif small:
                            acting.leave_small(p, in_hand, on_table, b.get("to", list(spot)), f0, end, hand or "right")
                        else:
                            acting.put_down(p, in_hand, on_table, b.get("to", list(spot)), f0, end, hand or "left",
                                            lean=float(b.get("lean", 9.0)), rest=getattr(room, "beside_laptop", None))
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
                        if point == "phone":
                            point = acting.phone_point(p, f1)
                        elif point == "pint":
                            point = list(room.prop("pint")[2])
                        if verb == "look_at":
                            p.look_at(f0, f1, point)
                        else:
                            p.reach(f0, f1, point, b.get("hand", "right"))
                    elif verb == "walk":  # (untimed, a walk takes what its steps need)
                        end = p.walk(f0, f1 if timed else None, b["to"], b.get("facing"), bool(b.get("curve")),
                                     b.get("heading"), float(b.get("stride", 0.5)))
                        walked.setdefault(p.id, []).append((f0, end))  # (a move can set off and land on them)
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
                    if verb in ("open_lid", "close_lid"):
                        end = f1 if timed else f0 + kit.n(14)
                        shut = verb == "close_lid"
                        room.move_lid(f0, end, shut)
                        if b.get("by"):  # fingertips on the lid's top edge: they set out before, and go with the lid
                            who = cast[b["by"]]
                            x = 0.16 * (1 if b.get("hand", "right") == "left" else -1)  # (on that hand's side of it)
                            who.push(f0, end, lambda f, x=x: room.lid_edge(x), b.get("hand", "right"), what="lid",
                                     rest=room.beside_laptop, clear=shut)
                    elif verb == "claude_type":
                        room.claude_type(f0, f1 if timed else f0 + kit.n(48), b.get("speed", "frantic"))
                    elif verb == "claude_turn":
                        room.claude_turn(f0, f1 if timed else f0 + kit.n(14), float(b["facing"]))
                    elif verb == "lights":  # (with "by", their hand reaches the switch on the beat's frame)
                        if b.get("by"):
                            # (a point that arrives: the fingertip on the switch on the beat's frame, and back)
                            cast[b["by"]].point(f0 - kit.n(10), f0 + kit.n(4), list(room.switch_point()),
                                                b.get("hand", "right"), touch="light switch")
                        room.lights(f0, int(b.get("banks", 3)), bool(b.get("stutter", True)),
                                    float(b.get("gap", 0.45)) * 24, pressed=bool(b.get("by")))
                    elif verb == "pull_in":  # whoever sits on the chair comes in to the desk with it
                        end = f1 if timed else f0 + kit.n(34)
                        room.events.append({"frame": f0, "kind": "chair"})
                        cast[b.get("who", default_who)].scoot(f0, end, (0, 0), rides=[room.chair], grip=room.desk_edge)
                    else:
                        end = f1 if timed else f0 + kit.n(18)
                        latch = F(t0 - float(b["latch"])) if verb == "open_door" and b.get("latch") is not None else None
                        room.open_door(f0, end, float(b.get("angle", 100)) if verb == "open_door" else 0.0, latch)
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
        if s["type"] == "fixed" and (s.get("cam") == "phone" or s.get("aim") == "face"):
            # a camera that looks back from the phone at the face: what the other end of the call sees
            who, mid = cast[subj], (f0 + f1) // 2
            s = dict(s)
            try:
                if s.get("cam") == "phone":
                    s["cam"] = list(acting.phone_point(who, mid, float(s.get("near", 0.03))))
            except ValueError as e:
                raise SpecError(f'shot {s["id"]!r}: {e}')
            if s.get("aim") == "face":
                sc.frame_set(mid)
                bpy.context.view_layer.update()
                s["aim"] = list(who.head.matrix_world @ Vector((0, -0.3, float(s.get("aim_drop", 0.0)))))
        scenery, board = (), None
        if s["type"] == "move":
            s = dict(s)
            walks = walked.get(subj, [])
            for k, at, pick in (("start", f0, 0), ("arrive", f1, 1)):
                v = s.get(k)
                if v in ("walk", "last_walk"):
                    if not walks:
                        raise SpecError(f'shot {s["id"]!r}: "{k}": "{v}" needs {subj} to walk in the shot')
                    # (a move sets off with their first walk, or with their last: after a stop on the way in)
                    s[k + "_frame"] = (walks[-1] if v == "last_walk" else walks[0])[0] if pick == 0 else walks[-1][1]
                elif v is not None:
                    s[k + "_frame"] = f0 + round(float(v) * fps)
                else:
                    s[k + "_frame"] = at
            scenery = [o for o in sc.objects if o.type == "MESH" and not o.hide_render and o.visible_camera
                       and o not in people and abs(o.matrix_world.determinant()) > 1e-12]
            board = room if kind == "lecture_hall" and getattr(room, "board_face", None) is not None else None
        try:
            cam, rep = cameras.solve(s, sub, f0, f1, seed=i, fps=fps, avoid=people, scenery=scenery, board=board)
        except ValueError as e:
            raise SpecError(str(e))
        sc.timeline_markers.new(s["id"], frame=f0).camera = cam
        cams.append((f0, f1, cam))
        if s.get("built_only"):
            rep["built_only"] = True
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
    kit.lamps = getattr(room, "lamps", None)  # (lights that change on purpose: the flicker check leaves them out)
    kit.board = _board_cover(room, p, spec, tl, cams) if kind == "lecture_hall" else None
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
    tipped, sunk_in = contacts.pop("carried_tilt", None), contacts.pop("carried_through", None)
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
    if tipped and tipped["frame"] is not None:  # something carried: is it level, is it clear of the furniture
        checks["carried_tilt_deg"] = round(tipped["deg"], 1)
        if tipped["deg"] > CARRIED_TILT:
            checks["passed"] = False
            problems.append(f'the {tipped["what"]} is tipped {tipped["deg"]:.0f} degrees in the hand at '
                            f'{(tipped["frame"] - 1) / tl.fps:.2f} s: a full mug is carried level')
    if sunk_in and sunk_in["frame"] is not None:
        checks["carried_through_m"] = round(sunk_in["m"], 3)
        if sunk_in["m"] > CARRIED_THROUGH:
            checks["passed"] = False
            problems.append(f'the {sunk_in["what"]} in the hand is {sunk_in["m"] * 100:.1f} cm into the {sunk_in["into"]} at '
                            f'{(sunk_in["frame"] - 1) / tl.fps:.2f} s')
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
        loosest = max(grips, key=lambda g: g["open"])
        checks["grip_open"] = loosest["open"]
        if loosest["open"] > GRIP_OPEN:
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + [
                f'{loosest["hand"]} hand is open on the {loosest["what"]} at {(loosest["frame"] - 1) / tl.fps:.2f} s (fingertips '
                f'{loosest["open"]:.2f} of a finger\'s length from it): it should close on what it holds']
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
    board = getattr(kit, "board", None)
    if board is not None:  # a lecture: the hall's size, and how much of the board the presenter hides
        hall = getattr(kit, "hall", None)
        checks["board"] = {**(hall or {}), **board, "limit": BOARD_COVER}
        said_board = []
        if board["share"] > BOARD_COVER:
            said_board.append(f'the presenter hides {100 * board["share"]:.0f}% of the board at '
                              f'{(board["frame"] - 1) / tl.fps:.2f} s (limit {100 * BOARD_COVER:.0f}%): stand them further '
                              f'from the board, or nearer its edge')
        if board["outside_window"]:
            said_board.append(f'the presenter reaches out of the shot\'s window on {board["outside_window"]} of the '
                              f'frames looked at (board pixels {board["box_px"]}): a clip of the window cuts them off')
        if said_board:
            checks["passed"] = False
            checks["body_problems"] = checks.get("body_problems", []) + said_board
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
    base = Path(args.spec).resolve().parent
    sounds = {name: [{**layer, "file": str(Path(layer["file"]) if Path(layer["file"]).is_absolute() else base / layer["file"])}
                     if "file" in layer else layer for layer in layers]
              for name, layers in (spec.get("sounds") or {}).items()}
    heard = {"fps": tl.fps, "duration": tl.duration, "events": events, "score": score}
    if sounds:  # recordings and tones the spec names for its events (sound.py places each on its frame)
        heard["sounds"], heard["mix"] = sounds, spec.get("mix") or {}
    (out / "events.json").write_text(json.dumps(heard, indent=2))
    if getattr(kit, "notes", None):
        checks["notes"] = kit.notes
        (out / "checks.json").write_text(json.dumps(checks, indent=2))
    if os.environ.get("BLENDER_CONTACT_TRACE"):  # every frame's contact numbers, to look into a failure
        (out / "contacts.json").write_text(json.dumps(getattr(kit, "contact_trace", [])))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))
    if args.mode == "render" and not args.bench:  # where still surfaces are in each frame: read against the render
        wanted_cams = [c for c in cams if not args.shots or c[2].name in [x.strip() for x in args.shots.split(",")]]
        found = cameras.still_points(wanted_cams)
        lamps = getattr(kit, "lamps", None)
        if lamps is not None and lamps.changes:  # (the frames where lights change on purpose: not flicker)
            found["light_changes"] = [list(c) for c in sorted({tuple(c) for c in lamps.changes})]
        (out / "still_points.json").write_text(json.dumps(found))

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
            changed = [b for a, b in getattr(getattr(kit, "lamps", None), "changes", []) if f0 <= a <= f1]
            if changed:  # (a hall that starts dark is judged once its lights hold, not part way through them)
                f = min(f1, max(f, max(changed) + 1))
            f = next((r["mark"]["frame"] for r in reports if r["id"] == cam.name and r.get("mark")), f)
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
