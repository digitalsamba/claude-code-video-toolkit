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
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpy  # noqa: E402

import cameras  # noqa: E402
import cast as casting  # noqa: E402
import kit  # noqa: E402
from character import FACES, LEANS, LOOKS, POSES, TYPING, ClayPerson  # noqa: E402
from sets import DeskRoom, confetti, title3d  # noqa: E402
from studio import PROP_KINDS, Studio  # noqa: E402
from timeline import SpecError, Timeline  # noqa: E402

SET_KINDS = ("desk_room", "desk", "studio")
SCORE_MOODS = ("calm", "tension", "resolve")
CHAR_VERBS = {"type", "pose", "expression", "look", "lean", "head_bob", "shake_head", "gulp", "blink", "sweat",
              "tremble", "tap", "celebrate", "hold_arms"}
SET_VERBS = {"alert", "work", "done"}
FX_VERBS = {"confetti", "title"}


def _one_of(beat, field, table, numbers=False, default=None):
    v = beat.get(field, default)
    if numbers and isinstance(v, (int, float)) and not isinstance(v, bool):
        return
    if not isinstance(v, str) or v not in table:
        raise SpecError(f'beat at {beat["at"]!r}: {beat["do"]} needs "{field}" to be one of {", ".join(table)}'
                        + (" or a number of degrees" if numbers else "") + f", not {v!r}")


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
    spec.setdefault("cast", [])
    if not isinstance(spec.get("shots"), list) or not spec["shots"]:
        raise SpecError('spec needs a "shots" list')
    if not isinstance(spec["cast"], list) or (not spec["cast"] and not studio):
        raise SpecError('spec needs a "cast" list (only a studio set can go without one)')
    if studio and spec["cast"]:
        raise SpecError("a studio set has props and no characters yet: leave out the cast")
    ids = [c.get("id") for c in spec["cast"]]
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
        if verb in CHAR_VERBS and b.get("who", ids[0]) not in ids:
            raise SpecError(f'beat at {b["at"]!r}: nobody in the cast is called {b["who"]!r} (have: {", ".join(ids)})')
        if verb == "type":
            _one_of(b, "speed", TYPING, default="calm")
        elif verb == "pose":
            _one_of(b, "pose", POSES)
        elif verb == "expression":
            _one_of(b, "face", FACES)
        elif verb == "look":
            _one_of(b, "target", LOOKS, numbers=True, default="screen")
        elif verb == "lean":
            _one_of(b, "amount", LEANS, numbers=True, default="none")
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
    # the picture's emission cancels the film's exposure, so under the Standard view its colours come out as made
    room = DeskRoom(st.get("palette"), scr, walls=kind == "desk_room", media_path=media_path,
                    media_start=tl.frame(tl.time(scr.get("at", 0))), media_offset=int(scr.get("skip_frames", 0)),
                    media_strength=float(scr.get("brightness", 1.0)) * 2 ** -exposure)
    cast = {}
    for c in spec.get("cast", []):
        if c.get("kind", "clay_person") != "clay_person":
            raise SpecError(f"unknown cast kind {c.get('kind')!r} (prototype has: clay_person)")
        if cast:
            raise SpecError("prototype supports one character per set")
        cast[c["id"]] = ClayPerson(c["id"], casting.resolve(c, base))
    if not cast:
        raise SpecError("spec needs a cast")
    default_who = next(iter(cast))

    fx_events = []
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
        elif verb in SET_VERBS:
            if verb == "alert":
                room.alert(f0)
            elif verb == "work":
                room.work(f0, f1)
            elif verb == "done":
                room.done(f0)
        elif verb == "confetti":
            confetti(f0, int(b.get("count", 60)), colors=b.get("colors"))
            fx_events.append({"frame": f0, "kind": "confetti"})
        elif verb == "title":
            title3d(f0, b["text"], tl.frames - 1, c=b.get("color", "#FFD24A"))
            fx_events.append({"frame": f0, "kind": "title"})

    for p in cast.values():
        p.breathe(1, tl.frames)

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

    p = cast[default_who]
    ik = p.ik_error(range(1, tl.frames, 6))
    events = [{"t": round((e["frame"] - 1) / fps, 3), "shot": tl.shot_at((e["frame"] - 1) / fps), **e}
              for e in sorted(p.events + room.events + fx_events, key=lambda e: e["frame"])]
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
    checks = {"passed": all(r["passed"] for r in reports) and ik < 0.01, "duration": tl.duration, "fps": tl.fps,
              "resolution": [sc.render.resolution_x, sc.render.resolution_y], "alpha": alpha,
              "engine": sc.render.engine, "blender": bpy.app.version_string,
              "frames": tl.frames - 1, "ik_max_error_m": round(ik, 4), "shots": reports}
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    (out / "events.json").write_text(json.dumps({"fps": tl.fps, "duration": tl.duration, "events": events,
                                                 "score": score}, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"))

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
            f = f1 if move == "out" else f0 if move == "in" else (f0 + f1) // 2
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
