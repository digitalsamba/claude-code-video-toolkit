#!/usr/bin/env python3
"""A lecture at a chalkboard: a fitted presenter rendered in Blender, standing at a board that Remotion draws.

A lecture is a script, data and a timeline, with no code of its own (templates/lecture is one, and its README is
the walkthrough). This tool turns the data into the Blender half and checks it:

    uv run tools/lecture.py timeline projects/my-talk    # audio + captions -> timeline.json (starts, words, windows)
    uv run tools/lecture.py specs    projects/my-talk    # lecture.json + timeline -> one presenter clip spec a segment
    uv run tools/lecture.py check    projects/my-talk    # each clip checked, one Blender at a time, then the joins
    uv run tools/lecture.py render   projects/my-talk    # the clips, one at a time, into public/blender
    uv run tools/lecture.py bg       projects/my-talk    # the inset's backdrop, cut from public/board/bg.png

What is where in a lecture project:
    lecture.json                     the lecture: presenter, staging, rules, and per segment the board and the acting
    public/audio/scenes/<id>.txt     what each segment says (voiced by voiceover.py --scene-dir, captioned by captions.py)
    timeline.json                    written by `timeline`: when each segment starts, its words by frame, the windows
    blender/presenter/P<nn>.json     written by `specs`: one transparent clip a segment, joined pose to pose
    public/blender/clips.json        written by `render`: the clips Remotion plays (lib/lecture)

The acting rules are options (lecture.json "rules"), each with the value the first lecture was made with. Every
cue is looked up in the narration's captions here, before Blender is asked, and a beat that would collide with
another is dropped with a line saying so.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The rules and the staging, as the first lecture (an eight-minute sprint review) was made with them, shared with
# lib/lecture: lib/lecture/defaults.json. A lecture.json "rules" or "staging" key overrides the default.
DEFAULTS = json.loads((ROOT / "lib" / "lecture" / "defaults.json").read_text(encoding="utf-8"))
RULES, STAGING, WINDOW = DEFAULTS["rules"], DEFAULTS["staging"], DEFAULTS["window"]
FACES = ("happy", "content", "worried", "neutral", "grin", "smile", "shocked")
ROOM_PER = 4.3  # seconds a frame for the opening's room shot (the dressed hall, opaque, 1920x1080) on an Apple M4


class LectureError(SystemExit):
    pass


# ------------------------------------------------------------------------------------------------ the narration
def norm(text: str) -> list[str]:
    """Words the way lib/components/cues.ts compares them: lower case, punctuation dropped, apostrophes kept inside."""
    s = text.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^a-z0-9']+", " ", s)
    s = re.sub(r"(^| )'+|'+( |$)", r"\1\2", s)
    return s.split()


class Words:
    """The narration's words, each with its frame in the video, for cue lookups."""

    def __init__(self, timeline: dict):
        self.tl = timeline
        self.by_seg = {s["id"]: s["words"] for s in timeline["segments"]}

    def cue(self, seg_id: str, phrase: str, nth: int = 1, edge: str = "start") -> int:
        want = norm(phrase)
        words = self.by_seg[seg_id]
        seen = 0
        for i in range(len(words) - len(want) + 1):
            if [w["w"] for w in words[i:i + len(want)]] == want:
                seen += 1
                if seen == nth:
                    return words[i]["f"] if edge == "start" else words[i + len(want) - 1]["e"]
        said = " ".join(w["w"] for w in words)
        raise LectureError(f'{seg_id}: "{phrase}" is not in the narration (match {nth}); it says: "{said[:400]}"')


def at_of(words: Words, seg: dict, at) -> int:
    """A board item's or a beat's frame: "words" (when they start), {"cue", "nth", "offset", "edge"}, "start"
    (the segment's first word), "end" (its last), or a number of frames after the first word."""
    if isinstance(at, (int, float)) and not isinstance(at, bool):
        return seg["startFrame"] + int(at)
    if at == "start":
        return seg["startFrame"]
    if at == "end":
        return seg["startFrame"] + seg["frames"]
    if isinstance(at, str):
        return words.cue(seg["id"], at)
    if isinstance(at, dict) and "cue" in at:
        return words.cue(seg["id"], at["cue"], int(at.get("nth", 1)), at.get("edge", "start")) + int(at.get("offset", 0))
    raise LectureError(f'{seg["id"]}: "{at}" is not a time: give words of the narration, {{"cue": ..., "offset": ...}}, '
                       f'"start", "end" or a number of frames')


# ------------------------------------------------------------------------------------------------ files
def load(project: Path) -> dict:
    p = project / "lecture.json"
    if not p.exists():
        raise LectureError(f"{p} not found: a lecture project has a lecture.json (start from templates/lecture)")
    lec = json.loads(p.read_text(encoding="utf-8"))
    lec["rules"] = {**RULES, **lec.get("rules", {})}
    lec["staging"] = {**STAGING, **lec.get("staging", {})}
    op = lec.get("opening")
    if op is True:
        op = {}
    if isinstance(op, dict):  # (the establishing shot: defaults under the lecture's own keys)
        base = DEFAULTS["opening"]
        lights = op.get("lights", base["lights"])
        if lights not in ("switch", "on"):
            raise LectureError('"opening": "lights" is "switch" (the hall starts dark and the presenter switches it on) '
                               'or "on" (lit from the start)')
        if lights == "on":  # (the lit hall's own times, as it was before the lights could be off)
            base = {k: v for k, v in {**base, **base["lit"]}.items()
                    if k not in ("step_in", "switch_spot", "switch_facing", "switch", "banks", "stutter", "gap")}
        lec["opening"] = {**base, **op, "camera": {**base["camera"], **op.get("camera", {})},
                          "sounds": {**base["sounds"], **op.get("sounds", {})}}
        o = lec["opening"]
        o.pop("lit", None)
        fade_end = o["fade"][0] + o["fade"][1]
        if lights == "on" and not (o["latch"] is not None and o["latch"] < o["door"]
                                   and fade_end < o["door"] < o["walk"][0] < o["walk"][1] < o["seconds"]):
            raise LectureError('"opening": the latch, the fade, the door, the walk and the cut have to come in that order '
                               f'(latch {o["latch"]}, fade {o["fade"]}, door {o["door"]}, walk {o["walk"]}, cut {o["seconds"]})')
        if lights == "switch" and not ((o["latch"] is None or o["latch"] < o["door"]) and fade_end < o["door"]
                                       < o["step_in"][0] < o["step_in"][1] < o["switch"] < o["walk"][0] < o["walk"][1]
                                       < o["seconds"]):
            raise LectureError('"opening": the fade, the door, the step in, the switch, the walk and the cut have to come '
                               f'in that order (fade {o["fade"]}, door {o["door"]}, step_in {o["step_in"]}, switch '
                               f'{o["switch"]}, walk {o["walk"]}, cut {o["seconds"]})')
    elif op not in (None, False):
        raise LectureError('"opening" is {} (or true) for the establishing shot, with any of its keys from lib/lecture/defaults.json')
    else:
        lec["opening"] = None
    ids = [s["id"] for s in lec.get("segments", [])]
    if not ids or len(set(ids)) != len(ids):
        raise LectureError('lecture.json needs "segments", each with its own "id" (the audio file stem)')
    return lec


def room_seconds(lec: dict) -> float:
    """How long the lecture is in the hall before the board: the opening's length, or "room_seconds"."""
    return float(lec["opening"]["seconds"]) if lec.get("opening") else float(lec.get("room_seconds", 0.0))


def probe_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def read_timeline(project: Path) -> dict:
    p = project / "timeline.json"
    if not p.exists():
        raise LectureError(f"{p} not found: run `uv run tools/lecture.py timeline {project}` after the voiceover")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ timeline
def cmd_timeline(project: Path, args) -> dict:
    """Segment starts from the audio's lengths and the rules' gaps; each segment's words by frame; the windows
    (where the camera goes into the board) as frames, for the board and the acting alike."""
    lec = load(project)
    fps = int(lec.get("fps", 30))
    r = lec["rules"]
    scenes = project / "public" / "audio" / "scenes"
    old = json.loads((project / "timeline.json").read_text()) if (project / "timeline.json").exists() else {}
    t = room_seconds(lec) + (lec["opening"]["first_word"] if lec.get("opening") else r["start"])
    segs = []
    for s in lec["segments"]:
        mp3, caps = scenes / f"{s['id']}.mp3", scenes / f"{s['id']}.captions.json"
        if not mp3.exists():
            raise LectureError(f"{mp3} not found: voice the script first (voiceover.py --scene-dir {scenes})")
        if not caps.exists():
            raise LectureError(f"{caps} not found: uv run tools/captions.py {scenes}")
        dur = probe_seconds(mp3)
        start_f = round(t * fps)
        words = []
        for c in json.loads(caps.read_text(encoding="utf-8")):
            for w in norm(c["text"]):
                words.append({"w": w, "f": start_f + round(c["startMs"] * fps / 1000),
                              "e": start_f + round(c["endMs"] * fps / 1000)})
        segs.append({"id": s["id"], "start": round(t, 4), "startFrame": start_f, "dur": round(dur, 3),
                     "frames": math.ceil(dur * fps), "words": words})
        t += dur + float(s.get("gap_after", r["gap"]))
    end = segs[-1]["start"] + segs[-1]["dur"] + r["tail"]
    tl = {"note": "written by tools/lecture.py timeline from the audio, the captions and lecture.json",
          "fps": fps, "segments": segs, "end": round(end, 3), "totalFrames": round(end * fps),
          "roomFrames": round(room_seconds(lec) * fps)}
    if lec.get("opening"):
        o = lec["opening"]
        tl["opening"] = {"fadeFrom": round(o["fade"][0] * fps), "fadeFrames": round(o["fade"][1] * fps)}
    words = Words(tl)
    windows, moves = [], []
    for seg, s in zip(segs, lec["segments"]):
        for w in s.get("windows", []):
            o = {**WINDOW, **w}
            a = at_of(words, seg, o["in"]) + (o["lead"] if not isinstance(o["in"], dict) else 0)
            b = (seg["startFrame"] + seg["frames"] + 2 if o.get("out", "end") == "end"
                 else at_of(words, seg, o["out"]) + (o["back"] if not isinstance(o["out"], dict) else 0))
            if b <= a + o["in_frames"]:
                raise LectureError(f'{seg["id"]}: the window from "{o["in"]}" to "{o.get("out", "end")}" is over before '
                                   f'the camera is in ({b - a} frames)')
            zoom = {"s": o["zoom"], "cx": o["cx"], "cy": o["cy"]}
            moves.append({"at": a, "dur": o["in_frames"], **zoom})
            for lean in o.get("moves", []):  # (a lean inside the window, on a phrase)
                moves.append({"at": at_of(words, seg, lean["at"]), "dur": int(lean.get("dur", 36)),
                              "s": lean.get("zoom", o["zoom"]), "cx": lean.get("cx", o["cx"]), "cy": lean.get("cy", o["cy"])})
            moves.append({"at": b, "dur": o["out_frames"], "s": 1, "cx": 960, "cy": 540})
            windows.append({"segment": seg["id"], "from": a + o["in_frames"], "to": b})
    tl["windows"] = windows
    tl["moves"] = sorted(moves, key=lambda m: m["at"])
    for k in ("presenter",):  # (what `check` measured stays until it is measured again)
        if k in old:
            tl[k] = old[k]
    (project / "timeline.json").write_text(json.dumps(tl, indent=1) + "\n")
    for s in segs:
        print(f"{s['id']}: {s['start']:7.2f} s (frame {s['startFrame']}), {s['dur']:.2f} s, {len(s['words'])} words")
    print(f"total {end:.1f} s ({tl['totalFrames']} frames), {len(windows)} windows")
    return tl


# ------------------------------------------------------------------------------------------------ specs
def clip_id(seg_id: str) -> str:
    return "P" + re.sub(r"[^A-Za-z0-9]", "", seg_id.split("_")[0])[:4]


def clip_spans(lec: dict, tl: dict) -> list[tuple[int, int]]:
    """Each segment's clip, in master frames: from lead_frames before its first word to the next clip's start
    (the first from the end of the room shots, or frame 0)."""
    segs, lead = tl["segments"], int(lec["rules"]["lead_frames"])
    first = round(room_seconds(lec) * tl["fps"])
    starts = [first] + [s["startFrame"] - lead for s in segs[1:]] + [tl["totalFrames"]]
    return [(starts[i], starts[i + 1]) for i in range(len(segs))]


def in_window(tl: dict, a: int, b: int, fade: int = 10) -> bool:
    return any(a <= w["to"] + fade and b >= w["from"] - fade for w in tl.get("windows", []))


def presenter_cast(project: Path, lec: dict) -> Path:
    p = lec.get("presenter", {}).get("cast")
    if not p:
        raise LectureError('lecture.json needs "presenter": {"cast": "<path to a character json>"}')
    path = (project / p).resolve()
    if not path.exists():
        raise LectureError(f"presenter cast {path} not found")
    desc = json.loads(path.read_text())
    model = desc.get("model")
    if model and not (path.parent / model).exists():
        raise LectureError(f"{path.name} names the fitted model {model}, which is not there yet: fit it once "
                           f"(docs/blender.md, 'A character from a generated 3D model'), e.g. for Mara "
                           f"`uv run tools/blender_render.py fit --model tests/fixtures/blender-fitted/mara.glb "
                           f"--fit tests/fixtures/blender-fitted/fit.json --out {path.parent / model}`")
    return path


def carried(lec: dict, tl: dict, spans, i: int) -> list[dict] | None:
    """The beats that cross the join at the start of clip i, in master frames (None for no carried move): by
    default they are already turning to the board where the new heading goes as the old one is wiped, and the new
    clip starts inside that turn. In a window the turn would show the inset their profile, so it is a front gesture."""
    r, fps = lec["rules"], tl["fps"]
    if i == 0 or r["carry"] == "still":
        return None
    seg_def = lec["segments"][i]
    if seg_def.get("carry", True) is False or not seg_def.get("heading"):
        return None
    join = spans[i][0]
    f0 = join - round(r["carry_lead"] * fps)
    speech = tl["segments"][i]["startFrame"]
    hold_to = speech + round(1.1 * fps)  # (held until the heading is chalked, then back to the hall)
    if in_window(tl, f0 + round(r["turn"] * fps), hold_to):
        kind = r["in_window"]
        return [{"f": f0, "do": "gesture", "kind": kind, "hand": other(lec), "dur": round((hold_to - f0) / fps, 3)}]
    target = seg_def.get("heading_at", r["heading_at"])
    return [{"f": f0, "do": "present", "target": {"board": list(target)}, "dur": round((hold_to - f0) / fps, 3)},
            {"f": hold_to + round(0.25 * fps), "do": "look_at", "target": "camera", "dur": 0.4}]


def table_of(st: dict) -> dict:
    """The close's table as the set takes it, placed by the board in the hall's measure."""
    t = st["table"]
    return {"at": {"board": list(t["at"])}, "chalk": {"board": list(t["chalk"])}, "size": list(t["size"]),
            "height": t["height"]}


def other(lec: dict) -> str:
    return "right" if lec["staging"]["chalk"] == "left" else "left"


def spec_for(project: Path, lec: dict, tl: dict, words: Words, spans, i: int, notes: list) -> tuple[str, dict]:
    r, st, fps = lec["rules"], lec["staging"], tl["fps"]
    seg, sdef = tl["segments"][i], lec["segments"][i]
    f0, f1 = spans[i]
    sid = clip_id(seg["id"])
    into = carried(lec, tl, spans, i)                 # the move this clip starts inside
    opening = lec.get("opening") if i == 0 else None  # the first clip comes in at the door, in the room shot
    # A clip that starts inside a move has a lead-in shot ("pre") that is built and not rendered: the move starts
    # from the stance in it, so the clip's first frame is mid-move. (A character is placed, and its stance keyed,
    # on a spec's first frame, so a move cannot start before it.) After an opening the lead-in is the opening:
    # the walk in is acted in this clip too, so the clip starts in the pose the room shot ends on.
    pre_f = round(float(r["pre_roll"]) * fps) if into else round(opening["seconds"] * fps) if opening else 0
    pre = pre_f / fps
    dur = (f1 - f0) / fps
    loc = lambda frame: round((frame - f0) / fps + pre, 3)  # noqa: E731  (master frame -> seconds into the spec)
    at = loc(seg["startFrame"])                       # where this segment's first word is, in the spec
    end = at + seg["dur"]
    clip_end = pre + dur
    cap = Path("../../public/audio/scenes") / f"{seg['id']}.captions.json"
    last = i == len(tl["segments"]) - 1
    hand, chalk = other(lec), st["chalk"]
    out = carried(lec, tl, spans, i + 1) if not last else None  # the move it ends inside
    setup_t = 0.0
    beats = [{"at": setup_t, "do": "hand", "shape": "loose", "hand": hand, "dur": 0},
             {"at": round(0.2 if opening else pre + 0.1, 3), "do": "blinks"},
             {"at": round(at, 4), "do": "talk", "until": round(end + 0.2, 4)}]
    if opening:  # the latch behind the door, the door, the walk to the mark (the room shot lands as it ends)
        o = opening
        door = {"at": o["door"], "do": "open_door", "dur": 0.8, "by": "presenter", "hand": hand}
        if o.get("latch") is not None:
            door["latch"] = round(o["door"] - o["latch"], 3)
        beats += [door,
                  # (the hand that let go of the handle hangs loose again, as every later clip starts it)
                  {"at": round(o["door"] + 0.9, 3), "do": "hand", "shape": "loose", "hand": hand, "dur": 0.35}]
        if o.get("lights") == "switch":  # in at the door, the switch beside it with the same hand, then the walk
            # (one bend from the doorway to side-on at the wall, the switch at that hand's shoulder: "reach" is as
            # far out as their own arm presses it straight)
            beats += [{"at": o["step_in"][0], "do": "walk", "to": {"switch": o["switch_spot"]},
                       "facing": o["switch_facing"], "curve": True, "heading": o["switch_facing"], "until": o["step_in"][1]},
                      {"at": o["switch"], "do": "lights", "by": "presenter", "hand": hand, "banks": o["banks"],
                       "stutter": o["stutter"], "gap": o["gap"]}]
        beats += [{"at": o["walk"][0], "do": "walk", "to": {"board": list(st["stand"])}, "facing": st["facing"],
                   "curve": True, "heading": o["heading"], "until": o["walk"][1], "stride": o["stride"]},
                  # (the arms come to hang as the later clips' placed stance has them, so those clips meet this one)
                  {"at": o["walk"][1], "do": "settle", "dur": 0.3}]
    elif not into:  # (a still join: head and eyes on the lens either side)
        beats.insert(0, {"at": -0.1, "do": "look_at", "target": "camera", "dur": 0.1})
    beats.append({"at": setup_t, "do": "hold", "prop": "chalk", "hand": chalk})
    arm_free, look_free = 0.0, 0.0
    if into:
        for b in into:
            beats.append({**{k: v for k, v in b.items() if k != "f"}, "at": loc(b["f"])})
        end_into = loc(into[0]["f"]) + into[0]["dur"]
        arm_free, look_free = end_into + 0.8, end_into + 0.6
    elif sdef.get("heading"):  # the first clip: the turn to the heading inside the clip
        a0 = at - 0.35
        if in_window(tl, f0 + round((a0 - pre + r["turn"]) * fps), f0 + round((at - pre + 1.35) * fps)):
            raise LectureError(f"{sid}: the heading is chalked while the inset is on screen; they would turn away in it")
        target = sdef.get("heading_at", r["heading_at"])
        beats += [{"at": round(a0, 3), "do": "present", "target": {"board": list(target)}, "dur": 1.5},
                  {"at": round(at + 1.35, 3), "do": "look_at", "target": "camera", "dur": 0.4}]
        arm_free, look_free = at + 2.4, at + 2.0
    # the end of the clip is kept for the move that crosses into the next one
    tail = loc(out[0]["f"]) - 0.1 if out else clip_end + 99
    for b in out or []:
        if loc(b["f"]) < clip_end:
            beats.append({**{k: v for k, v in b.items() if k != "f"}, "at": loc(b["f"])})
    holds = r["holds"]

    def free(t, busy_until, length):
        return t >= busy_until and t + length <= tail

    # (in the order they are said, whatever order lecture.json lists them in: a beat takes the arm from what follows)
    acts = sorted(((loc(at_of(words, seg, a[0])), k, a) for k, a in enumerate(sdef.get("acting", []))))
    for t, _, act in acts:
        cue, what, arg = act[0], act[1], act[2] if len(act) > 2 else None
        if what == "gesture":
            if arg not in holds:
                raise LectureError(f'{seg["id"]}: gesture "{arg}" on "{cue}" is not one of {", ".join(holds)}')
            if not free(t - 0.15, arm_free, holds[arg] + r["rest"]):
                notes.append(f"{sid}: dropped the {arg} gesture on '{cue}' (the arm is busy, or the join is near)")
                continue
            beats.append({"at": round(t - 0.15, 3), "do": "gesture", "kind": arg, "hand": hand, "dur": holds[arg]})
            arm_free = t - 0.15 + holds[arg] + r["rest"]
        elif what == "show":
            turned = (f0 + round((t - pre - 0.2 + r["turn"]) * fps), f0 + round((t - pre + 1.5) * fps))
            if in_window(tl, *turned):
                kind = act[3] if len(act) > 3 else r["in_window"]
                if not free(t - 0.15, arm_free, holds[kind] + r["rest"]):
                    notes.append(f"{sid}: dropped the gesture for the board look on '{cue}' (the arm is busy)")
                    continue
                beats.append({"at": round(t - 0.15, 3), "do": "gesture", "kind": kind, "hand": hand, "dur": holds[kind]})
                arm_free = t - 0.15 + holds[kind] + r["rest"]
                notes.append(f"{sid} frame {f0 + round((t - pre) * fps)}: '{cue}' faces the hall ({kind}): the camera is in a window")
                continue
            if not free(t - 0.2, arm_free, 2.3):
                notes.append(f"{sid}: dropped the board look on '{cue}' (the arm is busy, or the join is near)")
                continue
            beats += [{"at": round(t - 0.2, 3), "do": "present", "target": {"board": list(arg)}, "dur": 1.5},
                      {"at": round(t + 1.5, 3), "do": "look_at", "target": "camera", "dur": 0.4}]
            arm_free, look_free = t - 0.2 + 1.5 + 0.8, t + 2.0
        elif what == "look":
            if arg not in FACES:
                raise LectureError(f'{seg["id"]}: look "{arg}" on "{cue}" is not one of {", ".join(FACES)}')
            if t + 1.7 > tail:
                notes.append(f"{sid}: dropped the look on '{cue}' (the join is near)")
                continue
            beats += [{"at": round(max(t - 0.25, look_free), 3), "do": "look_at", "target": "camera", "dur": 0.3},
                      {"at": round(t, 3), "do": "expression", "face": arg, "dur": 0.3},
                      {"at": round(t + 1.6, 3), "do": "expression", "face": "neutral", "dur": 0.5}]
            look_free = t + 1.7
        elif what == "beat":  # any other verb, as the blender skill writes it, at the cue
            beats.append({**arg, "at": round(t, 3)})
        else:
            raise LectureError(f'{seg["id"]}: acting "{what}" on "{cue}": use gesture, show, look or beat')
    # between the looks at the lens and the board, they look out at the hall now and then
    lr = r["look_round"]
    looks = sorted(b["at"] for b in beats if b["do"] in ("look_at", "present"))
    t, k = at + lr["from"], i
    while t < min(end - 2.0, tail - 2.6):
        if all(abs(t - x) > lr["clear"] for x in looks):
            beats += [{"at": round(t, 3), "do": "look_at", "target": r["audience"][k % len(r["audience"])], "dur": 0.5},
                      {"at": round(t + 2.0, 3), "do": "look_at", "target": "camera", "dur": 0.5}]
            k += 1
            t += lr["every"]
        else:
            t += 0.7
    if last and r["close"]:  # chalk down on the table (just out of frame), hands dusted, a look and a smile
        beats += [{"at": round(end + 0.3, 3), "do": "put_down", "prop": "chalk", "hand": chalk, "dur": 0.75},
                  {"at": round(end + 1.95, 3), "do": "dust_hands", "dur": 1.15},
                  {"at": round(end + 3.0, 3), "do": "look_at", "target": "camera", "dur": 0.4},
                  {"at": round(end + 3.1, 3), "do": "expression", "face": "happy", "dur": 0.4}]
    cast = presenter_cast(project, lec)
    rel = Path(*[".."] * 2) / cast.relative_to(project) if cast.is_relative_to(project) else cast
    win = st["window"]
    w = round((win[2] - win[0]) * 1920 * st["density"] / 2) * 2
    h = round((win[3] - win[1]) * 1080 * st["density"] / 2) * 2
    table = {**table_of(st), "hidden": True, "cup_on": True}
    if lec.get("opening") and not r["close"]:  # (no table where the walk in goes: it is only there for the close)
        table = False
    spec = {
        "title": f"{sid} ({seg['id']})", "fps": fps, "resolution": f"{w}x{h}", "engine": "eevee", "transparent": True,
        "clock": round((f0 - pre_f) / fps, 4), "master_start_frame": f0 - pre_f,
        "look": {"view": "Standard", "exposure": -0.9},
        "set": {"kind": "lecture_hall", "walls": False, "table": table, **({"door": {}} if opening else {}),
                **({"lights": "off"} if opening and opening.get("lights") == "switch" else {})},
        "cast": [{"id": "presenter", "from": str(rel), "posture": "stand",
                  "at": {"door": opening["behind"]} if opening else {"board": list(st["stand"])},
                  "facing": 0 if opening else st["facing"]}],
        "voice": [{"captions": str(cap), "at": round(at, 4)}],
        "shots": ([{"id": "pre", "dur": round(pre, 4), "type": "fixed", "cam": "board", "lens": st["lens"],
                    "window": list(win), "built_only": True}] if pre_f else [])
        + [{"id": sid, "dur": round(dur, 4), "type": "fixed", "cam": "board", "lens": st["lens"], "window": list(win)}],
        "beats": sorted(beats, key=lambda b: b["at"]),
    }
    return sid, spec


def room_spec(lec: dict, first: dict) -> dict:
    """The opening's room shot: the first clip's spec (the same cast, beats and clock, so the pose at the cut is
    the same in both) with the hall drawn and dressed, the board showing the titled board, and the camera coming
    down the hall from the back to land on the board camera's framing as the walk ends. The first clip's own shot
    stays on as "tail", built and not rendered, so both specs run the same length."""
    o, st = lec["opening"], lec["staging"]
    room = json.loads(json.dumps(first))
    pre, clip = room["shots"][0], room["shots"][-1]
    room.update({"title": "E1 (the opening)", "resolution": "1920x1080", "transparent": False})
    room["set"] = {**room["set"], "walls": True, "hall": o["hall"],
                   "board": {"image": "../../public/board/title.png"}}
    c = o["camera"]
    room["shots"] = [{"id": "E1", "dur": pre["dur"], "type": "move",
                      "from": {"cam": {"hall": c["cam"]}, "aim": {"hall": c["aim"]}, "lens": c["lens"]},
                      "to": {"cam": "board", "lens": st["lens"]},
                      # (after a stop at the switch, the camera holds the hall as its lights come on)
                      "start": "last_walk" if o.get("lights") == "switch" else "walk", "arrive": "walk"},
                     {"id": "tail", "dur": clip["dur"], "type": "fixed", "cam": "board", "lens": st["lens"], "built_only": True}]
    return room


def cmd_specs(project: Path, args) -> list[dict]:
    lec, tl = load(project), read_timeline(project)
    words, spans, notes = Words(tl), clip_spans(lec, tl), []
    if [s["id"] for s in tl["segments"]] != [s["id"] for s in lec["segments"]]:
        raise LectureError("timeline.json is for other segments than lecture.json: run `timeline` again")
    d = project / "blender" / "presenter"
    d.mkdir(parents=True, exist_ok=True)
    index = []
    for i in range(len(tl["segments"])):
        sid, spec = spec_for(project, lec, tl, words, spans, i, notes)
        (d / f"{sid}.json").write_text(json.dumps(spec, indent=1) + "\n")
        if i == 0 and lec.get("opening"):
            (d / "E1.json").write_text(json.dumps(room_spec(lec, spec), indent=1) + "\n")
            index.append({"id": "E1", "room": True, "from": 0, "frames": spans[0][0], "pre": 0})
            print(f"E1: the opening, master frames 0 to {spans[0][0]} (the room shot; {sid} walks in with it)")
        pre = round(spec["shots"][0]["dur"] * tl["fps"]) if spec["shots"][0]["id"] == "pre" else 0
        index.append({"id": sid, "segment": tl["segments"][i]["id"], "from": spans[i][0], "frames": spans[i][1] - spans[i][0],
                      "pre": pre})
        print(f"{sid}: master frames {spans[i][0]} to {spans[i][1]} ({spans[i][1] - spans[i][0]} frames)"
              + (", starts inside the carried turn" if carried(lec, tl, spans, i) else ""))
    (d / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    for n in notes:
        print("  " + n)
    return index


# ------------------------------------------------------------------------------------------------ check, render
def blender_tool(*a, log: Path) -> int:
    cmd = ["uv", "run", "--project", str(ROOT), "python", str(ROOT / "tools" / "blender_render.py"), *a]
    with log.open("w") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def ids_of(project: Path, wanted: str | None) -> list[dict]:
    index = json.loads((project / "blender" / "presenter" / "index.json").read_text())
    if wanted:
        keep = {x.strip() for x in wanted.split(",")}
        index = [c for c in index if c["id"] in keep]
    return index


JOIN_PROBE = r'''
import bpy, json, sys
import numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
sc = bpy.context.scene
sc.frame_set(int(a[0]))
dg = bpy.context.evaluated_depsgraph_get()
out = {}
for o in bpy.data.objects:
    if o.type == "MESH" and o.name.endswith("_body") and o.vertex_groups.get("neck"):
        ev = o.evaluated_get(dg)
        m = ev.to_mesh()
        co = np.empty(len(m.vertices) * 3)
        m.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        mw = np.array(o.matrix_world)
        np.save(a[1] + ".npy", co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
    elif any(o.name.split(".")[0].endswith(k) for k in ("_head", "_pupil", "_lid", "_mouth", "_brow", "_neck")):
        out[o.name] = list(o.matrix_world.translation) + list(o.matrix_world.to_quaternion())
json.dump(out, open(a[1], "w"))
'''


def join_report(project: Path, index: list[dict], logs: Path) -> list[dict]:
    """Do clips that join meet in the same pose? Each clip's scene is asked for its body (every vertex), its head,
    pupils, lids, brows and mouth on the frame the next clip starts (one past its own last frame), and the next
    clip for its first frame. The worst vertex and the worst face part are reported."""
    import numpy as np
    from blender_render import installed_blender  # noqa: E402
    blender = installed_blender()
    out = []
    for a, b in zip(index, index[1:]):
        sa = project / "blender" / "presenter" / "out" / a["id"] / "scene.blend"
        sb = project / "blender" / "presenter" / "out" / b["id"] / "scene.blend"
        if not (sa.exists() and sb.exists()):
            continue
        ja, jb = logs / f"join-{a['id']}-out.json", logs / f"join-{b['id']}-in.json"
        for blend, frame, dst in ((sa, a.get("pre", 0) + a["frames"] + 1, ja), (sb, b.get("pre", 0) + 1, jb)):
            subprocess.run([blender, "-b", str(blend), "--python-expr", JOIN_PROBE, "--", str(frame), str(dst)],
                           capture_output=True, text=True)
        pa, pb = json.loads(ja.read_text()), json.loads(jb.read_text())
        va, vb = np.load(str(ja) + ".npy"), np.load(str(jb) + ".npy")
        body = float(np.linalg.norm(va - vb, axis=1).max()) if va.shape == vb.shape else 9.9
        face, what = 0.0, "body"
        for k in pa.keys() & pb.keys():
            m = math.dist(pa[k][:3], pb[k][:3])
            if m > face:
                face, what = m, k
        worst = max(body, face)
        out.append({"join": f"{a['id']}->{b['id']}", "frame": b["from"], "body_m": round(body, 4), "face_m": round(face, 4),
                    "worst": "body" if body >= face else what, "passed": worst <= 0.004})
    return out


def cmd_check(project: Path, args) -> bool:
    index = ids_of(project, args.ids)
    logs = project / "blender" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tl = read_timeline(project)
    ok, boards = True, []
    for c in index:
        spec = project / "blender" / "presenter" / f"{c['id']}.json"
        out = spec.parent / "out" / c["id"]
        t = time.time()
        code = blender_tool("check", str(spec), "--out", str(out), "--json", log=logs / f"check-{c['id']}.log")
        res = json.loads((out / "checks.json").read_text()) if (out / "checks.json").exists() else {}
        if res.get("spec_error"):
            print(f"{c['id']}: SPEC ERROR {res['spec_error']}")
            ok = False
            continue
        b = res.get("board") or {}
        if not c.get("room"):  # (the inset's crop and the board share are the presenter's clips')
            boards.append(b)
        for sh in res.get("shots", []):  # (the opening's move: where it sets off and lands, and how it lands)
            if sh.get("type") == "move":
                print(f"  {sh['id']}: sets off at frame {sh['sets_off']}, lands at {sh['lands']}; lens {sh['lens'][0]} to "
                      f"{sh['lens'][1]} mm; nearest the set {sh['nearest_set_m']} m; landed: {sh.get('landing')}")
        skin = res.get("skin") or {}
        print(f"{c['id']}: {'pass' if res.get('passed') else 'FAIL'} in {time.time() - t:.0f} s; board {100 * b.get('share', 0):.1f}%"
              f", stretch {skin.get('stretch', '-')}" + ("" if res.get("passed") else f" | {'; '.join(res.get('body_problems', []))}"))
        ok &= bool(res.get("passed")) or code == 0
    if boards and boards[0].get("head_px"):  # (where the head is, for the inset's crop; the box they ever reach)
        box = [min(b["box_px"][0] for b in boards), min(b["box_px"][1] for b in boards),
               max(b["box_px"][2] for b in boards), max(b["box_px"][3] for b in boards)]
        tl["presenter"] = {"head_px": boards[0]["head_px"], "box_px": box,
                           "share": max(b["share"] for b in boards), "scale": boards[0].get("scale")}
        (project / "timeline.json").write_text(json.dumps(tl, indent=1) + "\n")
    if len(index) > 1 and not args.no_joins:
        for j in join_report(project, index, logs):
            print(f"join {j['join']} at frame {j['frame']}: {'same pose' if j['passed'] else 'MOVES'} "
                  f"(body {1000 * j['body_m']:.1f} mm, face {1000 * j['face_m']:.1f} mm; worst: {j['worst']})")
            ok &= j["passed"]
    return ok


def cmd_render(project: Path, args) -> None:
    index = ids_of(project, args.ids)
    logs = project / "blender" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    frames = sum(c["frames"] for c in index)
    lec = load(project)
    st = lec["staging"]
    win = st["window"]
    pixels = (win[2] - win[0]) * 1920 * (win[3] - win[1]) * 1080 * st["density"] ** 2
    per = 0.81 * pixels / (864 * 836)  # (measured: 0.81 s a frame for Mara's 864x836 clips on an Apple M4; a heavier body is slower)
    room = sum(c["frames"] for c in index if c.get("room"))
    # (measured: the dressed hall at 1920x1080 is about ROOM_PER s a frame on an Apple M4)
    est = (frames - room) * per + room * ROOM_PER
    print(f"{len(index)} clips, {frames} frames: about {est / 60:.0f} min on an Apple M4, one at a time "
          f"(longer for a body of many faces)")
    dest = project / "public" / "blender"
    for c in index:
        spec = project / "blender" / "presenter" / f"{c['id']}.json"
        t = time.time()
        a = ["render", str(spec), "--clips", *([] if c.get("room") else ["--alpha"]), "--no-sound", "--force",
             "--to", str(dest), "--shots", c["id"]]
        if args.draft:
            a += ["--res", args.draft]
        code = blender_tool(*a, log=logs / f"render-{c['id']}.log")
        print(f"{c['id']}: exit {code} in {(time.time() - t) / 60:.1f} min")


def cmd_bg(project: Path, args) -> None:
    """The part of the bare board the presenter's clips cover, for the inset to draw behind their cropped clip."""
    lec = load(project)
    win = lec["staging"]["window"]
    x, y = round(win[0] * 1920), round((1 - win[3]) * 1080)
    w, h = round((win[2] - win[0]) * 1920), round((win[3] - win[1]) * 1080)
    src, dst = project / "public/board/bg.png", project / "public/board/bg-presenter.png"
    if not src.exists():
        raise LectureError(f"{src} not found: npx remotion still BoardBg public/board/bg.png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"crop={w}:{h}:{x}:{y}", str(dst)], check=True)
    print(f"{dst}: {w}x{h} from ({x}, {y})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Rules and staging: lecture.json \"rules\" and \"staging\"; the defaults are in this file (RULES, STAGING).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("timeline", "audio + captions -> timeline.json"), ("specs", "lecture.json -> one clip spec a segment"),
                           ("check", "check every clip (one Blender at a time), then the joins"),
                           ("render", "render the clips into public/blender, one at a time"),
                           ("bg", "cut the inset's backdrop out of the bare board")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("project", type=Path)
        if name in ("check", "render"):
            p.add_argument("--ids", help="only these clips, comma-separated (P02,P03)")
        if name == "check":
            p.add_argument("--no-joins", action="store_true", help="skip the pose check at the joins")
        if name == "render":
            p.add_argument("--draft", help="render smaller, e.g. 432x420, to judge the motion")
    args = ap.parse_args(argv)
    project = args.project.resolve()
    sys.path.insert(0, str(ROOT / "tools"))
    if args.cmd == "timeline":
        cmd_timeline(project, args)
    elif args.cmd == "specs":
        cmd_specs(project, args)
    elif args.cmd == "check":
        sys.exit(0 if cmd_check(project, args) else 1)
    elif args.cmd == "render":
        cmd_render(project, args)
    elif args.cmd == "bg":
        cmd_bg(project, args)


if __name__ == "__main__":
    main()
