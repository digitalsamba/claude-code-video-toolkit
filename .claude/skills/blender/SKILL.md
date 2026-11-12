---
name: blender
description: 3D scenes, character acting and product shots from a shot spec, rendered in Blender, as a finished film, as clips for any other build (moviepy, a video job, an editor), or as elements of a Remotion video (transparent characters over a scene, a Remotion render on a 3D laptop, phone or floating card that the camera pulls back from, an imported 3D model on a studio stage). A rigged cartoon cast described in JSON (build, face, hair, clothes, a seed), sets, shots by type, acting beats in seconds or on narration words, every camera checked by geometry, sound placed on the motion. Use when a video needs a consistent directable character (parody, explainer gag, reaction shot), a 3D reveal or element in a Remotion video, a device or product shot of an app or a model, or exact acting timing that AI video clips can't hold. Triggers include Blender, 3D, character animation, rigged character, cast, shot spec, acting, camera angles, transparent clip, alpha, pull out from the screen, phone mockup, device mockup, product shot, turntable, orbit, glTF, GLB, BlenderClip.
---

# Blender shot specs

`tools/blender_render.py` turns a JSON **shot spec** into 3D footage: a whole film, one clip per shot for another
build, or both from the same spec. The spec says *what happens*; `lib/blender/`
decides *where things go*: cameras are placed by shot type round their subject and checked (in frame, nothing in
the way) before they are kept; acting is verbs (`expression`, `pose`, `shake_head`) rather than keyframes; every
moment worth a sound is exported with its time, so the sound is on the motion by construction.

Three worked examples:

- `examples/blender-product-shot/spec.json`: a studio stage with no character: a picture on a floating card and
  on a phone, and an imported model, with `reveal`, `orbit` and `product` shots.
- `examples/blender-disk-panic/spec.json`: a film from one spec (22.5 s, 8 shots, 37 beats).
- `examples/blender-in-remotion/`: Blender clips inside a Remotion video. A terminal scene cuts to itself on a 3D
  laptop and the camera pulls back; then a transparent character acts beside the list of beats it is acting.

```bash
uv run tools/blender_render.py doctor                                         # Blender found? s/frame on this machine
uv run tools/blender_render.py check  examples/blender-disk-panic/spec.json   # ~5 s: checks + stills + contact sheet
uv run tools/blender_render.py render examples/blender-disk-panic/spec.json   # ~15 min (M4, EEVEE, 1080p)
uv run tools/blender_render.py render spec.json --res 640x360                 # a draft to judge motion, 0.35 s/frame
uv run tools/blender_render.py render spec.json --engine cycles               # physical light; ~10x slower (17 s/frame)
uv run tools/blender_render.py render spec.json --clips --alpha --to projects/my-video/public/blender
uv run tools/blender_render.py open   spec.json                               # the built scene in Blender
```

Output in `<spec dir>/out/` (`out/<name>/` for a spec not called `spec.json`): `checks.json`, `contact_sheet.png`, `stills/`, `events.json`, `scene.blend` (open it
in Blender and scrub), and after `render` the `.mp4` named after the spec's folder, or `clips/` with `--clips`.

Requires Blender 4.2 or later. It has been run on 5.3 (macOS) and 4.2 LTS (Linux, in a container); Windows is
supported by the code and not yet run. Optional component: nothing else in the toolkit needs it.
Install, timings and troubleshooting: `docs/blender.md`.

## What the render is for

These are uses of one spec format, not separate modes: the same spec renders as a film and as clips, and a video
can mix them (a product shot between two Remotion scenes, a character film with a Remotion title card).

| The video is | Do | Then |
|---|---|---|
| A 3D short on its own (a gag, a parody shot) | `render spec.json` | `qc.py --render`, add music with `addmusic.py` |
| A Remotion video with a 3D character in some scenes | `render --clips --alpha --to <project>/public/blender`, set kind `desk` | `<BlenderClip>` over the scene's field |
| A Remotion video that turns out to be on a screen in a room | render the scene with Remotion, name it in `set.screen.video`, a `pull_out` shot, `render --clips` | hard cut from the flat scene to the clip |
| An app, a UI or a product shown as an object | set kind `studio` with a `phone`, `card` or `model` prop; `reveal`, `orbit`, `product` shots | a clip between scenes, or `--alpha` over the scene's field |
| A moviepy build, a video job or an edit in another tool that wants a 3D shot | `render --clips` (a picture-only `.mp4` per shot, its effects as a `.wav`; `--prores` for editors) | `VideoFileClip`, a `file` scene in `video_job.py`, `cutdown.py` for 9:16 |
| Narrated, with acting on the words | a `voice` block and `"at": "cue:..."` | the spec follows the voice when it is re-timed |

For Remotion, set the spec's `"fps": 30` (the composition's): a clip plays frame for frame.

**When to suggest it.** While planning any video, if `blender_render.py doctor --no-bench` exits 0, offer a
Blender shot where one fits: a character reacting, a reveal, a product or device shot. If Blender isn't
installed, don't propose it unless the user asks for 3D; then say what it needs (`docs/blender.md`).

## The loop (do this, in this order)

1. **Write the spec** from the brief or narration (below: *From a brief to beats*).
2. **`check`** and read the report. Every shot prints PASS/FAIL, its exposure, and any camera fix the solver made
   ("swung +24 deg: nominal had view blocked by dev_hood"). Exit 1 on a failure, 2 on a spec error.
3. **Look at `contact_sheet.png`** (read the PNG). Geometry can't judge everything: a hand over the key text, an
   expression that reads wrong, a beat landing in the wrong shot. Green bar = passed checks, red = failed.
4. **Fix in the spec, not in Blender**: change a shot's `type`, `az`, `dist`, `dur`, or a beat's `at`. If the
   solver had to move a camera, consider writing its fix into the spec (`"az": ...`) so the intent is explicit.
5. Repeat 2–4 until it passes and reads right, then **`render`**. Check the result with
   `uv run tools/contact_sheet.py out/<name>.mp4` (cuts are detected), and `uv run tools/qc.py` for loudness.

`check` builds at the film's own shape (the spec's `resolution` scaled down), so a 9:16 spec is solved and judged
in 9:16. Never hand-edit `scene.blend` as the source of truth; it is rebuilt from the spec every run.

## Spec format

```json
{
  "title": "Disk Panic", "fps": 24, "resolution": "1920x1080", "engine": "eevee",
  "look": {"exposure": -0.9, "look": "AgX - Punchy"},
  "set":  {"kind": "desk_room", "palette": {...}, "screen": {...}},
  "cast": [{"id": "dev", "kind": "clay_person", "colors": {"top": "#F26B1F"}}],
  "shots": [{"id": "S1_wide", "dur": 3.0, "type": "wide"}, ...],
  "beats": [{"at": "S1_wide+1.6", "do": "blink"}, ...],
  "score": [{"from": 0, "to": "S2_alert+0.5", "mood": "calm"}, ...]
}
```

Also at the top level: `"transparent": true` (the same as `--alpha`) and `"voice"` (below).

**Acting takes the same time at any `fps`.** A blink, a typing rhythm and a head shake are written in seconds
inside the rig, so a 24 fps film and a 30 fps clip for Remotion act alike.

**A mistake in a spec is named, with the choices.** An unknown face, pose, shot type, subject or cast member is a
spec error (exit 2) that lists what there is, before anything is rendered.

**Times** are seconds: a number (absolute), or relative to a shot: `"S3"` (its start), `"S3+0.4"`,
`"S3.end-0.2"`, `"end"`. Write beats relative to the shot they play in, so changing one shot's `dur` keeps the
acting in step. Beats are applied in time order; a beat's length is `dur` (seconds) or `until` (a time).

**Times from the narration.** List the caption files `tools/captions.py` wrote, each with where its audio starts,
and a time can be the moment a word is said:

```json
"voice": [{"captions": "../public/audio/scenes/02_alert.captions.json", "at": "S2_alert+0.3"}],
"beats": [{"at": "cue:disk is full", "do": "alert"},
          {"at": "cue:full#2-0.1", "do": "expression", "face": "shocked"}]
```

`cue:` takes a word or a phrase (case and punctuation ignored), `#2` for its second occurrence, and an offset. A
phrase that isn't in the narration is a spec error that prints what the narration does say, as `cue()` does in
Remotion. Paths are relative to the spec.

### Cast

A cast member is a description, and every field is a named choice or a hex colour, so an agent can write one
from a sentence ("a woman in her forties with a grey bob, glasses and a green shirt") or from a photo:

```json
{"id": "maya", "kind": "clay_person", "seed": 12,
 "body":    {"build": "slim", "skin": "brown"},
 "face":    {"shape": "oval", "eyes": "brown", "brows": "thin", "nose": "small", "beard": "none"},
 "hair":    {"style": "bob", "color": "grey"},
 "clothes": {"top": "shirt", "top_color": "#2E7D5B", "pants": "#252B45", "shoes": "#F0F0EC"},
 "accessories": ["glasses"]}
```

| Field | Choices |
|---|---|
| `body.build` | `average`, `slim`, `broad` |
| `body.skin` | `pale`, `light`, `tan`, `brown`, `dark`, `deep`, or a hex colour |
| `face.shape` | `round`, `oval`, `square` |
| `face.eyes` | `brown`, `hazel`, `green`, `blue`, `grey`, or a hex colour |
| `face.brows`, `face.nose` | `medium`, `thin`, `thick`; `round`, `small`, `long` |
| `face.beard` | `none`, `moustache`, `full` |
| `hair.style` | `tufts`, `short`, `bob`, `long`, `bun`, `curly`, `bald` |
| `hair.color` | `black`, `dark`, `brown`, `chestnut`, `red`, `blonde`, `grey`, `white`, or a hex colour |
| `clothes.top` | `hoodie`, `tee`, `shirt`; `top_color`, `pants`, `shoes` are hex colours |
| `accessories` | any of `glasses`, `headphones`, `earrings` |

- **Left out with no `seed`:** the defaults, which is the character the examples use.
- **Left out with a `seed`:** drawn from the seed. `{"id": "extra", "seed": 41}` is a whole person, the same one on
  every machine, and another seed is someone else.
- **A standing cast.** Keep a character in its own file and name it: `{"id": "lead", "from":
  "../../kits/acme/cast/maya.json", "clothes": {"top_color": "#B23A48"}}`. Fields beside `from` override the
  file, so one video can change the costume and keep the person. `kits/example/cast/maya.json` is one.
- From a photo: choose the nearest named choice for each field and sample the hex colours from the picture. The
  result is a caricature in the toolkit's style, not a likeness.
- Every character has the same rig, so every verb works on every character. Check a new one with a `closeup` and
  a `medium` shot before writing the beats.
- **Where they start.** A cast member sits at the desk unless told otherwise: `"posture": "stand"`, `"at": [x, y]`
  (a spot on the floor, metres) and `"facing"` (degrees: 0 faces -Y, toward the desk; positive turns to their own
  left) put them on their feet anywhere in the room.

**A fitted model (experimental).** `"model": "lead.blend"` in a character file swaps the clay body for a
generated 3D character fitted to the rig by `lib/blender/fit_model.py` (an A-pose mesh from an image-to-3D
service, plus a `fit.json` of its joints and face). The model keeps its own shape: it is bound in its A-pose and
the rig poses it, so the same verbs drive it, and the chair, desk and laptop are placed for its legs and arms (seat
at its knee height, keyboard under its hands). `glasses` is the one accessory a model takes. Its face is made from its own marks (a lower
lip, teeth and lids on the model's lip line and eyes, with the cheeks and chin moving for a smile or an open
mouth), and with a second model of the same character with open hands the fit gives it fingers: hand shapes, a
grip on a door handle, hands laid flat on the desk. The steps and their costs are in `docs/blender.md`.

### Shots

| `type` | Subject | Use for |
|---|---|---|
| `wide` | character (3/4 front) | establishing: who, where |
| `front_wide` | character (straight on) | endings, titles above the head |
| `medium` | character (waist up) | gesture: hands on head, shrug |
| `closeup` | character (face) | the reaction; emotion lives here |
| `side` | character (profile) | effort, leaning in, typing hard |
| `over_shoulder` | `"subject": "screen"` | what they see: the alert lands |
| `insert` | `"subject": "screen"` | the detail: a bar draining, a number changing |
| `screen_fill` | `"subject": "screen"` | the screen covers the frame edge to edge, square on |
| `pull_out` | `"subject": "screen"` | starts as `screen_fill`, holds, then travels back round the shoulder to a side view of the desk |
| `push_in` | `"subject": "screen"` | `pull_out` reversed: ends on the screen, ready to cut to the flat scene |
| `full` | character, on their feet | head to toe; `"follow": true` travels with a walk |
| `feet` | character, on their feet | a low insert on the feet, travelling with them: an entrance before the face |
| `tilt_up` | character, on their feet | starts on the feet, holds, tilts up to the face (`hold` seconds at each end) |
| `product` | a studio prop | a three-quarter view of the prop, pushing in a little |
| `orbit` | a studio prop | the camera travels round the prop from `az` to `az_to` |
| `reveal` | a studio `phone` or `card` | starts on its screen edge to edge, holds, then pulls back to the `product` view |

Overrides per shot: `az` (degrees round the subject; 0 = straight in front, + = the subject's left), `dist` (m),
`h` (camera height above the target, m), `lens` (mm), `push` (fraction of the distance closed during the shot),
`shake` (handheld jitter, m; ~0.012 for urgency), `side` (`"left"`/`"right"` flips `az`).

The solver tries the nominal placement, then pulls back, swings ±12/24/36°, and rises, keeping the first that
passes at the shot's start, middle and end. What it changed is in the report.

A character who has left the chair is filmed where they are at the middle of the shot, facing the way they face
then, and the solver also tries the far side of them when a wall is in the way. `"follow": true` on any character
shot makes the camera travel with them. `"reveal": true` is for a shot the subject enters (through a door): it is
judged on its last frame only. `tilt_up` is judged where it ends, on the face.

The three moves take `az`, `elev` (degrees) and `dist` (m) for their far end, `hold` (seconds on the full screen
before moving, default 0.4) and `lens`. They are checked at 13 points along the way, also for the lens passing
within 0.3 m of a character ("lens 0.16 m from dev_body"), which is what makes a head fill the frame mid-move.
Their contact-sheet still is the far end. They pass in 16:9, 9:16 and 1:1.

`product`, `orbit` and `reveal` take their distance from the prop's size, so a 15 cm phone and a 2 m chair fill
the frame alike: `fill` (0 to 1, the share of the frame the prop takes at its widest moment), `az`, `az_to`,
`elev`, `lens`, or `dist` to set the distance by hand. A prop with a screen also reports `screen_readable`, the
share of the shot in which the screen is within 60 degrees of facing the lens; under half is a failure ("narrow
the orbit").

### Beats (`do`)

Character (`who` defaults to the first cast member):

| verb | fields | |
|---|---|---|
| `type` | `speed`: `calm`/`frantic`, `until` | hands typing; a click per key in the sound |
| `pose` | `pose`, `dur` | `type`, `rest`, `hands_on_head`, `arms_up`, `hands_behind_head` |
| `expression` | `face`, `dur` | `neutral`, `happy`, `content`, `shocked`, `worried`, `determined`, `relieved`, `grin` |
| `look` | `target`, `dur` | `screen`, `down`, `ahead`, `camera`, `up`, or degrees of head pitch |
| `lean` | `amount`, `dur` | `forward`, `none`, `back`, `far_back`, `recoil`, or degrees |
| `head_bob` / `shake_head` | `until` | idle nod to the work / "no no no" (whoosh each turn) |
| `gulp`, `blink`, `tap` | | single moments (`blink` takes `dur`, or `frames` counted at 24 fps) |
| `sweat`, `tremble` | `until` | a drop slides down / pupils jitter |
| `celebrate` | `until` | arms up, wiggling |
| `hold_arms` | | freeze the arms where they are |
| `stand_up` / `sit_down` | `dur` | off the chair onto their feet (the head lifts, hands to the knees, the chest comes forward over the feet, then up, the arms dropping last) / onto the chair behind them (a look down, the pelvis back before it goes down, hands to the knees, a small settle). Stand them a thigh's length in front of the seat first. About 1.7 s each reads well; under a second looks hurried |
| `walk` | `to`: `[x, y]`, `facing`, `until`, `curve`, `heading` | walk to a spot and stop, facing `facing` degrees. They turn on the spot first when the way to go is more than 30 degrees off, and again at the end. With no `until` or `dur` it takes what the steps need. A footstep in the sound per step. `"curve": true` walks a bend in place of a line and two turns: the feet land on the bend, the body banks into it and the head looks along it; `heading` is the way they are going as they arrive (degrees, like `facing`) |
| `turn` | `facing`, `dur` | turn on the spot, stepping round: the eyes and head go first, the chest follows, and it swings a little past at the end |
| `wink` | `side`: `left`/`right` | one eye, the brow dipping with it |
| `look_at` | `target`: `[x, y, z]`, `camera` or `screen`, `dur` | head and eyes to a point: the neck turns most of the way, the eyes the rest |
| `glance` | `yaw`, `pitch`, `dur` | the eyes alone, in degrees (up to about 25) |
| `reach` | `target`: `[x, y, z]`, `hand`, `dur` | a hand to a point in the room; an error says how far short it is |
| `hand` | `shape`: `open`, `relaxed`, `grip`, `flat`, `point`, `fist`; `hand`: `left`/`right`; `dur` | the fingers take a shape (both hands without `hand`). Only a fitted model whose fit found fingers has them: the clay cast's mittens are left as they are. Poses and verbs set shapes themselves (flat on the desk, a grip on a door handle, relaxed at the sides), so this is for a point or a fist |

Poses also has `down` (arms at the sides), which a standing character starts in; arms swing on a walk only from
`down`. A posture verb used in the wrong state is a named error ("dev is sitting: walk needs them standing
(stand_up first)"). `ik_max_error_m` in the report covers feet as well as hands, and `ik_worst` names the limb
and the second when it is over a centimetre: usually a step or a reach that was too far.

**Contact is checked for anyone on their feet.** The report measures, on every frame, what a still can't show:

| in `checks.json` | fails over | |
|---|---|---|
| `under_floor_m` | 1.5 cm | any of the body below the floor |
| `hand_in_body_m` | 3 cm | a hand inside the torso, head or legs |
| `through_m` | 3 cm | the body inside the seat, chair back, desk, laptop or door (a seat cushion gives 3 cm) |
| `foot_slide_m` | 3 cm | a foot travelling along the floor while it stands flat on it |
| `touch_miss_m` | 3 cm | a hand off the thing it was put on (the door handle, the desk's edge) |
| `grip_open` | over 0.75 | hands with fingers: how open a hand still is round what it holds (fingertips' distance from it over a finger's length; a flat hand is about 1) |
| `hands_apart_m` | under 0.5 cm | hands with fingers: the nearest the two hands come to each other |
| `palm_tilt_deg` | 35 degrees | hands with fingers: a hand resting over a table top, tipped from palm down |

A thin thing (a door leaf, a desk top) counts as gone through once anything is half way into it, and `through`
says how long it lasted ("from 14.50 s to 20.67 s"). A hand is checked at its palm and at every fingertip.

A failure is said in words in `body_problems` ("left hand 9 cm inside the body at 5.12 s") and each number gets an
`_at` with the frame. `moves` lists every posture verb with how far apart in time its parts start and stop and how
different left is from right: all zeros would be a machine. To look at a move yourself, render the shot as a clip
with `--keep-frames` and tile every second or third frame.

**More that `check` reports**, each failing the check in words when it is wrong:

- `walks`: each walk's bow off the straight line, its turn, and how far the body leans into a bend.
- `surfaces_fight` and `set_problems`: two objects of different materials with faces in one plane that a camera
  sees. The renderer draws one or the other from frame to frame, so the strip flickers. Move one, or end it
  inside the other. Graphics layered a little over a screen do not count.
- For a fitted model: `face` (`mouth_share`, the mouth's width against the face's, fails under 0.18; how far the
  lip stands off the face; how far its corners travel between a frown and a smile) and `skin`
  (`leg_above_hips`, vertices above the hips that move with a leg and tear a hem; `stretch`, fails over 4).

**A render is checked too.** `render` writes `still_points.json` (points of the set that stand still, and where
each is in every frame) and reads them back out of the picture: a still point that jumps for one frame and comes
back is flicker. Five such frames on one object in a shot fail (`render_checks.json`, and the tool exits 1);
fewer are listed as notes. It runs on the film and on `--clips`.

Set (`desk_room`): `alert` (popup + red screen light), `work` (`until`: bar drains bad → mid → good),
`done` (the "after" line pops in), `open_door` / `close_door` (`dur`, `angle`, and `by`: a cast member whose hand
takes the handle and goes with the door as far as the arm reaches; an error says how far short they stand),
`pull_in` (the chair and whoever sits on it come in to the desk: the feet walk it in with short steps and, once
the desk is in reach, the hands take its edge and pull). The screen's words and numbers come from `set.screen`
(`title`, `subtitle`, `before`, `after`, `fill_before`, `fill_after`, `alert` [2 lines], `working`).

Effects: `confetti` (`count`, `colors`: a list of hex colours, for the brand's), `title` (`text`, `color`): 3D text above the character, wobbles to the end.

### Sets

`desk_room` takes `"door": {}` (a doorway in the wall behind the chair, at `x` 2.6 unless given, as tall as the
character needs and with the handle where their hand falls, opening onto a dim hall) and `"chair_out"` (metres the
chair starts pulled back from the desk). An entrance: the character starts in the hall at `[2.6, 3.5]` (in reach
of the handle), `open_door` with `"by"`, `walk` in, `walk` to a thigh's length in front of the pulled-out chair, `sit_down`, `pull_in`.

| `kind` | |
|---|---|
| `desk_room` (default) | a room: walls, window, posters, rug, desk, chair, laptop, mug, plant. `palette` recolours it (`wall`, `wall2`, `floor`, `desk`, `accent1`, `accent2`, `rug`) |
| `desk` | the desk, chair and laptop with no room, for `--alpha`: the character sits on whatever the Remotion scene puts behind it |
| `studio` | a product stage: a backdrop curving from floor to wall, soft lights sized to the props, no character and no beats. `palette.backdrop` colours it; with `--alpha` the backdrop is left out |

**Studio props** (`set.props`, each with an `id` that shots name as their `subject`):

| `kind` | |
|---|---|
| `phone` | a phone standing on the stage; `screen.video` or `screen.image` is its picture (portrait or landscape, the phone follows the picture's shape) |
| `card` | a thin floating panel showing a picture or a video, `width` in metres (default 0.6): a UI screenshot or a Remotion scene with depth |
| `model` | a `.glb`, `.gltf`, `.obj`, `.fbx` or `.stl` file (`file`), scaled so its longest side is `size` metres (default 0.3) and set down on the floor |

Props stand side by side unless given `"at": [x, y]` (metres); `"turn"` (degrees) rotates one. Without a
picture, a phone or a card shows the palette's `accent`, so a layout can be checked before the render exists.
Screens take `skip_frames`, `at` and `brightness` as the laptop does, and want `"view": "Standard"` for the same
reason.

```json
"set": {"kind": "studio", "palette": {"backdrop": "#DCE2EE"},
        "props": [{"id": "ui", "kind": "card", "screen": {"video": "../out/scene.mp4"}},
                  {"id": "app", "kind": "phone", "screen": {"image": "app.png"}, "at": [0.42, -0.12], "turn": -18}]},
"shots": [{"id": "P1", "dur": 4, "type": "reveal", "subject": "ui"},
          {"id": "P2", "dur": 3, "type": "orbit", "subject": "app", "az": -35, "az_to": 35}]
```

An imported model keeps its own materials. Check its licence before it goes in a video, as with any asset.

**A Remotion render on the laptop.** `set.screen.video` (or `image`) replaces the storage UI with a picture, 16:9
and unlit so its colours survive:

```json
"look": {"view": "Standard", "exposure": -0.9},
"set": {"kind": "desk_room", "screen": {"video": "../out/screen.mp4", "skip_frames": 150}}
```

- The video advances one frame per scene frame, so render it at the spec's `fps`: a video at another rate is a
  spec error. `skip_frames` starts it part-way in; `at` (a time) delays it; past its end it holds the last frame.
  `brightness` scales it.
- Use `"view": "Standard"` when a flat scene cuts to the same scene on the laptop: the default AgX view compresses
  colour, and the cut would show. With Standard the first frame of a `pull_out` matches the flat frame.
- With a picture on the screen, `alert`, `work` and `done` still tint the screen's light and make their sounds,
  but draw nothing.

**Score**: `calm`, `tension`, `resolve` sections. This is simple synthesis for a draft; for the real bed run
`tools/music_gen.py` at the film's length and mix it under with `tools/addmusic.py`.

## From a brief to beats

A short gag is a turn: **normal → problem → reaction → effort → fix → payoff**. Give each its shot:

1. *Normal*: `wide`, 2.5–3 s, calm idle (`type` calm + `head_bob`), `calm` score.
2. *Problem*: `over_shoulder` on the screen, the `alert` ~0.5 s in; hold the character still (`hold_arms`).
3. *Reaction*: `closeup`, **slow**: the face changes 0.1 s in, then let it sit (`tremble`, `sweat`, a `gulp`
   around 2 s). Comedy is in the hold; 3–3.5 s.
4. *Escalation*: `medium` for the big gesture (`hands_on_head` + `shake_head`), `tension` score.
5. *Effort*: `side` with `shake`, `lean` forward, `determined`, `type` frantic; cut to an `insert` of the
   screen for `work` → `done`.
6. *Relief*: `closeup`, `relieved`, a slow `blink`, `lean` back.
7. *Payoff*: `front_wide`: `celebrate`, `grin`, `confetti`, `title`; end on a settle (`hands_behind_head`).

Rules of thumb: a beat lands 0.1–0.5 s *after* a cut, not on it; one idea per shot; cut to the screen whenever
the screen changes; keep expressions on screen at least 0.5 s; vary shot type between neighbours.

When the video has narration, take the times from the voice: a `voice` block and `cue:` times (above), one beat
on the word it illustrates.

## Blender clips in a Remotion video

```bash
uv run tools/blender_render.py render spec.json --clips --alpha --to projects/my-video/public/blender
uv run tools/blender_render.py render spec.json --clips --shots B1_act --to ...    # one shot again
```

One clip per shot lands in the project with a `clips.json` manifest: `.webm` (VP9 with alpha) with `--alpha`,
`.mp4` without, and a `.wav` of the shot's own effects. `--prores` adds ProRes 4444 for other editors.

Several specs can render `--to` one folder (a room and a transparent desk). Each clip in `clips.json` carries its
own `fps`, `resolution` and the `spec` it came from, so they can differ; give the specs' shots different ids (a
clash is warned about, and the later render wins). Rendering a whole spec drops its shots that are no longer in it.

```tsx
import manifest from '../public/blender/clips.json';
import { BlenderClip, blenderClip } from '../../../lib/components';

const act = blenderClip(manifest, 'B1_act');          // throws with the ids it has if the shot isn't there
<Sequence from={ACT} durationInFrames={act.frames}>
  <BlenderClip clip={act} style={{ left: 560, top: 270, width: 1440 }} />
</Sequence>
const ALERT = ACT + act.event('alert');               // composition frame of the alert, for a caption or a cut
```

What makes it sit in the scene:

- **Let the clip's bottom edge be the frame's.** A `wide` shot cuts the desk legs at the bottom of the picture;
  placed higher, they stop in mid-air.
- **Put type behind the clip and let the action cross it.** Order in the JSX is depth: a title, then the clip,
  and the confetti falls in front of the words. That is what reads as 3D.
- **Show the spec beside the acting.** The Remotion scene can import the spec JSON and light each beat as it
  plays; both read the same times, so they can't drift (`examples/blender-in-remotion/src/Video.tsx`).
- **Hard cut to a `pull_out`.** Use the flat scene's own component for the first part and cut on the frame the
  spec's `skip_frames` names. No fade is needed, and a fade would show the two as different pictures.
- A transparent clip has no cast shadow (EEVEE can't render one onto nothing). Keep the character on a flat
  field, where none is expected.
- A transparent render of a full room is a waste: use kind `desk`.

The order to build in when the screen shows the video's own scene: render the Remotion scene
(`npx remotion render ... Screen out/screen.mp4`), then the Blender clips, then the video.

## Extending

- New verb: a method on `ClayPerson` (`lib/blender/character.py`) or `DeskRoom` (`lib/blender/sets.py`) that takes
  frames and appends `{"frame", "kind"}` to `self.events` for its sound; then a branch in `compile.py` and, if it
  takes a named choice, a line in `validate()`. Write lengths as `n(frames at 24 fps)` and repeats with
  `ticks()` (`lib/blender/kit.py`), so the verb lasts the same time at any frame rate.
- New look: a choice in `lib/blender/cast.py` (`CHOICES`), then the geometry for it in `character.py`.
- New prop kind: `lib/blender/studio.py`.
- New shot type: a row in `SHOT_TYPES` (`lib/blender/cameras.py`).
- New sound: a `kind` branch in `lib/blender/sound.py`.
- Only `lib/blender/kit.py` touches the low-level API; fix Blender version changes there.
- `lib/blender/timeline.py` (times and cues) and `lib/blender/cast.py` (character descriptions) have no Blender
  in them and are tested in `tests/python/test_blender_spec.py`, which also builds real scenes when Blender is
  installed: every look, the moves in three frame shapes, the studio, and the spec errors.

## Looking round the scene, and Blender MCP

`blender_render.py open spec.json` opens the built `scene.blend`. Use it, or a Blender MCP server attached to
that window, to try a pose, a light or a camera with a person watching. Then write what worked into the spec or
`lib/blender/`: the `.blend` is rebuilt on the next run, and the spec is what makes the video repeatable. The
render itself always goes through `blender_render.py` (headless Blender Python), never through a live session.
The reasoning is in `docs/blender.md`.

## Blender gotchas (5.3)

- Blender exits 0 when the script it ran raised. `blender_render.py` looks for the compiler's `COMPILE_DONE`
  line, not the exit code.
- A movie's frame rate is on `bpy.data.movieclips.load(path).fps`, not on the image datablock.

- Video output: set `image_settings.media_type = "VIDEO"` *before* `file_format = "FFMPEG"` (5.0 change).
- EEVEE's engine id is `BLENDER_EEVEE` (it was `BLENDER_EEVEE_NEXT` in 4.2–4.4); `kit.setup_render` picks.
- IK on roll-0 arm bones needs `pole_angle = +90°` (−90 points the elbows backwards). Measure, don't guess:
  compare the solved elbow with the pole target.
- Distance-based skin weights bleed the chest into raised arms; use `parent_set(type="ARMATURE_AUTO")` (bone heat).
- AgX with default lights washes cartoon colours out; `exposure ≈ -0.9` and the `AgX - Punchy` look fix it.
- `-P script.py` runs with Blender's own Python: only `bpy`, `bmesh`, `mathutils` and the stdlib; no numpy
  guarantees, no toolkit imports. Host-side work (sound, exposure, sheets) stays in `tools/`.
- First Cycles frame compiles GPU kernels (minutes); time later frames when estimating (`doctor` does).
- An object scaled to zero (a hidden mouth shape) has no inverse matrix: skip it before `closest_point_on_mesh`.
- A movie texture needs `image_user.use_auto_refresh = True` or every frame renders the first one, and
  `frame_duration` set from the image, or it shows nothing.
- `film_transparent` gives straight alpha in the PNGs; encode with `-pix_fmt yuva420p -auto-alt-ref 0` or
  libvpx drops the alpha plane.

## Limits (experimental)

One character per set; the clay cast's hands are one bone (a mitten), so its `reach` puts the hand on a thing
but does not close round it (a fitted model can have fingers); nothing stops a walk going through furniture (walk
round it, or bend the walk with `curve`); the clay cast's hoodie hem is ragged when it stands, because its
body is made seated; a studio set has props and no character; cartoon "clay" style;
hand-keyed acting (no motion capture); no cast shadow on a transparent background; studio props don't move (the
camera does). Next on the roadmap: props that move and a tablet prop, cast shadows on transparent clips, two
characters in a set, hand shapes and props in the hand, your own `.blend` as a set, a `script` hook in the spec, imported rigs,
and a `blender` visual kind in `video_job.py`.
