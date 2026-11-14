# Tea first (a fitted character in the kitchen)

A 27.6 s film from `spec.json`, with a character made from a generated 3D model in place of the clay cast. Mara
comes through the door with her tea, crosses the kitchen, sits, puts the mug down by its handle, opens the laptop,
takes the mug up again and drinks, while Claude the CRT types on the counter. Eight shots, twelve beats.

Mara is invented and nobody's likeness. Her model lives in `tests/fixtures/blender-fitted`, so this runs from
the repo alone.

```bash
# 1. seat her model on the rig (8 s): writes out/mara-fitted.blend, which mara.json names
uv run tools/blender_render.py fit --model tests/fixtures/blender-fitted/mara.glb \
    --fit tests/fixtures/blender-fitted/fit.json --out examples/blender-kitchen/out/mara-fitted.blend

# 2. build the scene, verify every camera and the body, write a contact sheet (30 s)
uv run tools/blender_render.py check examples/blender-kitchen/spec.json

# 3. render (about 25 min on an M4: a fitted character in a lit room is 2.2 s a frame)
uv run tools/blender_render.py render examples/blender-kitchen/spec.json
```

Needs Blender 4.2 or later (`uv run tools/blender_render.py doctor`). Output lands in `out/` (gitignored):
`contact_sheet.png`, `checks.json`, `scene.blend`, the film.

What it shows:

- **A character from a model.** `mara.json` is one line: a `model` fitted to the shared rig. Every verb the clay
  cast has works on her, and she has a face and fingers of her own.
- **On her feet.** `open_door`, a `walk` that bends round the table (`"curve": true`), `sit_down`, `pull_in`.
- **A thing in the hand.** `carry` holds the mug by its handle and keeps it level through the walk and the sit;
  `put_down`, `pick_up` and `sip` do what they say.
- **The body is checked, not only the cameras.** `check` measures a hand inside the body, a limb through
  furniture, a sliding foot, a missed touch, a stretched sleeve and how far the mug tips (9.3 degrees here,
  limit 12), and fails the spec before a frame is rendered.
- **Cameras by type, and by hand where the type does not fit.** `full`, `wide`, `side` and `closeup` are placed
  round their subject; `check` swung the `side` camera 24 degrees because a cupboard was in the way. The walk,
  Claude and the last two-shot are `fixed` cameras, put where the spec says.

Two things this spec works round, both worth knowing before you write your own:

- **The sip is half height (`"part": 0.45`).** Raised all the way, Mara's sleeve stretches past the check's limit
  (4.7x against 4). The check is right and the rig is not finished here; a lower sip reads as a sip in a close-up.
- **The sip is a `closeup` turned 30 degrees (`"az": -30`).** From the front (`medium`) the laptop's lid hides
  the mug and both hands, and `over_shoulder` is a desk shot that lands under this table (`check` fails it).

Method and format: `.claude/skills/blender/SKILL.md`. How a model is fitted, and how a film like this is made
over several sessions: `docs/blender.md`.
