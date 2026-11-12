# Product shot (Blender studio set)

A 12 s product stage with no character: a picture on a floating card, the same app on a phone, and an imported
3D model, filmed in four shots. Everything is in `spec.json` and three small files in `assets/`.

```bash
uv run tools/blender_render.py check  examples/blender-product-shot/spec.json   # ~5 s
uv run tools/blender_render.py render examples/blender-product-shot/spec.json   # 360 frames, about 8 min on an M4
uv run tools/blender_render.py render examples/blender-product-shot/spec.json --res 640x360   # a draft to judge the moves
```

Needs Blender 4.2 or later (`uv run tools/blender_render.py doctor` finds it and times a frame on your machine).
Output lands in `out/` (gitignored): `contact_sheet.png`, `checks.json`, `scene.blend`,
`blender-product-shot.mp4`. A studio set has no beats, so the film is silent: check it with
`uv run tools/qc.py --render examples/blender-product-shot/out/blender-product-shot.mp4 --allow-silent`.

| Shot | Type | What it shows |
|---|---|---|
| `P1_reveal` | `reveal` on the card | Starts with the card's picture covering the frame, holds, then pulls back to the whole stage |
| `P2_phone` | `orbit` on the phone | The camera travels from 35 degrees on one side to 35 on the other; the screen stays readable |
| `P3_mug` | `product` on the model | A three-quarter view pushing in a little |
| `P4_spin` | `orbit` on the card | A wider arc, a little higher |

What it shows:

- **Distance comes from the prop's size.** The phone is 15 cm tall, the card half a metre wide and the mug 10 cm,
  and each fills its shot the same way with no distance written in the spec.
- **A screen is checked for being readable.** An orbit reports `screen_readable`, the share of the shot in which
  the screen faces the lens within 60 degrees; under half fails, with "narrow the orbit".
- **Any picture or video can be the screen.** Here the screens are PNG frames so the example needs nothing else.
  For a moving screen, name a Remotion render (`"screen": {"video": "../out/scene.mp4"}`) made at the spec's `fps`.
- **A model keeps its own materials.** `assets/mug.glb` was modelled for this example; swap in any `.glb`,
  `.gltf`, `.obj`, `.fbx` or `.stl` and set `size` to its longest side in metres.

For a clip to lay over a Remotion scene, add `--clips --alpha --to <project>/public/blender`: the backdrop is left
out and each shot becomes a transparent WebM for `BlenderClip`.

The stage is light, so its shots measure near the top of the exposure range (up to 0.66 against a limit of 0.70).
Cycles reads about 0.01 brighter than EEVEE, so leave that much room. A shot that is mostly backdrop can fail with `exposure 0.73 outside (0.08, 0.7)`: darken `palette.backdrop` a
step or bring the prop closer with `fill`.

Method and format: `.claude/skills/blender/SKILL.md`. Setup and timings: `docs/blender.md`. A character film:
`examples/blender-disk-panic`. Blender clips in a Remotion video: `examples/blender-in-remotion`.
