# Two fitted characters for the tests and the kitchen example

## Mara, who is invented

Mara is made up. She is not anyone's likeness, so she can live in the repository and the tests can fit a
generated model, give it a face and fingers and act with it on any machine with Blender.

- `mara.glb`: the body. Generated with Meshy (through MCP for Blender) from a text prompt, then reduced to 60,000
  faces with only its base colour texture at 1024 px (2 MB). The UV seams were welded before reducing: reduced as
  imported, the mesh opened along every seam.
- `mara_hands.glb`: the hands. Her generated hands have the fingers joined, so a second model was made the way
  `docs/blender.md` describes: a FLUX.2 edit of a front still of her with the fingers spread, then Meshy from that
  picture. Only its forearms and hands are kept, at full resolution (the fit finds fingers by the gaps between
  them, and fewer faces close the gaps).
- `fit.json`: her joints, clothes line and face, read off front and side stills with a grid. Her right brow is under
  her fringe, so it is in `brows_hidden` and nothing is erased there.
- `spec.json`, `mara.json`: a short scene the test checks her with (hand shapes, a point, expressions, a wink).
- `enter.json`: the walk-in (through the door, across the room, onto the chair, in to the desk, typing, a wink).
  Her build is not the one the rig was worked out on (short arms, long legs, narrow shoulders, long hands), so
  this is the spec that shows a number tuned on one character: the same film has to pass on any other.

To fit her by hand: `uv run tools/blender_render.py fit --model tests/fixtures/blender-fitted/mara.glb
--fit tests/fixtures/blender-fitted/fit.json --out mara-fitted.blend --preview fit/`, with `mara-fitted.blend` next
to `mara.json`.

## Conal, the toolkit's author

A caricature of the person who wrote the toolkit, shared by him. It is the body the rig was worked out on, so
its numbers are the ones the verbs were first measured against; Mara is the body that shows which of them were
tuned. The photos it was made from are not in the repository.

- `conal.glb`: the body, from an image-to-3D service, with its seams welded and reduced to 150,000 faces with
  its base colour at 1024 px (4.8 MB). That is the size the fit works at (`"faces"` in `conal-fit.json`):
  reduced to 60,000 it still fits, but the sleeve check fails on the walk-in (4.5x against 4), because the
  measure reads edges and coarser cloth has longer ones.
- `conal_hands.glb`: the same character generated again with open hands, at 60,000 faces (1.7 MB). The fit
  finds five fingers on each hand in it.
- `conal-fit.json`, `conal.json`: his joints, clothes line and face, and the character file (glasses, hair).
- `enter-conal.json`: the walk-in, the same film as `enter.json`.

To fit him by hand: `uv run tools/blender_render.py fit --model tests/fixtures/blender-fitted/conal.glb
--fit tests/fixtures/blender-fitted/conal-fit.json --out conal-fitted.blend`.
