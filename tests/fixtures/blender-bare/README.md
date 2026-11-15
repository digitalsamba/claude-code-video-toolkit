# Nell: a bare-armed invented character with her painted face kept

Nell is made up. She is not anyone's likeness, so she can live in the repository. She is here for what Mara
(`tests/fixtures/blender-fitted`) does not have: bare arms, so there is no cuff for the hands to be found by or to
hide the join, and a face painted nearer to life (eyes with a lid over the iris), so it is fitted with
`"paint": "keep"`.

- `nell.glb`: the body. A front view in an A-pose from `flux2.py` (a text prompt), then Meshy through MCP for
  Blender from that picture. Made smaller with `uv run tools/blender_render.py reduce --model nell_gen.glb --out
  nell.glb --faces 60000 --texture 1024` (2 MB), which thins the mesh inside the islands of its texture and never
  across their seams.
- `nell_hands.glb`: the hands. The front view cropped to the hips up (so the hands are bigger in the picture),
  a `flux2.py --input` edit to open the hands with the fingers spread, then Meshy from that. A first try from the
  full-length picture gave four fingers. Only the arms are kept. It came back 1.8 times the body's size, sized as
  if the half figure were a whole one: the fit scales a hands model to the body's arm span before looking for the
  hands.
- `fit.json`: her joints, clothes line and face, read off gridded stills and slices of the mesh. Her right arm
  hangs 3 cm nearer her body at the elbow than her left, so the right elbow is in `joints_right`: mirrored from
  the left it stood outside the arm, the inside of the arm went with her spine, and a sheet of skin hung from
  her right forearm to her hip (the fit fails that now; a test fits her without it to prove the check).
- `spec.json`, `nell.json`: the short scene the test checks her with (hand shapes, a point, expressions, a blink,
  a glance, a wink).
- `enter.json`: the walk-in every fitted character should pass. She passes all of it but one check: as she
  reaches back for the chair to sit, the back of her right armpit stretches 4.1x (limit 4). It is the shoulder's
  skinning when the arm swings back, the move where Conal's and Mara's own worst stretch is too (3.8x and 3.6x).
  Open.

To fit her by hand: `uv run tools/blender_render.py fit --model tests/fixtures/blender-bare/nell.glb
--fit tests/fixtures/blender-bare/fit.json --out nell-fitted.blend`, with `nell-fitted.blend` next to `nell.json`.
