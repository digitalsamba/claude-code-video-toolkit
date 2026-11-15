# Disk Panic (Blender shot spec)

A 22.5 s character short built entirely from `spec.json`: a rigged cartoon developer is typing, a "disk almost
full" alert pops up, he panics, cleans up, and celebrates "92 GB FREE". Eight shots, 37 acting beats, sound effects
placed on the frames the motion was keyed on.

```bash
uv run tools/blender_render.py check  examples/blender-disk-panic/spec.json   # ~5 s
uv run tools/blender_render.py render examples/blender-disk-panic/spec.json   # ~15 min on an M4
```

Needs Blender 4.2 or later (`uv run tools/blender_render.py doctor` finds it and times a frame on your machine).
Output lands in `out/` (gitignored): `contact_sheet.png`, `checks.json`, `scene.blend`, `blender-disk-panic.mp4`.

What it shows:

- **No coordinates or keyframes in the spec.** Shots are types (`wide`, `closeup`, `over_shoulder`...) placed
  round their subject; acting is verbs (`expression: shocked`, `pose: hands_on_head`, `shake_head`).
- **Cameras checked by geometry.** `check` verifies each shot keeps its subject in frame and unblocked at the
  start, middle and end; here it pulls the `side` shot back 15% because the face dropped off the bottom of the
  nominal framing.
- **Sound on the motion.** `events.json` lists every key click, the alert, the gulp, each head-shake whoosh and
  the cleanup arpeggio at the time Blender keyed it.

Method and format: `.claude/skills/blender/SKILL.md`. Setup and timings: `docs/blender.md`. The same character as
an element of a Remotion video: `examples/blender-in-remotion`.
