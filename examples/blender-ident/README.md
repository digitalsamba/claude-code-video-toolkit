# Idents from a logo (Blender studio set)

Two opening and closing idents for this toolkit's own mark, each from one spec and one SVG. Nothing here is drawn
by hand in Blender: the mark is `assets/mark-parts.svg`, the timing is the spec's beats, and the sounds are three
recordings and some sine tones named in the spec. Blender is optional in the toolkit; this example needs it.

| Spec | The ident | Open | Close |
|---|---|---|---|
| `clap.json` | The mark alone on a dark stage. A prompt (an upright bar and a caret) leans back, snaps to its lean, and the hit knocks the caret up into the play triangle; a light crosses it. The close plays it back and ends on a blinking prompt | 2.0 s | 3.5 s |
| `key.json` | The same clap on a keycap, for longer videos. The key is pressed, clicks, the bar snaps, the triangle comes up lit, and the camera rises to face the key | 2.4 s | 3.5 s |

```bash
uv run tools/blender_render.py check  examples/blender-ident/clap.json          # about 5 s: both shots, and the mark test
uv run tools/blender_render.py render examples/blender-ident/clap.json --clips --with-sound --res 960x540   # a draft, about 80 s
uv run tools/blender_render.py render examples/blender-ident/clap.json --clips --with-sound                 # 165 frames, about 6 min on an M4
uv run tools/blender_render.py render examples/blender-ident/clap.json --clips --with-sound --res 1080x1920 --out examples/blender-ident/out/clap-9x16
uv run tools/blender_render.py render examples/blender-ident/clap.json --clips --alpha --out examples/blender-ident/out/clap-alpha
```

Output lands in `out/` (gitignored): `clips/open.mp4` and `clips/close.mp4` with their sound in the file,
`open.wav` and `close.wav` beside them, `clips.json`, `contact_sheet.png`, `checks.json` and `render_checks.json`.
The rendered clips for the toolkit's brand are kept in `brands/video-toolkit/ident/`, the brand's ident slot
(`docs/creating-brands.md`, "Mark and ident"). `build.py` fills it: it copies the clips, names the default ident,
and measures from each open the frame of the hit and `title_frame`, the first frame from which the picture is at
rest. `templates/howto` plays what is there.

What it shows:

- **Any SVG logo becomes parts.** Each filled shape is extruded and bevelled in its own colour and can be moved
  by its `id`: `mark.slash`, `mark.play`, `mark.caret`. The caret is an `extra`: the film needs it and the
  finished mark does not have it.
- **The move is a list of beats.** `shear` leans the bar with its foot flat, `grow` brings the triangle up from
  its corner on a `spring`, `hit` shakes the camera and flashes, `press` pushes the key into its plate, `sweep`
  sends a light across the face, `blink` blinks the caret. Change a number, run `check`, look again.
- **The sound is on the hit.** `"sound": "clap"` on the `hit` puts the slate recording's loudest moment on that
  frame, with a low chord under it. After a render the clip is measured: the frame where the picture changes
  most should be the hit's frame.
- **The last frame is checked as a still.** The open ends on the mark and the close starts on it; `check`
  compares that frame with the flat SVG (each part in place, the caret gone, the lens square on) and reads each
  part's colour from the still. The dark stage's lights are set so `#3771e0` comes out as `#3771e0`.
- **Both shapes from one spec.** `--res 1080x1920` renders the vertical ident; on the keycap the camera stands
  back by itself while the key is seen from the side.

Colour: the hit is lit by a `flash` and the look's `bloom`, with `glow` kept under 1. Past 1 the blue clips
toward cyan. Bloom is left out of `--alpha` clips, which have nothing for the glow to sit on.

The transparent clips are for dark grounds: they keep the dark stage's colours, so the white triangle is faint
over a light scene. `ident.json` says so as `"alpha_ground": "dark"`.

Sounds (`sfx/`, each with its credit file, all CC0 from Freesound): a film slate clap, a clapperboard for body
under it, and a mechanical key. Nobody had listened to the mixes in this folder when it was written; the levels
and the sync are measured, and whether the sound is right is for a person to say.

To make your own: draw the mark flat, test it with `uv run tools/mark_sheet.py --svg mark.svg`, give each shape
an `id`, and start from `clap.json`. The bar for a mark and an ident, and the method, are in
`.claude/skills/video-design/idents.md`; the spec's fields are in `docs/blender.md` ("A logo and an ident").
