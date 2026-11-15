# Blender in Remotion

A 23.5 s Remotion video with Blender renders as two kinds of element:

1. **A Remotion scene inside a 3D scene.** A `Terminal` scene plays flat for five seconds. The video hard-cuts to
   the same scene on a laptop in a Blender room, on the same frame, and the camera pulls back round the shoulder
   of the character typing it. Then the character looks up.
2. **A 3D character inside a Remotion scene.** The character, at a desk on a transparent background, acts a short
   panic beside the list of beats it is acting; the list lights each beat as it plays. In the close the title sits
   behind the character and the confetti falls in front of it.

Nothing is timed by hand. Clip lengths come from `public/blender/clips.json`, and the beat list reads
`blender/desk.json`, the file Blender keyed the acting from.

## Build

Needs Node, Blender 4.2 or later and ffmpeg (`uv run tools/blender_render.py doctor` from the toolkit root).

```bash
npm install
npm run blender     # the screen video, then both Blender specs: about 10 minutes on an M4
npm run render      # out/blender-in-remotion.mp4, 90 s
```

The sound is effects only, so the render lands near -17 LUFS. For a copy at YouTube's -14:
`uv run tools/qc.py --render examples/blender-in-remotion/out/blender-in-remotion.mp4 --fix`.

`qc.py --render` warns once, "motion stops dead" at about 17.8 s. That is the character holding still after the
frantic typing until the cut at 19.2 s, a pause in the acting and not a clip that ran out.

`npm run blender` is three steps, in the order they depend on each other:

| Step | Makes | Time (M4) |
|---|---|---|
| `npm run screen` | `out/screen.mp4`: the terminal scene, which `blender/room.json` puts on the laptop | 20 s |
| `npm run blender:room` | `public/blender/A1_reveal.mp4`, `A2_face.mp4` (195 frames) | 5 min |
| `npm run blender:desk` | `public/blender/B1_act.webm`, `B2_cheer.webm`, transparent (360 frames) | 4.5 min |

The clips are not in the repository (video files are ignored); the specs, the manifest and the sound are.

## How it fits together

| File | What it is |
|---|---|
| `blender/room.json` | The room, with `set.screen.video` naming the Remotion render and a `pull_out` shot. `"view": "Standard"` keeps the screen's colours as Remotion made them, so the cut doesn't show |
| `blender/desk.json` | The transparent element: set kind `desk`, `"transparent": true`, two shots of acting beats |
| `src/Screen.tsx` | The terminal scene. `REVEAL` (frame 150) is where the video cuts to the laptop, and `skip_frames: 150` in `room.json` starts the laptop's picture on that frame |
| `src/Video.tsx` | The video: `blenderClip(manifest, id)` and `<BlenderClip>` from `lib/components` |
| `src/captured.json` | What `blender_render.py` printed when this example was built: the terminal's lines |

To change the acting, edit a beat in `blender/desk.json`, run `uv run tools/blender_render.py check` on it and
read the contact sheet, then `npm run blender:desk`. The list in the video follows, since it reads the same file.

Method and spec format: `.claude/skills/blender/SKILL.md`. Setup and timings: `docs/blender.md`.

## Sound

Typing: "Keyboard Typing 001.wav" by ristooooo1, CC0 (`public/audio/sfx/typing.attribution.txt`). Everything the
character does (key clicks, the alert, the head shake, the cleanup notes) is synthesised by `lib/blender/sound.py`
on the frames Blender keyed, and comes with each clip as a WAV.
