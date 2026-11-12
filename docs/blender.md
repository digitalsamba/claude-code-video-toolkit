# Blender in the toolkit

Blender is an optional component. Nothing else in the toolkit needs it. With it installed, a JSON **shot spec**
becomes a 3D scene: a rigged cartoon character acting at a desk, or a product stage with a phone, a floating card
or an imported model. The render is a finished film, a set of clips for another build (a Remotion video, a
moviepy timeline, a video job, an editor), or both from the same spec.

- **How to write a spec:** `.claude/skills/blender/SKILL.md`
- **A film from one spec:** `examples/blender-disk-panic`
- **Blender clips inside a Remotion video:** `examples/blender-in-remotion`
- **A product stage with no character:** `examples/blender-product-shot`

## Setup

1. Install Blender 4.2 or later. The toolkit is made on 5.3 (see *What it has been run on*, below).

   | Platform | Command |
   |---|---|
   | macOS | `brew install --cask blender` |
   | Windows | `winget install BlenderFoundation.Blender` |
   | Linux | `snap install blender --classic`, or the tarball from [blender.org](https://www.blender.org/download/) |

2. Ask the toolkit whether it can see it, and how fast this machine renders:

   ```bash
   uv run tools/blender_render.py doctor
   ```

   ```
   ok    Blender 5.3.0 at /opt/homebrew/bin/blender
   ok    ffmpeg
   ok    Pillow (contact sheets, exposure)
   ok    test render: 1.62 s/frame at 1920x1080 (BLENDER_EEVEE; the first frame took 1.7 s)
         blender-disk-panic: 540 frames, about 15 min to render here
   ```

   `doctor spec.json` times your own spec. `--no-bench` skips the test render.

Blender is found on `PATH`, in `/Applications/Blender.app`, under `Program Files\Blender Foundation`, or in
`/usr/bin`, `/usr/local/bin`, `/snap/bin` and `/opt/blender`. Anywhere else, set `BLENDER=/path/to/blender`.

Blender runs headless (`blender -b`), so this works over SSH and in CI. No window opens and no add-on is needed.
`ffmpeg` joins the picture and the sound and encodes clips; transparent clips need its `libvpx-vp9` encoder, which
the Homebrew, winget and distro builds include.

## What it has been run on

| System | Blender | Result |
|---|---|---|
| macOS, Apple M4 | 5.3 | Everything: both engines, films, transparent clips, the tests |
| Linux x64, in a container with no GPU | 4.2 LTS | All four example specs build and pass `check` under Cycles; the film, the clip and the transparent-clip paths render. EEVEE also runs, on Mesa's software OpenGL, slowly |
| Windows | | Not run yet. The code finds Blender under `Program Files` and reads and writes UTF-8 throughout; treat it as untested |

Transparent WebM clips have been played in Remotion on the Mac only.

## How long a render takes

Measured on a MacBook with an Apple M4 (10 cores, 24 GB), Blender 5.3, EEVEE, 32 samples:

| What | Time |
|---|---|
| `check` (build the scene, verify every camera, 8 stills, contact sheet) | 5 s |
| `render` at 640x360, a draft to judge motion | 0.35 s a frame |
| `render` at 1280x720 | 0.9 s a frame |
| `render` at 1920x1080 | 1.6 s a frame |
| `render --clips --alpha` at 1920x1080 (no room to light) | 0.75 s a frame |
| `render --engine cycles` at 1920x1080 | 17 s a frame |
| The 22.5 s example film (`blender-disk-panic`, 540 frames) | 15 min |
| All four clips of `blender-in-remotion` (555 frames) | 10 min |

So iterate in `check`, which is seconds, and render when the contact sheet reads right. A render prints its
progress and time left every 30 seconds. The first Cycles frame also compiles GPU kernels, which takes minutes
once per machine.

Cycles uses the first GPU backend the machine has: Metal on a Mac, then OptiX, CUDA, HIP and oneAPI. With none, it
renders on the CPU, which gives the same picture more slowly. EEVEE wants a GPU with OpenGL 4.3, Metal or Vulkan.
On a Linux server without one it still runs if Mesa's software renderer is installed, but every frame is drawn on
the CPU, so it is slow; `--engine cycles` is the dependable choice there. Time either with `doctor` before
committing to a long render.

## What you get

```bash
uv run tools/blender_render.py check  spec.json     # checks.json, stills/, contact_sheet.png, scene.blend
uv run tools/blender_render.py render spec.json     # the film: <folder>.mp4 with sound
uv run tools/blender_render.py render spec.json --clips --alpha --to projects/my-video/public/blender
uv run tools/blender_render.py open   spec.json     # the built scene in Blender, to look round
```

`--clips` writes one clip per shot and a `clips.json` manifest instead of one film:

| Flag | Result |
|---|---|
| `--clips` | `<shot>.mp4` per shot (H.264), `<shot>.wav` with that shot's sound effects, `clips.json` |
| `--alpha` | transparent background: `<shot>.webm` (VP9 with alpha), which Chrome and Remotion read |
| `--prores` | also `<shot>.mov` (ProRes 4444) for Premiere, Resolve and Final Cut |
| `--shots A,B` | only these shots, for a re-render after one beat changed |
| `--to DIR` | copy the clips into a project's `public/` folder and add them to its `clips.json`. Several specs can share the folder: each clip records its own `fps`, `resolution` and `spec` |

## In a Remotion video

Set the spec's `fps` to the composition's (30 in the toolkit's templates), render the clips into the project, and
play them with `BlenderClip` from `lib/components`:

```tsx
import manifest from '../public/blender/clips.json';
import { BlenderClip, blenderClip } from '../../../lib/components';

const act = blenderClip(manifest, 'B1_act');
<Sequence from={345} durationInFrames={act.frames}>
  <BlenderClip clip={act} style={{ left: 560, top: 270, width: 1440 }} />
</Sequence>
```

The clip plays once, holds its last frame, and brings its own effects. `act.event('alert')` is the clip frame the
alert lands on, for a caption or a cut that should hit the same moment.

It works in the other direction too. A spec's `set.screen.video` puts a Remotion render on the laptop in the
scene, and a `pull_out` shot starts with that picture covering the frame, so a flat scene can cut to itself in 3D.
`examples/blender-in-remotion` does both.

## In other builds

`render --clips` without `--alpha` writes an ordinary H.264 `.mp4` per shot, so a Blender shot goes wherever a
video file does: `VideoFileClip` in a moviepy build, a `file` scene in a `video_job.py` spec, `cutdown.py` for a
vertical cut. The clip is picture only; the shot's sound effects are the `.wav` beside it. `--prores` adds a
ProRes 4444 `.mov` for Premiere, Resolve and Final Cut (transparent with `--alpha`). `clips.json` lists each
clip's length and the frame of every sound event, for a build that wants to cut on the acting. A whole film
(`render` with no `--clips`) already has its sound, and takes a music bed from `addmusic.py`.

## Timing from the narration

A spec can also take its timing from the narration: list the scene's `.captions.json` under `voice`, and a beat's
time can be `"cue:disk is full"`, the moment those words are said.

A video on a screen (the laptop, a phone or a card) advances one of its frames per scene frame, so it must be
rendered at the spec's `fps`. One at another rate is refused before anything renders: `screen video scene.mp4 is
30 fps and the spec is 24 fps: render it at 24 fps, or set "fps": 30`.

## Characters

A cast member is a description in JSON, not a model file: a build, a face, hair, clothes and accessories, each a
named choice or a hex colour. An agent can write one from a sentence or a photo.

```json
{"id": "maya", "seed": 12,
 "body":    {"build": "slim", "skin": "brown"},
 "face":    {"shape": "oval", "eyes": "brown", "brows": "thin", "nose": "small", "beard": "none"},
 "hair":    {"style": "bob", "color": "grey"},
 "clothes": {"top": "shirt", "top_color": "#2E7D5B"},
 "accessories": ["glasses"]}
```

- Whatever is left out takes the default character's look, or, with a `seed`, is drawn from the seed: `{"id":
  "extra", "seed": 41}` is a whole person, and the same person on every machine.
- A character can live in its own file and be named with `"from"`; fields beside it override the file, so one
  video can change the costume and keep the person. A kit keeps its cast in `kits/<name>/cast/`
  (`kits/example/cast/maya.json`).
- A field that isn't one of its choices is a spec error that lists the choices.
- Every character is built on one rig, so every acting verb works on every character. They are caricatures in one
  cartoon style, not likenesses.

The full table of choices is in the skill.

## A product stage

Set kind `studio` is a backdrop that curves from floor to wall, with soft lights and no character. Its props are a
`phone` or a `card` showing a picture or a video, and a `model` (`.glb`, `.gltf`, `.obj`, `.fbx`, `.stl`) scaled to
a size in metres. Three shot types film them: `product` (a three-quarter view), `orbit` (round the prop) and
`reveal` (starts on a screen edge to edge, then pulls back). The camera's distance comes from the prop's size, and a
prop with a screen is checked for staying readable through the shot. With `--alpha` the backdrop is left out, for a
clip over a Remotion scene. `examples/blender-product-shot` is the worked example.

## Blender Python or a Blender MCP server?

The toolkit drives Blender with its own Python (`blender -b -P lib/blender/compile.py`), and that is the route to
build on:

- **The spec is the source.** The scene is rebuilt from JSON on every run, so a video can be re-rendered next
  month, by another agent, on another machine, and come out the same. A scene edited live has no such record.
- **It runs unattended.** No window, no add-on, no socket: it works in CI, over SSH and inside `video_job.py`.
- **Checks come back as data.** `check --json` says which camera failed and why, which is what an agent needs to
  fix a shot without looking at it.

A Blender MCP server (for example [blender-mcp](https://github.com/ahujasid/blender-mcp)) connects an agent to a
Blender window that is already open. That is useful for a different job: working out a look with a person
watching. Use it to try a pose, a light or a camera in `scene.blend`, to bring in a model from a library, or to
ask what an object in the scene is called. Then carry what you learned back into the spec or `lib/blender/`, where
it is kept. Don't treat the edited `.blend` as the result: the next `check` overwrites it.

The toolkit doesn't install or need an MCP server. `blender_render.py open spec.json` opens the built scene for
either a person or an MCP session to look at.

Installing `bpy` from PyPI is a third route. It pins a Python version and a 300 MB wheel into the toolkit's
environment for no gain over the Blender most people already have, so the toolkit calls the installed Blender.

## Limits today

9:16 works: set `"resolution": "1080x1920"` and read the contact sheet, since a `wide` shot crops the desk.
`check` builds its stills in the film's own shape (the spec's `resolution`, or `--res`), because where the solver
puts a camera depends on the shape of the frame.

One character per set, seated at a desk; three sets (`desk_room`, `desk` with no room for transparent renders,
and `studio`, which has props and no character); a cartoon style; acting keyed from verbs, with no motion
capture; studio props stand still while the camera moves. EEVEE cannot render a shadow onto a transparent
background, so a transparent character or prop has no cast shadow on the Remotion scene. The room's light is dim
on the darkest skin tones in a `wide` shot: raise `look.exposure` a little or use a `medium`. A `pull_out` passes
close to the character's hand on its way back, inside the limit the check enforces but large in frame for a
moment.

The roadmap goes two ways. More from the spec: props that move, a tablet prop, cast shadows on transparent clips,
two characters in a set, standing and walking, and a `blender` visual kind in `video_job.py`. And more of Blender
itself: your own `.blend` as a set, a `script` hook that runs your Python between the build and the checks, and
imported rigs whose actions become verbs.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Blender not found` | Install it, or set `BLENDER=/path/to/blender` (the program itself, not its folder) |
| `doctor` says the version is too old | 4.1 and earlier have the previous EEVEE and a different API. Install 4.2 or later |
| A shot is `FAIL: view blocked by ...` | Something is between the lens and the subject and the solver found no clear spot. Change the shot's `type`, `az` or `dist` |
| A shot is `FAIL: lens 0.16 m from dev_body` | A `pull_out` or `push_in` passes too close to the character. Give it a wider `az` or more `elev` |
| `BlenderClip ... was rendered at 24 fps` | The spec's `fps` isn't the composition's. Set it and render the clips again |
| The laptop's picture is dull or tinted | The default AgX view compresses colour. For a picture that must match a flat scene, set `"look": {"view": "Standard"}` |
| The transparent clip has a black background in a player | The player ignores VP9 alpha. Chrome and Remotion read it; use `--prores` for editors |
| EEVEE fails or crawls on a headless Linux box | No GPU. Install Mesa's software renderer for a slow EEVEE, or use `--engine cycles` |
| `screen video ... is 30 fps and the spec is 24 fps` | A screen plays one video frame per scene frame. Render the video at the spec's `fps`, or change the spec's |
| `cast member 'maya': hair.style is one of ...` | A character field that isn't one of its choices. Pick from the list in the message |
| A studio shot is `FAIL` with "faces the lens" | The orbit turns the screen away for more than half the shot. Narrow `az` to `az_to` |
| A studio shot fails with `exposure 0.73 outside (0.08, 0.7)` | A light backdrop fills the frame. Darken `palette.backdrop` a step, or raise the shot's `fill` |
