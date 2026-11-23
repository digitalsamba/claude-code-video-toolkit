# Manim: what a first spike found

[Manim Community](https://github.com/ManimCommunity/manim) is an open-source Python library (MIT) for animation
written as code: shapes, text, graphs, code listings and imported SVGs that are created, moved and turned into one
another. It is the look of 3Blue1Brown's maths videos.

**Optional and experimental.** Nothing in the toolkit depends on Manim. There is a render tool, a small helper
library and two examples; there is no template or skill yet. Most of this page is the notes from the first session
of tests (2026-11-23, Manim 0.22.1, macOS, Apple M4), kept so the next session starts from what was found. The
roadmap item is in `_internal/ROADMAP.md`.

## Using it

```bash
uv run tools/manim_render.py doctor                 # is Manim here, and what will and won't work
uv run tools/manim_render.py render examples/manim-ident/idents.py Typed --brand video-toolkit --out out/typed.mp4
uv run tools/manim_render.py render scene.py Flow --captions audio/flow.captions.json --alpha --out public/manim/flow.webm
uv run tools/manim_render.py render scene.py Fix --size 1080x1920 --out out/fix-9x16.mp4
```

`render` always renders at 30 fps and the size asked for, writes an `.mp4`, or with `--alpha` a `.webm` for
Remotion (or a `.mov` as ProRes 4444), and puts `<clip>.events.json` beside it when the scene noted any events.

A scene file starts with `from toolkit_manim import *` (`lib/manim`, put on the path by the tool). That gives
Manim itself and:

| Name | What it is for |
|------|----------------|
| `ToolkitScene`, `ToolkitScene3D` | `self.at(seconds)` waits for a narration cue and fails the render when the scene is already past it; `self.ev("key")` notes when something audible happens |
| `Voice()` | Word timings from `--captions`: `V.cue("the cache", nth=2, lead=0.35)` in seconds, `V.end` |
| `Brand()` | `--brand NAME`: `B.c("primary")` for a colour, `B.mark()` for the SVG |
| `Loop` | `self.run(build)` plays `build(phase)` for phase 0 to 1, for a background that loops without a jump |
| `PORTRAIT` | True for a portrait size, where the frame is 9 by 16 units |

Examples: `examples/manim-ident` (two idents with sound placed from the scene) and `examples/manim-diagram` (a
diagram that builds on the narration).

## What was tried, and what happened

| Test | Result |
|------|--------|
| A 19 s architecture diagram, each step on a narration word | Worked. 1080p30 rendered in about 4 s |
| The same diagram with a transparent background, inside a Remotion composition with the narration | Worked. Remotion drew the ground, a label and a footer; `qc.py --render` passed |
| An ident from a brand's `mark.svg`: a bar and a chevron that turn into the mark | Worked. 2.3 s clip in about 1.5 s; the last frame matched the flat mark pixel for pixel |
| The ident with alpha over a light and a navy ground | Worked, with one catch (see "Alpha") |
| A 9:16 code listing where one line becomes five, watched by a mascot made from the mark, with eyes that look and blink | Worked after two fixes (see "Vertical" and "Blinks") |
| A formula (`MathTex`) | Failed: no LaTeX on the machine |

## Installing it

- **`uv run --with manim` did not work on macOS.** Manim needs `pycairo`, and `pycairo` publishes wheels for
  Windows only. On macOS and Linux it is built from source, which needs a C compiler, `pkg-config`, cairo and
  pango. On the test machine the compiler itself was broken (command line tools older than the OS), so the build
  failed before it started.
- **`brew install manim` worked.** Homebrew has a prebuilt bottle with its own Python, about 60 MB, and gives a
  `manim` command. That makes Manim a program on the machine, like Blender, and not a Python dependency of the
  toolkit. A future `doctor` should look for the `manim` command first.
- **Homebrew upgraded ffmpeg as a side effect, and the new one has no text filters.** Installing the `manim`
  bottle moved ffmpeg from 8.0.1 to 9.0.2, and Homebrew's ffmpeg 9 no longer includes `drawtext` or `subtitles`
  (they are in a separate `ffmpeg-full` formula now). `tools/notebooklm_brand.py` uses `drawtext` and stops
  working; `tools/soulx.py` falls back to unlabelled output. A `doctor` should check for the filters a tool needs
  and say so, and the setup notes should warn before the install.
- **Linux and Windows were not tried.** The project publishes a Docker image (`manimcommunity/manim`). On Windows
  the wheels exist, so `uv` may be enough.
- **Formulas need LaTeX.** `MathTex` and `Tex` call `latex` and `dvisvgm`; without them the render stops with
  `FileNotFoundError: 'latex'`. `Text` (drawn by Pango from system fonts) needs neither, and every test above
  used only `Text`. A diagram, a code listing or an ident does not need LaTeX. A maths explainer does.
- Homebrew's build prints `SyntaxWarning` lines from a dependency on every run. `PYTHONWARNINGS=ignore` hides
  them.

## Timing animations to narration

Manim scenes are a sequence of `self.play(...)` and `self.wait(...)` calls, so the scene keeps its own clock.
Narration cues fit onto that directly. The same `<name>.captions.json` that `tools/captions.py` writes for Remotion
gives the second each word starts, and a scene waits until that second before it plays the next step:

```python
class CuedScene(Scene):
    def at(self, t, slack=1 / 30):
        now = self.renderer.time
        if t < now - slack:
            raise RuntimeError(f"cue at {t:.2f}s is {now - t:.2f}s in the past: shorten the animation before it")
        if t > now:
            self.wait(t - now)

# in construct():
self.at(V.cue("cache", lead=0.35))        # 0.35 s before the word, so the box has landed when it is said
self.play(FadeIn(cache), run_time=0.8)
```

`V.cue("phrase", nth=2)` finds the second time a phrase is said and raises when the phrase is not in the
narration, the same rule as `cue()` in `lib/components/cues.ts`.

- **Refuse to go back in time.** An animation that runs longer than the gap to the next cue would push everything
  after it late, silently. Raising in `at()` turns that into a failed render that names the cue.
- **Start before the word.** A shape that begins to appear on its word lands after it. A `lead` of 0.25 to 0.5 s
  put the finished shape on the word.
- **Free word timings.** The test narration was voiced with the operating system's voice and timed with the local
  whisper engine (`captions.py --engine whisper`), so a timing test costs nothing. It is a scratch voice only.

## Into Remotion

- `manim -t` renders with a transparent background as a QuickTime Animation `.mov` (`qtrle`, `argb`). It is large:
  12.7 MB for 19 s.
- **QuickTime Player will not open that `.mov`.** It reports the file as incompatible, because current macOS no
  longer plays the Animation codec. To look at a transparent clip, convert it to ProRes 4444, which keeps the alpha
  and plays (`ffmpeg -i in.mov -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le out.mov`), or lay it over a
  ground as an MP4.
- Remotion wants WebM with alpha. Convert with
  `ffmpeg -i in.mov -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 0 -crf 28 -auto-alt-ref 0 out.webm`. The 19 s clip
  came to 0.5 MB.
- **The conversion is the slow step.** The render took 3 s and the conversion 18 s.
- In Remotion it is `<OffthreadVideo src={staticFile('flow-alpha.webm')} transparent muted />`, the same as a
  transparent Blender clip, so `BlenderClip` should take one unchanged (not tried).
- **Who draws what.** Remotion drew the ground, the label and the footer; Manim drew only the diagram. That keeps
  type, colour fields and captions in one place and lets a Manim clip sit in any template.
- **Set the frame rate.** Manim's quality presets carry their own rates (`-ql` is 15 fps, `-qh` is 60). Pass
  `--fps 30 -r 1920,1080` every time.

## Look

- **The defaults are not ours.** Manim starts on black with its own colours. Set `config.background_color` and
  pass the brand's hex values; `Text(..., font="Helvetica Neue")` takes any installed font by name. A font that
  is not installed falls back without an error, so a render on another machine can change its type silently.
  Bundled fonts would need registering before use (not tried).
- **Sizes are in frame units, not pixels.** The frame is 8 units high and about 14.2 wide. Default text is small
  for video; `scale_to_fit_width(config.frame_width * 0.78)` sizes a title against the frame.
- **Morphs need looking at.** `Transform` between an open three-point chevron and a closed triangle passed
  through an odd, shrunken shape for a few frames. Check the middle frames of every morph on a contact sheet. A
  quick one (0.3 s) hides it; a slow one would not.

## Vertical

`-r 1080,1920` changes the pixels and nothing else. The frame stays 14.2 units wide, so everything is drawn as if
for 16:9 and lands in a band across the middle. Say what the frame measures:

```python
config.frame_width, config.frame_height = 9.0, 16.0
```

and lay the scene out from `config.frame_width`, so one scene file can serve both shapes.

## Alpha

An ident with a transparent background is only right over the ground it was drawn for. The toolkit's mark has a
near-black triangle, which nearly disappears over navy. A brand keeps `mark-on-dark.svg` for that, and the ident
manifest's `alpha_ground` already says which ground a clip is for. A Manim ident needs one render per ground,
from the matching SVG.

## Idents

- `SVGMobject("mark.svg")` reads a flat SVG and gives one object per shape, in the file's order, with their
  fills. The parts can be moved and transformed on their own.
- Ending on the unchanged `SVGMobject` (swap the animated parts for it on the last beat) made the last frame
  identical to a still of the flat mark. That is the same check the Blender ident runs, and it passed with a
  difference of zero.
- It has no depth, light or bloom. It is the 2D ident: seconds to render, no Blender.
- The how-to template already draws a 2D mark in Remotion when a brand has no ident. Before building a Manim
  ident maker, compare the two on the same mark.

## A flat character

A mascot was made from the mark: the SVG as a body, two white circles and two dots as eyes. It looks at a point
(the pupils move toward it inside the whites) and blinks.

- **Blinks.** Two `.animate` calls in a `Succession` both work out their target from the object as it is when the
  line runs, so "squash, then stretch back" stretched the open eye to ten times its height. One animation with
  `rate_func=there_and_back` goes and returns:
  `eyes.animate(run_time=0.18, rate_func=there_and_back).stretch(0.1, 1)`.
- This is as far as characters go here. There is no skeleton, no walk and no hands. A mascot that looks, blinks
  and points is in reach; acting is Blender's job.

## Ident variations that say what the toolkit does

Five were made from the same `mark.svg`, each two and a half to three and a half seconds, each ending on the flat
mark:

| Name | What happens |
|------|--------------|
| Prompt | A caret and a chevron (a prompt) snap into the slash and the play triangle |
| Typed | `/video` is typed at a caret; its slash becomes the mark's slash and the word folds into play |
| Chain | Lines of script become a waveform, the waveform a frame, the frame play, with one word under each |
| Playhead | The slash crosses four empty scenes as a playhead and each fills as it passes; the track closes into play |
| Frame | The chevron opens into a 16:9 frame, the frame fills and folds into play |

- **Keep the count of parts the same through a chain.** Five lines to five bars to five strips to five copies of
  the triangle morphs cleanly, one to one. Five strips standing in for one frame show hairline seams between them.
- **Many shapes into one:** transform each into its own copy of the target, then swap the copies for the real
  shape on the last beat.
- **Type by hand:** adding a `Text`'s letters one at a time with a short wait, and moving a caret after each, is
  simpler to time than the built-in letter animations.
- **An MP4's last frame is not exact.** Against a still of the flat mark, about 1,100 pixels along the edges
  differed after H.264. The lossless transparent render differed by none, so run that check on the `.mov`.

### Sound on an ident

Typed and Chain were given sound: a key click on each typed letter, a whoosh across each morph and a low key as
the mark lands. All three recordings are CC0 (the key is the one `examples/blender-ident` already uses; the
whooshes came from `tools/sfx_search.py --license cc0`).

- **Let the scene say when.** The scene notes the second each audible thing happens
  (`self.renderer.time` just before the `play`) and writes them to a JSON file as it finishes. The sound is built
  from that file, so a change to the animation moves the sound with it.
- **Place a sound by its own shape.** A click goes by its peak; a whoosh goes by the point where half its energy
  has passed, set a little past the middle of the morph it covers. Measured after the mix, every click sat within
  a millisecond of its letter.
- **Six clicks from one recording.** Small changes of pitch (a few percent) and level per letter stop one sample
  sounding like a machine gun. The landing sound is the same key at half speed.
- **Give the letters room.** At 75 ms a letter the clicks ran together; 100 ms reads as typing.
- Whether it sounds right is for a person to say. Only the timing was measured.

## Looping backgrounds

Three six-second loops were made in the brand's colours: a field of dots on navy with a wave crossing it, soft
lines breathing across a light ground, and thin rings turning around an off-centre point with one accent arc.
1080p30 took 2 to 11 s each.

- **A loop is a phase from 0 to 1.** One `ValueTracker` runs from 0 to 1 over the loop and `always_redraw`
  rebuilds the picture from it. Give every motion a whole number of turns over the phase (`TAU * p`, rings at
  whole-number rates) and the last frame leads back into the first. Measured: the change across the loop point
  was the size of one or two ordinary frame steps, with no jump.
- **Redrawing is the cost.** The dot field rebuilds 570 dots a frame and was the slow one (11 s). It is still
  fast, but a denser field would not be.
- **A background must be quieter than what sits on it.** The lines moved about fifteen times as much per frame as
  the dots. That is too lively to put text over; slow it or thin it before using it.
- **Not tried:** one of these behind a Remotion scene with text on it, or as the picture on a screen or wall in a
  Blender set. Both should only need the MP4.

## 3D geometry

Three loops with `ThreeDScene`: a wire icosahedron turning inside a dodecahedron, a sheet of squares with a
ripple running through it, and a trefoil knot drawn as a tube.

- **It works, and it is slower.** Six seconds at 1080p30 took 32 to 74 s, against 2 to 11 s for the flat loops.
  The renderer sorts every face by depth on each frame.
- **It is flat-shaded geometry, not a lit scene.** There are no shadows, materials or depth of field, and faces
  that pass through each other can sort wrongly. For a product or a character, use Blender. For a shape that
  turns behind a title, this is enough and takes a few lines.
- **The light changes your colours.** Surfaces are shaded by a built-in light, so a brand blue comes out lighter
  on faces turned toward it. `set_fill_by_checkerboard` with two close blues reads as one material; two far
  apart read as a pattern.
- **Any surface is a function.** `Surface(lambda u, v: point, ...)` drew the ripple and the knot; the knot's tube
  is a circle carried along a curve. `Dodecahedron` and `Icosahedron` are built in, with `faces_config` and
  `graph_config` for the faces, edges and corner dots.
- **A turn is a loop.** Rotating by `TAU` with a linear rate closes on itself.

## Code listings

`Code(code_string=..., language="python", background="window")` draws a highlighted listing. Its lines are
`code.code_lines[i]`, so a line can be boxed with `SurroundingRectangle`, faded, or transformed into a line of a
second listing. The background is `code.background` and can be transformed too, so the window grows with the code.

## What this suggests

- **Shape:** a program on the machine, found by a `doctor`; a scene file in, a clip out (transparent when
  asked); cues from the same captions files; Remotion does the composing. `tools/manim_render.py` and `lib/manim`
  are that shape, built after the tests.
- **First uses:** a diagram scene in a sprint review (a backend change that a screen recording can't show), a code
  change, a 2D ident for a brand without Blender.
- **Open:** Linux and Windows installs, bundled fonts, formulas with LaTeX, a clip through `BlenderClip`, a real
  voice, and whether a scene is better written by hand each time (a skill) or filled from a spec (a tool).
