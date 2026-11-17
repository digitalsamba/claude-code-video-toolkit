# Lecture

A presenter at a chalkboard: a fitted 3D character rendered in Blender, standing in front of a board that Remotion
draws, chalking points on the narration's words, turning to the board as each heading goes up, and shown small in
an inset while the camera goes into a window on the board. It needs Blender (optional in the toolkit: without it
the board, the chalk, the windows and the narration still render, and the presenter is left out).

A lecture is three things and no code of its own:

| File | What it is |
|---|---|
| `public/audio/scenes/<id>.txt` | what each segment says, one file a segment |
| `lecture.json` | the presenter, and per segment its heading, what is chalked and when, the windows, the acting |
| `timeline.json` | written by `tools/lecture.py timeline` from the audio: when each segment starts, its words by frame |

The code is `lib/lecture` (the board) and `tools/lecture.py` (the presenter's clips). This template ships as the
opening of "Introducing Blender support", with Mara, the invented character in `tests/fixtures/blender-fitted`.

## Make one

Run everything from the toolkit root. A full lecture is hours of rendering over a few sessions; three segments
(about a minute) render in 15 to 25 minutes on an Apple M4.

```bash
mkdir -p projects && cp -r templates/lecture projects/my-talk     # leave node_modules behind if the template has one
cd projects/my-talk && npm install && cd ../..

# 1. the presenter: fit Mara once (2 minutes), or point lecture.json "presenter.cast" at your own character
uv run tools/blender_render.py fit --model tests/fixtures/blender-fitted/mara.glb \
    --fit tests/fixtures/blender-fitted/fit.json --out projects/my-talk/cast/mara-fitted.blend

# 2. the words: edit public/audio/scenes/*.txt, copy-lint, voice, caption
uv run tools/copy_lint.py projects/my-talk/public/audio/scenes/*.txt
uv run tools/voiceover.py --scene-dir projects/my-talk/public/audio/scenes --voice-id <id> --model eleven_v4 \
    --stability 0.45 --max-wpm 168 --json
uv run tools/captions.py projects/my-talk/public/audio/scenes

# 3. the timeline, then the presenter's clips (check is seconds a clip; render is the long part)
uv run tools/lecture.py timeline projects/my-talk
uv run tools/lecture.py specs projects/my-talk       # says every beat it dropped, and why
uv run tools/lecture.py check projects/my-talk       # each clip, the board share, the pose at every join
uv run tools/lecture.py render projects/my-talk      # one clip at a time, with an estimate first

# 4. the board stills, the inset's backdrop, and the video
cd projects/my-talk && npm run board && cd ../.. && uv run tools/lecture.py bg projects/my-talk
cd projects/my-talk && npm run render && cd ../..
uv run tools/qc.py --project projects/my-talk --render projects/my-talk/out/lecture.mp4
```

`npm run studio` shows the board at any stage. Change a word in the script and run steps 2 and 3 again: every
frame (the chalk, the windows, the acting) is looked up from the captions, so nothing has to be moved by hand.

## lecture.json

```json
{"title": "My talk", "presenter": {"cast": "cast/presenter.json"},
 "segments": [{
   "id": "01_hello", "heading": "Blender support",
   "board": [
     {"at": "a 3D model", "chalk": "a 3D model", "x": 80, "y": 380, "size": "l"},
     {"at": "one file", "stroke": "under", "points": [80, 560, 96, 380]},
     {"at": "Here is", "pane": {"x": 60, "y": 330, "w": 1080, "h": 420, "title": "...", "code": "...", "out": "Before"}}],
   "windows": [{"in": "Here is", "out": "Before"}],
   "acting": [["fitted to its rig", "show", [300, 420]], ["one file", "gesture", "count"], ["by hand", "look", "content"]]
 }]}
```

- **Times** (`at`, `in`, `out`): words of that segment's narration (when they start), `{"cue": ..., "nth": 2,
  "offset": -4}`, `"start"`, `"end"`, or a number of frames after the first word. A cue that is not in the
  narration fails with the words that are.
- **Board items**: `chalk` (lettering written from x, y: sizes `xl l m s xs`, `color` white or yellow, `weight`,
  `tilt`), `stroke` (`line` and `arrow` take x1 y1 x2 y2, `box` x y w h, `ring` cx cy rx ry, `tick` and `cross` x y
  size, `under` x y text-size width, or `path` with `d`), `tap`, `pane` (a clean panel for something real: an
  `image` from public/ or a `code` listing, with a `title` and a `source` line), and `component` for a drawing of
  your own in `src/custom.tsx`. Board pixels are 1920x1080. Keep chalk left of x 1250: the presenter stands right.
- **Windows**: the camera goes into the board (1.4x on the left part, by default) from `in` to `out`; the
  presenter is out of shot, so they are shown in an inset at the bottom right. Keep a window's content left of
  x 1150 or above y 760 so the inset covers none of it.
- **Acting**, on the narration's words: `gesture` (`open`, `both`, `count`, `shrug`), `show` (they turn and point
  at a board pixel; in a window it becomes a gesture to the hall, so the inset keeps their face), `look` (at the
  lens, with a face: `happy`, `content`, `worried`, `neutral`), or `beat` with any verb from the blender skill.
  Two beats that would use the same arm at once: the later one is dropped, and `specs` says so.
- **The heading**: chalked as each segment starts, while they turn to it. That turn starts in the clip before
  (`rules.carry`), so the clips join in the middle of a move, not on a stance they return to each time.

## Staging and rules

`"staging"` and `"rules"` in lecture.json override the defaults in `lib/lecture/defaults.json`:
where they stand (a board column and metres out from the wall, in the hall's measure), the part of the frame their
clips cover (`window`), the board camera's lens, the gap between segments, how long a gesture is held, how often they
look out at the hall, what crosses a join (`carry`: `board` or `still`).

The hall is sized by the presenter: a fitted character of 2.25 m and one of 1.65 m stand at the same size beside
the board, and `check` fails a clip if the presenter hides more than 10% of the board.

`rules.close` (off by default) ends the last segment with the chalk put down on a hidden table, the hands dusted,
a look and a smile. The table's place (`staging.table`) has to suit the presenter's arms: on the two bodies it was
tried on, one put the chalk down cleanly and the other was 8 cm short or put a hand into the table top. Turn it
on, run `check`, and move the table until both its notes are gone.

## What is open

- The presenter does not write: lettering chalks itself in while they point at it.
- The close's table is placed by hand (above).
- `lib/lecture` imports Google Fonts from the project's `node_modules`; `remotion.config.ts` points webpack at it
  with `process.cwd()` (with `__dirname` it pointed elsewhere and the bundle could not find the package).

The sounds in `public/audio/sfx/` are CC0 recordings from Freesound; each has its `.attribution.txt`.
