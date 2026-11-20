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
  `image` or a `video` from public/, or a `code` listing, with a `title` and a `source` line; a video plays muted
  from the pane's `at`, `"from"` seconds in, and holds its last frame, its length read from the file by
  `timeline`), and `component` for a drawing of
  your own in `src/custom.tsx`. Board pixels are 1920x1080. Keep chalk left of x 1250: the presenter stands right.
- **Windows**: the camera goes into the board (1.4x on the left part, by default) from `in` to `out`; the
  presenter is out of shot, so they are shown in an inset at the bottom right. Keep a window's content left of
  x 1150 or above y 760 so the inset covers none of it.
- **Closer shots**: `"closes": [{"in": "and I'm made", "out": "The toolkit fitted", "type": "closeup"}]` cuts to a
  second camera on the presenter from one phrase to another (`out` defaults to the segment's last word). `closeup`
  is head and shoulders with the hall soft behind; `medium` is from the waist, with room for a gesture. Each is
  the segment's own clip seen from a camera placed round the presenter (`az`, `dist`, `h`, `lens`, `push` and `dof`
  move it, as on any shot in the blender skill), rendered with the hall and the board as it stands then, so the
  body is in the same pose either side of the cut and they look into the new lens as it cuts in. Use one for a
  line said to the viewer, a few seconds long; `timeline` refuses one while the camera is in a window, while the
  heading goes up, or within two seconds of another, and says which. Lettering chalked during a close is on the
  board when the cut comes back. After `specs`, run `uv run tools/lecture.py boards projects/my-talk` for the
  board stills behind them. Each renders at 1920x1080 at the room shot's speed (about 4 s a frame on an Apple M4).
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
look out at the hall, what crosses a join (`carry`: `board` or `still`), and their shadow on the board
(`shadow`: straight down and soft, as the ceiling lights throw it, faded in over two seconds after an opening).

Type on the board: every heading is the `l` size on one edge, so keep a segment's first line at y 360 or lower.
Keep to three chalk sizes in a video (`xl` for a word or two, `l`, `m`) and nothing said aloud below `m`: with the
pane's title and source line that is the five sizes a video should hold (the `video-design` skill).

The hall is sized by the presenter: a fitted character of 2.25 m and one of 1.65 m stand at the same size beside
the board, and `check` fails a clip if the presenter hides more than 10% of the board.

`rules.close` (off by default) ends the last segment with the chalk put down on a hidden table, the hands dusted,
a look and a smile. The table's place (`staging.table`) has to suit the presenter's arms: on the two bodies it was
tried on, one put the chalk down cleanly and the other was 8 cm short or put a hand into the table top. Turn it
on, run `check`, and move the table until both its notes are gone.

## Opening on the hall

`"opening": {}` in lecture.json starts the lecture on the whole hall, in the dark: it fades in from black onto
the hall lit only by its windows, the door opens and the corridor's light comes in round the presenter, they step
in and press the light switch beside the door, and the ceiling lights come on in three banks from the board back,
each with a fluorescent stutter and a clunk. Then they walk to their mark while the camera comes down the hall
from the back and lands on the board camera's framing on the frame the walk ends. It cuts to the board, which
looks the same, and the first word comes half a second later. 9 s to the cut.

`"opening": {"lights": "on"}` is the hall lit from the start: the latch clicks on black, the hall holds still for
a second, the door opens and the presenter walks in (about 7.8 s); its times are under `"lit"` in the defaults.
`"banks": 1, "stutter": false` switches everything on at once, with one click.

- `tools/lecture.py specs` writes the room shot as `blender/presenter/E1.json`: the first clip's own spec with the
  hall drawn and dressed (rows, windows, lights), so the walk in is acted in both and the pose at the cut is the
  same by construction (`check` measures it: the join `E1->P01`). The first clip keeps the opening as its lead-in.
- The board on the wall is `public/board/title.png` (`npm run board`): the title stays up after the cut and goes
  with the first segment's board.
- The camera's arrival is keyed to the walk's end, not to a typed time: change `walk` and the camera follows.
- Keys (seconds from black; defaults in `lib/lecture/defaults.json` "opening"): `lights` (`"switch"` or `"on"`),
  `seconds` (the cut), `first_word` (after the cut), `voice` (`{"id": "00_hook", "at": 0.7}`: a line said over the
  walk in, from `public/audio/scenes/00_hook.mp3` and its captions, so the video's first words come in its first
  second and not after the walk; it must end before the first segment starts), `latch` (none in the dark; a number brings the click back),
  `fade` ([start, length]), `door`, `step_in` ([start, end]: in at the door to the switch), `switch_spot`
  (`"reach"`: where their own arm presses it straight, or metres out from the wall), `switch_facing`, `switch`
  (the press), `banks`, `stutter`, `gap` (seconds between banks), `walk` ([start, end], to the mark), `behind`
  (where they wait, metres behind the door), `stride`, `heading`, `camera` (the start: `cam` and `aim` in the
  hall's measure, `lens`), `hall` (the set's `hall` options: `windows`, `audience`, `colors`, `corridor`, `spill`,
  `board_dark`, ...), and `sounds` for the latch, the door, the footsteps, the switch, each bank's stutter and its
  clunk (CC0 recordings in `public/audio/sfx/`, laid on the room shot's own event frames).
- It renders at 1920x1080, opaque: about 14 minutes on an Apple M4 (258 frames at 3.0 to 3.2 s a frame, either
  body) (`render` says before it starts).

## Shorts

A Short is some consecutive segments of the same lecture, made vertical. It uses the same clips, audio and
captions, so nothing is rendered again in Blender and nothing is voiced again:

```json
"shorts": [{"id": "own-character", "segments": ["08_capture", "09_photo_to_character"], "cover": ["09_photo_to_character", 3],
  "board": {"08_capture": [
    {"at": "It starts with photos", "chalk": "8 photos for your 3D double", "x": 80, "y": 370, "size": "l"},
    {"at": "The toolkit opens a page", "pane": {"x": 60, "y": 290, "w": 1000, "h": 646, "title": "The capture page",
      "video": "capture/capture.mp4", "out": "The photos stay",
      "takeover": {"size": [1280, 720], "label": "tools/reference_capture.py", "keys": [
        {"at": "The toolkit opens a page", "region": [0, 0, 1280, 720]},
        {"at": "eight poses", "region": [150, 100, 520, 742]},
        {"at": "keeps the sharpest", "region": [894, 100, 320, 456]}]}}}]}}]
```

```bash
cd projects/my-talk && npm run board      # also writes public/board/bg-short.png, the slate at 1080x1920
npx remotion render src/index.ts Short-own-character out/short.mp4 && cd ../..
uv run tools/qc.py --render projects/my-talk/out/short.mp4 --fix
```

Each short is a 1080x1920 composition, `Short-<id>`. A straight 9:16 crop of the long video cuts either the board
or the presenter, so the layout is rebuilt:
- **The board's chalk** is a column at the top: board pixels from x 10 to 1250, scaled to the width (chalk `l`
  comes out at 54 px and `m` at 39 px, both above the 28 px a phone reads).
- **The presenter** is below it, full width, over one slate drawn at the vertical size (its tray where their clip
  puts the board's). They stay in frame through the windows, so there is no inset and the camera does not move.
- **A pane** goes full width wherever the long video goes into a window, centred between the heading and the
  captions, with its title and source line 1.3 times their size.
- **A pane that takes over** (`takeover` on the pane) fills the frame under the board's lines: the presenter steps
  out, and the view moves to each `region` ([x, y, w, h] of the picture, `size` its pixels) on its word. Show the
  whole thing for a second or two, then punch in on what the narration names, and end it on a word that brings the
  presenter back. A `label` (the file name) shows while the whole picture is up and goes as the view moves in.
- **The captions** (`<Captions>` with `phrases`, from `public/audio/scenes/*.captions.json`) stay in one place for
  the whole short, above the presenter's head, and break at commas, not mid-phrase. Shorts are watched muted.
- **Its own board:** `"board": {"08_capture": [...]}` on the short replaces that segment's items in the short only,
  for one line that says what the step is (and a takeover); the long video keeps its own.
- **Frame 0 is the cover.** The first segment's heading is already written there. `cover` puts one of the short's
  panes (segment id, then the pane's place among that segment's panes, from 0) on frame 0, faded out by the first
  word. Use it when the presenter is mid-turn at the start, as they are after a carried heading turn.
- It starts at the first segment's clip and ends `tail` frames (12) after the last word. A pane that would fade out
  in those frames stays up.

- **A Short with its own words** (a new hook said to camera, a shorter step) is its own lecture: copy the project
  (leave `out/` and `public/blender/` behind), give it the new segments' `.txt`, copy the unchanged segments' `.mp3`,
  `.txt` and `.captions.json` under their new ids, voice and caption only the new ones, then `timeline`, `specs`,
  `check`, `render` and the short as above. The presenter's mouth follows the narration, so a new spoken line is a
  new clip; a Short of three segments (about 47 s) renders in about 14 minutes on an Apple M4.
- A takeover that runs to the last word (a walk in that lands) holds to the short's end: give the short a `tail`
  long enough for the video to finish.

Every layout number is under `"short"` in `lib/lecture/defaults.json`, and any of them can be set on one short.
`qc.py --layout` does not read a lecture: read stills (`npx remotion still ... --frame=N`) instead.

## What is open

- The presenter does not write: lettering chalks itself in while they point at it.
- The close's table is placed by hand (above).
- `lib/lecture` imports Google Fonts from the project's `node_modules`; `remotion.config.ts` points webpack at it
  with `process.cwd()` (with `__dirname` it pointed elsewhere and the bundle could not find the package).

The sounds in `public/audio/sfx/` are CC0 recordings from Freesound; each has its `.attribution.txt`.
