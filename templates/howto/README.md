# howto: tutorial episodes with a shared opening and close

A Remotion template for how-to and tutorial videos about a tool: an episode in chapters, each scene drawn on the
words of its narration, and a 9:16 Short cut from every chapter. It ships as the finished build of the toolkit's
own **Getting started 1: What it costs** (2:35, four chapters), so `npm run render` works before you change
anything. For the next episode you replace `src/episode.ts`, `facts.json` and the narration; the opening, chapter
cards, close and scene kinds stay.

## The opening and the close

One object carries the series: the **mark**. It is a terminal prompt, a chevron and a caret.

- **Opening:** the chevron fills into a play triangle and a track draws out from the caret, with a stop for each
  chapter. The title rises over it, line by line on the narration, and the stops light as the narration names
  what is coming.
- **Chapter cards:** the same track in the same place; the caret is now the playhead and moves to the chapter's
  stop. The card takes the chapter's colour.
- **Close:** the playhead runs to the end of the track while "next" is said, the track rolls back into the caret,
  the triangle opens into a prompt again and the caret blinks. The episode ends where it started: at a prompt.

The geometry (`railGeometry` in `src/howto/parts.tsx`) is shared, so the track never moves between them.

### With a brand's own mark and ident

A brand can carry a flat mark (`brands/<name>/assets/mark.svg`) and an ident (`brands/<name>/ident/`, a short
open and close with their sound; `docs/creating-brands.md`, "Mark and ident"). `scripts/ident.py` reads both for
the brand `src/brand.ts` names, copies the clips to `public/ident/` and writes `src/ident.json`:

```bash
uv run scripts/ident.py                          # the brand's default ident
uv run scripts/ident.py --episode key --short clap
uv run scripts/ident.py --none                   # back to the prompt mark above
```

- **The mark** takes the chevron's place at the head of the track, in the colours for each field.
- **An episode** opens with the ident's open clip, then the 2D opening from where its track draws. The close
  holds on the link, then the ident's close clip ends the video.
- **A Short** still opens on its chapter card (the cover and the hook) and ends on the ident's close clip, with
  the episode's name and the link over it.
- **Sound:** the ident plays its own, and the music bed starts after the open clip and is out before the close.
- **Levels:** an ident is mixed to sit beside narration at the toolkit's scene target, -16 LUFS. Keep the
  narration files there (measure with `ffmpeg -i file.mp3 -af ebur128 -f null -`): with a voice far under it the
  ident lands much louder than the voice, and the final normalising pass pumps the quiet close up.

The template ships filled from `brands/video-toolkit`. With `src/ident.json` as `{}`, or `ident: false` in
`src/episode.ts`, everything above this heading plays instead; nothing needs Blender either way, because the
clips are already rendered.

## One conversation

An episode plays as a conversation with an agent. Give a chapter an `ask` and its card types that question at a
prompt (`ask: 'what do the AI models cost?'`, under about 28 characters), and the chapter's first scene opens out
of the prompt as the answer. Without `ask` the card shows the chapter's title. A pen circles the value of the row
or bar being said and underlines the newest line under a figure, with no setup: it follows the cues the scene
already has. A ledger's `hero` keeps a figure's number on screen into the next scene.

In a Short the burned captions own a band low in the frame (their baseline is 330 px from the bottom). A scene
draws nothing under `CAPTION_TOP` (`src/howto/scenes.tsx`) but its source note: a line placed lower is written
over by the captions, and the horizontal render will not show it. After a change to a scene, look at a late frame
of its Short as well as the episode's.

- **A hook before the title.** `hook: { ask, scene }` in the episode plays a typed question and one scene before
  anything else, with no music: the most interesting thing the episode has to say (here, our real bill). The
  brand's ident and the title follow it, and the bed comes in with the title. Its narration is scene id `hook`.
- **Show the thing.** A ledger row can carry `show: { video }`, `{ images: [...] }` or `{ wave: true }`: a window
  beside the rows plays the clip, flicks through the images or draws a moving waveform while the narration is on
  that row. The pilot's are real outputs: an LTX-2.3 clip from `examples/sky-blue-short` and four FLUX.2 images
  (`public/buys/`).

## Scene kinds

| Kind | Shows | Lands on the narration |
|---|---|---|
| `ledger` | A receipt: rows of label, detail, dotted leader and value (a number counts up) | each row, its detail and its value on their own words; `allAt` lights every value ("all free") |
| `figure` | One big number, the lines that explain it underneath (replace, or `add` below) | the number on its word, each line on its words |
| `bars` | A few values against a limit line (a free allowance, a target) | each bar on its label's word, its amount on the amount's word; a closing line replaces the title |
| `statement` | Two or three lines | each line |
| `session` | An agent CLI session filling the frame (`TerminalSession`, any skin): the field is the terminal | every prompt, tool call, permission prompt, zoom |
| `command` | The one command to type, large, and what it does | typing starts on the word |
| `split` | Who does what: two columns ("You", "Claude Code"), a running count under each | each row on its words; a column's count takes the accent and the pen when it is said (`countAt`) |
| `screen` | A product's screen in a browser window, with a cursor and a camera (below) | every click, keystroke, page change and camera move |

A `split` row can stand for more than one thing (`count: 3` for "checks Node, uv and FFmpeg"), so the rows stay
few and the counts stay true to what was counted. The count climbs to each row's share over a few frames, and a
row can carry a `detail` line that lands on its own words (`detailAt`): a pointer, a condition, what the row is made of. A column holds six rows beside a title in 16:9 and eight
without one; in a Short the two columns stack, each count at the end of its head's line.

A ledger row's `show` window sits under the `hero` when there is one, and in the hero's place when there isn't,
so a ledger can show the thing without having a number to carry. One picture in the window is pushed in slowly;
a still that follows a moving waveform reads as a clip that froze (QC's "motion stops dead").

Every scene's props are a function of `cue`, so no frame number is typed by hand: `at: cue('three hundred')` is
the frame the narrator says it. `short` on a scene gives replacement props for the 9:16 Short (titles broken by
hand for the narrow frame).

## Screens: a product on the chapter's field

A `screen` scene shows an app or a website with no screen recording: the screen is static React rebuilt from the
product's source, and a scripted cursor and camera perform the step (the method is the `synthetic-ui-demo` skill;
the window is `Stage` from `lib/components`). The template supplies the scene; **the screen kit stays in your
project** (`src/kit/`), because it is the product's, not the template's.

`src/screens/example.tsx` is a whole one to copy from: a made-up settings page (the kit), a scene that types an
address, flips a toggle and saves, and its Short. Studio shows it as `Screen-example` and `Screen-example-short`,
timed by stand-in cues instead of narration; delete the file and those two compositions in `src/Root.tsx` once
your own screens are in.

```tsx
// src/screens/form.tsx: a scene is a function of cue, like every other kind
import { boxMid, visitPath } from '../../../../lib/components';

export const form = (cue: Cue): ScreenProps => {
  const T = { open: cue('schedule a call'), title: cue('give') };
  const { path, clicks } = visitPath({ x: 820, y: 640 }, [        // a visit per cue: there before it, click on it
    { at: T.open, to: boxMid(SCHEDULE_BTN), click: true },
    { at: T.title, to: boxMid(FORM.title), click: true },
  ]);
  return {
    screen: <FormScreen T={T} path={path} />,                      // reads useCurrentFrame() itself
    url: [{ at: 0, url: 'app.example.com/dashboard' }, { at: T.open + 8, url: 'app.example.com/schedule/create' }],
    path,
    clicks,
    camera: [{ f: 0, z: 1, x: 720, y: 405 }, { f: T.title, z: 1.65, x: 560, y: 300 }],
  };
};
// src/episode.ts
{ id: 'form', kind: 'screen', chapter: 0, props: form, short: (cue) => ({ camera: [...] }) },
```

- **Everything is a cue.** Clicks, typing, page changes and camera keys are frames looked up from the narration,
  so the scene follows the voice when a line is re-voiced. Positions are the kit's canvas pixels (1440 x 810).
- **On the field.** The window sits on the chapter's field with a hairline round it (no shadow), its edges on
  the title edge. The episode badge is left off: the screen is the picture.
- **The address bar** takes one address or a list by cue, so a click can lead from a website to the app
  (`url: [{ at: 0, url: 'example.com' }, { at: T.start, url: 'app.example.com/call' }]`). A website is a screen
  like any other: rebuild the page from its own markup and give the scene the site's address.
- **In a Short** the whole window is too small to read, so the scene crops it to a box above the captions
  (`screenBox`, nothing under `CAPTION_TOP`) and the camera stays on what is being talked about. With no camera
  in `short`, it follows the cursor at `follow` (default 3, about 30 px for 16 px text), holding still for small
  moves. When one thing is wider than the box (a long link beside its label), give `short` a camera at a lower
  zoom: the box shows `920 / (0.64 x zoom)` canvas pixels across.
- **A page no click leads to is a new scene.** When the narration moves on to a page the cursor did not open
  (the call's page after saving, a dashboard after sign-up), end the scene and start another with its own id
  and address: the cut does the move. A faked click to get there shows a step the product doesn't have.
- **Clicks are heard.** Each frame in `clicks` plays `public/audio/sfx/click.wav` (`sounds: false` turns it off).
- **People.** If the screen shows people who are not real, set `note: 'People shown are AI-generated'`. It sits
  over the picture's corner; `{ text, until }` takes it away on a cue when the people leave the screen.

Before rendering, look at a still at each click frame: a target is easily 20 px off.
`uv run scripts/stills.py --at "form@schedule a call@0,form@give@0"` takes stills on cues and tiles them. Then
`qc.py --layout` on the composition: the window is marked `data-qc="product"`, so the product's own small text is
reported apart from the video's.

## A series in one project

Episodes that share a kit and a brand can live in one project. Give each an `id` and list them:

```ts
// src/episode.ts
export const EPISODES: Episode[] = [EP1, EP2];     // each { id: 'ep1', ... }
```

- Narration goes in `public/audio/scenes/<id>/`; `scripts/timing.py` writes each under `episodes.<id>` and takes
  `layout.json` per episode as `{ "<id>": { "open": { "lead": 40 } } }`.
- Compositions are `Episode-<id>` and `Short-<id>-N`: `npx remotion render src/index.ts Episode-ep1 out/ep1.mp4`.
  `npm run render:shorts -- --episode ep1` renders that episode's Shorts to `out/ep1/`, and
  `scripts/chapters.py --episode ep1` writes its chapters.
- With one episode and no `id` nothing changes: `Episode`, `Short-N`, `public/audio/scenes/`.

## Files

| Path | What | Next episode |
|---|---|---|
| `src/episode.ts` | Series, number, title, chapters (each with its field), scenes, next episode, music | Replace |
| `facts.json` | Every number the episode shows or says, with source and the date checked | Replace, check on render day |
| `public/audio/scenes/` | One sentence per file: `<NN><letter>_<scene>.txt`, its `.mp3` and `.captions.json` | Replace |
| `public/audio/layout.json` | Lead, gap and tail frames per scene, where the defaults don't suit | Adjust |
| `public/audio/music/` | The bed (`bed_2.mp3`, ACE-Step, chosen by ear against the picture with `audition.py`) | Keep or replace |
| `scripts/timing.py` | Lays the phrases on the timeline and puts their word times in `src/timing.json` | Keep |
| `src/howto/timeline.ts` | The timeline itself (segments and their lengths), with no React in it, so scripts can read it | Keep |
| `scripts/timeline.mjs` | Prints that timeline as JSON: every segment's start frame, every chapter's | Keep |
| `scripts/chapters.py` | YouTube chapters from the timeline, to `out/publish/chapters.txt` | Keep |
| `scripts/render-shorts.mjs` | Renders a Short per chapter (`npm run render:shorts`), stops on a failure | Keep |
| `scripts/level.py` | Levels every narration phrase to -16 LUFS, so no line jumps out | Keep |
| `scripts/stills.py` | Stills of every scene, or at chosen cues, tiled into one sheet before a render | Keep |
| `scripts/accent.py` | Screens the narration for words a British voice said with an American r | Keep |
| `src/howto/` | Engine: timeline, scene kinds, the mark and track, theme | Keep |
| `src/screens/example.tsx` | A `screen` scene and its kit to copy from, previewed as `Screen-example` | Delete once yours are in |
| `src/brand.ts` | The one place the brand is chosen | Keep, or point at your brand |
| `scripts/ident.py` | Reads the brand's mark and ident into `src/ident.json` and `public/ident/` | Run again when the brand changes |
| `scripts/sounds.py` | Synthesises the sounds things land on, and the click, into `public/audio/sfx/` | Keep, or replace the files; `sounds: false` in the episode turns them off |
| `design.md` | Acts, fields, type steps, ground, layout: the decisions the code follows | Keep |

## Making an episode

From the toolkit root:

```bash
mkdir -p projects && cp -r templates/howto projects/my-episode && cd projects/my-episode
rm -rf node_modules out public/audio/scenes/* && npm install  # not the pilot's renders, narration or packages
# 1. write one sentence per file in public/audio/scenes/ (00a_open.txt, 01a_free.txt, ...), then lint it
uv run ../../tools/copy_lint.py public/audio/scenes/*.txt
# 2. voice it, level it, word-time it, lay it out
uv run ../../tools/voiceover.py --brand video-toolkit --scene-dir public/audio/scenes --json
uv run scripts/level.py                                        # every phrase to -16 LUFS
uv run ../../tools/captions.py public/audio/scenes
uv run scripts/timing.py
# 3. edit src/episode.ts (scenes and cues) and facts.json; preview, and look at every scene before a render
npm run studio
uv run scripts/stills.py                                       # out/stills/Episode-late.jpg
# 4. render, check, normalise
npm run render && npm run render:shorts
uv run ../../tools/qc.py --render out/episode.mp4 --fix        # writes out/episode.loudnorm.mp4 at -14 LUFS
uv run ../../tools/contact_sheet.py out/episode.loudnorm.mp4  # one frame per scene, for the critique
# 5. for YouTube: subtitles from the render, chapters from the timeline
uv run ../../tools/publish_kit.py --project . --video out/episode.loudnorm.mp4 --no-chapters
uv run scripts/chapters.py --video out/episode.loudnorm.mp4    # out/publish/chapters.txt
```

`npm run render:shorts` renders one Short per chapter, however many the episode has. Arguments after `--` go to
every render (`npm run render:shorts -- --log error`), and `--only 2,4` renders just those chapters. It stops at
the first render that fails and checks that each file was written by this run. Check that yourself after any
render command that went wrong: a render that did not run leaves the last one's file in `out/`, and that file
passes QC.

Chapters come from `scripts/chapters.py`, not from `publish_kit.py`: the publish kit makes a chapter per narration
file, and here a file is one sentence. The script reads the timeline the render plays, so a chapter starts where
its card does; the stretch before the first card (hook, ident, title) is the 0:00 chapter, named by the hook's
question. `--titles ask` names chapters by their typed questions. With `--video` it refuses to write chapters for
a render whose length is not the timeline's.

A copied `node_modules` has broken links (`remotion: MODULE_NOT_FOUND`), which is why the first line removes it
before `npm install`. `npx remotion compositions src/index.ts` lists the lengths but does not look up a single
cue: a cue is read when its scene draws. Render one still late in each scene
(`npx remotion still src/index.ts Episode out/still.png --frame N`, frames from `node scripts/timeline.mjs`) and
every cue in the scene is checked, in seconds, before a render that takes minutes. Do the same for each Short.

### Voicing with takes, a tag and a check

One take per line is step 2 above. For a narrator who drifts (pace, or accent on a word), make three and pick:

```bash
# text sent for voicing carries a direction the voice does not read aloud; the .txt in scenes/ never does
for t in 1 2 3; do mkdir -p public/audio/takes/t$t; done      # write <name>.txt in each: "[crisp British English accent] The line."
for t in 1 2 3; do uv run ../../tools/voiceover.py --brand video-toolkit --scene-dir public/audio/takes/t$t --json; done
# put the spoken words back in every take's .txt, THEN pick: the picker counts the tag's words as speech, reads
# every line as 30 to 60 wpm too fast and slows nearly all of them (31 of 33 lines on episode 2 before this)
uv run --extra prosody ../../tools/pick_takes.py --takes-dir public/audio/takes --output public/audio/picked --json
uv run --extra prosody scripts/accent.py                       # after captions.py: words said with an American r
```

Then level each picked file to the scene target with one gain and a limiter
(`volume=<gain>dB,alimiter=limit=0.891:level=disabled`, the gain from `ffmpeg -af ebur128`), copy it to
`public/audio/scenes/` with its spoken-words `.txt`, and run `captions.py` and `scripts/timing.py`. Any change to
an MP3 needs its captions made again.

A narration file does not have to be the narrator. Episode 2 opens on the voice line its own setup made
(`00a_hook.mp3` is that file, levelled, with its words in the `.txt`), so the result is heard before it is
explained, and `cue()` finds its words like any other line.

### A session scene is a copy of a real run

A `session` is drawn, so it can show anything, which is the reason it has to show only what happened:

- Run the real commands first, in a fresh clone, and save each one's command line, output, exit code and time to
  its own file in the project's `capture/` folder. Read build and call times off those files (a Modal deploy
  prints `App deployed in 590.573s`), not off a clock.
- `facts.json` lists each number with the capture file it came from, and marks every line that is **quoted**
  instead of captured: a step that needs a person at a browser (a first login, adding a card) is quoted from the
  tool's own source or docs, never invented. The agent's own sentences are quoted from the command file it
  follows, and the description says its wording differs from run to run.
- Count from the capture. If the video says "ten things you do", `facts.json` holds the ten, and a `split` scene's
  row counts add up to the same number.
- Nothing private on screen: no key values, account email, workspace name (it is in every Modal endpoint URL: cut
  the URL short) or home path. Read every session frame for this before the render is shown to anyone.

`open` and `close` are scene ids for the narration of the opening and the close. Both are optional: with no `open`
lines the opening is the title for about three seconds, which is what an episode with a hook wants. Scene files sort by name, so
`NN` sets the order. A cue that isn't in the narration fails the bundle with the words that are, so a rewritten
line can't put an animation on nothing.

## Fields and type

Three fields from `design.md`: `explain` (light), `work` (navy, the terminal's own colour) and `identity`
(brand blue, the close). A chapter names its field; colour changes only at a chapter turn. A brand can set any
value of a field in its `brand.json` (`"fields": { "work": { "ground": "#323E66" }, "explain": { "hair":
"transparent" } }`): its own dark ground, or a flat ground with no rows. A title with a long line takes
`titleSize: 'figure'` in the episode, since the opening sets it at display size. Inter and JetBrains
Mono are bundled in `public/fonts/` (SIL Open Font License) so a render is the same on every OS.
