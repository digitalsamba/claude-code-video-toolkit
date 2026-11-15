# howto: tutorial episodes with a shared opening and close

A Remotion template for how-to and tutorial videos about a tool: an episode in chapters, each scene drawn on the
words of its narration, and a 9:16 Short cut from every chapter. It ships as the finished build of the toolkit's
own **Getting started 1: What it costs** (2:34, four chapters), so `npm run render` works before you change
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

## Scene kinds

| Kind | Shows | Lands on the narration |
|---|---|---|
| `ledger` | A receipt: rows of label, detail, dotted leader and value (a number counts up) | each row, its detail and its value on their own words; `allAt` lights every value ("all free") |
| `figure` | One big number, the lines that explain it underneath (replace, or `add` below) | the number on its word, each line on its words |
| `bars` | A few values against a limit line (a free allowance, a target) | each bar on its label's word, its amount on the amount's word; a closing line replaces the title |
| `statement` | Two or three lines | each line |
| `session` | An agent CLI session filling the frame (`TerminalSession`, any skin): the field is the terminal | every prompt, tool call, permission prompt, zoom |
| `command` | The one command to type, large, and what it does | typing starts on the word |

Every scene's props are a function of `cue`, so no frame number is typed by hand: `at: cue('three hundred')` is
the frame the narrator says it. `short` on a scene gives replacement props for the 9:16 Short (titles broken by
hand for the narrow frame).

## Files

| Path | What | Next episode |
|---|---|---|
| `src/episode.ts` | Series, number, title, chapters (each with its field), scenes, next episode, music | Replace |
| `facts.json` | Every number the episode shows or says, with source and the date checked | Replace, check on render day |
| `public/audio/scenes/` | One sentence per file: `<NN><letter>_<scene>.txt`, its `.mp3` and `.captions.json` | Replace |
| `public/audio/layout.json` | Lead, gap and tail frames per scene, where the defaults don't suit | Adjust |
| `public/audio/music/` | The bed (`bed_1.mp3`, ACE-Step, chosen by `pick_music.py`) | Keep or replace |
| `scripts/timing.py` | Lays the phrases on the timeline and puts their word times in `src/timing.json` | Keep |
| `src/howto/` | Engine: timeline, scene kinds, the mark and track, theme | Keep |
| `src/brand.ts` | The one place the brand is chosen | Keep, or point at your brand |
| `design.md` | Acts, fields, type steps, ground, layout: the decisions the code follows | Keep |

## Making an episode

From the toolkit root:

```bash
cp -r templates/howto projects/my-episode && cd projects/my-episode && npm install
# 1. write one sentence per file in public/audio/scenes/ (00a_open.txt, 01a_free.txt, ...), then lint it
uv run ../../tools/copy_lint.py public/audio/scenes/*.txt
# 2. voice it, word-time it, lay it out
uv run ../../tools/voiceover.py --brand video-toolkit --scene-dir public/audio/scenes --json
uv run ../../tools/captions.py public/audio/scenes
uv run scripts/timing.py
# 3. edit src/episode.ts (scenes and cues) and facts.json; preview
npm run studio
# 4. render, check, normalise
npm run render && npm run render:shorts
uv run ../../tools/qc.py --render out/episode.mp4 --fix        # writes out/episode.loudnorm.mp4 at -14 LUFS
uv run ../../tools/contact_sheet.py out/episode.loudnorm.mp4  # one frame per scene, for the critique
```

`open` and `close` are scene ids for the narration of the opening and the close. Scene files sort by name, so
`NN` sets the order. A cue that isn't in the narration fails the bundle with the words that are, so a rewritten
line can't put an animation on nothing.

## Fields and type

Three fields from `design.md`: `explain` (light), `work` (navy, the terminal's own colour) and `identity`
(brand blue, the close). A chapter names its field; colour changes only at a chapter turn. Inter and JetBrains
Mono are bundled in `public/fonts/` (SIL Open Font License) so a render is the same on every OS.
