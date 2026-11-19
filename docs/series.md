# Running a series across sessions

A one-off video lives in one project. A series (a channel's episodes) needs more: a look every episode
shares, characters and places that come back, and approvals that span weeks. This is the shape that worked
for Ireland in the Telling (2026-11), written so another series can copy it.

## A hub folder per series

`projects/<series>/` holds what every episode shares; each episode's video is an ordinary toolkit project
(`projects/<series-prefix>-<episode>/`, with `project.json` naming the series), so `/video`, `qc.py` and
`/publish` work unchanged.

| In the hub | What |
|------------|------|
| `CLAUDE.md` | start here: what the series is, the rules, the pipeline, what's where |
| `series.json` | the state: channel, people, stages, characters, episodes, approvals, open questions, sessions |
| `series.py` | `status` (a one-screen dashboard), `check` (files on disk against the state), `stage`, `approve`, `log` |
| `BIBLE.md` | the series bible (below) |
| `characters/<id>/` | a character's description, the artist's originals in `art/` (never edited), derived files in `work/` |
| `sets/`, `props/` | the shared library |
| `episodes/<id>/` | the script, the brief, review notes |
| `DECISIONS.md` | what was decided, when and by whom |

Change state only through the script, and start every session with `status`. A fresh session then knows
where everything stands in one command.

## Approvals stay with people

Some decisions are people's: approving the bible, a character, an episode brief, the final cut, and making a
video public. The state records each one with who gave it and when, and the script refuses to move an episode
into a stage whose approval is missing (animation waits for approved characters, release waits for the final
cut). Record an approval only when the person has said it in words. Uploads stay private for review.

## The series bible

Write the bible before the first episode, and approve it, because every episode inherits it:

- **Look:** style frames (six to ten), palette, light, how characters are drawn, the set vocabulary.
- **Motion:** how the camera moves and how characters act; a ten-second motion test in the real pipeline.
- **Cast rules:** each character's recurring features (the ones to keep in every shot), from their sheet.
- **Sound:** the narration voice (picked by ear), the music's mood, how loud it sits under the voice.
- **Structure:** episode length, the opening and closing, titles, captions, chapters.
- **Words:** tone of the narration and of titles and descriptions (all through `copy_lint.py`).

The channel art is a good first draft of the bible: it already chose a palette, a type and a feel.

## Build a library, not everything per episode

Characters and places recur, so make each once, get it approved, and reuse it; per episode, make only what
that story needs (a ship, a flood, a landing). Each episode then costs less than the one before. With Blender,
the library is the cast JSON (`kits/<name>/cast/`), sets and imported props; see `docs/blender.md`.

## Per episode

script (the writer) -> brief (scenes with word budgets, characters, animation route, voice, music, cost; the
owner approves) -> art (approved per character) -> narration -> assembly -> animation -> captions -> checks
(`qc.py --render --layout`, `sync_check.py`, a contact sheet) -> private upload -> final cut (owner) -> public
(owner) -> stats (`youtube_stats.py`).

Pilot the whole pipeline on the first sixty seconds of episode one before building the rest of it.
