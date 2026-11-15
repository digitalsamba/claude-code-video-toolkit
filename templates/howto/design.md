# Design: howto (pilot: Getting started 1, "What it costs")

Source: ours (video-design act fields with the brand). Brand: `brands/video-toolkit` (site tokens: DS blue
#3771e0, navy, light grey). The idea comes from the subject: a terminal that makes video. The mark is a prompt
chevron and caret; the chevron fills into a play triangle and the caret becomes the playhead on a track with a
stop for each chapter. The opening draws the track, each chapter card moves the playhead, and the ending runs it
to the end and back into the caret, so an episode ends where it started: at a prompt.

## Acts

| Act | Scenes | Says | Field |
| --- | --- | --- | --- |
| Open | open | "Will this cost you money? Mostly, no." | work |
| 1 What's free / 2 Thirty dollars / 3 What we spend | free · thirty, buys · spent, extras | the facts, one at a time | explain |
| 4 Setting it up | setup, start | Claude Code does it; you add a card | work |
| Close | close | next episode, the repo | identity |

A template rule: explaining chapters are `explain`, chapters that happen in the terminal are `work`, the close is
`identity`. Fields change only at a chapter turn, so an episode has two or three cuts of colour.

## Fields

| Field | Ground | Ink | Data accent | Other data | Secondary ink | Hairline |
| --- | --- | --- | --- | --- | --- | --- |
| explain | #f1f2f4 | #121418 | #3771e0 | #c9cfdc | #4a5160 | ink 8% |
| work | #232d49 | #ffffff | #93b4f2 | #3d4a70 | #c9cfe0 | ink 7% |
| identity | #3771e0 | #ffffff | ink | #6d97e8 | ink 85% | ink 12% |

Light, dark, mid: neighbouring fields always differ in lightness. The terminal's own background is the `work`
ground, so a session fills the frame with no window: the field is the terminal.

## Type

Family: Inter (bundled, `public/fonts/inter`), weights 700 / 400. Mono: JetBrains Mono (bundled), for commands
and sessions only.
Steps (1080p landscape): display 200 · figure 132 · title 72 · label 36 · note 28.
Steps (1080x1920 Shorts): display 150 · figure 120 · title 64 · label 38 · note 30.
Display tracking -0.045em, leading 0.95. Figures tabular. Titles broken by hand, two lines at most.

## Ground

The terminal's rows: hairlines every 54 px (one row of a 36 px terminal), ink at 7-8%, on `statement`, `figure`
and chapter cards only. Ledgers, bars and sessions get none: they are the picture. The track (mark, line,
stops) is the one recurring object.

## Layout

Title edge: x = 120 (landscape), 80 (Shorts). Titles start at y = 120. The track sits at y = 790 (landscape),
1330 (Shorts), the same in the opening, every chapter card and the close.
Sources: one line, note size, bottom left at y = H - 72. Episode badge (series, number, toolkit version): note
size, bottom right, secondary ink, on every scene but the opening and close.

## Motion

Entrances 10 frames, spring damping 200, 14 px rise; nothing exits (the scene cuts). Every change lands on the
word that names it (`cue()`). The data accent moves to whatever is being said, and only one thing holds it.

## Rules turned off

- none
