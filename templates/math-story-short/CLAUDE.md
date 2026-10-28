# math-story-short template

9:16 Remotion template for MaxLearnsMath stories. See README.md for the pipeline.

## Rules that keep it working

- **Numbers come from `src/episode/data.ts`, never typed into visuals.** Narration quotes them,
  so changing the seed means re-reading the checkpoints and updating the narration.
- **`audioFile` must stay a string literal** (`'audio/scenes/<id>.mp3'`). `sync_timing.py` and
  `qc.py` parse the config with regexes and skip computed values.
- **Scene `id` = audio stem.** `npm run scripts` writes `public/audio/scenes/<id>.txt`, and
  voiceover.py writes `<id>.mp3` beside it.
- **Frames inside a visual are relative to its group**, not the scene. Use `beats[i].from` and
  `at(beats, i, frac)`; never hard-code absolute frames.
- **Every key number gets a moment:** a `Camera` push-in, a slam-in with a `thud`, or a `StatRow pulse`.
- `LineChart` uses fixed SVG ids (`plot-clip`, `glow`): one chart per visual.
- Captions start at y 1540 and the bottom ~260px stays clear for the platform's own UI. Nothing else is fixed.
- Run tools from the toolkit root (`uv run tools/...`), not from this directory.
- **Follow the production defaults in README.md** (voice, brisk pace, 0.35s scene gaps, quiet music from
  frame 0, deliver `video.loudnorm.mp4`). They are the user's agreed house style for every episode.
- **Show stills, not renders, while iterating:** `npm run stills` writes one PNG per scene end to
  `out/stills/`, and `--frames 72,150` checks moments inside an animation. Look at a contact sheet
  before showing the user; don't render video until the visuals are approved.

## Layout: three archetypes, chosen per visual group

The scene used to default to "content on top, Max bottom-left, two or three stat tiles to his
right". That triptych makes every scene look the same and shrinks the thing the viewer must see.
**Pick one archetype per visual group, from what the content needs:**

| Archetype | When | Max | Zone | Live numbers |
|---|---|---|---|---|
| **Showcase** | The visual *is* the point: a sample-space grid, a chart, a proof, a simulation | `hidden` (or a `side` cameo only if the visual leaves the bottom-left free) | Full bleed, y 300–1480 (1180px) | Badges laid *on* the visual (a corner pill, a count under the grid), not a stat column |
| **Presenter** | Max reacts to one object or statement: setup, wrong intuition, reveal | `side`, `pose: 'point'` toward the stage | `<Stage>` y 300–1090 | `<Aside>` `StatRow column` **only if the values change during the scene** |
| **Set piece** | Hook, engagement, punchline: one prop and Max's face | `hero` | Stage, or a custom box from y 140 when the header is hidden (hook) | none |

Rules:

- **StatRow never repeats what the stage already shows.** If "1/6" and "1/11" are 200px tall on
  the stage, tiles saying "most say 1/6 · answer 1/11" are noise: leave the aside empty or make
  Max `hero`. Stats earn their place when they tick (a count building to 11, a bankroll draining).
- **No StatRow in two consecutive visual groups.** Vary: showcase → presenter → set piece.
- **Showcase visuals take the whole width.** A 6×6 grid gets 114px+ cells; a chart's plot area is
  at least 900px wide. If it only fits by shrinking under the sizes below, hide Max.
- `<Stage>` and `<Aside>` are helpers for the presenter archetype, not the layout of every scene.
  Showcase visuals position at absolute coordinates (see the pinned-formula pattern: strip at
  y 290, working area y 570–1480).
- Max glides between layouts and off-screen on his own (`MaxLayer`), so switching archetypes
  between groups costs nothing. He hops on a mood change: change the mood on every reveal.

## Type scale (phone screen, 1080×1920)

Nothing the viewer must read is under 44px. The biggest element of each scene is *huge*: one
hero element per scene, and it is obvious which one.

| Role | Size | Font |
|---|---|---|
| Hero numeral or verdict (`1/11`, `11`, `6`) | 200–330px | Display (Fraunces 900, `theme.fonts.display`), lining figures |
| Key phrase (`both are sixes?`, `Hand them two dice.`) | 100–120px | Display |
| Statement / question lead-in | 46–56px | Inter 700 |
| Working equations | 54–72px wrapper (KaTeX scales up ×1.2) | STIX via `<Tex>` |
| Eyebrow label (`THE COMMON ANSWER`, `STEP 1 · NUMERATOR`) | 28–32px, letter-spacing 3–4, uppercase | JetBrains Mono |
| Tags on objects (`fixed at 6`, `6 faces`) | 26–30px | JetBrains Mono on a crimson or navy pill |
| Stat tiles | label 30 / value 56–68 | Mono / Inter 800 |
| Chart ticks | 40px | Inter |

Inter stays for narration, captions, chapter titles and body copy. Display is for the number or
phrase the episode is about, never for paragraphs.

## Objects, not icons

The episode's subject (dice, cards, coins, balls, chips, a roulette wheel) is drawn as a
**physical object**, not a flat pictogram: gradient body, a thicker bottom edge, a gloss line,
soft drop shadow, and a glow when it matters. `src/components/Die.tsx` is the reference: copy its
structure for a new object. Three tones carry meaning everywhere the object appears:

- **ivory** (cream face, ink pips): a real, neutral object
- **crimson**: the special case, the fixed value, the answer
- **navy**: unknown, neutral filler, dimmed context (matches the panels)

Objects move like objects: they roll in spinning, land with a `thud`, tumble while idle
(`Math.sin(frame / 18)`), and the same object at the same size recurs across scenes so the viewer
tracks it. `EditorPanel` (the code-editor window) is for plotted data only; never frame an object
in it.

## Scene recipes

- **Hook (`hook: true`)**: a *set piece with a prop the narration is literally about* (an exam
  sheet with university stamps slamming on, a casino chip stack, a lottery ticket). Not a list of
  labelled pills. The header is hidden, so the box can start at y 140. One slam per named thing.
- **Engagement / pause**: imitate the platform's own UI so the viewer's thumb twitches: a pause
  glyph in a cream disc with ripple rings, "Pause. Think." in display type, a comment box that
  starts typing an answer with a blinking cursor, an arrow to the comments.
- **Wrong intuition**: the flawed reasoning shown with the objects (one die locked crimson, the
  other still spinning), the wrong number huge in cream, then a crimson marker strike through it
  with a screen shake, and the right number slamming in bigger.
- **Count every case**: the full sample space as objects, revealed in a diagonal wave; the
  qualifying cases light crimson one by one (with a ticking count); the answer case goes ivory
  with a glow; `Camera` pushes into it, then back out.
- **Formula**: pinned formula strip at the top that fills in as each piece is solved (`\tfrac`
  inside the strip to keep it short, `\dfrac` in the working area), the working area restates the
  pieces with the objects (a mini sample-space grid with the counted cells lit), then the cancel,
  then the result at 300px+.
- **Why we're wrong / insight**: the brain's shortcut with the objects (fixed die + its six faces,
  "6 possibilities") against the real filter (the 11 pairs, "11, not 6̶").
- **Punchline**: Max `hero`, the two objects large, the line in display type with the key words crimson.

## Visual language (the brand's DNA)

- Navy backdrop with aurora glows and a drifting dot grid. Never flat.
- Crimson means Max, losses, the fixed value and the key result. Steel blue means axes, labels and
  "everyone else". Cream is text and the ivory objects. Ink is pips and the pause glyph.
- Mono (JetBrains) for labels, tags, numbers in tabs and the chapter badge. Inter for narration
  and titles. STIX Two for maths. Fraunces 900 for hero numerals and punchlines.
- Charts sit in an `EditorPanel` with a `.plot` filename. Objects and grids sit directly on the backdrop.
- The narration line is terminal-style (crimson `›`, block cursor), but words land whole: no letter-by-letter typing.
- Glass panels (`rgba(11,60,92,0.5)` + hairline border) hold equations and legends; cream paper
  (`#fdf0d5` → `#f4e3c3`, ruled lines, crimson margin) is for things from the real world (exam sheets, receipts).
