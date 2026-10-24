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
- **Keep the stage between y 300 and 1090.** Max stands from 1125 to 1515 (side layout), live numbers go
  in `<Aside>` to his right, and the captions start at 1540.
- **Every key number gets a moment:** a `Camera` push-in or a `StatRow pulse`, plus an `Sfx` cue.
- **Type size floor:** about 40px for anything the viewer must read. Chart ticks use 40px Inter.
- `LineChart` uses fixed SVG ids (`plot-clip`, `glow`): one chart per visual.
- Run tools from the toolkit root (`uv run tools/...`), not from this directory.

## Visual language (the brand's DNA, upgraded)

- Navy backdrop with aurora glows and a drifting dot grid. Never flat.
- Charts always sit in an `EditorPanel` with a `.plot` filename.
- Crimson means Max, losses and the key result. Steel blue means axes, labels and "everyone else".
  Cream is the text.
- Mono (JetBrains) for labels, numbers in tabs and the chapter badge. Inter for narration and
  titles. STIX Two for maths.
- The narration line is terminal-style (crimson `›`, block cursor), but words land whole: no letter-by-letter typing.
