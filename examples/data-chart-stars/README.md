# data-chart-stars

A 31s narrated chart video: the toolkit's GitHub star history, drawn by `lib/components/DataChart` and timed to
Lugh's voice. Nothing is pre-rendered: the line, axes, labels and bars are SVG, drawn frame by frame in the
Digital Samba brand.

```bash
npm ci
npm run studio      # preview
npm run render      # out/data-chart-stars.mp4
cd ../.. && uv run tools/qc.py --render examples/data-chart-stars/out/data-chart-stars.mp4 --fix   # -14 LUFS
```

Everything it needs is committed (the data, the voice phrases and their word timings), so it renders as is.

## What it shows

| Line | What the chart does |
|------|---------------------|
| "This is the video toolkit's star history on GitHub." | axes and line start on *history* |
| "It started slowly, ... in the first six weeks." | line reaches 28 Feb on *weeks* |
| "Then, at the end of March, four hundred stars arrived in three days." | spike drawn, "+402 in 3 days" band on *four hundred* |
| "Since then it has kept on climbing, ... past two thousand four hundred." | line runs on to today; counter lands on 2,435 |
| "Month by month, March is still the big one." | March bar lights on *March* |
| "But October was our strongest month since June." | October bar lights on *October* |

**The narration decides how fast each stretch of data is drawn.** `LineChart` takes `keys`, a list of
`{ frame, x }` pairs saying where the line has reached at which frame, and those frames come from `cue()`
(`lib/components/cues.ts`) looking words up in the captions. Regenerate a line and the chart re-times itself; rewrite
one so a cued word is gone and the bundle fails with the words that are there.

The other things the Matplotlib route ([data-viz-chart](../data-viz-chart/)) can't do without extra work:
- the y axis grows with the data as it's drawn (40 → 150 → 3,000), its gridlines crossfading between 1-2-5 steps
  instead of jumping;
- the counter reads the same head position the line is drawn to (`lineHead()`), so they can't disagree;
- labels are real text in the brand font, checked by `qc.py --layout` for size and contrast.

## Updating it

```bash
uv run examples/data-chart-stars/scripts/fetch_stars.py     # refresh data/star_series.json (GITHUB_TOKEN optional)
```

Totals and derived numbers (monthly gains, the March spike, the footer date) follow the data. **The narration
doesn't**: if the numbers move past what the script says ("past two thousand four hundred"), edit the `.txt` files
in `public/audio/scenes/`, then regenerate the voice:

```bash
# from the toolkit root; three takes, pick by pace and pitch, then word timings and the timeline
P=examples/data-chart-stars/public/audio
for t in 1 2 3; do mkdir -p $P/takes && cp -r $P/scenes $P/takes/t$t && rm -f $P/takes/t$t/*.mp3 $P/takes/t$t/*.json; \
  uv run tools/voiceover.py --provider qwen3 --cloud modal --language English \
    --ref-audio assets/voices/lugh-ref-expressive.wav --ref-text "<from assets/voices/lugh.json>" \
    --temperature 0.5 --scene-dir $P/takes/t$t; done
uv run tools/pick_takes.py --takes-dir $P/takes --output $P/scenes
uv run tools/captions.py $P/scenes
uv run examples/data-chart-stars/scripts/timing.py          # writes src/timing.json
```

Voice WAVs are gitignored, so `lugh-ref-expressive.wav` has to be present locally (see `assets/voices/lugh.json`).
Any other voice works the same way.

## Files

```
data/star_series.json      [date, cumulative stars] per day, from fetch_stars.py
public/audio/scenes/       one phrase per file: .txt (script), .mp3 (picked take), .captions.json (word timings)
scripts/fetch_stars.py     stargazer timestamps → daily cumulative series
scripts/timing.py          phrase starts and scene lengths → src/timing.json
src/Demo.tsx               the two scenes; every cue is at the top, under HEAD
```
