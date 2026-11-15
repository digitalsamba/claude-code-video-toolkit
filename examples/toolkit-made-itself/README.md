# toolkit-made-itself

A 93s video about claude-code-video-toolkit, made in the toolkit from one sentence typed into Claude Code:
*"Make a 90 second video about this toolkit, using this toolkit."* Each stage (narrate, score, generate, compose,
check) is shown as the real command in the `Terminal` component, then turned into what it made.

[Watch it on YouTube](https://www.youtube.com/watch?v=3HjPatdeal4)

```bash
npm ci
npm run studio      # preview
npm run render      # out/video.mp4
cd ../.. && uv run tools/qc.py --render examples/toolkit-made-itself/out/video.mp4 --fix   # -14 LUFS
```

Everything it needs is committed (the voice, its word timings, the chosen music take, the stills, the clip, the
sound effects and the captured terminal output), so it renders as is and spends nothing.

## Nothing on screen is typed in by hand

The rule this video was built to: every terminal line is real output from this build, every number is measured
from it, and the Check scene shows a problem `qc.py` really found in the draft.

1. **Capture.** Every command was run through `scripts/cap.sh NAME <command>`, which runs it from the toolkit
   root and saves what it printed to `capture/NAME.txt`, with the command as the first line and the time it took
   as the last. The paths in those files read `projects/toolkit-made-itself/` because that's where it was built.
2. **Convert.** `scripts/captured.py` turns the captures into `src/captured.json`, with each line's colour (the
   tools' ANSI codes) mapped to a `Terminal` line kind: ok, err, dim.
3. **Pick.** Scenes take lines with `pick('voiceover', '"wpm"', 'scenes/02_narrate.mp3')` (`src/captured.ts`). A
   line that isn't in the capture throws when the bundle builds, so a re-run that changes the output can't leave
   a stale line on screen.

The bill (`src/bill.json`, $0.33) came from `scripts/bill.py`: ElevenLabs history for the voice credits since the
build started, the `Est. cost` lines the Modal tools printed, Scribe seconds at $0.40 an hour, and the music takes
counted (acemusic is free). It needs the build session's ElevenLabs history, so it's kept as the record of how the
bill was measured rather than something to re-run.

## Timing

Narration is per scene (`public/audio/scenes/*.txt`, voiced by Marilyn from the ElevenLabs voice library through
`brands/video-toolkit`, with `--max-wpm 165`). `scripts/timing.py` reads each scene's length and its word timings
and writes `src/timing.json`; everything inside a scene waits for its word with `cue()`. The compose scene's
source code (`src/compose.tsx`, copied into `src/source.json` by `scripts/source.py`) highlights each line as the
narration names it.

## Compositions

| Id | What |
|----|------|
| `MadeWithItself` | The video. Its `music` prop picks the take (default 3, the one committed); `0` renders without music, for `audition.py` |
| `Thumbnail` | The 1280x720 YouTube thumbnail, a still of the brand and a terminal (`npx remotion still src/index.ts Thumbnail out/thumbnail.png`) |

The four music takes were compared with `audition.py` against a render without music; only the pick
(`public/audio/music/bg_3.mp3`) is committed. The waveforms in the Score scene are images of all four.

## Notes

- Uses `lib/components` (`Terminal`, `PageTurns`, `BarChart`, `Counter`) and `brands/video-toolkit` by relative
  path, so it stays inside the toolkit checkout.
- Published from a render on Remotion 4.0.518; the example is pinned to 4.0.539 like the rest of the toolkit. On
  4.0.539 a bar highlight that changed its label's font size every frame crashed Chrome mid-render ("Page
  crashed!"), so `BarChart` now scales the label instead. The picture is the same.
- The sound effects are from Freesound (CC0, credits in `public/audio/sfx/*.attribution.txt`) and ElevenLabs.
