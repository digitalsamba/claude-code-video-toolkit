# concept-explainer-short — guidance for Claude

Python/moviepy template (no Remotion, no npm). The whole video derives from
`scenes.json`; the pipeline is `gen_vo.py → gen_captions.py → build.py`, all
run **from the project directory**. Asset generation (Ideogram/LTX/music) runs
**from the toolkit root**. See README.md for the full workflow.

## Working on a project copy

1. **Plan in scenes.json first.** Narration text is the source of truth for
   both VO and captions. Budget words ÷ 2.4 ≈ scene seconds; warn the user
   past ~3 min total (YouTube Shorts/Reels limit; TikTok allows 10).
2. **Hook discipline:** scene 01 must earn the next 5 seconds. Question or
   tension in the first sentence; no throat-clearing.
3. **Visual rhythm:** alternate Ideogram cards (text-bearing) with LTX motion
   (atmosphere) — a pattern interrupt every ~15s. Author Ideogram JSON
   captions yourself (ideogram4 skill), one shared palette across all cards,
   vertical = `--resolution 1440x2560`. LTX vertical = `576x1024`,
   `--num-frames 161`; prompt for motion/atmosphere, never on-screen text.
4. **Render early, render often.** `build.py` works at every stage —
   placeholders before assets, estimates before VO, silent before audio.
   Show the user intermediate renders rather than describing them.

## Review checklist before calling it done

- Pull frames with ffmpeg at several timestamps and *look* at them:
  caption collisions, asset crops, placeholder cards left in.
- Check gen_vo.py's per-scene wpm output; anything flagged FAST/SLOW that
  `maxWpm` didn't catch needs a script edit or retake.
- Captions render the *script* text — if a scene's text changes, re-run
  gen_vo.py --force for that flow, then gen_captions.py.

## Gotchas (learned the hard way)

- Clone pacing inherits from the reference; `voice.maxWpm` is the safety
  net, not a substitute for a good 12–25s reference.
- Never burn whisper's own transcription — gen_captions.py aligns script
  words onto whisper timing for a reason.
- moviepy 2.x: `with_*` methods, `subclipped`, PIL for text (TextClip clips
  ascenders), `ColorClip` takes RGB tuples.
- Keep toolkit work (improving this template) separate from project work
  (making a video). Template fixes belong in `templates/`, not the copy.
