# youtube-longform — guidance for Claude

Python template (moviepy for decoding/encoding, a numpy compositor, ffmpeg for
audio and mastering; no Remotion, no npm). The whole video derives from
`scenes.json`; the pipeline is `gen_vo.py → gen_captions.py → build.py`, all
run **from the project directory**. Asset generation (Ideogram/FLUX/LTX/music)
runs **from the toolkit root**. See README.md for the full workflow, and the
`youtube-longform` skill for scripting and structure guidance.

## Working on a project copy

1. **Outline before script.** Agree the chapter list with the user first
   (5–8 chapters for 10 min), then write scenes chapter by chapter. Budget
   ~150 words/min; one scene ≈ 1–4 sentences (8–25s of VO).
2. **Hook discipline:** the first 30 seconds decide retention. Scene 01 opens
   on tension or a concrete promise; scene 02 says what the video will cover.
   No "hey guys, welcome back".
3. **Visual rhythm:** a new visual every ~8–15s. Use a list of assets for any
   scene over ~20s. build.py prints a warning for 30s+ scenes on one visual.
4. **Render early, render often.** `build.py --preview` works at every stage.
   Iterate on one chapter with `--chapter N` and look at the frames.
5. **Costs scale with length.** Before generating assets for a long video,
   tell the user the count (e.g. "~45 Ideogram cards ≈ $2–4, 8 LTX clips ≈
   $2") and confirm.

## Review checklist before calling it done

- `uv run tools/qc.py --project projects/NAME` (assets, pacing, loudness).
- Pull frames at several timestamps per chapter and *look*: placeholder cards
  left in, crops cutting text off, chapter card over important content.
- Read the chapter list build.py prints — ≥3 chapters, none under 10s.
- `out/captions.srt` exists (or the user chose to skip captions).
- Check gen_vo.py's per-scene wpm output for FAST/SLOW flags.

## Gotchas

- Changing `text` needs `gen_vo.py --force` (or delete that scene's MP3) and
  then `gen_captions.py` — captions re-run automatically only for MP3s newer
  than their timing file.
- The chapter cache keys on scenes, assets (size+mtime), captions and config —
  not on build.py itself. After editing build.py, bump `BUILD_VERSION` or use
  `--force`.
- `--chapter N` writes video only; the full build mixes audio and masters.
- Keep toolkit work (improving this template) separate from project work.
  Template fixes belong in `templates/youtube-longform/`, not the copy.
