# concept-explainer-short

A **concept explainer** template, 9:16 vertical by default (TikTok / Reels / YouTube Shorts)
or 16:9 landscape by config — see "Formats" below — — built entirely from AI-generated parts and composed with a single Python
build (no Remotion/Node). One concept, hook → explanation → payoff → CTA, with
big burned karaoke captions for sound-off viewing.

Everything is driven by **`scenes.json`**: each scene is narration text plus a
visual asset. The pipeline turns that into per-scene cloned (or built-in) TTS,
word-perfect caption timing, and an audio-anchored composite where timing drift
is impossible.

```
scenes.json ──► gen_vo.py ──► gen_captions.py ──► build.py ──► out/short.mp4
                  │                │
                  ▼                ▼
        audio/scenes/*.mp3   captions/words_*.json
        vo_durations.json
```

## Quick start

**From an agent, or hands-off:** `uv run tools/video_job.py submit --spec spec.json` copies
this template into `jobs/<id>/project/`, runs every step below and QC, and `result JOB_ID`
hands back the MP4 (see `docs/agent-video-jobs.md`). The manual path:


```bash
cp -r templates/concept-explainer-short projects/my-short
cd projects/my-short

# 0. Renders immediately — placeholder cards, estimated timing, silent
uv run build.py

# 1. Write your scenes (see scenes.json for the example)
# 2. Generate visuals (from the TOOLKIT ROOT — see "Visuals" below)
# 3. Voiceover → captions → final render (from the project dir)
uv run gen_vo.py
uv run gen_captions.py        # needs: uv sync --extra whisper
uv run build.py
```

## scenes.json

```json
{
  "title": "Why Is the Sky Blue?",
  "scenes": [
    { "id": "01", "slug": "hook", "visual": "ltx",
      "asset": "clips/01_hook.mp4",
      "text": "Look up on a clear day..." }
  ]
}
```

- `text` — the narration, verbatim. Captions burn *this* text (whisper only
  provides timing), so write it exactly as it should appear.
- `asset` — relative path; the extension decides the treatment:
  image → slow Ken Burns zoom, video → boomerang loop cut to the scene,
  missing → gradient placeholder (so you can render at any stage).
- `visual` — documentation of intent (`ideogram` / `ltx`); build.py goes by
  the asset extension.

**Pacing budget:** narration ÷ 2.4 ≈ seconds per scene. A 60s short is
~140 words total; the 4-scene example is ~45s. Keep the hook under 3 seconds
of setup. Platform limits: YouTube Shorts ≤ 3 min, Reels ≤ 3 min,
TikTok ≤ 10 min — gen_vo.py and build.py print a warning past 3 minutes.

## Formats

`config.json → format` is the canvas. The build adapts to any size (placeholder cards,
Ken Burns, boomerang loops and captions are all computed from it), so 16:9 is a config
change, not a different template:

```json
"format":   { "width": 1920, "height": 1080, "fps": 30 },
"captions": { "y": 900, "size": 64, "maxWords": 4, "maxChars": 30, ... }
```

Generate assets at the matching size: Ideogram `--resolution 1280x720`, LTX `1024x576`.
`tools/video_job.py` writes both configs for you (`--format 16:9`).

## Visuals

Generate from the **toolkit root**, save into the project's `images/` and
`clips/`. The intended split (see CLAUDE.md "FLUX.2 vs Ideogram 4"):

```bash
# Text-bearing cards (Ideogram 4, JSON captions; 9:16 = --resolution 1440x2560)
uv run tools/ideogram4.py --json caption.json --resolution 1440x2560 \
    --output projects/my-short/images/02_scattering.png

# Motion b-roll (LTX-2; 576x1024 scales exactly to 1080x1920)
uv run tools/ltx2.py --width 576 --height 1024 --num-frames 161 \
    --prompt "..." --output projects/my-short/clips/01_hook.mp4

# Optional music bed (looped + ducked automatically if present)
uv run tools/music_gen.py --prompt "..." --duration 120 \
    --output projects/my-short/audio/music.mp3
```

Alternate cards and motion so the viewer gets a pattern interrupt every
~15 seconds. Ideogram for anything with legible text; LTX for atmosphere.

## Voice

`config.json → voice` selects the narrator:

```json
"voice": {
  "provider": "qwen3", "cloud": "modal", "maxWpm": 165,
  "refAudio": "ref/my-voice.m4a",
  "refText": "Exact transcript of the reference recording.",
  "speaker": "Ryan", "tone": "",
  "brand": ""
}
```

- **Clone**: set `refAudio` + `refText`. Use 12–25s of varied, full-sentence
  speech *at narration pace* — the clone copies the reference's pace and
  temperature cannot fix a rushed reference (see `/voice-clone`).
- **Built-in**: leave `refAudio` empty; `speaker` + `tone` apply.
- `maxWpm` is the pacing safety net: rushed takes are slowed in place with
  pitch-preserving atempo. gen_vo.py prints per-scene wpm either way.
- `provider: "elevenlabs"` also works (uses your configured voice).
- **Brand voice**: set `"brand": "my-brand"` to take the voice from
  `brands/my-brand/voice.json` (voiceId and settings) instead of the ambient
  config. Works with any provider, and pairs with a brand's palette so one
  brand drives both look and narrator.

## Title and headline overlays

`scenes.json` may carry a top-level `"titleCard": {"text": "...", "seconds": 2.8}` and, per scene,
`"overlay": {"headline": "...", "subline": "..."}`. The build animates them in (slide up and fade,
an accent bar drawing across, the subline following) in the upper half of the frame, clear of the
caption band. With `timing.startPad` at 0 the first scene is on screen from frame 0, so the
Shorts/Reels cover is never black. `tools/video_job.py` writes both from a spec.

## Captions

Karaoke pills, 1–3 words at a time, timed from whisper word timestamps but
**force-aligned to your script text** so they're word-perfect. Tune position,
size, and chunking in `config.json → captions`; defaults clear platform UI and
the bottom rows of Ideogram list cards. Set `enabled: false` to skip.

`whisperModel: "auto"` (the default) uses whisper's `turbo` (large-v3-turbo,
~6GB VRAM; needs openai-whisper ≥ 20240930) when CUDA is available and `base`
on CPU. Set an explicit name (`"base"`, `"small"`, `"turbo"`, …) to override —
e.g. `"base"` on a GPU with less than ~6GB. Only timing comes from whisper, so
a bigger model buys tighter word boundaries, not different caption text.

## Re-rendering

Everything is idempotent and cached. Re-run `gen_vo.py --force` after script
edits (then `gen_captions.py` again), or just `build.py` after swapping an
asset or tweaking config. Timeline math always re-derives from
`vo_durations.json` — there is nothing to manually re-time.
