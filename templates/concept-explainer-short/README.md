# concept-explainer-short

A 9:16 vertical **concept explainer** template — TikTok / Reels / YouTube Shorts
style — built entirely from AI-generated parts and composed with a single Python
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

```bash
cp -r templates/concept-explainer-short projects/my-short
cd projects/my-short

# 0. Renders immediately — placeholder cards, estimated timing, silent
python3 build.py

# 1. Write your scenes (see scenes.json for the example)
# 2. Generate visuals (from the TOOLKIT ROOT — see "Visuals" below)
# 3. Voiceover → captions → final render (from the project dir)
python3 gen_vo.py
python3 gen_captions.py        # needs: pip install openai-whisper
python3 build.py
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

## Visuals

Generate from the **toolkit root**, save into the project's `images/` and
`clips/`. The intended split (see CLAUDE.md "FLUX.2 vs Ideogram 4"):

```bash
# Text-bearing cards (Ideogram 4, JSON captions; 9:16 = --resolution 1440x2560)
python3 tools/ideogram4.py --json caption.json --resolution 1440x2560 \
    --output projects/my-short/images/02_scattering.png

# Motion b-roll (LTX-2; 576x1024 scales exactly to 1080x1920)
python3 tools/ltx2.py --width 576 --height 1024 --num-frames 161 \
    --prompt "..." --output projects/my-short/clips/01_hook.mp4

# Optional music bed (looped + ducked automatically if present)
python3 tools/music_gen.py --prompt "..." --duration 120 \
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
  "speaker": "Ryan", "tone": ""
}
```

- **Clone**: set `refAudio` + `refText`. Use 12–25s of varied, full-sentence
  speech *at narration pace* — the clone copies the reference's pace and
  temperature cannot fix a rushed reference (see `/voice-clone`).
- **Built-in**: leave `refAudio` empty; `speaker` + `tone` apply.
- `maxWpm` is the pacing safety net: rushed takes are slowed in place with
  pitch-preserving atempo. gen_vo.py prints per-scene wpm either way.
- `provider: "elevenlabs"` also works (uses your configured voice).

## Captions

Karaoke pills, 1–3 words at a time, timed from whisper word timestamps but
**force-aligned to your script text** so they're word-perfect. Tune position,
size, and chunking in `config.json → captions`; defaults clear platform UI and
the bottom rows of Ideogram list cards. Set `enabled: false` to skip.

## Re-rendering

Everything is idempotent and cached. Re-run `gen_vo.py --force` after script
edits (then `gen_captions.py` again), or just `build.py` after swapping an
asset or tweaking config. Timeline math always re-derives from
`vo_durations.json` — there is nothing to manually re-time.
