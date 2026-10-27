# youtube-longform

A 16:9 **long-form YouTube** template (8–60 min explainers, essays, tutorials,
documentary-style narration) built from AI-generated parts and composed with a
Python build (no Remotion/Node). It is the landscape, chaptered sibling of
`concept-explainer-short`, rebuilt for length:

- **Chapters** — scenes group into chapters; YouTube chapter timestamps are
  written for you, plus a chapter title card on screen.
- **Chapter-by-chapter rendering with a cache** — edit one chapter and only
  that chapter re-renders. A fast numpy compositor (not moviepy's) keeps a
  20-minute 1080p render in the tens of minutes, not hours.
- **One continuous soundtrack** — VO anchored to real durations, music looped
  across the whole video and sidechain-ducked under the voice.
- **YouTube-ready master** — two-pass loudnorm to -14 LUFS / -1 dBTP,
  `+faststart`, an uploadable `captions.srt`, and a paste-ready description.

```
scenes.json ──► gen_vo.py ──► gen_captions.py ──► build.py
                  │                │                 │
                  ▼                ▼                 ├─► out/video.mp4
        audio/scenes/*.mp3   captions/words_*.json   ├─► out/youtube-description.txt
        vo_durations.json                            ├─► out/captions.srt
                                                     └─► out/timeline.json
```

## Quick start

```bash
cp -r templates/youtube-longform projects/my-video
cd projects/my-video

# 0. Renders immediately — placeholder cards, estimated timing, silent
uv run build.py --preview

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
  "title": "How GPS Actually Works",
  "description": "One- or two-line hook for the YouTube description.",
  "descriptionFooter": "#gps #science",
  "scenes": [
    { "id": "01", "slug": "hook", "chapter": "Intro", "visual": "ltx",
      "asset": "clips/01_hook.mp4",
      "text": "Right now, about thirty satellites are shouting the time at you..." },
    { "id": "03", "slug": "time-is-distance", "chapter": "Time Is Distance",
      "asset": ["images/03_signal.png", "clips/03_wave.mp4"],
      "text": "Start with one idea..." }
  ]
}
```

- `chapter` — a new value starts a new chapter; omit it to stay in the
  current one. Chapter titles become YouTube chapters **and** the on-screen
  chapter card. YouTube needs **≥3 chapters, the first at 0:00 (automatic),
  each ≥10s** — build.py warns when the list won't activate.
- `text` — the narration, verbatim. Captions/SRT use *this* text (whisper
  only provides timing), so write it exactly as it should appear.
- `asset` — a path or a **list of paths**; a list splits the scene's time
  evenly across them. Image → slow Ken Burns zoom, video → boomerang loop,
  missing → gradient placeholder. build.py flags scenes over 30s on a single
  visual — long-form needs a visual change every ~8–15 seconds.
- `id` must be unique and sortable (`"01"`, `"02"` … `"120"` is fine as
  long as you're consistent); `slug` names the audio file.

**Length budget:** ~150 words per minute. 10 min ≈ 1,500 words ≈ 40–60
scenes in 5–8 chapters. Over 8 min unlocks mid-roll ads.

## Commands

```bash
uv run build.py                # full 1080p render → out/video.mp4
uv run build.py --preview      # half-res fast encode → out/preview.mp4
uv run build.py --chapter 4    # just chapter 4 → out/chapters/04.mp4 (video only)
uv run build.py --force        # ignore the chapter cache
uv run gen_vo.py --force       # regenerate all VO after script edits
uv run gen_captions.py --force # redo caption timing
```

Chapters are cached by a hash of their scenes, assets, captions and config,
so a full re-run after changing one scene re-renders one chapter and re-mixes
the audio (seconds). Audio is always rebuilt in full — it's cheap.

## Visuals

Generate from the **toolkit root**, save into the project's `images/` and
`clips/`. 16:9 sizes:

```bash
# Text-bearing cards / diagrams (Ideogram 4 — JSON captions, 16:9)
uv run tools/ideogram4.py --json caption.json --resolution 2560x1440 \
    --output projects/my-video/images/04_spheres.png

# Text-free illustration / backgrounds (FLUX.2)
uv run tools/flux2.py --prompt "..., no text" --width 1920 --height 1080 \
    --output projects/my-video/images/05_bg.png

# Motion b-roll (LTX-2, 1024x576 scales exactly to 1920x1080)
uv run tools/ltx2.py --width 1024 --height 576 --num-frames 161 \
    --prompt "..." --output projects/my-video/clips/01_hook.mp4

# Music bed — one long track is looped across the video and ducked
uv run tools/music_gen.py --preset ambient --duration 180 \
    --output projects/my-video/audio/music.mp3
```

Mix stills and motion: a long video that is all Ken Burns feels like a
slideshow; one that is all LTX clips costs more and loops visibly. A good
default is one LTX clip per chapter opener and 2–4 Ideogram/FLUX stills per
minute in between.

## Voice

Same as concept-explainer-short — `config.json → voice`:

- **Clone**: `refAudio` + `refText` (12–25s of narration-paced speech).
- **Built-in**: `speaker` + `tone` (Qwen3), or `provider: "elevenlabs"`.
- **Brand voice**: `"brand": "my-brand"` reads `brands/my-brand/voice.json`.
- `maxWpm` (default 160) clamps rushed takes with pitch-preserving atempo.
  Over 10+ minutes, a consistent pace matters more than in a short.

## Captions

YouTube viewers can turn subtitles on, so the default is **not** to burn them:
`build.py` writes `out/captions.srt` (sentence-aware, ≤84 chars per cue) for
upload instead. Set `captions.burn: true` for burned subtitle pills (smaller,
6-word chunks, bottom of frame) — useful for sound-off social clips.

## Publishing

```bash
# from the toolkit root
uv run tools/youtube_upload.py --video projects/my-video/out/video.mp4 \
    --title "How GPS Actually Works" \
    --description-file projects/my-video/out/youtube-description.txt \
    --captions projects/my-video/out/captions.srt \
    --thumbnail projects/my-video/out/thumbnail.png --privacy private --json-out
```

Or run `/thumbnail` then `/publish`.

## config.json

| Key | Default | Notes |
|-----|---------|-------|
| `format` | 1920×1080 @ 30 | 3840×2160 works; renders ~4x slower |
| `chapterCard` | on, 3.5s, top-left | `numbered: false` drops the "01" |
| `captions.burn` | `false` | SRT is always written when timing exists |
| `timing.lead / tail` | 0.4 / 0.8 s | visual before / after each scene's VO |
| `music.volume / duck` | 0.18 / true | sidechain ducking under the voice |
| `audio.lufs / truePeak` | -14 / -1 | YouTube's loudness target |
