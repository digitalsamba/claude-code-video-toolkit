---
name: youtube-longform
description: Plan, script and produce long-form 16:9 YouTube videos (8–60 min explainers, video essays, tutorials, documentary-style narration) with the youtube-longform template — retention-first outlines, chaptered scripts, visual density planning, cost estimates, chapter timestamps, SRT captions and loudness-correct masters. Use when the user wants a long YouTube video, a video essay, a 10/15/20-minute explainer, a faceless YouTube channel video, chapters, or a landscape narrated video longer than a Short. Triggers include long video, long-form, YouTube video, video essay, explainer video, documentary, chapters, faceless channel, 10 minute video.
---

# Long-form YouTube with youtube-longform

The template lives in `templates/youtube-longform/` (README.md has every flag).
This skill is the *editorial* half: how to structure a long video so people keep
watching, how to plan visuals and spend, and the production order.

```
outline ─► script (scenes.json) ─► preview ─► assets ─► VO ─► captions ─► build ─► QC ─► thumbnail ─► publish
 (agree)       (agree)            (free)     ($)       ($)    (free)     (free)
```

Stop for the user's approval at **outline**, **script**, and **before spending
on assets**. Everything else can run straight through.

## 1. Outline (retention-first)

Ask for: topic, audience, target length, tone, voice (clone/built-in/brand),
and whether they have a brand in `brands/`. Then propose chapters.

| Length | Chapters | Scenes | Words |
|--------|----------|--------|-------|
| 8 min  | 4–6 | 30–40 | ~1,200 |
| 12 min | 6–8 | 45–60 | ~1,800 |
| 20 min | 8–12 | 75–100 | ~3,000 |

Shape that holds attention:

1. **Cold open (0:00–0:30)** — the most interesting moment, question or
   claim, *first*. No intro, no "welcome back", no channel plug.
2. **Promise (0:30–0:50)** — what they'll know by the end; tease the payoff
   of the final chapter ("…and why it would fail without Einstein").
3. **Body chapters** — each one opens a question and closes it. End each
   chapter on a hook into the next (an *open loop*): "But that only gets
   you to two possible points — and one of them is in space."
4. **Payoff** — deliver the promise from step 2 explicitly.
5. **Short outro (≤20s)** — one recap line, one ask. End screens take the
   last 5–20s, so keep the final scene's visual simple.

Chapter titles are what viewers scan in the progress bar: make them concrete
("Why Four Satellites"), not generic ("Part 2"). YouTube needs ≥3 chapters,
each ≥10s — build.py checks.

## 2. Script → scenes.json

- One scene = one idea = 1–4 sentences (8–25s of VO). Break long paragraphs
  into scenes so visuals can change.
- Write for the ear: short sentences, concrete numbers, "you". Spell numbers
  and symbols the way they should be *spoken* ("thirty eight microseconds",
  not "38μs") — the SRT shows exactly this text.
- Re-hook every 60–90 seconds: a question, a surprising number, a "here's
  the strange part".
- `chapter` on the first scene of each chapter (others inherit it).
- `asset` as a list when a scene runs >20s.

Check the budget before showing the script: `words ÷ 150 = minutes`. Then
`uv run build.py --preview` renders the whole timeline with placeholder cards
and estimated timing — show the user the chapter list it prints.

## 3. Visual plan and cost

Plan one visual per ~8–15s. A practical mix per minute of video:

| Visual | Tool | Use for | Approx cost |
|--------|------|---------|-------------|
| Text-bearing card / diagram / stat | `ideogram4.py --resolution 2560x1440` | definitions, numbers, lists, labelled diagrams | $0.03–0.10 each |
| Text-free illustration / background | `flux2.py --width 1920 --height 1080` | scenes, metaphors, atmosphere | Modal GPU, ~cents |
| Motion b-roll | `ltx2.py --width 1024 --height 576 --num-frames 161` | chapter openers, cold open, big moments | ~$0.23 per clip |
| Music bed | `music_gen.py --duration 180` (looped) | whole video | free key / cents |
| Voice | `gen_vo.py` (Qwen3 on Modal / ElevenLabs) | narration | ~$0.01/scene on Qwen3 |

Default recipe: **one LTX clip per chapter opener + the cold open, and 3–4
stills per minute** (alternating Ideogram cards and FLUX images). For a
10-minute video that's ~8 clips + ~35 stills ≈ **$3–6** total. Always state
the count and estimate and get a yes before generating.

Consistency across 40+ images matters more than any single image: fix one
palette (from `brands/<brand>/brand.json` or `config.json → palette`) and one
style phrase, and reuse them in every prompt. For Ideogram, author JSON
captions (see the `ideogram4` skill). LTX prompts describe motion and
atmosphere, never on-screen text.

Name files by scene id (`images/07_orbit.png`, `clips/12_launch.mp4`) so
missing ones are obvious in the build log and QC.

## 4. Audio

```bash
cd projects/NAME
uv run gen_vo.py               # per-scene TTS; prints wpm per scene + total minutes
uv run gen_captions.py         # whisper word timing aligned to the script (uv sync --extra whisper)
```

- Long videos expose pace drift: keep `voice.maxWpm` at 155–160 and look for
  FAST/SLOW flags.
- One music track (2–3 min) is looped and ducked under the voice for the whole
  video. Calm, low-variation beds (`ambient`, `lofi`, `corporate-bg`) loop
  best. Keep `music.volume` 0.12–0.2.

## 5. Build, review, iterate

```bash
uv run build.py --chapter 1    # look at one chapter while iterating
uv run build.py                # full master → out/video.mp4 (+ description, SRT, timeline)
cd ../.. && uv run tools/qc.py --project projects/NAME
```

Review by pulling frames, at least two per chapter, and looking at them:

```bash
ffmpeg -ss 95 -i out/video.mp4 -frames:v 1 -vf scale=960:-1 /tmp/f95.png
```

Only edited chapters re-render; audio always re-mixes. Render time is roughly
1–2x the video length at 1080p on a laptop (4K is ~4x slower).

## 6. Thumbnail and publish

- `/thumbnail` — 1280×720 Ideogram card; 2–4 word hook that adds to the title
  rather than repeating it.
- `/publish`, or directly:

```bash
uv run tools/youtube_upload.py --video projects/NAME/out/video.mp4 \
  --title "..." --description-file projects/NAME/out/youtube-description.txt \
  --captions projects/NAME/out/captions.srt --thumbnail projects/NAME/out/thumbnail.png \
  --privacy private --json-out
```

Upload private first, check chapters and captions in YouTube Studio, then
publish. Title ≤70 chars so it isn't truncated; put the main keyword early.

## Honesty rules

- If the user wants a *monetised* channel, say plainly that YouTube requires
  AI-generated realistic content to be disclosed ("altered or synthetic
  content" toggle at upload) and that mass-produced, repetitive videos are
  demonetised under the inauthentic-content policy. Original scripts and
  research are what make this work.
- Generation costs money and GPU time. "Unlimited" isn't a thing — give
  numbers.
