# math-story-short

9:16 animated **math story** shorts (Reels / TikTok / YouTube Shorts) in Remotion, built for
the MaxLearnsMath brand. Max hits an everyday situation, gets the intuition wrong, and the
maths shows why. Every visual is code, so the charts, counters and equations show the real
numbers instead of AI-generated pictures of numbers.

The example episode is **"Why the casino always wins"** (roulette and the house edge),
about 2:00.

```
story-config.ts ──► npm run scripts ──► voiceover.py ──► captions.py ──► sync_timing.py ──► npm run render
  (scenes +             (scene .txt)       (scene .mp3)    (word timing)    (real durations)
   narration)
```

## Quick start

```bash
cp -r templates/math-story-short projects/my-episode
cd projects/my-episode && npm install
npm run studio            # renders straight away: estimated timing, typewriter from the script, silent
```

From the **toolkit root**:

```bash
P=projects/my-episode
(cd $P && npm run scripts)                                            # narration -> public/audio/scenes/*.txt
uv run tools/voiceover.py --provider qwen3 --tone warm --scene-dir $P/public/audio/scenes --json
uv run tools/captions.py $P/public/audio/scenes                       # word timing for the typewriter
uv run tools/sync_timing.py --config $P/src/config/story-config.ts --apply --padding 0.6
uv run tools/qc.py --project $P
(cd $P && npm run render)                                             # out/video.mp4
uv run tools/qc.py --render $P/out/video.mp4 --fix                    # -14 LUFS for Reels/Shorts
```

## How an episode is built

| Piece | Where | What it does |
|---|---|---|
| Story | `src/config/story-config.ts` | Chapters, then scenes: narration, audio, duration, Max's mood/layout/thought, visual key |
| Visuals | `src/episode/visuals.tsx` | One component per visual key. Consecutive scenes with the same key share one mounted visual and get the scene boundaries as **beats**, so a chart keeps drawing across scenes |
| Data | `src/episode/data.ts` | Seeded simulation behind the charts. `scripts/seed-search.mjs` finds a seed whose results hit the story beats |
| Brand | `src/config/brand.ts`, `theme.ts` | Colours, fonts (Inter / JetBrains Mono / STIX Two), layout zones |

Persistent layers (backdrop, chapter header with story-style progress, Max, captions, grain)
live outside the scene sequences, so they never jump at a cut: Max glides between layouts and
hops when his mood changes.

## Building blocks (`src/components/`)

| Component | Use |
|---|---|
| `EditorPanel` | The brand's signature code-editor window: file tab, traffic-light dots, live status |
| `LineChart` | Animated SVG chart: linear or log x, animated domains (zoom-outs), glow, area fill, pulsing end marker |
| `StatRow` | Glass tiles with label + big tabular number; `tone: 'bad'` goes crimson |
| `EquationCard`, `Frac`, `Hl` | A derivation that builds line by line in a maths serif, with a highlighted result |
| `Max` / `MaxLayer` | The mascot: moods `happy thinking smug shocked confused sad`, layouts `hero side hidden`, thought bubbles |
| `TypewriterCaptions` | `›` prompt, text typed in sync with the voice, block cursor |
| `ChapterHeader`, `Backdrop`, `Outro` | Chapter progress + title (hidden during a `hook` scene), animated navy backdrop, end card |
| `Camera` (`Fx.tsx`) | Eased push-ins: keyframes of scale + focus point. Use it on the number that matters, then ease back out |
| `Sfx` (`Fx.tsx`) | Fire `public/sfx/*.mp3` at a frame. `node scripts/make-sfx.mjs` synthesises the pack with ffmpeg; swap in `tools/sfx.py` output under the same names |

Max directions per scene: `mood`, `layout` (`hero` / `side` / `hidden`), `pose: 'point'` (arm toward the
stage), `emote: '?' | '!'`, and `thought` (hero layout only).

**Zones in a visual:** `<Stage>` is the main picture. `<Aside>` is the strip right of Max in side layout,
for live readouts (`StatRow column`). Leaving it empty wastes a third of the screen.

## A new episode

1. Write the story beats first: **hook (payoff first) → setup → wrong intuition → surprise → rule out luck → reveal → scale → punchline**.
   The `hook: true` scene is a cold open: show the shocking result in the first second, then rewind.
2. Replace `chapters` and `scenes` in `story-config.ts`. Budget about 2.4 words/second plus about 1s per scene.
3. Write the simulation in `data.ts`, search a seed, and put the numbers it prints into the narration.
4. Add a visual component per group in `visuals.tsx` and register it in `visuals`.
5. Time reveals by **fraction of the beat** (`at(beats, 1, 0.4)`), so they survive `sync_timing --apply`.
   After the voiceover, nudge the fractions so the reveals land on the words.

## Layout (1080×1920)

Header 92–250 · stage 300–1090 · Max (side) 1125–1515 with the aside to his right (x ≥ 410) · captions from 1540. The bottom ~260px stays
clear for the platform's own UI.
