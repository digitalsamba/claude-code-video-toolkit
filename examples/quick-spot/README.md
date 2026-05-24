# quick-spot

A 15-second ad-style example built with **moviepy** instead of Remotion. Demonstrates the toolkit's "single-file `build.py`" pattern for short-form video.

Renders out of the box with **zero external assets** — solid colour backgrounds and PIL-rendered text only. Add per-scene voiceover and music when you're ready and the build picks them up automatically.

## Quick start

```bash
# From the toolkit root (one-time — installs moviepy, Pillow, matplotlib
# alongside the rest of the toolkit's optional Python deps):
python3 -m pip install -r tools/requirements.txt

# Then:
cd examples/quick-spot
python3 build.py
```

That's it. ~15 seconds later you have `out.mp4` — a silent text-on-colour spot. Open it to verify the timing and layout look right.

> If you get a `Missing dependency` error, the script will tell you exactly which package is missing and which `pip` command to run.

## Why moviepy instead of Remotion?

Use moviepy for sub-30s, tightly-timed content where every frame matters and where text must be deterministic — trailers, promos, news lower thirds, social captions. The Remotion templates (`sprint-review`, `product-demo`) are designed for longer, design-system-driven work; moviepy is the right tool for ad-style spots.

See the **moviepy** skill (`.claude/skills/moviepy/SKILL.md`) for the full pattern guide and the genres where this approach shines.

## What this example demonstrates

1. **PIL → ImageClip text rendering.** moviepy 2.x's `TextClip(method='label')` clips letter ascenders and descenders. The fix is to render text to a transparent PNG via Pillow and load it as an `ImageClip`. The cache key is the content hash so re-builds are free. See `render_text_png()` in `build.py`.

2. **Audio-anchored timeline.** The comment block at the top of `build()` is the source of truth. Every visual element has an absolute `start=` referencing it. Drift is impossible because durations come from real audio (when present) rather than estimates. See **CLAUDE.md → Video Timing → Audio-Anchored Timelines**.

3. **Optional VO + ducked music.** `build_audio()` checks for per-scene mp3 files and mixes them only if all are present, otherwise renders silent. Music is ducked to 22% volume under VO at 115% — tuned defaults from production use.

4. **Solid-colour backgrounds with fades.** No external video required. Swap these for `VideoFileClip("ltx_broll.mp4")` if you generate b-roll with `tools/ltx2.py`.

## Adding voiceover

```bash
# 1. Generate per-scene VO from the script (uses Qwen3-TTS by default; pass --provider elevenlabs for ElevenLabs)
python3 ../../tools/voiceover.py \
    --script VOICEOVER-SCRIPT.md \
    --scene-dir public/audio/scenes

# 2. Optional: generate background music
python3 ../../tools/music_gen.py \
    --preset cta --duration 15 \
    --output public/audio/music.mp3

# 3. Re-render — build.py picks up the new files automatically
python3 build.py
```

After the first audio render, run `python3 ../../tools/sync_timing.py --voiceover-json` against the generated audio to verify your scene-anchored timestamps still match the actual audio durations. If the audio is significantly longer/shorter than your timeline, edit the `start=` values and the comment block at the top of `build.py`.

## Adding LTX-2 b-roll

Replace any `color_bg(...)` call with a video clip:

```python
from moviepy import VideoFileClip

bg = (
    VideoFileClip(str(HERE / "public" / "broll" / "scene1.mp4"))
    .without_audio()
    .subclipped(0, 4.0)
    .resized(new_size=(W, H))
    .with_start(0.0)
)
clips.append(bg)

# Optional: tinted overlay so the text stays readable over busy footage
clips.append(
    ColorClip((W, H), color=NAVY_DARK)
    .with_duration(4.0)
    .with_start(0.0)
    .with_opacity(0.55)
)
```

Generate b-roll with:

```bash
python3 ../../tools/ltx2.py \
    --prompt "Dark moody abstract background, blue light streaks, cinematic" \
    --num-frames 121 \
    --output public/broll/scene1.mp4
```

See the **ltx2** skill for prompting guidance.

## File layout

```
quick-spot/
├── README.md           # this file
├── VOICEOVER-SCRIPT.md # narration script (used by voiceover.py)
├── build.py            # the entire video — runnable as-is
├── .text_cache/        # PIL text PNGs (auto-generated, gitignored)
├── public/
│   ├── audio/          # voiceover + music (gitignored, optional)
│   └── broll/          # LTX-2 generated b-roll (gitignored, optional)
└── out.mp4             # render output (gitignored)
```

## Customising

- **Colours** — edit the constants near the top of `build.py` (`NAVY`, `CORAL_HEX`, etc.) or pull them from a `brands/` profile.
- **Font** — change `FONT_BOLD` to any TTF on your system. The cache invalidates automatically when the font path changes (it's part of the hash key).
- **Duration** — extend `DURATION`, add scenes to the timeline comment block, then add `text_clip(...)` calls referencing the new starts.
- **Aspect ratio** — change `W, H` to `(1080, 1920)` for vertical (Reels/Shorts/TikTok). All `position=("center", ...)` values still work.

## Related

- **moviepy skill** — full pattern reference with genre fits and gotchas
- **ltx2 skill** — generating b-roll and stylised character cameos
- **acestep skill** — generating background music with `music_gen.py`
- **examples/data-viz-chart** — sibling example showing matplotlib + moviepy for data-driven scenes
