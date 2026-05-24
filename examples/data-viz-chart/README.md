# data-viz-chart

An animated time-series chart with deterministic text overlay. Demonstrates the **"matplotlib for data, moviepy for trustworthy text"** pattern — the natural production pipeline for news graphics, dashboards, and any data-driven scene where the labels need to be exact.

Renders out of the box with the included `data/star_series.json` (real GitHub star history of `digitalsamba/claude-code-video-toolkit`).

## Quick start

```bash
# From the toolkit root (one-time — installs matplotlib, moviepy, and
# Pillow alongside the rest of the toolkit's optional Python deps):
python3 -m pip install -r tools/requirements.txt

# Then:
cd examples/data-viz-chart
python3 build.py
```

First run takes ~30 seconds (matplotlib renders 450 frames, moviepy composites). Subsequent runs reuse the cached chart animation unless `data/star_series.json` is newer.

> If you get a `Missing dependency` error, the script will tell you exactly which package is missing and which `pip` command to run.

## Why split matplotlib and moviepy?

| Layer | Tool | Why |
|-------|------|-----|
| Data (line, fill, axes, ticks) | matplotlib | Genuinely good at drawing data from a series. Ships with reasonable defaults for axes, dates, and gridlines. |
| Trustworthy text (title, source attribution, headline number) | moviepy + PIL | Pixel-perfect, deterministic, version-controllable. Cannot be hallucinated or mis-styled. |

This is exactly how real news graphics are produced. The data layer comes from a data tool (R, Python, D3); the text layer is composited deterministically on top so that names, sources, and headline figures are guaranteed correct. AI video generation models cannot guarantee any of that — that's the whole reason this pattern exists.

The **moviepy** skill (`.claude/skills/moviepy/SKILL.md`) has the full "trustworthy text" framing and the genres where this approach matters most.

## What this example demonstrates

1. **`FuncAnimation` → mp4 → `VideoFileClip`.** matplotlib renders the animated chart to its own mp4 (`chart_anim.mp4`), then moviepy loads it as a clip and composites text on top. Two clean steps; either side can be re-run independently.

2. **PIL → ImageClip text rendering.** Same pattern as `examples/quick-spot` — moviepy 2.x's `TextClip(method='label')` clips letter ascenders and descenders. Render to a PNG via Pillow and load as an `ImageClip`. Cached by content hash so re-builds are free.

3. **Trustworthy source attribution.** The "Source: GitHub Stars API" footer is the point. It must be spelled correctly and styled consistently across every frame. moviepy guarantees both. The same pattern applies to interviewee name plates, breaking-news banners, pricing callouts, and anything else where text accuracy is non-negotiable.

4. **Cache-aware build.** The matplotlib step is the slow part (~25s for 450 frames). `build()` only re-renders the chart animation if `star_series.json` is newer than `chart_anim.mp4`, so iterating on the text overlay layer is fast.

## Swapping in your own data

`data/star_series.json` is just a list of `[date_string, value]` pairs:

```json
[["2026-01-20", 1], ["2026-01-22", 2], ["2026-01-24", 3], ...]
```

To plot something else:

1. Replace `data/star_series.json` with your own series.
2. Update the title in `build.py` (`text_clip("github.com/...", ...)`).
3. Update the y-axis label in `render_chart_animation()` (`ax.set_ylabel("Stars", ...)`).
4. Update the unit label (`text_clip("stars", ...)`) and source attribution.
5. Run `python3 build.py`.

The chart re-renders automatically because the data file mtime is newer than the cached animation.

## Adding voiceover

Same pattern as `examples/quick-spot`:

```bash
python3 ../../tools/voiceover.py \
    --script VOICEOVER-SCRIPT.md \
    --scene-dir public/audio/scenes
```

Then read the audio durations from the JSON output and anchor your text overlays to absolute timestamps in `build()`. See **CLAUDE.md → Video Timing → Audio-Anchored Timelines** for the pattern.

## File layout

```
data-viz-chart/
├── README.md           # this file
├── build.py            # the entire video — runnable as-is
├── data/
│   └── star_series.json # input series — replace with your own
├── chart_anim.mp4      # matplotlib output (auto-generated, gitignored)
├── .text_cache/        # PIL text PNGs (auto-generated, gitignored)
└── out.mp4             # final render (gitignored)
```

## Customising the chart

The matplotlib styling is in `render_chart_animation()`:

- **Colour palette** — `NAVY`, `CORAL_HEX`, `SLATE_HEX` constants near the top of `build.py`.
- **Axis ticks** — `mdates.MonthLocator()` for monthly ticks. Swap for `WeekdayLocator()`, `YearLocator()`, etc.
- **Line style** — `linewidth=5`, change to taste.
- **Fill area** — `alpha=0.22` on `fill_between`. Increase for more emphasis.
- **Aspect ratio** — `figsize=(16, 9)` is widescreen. Use `(9, 16)` for vertical / Reels.

## Related

- **moviepy skill** — full pattern reference, genre fits, gotchas
- **examples/quick-spot** — sibling example showing the audio-anchored timeline pattern for ad-style spots
- **CLAUDE.md → Video Timing → Audio-Anchored Timelines** — pairing this with per-scene voiceover
