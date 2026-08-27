# Render baseline

A fixed Remotion composition used to answer one question when Remotion is bumped:
**did the output change?**

## How it works

`scripts/render-baseline.mjs` renders one still per block of `src/Baseline.tsx` — twice,
on the same machine: once at the current pin, once at the candidate version — and
pixel-diffs the pairs. Nothing is committed as a "golden" image, so font, GPU and
Chrome differences between machines never produce false positives. Only the bump does.

Blocks (one still each, at the block's middle frame, `--scale=0.25`):

| # | Block | Exercises |
|---|-------|-----------|
| 01 | easing | `interpolate`, `spring`, `Easing.*` |
| 02 | sequence-text | `Sequence` nesting, text layout in the bundled font |
| 03 | offthreadvideo | `OffthreadVideo` on `public/testsrc.mp4` (the toolkit's media choice) |
| 04 | audio | `Audio` tag + frame-driven visual |
| 05 | transitions | `TransitionSeries` with `lib/transitions` `glitch` and `zoomBlur` |
| 06 | components | `lib/components` `AnimatedBackground`, `Label`, `Vignette`, `FilmGrain` |

Duration comes from `calculateMetadata`, so that API is covered too. The composition
bundles its own font (`public/fonts/BaselineSans.ttf`, a DejaVu Sans subset) — never
`system-ui` — and uses no network assets.

## Local use

```bash
cd tests/render-baseline && npm install && cd -

# Compare the current pin with a candidate in one go (restores the pin afterwards)
node scripts/render-baseline.mjs ab 4.0.518

# Or step by step
node scripts/render-baseline.mjs render before
node scripts/render-baseline.mjs install 4.0.518
node scripts/render-baseline.mjs render after
node scripts/render-baseline.mjs diff before after        # exit 1 on any change
```

Output lands in `out/<label>/` (stills) and `out/diff-<a>-vs-<b>/` (diff PNGs,
`report.md`, `report.json`). `--threshold=<percent>` relaxes the gate; it is `0` until
we have seen what real-world noise looks like across a few bumps.

## CI

`.github/workflows/render-baseline.yml` runs on PRs that touch a `package.json` under
`templates/`, `examples/`, `showcase/`, this directory, `lib/`, or the script itself. It
compares the base branch's pin with the PR's pin, posts the table as a PR comment, uploads
stills and diffs as artifacts, and fails the check when frames differ so a human looks.

`workflow_dispatch` takes a `version` input to try a candidate without opening a PR.

## Adding a block

Add a component to `BLOCKS` in `src/Baseline.tsx` and a matching label to
`BLOCK_LABELS` in `scripts/render-baseline.mjs`. Keep it deterministic: bundled font,
`random(seed)` from remotion rather than `Math.random`, no dates, no network.
