# Adaptive storyboard

One project.json produces landscape, portrait and square compositions. Edit
scene titles and onScreenText, then supply optional public-relative visual,
audioFile, captionsFile and narratorFile assets. Portrait and square stack text
and picture; landscape places them side by side. Media uses contain framing.

```bash
npm ci
npm run studio
npm run render
npm run render:portrait
npm run render:square
```

Pass `--props render-props.json` with `{"project": YOUR_MANIFEST, "safeZones": true}`
to preview conservative safe-zone guides. Disable guides for final delivery.
Margins leave room for typical social controls; inspect the destination app and
adjust `lib/formats.ts` for its current overlays. This template is the first
adapter for the shared manifest; legacy templates keep their own configuration.

No assets or credentials are needed for the included three-scene example. For
production content, keep titles short and run layout QC on all output formats.

From the toolkit root, `uv run tools/build.py --project PROJECT --builtin --json`
previews a complete build from the manifest. Add `--run` to generate missing
narration, prepare audio-driven scene lengths, render and check the output.
Generation may incur provider charges; the included silent sample needs none.
