# Verify production features

Run the offline Python checks, including a local provider → subtitle export →
real FFmpeg render → final QC → checkpoint resume workflow:

```bash
uv run python scripts/verify_features.py
```

For real template rendering and browser interaction tests, install the three
Node projects and Playwright's browser first:

```bash
npm ci --prefix playwright
npm ci --prefix templates/storyboard
npm ci --prefix examples/narration-cues
cd playwright
npx playwright install chromium
cd ..
uv run python scripts/verify_features.py --media --browser
```

FFmpeg/ffprobe must be on PATH. To use an installed Chromium browser, set
PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH to its absolute executable path. Verification
selects Remotion's Chrome for Testing mode for this full Chromium executable.
It runs headlessly and does not contact hosted generation providers. The report is
saved to scratch/feature-verification.json; browser screenshots and small real
renders accompany it. The report records only checks that actually ran.
Media verification includes the storyboard build adapter through final QC,
and an actual render with Spanish variant props.

The CI workflow runs on Windows and Linux. It adds production-path coverage to
the existing Remotion version baseline checks: changed inputs/outputs, failures,
resuming, estimate caps, asset history, translations, provider contracts, shared
schema, actual format dimensions and persisted review actions.
