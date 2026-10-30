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

## Production capabilities

Install the optional vision extra for actual CSRT tracking and compositing tests:

```bash
uv sync --extra vision
uv run python scripts/verify_features.py --capabilities
# All verification, including existing Remotion/browser checks:
uv run python scripts/verify_features.py --capabilities --media --browser
```

The capability checks measure music attenuation/recovery, two distinct utterances
at their assigned timestamps, sentence edit decisions versus output duration,
subject motion, chart pixel changes, source/claim mismatches, and composite pixels.
The eleven-task example renders six media outputs, runs final QC, then verifies
all eleven tasks resume. Artifacts and their hashes appear in the report under
`scratch/capability-verification-*/`. No paid provider calls occur.

FLUX/LTX transport tests use simulated responses through the existing engines.
Real reference conditioning quality and TTS voice quality require configured
providers and human review. WhisperX/pyannote and Demucs inference are optional
external environments; the default checks use supplied diarized/recorded inputs.

To exercise the real optional RVM model locally, install `--extra matting` and set
`VIDEO_TOOLKIT_RVM_MODEL` to the downloaded official FP32 ONNX model before
running verification. Otherwise the two RVM model tests explicitly skip. Model
weights are not included or downloaded in CI.
