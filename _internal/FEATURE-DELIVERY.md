# Production feature delivery

PR branch: `codex/ux-dx-production-workflows` on `AsharibAli/claude-code-video-toolkit`.
Target: `digitalsamba/claude-code-video-toolkit:main`, based on `eddb488`.

The feature implementations were developed on separate local branches. This
publication branch presents the same verified code as focused conventional
commits for implementation, tests, documentation and registry integration.
The original feature branches and integration checkout remain available locally.

| Feature | Main implementation commit | Delivered behavior |
| --- | --- | --- |
| Project manifests | `a3cbe35` | Validated, versioned JSON with legacy extension fields preserved. |
| Resumable builds | `f9d1384` | Preview or execute generation, timing, rendering and QC with checkpoints. |
| Selective regeneration | `fe7c8de` | Hash inputs/settings/outputs and rebuild affected tasks and dependents. |
| Asset history | `e47add6` | Search immutable snapshots and provenance, then restore checked bytes. |
| Narration-linked motion | `55586b8` | Resolve named words/phrases to cursor, camera and animation cues. |
| Visual storyboard review | `0a53e42` | Preview scenes and narration, save notes and expire approvals after edits. |
| Project budgets | `6193308` | Preview estimates, enforce caps, select profiles and reconcile charges. |
| Responsive formats | `ec19539` | Recompose one project for landscape, portrait and square. |
| Replayable browser demos | `663d3cb` | Record editable steps, replay them and export targets/timing/video. |
| Language/campaign versions | `057863e` | Validate translations and alternate hooks; coordinate audio, captions and output. |
| Local developer kit | `e7a11a9` | Discover/validate provider and template contracts; scaffold an offline template. |
| Workflow verification | `f861dab` | Exercise real CLI, media, browser and recovery paths on Windows/Linux CI. |

## Local verification

- 74 Python tests passed, including legacy manifests, failure recovery, content
  fingerprints, estimate caps, concurrent asset history, translations and Unicode.
- Strict TypeScript checks and focused JavaScript tests passed.
- Real landscape, portrait and square videos were rendered and probed.
- A Spanish variant rendered, and a storyboard build completed through final QC.
- Browser action recording/replay and saved review-page actions passed in a real browser.
- Ruff crash-class checks, registry idempotence and synchronized agent guidance passed.

Run `uv run python scripts/verify_features.py --media --browser` to regenerate
the local report at `scratch/feature-verification.json` and its media artifacts.
Setup is documented in `docs/verification.md`.

Windows was exercised locally. The PR configures Windows and Linux CI; remote
results are reported by GitHub. Hosted generation/billing were not called during
verification. Estimates are planning controls, not provider billing guarantees.

For usage and the scope of each adapter, read `docs/production-features.md`.
