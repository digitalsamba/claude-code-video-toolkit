# Production features

The production tools connect existing generation and rendering commands through
validated project data, resumable execution and local review. Each tool supports
`--help`. Run Python tools from the toolkit root through `uv run`.

| Task | Entry point | Guide |
| --- | --- | --- |
| Validate or migrate existing projects | `tools/project.py` | [Project manifest](project-manifest.md) |
| Execute/resume a video build | `/build-video`, `tools/build.py` | [Builds](build.md) |
| Rebuild only affected work | `tools/build.py --force-task ID` when needed | [Content fingerprints](build.md) |
| Forecast costs and select draft/final settings | `tools/build.py --budget-usd N --profile NAME` | [Budgets](budgets.md) |
| Review pictures, narration and QC | `/review-video`, `tools/review.py` | [Visual review](review.md) |
| Restore or search generated assets | `tools/asset_history.py` | [Asset history](asset-history.md) |
| Link clicks/zooms to narration words | `tools/cues.py`, `lib/cues.ts` | [Speech cues](narration-cues.md) |
| Compose landscape, portrait and square | `templates/storyboard` | [Responsive formats](multi-format.md) |
| Record reusable browser actions | `npm run flow` in playwright/ | [Demo flows](replayable-demos.md) |
| Make language/hook/CTA variations | `tools/variants.py` | [Variants](variants.md) |
| Export subtitle files | `tools/subtitles.py` | [Subtitle export](variants.md) |
| Add local templates or providers | `tools/extensions.py`, `tools/provider.py` | [Developer kit](developer-kit.md) |
| Verify workflows, browsers and video output | `scripts/verify_features.py` | [Verification](verification.md) |

Start with the included offline production plan or storyboard template. Builds
preview by default; generation executes only with `--run`. Hosted costs must be
estimated explicitly to enforce a cap. The visual page is local; it does not
replace Remotion Studio or provide hosted collaboration.

The storyboard template is the first data-driven adapter for project variants
and responsive layouts. Existing templates retain their working configurations
and can migrate individually. Translations are supplied/reviewed as JSON; no
unconfigured translation service is called. Local adapter contracts let you
extend providers without adding another hosted service to toolkit core.
