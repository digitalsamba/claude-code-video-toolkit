# Local production capability example

This creates a four-second synthetic recording, two tone "voices", background
music, reference art, a small dataset and source-backed text. No model, cloud
account or paid service is used. The tones exercise timing, not speech quality.

```bash
uv sync --extra vision
uv run examples/production-capabilities/create.py scratch/capabilities-demo
uv run tools/build.py --project scratch/capabilities-demo \
  --plan scratch/capabilities-demo/production.json --json
uv run tools/build.py --project scratch/capabilities-demo \
  --plan scratch/capabilities-demo/production.json --budget-usd 0 --run --json
```

Run the last command again to verify checkpoint resume. Use a new directory to
make another demo. Generated media is under the project's `out/` directory.

The 11 tasks lock/check evidence, lock/preview reference-conditioned character
shots, render a narrated chart, synchronize two existing utterances, duck music,
assemble highlights, track/reframe a moving subject, composite it with an attached
graphic, and run final QC. The character plan is previewed, not executed. Its
unknown provider estimates block a capped run until explicitly supplied.

For real speaker identification, generated voices, FLUX/LTX generation and neural
human matting, follow [the capability guide](../../docs/advanced-production.md).
