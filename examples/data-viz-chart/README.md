# data-viz-chart

An animated chart using Matplotlib for the data and Pillow text overlaid in MoviePy for titles,
source attribution and author-written headline numbers. The example delegates to `tools/data_video.py`,
so the bundled data and custom datasets use the same renderer.

```bash
uv sync
uv run examples/data-viz-chart/build.py
uv run examples/data-viz-chart/build.py --spec my-chart.json --brand digital-samba --output out/chart.mp4
```

The default `chart.json` renders the bundled `[date, value]` star series to a 15-second `out.mp4`.
Edit the spec to change title, source, headline, duration or size. CSV and JSON row datasets also
work, with columns `period`, `series`, `value`. Dataset paths are relative to the spec, and custom
column names use `"columns": {"period":"month", "series":"product", "value":"users"}`.
Every series must have a finite value for each period. `type` accepts `line` or `bar`.

`--brand <name>` loads colours and `fonts.primary` from `brands/<name>/brand.json`. The first
installed font in that CSS font list is used. If unavailable, the report records the fallback
to DejaVu Sans. Fonts are never downloaded. Title, source and headline text use that same font.

Supply author-recorded or generated narration with `--narration narration.wav` to anchor duration
to the audio. Write the script yourself. The tool records the dataset hash and never derives
narration or headline claims from the values.

Reruns overwrite completed outputs atomically. `--dry-run` validates and prints a JSON report
without writes. `--save-report provenance.json` saves the dataset/brand hashes after rendering.
Use the output with `uv run tools/qc.py --render out/chart.mp4 --fix` when delivering narration.

For charts inside Remotion, use the existing `lib/components/DataChart.tsx` instead.
