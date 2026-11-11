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
Its author-written "718 stars" headline starts fading in at 10 seconds over 0.5 seconds:

```json
{"headline": "718 stars", "headlineStartSeconds": 10, "headlineFadeSeconds": 0.5}
```

Set `headlineFadeSeconds` to zero for an immediate reveal. Timing is relative to the final video,
including when narration sets its duration. The headline must finish fading before the video ends.
Edit the spec to change title, source, headline, duration or size. CSV and JSON row datasets also
work, with columns `period`, `series`, `value`. Dataset paths are relative to the spec, and custom
column names use `"columns": {"period":"month", "series":"product", "value":"users"}`.
Every series must have a finite value for each period. `type` accepts `line` or `bar`.

Line periods that parse as dates use real calendar distances and readable month ticks such as
`Jan 2026`. Supported labels include ISO dates/timestamps, `2026-01`, and `Jan 2026` or `January 2026`.
Dates are sorted chronologically with their values. Missing days or months retain their true gap.
Use all dates or all categorical labels in one line chart; categories such as `Alpha` and `Beta`
keep equal spacing. Duplicate date instants are rejected.

Bar charts show the period above the plot. While values interpolate, the label shows the transition,
such as `Jan → Feb`, then holds the final period with its values. Bars and the first line use the
brand primary. Additional lines share one neutral, and a single line has no redundant legend.

`--brand <name>` loads colours and `fonts.primary` from `brands/<name>/brand.json`. The first
installed font in that CSS font list is used. `system-ui` resolves to an available platform font;
generic families such as `sans-serif` also resolve without reporting a missing font. `fontFallback`
is true when an earlier requested family is unavailable and a later family or DejaVu Sans is used.
The report names the actual font. Fonts are never downloaded. Overlays and chart labels use it.

Supply author-recorded or generated narration with `--narration narration.wav` to anchor duration
to the audio. Write the script yourself. The tool records the dataset hash and never derives
narration or headline claims from the values.

Reruns overwrite completed outputs atomically. `--dry-run` validates and prints a JSON report
without writes. `--save-report provenance.json` saves the dataset/brand hashes after rendering.
Use the output with `uv run tools/qc.py --render out/chart.mp4 --fix` when delivering narration.

For charts inside Remotion, use the existing `lib/components/DataChart.tsx` instead.
