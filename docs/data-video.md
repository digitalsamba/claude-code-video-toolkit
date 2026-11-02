# Videos from CSV or JSON charts

The existing data-viz-chart example is authored for one dataset. This command validates caller-supplied CSV/JSON and chart settings, creates a bar or line animation, derives narration text from the values and records the source hash.

Uses The existing Matplotlib and NumPy dependencies, encoded with FFmpeg. No new dependencies, hosted service or model download. Matplotlib has its BSD-style license and NumPy is BSD-3-Clause; toolkit code is MIT; FFmpeg terms depend on its build.

## Usage

```bash
uv run tools/data_video.py --spec chart.json --output chart.mp4 --script-output narration.txt
uv run tools/data_video.py --spec chart.json --output chart.mp4 --narration voice.wav --save-report chart-report.json --run
```

Create `data.csv` and `chart.json` in the same directory:

```csv
period,series,value
Jan,Product A,10
Feb,Product A,20
Jan,Product B,15
Feb,Product B,12
```

```json
{
  "version": 1,
  "type": "line",
  "dataset": "data.csv",
  "title": "Monthly usage",
  "unit": "users",
  "width": 1280,
  "height": 720,
  "fps": 30,
  "durationSeconds": 6
}
```

Use `"type": "bar"` for animated bars. JSON datasets use an array of row objects
with the same columns. Optional `columns` maps `period`, `series` and `value` to
different column names. Rows determine period order; each series needs a value
in every period. Limits are 12,000 rows, 500 periods and 24 series.

Preview validates data and reports the script and provenance without rendering.
`--script-output` and `--save-report` explicitly write new files. Use `--run` to
publish the video. Supplied narration sets its duration, rounded up to the next
video frame; otherwise `durationSeconds` defaults to 6. Inputs must contain finite
numeric values. Existing outputs are rejected. The narration text describes
values, not their meaning. A changed dataset invalidates the prepared report.

## Implementation reference

[Manim](https://github.com/ManimCommunity/manim), [bar_chart_race](https://github.com/dexplo/bar_chart_race) and [Matplotlib](https://github.com/matplotlib/matplotlib) were compared. Reuse Matplotlib already installed by the toolkit.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_data_video.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
