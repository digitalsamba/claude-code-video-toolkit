# Tracked video reframing

cutdown.py has static or per-segment crop focus. This command follows a selected subject or face per frame, smooths camera movement, and produces a portrait or square crop with the source audio.

Uses OpenCV CSRT or bundled Haar face detection, then FFmpeg encoding. Optional OpenCV contrib headless is installed through the vision extra. No hosted services, tokens or separately fetched model weights. OpenCV is Apache-2.0; toolkit code is MIT; FFmpeg terms depend on its build.

## Usage

```bash
uv sync --extra vision
uv run tools/reframe.py --video input.mp4 --box 100,50,180,240 --width 1080 --height 1920 --output portrait.mp4
uv run tools/reframe.py --video input.mp4 --box 100,50,180,240 --width 1080 --height 1920 --output portrait.mp4 --run
```

`--box` is x,y,width,height in pixels on the first source frame. Alternatively use `--faces`, which follows the nearest detected face and refuses large identity jumps. `--smoothing` and the lost-frame limit control the track. Lost detections are marked as held frames; split the source at cuts or occlusions that exceed the limit.

The output dimensions must be even and at most 4096. Constant-frame-rate media is required. `--run` performs tracking and writes both the reframed video and a source-hash-bound `.track.json` report. Preview validates inputs without running the tracker or writing outputs. Tracking is a heuristic, not identity recognition; review the crop. Existing output files are rejected.

## Implementation reference

[OpenCV tracking](https://github.com/opencv/opencv_contrib/tree/4.x/modules/tracking) was compared with [Norfair](https://github.com/tryolabs/norfair). CSRT tracks the supplied box without an additional detector runtime.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_reframe.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
