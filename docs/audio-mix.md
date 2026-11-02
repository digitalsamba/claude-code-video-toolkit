# Speech-aware audio mixing

addmusic.py mixes a music bed at a fixed volume. This command reduces music during speech, lets it recover in pauses, and places timed sound effects on an existing video.

Uses FFmpeg sidechaincompress, amix and adelay. No new Python dependencies or paid calls. Uses the installed FFmpeg/ffprobe. The toolkit is MIT; FFmpeg licensing depends on the installed build.

## Usage

```bash
uv run tools/audio_mix.py --video input.mp4 --music music.wav --speech narration.wav --output mixed.mp4
uv run tools/audio_mix.py --video input.mp4 --music music.wav --speech narration.wav --output mixed.mp4 --run
```

Music loops to the video's duration. If `--speech` is omitted, the video's audio controls ducking and is kept as the spoken track. With an isolated speech file, that file replaces the video's original audio in the mix. `--music-gain`, `--speech-gain`, `--threshold`, `--ratio`, `--attack` and `--release` control the mix. Attack/release use milliseconds. The limiter controls peaks; this command does not perform final loudness normalization.

`--events effects.json` accepts an array such as `[{"path":"chime.wav","atSeconds":1.5,"gain":0.5}]`. Paths resolve relative to the JSON file. There can be up to 128 events. Existing outputs and input/output collisions are rejected. Preview prints the planned FFmpeg command; `--run` publishes a completed output.

## Implementation reference

[FFmpeg](https://github.com/FFmpeg/FFmpeg) filters were compared with the [Shotcut](https://github.com/mltframework/shotcut) editor stack; use the existing FFmpeg runtime.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_audio_mix.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
