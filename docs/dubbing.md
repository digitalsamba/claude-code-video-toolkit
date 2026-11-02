# Speaker-timed replacement audio

redub.py handles a replacement narration track. This command assigns supplied recordings or existing TTS voices to labelled speaker segments, fits them into their original slots and optionally preserves a supplied background stem.

Uses Supplied speaker labels, existing voiceover.py and FFmpeg atempo/adelay. Supplied recordings run locally with no API cost or new dependency. Existing TTS providers keep their normal cost and model terms. No WhisperX, pyannote or Demucs integration or new hosted provider is included. Toolkit code is MIT; FFmpeg terms depend on its build.

## Usage

```bash
uv run tools/dubbing.py import speakers.json --output dub.json
uv run tools/dubbing.py render --video input.mp4 --spec dub.json --output dubbed.mp4
uv run tools/dubbing.py render --video input.mp4 --spec dub.json --output dubbed.mp4 --run
```

Import accepts `{"segments":[{"start":0,"end":1,"text":"Hello.","speaker":"A"}]}` with times in seconds. It writes a version-1 dubbing spec with `speakers` and `segments`. Edit every placeholder voice or supply each segment's `audio` recording relative to the spec file.

For example, a segment can be `{"id":"line-1","speaker":"A","startSeconds":0,"endSeconds":1,"text":"Hello.","audio":"hello.wav"}`. With supplied recordings, the speakers entry can be `{"A":{}}`. A top-level `background` path preserves a supplied music/effects stem. The original source audio is replaced; no stem is extracted automatically.

Existing `elevenlabs`/`60db` mappings use `voiceId`; `qwen3` uses `speaker`, with optional `refAudio`/`refText`. Reusing these providers can incur their normal charges. Preview never synthesizes. `--run` fits utterances with pitch-preserving tempo changes and refuses fits beyond the configured bounds. It changes audio timing, not mouth motion. Model-based diarization and stem extraction are intentionally excluded under CONTRIBUTING.

## Implementation reference

[WhisperX](https://github.com/m-bain/whisperX), [pyannote.audio](https://github.com/pyannote/pyannote-audio) and [Demucs](https://github.com/facebookresearch/demucs) were compared. This PR accepts labelled segments/stems supplied by the caller; external analysis/model stacks remain outside the toolkit.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_dubbing.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
