# Transcript-based highlight cuts

cutdown.py renders ranges you choose. This command proposes those ranges from complete transcript sentences, removes repeated takes and shortens long aligned pauses before rendering them.

Uses Python sentence/keyword ranking and FFmpeg trim/concat. No new dependencies when a transcript is supplied. Optional local transcription reuses the existing whisper extra and may download weights. No hosted API is added. Toolkit code is MIT; OpenAI Whisper is MIT and FFmpeg terms depend on the installed build.

## Usage

```bash
uv run tools/highlights.py --video input.mp4 --transcript words.json --seconds 45 --keyword launch --output highlights.mp4 --save-plan edit.json
uv run tools/highlights.py --video input.mp4 --transcript words.json --seconds 45 --keyword launch --output highlights.mp4 --run
```

Input is either toolkit `Caption[]` with `startMs`, `endMs`, `text`, or a Whisper-style object with `segments` and optional aligned `words`. Timestamps in Whisper objects are seconds. Sentences must end in punctuation. Missing aligned words keep the whole segment together; incomplete trailing text is omitted explicitly in the report.

Repeated `--keyword` values affect ranking. Selection is deterministic first-pass editing, not an assessment of editorial quality. Cuts stay in source order, preserve complete sentences, avoid adjacent speech with padding, and fit the duration budget. Review the printed edit plan before `--run`. `--save-plan` explicitly writes a new JSON file even in preview. `--transcribe` uses the existing optional Whisper extra and can download weights; supplied transcripts need no model. Outputs never overwrite existing files.

## Implementation reference

[Auto-Editor](https://github.com/WyattBlue/auto-editor) audio activity and [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) visual cuts were compared. Neither supplies sentence-aware selection on its own.

No source code or model weights from those projects are copied into this change.
This utility reads files and writes a separate output. It does not read a project
manifest or use a task graph, queue, provider registry or new CI workflow.

## Tests

```bash
uv run python -m unittest discover -s tests/python -p test_highlights.py -v
```

Tests use local fixtures and actual media processing where applicable. No paid inference is part of verification.
