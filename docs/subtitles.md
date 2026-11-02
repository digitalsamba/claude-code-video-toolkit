# Export SRT and WebVTT subtitles

`captions.py` creates word timings for burned captions. `subtitles.py` turns the
same Caption[] JSON into a downloadable subtitle track for players and publishing.
It does not transcribe audio, alter a video or use a project manifest.

```bash
uv run tools/subtitles.py narration.captions.json --output narration.srt
uv run tools/subtitles.py narration.captions.json --format vtt --max-words 7 --output narration.vtt
```

Input entries contain `text`, `startMs` and `endMs`. Times must be finite,
nonnegative, ordered and non-overlapping. Groups keep the first start and last
end timestamp. Text is escaped so it cannot introduce markup or extra cues.
Existing outputs and replacing the input file are rejected. Files publish only
after the complete UTF-8 track has been written.

This extracts subtitle export from the variants feature in #142. It reuses the
same local output helper as the other standalone tools. No dependencies, API
calls or model downloads are added. Toolkit code is MIT.

```bash
uv run python -m unittest discover -s tests/python -p test_subtitles.py -v
```
