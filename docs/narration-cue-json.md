# Export named narration cues as JSON

The existing `voice()` helper in `lib/components/cues.ts` resolves narration cues
inside Stage/Remotion compositions. Keep using it there. `tools/cues.py` exports
a batch of named timestamps for external tools and non-Remotion animations.
It reads existing Caption[] files and does not transcribe or generate audio.

```bash
uv run tools/cues.py --captions examples/narration-cues/public/narration.captions.json --spec examples/narration-cues/cue-spec.json --fps 30 --output cues.json
```

Each named spec entry has a `phrase`, optional one-based `occurrence`, `anchor`
of `start` or `end`, and `offsetMs`. Ambiguous repeated phrases require an
explicit occurrence. `--audio-offset-ms` positions the narration on a timeline.
The output retains milliseconds; optional cached frames use the requested FPS.
Regenerate it whenever narration or caption timing changes.

`lib/cues.ts` validates exported JSON through `parseCueDocument(unknown)` and
converts a named cue with `cueFrame({document, name, fps})`. It uses the actual
composition FPS, including fractional rates, rather than trusting cached frames.
It does not replace the existing Stage cue API or alter current scenes.

No dependencies, hosted services, model downloads or API costs are added.
Toolkit code remains MIT. Node 24 can run the TypeScript consumer tests directly:

```bash
uv run python -m unittest discover -s tests/python -p test_cues.py -v
node --experimental-strip-types --test tests/cues.test.cjs
```
