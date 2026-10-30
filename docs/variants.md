# Language and campaign versions

Keep one source project and provide reviewed translations or alternate hooks in
a variant specification. The tool validates every version before writing any
files. It preserves the source and discards stale narration/captions after a
language or narration change. It never translates text silently.

```json
{
  "version": 1,
  "glossary": {"Remotion": "Remotion"},
  "variants": [
    {"id": "es", "kind": "locale", "locale": "es", "scenes": {
      "intro": {"title": "Hola Remotion", "onScreenText": "Tu próximo vídeo", "narration": "Bienvenido"}
    }},
    {"id": "hook-b", "kind": "campaign", "scenes": {
      "intro": {"title": "Ship your next video"}
    }}
  ]
}
```

Language variants require explicit translations for every nonempty title,
onScreenText and narration. Campaign variants override only chosen fields.
Glossary entries enforce exact terms. Voice settings may supply provider, cloud
and voiceId per variant. Supply translated audio/captions as scene overrides,
or plan generation with `--with-audio`.

```bash
uv run tools/variants.py --project projects/demo --spec translations.json
uv run tools/variants.py --project projects/demo --spec translations.json --with-audio --write
uv run tools/build.py --project projects/demo --plan projects/demo/variants/production.json --json
uv run tools/build.py --project projects/demo --plan projects/demo/variants/production.json --run
```

Outputs are snapshot manifests, render props and per-scene scripts under
`variants/<id>/`. Paths inside these snapshots remain relative to the **source
project**, where assets are shared. The storyboard template consumes these
props directly and receives separate output paths. Its generated production
plan connects narration, captions, actual-audio retiming and rendering. Other
templates can consume snapshot manifests through their own adapters.

`--with-audio` plans hosted generation but never executes it. Generation may
incur charges when you later run the plan. Estimate unknown hosted costs before
using a build budget. Existing variant directories are never overwritten; use
another ID to preserve older work. Edit typography and check layout for every
language. Translation quality and voice language support require review.

Export existing word captions as downloadable subtitle tracks:

```bash
uv run tools/subtitles.py public/audio/intro.captions.json --format srt --output out/intro.srt
uv run tools/subtitles.py public/audio/intro.captions.json --format vtt --output out/intro.vtt
```

No new hosted translation provider is introduced. Claude can draft the JSON,
and a reviewer can approve wording before generation.
