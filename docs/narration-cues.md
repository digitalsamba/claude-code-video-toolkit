# Motion that follows narration

Use `tools/cues.py` to place clicks, zooms, highlights, and other events on words
in narration. It reads existing word-level `.captions.json` files from
`tools/captions.py`, so it works with any voice provider and makes no API calls.

## Define named phrases

```json
{
  "version": 1,
  "cues": {
    "openSettings": { "phrase": "open settings", "offsetMs": -150 },
    "save": { "phrase": "apply", "occurrence": 2, "anchor": "end" }
  }
}
```

`phrase` matches contiguous whole words, ignoring case and punctuation. Curly
and straight apostrophes match. It does not guess approximate wording. When a
phrase appears more than once, set its one-based `occurrence`. An ambiguous or
missing phrase stops generation and names the cue that needs an edit.

`anchor` defaults to `start`; `end` uses the end of the final word. `offsetMs`
defaults to zero. A negative offset starts a movement before the phrase. A cue
that would fall before frame zero is an error.

## Generate timestamps

Run from the toolkit root:

```bash
uv run tools/cues.py --captions projects/my-video/public/audio/scenes/01-title.captions.json --spec projects/my-video/cue-spec.json --output projects/my-video/src/cues.json --fps 30
```

If the audio starts two seconds into its scene, add `--audio-offset-ms 2000`.
The output's `timeMs`, `startMs`, and `endMs` then use scene time. Place the
animation in that scene's Sequence. Leave the offset at zero when the animation
and audio already share a Sequence starting at the audio's first sample.

The output is versioned JSON with a `cues` object. Each named cue includes
`timeMs`, the phrase's `startMs` and `endMs`, and its matching occurrence.
`--fps` optionally adds frame numbers. Without `--output`, JSON goes to stdout.
Invalid timestamps, out-of-order captions, duplicate JSON keys, unknown spec
fields, and missing phrases fail with a nonzero exit code. A failed run keeps
the previous output file.

After changing narration, regenerate the audio, its captions, and then the cues.
This tool follows the timestamps it receives; stale captions cannot reflect a
new recording. Scene durations still need enough room for the new audio.

## Use in Remotion

```tsx
import { cueFrame, parseCueDocument } from '../../../lib/cues';
import cueData from './cues.json';

const cues = parseCueDocument(cueData);

// Inside the scene component, use the actual composition FPS:
const { fps } = useVideoConfig();
const clickAt = cueFrame({ document: cues, name: 'openSettings', fps });

// Use clickAt in Stage clicks, cursor/camera keyframes, Sequence.from,
// or a condition such as frame >= clickAt.
```

The helper computes from milliseconds each time. Changing a composition from
30 to 60 FPS therefore keeps events at the same spoken word, even if the cue
file contains cached 30 FPS frames. Frame rounding chooses the nearest frame,
with exact halves rounded up.

The [runnable example](../examples/narration-cues/README.md) connects generated
cues to the shared Stage, Cursor, camera, and UI state with included sample
captions. It needs no external assets.

## Verify

The Python checks use only the standard library:

```bash
uv run --no-project python -m unittest discover -s tests -p test_cues.py
```

With the example's Node dependencies installed, compile and check the helper
from the toolkit root:

```bash
examples/narration-cues/node_modules/.bin/tsc lib/cues.ts --target ES2020 --module commonjs --strict --outDir scratch/cues-ts
node --test tests/cues.test.cjs
cd examples/narration-cues
npm run typecheck
```
