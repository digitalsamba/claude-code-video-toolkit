# manim-diagram: a diagram that builds on the narration

A 19 second diagram drawn by [Manim](https://github.com/ManimCommunity/manim): a request goes from the browser to
the API to the database, a cache goes in between, and the response time falls from 400 ms to 90. Each step appears
on the word that names it. It is the kind of scene a sprint review needs for a backend change that a screen
recording can't show. Manim is optional and experimental here (`uv run tools/manim_render.py doctor`).

```bash
uv run tools/manim_render.py render examples/manim-diagram/flow.py Flow --brand video-toolkit \
  --captions examples/manim-diagram/audio/flow.captions.json --out examples/manim-diagram/out/flow.mp4

# transparent, for a Remotion scene that draws the ground, the title and the captions
uv run tools/manim_render.py render examples/manim-diagram/flow.py Flow --brand video-toolkit \
  --captions examples/manim-diagram/audio/flow.captions.json --alpha --out examples/manim-diagram/out/flow.webm
```

About 4 seconds to render; the transparent version takes about 20 more to convert.

## What is here, and what is not

- `flow.py`: the scene. Every step is `self.at(V.cue("cache", lead=0.35))` then a `play`.
- `audio/flow.txt`: the narration. `audio/flow.captions.json`: when each word is said.
- **No audio file.** The word timings came from a scratch voice that is not ours to ship. Voice `flow.txt` with
  `tools/voiceover.py`, run `tools/captions.py` on the result, and render again: the diagram follows the new
  timings, and fails with the cue's name when an animation no longer fits before the next word.

In Remotion the transparent clip is `<OffthreadVideo src={staticFile('manim/flow.webm')} transparent muted />`
with the narration as an `<Audio>` beside it.

Lessons from making this are in [`docs/manim.md`](../../docs/manim.md).
