# manim-ident: idents that say what the toolkit does

Two short idents drawn by [Manim](https://github.com/ManimCommunity/manim) from a brand's own `mark.svg`, each
ending on the flat mark, with sound placed from the scene. Manim is optional and experimental here; this example
needs it (`uv run tools/manim_render.py doctor`).

| Scene | What happens | Length |
|---|---|---|
| `Typed` | `/video` is typed at a caret; its slash becomes the mark's slash and the word folds into the play triangle | 2.8 s |
| `Chain` | Five lines of script become a waveform, the waveform a frame, the frame play, with one word under each | 3.5 s |

```bash
uv run tools/manim_render.py render examples/manim-ident/idents.py Typed --brand video-toolkit --out examples/manim-ident/out/typed.mp4
uv run examples/manim-ident/sound.py examples/manim-ident/out/typed.mp4      # typed-sound.mp4

# transparent, for a Remotion scene
uv run tools/manim_render.py render examples/manim-ident/idents.py Typed --brand video-toolkit --alpha --out examples/manim-ident/out/typed.webm
```

Each renders in a few seconds. Output lands in `out/` (gitignored).

## How the sound is placed

The scene notes the second each audible thing happens (`self.ev("key")` before a letter appears,
`self.ev("fold:0.55")` before the fold) and `manim_render.py` writes them beside the clip as
`<clip>.events.json`. `sound.py` reads that file: a key click on each letter, a whoosh across each morph, the same
key at half speed as the mark lands. A click is placed by its peak and a whoosh by the point where half its energy
has passed, and the script prints how far the clicks landed from their letters (under a millisecond). Change the
animation and the sound moves with it. Whether it sounds right is for a person to say.

Sounds (`sfx/`, each with its credit file, all CC0 from Freesound): a mechanical key and two whooshes.

## For another brand

`--brand NAME` takes the colours and `brands/NAME/assets/mark.svg`. The scenes assume the mark's first shape is
what the caret or bar becomes and its second is what everything folds into; a mark with other parts needs the
last beat rewritten, and the rest stands. A transparent ident is only right over the ground it was drawn for:
render a dark version from `mark-on-dark.svg`.

Lessons from making these are in [`docs/manim.md`](../../docs/manim.md).
