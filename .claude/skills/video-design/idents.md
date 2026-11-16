# Marks and idents

A mark is the logo's shape. An ident is the mark moving, with a sound, at the start and the end of a video. This
file is the bar for both and the method that produced the toolkit's own. Making a 3D ident needs Blender, which
is optional; a mark and its sheet need nothing but the toolkit.

## Marks

**The bar.** A mark has to work where it will be seen, and most of those places are small:

- at 16 px (a favicon, a browser tab) and 32 px, where a thin gap closes and a small part disappears;
- on the brand's light ground, its dark ground and its own colour, which usually takes two or three versions
  (as drawn, on dark, one colour);
- inside a circle, because avatars are round and the corners of the square are cut off;
- in one colour at 70% in the corner of a video frame, as a watermark;
- as the last frame of an ident, so it has to be something that can stand still.

**The method.**

1. **Start from the subject, in words.** Write what the thing is in one line ("a terminal that makes video") and
   list what a viewer already associates with each half. Ideas come from joining the two lists, not from
   decorating a letter.
2. **Draw five directions, not one.** Flat SVG, by hand, one colour plus the brand colour at most. Give every
   shape an `id`: a 3D ident moves the parts by name. One of the five should be the obvious idea, so the others
   are measured against it.
3. **Run the sheet on each.** `uv run tools/mark_sheet.py --svg mark.svg --brand <name>` draws the sizes, the
   pixel zooms, the grounds, the circle and the watermark on one PNG and warns about contrast under 3:1, a part
   the circle cuts, and a part lost at 16 px. Add `--on-dark` and `--mono` once those versions exist.
4. **Say how each would move, in one sentence.** A mark with no move in it makes a weak ident. The toolkit's
   mark is a slash and a play triangle; its move is a clap, so the sound is the move.
5. **Critique each against the bar, rank them, recommend one.** The still at 16 px decides more often than the
   drawing at 512 px. The toolkit's first draft (a prompt chevron and a caret) ranked third because at rest it is
   the `>_` every terminal app has.
6. **Search for lookalikes before it is public.** An image search of the mark on white. Note the nearest shapes
   and anything in the same field with the same colours. That is a first pass, not a trademark search.

Keep the mark in `brands/<name>/assets/` as `mark.svg`, `mark-on-dark.svg` and `mark-white.svg`, with the same
shapes and ids in each.

## Idents

**The bar.**

- **One mark, one move, one sound.** The move and the sound do the work; the mark stays simple. If the move needs
  three sentences it is three moves.
- **Short.** An open of 1.5 to 2.5 s and a close of 3 to 5 s. The open must not delay the hook, so a Short gets
  the close only, or the short open.
- **Siblings.** The close is the open played back or completed, so the pair reads as one thing.
- **It works with the sound off.** The move alone has to say what happened.
- **It ends as the mark.** The last frame of the open is the avatar. Anything else on screen then is a mistake.
- **The brand's colours survive it.** A flash or a glow that turns the brand colour into another colour is not
  the brand, even for three frames at rest. At the hit, a brighter version of the same hue is the limit.
- **It is yours.** No other brand's mark, and no copy of a famous ident's move or sound.

**The method.**

1. **Animatics first.** Three ideas as rough drafts with a sound sketch each, shown side by side. Choose with
   the sound on. A choice made from stills picks the best still, not the best move.
2. **One family, not one ident.** Keep the move and the sound; vary the staging. The toolkit has the mark alone
   on a dark stage (everywhere, and the only one for Shorts) and the same clap on a keycap (full episodes).
3. **Build from a spec.** `tools/blender_render.py` with a `logo` prop and an `ident` shot (`docs/blender.md`,
   "A logo and an ident"; the `blender` skill has the verbs). The timing is the list of beats, so a change is a
   number, and the vertical version is the same spec at `--res 1080x1920`.
4. **Put the sound on the picture's biggest change.** That frame is what the eye takes as the hit. `render`
   checks it for every `hit`. With `tools/sync_check.py`, measure a hit with `--kind peak`: as an `impact` it
   reads late, because a hit sets off something bigger that stops later.
5. **Check the last frame as a still.** `check` compares it with the flat SVG: parts in place, extras gone, the
   lens square on, each colour within 10 Delta E.
6. **Leave room.** If a wordmark or a title may sit with the mark, move the mark in the frame with the shot's
   `offset` now, so adding the words later does not change the ident.
7. **Say what nobody has heard.** Levels and sync can be measured. Whether the sound is right cannot: a person
   listens before it is used.

8. **Put it in the brand's slot.** `brands/<name>/ident/` with an `ident.json` (lengths, the hit's frame, the
   frame a title may come in) and the flat mark in `brands/<name>/assets/mark.svg`; the format is in
   `docs/creating-brands.md`. A template that reads the slot plays the ident, and keeps its own 2D opening when
   the slot is empty, so a video never depends on the ident being there.

Without Blender, the same bar holds for a 2D ident in Remotion: the mark's shapes as SVG, one move on a `cue()`
frame, one sound on that frame.
