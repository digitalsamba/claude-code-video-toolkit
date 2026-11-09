# Formats: vertical cuts and brand editions

One parody short usually ships as several cuts. None of them need new GPU work: the cast clips, backgrounds,
voice and music are shared, and only the Remotion layout changes. Each cut is a separate `<Composition>` in the
same project, built from a copy of the 16:9 shots file with new positions.

## Vertical (9:16) for phones

Rebuild the layout for the phone, don't crop the 16:9 render (`cutdown.py` is for videos that can't be
re-rendered).

- **Portrait clips fit better than in 16:9.** The 704x768 LTX clips fill a 2x2 grid of tall tiles, or the whole
  screen, with no side bars or blended edges.
- **A video call is already phone-shaped:** the speaker fills the top, the others sit in a row beneath it, the
  toolbar runs along the bottom.
- **Split a two-person beat top and bottom** (a desk dispute, a standoff) instead of left and right.
- **Break titles onto two lines** ("YOU'RE / ON MUTE"); a 16:9 title is too wide at a readable size.
- **Lower thirds and toasts move up and in:** keep text above the bottom ~300 px and inside a 60 px side margin,
  where phone apps put their own buttons.
- **Frame 0 is the cover** on Shorts and Reels: start with the title or the most recognisable shot already on
  screen, not a dark establishing shot (`qc.py --render` warns).
- Keep shot timings identical to the 16:9 cut, so the same narration, music and sound effects line up.

## A brand edition

A company can have its own version: the same video, with its product's real UI where the generic one used a
made-up interface. Learned making a Digital Samba edition of a video-call horror short (2026-11).

- **Reuse a rebuilt UI kit, don't screenshot.** If the toolkit already rebuilt the product's screens from source
  (the `synthetic-ui-demo` method; `templates/ds-release-notes/src/room/` is Digital Samba's room), copy the kit
  into the project. Scale its logical canvas to the frame (1440x810 is exactly 1080p at 4/3) and fit the old
  content box into its stage with one uniform scale, so the shots keep their coordinates.
- **Give the brand chrome the same props as the generic chrome** (timer, a "leave" offset, mic state), and
  swap it in with an import alias. The shots don't change.
- **Gags move with the UI:** a "Leave" button that runs away becomes the product's own hang-up button fleeing
  up its toolbar.
- **Don't let a joke land on the product.** In the generic cut, "connection unstable" and a frozen face are
  about video calls in general. On the brand's UI they read as the product failing. Blame something outside it
  ("Theo's home Wi-Fi is unstable") or cut the gag.
- **Keep unreleased features out**, even as background props.
- The end card gets the brand's logo; the generic cut stays brand-free, for the public showcase.
- Brand editions are the brand's call: render them locally, and publish only when the brand owner says so.
