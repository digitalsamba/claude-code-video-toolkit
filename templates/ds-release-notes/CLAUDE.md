# ds-release-notes template

Synthetic release-notes video for Digital Samba. Read `README.md` first, then the `synthetic-ui-demo` skill. The
template ships as the release 650 build so it renders as is.

## Where the 650 screens came from (the repos are not needed to build)
- Transcription language: `react-frontend` `src/features/room/settings/tabs/CaptionsTab.tsx`, strings `settings:captions.*`.
- Samsung Internet join: `react-frontend` `src/features/join/shared/PendingMediaNotice.tsx`, strings `join:pending_media.*`.
- Pinned avatar: `react-frontend` `TiledWithContent/RoomGrid.tsx` (content column 80% default, side strip ≥166px);
  avatar clamp `min(size, max(100cqmin - 32px, 60cqmin))`.
- Hide tiles for (DS Free): `react-account-center` `settings/templates/_components/sections/recordings.js` and
  `form-controls/roles-select.js`.
- Custom room link warning (DS Free): `react-account-center` `schedule/_components/custom-slug.js`. Inline red helper
  text, only when editing, longer text when invites are on.
- The room's icons come from `react-frontend/src/assets/fonts/icons/selection.json` (extracted to
  `src/room/icon-paths.json`); the room theme from `src/theme/`.

## Layout
- `src/Release.tsx`: scene order, lengths, VO start per phrase (`voice: [file, at][]`), SFX.
- `src/Stage.tsx`: browser window, camera and cursor. Every UI scene sits inside one.
- `src/room/`: room kit. Toolbar on the left (the default); active speaker shown only by a green mic.
- `src/ac/`: account-center kit (shared with `examples/ds-free-schedule-demo`).
- `src/scenes/`: one file per release item; click frames and targets are constants at the top.

## Audio
- Clone Lugh from `assets/voices/lugh-ref-expressive.wav` (pitch-expanded, slowed) and never from the raw clip; the
  raw clip gives flat lines at 200+ wpm that then need an audible stretch.
- One `.txt` per sentence, three takes, `tools/pick_takes.py` to choose, `tools/captions.py` for word timings.
- Short closing sentences still rush (~185 wpm); if one grates, pick another take from `takes.json` by hand.

## Gotchas
- `logo_mono.svg` is black; it needs `filter: brightness(0) invert(1)` on dark backgrounds.
- Measure click targets off rendered stills. A guessed coordinate is usually off by 20-50px once the camera zooms.
- A changed phrase moves everything after it: re-measure, update the header comment and scene lengths, re-check stills.
- People are FLUX.2 portraits, labelled "AI-generated" on screen. Keep names and data plainly fictional.
- Don't invent UI copy. If a string wasn't pulled from the repo, leave the row without it.
