# QC before and after a render

What `qc.py` checks in a project and in a render, and how to place a sound effect on a generated movement. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Pre-render QC (before every render)

```bash
uv run tools/qc.py --project projects/my-video                       # assets, timing, pacing, loudness
uv run tools/qc.py --project projects/my-video --render projects/my-video/out/video.mp4
uv run tools/qc.py --project projects/my-video --fix                 # normalise scene VO loudness
uv run tools/qc.py --render projects/my-video/out/video.mp4 --fix    # writes video.loudnorm.mp4
```

Exit 1 on errors (narration cut off, a missing asset the template doesn't guard), so it can
gate a render. **Loudness is the check nothing else does:** the templates mix VO + music
without normalising, and renders land well under YouTube's -14 LUFS, which YouTube turns
*down* when too loud but never *up* when too quiet. Scene VO targets -16 LUFS (headroom for
music); the render targets -14 LUFS / -1 dBTP. Optional keys (`backgroundMusicFile`,
`chimeFile`, `screenshotFile`, `voiceoverFile`) are warnings, not errors: the templates skip
them with `getStaticFiles()`, so the render succeeds silently without them.

**`--render` also watches the picture** (a few seconds even for a 4-minute render; `--skip-frames` to
skip):
- **Still stretches:** more than 3s (`--max-still`) where nothing on screen changes. The comparison is per
  tile, so a moving cursor or a talking-head PiP counts as change. The last 3s may hold.
- **Black gaps:** black for 0.5s or more mid-video, usually a gap between sequences or media that didn't load.
- **Motion that stops dead:** part of the frame moving for 3s or more, then frozen for 2s or more while the rest of the
  video goes on. Usually an image-to-video clip that ran out and held its last frame; lengthen it with `tools/ltx_extend.py`
  rather than holding or looping it. The whole-frame still check misses it because something else is always moving.
- **Empty first frame:** background only. A warning on vertical and square renders, whose frame 0 is
  the Shorts/Reels cover, and info on landscape renders, which get a custom thumbnail.
- **Audio:** a missing or silent audio track is an error. `--allow-silent` makes it info, for a deliverable that is
  silent on purpose; the picture checks still run.

**On a short vertical or square render it also measures the opening** (3 minutes or less, the length YouTube
treats as a Short), because a viewer in a feed stays or swipes on the first seconds. One line of numbers, and a
warning when:
- nothing on screen moves in the first 2 s;
- the first shot runs over 5 s before the picture changes (a move inside the shot or new text is not counted);
- the first sound comes later than 1 s, or the first 2 s are 10 dB or more under the video as a whole.

`--layout` adds a warning when a vertical composition has no text on screen in its first 2 s. A slow, quiet
opening can be the point; the check makes it a choice. Speech is not detected: it hears sound, not words.

Still stretches are warnings, not defects: a hold the narration needs is fine, but give it a next action
(see `.claude/skills/remotion/motion-craft.md`).

**`--layout` measures the text itself** (Remotion projects; needs Node, `playwright/` installed and Chrome or
Playwright's Chromium). It bundles the project, opens it in a browser, seeks a frame a second with Remotion's own
hooks and reports, as warnings:
- text off the canvas or clipped by its own box, held across two samples (entrances and exits pass through);
- contrast under WCAG AA at the size the text is shown, grouped by colour pair, with a passing colour suggested;
- text that never reaches 2.6% of the short side (28px at 1080p), the smallest a phone viewer reads.

Text under `data-qc="product"` is the product's own UI (Stage marks its browser window; mark a hand-built phone or
tablet mockup the same way). It's reported separately, as info, without the overflow check, since the camera crops
it on purpose: the video can't fix the product's colours, but it's worth knowing where the UI is hard to read.
`data-qc-ignore` skips an element. About 10s to bundle, then ~0.15s a sample.

## Sound effects on generated movement

A sound laid over a generated clip (a sneeze over an LTX face, a thud when a jacket lands) is placed at a frame by
hand, and is easily half a second off the movement. Place it by measurement, then check the render:

```bash
# before rendering: where in the clip should the sound start? (its loudest moment meets the movement)
uv run tools/sync_check.py --clip public/people/theo-sneeze.mp4 --sfx public/audio/sfx/sneeze.mp3 --window 2.5:4.5
uv run tools/sync_check.py --clip public/people/theo-desk.mp4 --sfx public/audio/sfx/thump.mp3 --kind impact --window 1.5:3.8

# after rendering, before presenting: every cue in one go (exit 1 if any is off by more than 0.1 s)
uv run tools/sync_check.py --video out/video.mp4 --cues sync-cues.json
```

`--kind peak` (default) puts the sound on the biggest movement: a sneeze, a laugh. `--kind impact` puts it where a
movement stops: a landing, a stamp. `--kind onset` puts it where a sustained movement starts (a badge lifted
slowly), which has no single spike for `peak` to find. A hit that sets off something bigger (a clap that knocks a
shape up, a light that comes on) is `peak`: as `impact` it reads up to 0.36 s late, on whatever it set off. Something picked up (chalk off a table, a cup raised) is
`onset` too: measured as `impact`, a chalk tap read 0.37 s early, because the check found the hand settling
after the lift. Give a `--window` around the action: an extended clip's seam
or a later gesture can be bigger than the one the sound is for. Keep the cues in the project's `sync-cues.json`
(`{name, sfx, at, region, kind, window}`, times in the render, regions as `w:h:x:y`; sound paths relative to the
cue file or the toolkit root) so the check re-runs on every render. Keep the window clear of the shot's dissolve and
any text that animates in near the movement.
