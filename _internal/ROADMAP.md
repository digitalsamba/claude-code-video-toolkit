# Video Toolkit Roadmap

What we're building next for claude-code-video-toolkit, and what shipped recently. Unscheduled ideas live in
[`BACKLOG.md`](BACKLOG.md), every change in [`CHANGELOG.md`](CHANGELOG.md), and the full inventory (skills, tools,
templates, components, their status) in [`toolkit-registry.json`](toolkit-registry.json).

**Repository:** https://github.com/digitalsamba/claude-code-video-toolkit

---

## Status

| | |
|---|---|
| **Release** | v0.30.0 (2026-11-22) |
| **Phase** | 4, polish and advanced |
| **Focus** | Lectures and how-to episodes as templates, 3D characters fitted from a generated model (Blender, optional), and the long-form and translation gaps contributors keep pointing at |

---

## In flight

| What | Where | State |
|------|-------|-------|
| Tutorials for a product: a `screen` scene kind in the `howto` template, `Stage` changes, a brand's house style | branch | Built, waiting on feedback |
| Thai explainers: a script skill and a `font` and caption language for `concept-explainer-short` | #253 | Split requested |
| Kit design layer (fonts, layouts, motif, logo) | #188 | Parked draft |
| Working in other coding agents | #240 | Discussion. The name stays; first step is the same brief in Claude Code and Codex, compared |

---

## Next up

Roughly in priority order within each group. Each item is sized to one session or one PR.

### Video types

- [ ] **Tutorial template and batch production.** One task per video, steps called out on screen, the same intro
  and outro, synthetic UI from the screen kits. A batch is one spec per tutorial and an upload queue that spreads
  it over days (the YouTube quota is about 6 uploads a day). The `howto` template shipped in v0.30.0; the batch
  and the upload queue are open.
- [ ] **Chaptered long-form template.** 10 to 30 minute videos rendered chapter by chapter with a cache, one
  ducked music bed, -14 LUFS, chapters and an `.srt`. Likely on concept-explainer-short's 16:9 path. #127 had the
  right design.
- [ ] **Translate a finished video.** Translate each scene's `.txt`, re-voice with the same clone, re-caption and
  re-time. Synthetic UI videos can also switch the rebuilt screens to the app's own locale files, which a screen
  recording can't do.
- [ ] **Data segments for recurring reviews.** One JSON per metric, refreshed by a script, shown at a quarterly
  zoom (townhalls) and a monthly one (KPI reviews). Only lines with numbers are re-voiced, and a data check fails
  the render when a number no longer matches the narration.
- [ ] **`DataChart` CSV loading.**

### Sound

- [ ] **Music that follows the picture.** Today music only matches the video's length. Cheapest first:
  1. Snap flexible frames (act cuts, line entrances, never narration) to the chosen take's beats, within 0.3 s.
  2. Pick the BPM so act cuts land on bar lines. Measure first how well ACE-Step keeps the BPM asked for.
  3. `music_gen.py --repaint` a short window round each act cut so the music turns when the picture does.

  Judge each by ear with `audition.py`. Pilot: a project with no voice and clear act turns.

### Agents

- [ ] **`video_job.py wait`:** block for a few minutes streaming stage lines, for agents with a per-call time limit.
- [ ] **Release-notes jobs.** A `ds-release-notes` path in `video_job.py`: scenes as "screen, action, target,
  narration" over the template's screen kits. Build it once a few hand-made release briefs show which scenes recur.
- [ ] **Other coding agents (#240).** Run one brief through Claude Code and Codex and write down what differs.

### 3D with Blender (optional)

Blender stays optional: nothing else in the toolkit depends on it.

- [ ] **A brief and a storyboard as tools.** A guided brief (idea, lines, tone, length, what the agent may not
  decide) and a `storyboard` step that builds the board from `check` stills and the timeline, one row per beat.
- [ ] **Rig:** shoulder skinning for a full reach back, arms over the head, and a full sip that stretches a
  fitted sleeve.
- [ ] **Product stage:** props that move, a `text` prop, a tablet and a browser window, and its own exposure range.
- [ ] **Contact shadows on transparent clips**, rendered as a pass and laid under the clip by `BlenderClip`.
- [ ] **Your own `.blend` as a set**, with its objects named as subjects so cameras, checks and clips work on it.
- [ ] **A `script` hook in the spec:** a Python file run after the build and before `check`, so anything `bpy` can
  do is repeatable and still checked.
- [ ] **Imported rigs:** a rigged glTF or FBX whose named actions become verbs.
- [ ] **A `blender` visual kind in `video_job.py`**, casting from a kit's `cast/`, with the ETA from `doctor`.
- [ ] **Platform proof:** a first Windows run and a weekly CI job running `check` on the examples. (Linux on
  Blender 4.2 passes.)
- [ ] **More checks:** face brightness on dark skin; an idle big enough that a held pose doesn't read as a stopped
  clip to `qc.py`; what a shot is for (low angle, a prop in view on a word) as rows `check` tests.
- [ ] **Cloud render on Modal, spike first.** Opt in only, with an estimate and a `--max-usd` cap. Measure one
  EEVEE frame headless on an L4 (it may lack a graphics stack) against Cycles with OptiX; no cost goes in the docs
  until it is measured.
- [ ] **Self-hosted image-to-3D on Modal**, so a fitted character needs no paid vendor key. Read each model's
  licence first.
- [ ] **First real uses:** a video timed with `cue:`, 9:16 framing of the desk, a device reveal in
  `ds-release-notes`.

Ideas from other Blender and AI projects (surveyed 2026-11-14, nothing copied): a strip of frames per acting beat
and a bones view beside the contact sheet; studio light ratios by material and an HDRI world; dropping noisy maps
from generated models and a weights check in the fit; library props by Poly Haven id with a credit file; shot
templates for two people. Not yet: motion capture clips as verbs.

### Parody shorts

- [ ] More context sources (tickets, `git log`, postmortems, calendars, CI), noting which gave a line that landed.
- [ ] A context pass as a step: the agent writes `bible.md`, the person strikes lines before the script.
- [ ] Genre kits: each genre's overlay, fonts, palette and shot helpers as a folder a new project copies.
- [ ] A photo cast kept in a kit, so the next video only re-costumes.
- [ ] Cast members who speak their catchphrase with SoulX.
- [ ] One layout table per shot with 16:9 and 9:16 positions side by side.
- [ ] A public example from the invented cast (check the size budget: about 60 MB of clips).

### Platform and housekeeping

- [ ] Qwen3-TTS as the default voiceover provider instead of ElevenLabs.
- [ ] Publish destinations beyond YouTube (LinkedIn, X, an R2 CDN).
- [ ] Multi-format output (WebM, GIF).
- [ ] Monthly engineering-review template: in the tree, not ready to publish.
- [ ] An end-to-end test of project resumption and filesystem reconciliation.
- [ ] Skills for video accessibility and video timing.
- [ ] Push the Qwen3-TTS v0.2 image to RunPod (built; waiting on a registry token).

---

## Shipped

Detail for each release is in [`CHANGELOG.md`](CHANGELOG.md).

| Release | Date | Headline |
|---------|------|----------|
| v0.30.0 | 2026-11-22 | `templates/lecture` and `templates/howto`, a logo in 3D and a brand's ident, `TerminalSession`, `data_video.py`, fitted characters in close shots |
| v0.29.x | 2026-11-14 | Fitted 3D characters that stand, walk, sit and carry a mug; kitchen and lecture hall sets; `reference_capture.py` |
| v0.28.0 | 2026-11-12 | Blender shot specs and `BlenderClip`; `Terminal`; several narrators per brand; `publish_kit.py`; `youtube_upload.py --check` |
| v0.27.0 | 2026-11-09 | `sfx_search.py`, `addmusic.py --duck`, SRT and VTT captions, face-tracked cut-downs, `highlights.py`, `youtube_stats.py`, `copy_lint.py`, the Pages site |
| v0.26.0 | 2026-11-08 | `parody-short` skill, `LivingPhoto`, `ltx_extend.py`, `sync_check.py`, motion-stop QC, agent jobs with the person's own media |
| v0.25.0 | 2026-11-05 | `video-design` with design sources, `DataChart`, `PageTurns`, narration cues, `pick_music.py`, `audition.py`, `contact_sheet.py` |
| v0.20 to v0.24 | 2026-08 to 11 | SoulX talking heads, `qc.py`, burned captions, `cutdown.py`, `/thumbnail`, `modal_setup.py`, synthetic UI demos, clone prosody tools, agent video jobs |

Phases 1 to 3 (foundation, skills and automation, open source release, templates and brands) are complete.

---

## Skill maturity

Each skill's current status is in the registry (`skills[].status`).

| Status | Meaning | To move up |
|--------|---------|------------|
| **draft** | Just created, untested | Verify the examples, test the core path, note issues in `reviews/`, fix the critical ones |
| **beta** | Works, needs real-world use | Use it in a real project, gather feedback, finish the docs, no known critical issues |
| **stable** | Proven and documented | |
