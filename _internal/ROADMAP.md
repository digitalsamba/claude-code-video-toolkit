# Video Toolkit Roadmap

This document tracks the development of claude-code-video-toolkit.

**Repository:** https://github.com/digitalsamba/claude-code-video-toolkit

---

## Vision

An open-source, AI-native video production workspace for Claude Code, featuring:
- Reusable templates for common video types
- Brand profiles for consistent visual identity
- Claude skills providing deep domain knowledge
- Automated asset pipelines (recording, conversion, audio generation)
- Slash commands for guided workflows

---

## Current Status

**Release:** v0.24.0 (2026-11-01)
**Phase:** 4 - Polish & Advanced
**Focus:** making what shipped in v0.21–v0.24 solid (agent video jobs, synthetic UI demos), then the
gaps contributors keep pointing at: long-form video and transcript-driven editing.
**Recent (v0.20–v0.24):** SoulX talking heads, `qc.py` (loudness, picture and layout checks), burned
captions, `cutdown.py`, `/thumbnail`, `modal_setup.py`, synthetic UI demos (`Stage` + `Cursor`) and the
`ds-release-notes` template, clone prosody tools, agent video jobs (experimental), concept-explainer-short in 16:9.

---

## Next Up

Ordered roughly by priority. Each is sized to one session or one PR.

- [x] **Harden agent video jobs.** Done in #156 (2026-11-02): 10 MB-safe renders, a 3x faster build, a
      cost event on every exit, `resume`, `prune`, `events`, `warnings`, per-host ETAs, MCP run end to end.
      Windows is reviewed but still unrun.
- [x] **Narration-linked cues for `Stage`.** Done: `voice()` / `cue('save')` in `lib/components/cues.ts`, with
      `FreeCustomLink` in ds-release-notes as the worked example (render unchanged frame for frame).
- [ ] **Blender shot specs, the next round.** Shipped experimental and optional (`tools/blender_render.py`,
      `lib/blender/`, the `blender` skill, three examples). In order:
      1. *Finish the product stage:* props that move (turn, slide in), a `text` prop, a tablet and a
         browser-window prop, and its own exposure range so a light backdrop doesn't sit at the limit.
      2. *Contact shadows on transparent clips:* a shadow pass rendered on its own and laid under the clip by
         `BlenderClip`.
      3. *A second character and a standing pose*, then walking.
      4. *Your own `.blend` as a set:* a set kind that loads the user's scene and names its objects as subjects,
         so the cameras, checks and clips work on anything built in Blender.
      5. *A `script` hook in the spec:* a Python file run after the build and before `check`, so anything `bpy`
         can do is repeatable and still verified. With 4, this is the path to all of Blender for a toolkit user.
      6. *Imported rigs and actions:* a cast kind for a rigged glTF or FBX whose named actions become verbs.
      7. *A `blender` visual kind in `video_job.py`:* the cast from a kit's `cast/`, the ETA from `doctor`.
      8. *Platform proof:* a first Windows run, a Linux 4.2 re-run, and a weekly CI job that runs `check` on the
         examples.
      9. *Checks:* face brightness on dark skin; an idle big enough that a held pose doesn't read as a stopped
         clip to `qc.py`.
      10. *First real uses:* a video timed with `cue:`, 9:16 framing of the desk, a device reveal in
          `ds-release-notes` (design note in its README), and a second "Made with itself".
- [ ] **Chaptered long-form template.** 10–30 min YouTube videos: render chapter by chapter with a cache,
      one continuous ducked music bed, -14 LUFS, chapter timestamps and an `.srt`. Probably built on
      concept-explainer-short's 16:9 path. A contributor attempt (#127) had the right design.
- [x] **Transcript-driven highlights for `cutdown.py`.** Done in #162: `highlights.py` picks whole sentences
      from a word-timed transcript and writes the ranges as `cutdown.py --segment` arguments.
- [ ] **Tutorial template and batch production.** "How to do X" videos, one task each: steps called out on
      screen, the same intro and outro, synthetic UI from the screen kits (the account-center kit from
      `ds-free-schedule-demo` first). A batch is one spec per tutorial built in a row, and an upload queue that
      spreads it over days and schedules each video, since the YouTube quota is about 6 uploads a day. Pilot: DS
      Free account-center tutorials for the Digital Samba channel.
- [ ] **`Terminal` component.** A synthetic terminal in `lib/components` that types commands and prints output on
      narration cues, as `Stage` does for a browser, so toolkit demos and tutorials show Claude Code without a
      screen recording. The design-sources sample has a hand-built one to start from. Covers the "terminal
      recording" skill below without recording anything.
- [ ] **Translate a finished video.** Translate each scene's `.txt`, re-voice with the same clone (Qwen3 is
      multilingual), re-caption and re-time with the existing tools. Synthetic UI videos can also switch the
      rebuilt screens to the app's own locale files (`public/locales/<lang>/`), so the product UI is translated
      too, which a screen recording can't do.
- [ ] **Publish kit for a render.** Chapters for the description from the scene list, one SRT for the whole
      video (per-scene captions joined at their offsets) uploaded with `--captions`, a default thumbnail, and the
      site and playlist links, all from `project.json`, so `/publish` is one step. Overlaps with the chaptered
      long-form template above.
- [ ] **Spec-driven release-notes job for Lugh.** A `ds-release-notes` path in `video_job.py`: the agent
      writes scenes as "screen, action, target, narration" over the screen kits the template already has,
      and gets the same submit/status/result flow. Covers only screens a kit already rebuilds. Until then
      Lugh writes a release brief (`memory/release-videos/release-<N>.md` on longarm) and a Claude Code
      session builds from it; build this once a few briefs show which scenes recur.
- [x] **`<DataChart>` in `lib/components`.** `LineChart` (narration keyframes, growing y axis, cue-timed bands,
      several series with a legend), `BarChart`, `Counter` and `SplitBar`, themed from a brand
      (`examples/data-chart-stars`, #175 and the second cut). Still open: CSV loading.
- [ ] **Data segments for recurring internal reviews.** One JSON per metric, refreshed by a fetch script, rendered
      at two zooms from the same components: a **quarterly** view for townhalls (the whole period, trend and
      milestones) and a **monthly** view for KPI reviews (one month by week or day, against the month before).
      Only the narration lines that carry numbers get re-voiced, and the scene's data checks fail the render
      when a refreshed number no longer matches what is said.
- [ ] **`video_job.py wait`**: block up to a few minutes streaming stage lines, for agents with a
      per-call time limit.
- [ ] **Music that follows the picture.** Today music only matches the video's length (`--duration`), and
      `pick_music.py` only checks after the fact whether a take happens to have a note on the act cuts. Three
      ways, cheapest first:
      1. *Picture to the music:* snap the flexible frames (act cuts, line entrances; never narration) to the
         chosen take's beats within ±0.3s, from the onsets `pick_music.py` already finds.
      2. *Tempo from the timeline:* choose the BPM so the act cuts fall on bar lines, or move the cuts to the
         bars. Measure first how closely ACE-Step keeps the BPM it's asked for.
      3. *The music turns when the picture turns:* the colour rule for sound. Generate the bed, then
         `music_gen.py --repaint` a short window round each act cut (times from `design.md`'s acts) with a
         prompt for the change. Least proven: repaint seams are unheard.
      Judge each with `audition.py` renders, by ear, not by score. Pilot: `projects/still-wheel-hour` (no
      voice, four acts with clear turns).
- [ ] **Parody shorts, the next round.** The `parody-short` skill (2026-11-07) is proven on Slack and an event
      runsheet for context, and (same day) on an invented cast, vertical cuts and a brand edition. Still to try
      or build:
      1. *More context sources:* tickets, `git log` and PR titles, postmortems, calendars, CI and dashboards
         (listed untested in `context.md`). Record which ones gave a line that got a laugh.
      2. *A context pass as a step:* the agent writes `bible.md` from whatever the user has connected, and the
         person asking strikes lines before the script is written.
      3. *Reusable genre kits:* each genre's overlay, fonts, palette and shot helpers (VHS, creature fog,
         Dickens grade) as a folder a new project copies, so the second Christmas video starts from the first.
      4. *Cast reuse:* done by hand for an invented cast (`projects/cast/`, two videos). Still to do: keep a cast in
         a kit (`kits/`) so the next video only re-costumes.
      5. *Faces that talk:* a costumed cast member speaking their catchphrase with SoulX, not just moving.
      6. *Audience test:* show to the team and note which moments got a laugh, to rank the "why it works" list.
      7. *One layout, two shapes:* the vertical cuts were a copy of the 16:9 shots file with new positions. A layout
         table per shot (16:9 and 9:16 coordinates side by side) would keep the two in step when a gag changes.
      8. *A public example:* the invented-cast shorts have no real people, so one could go in `examples/` (assets
         are ~60 MB of clips; check the size budget first).

---

## Phases

### Phase 1: Foundation ✅ COMPLETE

- [x] Sprint review template with theme system
- [x] Config-driven video content
- [x] `/video` slash command (unified project creation)
- [x] Narrator PiP component
- [x] Remotion skill (stable)
- [x] ElevenLabs skill (stable)

### Phase 2: Skills & Automation ✅ COMPLETE

**Skills:**
- [x] FFmpeg skill (beta)
- [x] Playwright recording skill (beta)

**Python Tools:**
- [x] `voiceover.py` - CLI for ElevenLabs TTS
- [x] `music.py` - CLI for background music
- [x] `sfx.py` - CLI for sound effects

**Commands:**
- [x] `/generate-voiceover` - streamlined audio generation
- [x] `/record-demo` - guided Playwright recording

**Infrastructure:**
- [x] Playwright recording setup (`playwright/`)
- [x] Centralized config (env var with registry fallback)

### Phase 2.5: Open Source Release ✅ COMPLETE

- [x] Directory restructure for public release
- [x] Brand profiles system (`brands/`)
- [x] Environment variable support
- [x] README, LICENSE (MIT), CONTRIBUTING.md
- [x] Documentation (`docs/`)
- [x] GitHub repo published

### Phase 3: Templates & Brands ✅ COMPLETE

Open items moved to Phase 4 below.

- [x] Brand profiles: default, Digital Samba, `/brand` command
- [x] Templates: sprint-review, sprint-review-v2, product-demo, concept-explainer-short (9:16 and 16:9),
      ds-release-notes (the changelog/release-notes template)
- [x] `/video` and `/template` commands, shared component library (`lib/`)
- [x] Transitions library (`lib/transitions/`) and gallery (`showcase/transitions/`)
- [x] Brand loader (`lib/brand.ts`); `/video` generates `brand.ts`
- [x] Multi-session project system: schema, filesystem reconciliation, per-project CLAUDE.md, `/skills`
- [x] `/scene-review`, pre-render QC and asset validation (`tools/qc.py`)
- [x] Qwen3-TTS: standalone tool, `--provider qwen3`, cloning, `/voice-clone`, VoiceDesign (v0.2)
- [x] Modal as a second cloud GPU provider for every GPU tool; one-command setup (`tools/modal_setup.py`)
- [x] Generators: FLUX.2 Klein, Ideogram 4 (and 4.5 opt-in), LTX-2.3 with style LoRAs, ACE-Step 1.5
- [x] 60db TTS provider; TTS pacing QC and `--max-wpm`
- [x] YouTube publishing (`/publish`, OAuth, scheduling, thumbnails)
- [x] `/contribute`, `examples/`, CONTRIBUTORS.md, the Community add-ons list
- [x] Exact Remotion pins, render-baseline A/B in CI, grouped monthly Dependabot bumps, and a CI check
      that every pinned project is covered (#23)

### Phase 4: Polish & Advanced 🔄 IN PROGRESS

**Shipped:**
- [x] Social cut-downs: 9:16 / 1:1 / 4:5 (`tools/cutdown.py`)
- [x] Burned word-level captions (`tools/captions.py` + `<Captions>`)
- [x] Thumbnails (`/thumbnail`, Ideogram 4 from `project.json` + brand)
- [x] `qc.py --render` picture checks and `--layout` text checks (overflow, WCAG contrast, minimum size)
- [x] Talking heads: SoulX-FlashHead as the default (EchoMimicV3 tried and removed)
- [x] Synthetic UI demos: `Stage` + `Cursor` in `lib/components`, the `synthetic-ui-demo` skill
- [x] Clone prosody: `voice_prosody.py` measure/expand, `pick_takes.py`
- [x] Agent video jobs (experimental): `video_job.py`, `video_mcp.py`, `skills/video-toolkit/`
- [x] Blender shot specs (experimental, optional): `blender_render.py`, `lib/blender/`, `BlenderClip`, the `blender` skill

**Open:**
- [ ] Make Qwen3-TTS the default voiceover provider (replacing ElevenLabs)
- [ ] Other publish destinations (LinkedIn, X, S3/R2 CDN)
- [ ] Multi-format output (WebM, GIF)
- [ ] Tutorial template
- [ ] Monthly engineering-review template (ticket-driven, presenter-narrated) — in-tree but not yet ready to publish
- [ ] Test project resumption (multi-session) and filesystem reconciliation end to end
- [ ] Skills: video accessibility, terminal recording (asciinema), video timing
- [ ] Qwen3-TTS v0.2 image on RunPod (built; push waiting on a registry token refresh)

---

## Skill Maturity Levels

| Status | Meaning |
|--------|---------|
| **draft** | Just created, untested, may have errors |
| **beta** | Functional, needs real-world validation |
| **stable** | Battle-tested, well-documented, recommended |

### Current Skill Status

| Skill | Status | Notes |
|-------|--------|-------|
| remotion | stable | Core framework knowledge |
| remotion-official | stable | Synced from remotion-dev/skills (weekly via GitHub Actions) |
| elevenlabs | stable | Audio generation |
| ffmpeg | beta | Asset conversion |
| playwright-recording | beta | Browser demo capture |
| video-design | beta | Colour by act, type and chart rules, design.md handoff from any design source, contact-sheet critique |
| qwen-edit | stable | AI image editing prompting patterns |
| runpod | stable | Cloud GPU setup, Docker images, endpoint management |
| modal | beta | Alternative cloud GPU provider — faster cold starts, every GPU tool deployed |
| acestep | stable | ACE-Step 1.5 music generation prompting |
| ltx2 | beta | LTX-2 video generation prompting, style LoRAs |
| moviepy | stable | Python composition, deterministic text overlay |
| ideogram4 | beta | Structured JSON captions for in-image text |
| synthetic-ui-demo | beta | Product screens rebuilt from source and driven by `Stage`; used for DS releases 649 and 650 |
| video-toolkit | draft | Agent-facing skill for video jobs (`skills/video-toolkit/`); experimental |

---

## Review Process

**draft → beta:**
- Verify code examples work
- Test core functionality
- Document issues in `_internal/reviews/`
- Fix critical issues

**beta → stable:**
- Use in a real project
- Gather feedback
- Complete documentation
- No known critical issues

---

## Metrics

| Category | Count | Items |
|----------|-------|-------|
| Templates | 5 | sprint-review, sprint-review-v2, product-demo, concept-explainer-short, ds-release-notes |
| Brands | 2 | default, digital-samba |
| Skills | 14 | remotion, remotion-official, elevenlabs, ffmpeg, playwright-recording, video-design, qwen-edit, runpod, acestep, ltx2, moviepy, ideogram4, synthetic-ui-demo, video-toolkit |
| Tools | 31 | voiceover, qwen3_tts, sixtydb_tts, music, music_gen, sfx, redub, addmusic, sync_timing, qc, align_captions, captions, cutdown, dewatermark, locate_watermark, notebooklm_brand, image_edit, upscale, flux2, ideogram4, ltx2, chain_video, sadtalker, soulx, verify_setup, modal_setup, youtube_upload, voice_prosody, pick_takes, video_job, video_mcp |
| Commands | 15 | setup, video, brand, template, skills, contribute, record-demo, generate-voiceover, scene-review, design, versions, redub, voice-clone, thumbnail, publish |
| Components | 14 | AnimatedBackground, SlideTransition, Label, Vignette, FilmGrain, LogoWatermark, SplitScreen, NarratorPiP, Captions, Envelope, PointingHand, MazeDecoration, Stage, Cursor |
| Transitions | 7 | glitch, rgbSplit, zoomBlur, lightLeak, clockWipe, pixelate, checkerboard |
| Examples | 10 | hello-world, quick-spot, data-viz-chart, ds-crt-stinger, sky-blue-short, digital-samba-skill-demo, sprint-review-cho-oyu, the-space-between, ds-release-649, ds-free-schedule-demo |

---

## Related Files

| File | Purpose |
|------|---------|
| `BACKLOG.md` | Unscheduled ideas and future enhancements |
| `CHANGELOG.md` | Historical record of changes |
| `toolkit-registry.json` | Machine-readable inventory |
