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
- [ ] **Chaptered long-form template.** 10–30 min YouTube videos: render chapter by chapter with a cache,
      one continuous ducked music bed, -14 LUFS, chapter timestamps and an `.srt`. Probably built on
      concept-explainer-short's 16:9 path. A contributor attempt (#127) had the right design.
- [ ] **Transcript-driven highlights for `cutdown.py`.** Pick whole sentences from the captions JSON by
      keyword or score, so a short from a long render doesn't need hand-typed `--segment` times.
- [ ] **Spec-driven release-notes job for Lugh.** A `ds-release-notes` path in `video_job.py`: the agent
      writes scenes as "screen, action, target, narration" over the screen kits the template already has,
      and gets the same submit/status/result flow. Covers only screens a kit already rebuilds. Until then
      Lugh writes a release brief (`memory/release-videos/release-<N>.md` on longarm) and a Claude Code
      session builds from it; build this once a few briefs show which scenes recur.
- [ ] **`video_job.py wait`**: block up to a few minutes streaming stage lines, for agents with a
      per-call time limit.

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
| frontend-design | stable | Visual design refinement |
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
| Skills | 14 | remotion, remotion-official, elevenlabs, ffmpeg, playwright-recording, frontend-design, qwen-edit, runpod, acestep, ltx2, moviepy, ideogram4, synthetic-ui-demo, video-toolkit |
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
