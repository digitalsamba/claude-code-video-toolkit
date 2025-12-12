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

**Phase:** 3 - Templates & Brands
**Focus:** Multi-session project system, unified commands

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

### Phase 3: Templates & Brands 🔄 IN PROGRESS

**Brand Profiles:**
- [x] Default brand profile
- [x] Digital Samba brand profile
- [x] `/brand` command - list, edit, or create brands

**Templates:**
- [x] Product demo template
- [x] `/video` command - unified project management
- [x] `/template` command - list available templates
- [x] Shared component library (`lib/`)
- [ ] Tutorial template
- [ ] Changelog/release notes template

**Template-Brand Integration:**
- [x] Brand loader utility (`lib/brand.ts`)
- [x] Templates use `brand.ts` for theming
- [x] `/video` generates brand.ts from selected brand

**Multi-Session Project System:**
- [x] Project schema (`lib/project/types.ts`)
- [x] Filesystem reconciliation
- [x] Auto-generated CLAUDE.md per project
- [x] `/skills` command

**Review & Validation:**
- [x] `/scene-review` command - dedicated scene-by-scene review with Remotion Studio
  - [x] Starts Remotion Studio for visual verification
  - [x] Walks through scenes one by one (not summary tables)
  - [x] Generic - works with any template's config
  - [x] `/video` delegates to `/scene-review` when phase is `review`
  - [x] `/generate-voiceover` warns if review incomplete
- [ ] Pre-render review (timing, sync, polish)
- [ ] Asset validation (ffprobe checks)
- [ ] Enhancement suggestions
- Note: `/review` name clashes with Claude Code built-in PR review - using `/scene-review`

**Contribution & Examples:**
- [x] `/contribute` command
- [x] `examples/` directory
- [x] CONTRIBUTORS.md

**Testing:**
- [x] Test new project creation with scene-centric flow
- [ ] Test project resumption (multi-session)
- [ ] Verify filesystem reconciliation
- [x] Verify CLAUDE.md auto-generation

### Phase 4: Polish & Advanced

**Output & Accessibility:**
- [ ] Multi-format output (MP4, WebM, GIF, social formats)
- [ ] Subtitle generation from voiceover scripts
- [ ] Thumbnail auto-generation

**Skills:**
- [ ] Video accessibility skill
- [ ] Terminal recording skill (asciinema)
- [ ] Video timing skill

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
| elevenlabs | stable | Audio generation |
| ffmpeg | beta | Asset conversion |
| playwright-recording | beta | Browser demo capture |

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
| Templates | 2 | sprint-review, product-demo |
| Brands | 2 | default, digital-samba |
| Skills | 4 | 2 stable, 2 beta |
| Tools | 3 | voiceover, music, sfx |
| Commands | 8 | video, brand, template, skills, contribute, record-demo, generate-voiceover, scene-review |
| Components | 9 | AnimatedBackground, SlideTransition, Label, Vignette, LogoWatermark, SplitScreen, NarratorPiP, Envelope, PointingHand |
| Examples | 2 | digital-samba-skill-demo, sprint-review-cho-oyu |

---

## Related Files

| File | Purpose |
|------|---------|
| `BACKLOG.md` | Unscheduled ideas and future enhancements |
| `CHANGELOG.md` | Historical record of changes |
| `toolkit-registry.json` | Machine-readable inventory |
