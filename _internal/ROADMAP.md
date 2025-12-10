# Video Toolkit Roadmap

This document tracks the development of claude-code-video-toolkit.

> **Note:** When committing progress, update `_internal/CHANGELOG.md` with changes before pushing.

## Vision

An open-source, AI-native video production workspace for Claude Code, featuring:
- Reusable templates for common video types
- Brand profiles for consistent visual identity
- Claude skills providing deep domain knowledge
- Automated asset pipelines (recording, conversion, audio generation)
- Slash commands for guided workflows

**Repository:** https://github.com/digitalsamba/claude-code-video-toolkit

---

## Current Status

**Phase:** 3 - Templates & Brands
**Focus:** Multi-session project system, unified commands

---

## Phases

### Phase 1: Foundation ✅ COMPLETE

- [x] Sprint review template with theme system
- [x] Config-driven video content
- [x] `/video` slash command (unified project creation, replaced `/new-video`)
- [x] Narrator PiP component
- [x] Remotion skill (stable)
- [x] ElevenLabs skill (stable)

### Phase 2: Skills & Automation ✅ COMPLETE

**Skills:**
- [x] FFmpeg skill (beta) - common video/audio conversions
- [x] Playwright recording skill (beta) - browser demo capture
- [x] Review and validate FFmpeg skill
- [x] Review and validate Playwright skill

**Python Tools:**
- [x] `voiceover.py` - CLI for ElevenLabs TTS generation
- [x] `music.py` - CLI for background music generation
- [x] `sfx.py` - CLI for sound effects with presets
- [x] `config.py` - Shared configuration (env vars + registry fallback)

**Commands:**
- [x] `/generate-voiceover` - streamlined audio generation
- [x] `/record-demo` - guided Playwright recording
- [x] `/video-status` - replaced by `/video` with built-in project scanning

**Infrastructure:**
- [x] Playwright recording setup (`playwright/`)
- [x] Centralize voice ID (env var with registry fallback)

### Phase 2.5: Open Source Release ✅ COMPLETE

- [x] Directory restructure for public release
  - `templates/` - video templates
  - `projects/` - user video projects
  - `brands/` - brand profiles
  - `docs/` - documentation
  - `_internal/` - toolkit metadata (renamed from `_toolkit/`)
- [x] Brand profiles system (`brands/default/`)
  - `brand.json` - colors, fonts, typography
  - `voice.json` - ElevenLabs voice settings
  - `assets/` - logos, backgrounds
- [x] Secrets audit and `.gitignore`
- [x] Environment variable support (`ELEVENLABS_VOICE_ID`)
- [x] README, LICENSE (MIT), CONTRIBUTING.md
- [x] Documentation (`docs/getting-started.md`, `creating-brands.md`, `creating-templates.md`)
- [x] GitHub repo: digitalsamba/claude-code-video-toolkit
- [x] Initial commit and push

### Phase 3: Templates & Brands 🔄 IN PROGRESS

**Brand Profiles:**
- [x] Default brand profile
- [x] Digital Samba brand profile (public example)
  - [x] Extract colors from digitalsamba.com
  - [x] Add DS logos to `brands/digital-samba/assets/`
  - [x] Configure voice settings
- [x] `/brand` command (replaced `/new-brand`) - list, edit, or create brands
  - [x] Mine colors/fonts from URL
  - [x] Interactive color picker
  - [x] Logo upload guidance
  - [x] Voice selection

**Templates:**
- [x] Product demo template (extract from digital-samba-skill-demo)
- [x] `/video` command (replaced `/new-video`) - unified project management
- [x] `/template` command - list available templates
- [x] Shared component library (`lib/`) ⭐
  - [x] Theme system (`lib/theme/`) - ThemeProvider, useTheme, types
  - [x] Core components (`lib/components/`) - AnimatedBackground, SlideTransition, Label, Vignette, LogoWatermark, SplitScreen
  - [x] NarratorPiP (needs refinement - different APIs in templates)
  - [x] Templates updated to import from lib
- [ ] Tutorial template
- [ ] Changelog/release notes template

**Template-Brand Integration:**
- [x] Brand loader utility (`lib/brand.ts`)
- [x] Templates use `brand.ts` for theming (generated at project creation)
- [x] `/video` generates brand.ts from selected brand
- [x] project.json tracks brand selection

**Multi-Session Project System:**
- [x] Project schema (`lib/project/types.ts`) - phases, scenes, assets, session history
- [x] Filesystem reconciliation (compare project.json intent vs actual files)
- [x] Auto-generated CLAUDE.md per project
- [x] `/skills` command - list installed skills or create new ones (renamed from `/skill` due to Claude Code conflict)
- [x] Contribution/feedback prompts in all commands and skills

**Contribution & Examples:**
- [x] `/contribute` command - guided contribution workflow
- [x] `examples/` directory for shareable showcase projects
- [x] Contributor recognition with backlinks (CONTRIBUTORS.md)
- [x] Evolution narrative across all commands
- [x] FEEDBACK.md for capturing improvement ideas
- [x] Example: digital-samba-skill-demo (with finished video link)
- [x] Example: sprint-review-cho-oyu (with finished video link)

### Phase 4: Polish & Advanced

**Commands:**
- [x] ~~`/video-status`~~ - replaced by `/video` with built-in scanning
- [x] ~~`/convert-asset`~~ - removed, FFmpeg skill handles conversationally
- [x] ~~`/sync-timing`~~ - removed, timing knowledge in CLAUDE.md

**Output & Accessibility:**
- [ ] Multi-format output (MP4, WebM, GIF, social formats)
- [ ] Subtitle generation from voiceover scripts
- [ ] Thumbnail auto-generation
- [ ] Pre-render validation command

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

**Templates:** 2 (sprint-review, product-demo)
**Brands:** 2 (default, digital-samba)
**Skills:** 4 (2 stable, 2 beta)
**Tools:** 3 (voiceover, music, sfx)
**Commands:** 7 (video, brand, template, skill, contribute, record-demo, generate-voiceover)
**Shared Components:** 7 (AnimatedBackground, SlideTransition, Label, Vignette, LogoWatermark, SplitScreen, NarratorPiP)
**Examples:** 2 with finished video links (digital-samba-skill-demo, sprint-review-cho-oyu)

---

## Next Actions

1. **Upload example videos to YouTube** (optional, for discoverability)
2. **Test `/video` workflow end-to-end**
   - Test new project creation with scene-centric flow
   - Test project resumption (multi-session)
   - Verify filesystem reconciliation
   - Verify CLAUDE.md auto-generation
3. Document narrator video creation workflow (see BACKLOG.md)
4. Create tutorial template
5. Promote ffmpeg and playwright-recording skills to stable after testing
