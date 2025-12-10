# Toolkit Changelog

All notable changes to claude-code-video-toolkit.

> **Status:** Work in progress - no official releases yet.

---

## 2025-12-10

### Fixed
- **Slash commands not loading** - Renamed `/skill` to `/skills` to avoid conflict with built-in `Skill` tool. The naming collision was silently preventing ALL custom commands from loading. Bug reported to Anthropic.

### Added
- **`/contribute` command** - Guided contribution workflow
  - Report issues via `gh issue create`
  - Submit PRs for improvements
  - Share skills and templates
  - Safety checks to exclude private project work
- **Evolution narrative** across all commands
  - Consistent "## Evolution" section in each command
  - Local improvement workflow (say "improve this" → FEEDBACK.md → CHANGELOG.md)
  - Remote contribution links (GitHub issues + PRs)
  - Command history tracking
- **FEEDBACK.md** (`_internal/`) - Central feedback capture and evolution principles
- **Template evolution guidance** in `/template` command
  - How to add features to existing templates
  - Template maturity indicators
  - Pattern extraction to shared lib/

### Changed
- **`projects/` now fully gitignored** - User video work stays completely private
  - Only `.gitkeep` is tracked to preserve directory structure
  - Safe to contribute without exposing project content
- **New `examples/` directory** for shareable showcase projects
  - Configs, scripts, and docs are tracked
  - Large media files (mp4, mp3) are gitignored
  - Each example includes `ASSETS-NEEDED.md` documenting required media
- **`/contribute` now supports example projects** (Option 5)
  - Guides copying from projects/ to examples/
  - Auto-generates ASSETS-NEEDED.md
  - Creates README with quick start instructions
  - **Contributor recognition** with backlinks to website/org
- **CONTRIBUTORS.md** - Recognition for organizations and individuals who share examples

---

## 2025-12-09

### Added
- **Multi-session project system** (`lib/project/`)
  - `types.ts` - TypeScript schema for project.json
  - `README.md` - Documentation for project lifecycle and phases
  - Projects now track: phase, scenes, assets, audio, session history
  - Filesystem reconciliation (compares intent vs reality)
  - Auto-generated CLAUDE.md per project for instant context

- **Unified commands** - Context-aware entry points that list existing items or create new:
  - `/video` - Replaces `/new-video`. Scans projects, offers resume or new
  - `/brand` - Replaces `/new-brand`. Lists brands, edit or create new
  - `/template` - Lists templates or creates new ones (copy, minimal, from project)
  - `/skill` - Lists installed skills or creates new ones

### Changed
- **Command pattern unified** - All domain commands now scan first, then offer actions
- Commands integrate with project.json for state tracking
- README.md updated with new commands and multi-session workflow

### Removed
- `/new-video` - Replaced by `/video`
- `/new-brand` - Replaced by `/brand`

### Notes
- After creating/modifying commands or skills, restart Claude Code to load changes

---

## 2024-12-09

### Added
- **Shared component library** (`lib/`)
  - `lib/theme/` - ThemeProvider, useTheme, createStyles, type definitions
  - `lib/components/` - Reusable video components:
    - AnimatedBackground (supports subtle, tech, warm, dark variants)
    - SlideTransition (fade, zoom, slide-up, blur-fade)
    - Label (floating badge with optional JIRA ref)
    - Vignette (cinematic edge darkening)
    - LogoWatermark (corner branding)
    - SplitScreen (side-by-side video layout)
    - NarratorPiP (picture-in-picture presenter - needs refinement)
  - `lib/index.ts` - Unified exports for templates

### Changed
- **Templates now import from shared library**
  - sprint-review: Uses lib components via re-exports
  - product-demo: Uses lib components (keeps local NarratorPiP for different API)
  - Both templates use shared ThemeProvider from lib/theme
  - Deleted duplicate component files from templates
- Updated ROADMAP.md - Shared component library marked complete
- Updated BACKLOG.md - Added NarratorPiP API refinement task

---

## 2024-12-09 (earlier)

### Added
- **Product demo template** (`templates/product-demo/`)
  - Scene-based composition (title, problem, solution, demo, stats, CTA)
  - Config-driven content via `demo-config.ts`
  - Dark tech aesthetic with animated background
  - Narrator PiP (picture-in-picture presenter)
  - Browser/terminal chrome for demo videos
  - Stats cards with spring animations
- **`/new-brand` command** - guided brand profile creation
  - Extract colors from website URL
  - Manual color entry with palette generation
  - Logo and voice configuration guidance
- **Digital Samba brand profile** (`brands/digital-samba/`)
- **Template-brand integration**
  - Brand loader utility (`lib/brand.ts`)
  - `brand.ts` in each template (generated at project creation)
  - `project.json` for tracking project metadata
  - Brand generator script (`lib/generate-brand-ts.ts`)
- **`/new-video` command** - unified project creation wizard
  - Choose template (sprint-review, product-demo)
  - Choose brand from available brands
  - **Scene-centric workflow:**
    - Content gathering (URLs, notes, paste)
    - Claude proposes scene breakdown
    - Interactive scene refinement
    - Scene types: title, overview, demo, split-demo, stats, credits, problem, solution, feature, cta
    - Visual types: `[DEMO]`, `[SCREENSHOT]`, `[EXTERNAL VIDEO]`, `[SLIDE]`
  - Generates VOICEOVER-SCRIPT.md with narration and asset checklist
  - Creates project.json with scene tracking
  - Guides through asset creation phase
  - Ongoing project support (resume any project)

### Changed
- **Replaced `/new-sprint-video` with `/new-video`** - single entry point for all templates
- Templates now load theme from `brand.ts` instead of hardcoded values
- Updated CLAUDE.md with new workflow and commands
- Updated ROADMAP.md - Phase 3 template-brand integration complete
- **Added Video Timing section to CLAUDE.md** - pacing rules, scene durations, timing calculations
- **Removed `/convert-asset` from backlog** - FFmpeg skill handles this conversationally
- **Removed `/sync-timing` from backlog** - timing knowledge now in CLAUDE.md

---

## 2024-12-08

### Added
- **Open source release** - Published to GitHub at digitalsamba/claude-code-video-toolkit
- **Brand profiles system** (`brands/`)
  - `brand.json` for colors, fonts, typography
  - `voice.json` for ElevenLabs voice settings
  - `assets/` for logos and backgrounds
  - Default brand profile included
- **Documentation** (`docs/`)
  - `getting-started.md` - First video walkthrough
  - `creating-brands.md` - Brand profile guide
  - `creating-templates.md` - Template creation guide
- **Environment variable support**
  - `ELEVENLABS_VOICE_ID` - Set voice ID via env var
  - Falls back to `_internal/skills-registry.json` if not set
- `/generate-voiceover` command - guided ElevenLabs TTS generation
- `/record-demo` command - guided Playwright browser recording
- Interactive recording stop controls (Escape key, Stop button)
- Window scaling for laptop screens (`--scale` option, default 0.75)
- FFmpeg skill (beta) - common video/audio conversion commands
- Playwright recording skill (beta) - browser demo capture
- Playwright infrastructure (`playwright/`) with recording scripts
- Python tools: `voiceover.py`, `music.py`, `sfx.py`
- Skills registry for centralized config
- README.md, LICENSE (MIT), CONTRIBUTING.md
- `.env.example` template

### Changed
- **Directory restructure for open source:**
  - `templates/` - Video templates (moved from root)
  - `projects/` - User video projects (moved from root)
  - `brands/` - Brand profiles (new)
  - `assets/` - Shared assets (consolidated)
  - `_internal/` - Toolkit metadata (renamed from `_toolkit/`)
- Updated `/new-sprint-video` command paths
- `tools/config.py` reads from `_internal/` and supports env vars
- Playwright recordings output at 30fps (matches Remotion)

### Fixed
- Removed hardcoded voice ID from committed files
- Proper `.gitignore` for secrets and build artifacts
- FFmpeg trim command syntax (use `-to` not `-t` for end time)
- Playwright double navigation issue
- Recording frame rate mismatch (was 25fps, now 30fps)

---

## 2024-12-04

### Added
- Sprint review template (`templates/sprint-review/`)
  - Theme system with colors, fonts, spacing
  - Config-driven content via `sprint-config.ts`
  - Slide components: Title, Overview, Summary, EndCredits
  - Demo components: DemoSection, SplitScreen
  - NarratorPiP component for picture-in-picture narrator
  - Audio integration (voiceover, background music, SFX)
- `/new-sprint-video` slash command for guided project creation
- Initial workspace setup
- Remotion skill documentation
- ElevenLabs skill documentation
- First video project: sprint-review-cho-oyu
- Voice cloning workflow with ElevenLabs
