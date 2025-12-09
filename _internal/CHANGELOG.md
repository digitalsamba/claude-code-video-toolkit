# Toolkit Changelog

All notable changes to claude-code-video-toolkit.

## [0.3.0] - 2024-12-09

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
- CLAUDE.md updated for new structure

### Fixed
- Removed hardcoded voice ID from committed files
- Proper `.gitignore` for secrets and build artifacts

---

## [Unreleased]

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
- `/generate-voiceover` command - guided ElevenLabs TTS generation
- `/record-demo` command - guided Playwright browser recording
- Interactive recording stop controls (Escape key, Stop button)
- Window scaling for laptop screens (`--scale` option, default 0.75)
- FFmpeg skill (beta) - common video/audio conversion commands
- Playwright recording skill (beta) - browser demo capture
- Playwright infrastructure (`playwright/`) with recording scripts
- Python tools: `voiceover.py`, `music.py`, `sfx.py`
- Skills registry for centralized config

### Changed
- Playwright recordings output at 30fps (matches Remotion)
- Updated CLAUDE.md with new templates and commands

### Fixed
- FFmpeg trim command syntax (use `-to` not `-t` for end time)
- Playwright double navigation issue
- Recording frame rate mismatch (was 25fps, now 30fps)

### Planned (see BACKLOG.md)
- Shared component library (AnimatedBackground, SplitScreen, NarratorPiP, etc.)
- Narrator video creation guide

---

## [0.2.0] - 2024-12-09

### Added
- Sprint review template (`templates/sprint-review/`)
  - Theme system with colors, fonts, spacing
  - Config-driven content via `sprint-config.ts`
  - Slide components: Title, Overview, Summary, EndCredits
  - Demo components: DemoSection, SplitScreen
  - NarratorPiP component for picture-in-picture narrator
  - Audio integration (voiceover, background music, SFX)
- `/new-sprint-video` slash command for guided project creation

---

## [0.1.0] - 2024-12-04

### Added
- Initial workspace setup
- Remotion skill documentation
- ElevenLabs skill documentation
- First video project: sprint-review-cho-oyu
- Voice cloning workflow with ElevenLabs
