# Toolkit Changelog

All notable changes to claude-code-video-toolkit.

> Releases are automated via GitHub Actions. See `.github/workflows/release.yml`.

---

## 2026-04-09 (v0.14.2)

### Added
- **moviepy skill** (`.claude/skills/moviepy/`) — Python video composition for overlaying deterministic text on AI-generated video (LTX-2, SadTalker). Leads with the "trustworthy text" framing: news, trailers, lower thirds, and social captions all need deterministic overlay because AI models can't guarantee readable text. Covers the PIL workaround for moviepy 2.x's TextClip ascender-clipping bug, audio-anchored timelines, common recipes (labels on LTX-2 clips, lower thirds on SadTalker heads, tinted overlays for contrast, split-screen composites, music + VO mixing), 2.x API gotchas, and a genres-where-this-shines table.
- **`examples/quick-spot`** — runnable 15-second ad-style moviepy example. Audio-anchored timeline, PIL text rendering with cross-platform font loading, optional per-scene VO + ducked music, solid-colour backgrounds. Renders with `python3 build.py` and zero external assets.
- **`examples/data-viz-chart`** — runnable animated time-series chart (real GitHub star history) demonstrating the "matplotlib for data + moviepy for trustworthy text" split that mirrors real news-graphics pipelines. Cache-aware matplotlib step.
- **`examples/hello-world`** — minimal Remotion sprint-review example finally committed. The root README's quick-start callout has been pointing at `cd examples/hello-world && npm install && npm run render` since Feb 23, but the example had zero git history on any branch. Fixes the broken link for anyone cloning the repo.
- **LTX-2 skill: Stylized Character Cameo** — new use case in `.claude/skills/ltx2/SKILL.md` documenting LTX-2 image-to-video as a SadTalker alternative for non-photoreal faces (fantasy characters, heavy beards, masks, helmets) where lip-sync precision matters less than motion + atmosphere.
- **CLAUDE.md: Audio-Anchored Timelines** — new subsection in the Video Timing guide complementing the existing reactive `sync_timing.py` flow with a prevention-first pattern (generate audio first, anchor visuals to absolute timestamps). Especially useful for single-file moviepy `build.py` projects.

### Changed
- **`tools/requirements.txt`** adds `Pillow>=10.0`, `moviepy>=2.0`, `matplotlib>=3.7`. One install command (`python3 -m pip install -r tools/requirements.txt`) now covers every Python feature in the toolkit: AI voiceover, image gen, music gen, and the new moviepy examples.
- **Root README quick-start** switches to `python3 -m pip install` — the [PyPA-recommended invocation](https://pip.pypa.io/en/stable/user_guide/#running-pip) that works identically on macOS, Linux, and Windows and guarantees `pip` and `python3` resolve to the same environment.
- **`docs/sadtalker.md`** adds a "When NOT to Use SadTalker" section pointing at LTX-2 image-to-video for stylized characters, heavy facial hair, masks, and non-frontal angles.
- **`examples/README.md`** splits "Using Examples" into Remotion (`npm install` flow) and moviepy (`python3 build.py` flow) sections and adds both new examples to the listing table.

### Fixed
- **Hardcoded macOS font paths in `examples/quick-spot/build.py` and `examples/data-viz-chart/build.py`** — both examples used `/System/Library/Fonts/Supplemental/Arial Bold.ttf` directly, which would crash with `FileNotFoundError` on Linux and Windows. Replaced with a `platform.system()`-keyed fallback chain that tries OS-appropriate sans-serif fonts (Arial on macOS and Windows, DejaVu Sans on Linux, with multiple candidate paths each) and falls back to `ImageFont.load_default()` if nothing matches, so the examples never crash on unusual setups. Both examples now genuinely run cross-platform as their READMEs promise.
- **Latent Pillow gap in `tools/flux2.py` and `tools/image_edit.py`** — both tools imported PIL inside a `try/except ImportError` guard with a friendly error message, but Pillow was never actually declared in `tools/requirements.txt`. Users had to install Pillow manually after hitting the runtime error. Free-rider fix via the moviepy declaration.
- **Guarded imports** added to both moviepy example `build.py` files — matches the existing `flux2.py` / `image_edit.py` friendly-error pattern. Prevents bare `ModuleNotFoundError` tracebacks for users who run the examples before installing dependencies.

### Docs
- **README examples table** adds `q2-townhall-longarm-ad` (Super Bowl-style launch ad with LTX-2 animated Lugh cameo) and `q2-townhall-stars` (GitHub star history time-lapse) as finished demo entries dated 2026-04-08.
- **moviepy skill** added to the README's skills table and `_internal/toolkit-registry.json`.

---

## 2026-04-04 (v0.14.1)

### Added
- **`tools/chain_video.py`** — Chain LTX-2 clips with visual continuity (each scene uses previous clip's last frame as input). Supports per-scene prompts via JSON, auto-resume of interrupted runs, and structured progress reporting.
- **Structured progress reporting** (`--progress json`) for all cloud GPU tools
- **OpenClaw skill: yieldMs polling** — agent stays in poll loop for live progress instead of going silent with `background:true`
- **OpenClaw skill: style drift prevention** — rules to prevent anime/Asian style drift in chained LTX-2 sequences (per-scene prompts + negative prompts)

### Changed
- **acemusic cloud API** is now default music generation provider (official API, XL Turbo 4B model with 5Hz LM thinking)
- **SadTalker timeouts** increased: 20x multiplier + 300s buffer for size=512+gfpgan on A10G
- Synced official Remotion skills (upstream d5d3955)

### Fixed
- chain_video: glob matching for manually-renamed clips (e.g. `chain-05-brigid.mp4`)
- chain_video: 20-min subprocess timeout, stderr in error messages, JSON validation, temp file cleanup
- chain_video: registered in toolkit-registry.json
- R2 recovery note for SadTalker client timeouts

---

## 2026-03-26 (v0.13.2)

### Changed
- **Default cloud provider switched to Modal** — All 6 cloud GPU tools (flux2, image_edit, music_gen, qwen3_tts, sadtalker, voiceover) now default to `--cloud modal` instead of `--cloud runpod`. RunPod remains available via `--cloud runpod`.

### Fixed
- Remove unused imports in `tools/ltx2.py` (`json`, `requests`) — flagged by code quality bot

---

## 2026-03-25 (v0.13.1)

### Added
- **`tools/ltx2.py`** — AI video generation using LTX-2.3 22B DiT model
  - Text-to-video generation from prompts (~5s clips at up to 1024x1536)
  - Image-to-video — animate still images with motion prompts
  - Joint audio+video generation (synchronized ambient audio)
  - Quality presets: `standard` (30 steps) and `fast` (15 steps)
  - Valid frame counts: 25-193 frames ((n-1)%8==0), 24fps default
  - Modal deployment on A100-80GB with baked weights (~55GB)
  - Estimated cost: ~$0.23 per 5-second clip
- **LTX-2 skill** (`.claude/skills/ltx2/`) — prompting guide, parameters, video production use cases
- **Openclaw skill updated** — LTX-2 added as section 4d (video clips), setup instructions, cost table

### Technical Notes
- `torch.inference_mode()` required around pipeline call — without it, Gemma text encoder retains ~37GB of autograd activations causing OOM (upstream bug: Lightricks/LTX-2#152)
- Weights hosted on `Lightricks/LTX-2.3` HuggingFace repo (separate from the code repo `Lightricks/LTX-2`)
- `transformers` pinned to 4.57.6 (5.x breaks `Gemma3TextConfig.rope_local_base_freq`)
- HuggingFace token required for both LTX-2 (rate limiting) and Gemma 3 (gated model)

---

## 2026-03-22 (v0.12.0)

### Added
- **`tools/music_gen.py`** — AI music generation using ACE-Step 1.5
  - Text-to-music with precise BPM, key, time signature, and duration control
  - Vocal music with lyrics in 50+ languages
  - Cover / style transfer from reference audio (`--cover`)
  - Stem extraction — isolate vocals, drums, bass, etc. (`--extract`)
  - **8 scene presets** for video production: `corporate-bg`, `upbeat-tech`, `ambient`, `dramatic`, `tension`, `hopeful`, `cta`, `lofi`
  - **Brand-aware generation** — `--brand` loads style hints from brand.json
  - `--setup` creates RunPod template + endpoint via GraphQL
  - Docker image: `ghcr.io/conalmullan/video-toolkit-acestep:latest` (CUDA 12.8, baked model weights)
  - MIT licensed model — free alternative to ElevenLabs Music with more control
  - ~2-3s inference on GPU (turbo mode, 8 steps)
- **ACE-Step skill** (`.claude/skills/acestep/`) — prompt engineering patterns, lyrics formatting, scene preset guide, video production integration

---

## 2026-03-22 (v0.11.1)

### Fixed
- **Queue timeout safeguard** — All 6 RunPod tools now auto-cancel jobs stuck in queue for >5 min, preventing runaway billing from GPU unavailability
- **Cancel on exit** — Jobs are cancelled on RunPod when client polling times out (previously left orphaned jobs in queue indefinitely)
- **R2 upload support** — flux2.py and image_edit.py now pass R2 config to handlers (was already supported server-side but never triggered)

### Changed
- **flux2 default GPU** — Changed from ADA_24 (RTX 4090, frequently throttled) to AMPERE_24,ADA_24 (fallback chain)
- **flux2 setup** — `--setup-gpu` now accepts comma-separated GPU types for fallback
- **RunPod skill** — Added API reference (GraphQL queries/mutations, REST endpoints, GPU IDs, R2 CLI patterns)
- **R2 docs** — Added operations guide (listing, cleanup, `--region auto` gotcha) to runpod-setup.md

---

## 2026-03-15 (v0.11.0)

### Added
- **`tools/flux2.py`** — AI image generation and editing using FLUX.2 Klein 4B
  - Text-to-image generation from prompts (~2.5s fast mode, ~8s quality mode)
  - Image editing with reference images (`--input photo.jpg --prompt "..."`)
  - Multi-image compositing (up to 3 reference images)
  - **8 scene presets** for video production: `title-bg`, `problem`, `solution`, `demo-bg`, `stats-bg`, `cta`, `thumbnail`, `portrait-bg`
  - **Brand-aware generation** — `--brand digital-samba` reads brand.json and injects color palette into prompts
  - Preset + prompt layering — preset provides style/mood, `--prompt` adds subject context
  - `--setup` creates RunPod template + endpoint via GraphQL
  - `--list-presets` shows all available scene presets
  - Docker image: `ghcr.io/conalmullan/video-toolkit-flux2:latest` (baked model weights, ~15GB)
  - Apache 2.0 licensed model (commercial OK)
- **"The Space Between"** — [showcase video](https://demos.digitalsamba.com/video/the-space-between.mp4) demonstrating end-to-end AI video creation
  - Avatar generated with flux2, voiced with Qwen3-TTS, animated with SadTalker, composed in Remotion
  - Concept, script, voice, visuals — all AI-generated
  - New video format: video essay (beyond sprint reviews and product demos)

### Changed
- **Baked Docker images rebuilt** — both `flux2` and `qwen3-tts` images now include model weights baked in, eliminating cold-start model downloads (~30s cold start vs minutes)
- Updated `_internal/toolkit-registry.json` with flux2 tool entry, cloud endpoint, scene presets, and showcase example
- Updated README with flux2 tool, Docker image, demo table entry, and Cloud GPU tools table

---

## 2026-02-25 (v0.10.1)

### Added
- **`tools/sync_timing.py`** — Audio-to-config timing sync tool
  - Measures actual per-scene audio durations via ffprobe
  - Auto-detects config file and template type (sprint-review v1/v2, product-demo)
  - 3-pass audio-to-scene matching: `audioFile` field → index → name
  - Comparison table with delta indicators; skips changes < 0.3s
  - `--apply` updates `durationSeconds` in-place (creates `.bak` backup)
  - `--voiceover-json` accepts voiceover.py output directly
  - Suggests `playbackRate` adjustments for demo scenes

### Changed
- **CLAUDE.md slimmed 44%** (861 → 480 lines)
  - Removed catalog data duplicated in `toolkit-registry.json` (skills, commands, components, transitions, presets, Docker images, duplicate CLI examples)
  - Added cross-references to registry for structured data
  - All workflow guidance, timing knowledge, code patterns, and tool-specific gotchas retained
- Integrated `sync_timing.py` into Video Production Workflow (step 7) and TTS drift feedback loop
- Updated toolkit-registry.json with `sync_timing` tool entry

---

## 2026-02-24 (v0.10.0)

### Added
- **Qwen3-TTS integration** — Self-hosted TTS via RunPod as free alternative to ElevenLabs
  - `tools/qwen3_tts.py` standalone CLI with 9 speakers, tone presets, voice cloning
  - `voiceover.py --provider qwen3` for per-scene generation
  - Docker image: `ghcr.io/conalmullan/video-toolkit-qwen3-tts:latest`
  - Temperature/top_p params for expressiveness control
- **`/voice-clone` command** — Record, test, and save a cloned voice to a brand profile
- **`sprint-review-v2` template** — Composable scene-based architecture for sprint reviews
- **`FilmGrain` component** — SVG noise overlay for cinematic film texture
- **`hello-world` example** — Minimal 25s video, zero config, renders in 2 minutes
- **`runpod` skill** — Cloud GPU setup, Docker images, endpoint management, troubleshooting
- **Remotion skills sync update** — 6 new upstream rules (audio-visualization, ffmpeg, light-leaks, subtitles, transparent-videos, voiceover)

### Improved
- **Onboarding & developer experience** — All API keys now optional, videos render with just Node.js
  - `.env.example` reorganized with sections, all keys commented out by default
  - README and getting-started.md restructured with "Try It Now" quick start
  - All Python tools show actionable guidance when API keys are missing (add key, use alternative, or skip)
  - `voiceover.py` catches missing `elevenlabs` pip package with helpful message pointing to Qwen3-TTS

### Changed
- Updated README with prerequisites table, Qwen3-TTS, Docker images, voice-clone command
- Updated CLAUDE.md with FilmGrain component and checkerboard transition
- Updated roadmap metrics and skill status table
- Fixed cho-oyu demo link; added cortina sprint to demos table

---

## 2026-02-19 (v0.9.3)

### Added
- **Official Remotion skills** — Synced 33 rule files from [remotion-dev/skills](https://github.com/remotion-dev/skills) into `.claude/skills/remotion-official/`
- **Weekly sync workflow** — GitHub Actions checks upstream every Monday and opens a PR if files changed
- **Sync documentation** — `docs/remotion-skills-sync.md` explaining the split and sync process

### Changed
- **Remotion skill split** — Custom `remotion` skill trimmed from ~470 to ~160 lines, now covers only toolkit-specific patterns (transitions, components, conventions). Core framework knowledge deferred to `remotion-official`
- Updated CLAUDE.md skills table with both remotion skills

---

## 2026-01-25 (v0.9.2)

### Added
- **Per-scene voiceover generation** - New recommended workflow for audio
  - `voiceover.py --scene-dir` processes all `.txt` files in a directory
  - `voiceover.py --concat` joins scene audio for SadTalker narrator
  - Each scene's `<Audio>` starts at frame 0 within its `Series.Sequence`
  - Regenerate individual scenes without re-doing the whole video
  - Scene durations match audio naturally (no manual offset calculations)

- **Sprint-review template per-scene audio support**
  - `audioFile` field on `DemoConfig`, `info`, `overview`, `summary`
  - SprintReview.tsx renders per-scene `<Audio>` elements
  - Backward compatible: global voiceover track still works

- **Updated `/generate-voiceover` command**
  - Detects `public/audio/scenes/*.txt` and offers per-scene mode
  - Per-scene is now the default when scene scripts exist
  - Concat option for SadTalker integration

### Changed
- Updated documentation (CLAUDE.md, README.md, getting-started.md)

---

## 2025-12-30 (v0.7.0)

### Added
- **`tools/dewatermark.py`** - AI-powered watermark removal using ProPainter
  - Cloud GPU processing via RunPod serverless (~$0.05-0.30/video)
  - Local processing for NVIDIA GPU users (8GB+ VRAM)
  - Auto resize-ratio: 1.0 for <30s, 0.75 for <1min, 0.5 for longer
  - Chunked processing for long videos with overlap stitching
  - Cloudflare R2 for reliable file transfer
  - Automated `--setup` for one-command RunPod configuration
  - Presets: notebooklm, tiktok, sora, stock-br, stock-bl, stock-center

- **`tools/locate_watermark.py`** - Helper tool for finding watermark coordinates
  - Coordinate grid overlay for visual identification
  - Region verification across multiple frames
  - Preset support for common watermarks
  - Requires ImageMagick (`brew install imagemagick`)

- **`tools/notebooklm_brand.py`** - Post-processing for NotebookLM videos
  - Trims NotebookLM visual outro while preserving full audio
  - Adds custom branded outro with logo and URL
  - Handles freeze-frame bridging when audio exceeds video

- **Docker infrastructure** for RunPod serverless (`docker/runpod-propainter/`)
  - Dockerfile, handler.py, README
  - Public image available for quick deployment

- **Documentation**
  - `docs/optional-components.md` - ML component installation guide
  - `docs/runpod-setup.md` - Cloud GPU setup guide

### Changed
- ElevenLabs TTS: Added `eleven_v3` (alpha) model option
- Updated `CLAUDE.md` with dewatermark and locate_watermark documentation
- Updated `.env.example` with RunPod and R2 configuration

---

## 2025-12-28 (v0.6.0)

### Added
- **`redub.py --sync`** - Word-level time remapping for TTS synchronization
  - Uses ElevenLabs `convert_with_timestamps` API for character-level alignment
  - Uses Scribe word-level timestamps from original audio
  - Builds segment mapping (configurable via `--segment-size`, default: 15 words)
  - Applies variable speed per segment via FFmpeg filtergraph
  - Fixes audio drift when TTS voice speaks at different pace than original
  - Solves: TTS often starts fast and ends slow, causing 3-4+ second drift

### Changed
- Updated `CLAUDE.md` with Redub Sync Mode documentation
- TTS duration now measured via ffprobe (more accurate than timestamp data)

---

## 2025-12-28 (v0.5.0)

### Added
- **`tools/addmusic.py`** - Add background music to existing videos
  - Generate music via ElevenLabs or use existing audio file
  - Options: `--music-volume`, `--fade-in`, `--fade-out`
  - Works on any video file without requiring a project structure

### Changed
- Updated `CLAUDE.md` with addmusic documentation

---

## 2025-12-28 (v0.4.0)

### Added
- **`/redub` command** - Redub existing videos with a different voice
  - Guided workflow for voice selection and transcript handling
  - Supports transcript review/editing before TTS generation
  - Works on any video file without requiring a project structure

- **`tools/redub.py`** - Voice replacement utility tool
  - Extracts audio from video (FFmpeg)
  - Transcribes audio to text (ElevenLabs Scribe STT)
  - Generates new audio with target voice (ElevenLabs TTS)
  - Replaces audio track in video (FFmpeg)
  - Options: `--save-transcript`, `--transcript`, `--keep-temp`, `--speed`
  - Warns about duration mismatches between original and new audio

- **Utility tools concept** - Documented distinction between project tools and utility tools
  - Project tools (voiceover, music, sfx): Used during video creation workflow
  - Utility tools (redub): Quick transformations on existing videos, no project needed

### Changed
- Updated `CLAUDE.md` with utility tools documentation
- Updated Python Tools section to include redub

---

## 2025-12-20

### Added
- **GitHub Actions release automation** (`.github/workflows/release.yml`)
  - Tag-triggered releases (`v*` tags)
  - Manual dispatch option with version input
  - Auto-generates changelog from commits since last tag
  - Reads version from `toolkit-registry.json`

- **`/versions` command** - Check dependency versions and toolkit updates
  - Detects Remotion package version mismatches in projects
  - Compares local toolkit version against GitHub releases
  - Offers to fix version mismatches by pinning and reinstalling
  - Documents common issues and prevention strategies

### Fixed
- **Remotion version mismatch** in digital-samba-free-intro project
  - Pinned all Remotion packages to 4.0.387 (was mixing 4.0.383 and 4.0.387)
  - Removed `^` prefix to prevent future drift

---

## 2025-12-10

### Added
- **`/scene-review` command** - Dedicated scene-by-scene review with Remotion Studio
  - Starts Remotion Studio for visual verification
  - Walks through scenes one by one (not summary tables)
  - Generic - works with any template's config
  - `/video` now delegates to `/scene-review` when phase is `review`
  - `/generate-voiceover` warns if review not complete
  - Fixes: Review kept getting skipped because `/video` command was too long

### Changed
- **Consolidated tracking files** - Simplified from 4 files to 3:
  - `ROADMAP.md` - What we're building (removed duplicate "Next Actions" and "In Progress" sections)
  - `BACKLOG.md` - What we might build (removed all implemented items, now only future ideas)
  - `CHANGELOG.md` - What we built (historical record)
  - Removed `FEEDBACK.md` - Evolution principles moved to `docs/contributing.md`
- Created `docs/contributing.md` with evolution principles and contribution workflow

### Fixed
- **Slash commands not loading** - Renamed `/skill` to `/skills` to avoid conflict with built-in `Skill` tool. The naming collision was silently preventing ALL custom commands from loading. Bug reported to Anthropic.

### Removed
- **`/review` command** - Clashed with Claude Code's built-in PR review command. Replaced by `/scene-review`.

### Added
- **Animation components** (`lib/components/`)
  - Envelope - 3D envelope with opening flap animation, configurable message
  - PointingHand - Animated hand emoji with directional slide-in and pulse effect
- **`/contribute` command** - Guided contribution workflow
  - Report issues via `gh issue create`
  - Submit PRs for improvements
  - Share skills and templates
  - Safety checks to exclude private project work
- **Evolution narrative** across all commands
  - Consistent "## Evolution" section in each command
  - Local improvement workflow
  - Remote contribution links (GitHub issues + PRs)
  - Command history tracking
- **Template evolution guidance** in `/template` command
  - How to add features to existing templates
  - Template maturity indicators
  - Pattern extraction to shared lib/
- **Product demo template README** (`templates/product-demo/README.md`)
  - Quick start guide, configuration, project structure
  - All 7 scene types documented with examples
  - Narrator PiP and demo chrome options

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
- **Documentation updates**
  - `docs/getting-started.md` - Updated for `/video` command, added multi-session workflow
  - `docs/creating-brands.md` - Updated for `/brand` command integration
- **Renamed `skills-registry.json` → `toolkit-registry.json`**
  - Consistent format across all entries (path, description, status, created, updated)
  - Added `components` section for shared lib/components
  - Synced with actual commands (7), templates (2), components (9)

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

## 2025-12-09

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

## 2025-12-09 (earlier)

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

## 2025-12-08

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

## 2025-12-04

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
