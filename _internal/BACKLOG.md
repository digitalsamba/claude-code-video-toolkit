# Toolkit Backlog

Ideas and enhancements for claude-code-video-toolkit. Items here are **not yet scheduled** - they're captured for future consideration.

> **Note:** When an item is implemented, remove it from this file and add the change to `CHANGELOG.md`.

---

## Commands

### `/toolkit-status`
Meta command for toolkit development:
- Show current roadmap phase
- List skill maturity status
- Show recent changes
- List backlog items

### `/components`
Browse, preview, and manage reusable animation components:
- **List mode**: Show all components in `lib/components/` with descriptions
- **View mode**: Display component props, usage examples, and preview instructions
- **Categories**: Backgrounds, Overlays, Animations, Layouts, Transitions
- **Preview**: Generate a quick Remotion preview of a specific component
- **Add mode**: Create new component from template or extract from project

**Example usage:**
```
/components              # List all components
/components Envelope     # View Envelope component details
/components --category animations  # List animation components
```

**Component documentation format:**
Each component should have:
- JSDoc with `@example` showing usage
- Props interface with descriptions
- Category tag for filtering

### `/discover-app`
Automated web app exploration for demo planning:
- Crawl all links from a starting URL
- Identify interactive elements (buttons, forms, dropdowns, modals)
- Map navigation flows and page hierarchy
- Detect authentication requirements
- Screenshot each discovered page
- Output site map, suggested recording scripts, and asset manifest

---

## Skills

### Brand Mining Skill
Extract brand identity from websites:
- Screenshot capture
- Dominant color extraction
- Font detection (via CSS inspection)
- Logo detection and download
- Output as draft `brand.json`

### App Discovery Skill
Playwright-based web app exploration and analysis:
- **Crawling**: Discover pages within a domain
- **Element detection**: Find clickable elements, forms, navigation patterns
- **Flow mapping**: Identify common user journeys (login, signup, CRUD)
- **Screenshot capture**: Visual inventory of all discovered pages
- **Auth detection**: Identify login walls and protected routes
- **Output formats**:
  - Mermaid site map / flow diagram
  - JSON structure of pages, elements, and actions
  - Recording script templates for each discovered flow

### Terminal Recording Skill
- Asciinema recording and conversion
- svg-term-cli usage
- Typing effect animations in Remotion

### Video Timing Skill
- Scene duration guidelines
- Voiceover pacing recommendations
- Break tag usage patterns
- Demo playback rate calculations

### Video Accessibility Skill
- Subtitle/caption generation
- Transcript creation
- Color contrast guidelines
- Audio description patterns

### Frontend-Design Skill Rework
The current frontend-design skill is generic web-focused. Rework to be Remotion/video-specific:
- Frame-based animation patterns (not CSS transitions)
- Remotion primitives: `interpolate`, `spring`, `Sequence`, `Series`
- Video-appropriate motion (30fps constraints, render performance)
- Slide component patterns (TitleSlide, StatsSlide, etc.)
- Visual narrative arc across scenes
- Color/typography choices that work at video resolution
- Remove web-specific guidance (responsive, hover states, etc.)

---

## Templates

### Tutorial Template
- Chapter-based structure
- Progress indicator
- Step-by-step sections
- Code highlighting

### Comparison Template
- Before/After split screen
- Feature comparison cards
- Toggle animations

---

## Components

### Still Needed
- CodeHighlight (syntax-highlighted code blocks)
- TypeWriter (animated text typing effect)

### NarratorPiP API Refinement
The NarratorPiP component has two different APIs:
- **sprint-review**: Props-based (`videoFile`, `position`, `size` as direct props)
- **product-demo**: Config-based (`config` object containing all settings)

**Completed this session:**
- [x] Added `objectPosition` prop for video framing control
- [x] Changed default to `objectFit: contain` to show full video

**Still needed:**
- [ ] Unify into a single API
- [ ] Better timing control (startFrame, endFrame)
- [ ] Green screen / background removal support
- [ ] Multiple narrator support
- [ ] Auto-framing based on face detection
- [ ] Aspect ratio prop for container shape

### Narrator Video Creation Guide
Document best practices for creating narrator PiP videos:
- Recording setup (camera, lighting, framing)
- Green screen vs natural background
- Video specifications (resolution, format, duration)
- Syncing with voiceover timing
- Post-processing (cropping, compression)
- Example workflow from raw recording to final asset

---

## Infrastructure

### Multi-Format Output
Render pipeline for (square and vertical cut-downs are done in `tools/cutdown.py`):
- WebM (web fallback)
- GIF (preview/social)

### Python Compatibility
The documentation uses `python` but macOS and some systems require `python3`:
- Detect which command is available at runtime
- Add wrapper script or shell function
- Update CLAUDE.md examples to be cross-platform
- Consider: `#!/usr/bin/env python3` shebangs for direct script execution

### Cost Tracking
ElevenLabs usage monitoring:
- Log character counts per generation
- Track music minutes
- Monthly usage summary

### R2 Storage Monitoring
Cloudflare R2 usage for dewatermark tool:
- `--r2-status` flag: Show bucket size, object count, recent objects
- Cleanup orphaned files from failed jobs
- Usage trends and cost estimation
- Auto-cleanup policies for old objects

### Docker Package Namespace
Move GPU worker images from personal to org namespace:
- Currently: `ghcr.io/conalmullan/video-toolkit-*`
- Target: `ghcr.io/digitalsamba/video-toolkit-*`
- Add OCI labels to Dockerfiles for auto-linking
- Update CLAUDE.md and docs references
- Packages: qwen-edit, realesrgan, propainter (+ animate, wan-i2v if used)

---

## Improvements

### Per-Scene Voiceover Generation ✅ IMPLEMENTED
Tested in `android-screenshare-sprint` (Jan 2026). Now promoted to default workflow.

**Implementation complete:**
- [x] `voiceover.py` supports `--scene-dir` for processing .txt files in a directory
- [x] `voiceover.py` supports `--concat` for ffmpeg concatenation (for SadTalker)
- [x] `audioFile` field added to DemoConfig, info, overview, summary, credits configs
- [x] SprintReview.tsx renders per-scene `<Audio>` within each `Series.Sequence`
- [x] Backward compatibility: global voiceover track still works when no per-scene audio
- [x] `/generate-voiceover` command updated to detect and prefer per-scene mode
- [x] Documentation updated (CLAUDE.md, README.md, getting-started.md)

**Remaining ideas (not blocking):**
- [ ] Store concat list file in project for reproducibility
- [ ] Auto-split VOICEOVER-SCRIPT.md into scene scripts

### Qwen3-TTS Generation Parameters (Partially Done)

`temperature` and `top_p` are now supported end-to-end (handler.py, qwen3_tts.py, voiceover.py). Remaining parameters need a future Docker rebuild.

**Done:**

| Parameter | Effect | Status |
|-----------|--------|--------|
| `temperature` | Expressiveness/randomness (0.4 = consistent, 0.7 = natural, 1.2+ = varied) | **Done** |
| `top_p` | Nucleus sampling cutoff (controls quality/diversity tradeoff) | **Done** |

**Remaining (future Docker rebuild):**

| Parameter | Effect | Priority |
|-----------|--------|----------|
| `max_new_tokens` | Max audio length — needed for long texts | Medium |
| `x_vector_only_mode` | Fast voice cloning — speaker embedding only, skips full prompt encoding. Faster but lower quality. Good for previews. | Medium |
| `do_sample` | Toggle sampling vs greedy | Low |
| `repetition_penalty` | Reduce repetitive audio patterns | Low |

**Reference:** Qwen3-TTS models accept all `model.generate()` kwargs: https://github.com/QwenLM/Qwen3-TTS

### Voice Management
- Support multiple voices per project
- Voice settings presets (narrator, character, etc.)
- Voice preview before generation

### Playwright Enhancements
- Auth state persistence between recordings
- Click ripple effect improvements
- Slow typing simulation
- Scroll smoothing

### Template Improvements
- Additional color themes
- Progress bar component

### Brand System Enhancements
- Brand inheritance (extend another brand)
- Dark/light mode variants per brand
- Brand preview command

---

## SadTalker Refinements (In Progress)

### Completed This Session
- [x] `--retrieve JOB_ID` flag to download results from completed jobs
- [x] Auto-timeout calculation based on audio duration (~4min/min + 2min buffer)
- [x] Fixed `video_url` handling (was looking for `r2_url`)
- [x] Updated docs with job recovery section

### Still Needed
- [ ] **Webhook support** (`--webhook URL`) - fire and forget, get notified when done
- [ ] **Async mode** (`--async`) - submit and exit, retrieve later with job ID
- [ ] **Job ID persistence** - save job ID to local file for crash recovery
- [ ] **Progress streaming** - show chunk progress during long jobs
- [ ] **R2 read permissions** - ensure credentials have Object Read & Write
- [ ] **Image preprocessing** - auto-crop/center face for better framing
- [ ] **Output reframing** - ffmpeg post-process to center face in output

### Integration with NarratorPiP
The current workflow has a framing issue:
1. Source image has face positioned high in frame
2. SadTalker preserves this framing
3. NarratorPiP crops further, cutting off mouth

**Solutions to explore:**
- Pre-process source image to center face before SadTalker
- Post-process SadTalker output to reframe/crop
- Add face detection to auto-adjust NarratorPiP `objectPosition`
- Document image requirements for good PiP framing

### Current sadtalker.py Changes (uncommitted)
```bash
git diff tools/sadtalker.py  # ~150 lines of additions
```
- `get_audio_duration()` - ffprobe helper
- `calculate_timeout()` - auto-timeout logic
- `retrieve_job_result()` - download from job ID
- `--retrieve` CLI flag
- `--timeout` default changed to 0 (auto-calculate)

---

## Documentation

- [ ] Video tutorial: Using the toolkit
- [ ] Skill creation guide
- [ ] Template customization guide
- [ ] Troubleshooting guide
- [ ] Brand mining walkthrough
- [ ] Add `tools/README.md` for Python utilities
- [ ] Add `lib/components/README.md`
