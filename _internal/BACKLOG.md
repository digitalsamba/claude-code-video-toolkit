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

---

## Templates

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

## Integrations

### An editable timeline out
Today a finished video is a flat MP4. An editor who wants to move a cut or swap a shot has nothing to open.
- Export the project as a timeline (OpenTimelineIO, which DaVinci Resolve reads, with adapters for Premiere and
  Final Cut): one clip per scene, narration, music and sound effects on their own tracks, captions as a track
- Scenes drawn in Remotion go out as rendered clips; generated clips and stills go out as their source files
- Start from what `timeline.json` and the audio-anchored builds already know
- Open question: whether it pays for itself without an editor among our users asking for it

### ComfyUI as a backend
The image and video tools run our own images on Modal or RunPod. ComfyUI is where most people already run open
models on their own GPU, and it takes a workflow as JSON over a local API.
- A provider beside `--cloud modal|runpod`: send a workflow to a running ComfyUI and collect the result
- Would let a user bring any model ComfyUI supports without us building an image for it
- Optional, like Blender: nothing depends on it, and a `doctor` says whether one is reachable
- Open question: which workflows to ship and keep working, given how fast its nodes change

---

## Infrastructure

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
| `max_new_tokens` | Max audio length, needed for long texts | Medium |
| `x_vector_only_mode` | Fast voice cloning: speaker embedding only, skips full prompt encoding. Faster but lower quality. Good for previews. | Medium |
| `do_sample` | Toggle sampling vs greedy | Low |
| `repetition_penalty` | Reduce repetitive audio patterns | Low |

**Reference:** Qwen3-TTS models accept all `model.generate()` kwargs: https://github.com/QwenLM/Qwen3-TTS

### Voice Management
- Voice settings presets (narrator, character, etc.)
- Voice preview before generation

### Playwright Enhancements
- Auth state persistence between recordings
- Click ripple effect improvements
- Slow typing simulation
- Scroll smoothing

### Voiceover Script
- Auto-split VOICEOVER-SCRIPT.md into per-scene scripts

### Template Improvements
- Additional color themes
- Progress bar component

### Brand System Enhancements
- Brand inheritance (extend another brand)
- Dark/light mode variants per brand
- Brand preview command

---

## Documentation

- [ ] Video tutorial: Using the toolkit
- [ ] Skill creation guide
- [ ] Template customization guide
- [ ] Troubleshooting guide
- [ ] Brand mining walkthrough
- [ ] Add `tools/README.md` for Python utilities
- [ ] Add `lib/components/README.md`
