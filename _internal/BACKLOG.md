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

---

## Templates

### Tutorial Template
- Chapter-based structure
- Progress indicator
- Step-by-step sections
- Code highlighting

### Changelog Template
- Version header
- Feature list with icons
- Breaking changes section
- Compact format

### Comparison Template
- Before/After split screen
- Feature comparison cards
- Toggle animations

---

## Components

### Still Needed
- CodeHighlight (syntax-highlighted code blocks)
- ClickRipple (mouse click effect for demos)
- TypeWriter (animated text typing effect)

### NarratorPiP API Refinement
The NarratorPiP component has two different APIs:
- **sprint-review**: Props-based (`videoFile`, `position`, `size` as direct props)
- **product-demo**: Config-based (`config` object containing all settings)

Need to unify into a single API. Consider:
- Simpler props-based API for most use cases
- Optional config object for complex scenarios
- Better timing control (startFrame, endFrame)
- Green screen / background removal support
- Multiple narrator support

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

### Asset Validation Script
Pre-render check:
- All referenced videos exist
- All audio files exist
- Duration matches config
- TypeScript compiles

### Multi-Format Output
Render pipeline for:
- MP4 (primary)
- WebM (web fallback)
- GIF (preview/social)
- Square format (social)
- Vertical format (mobile/stories)

### Cost Tracking
ElevenLabs usage monitoring:
- Log character counts per generation
- Track music minutes
- Monthly usage summary

---

## Improvements

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

### Transitions Library (In Progress - experiment/transitions branch)
**Status:** Built but needs testing before merge

New `lib/transitions/` with 6 custom transitions + official Remotion transitions:
- `glitch()` - Digital distortion, slice displacement, RGB separation
- `rgbSplit()` - Chromatic aberration
- `zoomBlur()` - Radial motion blur with scale
- `lightLeak()` - Cinematic lens flare
- `clockWipe()` - Radial sweep
- `pixelate()` - Digital mosaic

**TODO before merge:**
- [ ] Test all transitions in gallery (`showcase/transitions/`)
- [ ] Fix any broken transitions (glitch was fixed but needs verification)
- [ ] Copy fixed presentations back to `lib/transitions/presentations/`
- [ ] Test integration with templates
- [ ] Update lib/transitions/TransitionGallery.tsx to match showcase version

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
