# Toolkit Backlog

Ideas and enhancements for claude-code-video-toolkit. Items here are not yet scheduled - they're captured for future consideration.

---

## Commands

### `/new-brand`
Guided brand profile creation:
- Ask for brand name
- Option to mine colors/fonts from a URL (Playwright screenshot + color extraction)
- Interactive color picker for manual entry
- Logo upload guidance (where to place files)
- Voice selection (list available ElevenLabs voices or use existing clone)
- Generate `brand.json`, `voice.json`, create `assets/` folder

### `/convert-asset`
FFmpeg helper command:
- GIF → MP4
- Resize video
- Compress for web
- Extract audio
- Trim video

### `/sync-timing`
Timing calculator:
- Analyze voiceover duration
- Calculate demo segment timings
- Suggest playbackRate adjustments
- Update sprint-config.ts automatically

### `/video-status`
Project dashboard:
- List required vs existing assets
- Show total duration calculation
- Timeline visualization
- Missing asset warnings

### `/toolkit-status`
Meta command for toolkit development:
- Show current roadmap phase
- List skill maturity status
- Show recent changes
- List backlog items

### `/discover-app`
Automated web app exploration for demo planning:
- Crawl all links from a starting URL
- Identify interactive elements (buttons, forms, dropdowns, modals)
- Map navigation flows and page hierarchy
- Detect authentication requirements
- Screenshot each discovered page
- Output site map, suggested recording scripts, and asset manifest

### `/new-skill`
Guided skill creation for extending the toolkit:
- Ask for skill name and description
- Ask for trigger phrases (when should Claude use this skill?)
- Create `SKILL.md` template with proper frontmatter
- Optionally create `reference.md` for detailed docs
- Register in `_internal/skills-registry.json`
- Provide guidance on skill structure and best practices

**Considerations:**
- Skills live in `.claude/skills/` which is a Claude Code convention
- Users may want video-related skills (new tools, services, techniques)
- Or non-video skills that complement their workflow
- Need clear documentation on SKILL.md format and frontmatter

### `/new-template`
Guided template creation:
- Ask for template name and video type
- Choose starting point (blank, copy existing, from project)
- Set up directory structure with config files
- Create placeholder components
- Register in `_internal/skills-registry.json`

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

### Product Demo Template
Extract from digital-samba-skill-demo:
- Dark theme
- Problem → Solution → Demo → CTA flow
- Code snippet components
- Stats cards

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

## Infrastructure

### Shared Component Library ⭐ HIGH PRIORITY
Extract common components to `components/` at workspace level:
- AnimatedBackground (floating shapes, grid lines, gradient overlays)
- SlideTransition (fade, slide, zoom transitions)
- Label (floating label badges with JIRA refs)
- NarratorPiP (picture-in-picture presenter overlay)
- SplitScreen (side-by-side video comparison)
- Vignette (cinematic edge darkening)
- LogoWatermark (corner logo branding)
- CodeHighlight (syntax-highlighted code blocks)

**Why this matters:** These "power tools" enable users to create professional videos quickly. Currently duplicated across templates - should be a single import.

### Narrator Video Creation Guide ⭐ NEEDS REVIEW
Document best practices for creating narrator PiP videos:
- Recording setup (camera, lighting, framing)
- Green screen vs natural background
- Video specifications (resolution, format, duration)
- Syncing with voiceover timing
- Post-processing (cropping, compression)
- Example workflow from raw recording to final asset

**Why this matters:** The narrator PiP is a powerful feature but users need guidance on creating the source video.

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

### Brand Loader Utility
TypeScript utility for templates to load brands:
- `loadBrand('digital-samba')` returns typed brand config
- Merge with template defaults
- Type-safe theme integration

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
- More transition styles
- Additional color themes
- Logo watermark component
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
