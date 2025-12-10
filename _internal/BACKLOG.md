# Toolkit Backlog

Ideas and enhancements for claude-code-video-toolkit. Items here are not yet scheduled - they're captured for future consideration.

## Workflow: Marking Items Complete

When implementing a backlog item:

1. **During implementation** - Add note to CHANGELOG.md
2. **After completion** - Mark item in this file with `✅ IMPLEMENTED`
3. **If merged/removed** - Mark with `✅ MERGED INTO /command` or `❌ REMOVED` with reason
4. **Update ROADMAP.md** - If the item was on the roadmap, mark it there too

**Quick check command:**
```bash
# Find implemented items that might be unmarked
grep -r "IMPLEMENTED" _internal/BACKLOG.md | head -5
```

---

## Commands

### `/brand` ✅ IMPLEMENTED (was `/new-brand`)
Unified brand command - lists existing brands or creates new ones.

### `/video` ✅ IMPLEMENTED (was `/new-video`)
Unified video command with multi-session support:
- Scans for existing projects
- Resume or create new
- Scene-centric workflow with asset markers
- project.json tracking with phases
- Auto-generated CLAUDE.md per project

### `/template` ✅ IMPLEMENTED
Lists available templates with details.

### `/skill` ✅ IMPLEMENTED (was `/new-skill`)
Lists installed skills or creates new ones.

### `/video-status` ✅ MERGED INTO `/video`
Now built into `/video` command - scans projects on invocation.

### `/convert-asset` ❌ REMOVED
Not needed - FFmpeg skill handles this conversationally.

### `/sync-timing` ❌ REMOVED
Not needed as a command - timing knowledge in CLAUDE.md.

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

### `/new-template` ✅ MERGED INTO `/template`
Now part of `/template` command:
- List existing templates or create new
- Choose starting point (copy existing, minimal, from project)
- Define scene types
- Set up directory structure

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

### Product Demo Template ✅ IMPLEMENTED
Extracted to `templates/product-demo/`:
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

### Shared Component Library ✅ IMPLEMENTED
Extracted to `lib/components/`:
- AnimatedBackground (floating shapes, grid lines, gradient overlays)
- SlideTransition (fade, slide, zoom transitions)
- Label (floating label badges with JIRA refs)
- NarratorPiP (picture-in-picture presenter overlay) - **needs API refinement**
- SplitScreen (side-by-side video comparison)
- Vignette (cinematic edge darkening)
- LogoWatermark (corner logo branding)

**Still needed:**
- CodeHighlight (syntax-highlighted code blocks)

### NarratorPiP API Refinement ⭐ NEEDS WORK
The NarratorPiP component has two different APIs:
- **sprint-review**: Props-based (`videoFile`, `position`, `size` as direct props)
- **product-demo**: Config-based (`config` object containing all settings)

Need to unify into a single API. Consider:
- Simpler props-based API for most use cases
- Optional config object for complex scenarios
- Better timing control (startFrame, endFrame)
- Green screen / background removal support
- Multiple narrator support

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

### Brand Loader Utility ✅ IMPLEMENTED
Implemented in `lib/brand.ts` and `lib/generate-brand-ts.ts`:
- `loadBrand('digital-samba')` returns typed brand config
- Merge with template defaults
- Type-safe theme integration
- `/video` generates `brand.ts` from selected brand at project creation

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
