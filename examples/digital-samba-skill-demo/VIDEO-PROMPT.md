# Video Generation Prompt

Use this prompt in the `~/work/video` workspace to generate the Remotion video composition.

---

## Prompt

Create a Remotion video composition for the Digital Samba Skill demo. This is a marketing video showcasing how the Digital Samba Claude Code skill enables rapid video conferencing app development.

### Project Location
`~/work/video/digital-samba-skill-demo/remotion/`

### Available Assets

**Demo Videos** (`public/demos/`):
- `skill-install.mp4` - GIF conversion showing Claude Code skill installation (382KB)
- `app-walkthrough.mp4` - Playwright recording of full interview room flow (892KB, ~31s)

**Images** (`public/images/`):
- `ds-logo.png` - Digital Samba logo
- `embedded-app-icon.png` - SDK app icon

**Asciinema Casts** (`assets/` - need conversion or player embed):
- `build-app.cast` - Terminal recording of Claude building the app
- `skill-install.cast` - Terminal recording of skill installation

### Branding
- **Digital Samba Blue**: `#0066FF`
- **Accent Teal**: `#00D4AA`
- **Background**: Dark (`#0a0a0a` or `#111`)
- **Style**: Modern tech, terminal aesthetics, clean animations

### Video Structure (~2:00-2:30)

1. **Title Slide** (5s) - DS logo + "Build Video Conferencing in Minutes" + "Claude Code + Digital Samba"

2. **Problem Statement** (8s) - "Video conferencing integration is complex..." with visual of API complexity

3. **Solution Intro** (7s) - "What if AI understood your video API?" - terminal aesthetic

4. **Skill Installation** (15s) - Play `skill-install.mp4`, voiceover explains one-command install

5. **Building the App** (25s) - Show Claude building the app. Key message: "We gave Claude creative freedom to interpret what an online interview should look like"

6. **App Walkthrough** (35s) - Play `app-walkthrough.mp4` showing:
   - Home page with dual cards (Interviewer/Candidate)
   - Create room form
   - Room code display
   - Candidate joining
   - Live video room

7. **Summary Stats** (7s) - Animated cards: "Minutes to working app", "Zero infrastructure", "Full SDK control"

8. **CTA** (8s) - "Get the skill: github.com/digitalsamba/digital-samba-skill" + DS logo

### Voiceover Script
See `VOICEOVER-SCRIPT.md` for full timed narration. Key addition:
> "We gave Claude creative freedom to interpret what an online interview experience should look like."

### Technical Requirements
- Resolution: 1920x1080
- FPS: 30
- Duration: ~4500 frames (2:30) - adjust based on asset timing
- Codec: h264

### Component Suggestions
- Use `OffthreadVideo` for demo clips
- Use `Series.Sequence` for timeline
- Add subtle animated background (floating shapes)
- Use `interpolate` for all animations
- Add `SlideTransition` component for scene changes

### Output
Render to: `out/digital-samba-skill-demo.mp4`

---

## Quick Start

```bash
cd ~/work/video/digital-samba-skill-demo/remotion
npx create-video@latest .  # If scaffolding needed
npm install
npm run studio  # Preview
npm run render  # Final output
```
