# Example: Digital Samba Skill Demo

Marketing video demonstrating how the Digital Samba Claude Code skill accelerates video conferencing app development.

> Contributed by [Digital Samba](https://digitalsamba.com)

## Watch the Finished Video

**[Download MP4](https://demos.digitalsamba.com/video/digital-samba-skill-demo.mp4)**

*See the finished result before you start!*

## What This Demonstrates

- **Product demo template** - Dark tech aesthetic with animated backgrounds
- **Problem → Solution narrative** - Setting up the pain point before the demo
- **GIF to video conversion** - Using FFmpeg to prepare assets
- **Browser walkthrough** - Playwright-recorded app demonstration
- **Stats cards** - Spring-animated summary slide
- **CTA ending** - Call-to-action with links

## Video Overview

**Duration:** ~2:30
**Template:** product-demo style
**Audience:** Developers evaluating Digital Samba, Claude Code users

### Scenes

1. Title (8s) - Dual logos + headline
2. Problem Statement (10s) - "Video conferencing is complex..."
3. Solution Intro (7s) - "What if AI understood your video API?"
4. Skill Installation (20s) - GIF demo of `claude mcp add`
5. Building the App (30s) - Terminal demo
6. App Walkthrough (35s) - Browser recording of working app
7. Live Video Call (15s) - Demo of actual video conferencing
8. Code Highlight (10s) - Key generated code snippets
9. Summary Stats (7s) - "5 minutes to working app"
10. CTA + Credits (8s) - GitHub link, website

## Quick Start

```bash
# Copy to your projects
cp -r examples/digital-samba-skill-demo projects/my-version
cd projects/my-version

# Create required assets (see ASSETS-NEEDED.md)
# ...

# Preview
cd remotion
npm install
npm run studio

# Render
npm run render
```

## Project Structure

```
digital-samba-skill-demo/
├── README.md              # This file
├── ASSETS-NEEDED.md       # What media to create
├── VIDEO-SPEC.md          # Full video specification
├── VOICEOVER-SCRIPT.md    # Narration script
├── CLAUDE.md              # AI context
├── assets/
│   ├── claude-code-install.gif     # Source for conversion
│   └── claude-desktop-install.gif  # Alternative
└── remotion/
    ├── src/
    │   └── config/        # Video configuration
    └── public/
        ├── images/        # Logos (included)
        ├── demos/         # Add your recordings here
        └── audio/         # Add voiceover/music here
```

## Assets

Large media files are not included. See `ASSETS-NEEDED.md` for:
- Demo video recordings
- Voiceover generation
- Background music

## Branding

This example uses Digital Samba branding:
- Primary: `#0066FF` (blue)
- Accent: `#00D4AA` (teal)
- Dark backgrounds with terminal aesthetics

---

*Contributed by [Digital Samba](https://digitalsamba.com) - European video conferencing API*
