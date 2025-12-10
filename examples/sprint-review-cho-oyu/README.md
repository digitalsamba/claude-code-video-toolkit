# Example: Sprint Review Cho-Oyu

Internal sprint review video for Digital Samba Mobile iOS v4.0.2, demonstrating bug fixes and improvements.

> Contributed by [Digital Samba](https://digitalsamba.com)

## Watch the Finished Video

**[Download MP4](https://demos.digitalsamba.com/video/sprint-review.mp4)**

*See the finished result before you start!*

## What This Demonstrates

- **Sprint review template** - Config-driven demo showcase
- **Split-screen demos** - Phone + browser side-by-side
- **JIRA integration** - Reference labels on demo clips
- **Multi-demo structure** - 6 separate demo recordings
- **Professional narration** - AI-generated voiceover
- **Sound effects** - Whoosh, chime, click transitions

## Video Overview

**Duration:** ~3-5 minutes
**Template:** sprint-review
**Audience:** Internal dev team and stakeholders

### Content

**What's New in iOS v4.0.2:**
- App renamed to "Digital Samba"
- 5 screenshare stability fixes
- Sleep/lock connection persistence
- Deep link warm start fix

### Demo Sections

1. **Title** (5s) - Sprint name, version, date
2. **Overview** (15s) - Bullet point summary
3. **App Rename** (10s) - New branding demo
4. **Sleep/Lock Fix** (30s) - Connection persistence [SAMBA-8059]
5. **Tile Timing** (30s) - Split-screen phone + browser [SAMBA-8053, 8057]
6. **Button State** (30s) - Reconnection sync [SAMBA-8054]
7. **Recording Indicator** (20s) - Cleanup on app close [SAMBA-8056]
8. **Deep Links** (20s) - Warm start handling [SAMBA-8112]
9. **Summary** (10s) - Stats and availability

## Quick Start

```bash
# Copy to your projects
cp -r examples/sprint-review-cho-oyu projects/my-version
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
sprint-review-cho-oyu/
├── README.md              # This file
├── ASSETS-NEEDED.md       # What media to create
├── VIDEO-SPEC.md          # Full video specification
├── VOICEOVER-SCRIPT.md    # Narration script (~3 min)
├── CLAUDE.md              # AI context
├── PROJECT-STATUS.md      # Production tracking
└── remotion/
    ├── src/
    │   └── config/
    │       └── sprint-config.ts   # Demo configuration
    └── public/
        ├── images/        # Logos (included)
        ├── demos/         # Add your recordings here
        └── audio/         # Add voiceover/music/SFX here
```

## Assets

Large media files are not included. See `ASSETS-NEEDED.md` for:
- 7 demo video recordings (iPhone + browser)
- Voiceover generation
- Background music
- Sound effects (whoosh, chime, click)

## Recording Requirements

- **iPhone**: Physical device required (screenshare doesn't work on simulator)
- **Browser**: Mac for recording remote participant view
- **Split-screen**: Record phone and browser separately

## Adapting for Your Sprint

1. Update `VIDEO-SPEC.md` with your sprint content
2. Modify `VOICEOVER-SCRIPT.md` for your narration
3. Edit `remotion/src/config/sprint-config.ts`:
   ```typescript
   export const sprintConfig = {
     info: {
       name: 'Your Sprint Name',
       version: '1.0.0',
       dateRange: 'Jan 15 - Jan 29',
     },
     demos: [
       { type: 'single', videoFile: 'your-demo.mp4', ... }
     ]
   };
   ```
4. Record new demos following the spec
5. Generate new voiceover

---

*Contributed by [Digital Samba](https://digitalsamba.com) - European video conferencing API*
