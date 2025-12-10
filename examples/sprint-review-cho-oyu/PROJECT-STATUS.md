# Project Status: Sprint Review Video
## Digital Samba Mobile iOS v4.0.2 - Sprint Cho Oyu

**Last Updated**: 2025-12-04
**Target Duration**: ~3:47 (director's cut) / ~2:27 (sprint review cut)
**Output**: MP4 (1920x1080, 30fps)

---

## Current Phase: Voiceover Script Complete - Ready for Recording

| Phase | Status | Notes |
|-------|--------|-------|
| Spec & Script | Done | VIDEO-SPEC.md, VOICEOVER-SCRIPT.md |
| Remotion Scaffold | Done | All components built |
| Branding/Styling | Done | Animated bg, transitions, orange accent |
| Slide Animations | Done | All slides polished, larger fonts |
| Director's Cut | Done | "Making Of" sequence (commented out for now) |
| Asset Recording | Done | All demo videos recorded + trimmed clips |
| Visual Review | Done | All slides reviewed and adjusted |
| Voiceover Script | **Done** | Script reviewed and timed to match video |
| Voiceover Recording | **Next** | Record or use TTS |
| Final Render | Not Started | Ready when voiceover complete |

---

## Components Created

### Core Slides
- [x] **TitleSlide** - Scale-up + staggered fade animation, larger fonts (88px title)
- [x] **OverviewSlide** - Centered layout, staggered bullet animations, larger fonts (78px title, 44px bullets)
- [x] **AppRenameSlide** - Horizontal Step 1/Step 2 layout with actual app icons
  - Uses legacy.png and ds-logo.png
  - RN upgrade animation 0.68 → 0.77
  - Larger fonts and icons (130px)
- [x] **SummarySlide** - Two-phase: stats with spring animation, then screenshot overlay
  - Stats: 1 App Rename, 9 Bug Fixes, 6 Screenshare Fixes (140px numbers)
  - Screenshot springs in with bounce effect
- [x] **DemoSection** - Video with label/JIRA overlay, playbackRate support (32px labels)
- [x] **SplitScreen** - Side-by-side demos, playbackRate support (32px labels)
- [x] **TimelapseDemo** - 24-style yellow elapsed timer, speed indicator
- [x] **TimelapseDemoWithSkip** - Shows 0-1min, skips to 6min, then outro
- [x] **MakingOf** - Director's cut finale (80s, currently commented out)

### Visual Polish
- [x] **AnimatedBackground** - Subtle floating shapes, persistent across slides
- [x] **SlideTransition** - Zoom/blur-fade transitions between sequences
- [x] Transparent slide backgrounds (animated bg shows through)
- [x] Spring animations on stats and screenshots
- [x] 24-style digital clock (yellow glow, 00:00 format)

---

## Asset Checklist

### Images (public/images/) - DONE
| File | Status |
|------|--------|
| ds-logo.png | ✅ Added |
| legacy.png | ✅ Added |
| app-icon-after.png | ✅ Added |
| 4.0.2-pending-release.png | ✅ Added (App Store Connect screenshot) |

### Demo Recordings (public/demos/) - DONE
| File | Duration | Playback | Sequence | Notes |
|------|----------|----------|----------|-------|
| sleep-lock.mp4 | 29s | 1.4x | 21s | Lock device, wait, unlock |
| background-screenshare-intro.mp4 | 60s | 12x | 5s | First minute of demo |
| background-screenshare-return.mp4 | 14s | 4.7x | 3s | 6:00-6:14 section |
| background-screenshare-outro.mp4 | 6s | 1x | 6s | Final exit (6:14-6:20) |
| tile-timing-phone.mp4 | 29s | 1x | 12.5s | iPhone view, cut at frame 375 |
| tile-timing-browser.mp4 | 71s | 1x | 12.5s | Browser view, startFrom=300 |
| button-state.mp4 | 33s | 1.4x | 24s | WiFi toggle, reconnection |
| recording-indicator.mp4 | 17s | 1.4x | 12s | Force quit during screenshare |
| deep-links.mp4 | 23s | 1.4x | 16s | Discord link demo |

### Audio (public/audio/) - PENDING
| File | Duration | Status | Notes |
|------|----------|--------|-------|
| voiceover.mp3 | ~2.5min | Pending | Script needs timing update |
| background-music.mp3 | ~3min | Optional | Ambient track if desired |

---

## Timeline Structure

### Sprint Review Cut (~2:27 / 147s)
```
0:00  [5s]    Title Card
0:05  [15s]   Overview Slide
0:20  [15s]   App Rename/Migration    SAMBA-8117
0:35  [21s]   Sleep/Lock Demo         SAMBA-8059 (1.4x)
0:56  [15s]   Background Screenshare  SAMBA-8094 (skip middle)
1:11  [12.5s] Tile Timing (split)     SAMBA-8053/8057
1:24  [24s]   Button State Demo       SAMBA-8054 (1.4x)
1:48  [12s]   Recording Indicator     SAMBA-8056 (1.4x)
2:00  [16s]   Deep Links Demo         SAMBA-8112 (1.4x)
2:16  [12s]   Summary Slide
2:27  END
```

### Director's Cut (~3:47 / 227s) - Currently disabled
```
... same as above through 2:27 ...
2:27  [80s]   The Making Of (AI workflow reveal)
3:47  END
```

---

## Next Steps

### Next Session: Voiceover Recording
1. Record voiceover audio (or use TTS) based on VOICEOVER-SCRIPT.md
2. Place audio file at `public/audio/voiceover.mp3`
3. Optional: Add background music
4. Final render: `npm run render`
5. Decide on Director's Cut inclusion

---

## Commands Reference

```bash
cd /Users/conalmullan/work/video/sprint-review-cho-oyu/remotion

# Preview in browser
npm run studio

# Render full video
npm run render

# Render preview (first 10 seconds)
npm run render:preview
```

---

## Session Notes

### 2025-12-03 - Session 1: Scaffold & Styling
- Created Remotion scaffold with all components
- Added placeholder handling for missing videos
- Created AppRenameSlide with two-row animation
- Applied Digital Samba branding

### 2025-12-04 - Session 2: Polish & Director's Cut
- Updated TitleSlide: Sprint "Cho Oyu", dates, iOS Embedded App Update
- Improved OverviewSlide: centered, slower staggered animations
- Restructured AppRenameSlide: horizontal layout, dual version animations
- Added App Version 4.0.1→4.0.2 animation, Legacy App 1.35→1.36 subtext
- Updated SummarySlide: counting stats, App Store screenshot
- Added AnimatedBackground (subtle floating shapes)
- Added SlideTransition (zoom/blur-fade effects)
- Made all slide backgrounds transparent
- Created MakingOf component for director's cut

### 2025-12-04 - Session 3: Demo Recording & Integration
- Added all 7 demo video recordings
- Added SAMBA-8094 (background screenshare) as new section
- Created TimelapseDemo component with 24-style elapsed timer
- Trimmed background-screenshare.mp4 to 6:20 using ffmpeg
- Updated SplitScreen with independent leftStartFrom/rightStartFrom props
- Added playbackRate support to DemoSection
- Total duration was 285s (~4:45 with Making Of)

### 2025-12-04 - Session 4: Visual Review & Optimization
- **Font sizes increased** across all slides for better readability
  - Title: 72→88px, Overview: 64→78px, bullets: 36→44px
  - Labels: 24→32px, JIRA refs: 18→24px
  - App icons: 100→130px
- **TitleSlide**: Changed from slide-up to scale-up + staggered fade animation
- **AppRenameSlide**: Added actual app icons (legacy.png, ds-logo.png)
- **SummarySlide**: Redesigned with two-phase animation (stats then screenshot overlay with spring)
- **TimelapseDemo**: 24-style yellow digital clock, speed indicator overlay
- **TimelapseDemoWithSkip**: New component - shows 0-1min, skips middle, shows 6-6:20
- **Demo speedups**: Most demos now play at 1.4x to reduce overall length
- **Tile Timing**: Cut to 375 frames (12.5s) - shows key sync moment
- **Making Of**: Commented out for sprint review cut
- **Total duration reduced**: 4:45 → 2:27 (sprint review cut)

### 2025-12-04 - Session 5: Voiceover Script Review
- Reviewed script section-by-section to match new video timings
- **Overview**: Updated focus to "screenshare lifecycle reliability and connection persistence while backgrounded"
- **App Rename**: Added context about RN upgrade (0.68→0.77) being prerequisite for rename, previous attempts had ultimately failed
- **Sleep/Lock**: Added Hanna found issue, common scenario explanation, iOS privacy restriction detail
- **Background Screenshare**: Condensed significantly (30s→15s), kept audio session caveat
- **Tile Timing**: Condensed (30s→12.5s), added Janus connection technical detail
- **Button State**: Updated demo reference to airplane mode toggle
- **Recording Indicator**: Trimmed technical detail about broadcast extension
- **Summary**: Updated to "passed App Store review and pending developer release"
- **Total script duration**: 3:25 → 2:27 (matches video)

---

## Decisions Log

| Decision | Choice | Date | Rationale |
|----------|--------|------|-----------|
| Framework | Remotion | 2025-12-03 | React-based, programmatic |
| Resolution | 1920x1080 | 2025-12-03 | Standard HD |
| Two versions | Sprint + Director's cut | 2025-12-04 | Stakeholder context |
| Director's cut content | AI workflow reveal | 2025-12-04 | Meta demonstration |
| Animation style | Subtle, professional | 2025-12-04 | Not flashy, polished |
| SAMBA-8094 placement | After Sleep/Lock | 2025-12-04 | Both are connection persistence fixes |
| Timelapse style | 24-style yellow clock | 2025-12-04 | Dramatic, professional |
| Demo playback | 1.4x for most demos | 2025-12-04 | Reduce runtime, still watchable |
| Background screenshare | Skip middle section | 2025-12-04 | Show start (1min) + end (20s) |
| Font sizes | ~20% larger | 2025-12-04 | Better readability |
| Sprint review length | ~2:27 | 2025-12-04 | Concise for stakeholder review |
