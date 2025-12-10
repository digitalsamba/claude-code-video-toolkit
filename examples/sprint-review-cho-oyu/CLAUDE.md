# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **video production project** for creating a sprint review presentation of Digital Samba Mobile iOS v4.0.2 (BUILD 233). The target is a 3-5 minute video demonstrating bug fixes and improvements.

## Key Files

- `VIDEO-SPEC.md` - Complete video specification including structure, demo sequences, recording requirements, and asset checklist
- `VOICEOVER-SCRIPT.md` - Timed voiceover script (~3 minutes) with section breakdowns

## Video Content Summary

The sprint review covers:
- App rename to "Digital Samba"
- 5 screenshare stability fixes (tile timing, button state sync, recording indicator cleanup)
- Sleep/lock connection persistence fix
- Deep link warm start fix

## Production Workflow

### Recording Method
- Use Digital Samba video conferencing platform for iPad screenshare recording
- Browser view captures for split-screen demos (showing remote participant perspective)
- Combine segments into final MP4

### Assets Required
1. Title/Overview/Summary slides (Keynote or similar)
2. iPhone screen recordings (5-6 clips per VIDEO-SPEC.md section demos)
3. Browser recordings for split-screen demos
4. Voiceover audio (text-to-speech or recorded narration)

### Demo Sections (from VIDEO-SPEC.md)
Each demo has specific JIRA references and step-by-step recording instructions:
- Sleep/Lock Fix (SAMBA-8059)
- Screenshare Tile Timing (SAMBA-8053, SAMBA-8057)
- Button State After Reconnection (SAMBA-8054)
- Recording Indicator Cleanup (SAMBA-8056)
- Deep Links (SAMBA-8112)

## Technical Context

The underlying product is a React Native video conferencing app with native iOS screenshare capabilities. Key native modules referenced:
- `RoomSessionModule.swift` - Background persistence
- `WebRTCManager.swift` - Two-phase screenshare
- `ScreenshareModule.swift` - State queries
