# Sprint Review Video Spec
## Digital Samba Mobile - iOS v4.0.2

**Created**: 2025-12-03
**Purpose**: Sprint review presentation of iOS bug fixes and improvements
**Target Length**: 3-5 minutes

---

## Context

Digital Samba Mobile is a React Native video conferencing app with native screenshare capabilities. Version 4.0.2 (BUILD 233) includes significant bug fixes for iOS, primarily around screenshare stability and connection persistence.

---

## What's New (Official Copy)

```
App renamed to 'Digital Samba'

Screen Sharing Improvements
- Fixed disconnect when device locked or app in background
- Button state now syncs correctly after network reconnection
- Recording indicator reliably dismisses when sharing stops
- Improved state management for stop/restart sessions
- Remote users only see tile when broadcast actually starts

Deep Linking
- Fixed deep links not opening room when app already running
```

---

## Video Structure

### 1. Title Card (5 seconds)
- Title: "Digital Samba Mobile - iOS Sprint Review"
- Subtitle: "Version 4.0.2 (BUILD 233)"
- Date: December 2025

### 2. Overview Slide (15 seconds)
**Voiceover**: "This sprint we focused on stability improvements for the iOS app, with a major focus on screenshare reliability and connection persistence."

**On Screen**:
- App renamed to 'Digital Samba'
- Screen Sharing: 5 improvements
- Deep Linking: 1 fix
- Connection: 1 fix

### 3. Demo: App Rename (10 seconds)
**What to Show**:
- iPhone home screen with app icon showing "Digital Samba" name
- App launch showing new branding

**Voiceover**: "The app has been renamed from 'Digital Samba E' to 'Digital Samba' following updates to our embedded SDK."

### 4. Demo: Sleep/Lock Connection Fix (30 seconds)
**JIRA**: SAMBA-8059

**What to Show**:
1. Join a room with camera/mic enabled
2. Show video working
3. Lock the device (press power button)
4. Wait 10-15 seconds
5. Unlock device
6. Show still connected to room, video resumes

**Voiceover**: "Previously, locking your device would cause a disconnect from the conference. We've implemented an audio session module that maintains the connection during sleep. When you unlock, you're right back in the meeting."

**Technical Note**: Camera goes black during lock (iOS limitation) but connection persists and camera resumes on unlock.

### 5. Demo: Screenshare - Remote Tile Timing (30 seconds)
**JIRA**: SAMBA-8053, SAMBA-8057

**What to Show** (split screen - phone + remote browser view):
1. Tap screenshare button on iPhone
2. iOS broadcast picker appears
3. Remote view shows NOTHING yet (this is the fix)
4. Tap "Start Broadcast" on picker
5. Remote view now shows screenshare tile

**Voiceover**: "We reworked the screenshare lifecycle. Previously, remote participants would see a screenshare tile as soon as you tapped the button, before the broadcast actually started. Now the tile only appears when you're actually sharing."

### 6. Demo: Screenshare - Button State After Reconnection (30 seconds)
**JIRA**: SAMBA-8054

**What to Show**:
1. Start screenshare (remote sees it)
2. Toggle WiFi off on iPhone
3. Wait for disconnect
4. Toggle WiFi back on
5. App reconnects
6. Screenshare button correctly shows active state
7. Tap to stop - it works

**Voiceover**: "After a network reconnection, the screenshare button now correctly reflects the actual broadcast state. You can stop your screenshare normally instead of getting stuck with an inactive button."

### 7. Demo: Screenshare - Recording Indicator Cleanup (20 seconds)
**JIRA**: SAMBA-8056

**What to Show**:
1. Start screenshare (red recording indicator in status bar)
2. Force quit the app (swipe up from app switcher)
3. Recording indicator disappears within seconds

**Voiceover**: "When you close the app during screenshare, the red recording indicator now clears properly. The broadcast extension receives the termination signal and stops gracefully."

### 8. Demo: Deep Links (20 seconds)
**JIRA**: SAMBA-8112

**What to Show**:
1. App is already running (show it in app switcher)
2. Open Messages or Safari with a meeting link
3. Tap the link
4. App opens and joins the room

**Voiceover**: "Deep links now work correctly when the app is already running. Previously, tapping a meeting link would open the app but not navigate to the room."

### 9. Summary Slide (10 seconds)
**On Screen**:
- 8 bug fixes total
- Focus: Screenshare stability
- Available: BUILD 233
- Pending: App Store review

**Voiceover**: "That's a wrap on iOS improvements for this sprint. BUILD 233 is archived and pending App Store review."

---

## Recording Requirements

### Equipment
- Physical iPhone (screenshare doesn't work on simulator)
- Mac for recording "remote participant" browser view
- Quiet environment for voiceover

### iPhone Setup
- Enable Do Not Disturb (clean status bar)
- Remove sensitive notifications
- Use test room (not production with real users)
- Good lighting if showing device physically

### Recording Software
- iPhone: Built-in Screen Recording (Control Center)
- Mac browser: QuickTime Player > New Screen Recording
- Alternative: OBS for picture-in-picture layouts

### Split Screen Shots
For screenshare demos, consider side-by-side layout:
```
+------------------+------------------+
|                  |                  |
|   iPhone View    |  Remote Browser  |
|                  |     (Mac)        |
|                  |                  |
+------------------+------------------+
```

---

## Assets Needed

| Asset | Status | Notes |
|-------|--------|-------|
| Title slide | TO CREATE | Keynote/Figma |
| Overview slide | TO CREATE | Bullet points |
| Summary slide | TO CREATE | Key stats |
| iPhone recordings | TO RECORD | 5-6 clips |
| Browser recordings | TO RECORD | For split-screen demos |
| Voiceover audio | TO RECORD | ~3 min narration |
| Background music | OPTIONAL | Subtle, royalty-free |

---

## Technical Details (for reference)

### Commits Included
```
4867525 fix(android): Resolve double-tap to stop screenshare issue [SAMBA-8050]
b6824e8 fix(android): Add join timeout to handle permission denial hang [SAMBA-8113]
c255e9d chore(ios): Bump to BUILD 233, rename app to 'Digital Samba'
1ef7730 fix(ios): Robust state sync with WebRTC connection check [SAMBA-8054]
c4ae18a fix(ios): State sync on foreground + app termination cleanup [SAMBA-8054/8056]
96c123a fix(ios): Two-phase screenshare to prevent premature tile [SAMBA-8053/8057]
bc46d52 fix(deeplink): Handle warm start deep links on iOS and Android [SAMBA-8112]
1f4fa14 fix(ios): Add RoomSessionModule for background connection persistence [SAMBA-8059]
```

### Key Files Changed
- `ios/DigitalSambaMobile/NativeModules/RoomSessionModule.swift` - Background persistence
- `ios/DigitalSambaMobile/NativeModules/WebRTCManager.swift` - Two-phase screenshare
- `ios/DigitalSambaMobile/NativeModules/ScreenshareModule.swift` - State queries
- `ios/DigitalSambaMobile/AppDelegate.swift` - Termination handling, deep links
- `src/hooks/useScreenshare.ts` - State sync on foreground

---

## Notes for Video AI Agent

1. This is an internal sprint review, not customer-facing marketing
2. Focus on demonstrating the fixes work, not flashy production
3. Voiceover can be casual/technical - audience is dev team and stakeholders
4. Each demo should clearly show the "after" behavior (we don't have "before" recordings)
5. Keep it concise - aim for 3-5 minutes total
