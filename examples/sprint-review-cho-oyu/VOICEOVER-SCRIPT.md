# Voiceover Script
## Digital Samba Mobile iOS Sprint Review

**Total Duration**: ~2:27

---

### [0:00] Title (5s)

*[No voiceover - title card on screen]*

---

### [0:05] Overview (15s)

This sprint we focused on stability improvements for the iOS app, with a major focus on screenshare lifecycle reliability and connection persistence while the app is backgrounded. Let's walk through what we fixed.

---

### [0:20] App Rename (15s)

The app has been renamed from "Digital Samba E" to "Digital Samba". To make this possible, we had to upgrade the legacy app's React Native - from 0.68 to 0.77, spanning nearly three years and nine versions. Previous attempts had ultimately failed, but this time we got it right. All functionality retained, including screenshare.

---

### [0:35] Sleep/Lock (21s)

Hanna found this issue during testing - the app was disconnecting when users locked their device. This is a common scenario - a device may go into lock state during a video conference even if the user is still following along.

*[Demo: Lock device, wait, unlock]*

Now when you unlock, you're right back in the meeting. The user's camera goes black to other participants while locked - that's an iOS privacy restriction. Similar behaviour occurs when the app is backgrounded during screenshare.

---

### [0:56] Background Screenshare (15s)

A related issue from Mediatii - screensharing while backgrounded would disconnect after a minute or two. Watch the timer - we're six minutes in and still connected. One note - this requires audio active in the room, as iOS uses the audio session to keep the app alive.

---

### [1:11] Tile Timing (12.5s)

Previously, remote participants would see a screenshare tile before you'd even started broadcasting. We now bring up the Janus connection before the broadcast picker is shown - so the tile only appears when you actually press start.

---

### [1:24] Button State (24s)

Another issue was button state after network reconnection. Let me demonstrate.

*[Demo: Toggle airplane mode]*

I've toggled airplane mode on. The app disconnects... now airplane mode off... and we're reconnected. Notice the screenshare button correctly shows it's still active. I can tap to stop, and it works. Previously, this button would be stuck and unresponsive.

---

### [1:48] Recording Indicator (12s)

We also fixed an issue where force-quitting during screenshare would leave the red recording indicator stuck in the status bar. Now when I swipe up to close the app, the indicator clears within seconds.

---

### [2:00] Deep Links (16s)

Finally, deep links - another fix requested by Mediatii. When the app was already running in the background, tapping a meeting link wouldn't navigate to the room.

*[Demo: Tap meeting link from Discord, app joins Nina's WebSummit room]*

Now it works correctly - here we tap a link from Discord, and the app comes to foreground and joins the meeting as expected.

---

### [2:16] Summary (12s)

That's a wrap on iOS improvements for this sprint. Nine bug fixes total, with a heavy focus on screenshare stability and connection persistence. BUILD 233 has passed App Store review and is pending developer release.

*[End card]*

---

## Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Title | 5s | 0:05 |
| Overview | 15s | 0:20 |
| App Rename | 15s | 0:35 |
| Sleep/Lock | 21s | 0:56 |
| Background Screenshare | 15s | 1:11 |
| Tile Timing | 12.5s | 1:24 |
| Button State | 24s | 1:48 |
| Recording Indicator | 12s | 2:00 |
| Deep Links | 16s | 2:16 |
| Summary | 12s | 2:27 |

**Total: ~2:27** (+ 80s Making Of for director's cut = ~3:47)
