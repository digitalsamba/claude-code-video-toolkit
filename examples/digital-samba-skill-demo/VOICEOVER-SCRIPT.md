# Voiceover Script - Digital Samba Skill Demo

**Total Duration**: ~2:20 (voiceover starts at 0:04, ends at 2:28)
**Voice**: Professional, conversational, developer-friendly
**Pace**: Moderate, with pauses for visual focus

---

## Script with Timing

### [0:04 - 0:08] Title
> "Add video calls to your app in an afternoon."

### [0:08 - 0:18] Problem Statement
> "Video conferencing integration usually means weeks of work - wading through API docs, wrangling authentication, dealing with WebRTC headaches. There's a faster way."

### [0:18 - 0:25] Solution Intro
> "The Digital Samba skill gives Claude everything it needs to build video apps with you."

### [0:25 - 0:45] Skill Installation
> "A few simple commands to install. Clone the repo, and Claude learns our REST API, SDK methods, and common integration patterns. No configuration needed."

### [0:45 - 1:15] Building the Demo App
> "Let's build something real. We asked Claude to create an interview room app from scratch. It picked Next.js, set up API routes, and handled JWT authentication - all using patterns from the skill. A few minutes later, we have a working app."

### [1:15 - 1:50] App Walkthrough
> "Here's how it works. Interviewers create a room, get a code to share. Candidates enter the code and join. The API handles the room, enables recording, generates tokens. Simple flow, no fuss."

### [1:50 - 2:05] Live Video Call
> "And we're connected. Video, audio, screen sharing, recording - all running on Digital Samba's infrastructure. You don't manage any of it."

### [2:05 - 2:15] Code Highlight
> "The generated code follows best practices - secure JWT token generation, proper SDK initialization, and clean API integration patterns."

### [2:15 - 2:22] Summary
> "Five minutes. No infrastructure to manage. Full control over the experience."

### [2:22 - 2:28] CTA
> "The Digital Samba skill is on GitHub. Go give it a try."

---

## Script Statistics

- **Word count**: ~220 words
- **Speaking time**: ~1:50 at natural pace
- **Style**: Free-flowing, conversational - no artificial pauses

---

## ElevenLabs Generation Notes

### Voice Settings (Recommended)
```python
voice_settings = {
    "stability": 0.80,        # Slightly more variation for natural feel
    "similarity_boost": 0.90,  # Strong voice match
    "style": 0.15,            # Subtle expressiveness
    "use_speaker_boost": True
}
```

### Model
- Use `eleven_multilingual_v2` for natural prosody
- Alternative: `eleven_turbo_v2_5` for faster generation

### Background Music
Generate upbeat, energetic tech background music:
- **Style**: Modern electronic, tech product demo, upbeat and driving
- **Tempo**: Medium-fast (~120 BPM)
- **Volume**: 0.20-0.25 (under voiceover but noticeable energy)
- **Duration**: 2:30 to cover full video

---

## Plain Text Version (for TTS)

```
Add video calls to your app in an afternoon.

Video conferencing integration usually means weeks of work - wading through API docs, wrangling authentication, dealing with WebRTC headaches. There's a faster way.

The Digital Samba skill gives Claude everything it needs to build video apps with you.

A few simple commands to install. Clone the repo, and Claude learns our REST API, SDK methods, and common integration patterns. No configuration needed.

Let's build something real. We asked Claude to create an interview room app from scratch. It picked Next.js, set up API routes, and handled JWT authentication - all using patterns from the skill. A few minutes later, we have a working app.

Here's how it works. Interviewers create a room, get a code to share. Candidates enter the code and join. The API handles the room, enables recording, generates tokens. Simple flow, no fuss.

And we're connected. Video, audio, screen sharing, recording - all running on Digital Samba's infrastructure. You don't manage any of it.

The generated code follows best practices - secure JWT token generation, proper SDK initialization, and clean API integration patterns.

Five minutes. No infrastructure to manage. Full control over the experience.

The Digital Samba skill is on GitHub. Go give it a try.
```
