---
name: elevenlabs
description: Generate AI voiceovers, sound effects, and music using ElevenLabs APIs. Use when creating audio content for videos, podcasts, or games. Triggers include generating voiceovers, narration, dialogue, sound effects from descriptions, background music, soundtrack generation, voice cloning, or any audio synthesis task.
---

# ElevenLabs Audio Generation

ElevenLabs provides AI-powered audio generation including text-to-speech, sound effects, and music. Requires API key as `ELEVENLABS_API_KEY`.

## Setup

```bash
pip install elevenlabs python-dotenv
```

Store API key in `.env`:
```
ELEVENLABS_API_KEY=your_key_here
```

## Text-to-Speech (Voiceover)

```python
from elevenlabs.client import ElevenLabs
from elevenlabs import save, VoiceSettings
import os

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

audio = client.text_to_speech.convert(
    text="Welcome to my video!",
    voice_id="JBFqnCBsd6RMkjVDRZzb",  # George - warm narrator
    model_id="eleven_multilingual_v2",
    voice_settings=VoiceSettings(
        stability=0.5,        # 0-1: Lower = more expressive
        similarity_boost=0.75,
        style=0.5,
        speed=1.0
    )
)
save(audio, "voiceover.mp3")
```

**Models:**
- `eleven_flash_v2_5` - Ultra-low latency (~75ms)
- `eleven_multilingual_v2` - Highest quality, 32 languages
- `eleven_turbo_v2_5` - Balanced quality/speed

**Popular Voices:**
- George: `JBFqnCBsd6RMkjVDRZzb` (warm narrator)
- Rachel: `21m00Tcm4TlvDq8ikWAM` (clear female)
- Adam: `pNInz6obpgDQGcFmaJgB` (professional male)

List all voices: `client.voices.get_all()`

## Voice Cloning

Clone your own voice for personalized voiceovers.

### Instant Voice Clone (1-2 min audio, instant)

```python
# Files must be opened as binary file handles, not paths
with open("sample.mp3", "rb") as f:
    voice = client.voices.ivc.create(
        name="My Voice",
        files=[f],
        remove_background_noise=True
    )

print(f"Voice ID: {voice.voice_id}")  # Save this for reuse

# Use cloned voice
audio = client.text_to_speech.convert(
    text="This is my cloned voice!",
    voice_id=voice.voice_id,
    model_id="eleven_multilingual_v2"
)
```

**Important:**
- Use `client.voices.ivc.create()` (not `client.voices.clone()`)
- Pass file handles opened in binary mode (`"rb"`), not file paths
- Convert m4a to mp3 first if needed: `ffmpeg -i input.m4a -codec:a libmp3lame -qscale:a 2 output.mp3`
- Multiple samples (2-3 clips) improve voice accuracy
- Save the voice ID for reuse to avoid creating new clones each run

**Recording tips:** Quiet room, 1-2 min total (longer is better), consistent tone, USB mic or phone sufficient.

**Voice Settings for natural speech:**
- `stability`: 0.75-0.85 (higher = more consistent, fewer odd inflections)
- `similarity_boost`: 0.9-0.95 (higher = closer to original voice)
- `style`: 0.0-0.1 (lower = more neutral, less "character")
- `speed`: 1.0 (adjust if timing doesn't match video)

**Script pauses:** Use `<break time="1.0s" />` tags in script text to add pauses between sections.

### Professional Voice Clone (30+ min audio, 3-6 hours training)

Requires Creator plan+. Higher fidelity, requires voice verification. See [reference.md](reference.md) for full PVC workflow.

## Sound Effects

Generate sound effects from text descriptions (max 22 seconds).

```python
result = client.text_to_sound_effects.convert(
    text="Thunder rumbling followed by heavy rain",
    duration_seconds=10,
    prompt_influence=0.3  # 0-1: How strictly to follow prompt
)

with open("thunder.mp3", "wb") as f:
    for chunk in result:
        f.write(chunk)
```

**Prompt tips:** Be specific - "Heavy footsteps on wooden floorboards, slow and deliberate, with creaking"

## Music Generation (Eleven Music)

Generate studio-grade music (10 seconds to 5 minutes).

```python
result = client.music.compose(
    prompt="Upbeat indie rock, catchy guitar riff, energetic drums, travel vlog",
    music_length_ms=60000,  # 60 seconds
    force_instrumental=True
)

with open("background_music.mp3", "wb") as f:
    for chunk in result:
        f.write(chunk)
```

**Important:** Use `client.music.compose()` (not `client.music.generate()`)

**Prompt structure:** Include genre, mood, instruments, tempo, use case.

## Remotion Integration

Save audio to `public/` folder for Remotion projects:

```python
save(audio, "public/audio/voiceover.mp3")
```

Then in Remotion:
```tsx
import { Audio, staticFile } from 'remotion';

<Audio src={staticFile('audio/voiceover.mp3')} />
```

## Advanced API

For detailed API documentation including all models, voice settings, output formats, rate limits, and Professional Voice Cloning workflow, see [reference.md](reference.md).

---

## Feedback & Contributions

If this skill is missing information or could be improved:

- **Missing a pattern?** Describe what you needed
- **Found an error?** Let me know what's wrong
- **Want to contribute?** I can help you:
  1. Update this skill with improvements
  2. Create a PR to github.com/digitalsamba/claude-code-video-toolkit

Just say "improve this skill" and I'll guide you through updating `.claude/skills/elevenlabs/SKILL.md`.
