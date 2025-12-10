#!/usr/bin/env python3
"""Generate voiceover and background music for Digital Samba Skill Demo."""

import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save, VoiceSettings

# Load API key from .env
load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Enhanced voice clone
VOICE_ID = "lTB2h59fts8whFjtZNrF"

# Voiceover script - full version synced to scene durations
# Scene timing: Title(8s), Problem(10s), Solution(7s), Install(20s), Build(30s), Walkthrough(50s), Summary(7s), CTA(25s)
# Voiceover starts at 4s into video. Total video: 157s
VOICEOVER_SCRIPT = """
Video conferencing for your app. In minutes.

<break time="3.0s" />

Adding video calls to your application usually means weeks of work. Wading through API documentation, implementing authentication flows, managing WebRTC complexity. What if AI could help?

<break time="1.0s" />

Introducing the Digital Samba skill for Claude. It gives Claude deep knowledge of video conferencing integration - APIs, SDKs, authentication patterns - all the context Claude needs to build your integration.

<break time="1.0s" />

Setup is simple. Clone the skill repository into your project's claude folder, and Claude instantly understands Digital Samba's REST API, embedded SDK, and integration best practices. The skill includes comprehensive documentation, code examples, and proven patterns for common use cases.

<break time="1.5s" />

Let's see it in action. We asked Claude to build an interview room application - a real use case where recruiters conduct video interviews with candidates. We gave Claude creative freedom to design the experience. It built a complete Next.js application with JWT authentication, room management, and embedded video calls. The skill provided all the context - API endpoints, SDK methods, authentication patterns. In minutes, a working application designed and built by AI.

<break time="1.5s" />

The app has two user flows. Interviewers create rooms and get a shareable code. Candidates enter the code to join. Behind the scenes, the API creates private rooms with recording enabled, and JWT tokens control access permissions. The experience feels native to the product - not a third-party add-on.

<break time="1.0s" />

And just like that - a fully functional video interview room. Real-time video, screen sharing, recording controls. All running on Digital Samba's infrastructure, branded as your own. Whether you're building telehealth, online education, or collaboration tools - this is how fast integration can be.

<break time="1.0s" />

From idea to working integration in minutes. No video infrastructure to manage. And the flexibility to customize every aspect of the experience.

<break time="1.5s" />

Get the Digital Samba skill on GitHub. Add video conferencing to your product today.

<break time="3.0s" />
"""

def generate_voiceover():
    """Generate the voiceover audio."""
    print("Generating voiceover...")
    audio = client.text_to_speech.convert(
        text=VOICEOVER_SCRIPT.strip(),
        voice_id=VOICE_ID,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.80,
            similarity_boost=0.90,
            style=0.15,
            speed=1.0
        )
    )

    output_path = "remotion/public/audio/voiceover.mp3"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save(audio, output_path)
    print(f"Voiceover saved to: {output_path}")

def generate_background_music():
    """Generate upbeat energetic background music."""
    print("Generating background music...")
    result = client.music.compose(
        prompt="Upbeat electronic, energetic tech product demo, problem-solving momentum, driving synths, modern and confident, startup energy, 120 BPM, light drums, inspirational build-up, achievement feeling",
        music_length_ms=150000,  # 2:30 = 150 seconds
        force_instrumental=True
    )

    output_path = "remotion/public/audio/background-music.mp3"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in result:
            f.write(chunk)
    print(f"Background music saved to: {output_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "voiceover":
            generate_voiceover()
        elif sys.argv[1] == "music":
            generate_background_music()
        else:
            print("Usage: python generate_audio.py [voiceover|music]")
    else:
        # Generate both
        generate_voiceover()
        generate_background_music()
        print("\nAll audio generated successfully!")
