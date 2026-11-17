# Voiceover

Narration with ElevenLabs, Qwen3-TTS or 60db, and how to fix a cloned voice that sounds flat or rushed. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Voiceover Generation

Three TTS providers share one `voiceover.py` interface (`--provider`):
**elevenlabs** (default), **qwen3** (self-hosted, free), and **60db** (premium cloud).
Voice settings (`--stability`, `--similarity`, `--speed`) use a **unified 0-1 scale**
across all providers, for 60db they are auto-converted to its native 0-100 scale.

```bash
# Per-scene generation (recommended, ElevenLabs default)
uv run tools/voiceover.py --scene-dir public/audio/scenes --json

# Using Qwen3-TTS (self-hosted, free alternative to ElevenLabs)
uv run tools/voiceover.py --provider qwen3 --tone warm --scene-dir public/audio/scenes --json

# Using 60db (premium cloud TTS — needs SIXTYDB_API_KEY)
uv run tools/voiceover.py --provider 60db --scene-dir public/audio/scenes --json
uv run tools/voiceover.py --provider 60db --voice-id <uuid> --stability 0.6 --script SCRIPT.md --output out.mp3

# Single file (legacy)
uv run tools/voiceover.py --script SCRIPT.md --output out.mp3
```

### 60db (standalone)

`tools/sixtydb_tts.py` is the dedicated 60db tool (counterpart to `qwen3_tts.py`).
It exposes a `generate_audio()` used by `voiceover.py` and `redub.py`, plus a CLI.
Three transports all produce a finished audio file: `synthesize` (REST, default),
`stream` (NDJSON), and `websocket` (realtime; needs `pip install websocket-client`).

```bash
uv run tools/sixtydb_tts.py --text "Hello world" --output hello.mp3
uv run tools/sixtydb_tts.py --text "Hello" --transport stream --output hello.mp3
uv run tools/sixtydb_tts.py --list-voices          # GET /myvoices
```

Config: `SIXTYDB_API_KEY` (required), `SIXTYDB_VOICE_ID` (optional, falls back to
60db's default voice). Brands carry a `sixtydb` block in `voice.json` (voiceId +
settings). `redub.py --tts-provider 60db` uses 60db for the new voice while
transcription stays on ElevenLabs Scribe (60db has no STT); in `--sync` mode the
60db output is run back through Scribe to recover word timestamps.

## Clone Prosody (when a cloned voice sounds flat or rushed)

A clone copies its reference clip's **pace and intonation**. If the lines come out fast and flat, the clip is
fast and flat, and `--max-wpm` stretching below ~0.9x only smears them. Measure, then fix the clip:

```bash
uv sync --extra prosody
uv run tools/voice_prosody.py measure assets/voices/lugh-ref.wav public/audio/scenes/*.mp3   # sd < 2 st = flat
uv run tools/voice_prosody.py expand assets/voices/lugh-ref.wav assets/voices/lugh-ref-expressive.wav \
    --pitch 2.0 --tempo 0.85          # same voice and accent, bigger melody, 15% slower; clone from this
```

Then generate **one `.txt` per sentence**, three takes, and let `pick_takes.py` choose by pace then pitch spread
(it stretches only a winner still over 170 wpm, never below 0.9x):

```bash
for t in 1 2 3; do cp -r public/audio/phrases public/audio/takes/t$t; \
  uv run tools/voiceover.py --provider qwen3 --ref-audio assets/voices/lugh-ref-expressive.wav \
    --ref-text "..." --scene-dir public/audio/takes/t$t; done
uv run tools/pick_takes.py --takes-dir public/audio/takes --output public/audio/scenes --json
```

**Pronunciation (Qwen3 has no phoneme tags):** short words that look like acronyms can come out spelled, so "ops
escalations" became "O-P-S escalations". Fix it in the text: sentence case ("Ops"), or let a phrase carry it ("the ops
team"). The `.txt` is also the caption text, so prefer a rewording to a respelling ("opps"). To check by transcript,
copy the MP3s to a folder without their `.txt` and run `captions.py` there: the raw transcript shows letters as
"OP-IS" or "OPS".

Lugh's expressive reference lives at `assets/voices/lugh-ref-expressive.wav`. Measured on release 650: the raw
clip gave clones at 1.7-2.1 st sd and 180-215 wpm; the expanded clip gave 3-4 st at 145-165 wpm with no stretch.
Short closing sentences still rush; take selection handles most of it.

## Qwen3-TTS (Standalone)

```bash
uv run tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3
uv run tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3
uv run tools/qwen3_tts.py --text "Hello" --instruct "Speak enthusiastically" --output excited.mp3
uv run tools/qwen3_tts.py --text "Hello" --ref-audio sample.wav --ref-text "transcript" --output cloned.mp3
uv run tools/qwen3_tts.py --list-voices   # 9 speakers: Ryan, Aiden, Vivian, etc.
uv run tools/qwen3_tts.py --list-tones    # neutral, warm, professional, excited, etc.
```

Temperature controls expressiveness: `--temperature 1.2` (more expressive) or `--temperature 0.4` (more consistent).
