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

### PD Clone Voice (Thai, standalone)

`tools/pdclone_tts.py` voices Thai narration with PD Clone Voice. It is built for the free account's 20 calls a
day, so it is not a `voiceover.py` provider (that makes one call per scene):

- Scenes are joined with ` ... ` and sent in as few calls as fit 2,000 characters, then the WAV is cut back into
  one file per scene at the pauses and trimmed of silence at both ends.
- English words are rewritten as Thai readings before voicing (Claude to คล็อด). Built-in readings cover common
  names; add a project's own with `--lexicon words.json`. A Latin word left without a reading stops the run
  before a call is spent. The manifest keeps the original text as `caption` for subtitles.
- Each part's characters per second is compared with the others; a part that is much faster is flagged as
  likely to have skipped a word.
- A rerun with the same text reuses the saved raw WAV and spends no call.
- HTTP 402 or 429 means the day's quota is used up: the tool exits with code 3 and never retries.

```bash
uv run tools/pdclone_tts.py --list-voices
uv run tools/pdclone_tts.py --scenes scenes.json --out-dir public/audio/scenes --dry-run   # plan, no call
uv run tools/pdclone_tts.py --scenes scenes.json --out-dir public/audio/scenes --voice "พี่เมย์" --speed 1.0
```

`--scenes` takes a list of `{id, text}` (or `narration`), or `--scene-dir` one `.txt` per scene. Output is
`<id>.mp3` per scene plus `pdclone-manifest.json` (durations, frames at 30fps, caption and spoken text).
Config: `PDCLONE_API_KEY` in `.env` (or `PD_VOICE_KEY` in the environment), `PDCLONE_VOICE_ID` optional.
With `--scenes` from concept-explainer-short (scenes with a `slug`), files are named `<id>_<slug>.mp3` and
`--vo-durations vo_durations.json` writes the durations `build.py` reads.

**Prefer the one take over the pieces.** The raw WAV in `<out-dir>/_raw/` is a single read of every scene, so its
breaths and intonation run across the scene breaks. Played as one track, it sounds continuous where four trimmed
pieces with gaps between them sound stop-start. Use the per-scene files only to find where each scene starts and
ends, then time the picture to the take. For a quicker read with no gaps, cutting every pause over 0.1s and
speeding the take 10% (ffmpeg `atempo`, pitch kept) was approved by ear on a Thai short, taking a 4 scene take
from 27.2s to 22.7s. Beyond about 15%, voice again with `--speed` instead, which costs a call. Whisper's word
timing does not work on Thai, so time Thai captions by each phrase's share of the scene's characters, snapped to
the nearest pause in the take.

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
