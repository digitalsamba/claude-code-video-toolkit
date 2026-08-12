---
name: minimax-music
description: Generate instrumental or vocal music with MiniMax's hosted music API. Use for MiniMax music generation, global or mainland China endpoints, streamed audio, generated lyrics, or URL output.
---

# MiniMax Music

Use `tools/minimax_music.py` for hosted text-to-music generation. It supports both API regions and saves the returned audio to a local file.

## Setup

Add your key to `.env`:

```bash
MINIMAX_API_KEY=your_key_here
```

The tool sends it as a bearer token and never writes it to output files.

## Models

- `music-3.0` (default)
- `music-2.6`
- `music-3.0-free`
- `music-2.6-free`

Cover models and reference-audio fields are outside this text-to-music tool.

## Common Workflows

Generate an instrumental track:

```bash
python3 tools/minimax_music.py \
  --instrumental \
  --prompt "Warm cinematic ambient score, slow piano, soft strings" \
  --output public/audio/score.mp3
```

Generate a vocal song from lyrics:

```bash
python3 tools/minimax_music.py \
  --prompt "Bright indie pop, energetic chorus, polished production" \
  --lyrics "[Verse]\nMorning light across the city\n[Chorus]\nWe begin again" \
  --output public/audio/song.mp3
```

Generate lyrics from a prompt:

```bash
python3 tools/minimax_music.py \
  --lyrics-optimizer \
  --prompt "Hopeful acoustic folk song about a new beginning" \
  --output public/audio/song.mp3
```

Use the mainland China endpoint and its optional non-streaming watermark:

```bash
python3 tools/minimax_music.py \
  --region cn_zh \
  --aigc-watermark \
  --instrumental \
  --prompt "Gentle acoustic background music" \
  --output public/audio/background.mp3
```

## Request Controls

- `--region global_en` uses `https://api.minimax.io/v1/music_generation`.
- `--region cn_zh` uses `https://api.minimaxi.com/v1/music_generation`.
- `--output-format hex` is the default and supports regular or streamed responses.
- `--output-format url` downloads the returned link immediately; provider links expire after 24 hours.
- `--stream` accepts hex output only.
- `--format` accepts `mp3`, `wav`, or `pcm`.
- `--sample-rate` accepts `16000`, `24000`, `32000`, or `44100`.
- `--bitrate` accepts `32000`, `64000`, `128000`, or `256000`.
- `--json` emits a machine-readable success or error object.

For vocal generation, provide `--lyrics` or use `--lyrics-optimizer`. Instrumental generation requires `--prompt`.
