---
name: acestep
description: AI music generation with ACE-Step 1.5 — background music, vocal tracks, covers, stem extraction, audio repainting, and continuation for video production. Use when generating music, soundtracks, jingles, or working with audio stems. Triggers include background music, soundtrack, jingle, music generation, stem extraction, cover, style transfer, repaint, continuation, or musical composition tasks.
---

# ACE-Step 1.5 Music Generation

Open-source music generation via `tools/music_gen.py`.

**Cloud providers:**
- **modal** (the default when its endpoint is in `.env`): Self-hosted ACE-Step 1.5 XL Turbo (4B DiT, no LM) on Modal A10G. Requires `MODAL_MUSIC_GEN_ENDPOINT_URL`.
- **runpod**: Self-hosted ACE-Step 2B Turbo on RunPod. Requires `RUNPOD_ACESTEP_ENDPOINT_ID`.
- With neither set up, `music_gen.py` stops and gives the setup command.
- **Licence and cost:** the ACE-Step 1.5 weights and code are MIT, so commercial use of the music is free and
  needs no credit; you pay only GPU time, about $0.01-0.03 a take on Modal.
  A take can still resemble an existing song; listen before publishing.

## Setup

```bash
uv run tools/modal_setup.py music_gen        # Modal (recommended); writes MODAL_MUSIC_GEN_ENDPOINT_URL to .env
uv run tools/music_gen.py --setup             # RunPod
```

## Quick Reference

```bash
# Basic generation (on Modal when its endpoint is set up)
uv run tools/music_gen.py --prompt "Upbeat tech corporate" --duration 60 --output bg.mp3

# Generate 4 variations (ambient_1.mp3 ... ambient_4.mp3), then recommend one for the video
uv run tools/music_gen.py --prompt "Calm ambient piano" --duration 30 --variations 4 --output ambient.mp3
uv run tools/pick_music.py ambient_*.mp3 --target 30 --cuts 10,21 --under voice --pick ambient.mp3

# Fast mode (disable thinking)
uv run tools/music_gen.py --no-thinking --prompt "Quick draft" --duration 30 --output draft.mp3

# With musical control
uv run tools/music_gen.py --prompt "Calm ambient piano" --duration 30 --bpm 72 --key "D Major" --output ambient.mp3

# Scene presets (video production)
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
uv run tools/music_gen.py --preset tension --duration 20 --output problem.mp3
uv run tools/music_gen.py --preset cta --brand digital-samba --duration 15 --output cta.mp3

# Vocals with lyrics
uv run tools/music_gen.py --prompt "Indie pop jingle" --lyrics "[verse]\nBuild it better\nShip it faster" --duration 30 --output jingle.mp3

# Cover / style transfer
uv run tools/music_gen.py --cover --reference theme.mp3 --prompt "Jazz piano version" --duration 60 --output jazz_cover.mp3

# Stem extraction
uv run tools/music_gen.py --extract vocals --input mixed.mp3 --output vocals.mp3

# Fall back to self-hosted
uv run tools/music_gen.py --cloud modal --prompt "Background music" --duration 60 --output bg.mp3
```

## Fixing "Samey" Output

If generated music sounds repetitive or lacks variety, try these in order:

1. **Use Modal over RunPod**: Modal runs the XL Turbo 4B DiT, RunPod is still 2B
2. **Write a fuller prompt**: the endpoints run without the 5Hz LM, so nothing enriches a sparse prompt; name instruments, tempo and mood yourself
3. **Generate variations** — `--variations 4` generates 4 takes; `pick_music.py` recommends one with its reasons (it measures, it can't hear), and `audition.py` puts each take under the finished picture so the user chooses by watching
5. **Vary BPM and key across scenes** — don't use the same preset for every scene
6. **Write sparser prompts** — "Upbeat indie rock" gives the model more creative freedom than a hyper-detailed description
7. **Vary seeds** — omit `--seed` to let each generation be unique

## Creating a Song (Step by Step)

### 1. Instrumental background track (simplest)
```bash
uv run tools/music_gen.py --prompt "Upbeat indie rock, driving drums, jangly guitar" --duration 60 --bpm 120 --key "G Major" --output track.mp3
```

### 2. Song with vocals and lyrics
Write lyrics in a temp file or pass inline. Use structure tags to control song sections.

```bash
# Write lyrics to a file first (recommended for longer songs)
cat > /tmp/lyrics.txt << 'LYRICS'
[Verse 1]
Walking through the morning light
Coffee in my hand feels right
Another day to build and dream
Nothing's ever what it seems

[Chorus - anthemic]
WE KEEP MOVING FORWARD
Through the noise and doubt
We keep moving forward
That's what it's about

[Verse 2]
Screens are glowing late at night
Shipping code until it's right
The deadline's close but so are we
Almost there, just wait and see

[Chorus - bigger]
WE KEEP MOVING FORWARD
Through the noise and doubt
We keep moving forward
That's what it's about

[Outro - fade]
(Moving forward...)
LYRICS

# Generate the song
uv run tools/music_gen.py \
  --prompt "Upbeat indie rock anthem, male vocal, driving drums, electric guitar, studio polish" \
  --lyrics "$(cat /tmp/lyrics.txt)" \
  --duration 60 \
  --bpm 128 \
  --key "G Major" \
  --output my_song.mp3
```

### 3. A weak section
Repaint and continuation aren't on the endpoints yet (see "repainting and continuation" below): make a few takes
with `--variations` and keep the best, or cut and crossfade two takes with ffmpeg.

### Key tips for good results
- **Caption = overall style** (genre, instruments, mood, production quality)
- **Lyrics = temporal structure** (verse/chorus flow, vocal delivery)
- **UPPERCASE in lyrics** = high vocal intensity
- **Parentheses** = background vocals: "We rise (together)"
- **Keep 6-10 syllables per line** for natural rhythm
- **Don't describe the melody in the caption** — describe the *sound* and *feeling*
- **Use `--seed`** to lock randomness when iterating on prompt/lyrics

### Controlling vocal gender
The model doesn't reliably follow "female vocal" or "male vocal" on its own. Use **both** of these together:
1. **In the prompt**: Be explicit — "solo female singer, alto voice" or "female vocalist only, breathy intimate voice". Adding an artist reference helps (e.g., "Brandi Carlile style").
2. **In the lyrics**: Add `[female vocal]` tags before each section:
```
[female vocal]
[Verse 1]
Walking through the morning light...

[female vocal]
[Chorus - anthemic]
WE KEEP MOVING FORWARD...
```
Just saying "female vocal" in the prompt alone is often ignored. The combination of prompt + lyrics tags is what works.

### Duets and vocal trading
For duets with male/female vocals trading verses, use both the prompt and per-section lyrics tags:
- **Prompt**: "duet, male and female vocals trading verses, warm harmonies on chorus"
- **Lyrics**: Tag each section with who sings it:
```
[Verse 1 - male vocal, storytelling]
First verse lyrics here...

[Chorus - male and female duet, harmonies]
Chorus lyrics here...

[Verse 2 - female vocal, wry]
Second verse lyrics here...

[Bridge - male vocal, spoken]
Spoken bridge...

[Bridge - female vocal, sung]
Sung response...
```
This reliably produces vocal trading between sections and harmonies on shared parts.

## Scene Presets

| Preset | BPM | Key | Use Case |
|--------|-----|-----|----------|
| `corporate-bg` | 110 | C Major | Professional background, presentations |
| `upbeat-tech` | 128 | G Major | Product launches, tech demos |
| `ambient` | 72 | D Major | Overview slides, reflective content |
| `dramatic` | 90 | D Minor | Reveals, announcements |
| `tension` | 85 | A Minor | Problem statements, challenges |
| `hopeful` | 120 | C Major | Solution reveals, resolutions |
| `cta` | 135 | E Major | Call to action, closing energy |
| `lofi` | 85 | F Major | Screen recordings, coding demos |

## Task Types

### text2music (default)
Generate music from text prompt + optional lyrics.

### cover
Style transfer from reference audio. Control blend with `--cover-strength` (0.0-1.0):
- **0.2** — Loose style inspiration (more creative freedom)
- **0.5** — Balanced style transfer
- **0.7** — Close to original structure (default)
- **1.0** — Maximum fidelity to source

### extract
Stem separation — isolate individual tracks from mixed audio.
Tracks: `vocals`, `drums`, `bass`, `guitar`, `piano`, `keyboard`, `strings`, `brass`, `woodwinds`, `other`

> Upstream lists extract as a **base-model** task (`acestep-v15-base` / `-xl-base`). The turbo DiTs
> the self-hosted endpoints serve (XL Turbo on Modal, 2B Turbo on RunPod) don't support it, so
> expect weak stems there.

### repainting and continuation (not on the endpoints yet)
Upstream ACE-Step 1.5 has both as tasks; the Modal and RunPod endpoints don't serve them yet. Until they do,
regenerate the take, or cut and crossfade two takes with ffmpeg.

## Prompt Engineering

### Caption Writing — Layer Dimensions

Write captions by layering multiple descriptive dimensions rather than single-word descriptions.

**Dimensions to include:**
- **Genre/Style**: pop, rock, jazz, electronic, lo-fi, synthwave, orchestral
- **Emotion/Mood**: melancholic, euphoric, dreamy, nostalgic, intimate, tense
- **Instruments**: acoustic guitar, synth pads, 808 drums, strings, brass, piano
- **Timbre**: warm, crisp, airy, punchy, lush, polished, raw
- **Era**: "80s synth-pop", "modern indie", "classical romantic"
- **Production**: lo-fi, studio-polished, live recording, cinematic
- **Vocal**: breathy, powerful, falsetto, raspy, spoken word (or "instrumental")

**Good**: "Slow melancholic piano ballad with intimate female vocal, warm strings building to powerful chorus, studio-polished production"
**Bad**: "Sad song"

### Key Principles

1. **Specificity over vagueness** — describe instruments, mood, production style
2. **Avoid contradictions** — don't request "classical strings" and "hardcore metal" simultaneously
3. **Repetition reinforces priority** — repeat important elements for emphasis
4. **Sparse captions = more creative freedom** — detailed captions constrain the model
5. **Use metadata params for BPM/key** — don't write "120 BPM" in the caption, use `--bpm 120`

### Lyrics Formatting

**Structure tags** (use in lyrics, not caption):
```
[Intro]
[Verse]
[Chorus]
[Bridge]
[Outro]
[Instrumental]
[Guitar Solo]
[Build]
[Drop]
[Breakdown]
```

**Vocal control** (prefix lines or sections):
```
[raspy vocal]
[whispered]
[falsetto]
[powerful belting]
[harmonies]
[ad-lib]
```

**Energy indicators:**
- UPPERCASE = high intensity ("WE RISE ABOVE")
- Parentheses = background vocals ("We rise (together)")
- Keep 6-10 syllables per line within sections for natural rhythm

## Video Production Integration

### Music for Scene Types

| Scene | Preset | Duration | Notes |
|-------|--------|----------|-------|
| Title | `dramatic` or `ambient` | 3-5s | Short, mood-setting |
| Problem | `tension` | 10-15s | Dark, unsettling |
| Solution | `hopeful` | 10-15s | Relief, optimism |
| Demo | `lofi` or `corporate-bg` | 30-120s | Non-distracting, matches demo length |
| Stats | `upbeat-tech` | 8-12s | Building credibility |
| CTA | `cta` | 5-10s | Maximum energy, punchy |
| Credits | `ambient` | 5-10s | Gentle fade-out |

### Timing Workflow

1. Plan scene durations first (from voiceover script)
2. Generate music to match: `--duration <scene_seconds>`
3. Music duration is precise (within 0.1s of requested)
4. For background music spanning multiple scenes: generate one long track

### Combining with Voiceover

Background music should be mixed at 10-20% volume in Remotion:
```tsx
<Audio src={staticFile('voiceover.mp3')} volume={1} />
<Audio src={staticFile('bg-music.mp3')} volume={0.15} />
```

For music under narration: use instrumental presets (`corporate-bg`, `ambient`, `lofi`).
For music-forward scenes (title, CTA): can use higher volume or vocal tracks.

### Brand Consistency

Use `--brand <name>` to load hints from `brands/<name>/brand.json`.
Use `--cover --reference brand_theme.mp3` to create variations of a brand's sonic identity.
For consistent sound across a project: fix the seed (`--seed 42`) and vary only duration/prompt.

## Advanced Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--variations N` | 1 | Generate N takes (1-8), saved as `<output>_1` to `<output>_N`; a given `--seed` moves on by one per take |
| `--seed` | random | Lock randomness for reproducibility |

## Technical Details

- **Modal**: XL Turbo 4B DiT (ACE-Step-1.5 v0.1.8), no LM, 8 steps at shift 3.0 (upstream's turbo recommendation; override with `shift` in the request), DCW sampler on. ~4s warm for 10s of audio, ~50s cold
- **RunPod**: Standard Turbo 2B DiT, no LM, ~2-3s per generation
- **Output**: 48kHz MP3/WAV/FLAC
- **Duration range**: 10-600 seconds
- **BPM range**: 30-300

### When NOT to use ACE-Step
- **Voice cloning** — use Qwen3-TTS or ElevenLabs instead
- **Sound effects** — use ElevenLabs SFX (`tools/sfx.py`)
- **Speech/narration** — use voiceover tools, not music gen
- **Stem extraction from video** — extract audio first with FFmpeg, then use `--extract`
