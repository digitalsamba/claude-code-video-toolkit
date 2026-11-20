# Music and sound effects

Music with ACE-Step, choosing between takes, mixing a bed under speech, and finding stock sound effects. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## AI Music Generation (ACE-Step 1.5)

Default provider is **acemusic** (official cloud API, free key from [acemusic.ai/api-key](https://acemusic.ai/api-key)). Uses XL Turbo 4B model with 5Hz LM thinking mode. Falls back to Modal (self-hosted XL Turbo 4B on ACE-Step-1.5 v0.1.8, no LM thinking) or RunPod (2B turbo).

```bash
# Background music (acemusic cloud API by default)
uv run tools/music_gen.py --prompt "Upbeat tech corporate" --duration 60 --bpm 128 --key "G Major" --output music.mp3

# Generate 4 variations (bg_1.mp3 ... bg_4.mp3), then recommend one against the video
uv run tools/music_gen.py --prompt "Subtle corporate tech" --duration 60 --variations 4 --output bg.mp3
uv run tools/pick_music.py bg_*.mp3 --target 58 --cuts 12,31,50 --under voice --fade-out 1.5 --pick bg.mp3

# Fast mode (disable thinking)
uv run tools/music_gen.py --no-thinking --prompt "Quick draft" --duration 30 --output draft.mp3

# Scene presets for video production
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
uv run tools/music_gen.py --preset tension --duration 20 --output problem.mp3
uv run tools/music_gen.py --preset cta --brand digital-samba --output cta.mp3

# Song with vocals and lyrics (use structure tags for sections)
uv run tools/music_gen.py \
  --prompt "Indie pop anthem, male vocal, bright guitar, studio polish" \
  --lyrics "[Verse]\nWalking through the morning light\nCoffee in my hand feels right\n\n[Chorus - anthemic]\nWE KEEP MOVING FORWARD\nThrough the noise and doubt\n\n[Outro - fade]\n(Moving forward...)" \
  --duration 60 --bpm 128 --key "G Major" --output song.mp3

# Cover / style transfer
uv run tools/music_gen.py --cover --reference theme.mp3 --prompt "Jazz piano version" --output cover.mp3

# Repaint a weak section (acemusic only)
uv run tools/music_gen.py --repaint --input track.mp3 --repaint-start 15 --repaint-end 25 --prompt "Guitar solo" --output fixed.mp3

# Continue from existing audio (acemusic only)
uv run tools/music_gen.py --continuation --input track.mp3 --prompt "Continue with jazz piano" --output extended.mp3

# Stem extraction
uv run tools/music_gen.py --extract vocals --input mixed.mp3 --output vocals.mp3

# Fall back to self-hosted
uv run tools/music_gen.py --cloud modal --prompt "Background music" --duration 60 --output bg.mp3

# List presets
uv run tools/music_gen.py --list-presets
```

**Choosing between takes:** `pick_music.py` measures, it can't listen. It rules out a take shorter than the video,
clipping, or with a gap mid-track, then prefers a calm bed under narration or text, some movement, a clean ending
(or one the video's own `--fade-out` covers) and a note or level change on the act cuts, and says why against the
runner-up. An agent with nobody to ask takes its pick and records the reasons.

**With a person, choose against the picture.** A take that sounds best alone may not be the one that sits best
under the video, and clean takes often tie on the measures. The picture doesn't depend on the music, so one render
serves every take: `audition.py` mixes each in with ffmpeg in seconds, at the composition's level and fades, and
normalised to -14 LUFS. Show its measured pick first, then ask the user to watch every render and choose.

```bash
# a narrated video: render once with the music file moved aside (the templates skip it), then
uv run tools/audition.py --video out/video.nomusic.mp4 --music public/audio/bg_*.mp3 --volume 0.15 --fade-out 1.5 \
    --recommend --cuts 12,31,50 --under voice                  # -> out/audition/video.bg_1.mp4 ...
# music only (the render's audio is the old bed): --drop-audio
```

8 scene presets: `corporate-bg`, `upbeat-tech`, `ambient`, `dramatic`, `tension`, `hopeful`, `cta`, `lofi`. See `.claude/skills/acestep/` for prompt engineering patterns and video production integration guide.

`addmusic.py --duck` lowers an existing music bed while speech is active and lets it recover in
pauses. Omit `--duck` for a fixed level. `--speech narration.wav` replaces the video's audio with
isolated narration; otherwise the video's audio controls ducking. Music loops to the video length.

```bash
uv run tools/addmusic.py -i input.mp4 -m music.wav --duck --events effects.json -o mixed.mp4
uv run tools/qc.py --render mixed.mp4 --fix
uv run tools/qc.py --render mixed.loudnorm.mp4
```

Effects use `[{"path":"chime.wav","atSeconds":1.5,"gain":0.5}]`, with paths relative to the JSON
(generate them with `sfx.py` or find a recording with `sfx_search.py`).
Attack/release controls (`--duck-attack`, `--duck-release`) use milliseconds. The peak limiter does
not set integrated loudness; use QC's normalized output for the -14 LUFS delivery target. Reruns
replace completed outputs atomically. `--dry-run` performs no writes or API calls.

## Stock Sound Effects (Openverse)

`sfx.py` generates a sound with ElevenLabs; `sfx_search.py` finds a real recording on Freesound through the
Openverse API. It's free and needs no key (20 requests/minute, 200/day).

```bash
uv run tools/sfx_search.py --query "camera shutter" --max-duration 3   # list matches
uv run tools/sfx_search.py --query "whoosh" --output public/audio/whoosh.mp3   # top match
uv run tools/sfx_search.py --id ID --output public/audio/whoosh.mp3    # one from the list
```

- **Licenses:** only `cc0,by` by default, the two that allow commercial use with no share-alike. `--license` widens
  it, and an `nc` or `nd` download prints a warning.
- **Credit CC BY sounds in the video.** Every download leaves a `<name>.attribution.txt` with the credit line.
- Don't filter on `category=sound_effect`: Openverse leaves `category` empty on Freesound records, so it returns
  nothing. The tool filters on `source=freesound` instead.
- Files are Freesound's HQ MP3 previews, fine for a mix. Search in English.
- Openverse copies licence data from Freesound and doesn't guarantee it: before commercial use, open the source
  link in the credit file and confirm the licence there.
