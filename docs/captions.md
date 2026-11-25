# Burned captions and subtitles

Word-level captions from the scene audio, and SRT or VTT subtitles. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Burned Captions (after voiceover)

```bash
uv run tools/captions.py projects/my-video/public/audio/scenes            # every scene MP3
uv run tools/captions.py projects/my-video/public/audio/scenes --engine whisper   # local, free
uv run tools/captions.py projects/my-video/public/audio/voiceover.mp3 --script narration.txt
```

Writes `<name>.captions.json` (word-level, `@remotion/captions` `Caption[]` shape) beside each
audio file. sprint-review, sprint-review-v2 and product-demo render it through the shared
`<Captions>` component (`lib/components/Captions.tsx`) as soon as the file exists, with karaoke
highlighting in the theme's primary colour. `captions: { enabled: false }` in the config turns them
off; any other `Captions` prop in that block restyles them (`fontSize`, `bottom`, `karaoke`, ...).

- **Captions spell what the script says, not what was heard.** The words of the sibling `.txt`
  (which `voiceover.py --scene-dir` leaves beside every MP3) are aligned onto the transcript's
  timing, so "Remotion" never becomes "remote". A match below 60% means the `.txt` is stale.
  `--script` must be narration only, markdown headings in it would become caption words.
- **Engines:** `scribe` (ElevenLabs, default when `ELEVENLABS_API_KEY` is set) or `whisper`
  (`uv sync --extra whisper`). Both just provide timing when a script is present.
- **Re-run after re-timing audio.** The JSON holds times relative to the audio file, so changing
  `durationSeconds` is fine, but regenerating or `--max-wpm`-clamping a scene's MP3 needs new captions.
- `align_captions.py` is different: it re-times hand-written `cap()` phrases in a config, rather than
  generating word captions.

`--srt` and/or `--vtt` also write sibling subtitles, splitting cues at sentence ends, pauses
over 0.6 seconds, and `--max-words 7`. For an existing caption JSON, export offline without
transcribing again: `uv run tools/captions.py voiceover.captions.json --srt --vtt`.
Outputs overwrite atomically. Each subtitle uses the same audio-relative timeline as its JSON;
scene captions need offsets before use as subtitles for a complete render.

## Word cues on joined audio

Transcribe the sentence clips individually when animating on a spoken word. Whisper can put the
first word of a sentence in the preceding silence when given the joined file.

```bash
uv run --extra whisper tools/captions.py audio/phrases --engine whisper \
    --joined audio/narration.captions.json --gap-ms 250
```

Pass the same sentence clips in the same order as the audio edit. A directory is read in filename
order, so use names such as `01-intro.wav` and `02-example.wav`. `--gap-ms` is the silence between
consecutive clips and defaults to zero. Use the same silence when joining the decoded audio.

The CLI shifts each clip's word timings using decoded sample lengths before rounding to milliseconds.
This keeps the cues on the joined clock when the clips are placed end to end with the chosen gap.
The timing of a word within its clip comes from the transcriber. `--srt` and `--vtt` export the joined
timeline too. If any clip fails, the command leaves the previous joined caption file in place.
