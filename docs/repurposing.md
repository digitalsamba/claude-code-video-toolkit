# Reusing a finished video

Social cut-downs, transcript highlights, redubbing and NotebookLM branding. Run every tool from the toolkit root with `uv run`. Each tool also has `--help`, and the registry lists its options.

## Social Cut-downs (16:9 → 9:16 / 1:1 / 4:5)

```bash
uv run tools/cutdown.py --input out/video.mp4 --platform shorts                 # whole video, 9:16
uv run tools/cutdown.py --input out/video.mp4 -f 9:16 -f 1:1 --start 1:05 --end 1:50
uv run tools/cutdown.py --input out/video.mp4 --mode crop \
    --segment 0:04-0:11 --segment 0:42-0:58@0.25                                 # highlight reel
```

Outputs land beside the input as `<name>-9x16.mp4` etc. `blur`, `crop` and `pad` need only FFmpeg.
`--mode track` follows a single face, or a textured subject selected with `--box x,y,width,height`
in source pixels at each segment's start. Install its optional dependency with `uv sync --extra vision`.
It supports the same `--platform`, repeated `-f` and `--segment` options as other modes:

```bash
uv run tools/cutdown.py -i input.mp4 --mode track --platform shorts -f 9:16 -f 1:1 --segment 4-11 --segment 42-58
```

Tracking holds the last good position during occlusion, ambiguous multiple faces or a hard scene cut;
before a subject is found it uses `--focus` (including per-segment `@FOCUS`). Split at cuts to start
tracking a new subject. `--track-smoothing` and `--max-pan-speed` limit camera movement. Constant FPS
is required. Reruns overwrite outputs via temporary files and `os.replace`; `--dry-run` writes nothing.

- **Pick the mode by content.** `blur` (default) fits the whole frame over a blurred copy, so
  nothing is cut: right for slides and screen recordings. `crop` fills the frame by cutting the
  sides: right for talking heads and single-subject shots; `--focus 0-1` (or `@FOCUS` per
  segment) picks what to keep. `pad` is `blur` over a solid colour (`--brand` uses `bgDark`).
- **A subject that crosses the frame.** Try `--mode track` first: it follows a face that is in frame when the
  segment starts. When the thing to keep is not the face (a mug set down, a laptop lid), or the subject walks in
  after the start, give the segment several focus values and the crop moves through them, evenly spaced in time:
  `--segment 6.5-14.1@0.57,0.7,0.4`. On a sit-down shot, track held the face and let the mug leave the frame as
  it was put down; the focus values kept both.
- **Fitted modes shrink text.** In 9:16 a 1080p source shows at 56% scale, so burned captions
  get small. `--zoom 1.3` trades a little of each side for size. For a vertical cut of a
  captioned Remotion video, consider re-rendering it with larger captions first.
- **Shorts and Reels cap at 3 minutes** (TikTok 10). `--platform` warns past the limit; a
  longer upload just posts as a regular video. YouTube makes any vertical or square upload ≤3 min a Short,
  so `/publish` needs no special flag.
- `--overlay frame.png` lays a full-frame PNG (with transparency) on top, e.g. a branded header
  band made with Ideogram or by hand.

## Transcript highlights

`highlights.py` selects complete sentences, keeps the last repeated take (`--keep first` overrides),
and shortens long pauses. It runs by default and overwrites its output. Use `--dry-run` to review
the JSON plan without writes. `--transcript` must cover the **whole source video timeline**;
per-scene captions are relative to their audio file and cannot be passed without offsetting them.
`--transcribe` uses the existing optional Whisper extra locally and may download model weights.

```bash
uv run tools/highlights.py --video input.mp4 --transcript words.json --seconds 45 --keyword launch --output highlights.mp4 --save-plan edit.json
```

The plan's `cutdownArgs` array contains repeated `--segment START-END` arguments. Pass them to
`cutdown.py --input input.mp4 --platform shorts` to crop the selected ranges directly from the
original video. Review the selection before publishing; keyword ranking is a first editing pass.

## Redub Sync Mode

```bash
uv run tools/redub.py --input video.mp4 --voice-id VOICE_ID --sync --output dubbed.mp4
```

The `--sync` flag enables word-level time remapping, essential when TTS voice pacing differs from original. Without it, audio can drift 3-4+ seconds by the end.

**How it works:** Scribe transcribes original → TTS generates new audio with timestamps → segment mapping (15 words/segment) → FFmpeg variable speed per segment.

## NotebookLM Branding

Post-processes NotebookLM videos with custom branding. Solves the problem where redubbed TTS audio extends beyond the safe visual trim point.

```bash
uv run tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --logo assets/logo.png \
    --url "mysite.com" \
    --output video_final.mp4
```

Trims NotebookLM visuals, keeps full audio, bridges with freeze frame, adds branded outro.
