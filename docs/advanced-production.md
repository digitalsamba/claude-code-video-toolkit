# Advanced production capabilities

These tools extend the existing project/build workflow. Commands below run from
the toolkit root. They preview by default; `--run` creates media. Existing files
are protected. Use new output names for revisions.

Approach comparisons and upstream licenses are in
[the research record](advanced-production-research.md).

## Speech-aware audio mixing

```bash
uv run tools/audio_mix.py --video demo.mp4 --speech narration.wav \
  --music music.wav --output out/mixed.mp4
uv run tools/audio_mix.py --video demo.mp4 --speech narration.wav \
  --music music.wav --output out/mixed.mp4 --run
```

Omit `--speech` to use the video's first audio stream. An isolated narration
track gives better results when the original already contains music. Music loops
to the video length. Speech controls a sidechain compressor, so music returns
between phrases. `--threshold`, `--ratio`, `--attack` and `--release` tune this
behavior. Attack/release are milliseconds. Gains are linear, not decibels.

Use `--events effects.json` for timestamped effects. Paths are relative to that
JSON file. Resolve narration phrases with `tools/cues.py`, then use their
`timeMs / 1000` as `atSeconds`.

```json
[{"path":"click.wav","atSeconds":1.25,"gain":0.6}]
```

This uses FFmpeg's sidechaincompress, adelay, amix and limiter filters. It copies
the source video stream and replaces its audio. It does not classify music or
separate voices automatically.

## Automatic highlights

```bash
uv run tools/highlights.py --video recording.mp4 --transcript words.json \
  --seconds 45 --keyword release --output out/highlights.mp4 --save-plan out/edits.json
# Add --run to render. Or replace --transcript with --transcribe --model base.
```

Accepts toolkit Caption[] milliseconds or Whisper/WhisperX `segments` in seconds.
Local Whisper uses the existing optional `whisper` extra. It may download model
weights on first use. No hosted transcription is invoked.

The tool ranks complete punctuated sentences by keyword matches and uncommon
terms, drops near-duplicate takes, selects within the duration budget, and keeps
source order. Long internal pauses are shortened between aligned words. A
segment without word alignment stays intact. An unfinished final sentence is
omitted. This is deterministic first-pass editing; inspect the exported edit
decisions for context and editorial quality. It cannot infer which claims are
interesting or correct. FFmpeg re-encodes cuts to preserve accurate boundaries.

## Subject tracking and reframing

```bash
uv sync --extra vision
uv run tools/reframe.py --video demo.mp4 --box 220,90,180,240 \
  --width 1080 --height 1920 --output out/portrait.mp4 --run
# Or use --faces instead of --box for automatic face detection.
```

CSRT follows the first-frame subject box. Face mode starts with the largest face,
then follows the nearest face; it is not face recognition. Both modes report
temporary losses and fail after `--max-lost` frames. Split footage at scene cuts
or reinitialize with another first-frame box. Neither mode can guarantee identity
through occlusion or reliably recognize the active speaker among several people.

The crop follows the tracked center with smoothing, clamps at image edges, and
preserves the target aspect ratio. `--smoothing 1` follows immediately. The source
must have constant FPS. Original audio is retained. The adjacent `.track.json`
contains frame boxes, loss markers and a source hash for reuse by compositing.

## Multi-speaker dubbing

```bash
uv run tools/dubbing.py import diarized-whisperx.json --output dub.json
# Edit dub.json with translated/replacement text and voice identities.
uv run tools/dubbing.py render --video source.mp4 --spec dub.json \
  --output out/dubbed.mp4 --run
```

WhisperX identifies speaker IDs, not people's names. Assign each ID a stable
`provider` and `voiceId`, or `speaker` for Qwen3. Existing providers are
`elevenlabs`, `qwen3`, and `60db`. Qwen3 supports `refAudio` plus `refText`.
Set `audio` on individual segments to use existing recordings without TTS.
Paths are relative to the spec. Review diarization and replacement text before
rendering. The tool does not translate text automatically.

```json
{"version":1,"speakers":{"A":{"provider":"qwen3","speaker":"Ryan"}},
 "background":"background.wav","segments":[
  {"id":"greeting","speaker":"A","startSeconds":1,"endSeconds":3,
   "text":"Welcome to the release."}]}
```

The renderer fits each utterance to its original slot with pitch-preserving
`atempo`, places it at the original timestamp, and mixes overlapping speakers.
It rejects tempo outside `--min-tempo .75 --max-tempo 1.5` instead of truncating
speech. Revise text or timing when a translation does not fit. This synchronizes
dialogue timing; it does not modify a presenter's mouth movements.

Provide an isolated `background` stem to retain music/ambience. The original
dialogue is never added back. Without a stem, gaps are silent. If Demucs is
installed in another environment, extract a stem explicitly:

```bash
uv run tools/dubbing.py separate --video source.mp4 --output background.wav \
  --python /path/to/demucs/python --run
```

Separation is approximate and can remove effects or retain speech leakage.
For automatic transcription and diarization, install WhisperX in its own
environment following its upstream instructions, set `HF_TOKEN`, accept its
selected pyannote model conditions, then run:

```bash
/path/to/whisperx/python tools/dubbing.py diarize --video source.mp4 \
  --output diarized-whisperx.json --model base --device cpu
```

This explicit analysis command downloads and runs local models. Keep the heavy
WhisperX/Demucs environments separate from the toolkit's core dependencies.
Preview/render tests use supplied utterances; paid TTS is only called by `--run`
when a segment lacks `audio`.

## Character consistency across shots

Define stable appearance, wardrobe, visual style and reference images in a
project-relative character spec:

```json
{"version":1,"seed":42,"style":"Soft daylight, watercolor",
 "characters":{"lea":{"description":"Short brown hair, round glasses",
   "wardrobe":"Blue jacket","references":["public/lea.png"]}},
 "shots":[{"id":"intro","characters":["lea"],"prompt":"Standing in a studio",
   "motionPrompt":"A small wave","imageCostUsd":0.05,"videoCostUsd":0.2}]}
```

```bash
uv run tools/characters.py lock --project projects/story --spec characters.json \
  --output projects/story/lookbook.lock.json
uv run tools/characters.py plan --project projects/story \
  --lock projects/story/lookbook.lock.json --save-plan projects/story/shots.plan.json
uv run tools/build.py --project projects/story --plan projects/story/shots.plan.json --json
# Add --run and an appropriate --budget-usd only when ready to call configured endpoints.
```

The lock records reference hashes and deterministic shot seeds. Changed images
require a new reviewed lock. FLUX receives the original character references for
every shot, with the same appearance/style constraints. An optional motion prompt
adds an LTX image-to-video task that depends on that shot's still. The current
FLUX endpoint accepts up to three references in total per shot. LTX uses Modal;
`--cloud` selects the FLUX provider. Asset history records generated outputs.

This conditions generation; it does not guarantee identity or prevent model
artifacts. Inspect each shot. Estimates are supplied by you, not provider quotes.
Unknown estimates block capped builds. Use a new `--output-dir` for revisions so
old generated assets stay intact. Preview and lock creation do not call providers.

## Reusable data videos

```bash
uv run tools/data_video.py --spec examples/data-to-video/chart.json \
  --output out/chart.mp4 --script-output out/chart-narration.txt
# Generate speech from that script with the existing voiceover tool, then:
uv run tools/data_video.py --spec examples/data-to-video/chart.json \
  --narration narration.wav --output out/narrated-chart.mp4 --run
```

Specs select `line` or `bar`, dataset, title, unit, dimensions, FPS and duration.
CSV and JSON use long rows with `period`, `series`, `value`; override names with
`columns`. Periods follow their first appearance, so sort source rows beforehand.
Each series must have one finite numeric value per period. Missing/duplicate
cells fail validation. Limits are 500 periods, 24 series, and 12000 rows.

Matplotlib animates the values and holds the final state. The optional narration
track determines duration. The generated script states first/last values and
absolute changes, without inventing trends or explanations. `--save-report`
writes the source hash, normalized data and narration text for audit. The sample
dataset is fictional. Review labels, units and source data before publishing.

## Source-backed content

```json
{"version":1,"script":"script.md","sources":{"report":{"path":"report.csv"}},
 "claims":[{"id":"may-users","text":"May had 120 users.","source":"report",
   "selector":{"row":0,"column":"users"},"expected":120,"display":"120"}]}
```

```bash
uv run tools/evidence.py lock --spec evidence.json --output evidence.lock.json
uv run tools/evidence.py check --spec evidence.lock.json --output out/evidence-report.json
```

Supported sources are local UTF-8 TXT, Markdown, CSV and JSON. Claims must quote
their exact script text. Evidence is either a `quote` from a document or a
`selector` for a CSV cell/JSON pointer, with `expected` value. CSV rows are
zero-based, excluding the header. JSON pointers follow RFC 6901. Numeric display
values use plain decimal notation, without thousands separators or unit scaling.

The lock records script and source hashes. Checks flag changed/missing files,
missing quotes, wrong values, unknown sources and script content with no claim.
`quote-found` means exact text was located, not that the source is correct.
Dataset matches and paraphrases produce review warnings because labels, units,
context and meaning still need judgment. `--strict` also fails on those warnings.
The tool never silently treats retrieval or matching numbers as proof of truth.
Reports use `findings` with error/warning levels, compatible with local review QC.

To check displayed storyboard content as well as narration, use
`"project":"project.json"` instead of `script`. Each claim then needs `sceneId`.
The default fields are `title`, `narration` and `onScreenText`; `textFields` can
select other plain-text scene fields. Missing coverage is reported per scene
with its start timestamp. This checks manifest content, not text embedded inside
images or arbitrary template code. To display findings in local review, write
the report to the project's `out/qc.json` or merge its findings into the build's
QC report.

## Moving backgrounds and tracked graphics

```bash
uv sync --extra vision
uv run tools/composite.py --foreground presenter.mp4 --background studio.png \
  --key '#00ff00' --output out/virtual-set.mp4 --run
# Replace --key with --alpha mask.mp4 for an existing matte, or:
uv sync --extra vision --extra matting
uv run tools/composite.py --foreground presenter.mp4 --background studio.mp4 \
  --rvm-model /path/to/rvm_mobilenetv3_fp32.onnx --loop-background \
  --output out/matted.mp4 --run
```

Chroma key uses FFmpeg. A supplied mask uses white for opaque, black for
transparent, and gray for soft edges; it must match foreground dimensions and
FPS. A still mask is also supported. RVM uses the official FP32 ONNX model,
processes frames sequentially, and passes its recurrent states into each next
frame. Model weights are never downloaded automatically. Review the upstream
GPL-3.0/model terms linked in the research record. RVM specializes in humans,
not arbitrary object removal. Adjust `--ratio` for detail versus processing cost.

Backgrounds can be still images or videos. They are resized to cover the frame
and cropped centrally. A short video fails unless `--loop-background` is set.
Foregrounds require constant FPS. Their audio is preserved; background audio is
not mixed into the result.

Add `--overlays labels.json --track out/portrait.track.json` to attach transparent
graphics to the boxes from `reframe.py`. The track must belong to the original
foreground video, not the reframed render. Overlay paths are relative to the
overlay spec. Use either `position: [x,y]` or `attachToTrack: true`:

```json
[{"path":"label.png","attachToTrack":true,"offset":[0,-8],
  "size":[160,48],"startSeconds":1,"endSeconds":4}]
```

Attached labels center just above the subject box. Off-frame portions are clipped.
Both held tracking boxes and errors remain visible in the original track report.
