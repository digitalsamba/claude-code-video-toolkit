# claude-code-video-toolkit

This file provides guidance to Claude Code (claude.ai/code) when working with this video production toolkit.

## Overview

**claude-code-video-toolkit** is an AI-native video production workspace. It provides Claude Code with the skills, commands, and tools to create professional videos from concept to final render.

**Key capabilities:**
- Programmatic video creation with Remotion (React-based)
- AI voiceover generation with ElevenLabs or Qwen3-TTS
- AI music generation with ACE-Step 1.5 (text-to-music, vocals, covers, stems)
- Browser demo recording with Playwright
- Asset processing with FFmpeg

## Directory Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Guided workflows
├── tools/               # Python CLI automation
├── templates/           # Video templates
│   ├── sprint-review/   # Sprint review video template
│   └── product-demo/    # Marketing/product demo template
├── brands/              # Brand profiles (colors, fonts, voice)
├── projects/            # Your video projects go here (gitignored)
├── examples/            # Curated showcase projects (shared)
├── assets/              # Shared assets (voices, images)
├── playwright/          # Browser recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata & registry
```

## Registry

`_internal/toolkit-registry.json` is the canonical source for all skills, commands, tools, templates, components, transitions, and cloud endpoints — including their paths, status, options, presets, and env vars. Consult it for structured data. This file focuses on **workflow guidance, patterns, and knowledge** that the registry can't capture.

## Quick Start

**First-time setup (optional, ~5 minutes):**
```
/setup
```

Walks through cloud GPU, file transfer (R2), and voice configuration. Most features are free. Skip this if you just want to render videos with Node.js.

**Work on a video project:**
```
/video
```

This command will:
1. Scan for existing projects (resume or create new)
2. Choose template (sprint-review, product-demo)
3. Choose brand (or create one with `/brand`)
4. Plan scenes interactively
5. Create project with VOICEOVER-SCRIPT.md

**Multi-session support:** Projects span multiple sessions. Run `/video` to resume where you left off. Each project tracks its phase, scenes, assets, and session history in `project.json`.

**Or manually:**
```bash
cp -r templates/sprint-review projects/my-video
cd projects/my-video
npm install
npm run studio   # Preview
npm run render   # Export
```

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

## Templates

Templates live in `templates/`. Most are standalone Remotion projects; concept-explainer-short is pure Python. See registry `templates` section for the full list.

### sprint-review
Config-driven sprint review videos with theme system, config-driven content (`sprint-config.ts`), pre-built slides (Title, Overview, Summary, Credits), demo components (single video, split-screen), and audio integration.

### product-demo
Marketing/product demo videos with dark tech aesthetic, scene-based composition (title, problem, solution, demo, stats, CTA), animated background, Narrator PiP, browser/terminal chrome, and stats cards with spring animations.

### ds-release-notes
Synthetic release-notes videos for Digital Samba (nothing screen-recorded): room and account-center screens rebuilt
from source as static React, driven by the `Stage` cursor and camera, with sentence-level cloned narration picked by
pace and pitch. Ships as the finished release 650 build, so it renders as is; for a new release keep the kits and
replace `src/scenes/` and the voice phrases. Method in the `synthetic-ui-demo` skill.

### concept-explainer-short
9:16 vertical concept-explainer shorts (TikTok/Reels/YouTube Shorts). **Python/moviepy, not Remotion** — the whole video derives from `scenes.json` (per-scene narration + visual asset). Pipeline: `gen_vo.py` (per-scene TTS via voiceover.py, clone or built-in, `--max-wpm` pacing clamp) → `gen_captions.py` (whisper word timing force-aligned to script text, needs `uv sync --extra whisper`) → `build.py` (audio-anchored composite: Ken Burns on stills, boomerang-looped clips, burned karaoke caption pills, ducked music). Renders at every stage — placeholder cards before assets, silent before audio. Visuals follow the FLUX/Ideogram/LTX split: Ideogram cards at `1440x2560` for anything text-bearing, LTX b-roll at `576x1024` for motion.

## Blender shot specs (experimental, optional)

A general 3D tool, not a template: 3D scenes, character acting and product shots from a JSON spec: `tools/blender_render.py check|render spec.json`,
built by `lib/blender/` (a rigged cartoon cast, a set, shots by type, acting beats in seconds or on narration words
with `cue:`). `check` (about 5 s) verifies every camera by geometry (subject in frame, nothing in the way, a moving
lens clear of the cast; failing cameras are moved by a solver) and writes a labelled contact sheet to read before
rendering; sound is placed on the frames the motion was keyed on. `doctor` finds Blender and times a frame on
this machine (about 1.6 s at 1080p on an M4).

One spec gives a whole film, clips for another build, or both; the uses below mix freely:
- `render spec.json` is the whole film with sound (`examples/blender-disk-panic`).
- `render --clips` is one MP4 per shot for any build: a moviepy timeline, a `file` scene in a video job, an editor
  (`--prores`), or `cutdown.py` for a vertical cut.
- `render --clips --alpha --to <project>/public/blender` is one clip per shot with a transparent background and a
  `clips.json` manifest; `BlenderClip` and `blenderClip()` (`lib/components`) play them over a scene's own field,
  with type behind the character. Set the spec's `fps` to 30.
- `set.screen.video` puts a Remotion render on the laptop in the scene, and a `pull_out` shot starts with it
  covering the frame, so a flat scene hard-cuts to itself in 3D and the camera pulls back (`push_in` returns).
  Render that video at the spec's `fps`; another rate is a spec error.
- Set kind `studio` is a product stage with no character: a `phone` or a floating `card` showing a picture or a
  video, and an imported `model` (glTF, OBJ, FBX, STL), filmed with `product`, `orbit` and `reveal` shots that take
  their distance from the prop's size (`examples/blender-product-shot`).

**Cast:** a character is a JSON description (`body`, `face`, `hair`, `clothes`, `accessories`, each a named choice
or a hex colour, and a `seed` that fills what is left out), so an agent writes one from a sentence or a photo. Keep
a standing character in `kits/<name>/cast/<id>.json` and name it with `"from"`; fields beside `from` re-costume
it. All characters share one rig, so every verb works on each. They are caricatures in one cartoon style, not
likenesses. A mistake in a spec (an unknown face, pose, shot type or hair style) is a named error listing the
choices, before anything renders.

Use it when a video needs a consistent, directable character, a 3D reveal, a product or device shot, or acting
timing that LTX clips can't hold. **Suggest it while planning a video when it is installed**
(`uv run tools/blender_render.py doctor --no-bench` exits 0); without Blender nothing else changes, so don't
propose it unless the user asks for 3D. Blender runs headless through its own Python; a Blender MCP server is for trying a look in an open window,
and what it finds goes back into the spec. Needs Blender 4.2+ (optional): run on 5.3 on macOS and 4.2 LTS on Linux
(Cycles on a server with no GPU; EEVEE works there on software GL, slowly); Windows is not yet run. Format, verbs
and the shot grammar are in the `blender` skill, setup and timings in `docs/blender.md`, and there is an example
of each use: `examples/blender-disk-panic` (a film), `blender-product-shot` (a studio stage) and
`blender-in-remotion` (clips in a Remotion video).

## Brand Profiles

Brands live in `brands/`. Each defines visual identity:

```
brands/my-brand/
├── brand.json    # Colors, fonts, typography
├── voice.json    # ElevenLabs voice settings
└── assets/       # Logo, backgrounds
```

See `docs/creating-brands.md` for details.

## Shared Components

Reusable video components in `lib/components/`. See registry `components` section for the full list with descriptions. Import in templates via:

```tsx
import { AnimatedBackground, SlideTransition, Label } from '../../../../lib/components';
```

**Synthetic UI demos:** `Stage` (browser window, camera zoom/pan keyframes, cursor layer) and `Cursor` (eased
keyframe path, click ripples) drive product screens rebuilt as static React from the app's source, so nothing is
screen-recorded. Coordinates are canvas pixels (1440x810 by default) and every click is a frame number, looked up
from the narration with `cue('save')` (`lib/components/cues.ts`) so the scene follows the voice when it is
re-timed. For product walkthroughs and release-notes videos, prefer this to a Playwright recording when the app's
source is readable. The method and its gotchas are in `.claude/skills/synthetic-ui-demo/`. The reference builds
are `examples/ds-free-schedule-demo` and `examples/ds-release-649`, and their screen kits stay in the examples.

**Terminal demos:** `Terminal` (`lib/components/Terminal.tsx`) is the same idea for a CLI: commands type, output
prints (`kind: 'ok' | 'err' | 'dim'`) and menus select, each on a `cue()` frame, and the window scrolls once full.
Theme it with `terminalTheme(brand)`; `brands/video-toolkit` carries a terminal palette. Start a line's output after
`typedUntil(command)`, or the command is cut off mid-typing.

**Data charts:** `LineChart` and `BarChart` (`lib/components/DataChart.tsx`) draw charts as SVG in the brand
(`chartTheme(brand.colors, brand.fonts, 'dark' | 'light')`) instead of embedding a Matplotlib MP4. The line draws to
`keys` (`{ frame, x }` pairs, frames from `cue()`), so each stretch of data is drawn while the narration talks about
it; annotation bands and bar highlights take cue frames too, and `lineHead()` gives the same position for a counter.
`series` puts several lines on one chart with a legend; `Counter` counts a number up from a cue; `SplitBar` shows shares of a
whole (who did the work), and still labels a zero share so the narration can say it.
Prefer this inside Remotion videos; keep the `data-viz-chart` Matplotlib pattern for pure-Python builds. Reference:
`examples/data-chart-stars`.

## Python Tools

Audio, video, and image tools in `tools/`. See registry `tools` section for the full catalog with descriptions, options, presets, and env vars. Every tool supports `--help`.

```bash
# Setup (one-time) — uv creates .venv/ and installs all locked dependencies
uv sync
uv sync --extra whisper   # optional: karaoke captions (heavy, pulls in torch)
uv sync --extra modal     # optional: Modal CLI for self-hosted cloud GPU
```

Run every Python tool through the project environment with `uv run` — it auto-syncs the venv, so no activation is needed.

**Important: always invoke tools from the toolkit root directory.** When working inside a project (`projects/my-video/`), tool paths like `uv run tools/upscale.py` will fail because `tools/` is relative. Always use:
```bash
cd /path/to/claude-code-video-toolkit && uv run tools/upscale.py ...
```
This is especially critical for background commands where the working directory may not be obvious.

### Tool Categories

| Type | Tools | When to Use |
|------|-------|-------------|
| **Project tools** | voiceover, music, music_gen, sfx, sfx_search, sync_timing, qc | During video creation workflow |
| **Utility tools** | redub, addmusic, notebooklm_brand, locate_watermark, cutdown | Quick transformations on existing videos |
| **Cloud GPU** | image_edit, upscale, dewatermark, sadtalker, soulx, qwen3_tts, music_gen, flux2 | AI processing via RunPod or Modal (`--cloud runpod\|modal`; soulx is Modal-only) |
| **Publishing** | youtube_upload, youtube_stats | Upload a finished render to YouTube (use `/publish` for the guided workflow); see how it does |
| **3D (optional)** | blender_render | Character acting, 3D reveals and product shots from a shot spec; needs Blender installed |

Utility tools work on any video file without requiring a project structure.

### Voiceover Generation

Three TTS providers share one `voiceover.py` interface (`--provider`):
**elevenlabs** (default), **qwen3** (self-hosted, free), and **60db** (premium cloud).
Voice settings (`--stability`, `--similarity`, `--speed`) use a **unified 0-1 scale**
across all providers — for 60db they are auto-converted to its native 0-100 scale.

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

#### 60db (standalone)

`tools/sixtydb_tts.py` is the dedicated 60db tool (counterpart to `qwen3_tts.py`).
It exposes a `generate_audio()` used by `voiceover.py` and `redub.py`, plus a CLI.
Three transports all produce a finished audio file: `synthesize` (REST, default),
`stream` (NDJSON), and `websocket` (realtime; needs `pip install websocket-client`).

```bash
uv run tools/sixtydb_tts.py --text "Hello world" --output hello.mp3
uv run tools/sixtydb_tts.py --text "Hello" --transport stream --output hello.mp3
uv run tools/sixtydb_tts.py --list-voices          # GET /myvoices
```

Config: `SIXTYDB_API_KEY` (required), `SIXTYDB_VOICE_ID` (optional — falls back to
60db's default voice). Brands carry a `sixtydb` block in `voice.json` (voiceId +
settings). `redub.py --tts-provider 60db` uses 60db for the new voice while
transcription stays on ElevenLabs Scribe (60db has no STT); in `--sync` mode the
60db output is run back through Scribe to recover word timestamps.

### Clone Prosody (when a cloned voice sounds flat or rushed)

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

### Timing Sync (after voiceover)

```bash
uv run tools/sync_timing.py                          # Dry run comparison
uv run tools/sync_timing.py --apply                  # Update config (1s default padding)
uv run tools/sync_timing.py --apply --padding 1.5    # Custom padding
uv run tools/sync_timing.py --voiceover-json vo.json # Use voiceover.py output
uv run tools/sync_timing.py --json                   # Machine-readable output
```

Knows sprint-review v1/v2 and product-demo configs, and falls back to a **generic mode** for any
config that pairs `*AudioFile` with `*DurationSeconds` fields (e.g. `tickets[]` with intro/outro
narration) — so new template shapes work without touching the tool.

### Pre-render QC (before every render)

```bash
uv run tools/qc.py --project projects/my-video                       # assets, timing, pacing, loudness
uv run tools/qc.py --project projects/my-video --render projects/my-video/out/video.mp4
uv run tools/qc.py --project projects/my-video --fix                 # normalise scene VO loudness
uv run tools/qc.py --render projects/my-video/out/video.mp4 --fix    # writes video.loudnorm.mp4
```

Exit 1 on errors (narration cut off, a missing asset the template doesn't guard), so it can
gate a render. **Loudness is the check nothing else does:** the templates mix VO + music
without normalising, and renders land well under YouTube's -14 LUFS — which YouTube turns
*down* when too loud but never *up* when too quiet. Scene VO targets -16 LUFS (headroom for
music); the render targets -14 LUFS / -1 dBTP. Optional keys (`backgroundMusicFile`,
`chimeFile`, `screenshotFile`, `voiceoverFile`) are warnings, not errors: the templates skip
them with `getStaticFiles()`, so the render succeeds silently without them.

**`--render` also watches the picture** (a few seconds even for a 4-minute render; `--skip-frames` to
skip):
- **Still stretches:** more than 3s (`--max-still`) where nothing on screen changes. The comparison is per
  tile, so a moving cursor or a talking-head PiP counts as change. The last 3s may hold.
- **Black gaps:** black for 0.5s or more mid-video, usually a gap between sequences or media that didn't load.
- **Motion that stops dead:** part of the frame moving for 3s or more, then frozen for 2s or more while the rest of the
  video goes on. Usually an image-to-video clip that ran out and held its last frame; lengthen it with `tools/ltx_extend.py`
  rather than holding or looping it. The whole-frame still check misses it because something else is always moving.
- **Empty first frame:** background only. A warning on vertical and square renders, whose frame 0 is
  the Shorts/Reels cover, and info on landscape renders, which get a custom thumbnail.
- **Audio:** a missing or silent audio track is an error. `--allow-silent` makes it info, for a deliverable that is
  silent on purpose; the picture checks still run.

Still stretches are warnings, not defects: a hold the narration needs is fine, but give it a next action
(see `.claude/skills/remotion/motion-craft.md`).

**`--layout` measures the text itself** (Remotion projects; needs Node, `playwright/` installed and Chrome or
Playwright's Chromium). It bundles the project, opens it in a browser, seeks a frame a second with Remotion's own
hooks and reports, as warnings:
- text off the canvas or clipped by its own box, held across two samples (entrances and exits pass through);
- contrast under WCAG AA at the size the text is shown, grouped by colour pair, with a passing colour suggested;
- text that never reaches 2.6% of the short side (28px at 1080p), the smallest a phone viewer reads.

Text under `data-qc="product"` is the product's own UI (Stage marks its browser window; mark a hand-built phone or
tablet mockup the same way). It's reported separately, as info, without the overflow check, since the camera crops
it on purpose: the video can't fix the product's colours, but it's worth knowing where the UI is hard to read.
`data-qc-ignore` skips an element. About 10s to bundle, then ~0.15s a sample.

### Sound effects on generated movement

A sound laid over a generated clip (a sneeze over an LTX face, a thud when a jacket lands) is placed at a frame by
hand, and is easily half a second off the movement. Place it by measurement, then check the render:

```bash
# before rendering: where in the clip should the sound start? (its loudest moment meets the movement)
uv run tools/sync_check.py --clip public/people/theo-sneeze.mp4 --sfx public/audio/sfx/sneeze.mp3 --window 2.5:4.5
uv run tools/sync_check.py --clip public/people/theo-desk.mp4 --sfx public/audio/sfx/thump.mp3 --kind impact --window 1.5:3.8

# after rendering, before presenting: every cue in one go (exit 1 if any is off by more than 0.1 s)
uv run tools/sync_check.py --video out/video.mp4 --cues sync-cues.json
```

`--kind peak` (default) puts the sound on the biggest movement: a sneeze, a laugh. `--kind impact` puts it where a
movement stops: a landing, a stamp. `--kind onset` puts it where a sustained movement starts (a badge lifted
slowly), which has no single spike for `peak` to find. Give a `--window` around the action: an extended clip's seam
or a later gesture can be bigger than the one the sound is for. Keep the cues in the project's `sync-cues.json`
(`{name, sfx, at, region, kind, window}`, times in the render, regions as `w:h:x:y`; sound paths relative to the
cue file or the toolkit root) so the check re-runs on every render. Keep the window clear of the shot's dissolve and
any text that animates in near the movement.

### Burned Captions (after voiceover)

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
  `--script` must be narration only — markdown headings in it would become caption words.
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

### Qwen3-TTS (Standalone)

```bash
uv run tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3
uv run tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3
uv run tools/qwen3_tts.py --text "Hello" --instruct "Speak enthusiastically" --output excited.mp3
uv run tools/qwen3_tts.py --text "Hello" --ref-audio sample.wav --ref-text "transcript" --output cloned.mp3
uv run tools/qwen3_tts.py --list-voices   # 9 speakers: Ryan, Aiden, Vivian, etc.
uv run tools/qwen3_tts.py --list-tones    # neutral, warm, professional, excited, etc.
```

Temperature controls expressiveness: `--temperature 1.2` (more expressive) or `--temperature 0.4` (more consistent).

### Cloud GPU Providers

All cloud GPU tools support two providers via `--cloud runpod|modal`. RunPod is the default. Modal was added as a reliability fallback after RunPod outages, and offers faster cold starts.

```bash
# --- RunPod setup (automated, one-time per tool) ---
echo "RUNPOD_API_KEY=your_key_here" >> .env
uv run tools/image_edit.py --setup
uv run tools/upscale.py --setup
uv run tools/qwen3_tts.py --setup
uv run tools/music_gen.py --setup

# --- Modal setup (deploys + writes each endpoint URL to .env) ---
uv sync --extra modal && uv run modal setup
uv run tools/modal_setup.py upscale image_edit         # or --all (everything but ltx2)
uv run tools/modal_setup.py --list                     # deployed? .env in sync?
uv run tools/modal_setup.py --sync-env                 # deployed by hand? fill .env
# See docs/modal-setup.md for full guide
```

`modal_setup.py` reads URLs from Modal's API, never from the deploy log, which wraps and
truncates them. Gated apps (`ltx2`, `flux2` with `FLUX2_MODEL=klein-9b`) need a Modal
secret with `HF_TOKEN`. The apps default to the name `huggingface-token`, but Modal's
template names it `huggingface-secret`. The tool detects that and sets `HF_SECRET_NAME`;
`--hf-secret NAME` overrides it.

### AI Image Generation (FLUX.2 vs Ideogram 4)

The toolkit has **two** text-to-image generators. They barely overlap — the deciding factor is
**whether the image needs legible baked-in text**.

```bash
# FLUX.2 Klein — text-FREE backgrounds + image editing (self-hosted, free)
#   klein-4B by default (Apache-2.0). FLUX2_MODEL=klein-9b at Modal deploy time opts into
#   klein-9B (FLUX Non-Commercial, outputs need review) -- see docs/modal-setup.md
uv run tools/flux2.py --preset title-bg --brand digital-samba   # background for Remotion text overlay
uv run tools/flux2.py --prompt "Abstract tech background, no text"

# Ideogram 4 — legible IN-IMAGE text + exact color/layout (hosted API, ~$0.03-0.10/img, commercial-OK)
uv run tools/ideogram4.py --json caption.json --output title.png   # text baked into the image
uv run tools/ideogram4.py --prompt "Thumbnail: 'SHIP FASTER' bold" --output thumb.png
```

**The key distinction — baked-in text vs. background-for-overlay:**

- **FLUX.2 presets deliberately produce text-*free* backgrounds** (`title-bg`, `cta`, `thumbnail`
  all end with "no text, no words, no letters"). The intended pattern is **FLUX background →
  Remotion renders the text on top**, so the text stays editable, animatable per-letter, and
  frame-accurate. This is the right default for sprint-review **title cards / lower-thirds**.
- **Ideogram 4 bakes legible designed text *into* a flat PNG.** Pixel-perfect typography + exact
  hex colors in one shot, but static. Reach for it when **the text *is* the design**: YouTube/social
  thumbnails (static anyway), quote/stat cards, in-scene signage/logos, stylized text effects that
  Remotion overlays can't easily do. It fills a real gap — FLUX and LTX-2 both garble in-image text.

| Need | Use |
|------|-----|
| Animated/editable title text, lower-thirds, sprint-review title cards | **FLUX.2 bg + Remotion text** |
| Atmospheric/abstract backgrounds, problem/solution illustrations, presenter backdrops | **FLUX.2** presets |
| Finished graphic where typography is the design — thumbnails, quote cards, in-scene signage | **Ideogram 4** |
| Edit an existing photo (clothing, reframe, style, background) | **image_edit** (neither generator edits) |

Ideogram 4 chains with the processors like any generator: `ideogram4 → upscale.py` (crisp 4K),
`ideogram4 → ltx2.py --input` (animate the still), or drop the PNG straight into Remotion `<Img>`.
Ideogram 4 is generate-only here (no editing). See `.claude/skills/ideogram4/` for the JSON caption
format — Claude authors the caption as the "magic prompt" expander; needs `IDEOGRAM_API_KEY` in `.env`.
`--model 4.5` opts into Ideogram 4.5 on the v2 API (same flags; `--speed` maps to its quality
tiers). 4.5 has been run live: JSON captions render near-perfect text, plain prompts add
garbled filler text, so always use JSON. 4.0 stays the default because 4.5 sizes must be
multiples of 32, which rules out 1280x720 (`/thumbnail` relies on 4.0 for that exact size).

### AI Image Editing

```bash

# Image editing (Qwen-Image-Edit-2511, Apache-2.0; IMAGE_EDIT_MODEL=qwen-image-2.1 at Modal
# deploy time opts into Qwen-Image-2.1, non-commercial)
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses"
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal
uv run tools/image_edit.py --input photo.jpg --style cyberpunk
uv run tools/image_edit.py --input photo.jpg --background office
uv run tools/image_edit.py --list-presets  # Full preset list

# Upscaling (RealESRGAN)
uv run tools/upscale.py --input photo.jpg --output photo_4x.png --cloud runpod
uv run tools/upscale.py --input photo.jpg --scale 2 --model anime --face-enhance --cloud runpod
```

See `docs/qwen-edit-patterns.md` and `.claude/skills/qwen-edit/` for prompting guidance.

### AI Music Generation (ACE-Step 1.5)

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

### Stock Sound Effects (Openverse)

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

### Social Cut-downs (16:9 → 9:16 / 1:1 / 4:5)

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
- **Fitted modes shrink text.** In 9:16 a 1080p source shows at 56% scale, so burned captions
  get small. `--zoom 1.3` trades a little of each side for size. For a vertical cut of a
  captioned Remotion video, consider re-rendering it with larger captions first.
- **Shorts and Reels cap at 3 minutes** (TikTok 10). `--platform` warns past the limit; a
  longer upload just posts as a regular video. YouTube makes any vertical or square upload ≤3 min a Short,
  so `/publish` needs no special flag.
- `--overlay frame.png` lays a full-frame PNG (with transparency) on top, e.g. a branded header
  band made with Ideogram or by hand.

### Transcript highlights

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

### Watermark Removal

```bash
# Locate watermark coordinates
uv run tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/
uv run tools/locate_watermark.py --input video.mp4 --preset notebooklm --verify

# Remove watermark (RunPod)
uv run tools/dewatermark.py --input video.mp4 --region 1080,660,195,40 --output clean.mp4 --runpod
uv run tools/dewatermark.py --setup  # One-time setup
```

**Workflow:** grid overlay → note coordinates → verify with `--region` → remove with dewatermark.

**Local mode** requires NVIDIA GPU (8GB+ VRAM). Mac users should use `--runpod`.

### Talking Head Generation (SoulX-FlashHead Pro vs Lite vs SadTalker)

Two generators, and the deciding factor is **whether a viewer actually watches the shot**.

```bash
# SoulX-FlashHead Pro — the default. Diffusion, follows the input aspect ratio (Modal only)
uv run tools/soulx.py --image presenter_16x9.png --audio voiceover.mp3 \
  --size 768 --output narrator.mp4

# SoulX-FlashHead Lite: drafts only (long-take drift unmeasured). Same endpoint,
# ~8x faster per chunk, no compile
uv run tools/soulx.py --image presenter_16x9.png --audio voiceover.mp3 \
  --model lite --width 512 --height 288 --output draft.mp4

# SadTalker: warp-based, cheap and near-realtime drafts, and the option when Modal is unavailable;
# square crop unless --preprocess full
uv run tools/sadtalker.py --image presenter_16x9.png --audio voiceover.mp3 \
  --preprocess full --still --expression-scale 0.8 --output narrator.mp4
```

| Need | Use |
|------|-----|
| Anything a viewer watches — narrator in frame, held shot, finished video | **soulx** |
| Throwaway drafts, or many takes to choose between | **soulx --model lite** or **sadtalker** (Lite: drafts only; long-take drift unmeasured) |
| Non-square source image you don't want to fight | **soulx** (no `--preprocess` needed) |
| Modal unavailable (RunPod only) | **sadtalker** |

**Identity holds over long takes, which is why this is the default.** Segment-chained
talking heads re-anchor each segment on the previous segment's output, so the failure is
*absorbing*: one bad segment poisons everything after it. SoulX is trained with
Oracle-Guided Bidirectional Distillation against exactly that. Measured at **97% of
frame-zero sharpness at 70s**, flat across all 72 segments. There is no short-render
ceiling to design around, so per-scene generation is a choice rather than a workaround.

**Cost:** Pro ~$0.0024 per second of output, Lite ~$0.0004, SadTalker ~$0.0014. Lite is the
cheapest per second, but it is for drafts only: it has been tested on ~2.6s clips, its
long-take drift is unmeasured, and the 70s sharpness figure above is Pro's.

**Key flags:**
- soulx: `--size 768` — aspect follows the image and snaps to the model's grid
- soulx: `--width`/`--height` for exact dimensions; **Pro needs multiples of 16, Lite multiples
  of 32** (768x432 is illegal on Lite; 512x288 and 1024x576 are exact 16:9). The tool checks
  per model; nothing upstream does (an off-grid size floors silently and renders wrong)
- soulx: `--model lite` never compiles and is unloaded before any Pro render, so mixing the
  two in one container doesn't eat Pro's VRAM headroom
- sadtalker: `--preprocess full` — **Critical!** Preserves input dimensions (default `crop`
  outputs square)
- sadtalker: `--still` and `--expression-scale 0.8` — calmer, more professional look

**Image requirements (both):** Face 30-70% of frame, front-facing, 16:9 for NarratorPiP, 512px+.

**Gotchas that cost hours** (full list in `docs/soulx.md`):
- `torch.compile` is on upstream, costs ~600s on a cold container, and **recompiles on every
  resolution change**. Pick one narrator resolution per project; a batch at one size
  amortises it, switching size per scene pays it per scene.
- Width and height must be multiples of 16. `--size` handles this; exact dimensions do not
  forgive you.
- `--face-crop` is unconditionally square whatever the target size says, so it throws away a
  16:9 framing. Leave it off.
- Don't score talking-head quality with an automated similarity metric. Two have now given
  confidently wrong answers — one ranked highest a variant with a visible eye defect, the
  other plateaued straight through a total collapse. Sharpness-over-time plus a contact
  sheet, or human review.

See `docs/soulx.md` and `docs/sadtalker.md` for detailed options and troubleshooting.

### Redub Sync Mode

```bash
uv run tools/redub.py --input video.mp4 --voice-id VOICE_ID --sync --output dubbed.mp4
```

The `--sync` flag enables word-level time remapping — essential when TTS voice pacing differs from original. Without it, audio can drift 3-4+ seconds by the end.

**How it works:** Scribe transcribes original → TTS generates new audio with timestamps → segment mapping (15 words/segment) → FFmpeg variable speed per segment.

### NotebookLM Branding

Post-processes NotebookLM videos with custom branding. Solves the problem where redubbed TTS audio extends beyond the safe visual trim point.

```bash
uv run tools/notebooklm_brand.py \
    --input video_synced.mp4 \
    --logo assets/logo.png \
    --url "mysite.com" \
    --output video_final.mp4
```

Trims NotebookLM visuals, keeps full audio, bridges with freeze frame, adds branded outro.

### Agent Video Jobs (brief in, MP4 out)

`tools/video_job.py` is the async entrypoint for agents that can't babysit a render: `submit --spec spec.json
--max-usd 1.00` returns in seconds with a job id and estimate, a detached worker runs voiceover → visuals →
captions → build → `qc.py --fix`, and `status JOB_ID --json` / `result JOB_ID --json` read state from disk from
any fresh process. The spec is scenes with narration text and one visual each (`ltx`, `ideogram`, `flux`, `text`,
`file` for the person's own photo or clip, `none`), optionally naming a **kit** (`kits/README.md`: a brand, voice
and photo folder set up once in a session, so later videos are one prompt); `text` scenes fall back to a local
brand-colour background on a host without FLUX; the compositor is `concept-explainer-short` in 9:16 or 16:9. `estimate` is a pure function of the
spec and the module's `PRICES` table, `submit` refuses over the cap before any spend, one job runs at a time,
and receipts land in `logs/media-spend.log` as JSON lines (or the agent's workspace with `VIDEO_JOB_WORKSPACE`). A job that dies part-way keeps what
it bought: `resume JOB_ID` redoes only the missing outputs. `status` carries timestamped `events`, `warnings` and
an ETA calibrated per host, `submit` prunes old jobs' project trees, and renders are encoded under 9.5 MB so they
attach in Discord. Brief → spec needs an LLM: agents write
the spec themselves, or set `VIDEO_JOB_EXPANDER` to any command that turns a brief into one. Agents without a
shell get the same calls from `tools/video_mcp.py` (MCP over stdio, `uv sync --extra mcp`). The
agent-facing skill is `skills/video-toolkit/` (it opens with a routes table: which template or skill fits what the
user asks, and whether it is a one-prompt `job` or a working `session`; `video_job.py catalog` lists them from the registry); the full contract is in
`docs/agent-video-jobs.md`. When you add a tool, template, component, skill or design source to the registry,
mention it in that skill too, if an agent outside Claude Code could use it.

### Publishing to YouTube

Upload a finished render to YouTube via the Data API v3. Use the `/publish` command for the
guided workflow (it auto-fills title/description/tags from `project.json`), or call the tool directly.

```bash
# One-time login (opens a browser, caches a refresh token under _internal/.youtube/)
uv run tools/youtube_upload.py --auth

# Upload privately (the safe default)
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \
    --description-file DESCRIPTION.md --tags "ai,agents,explainer" --json-out

# Schedule a public go-live
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" \
    --publish-at 2026-06-10T09:00:00Z --thumbnail out/thumb.png --json-out

# Validate everything without uploading (also reports auth readiness)
uv run tools/youtube_upload.py --video out/video.mp4 --title "Test" --dry-run --json-out

# Setup check: channel, scopes, Data/Analytics API on or off, with the fix for each (read-only)
uv run tools/youtube_upload.py --check --account personal

# Subtitles and chapters from the render: one SRT for --captions, chapter lines for the description
uv run tools/publish_kit.py --project projects/my-video --video out/video.mp4
```

The publish kit transcribes the render once (Scribe, about $0.01 for 90 s, or `--engine whisper`) and aligns it to
the scene scripts, so the subtitles are spelled as written and timed to the render, with no scene offsets to add
up. Chapters sit at each scene's cut, titled from `project.json` `scenes[].title` or the file names, under
YouTube's rules (first at 0:00, three or more, 10 s each). Re-runs reuse the saved transcript.

**Setup is OAuth, not an API key** (uploads act on behalf of a channel). See
`docs/youtube-upload.md` for the one-time Google Cloud Console walkthrough and
`YOUTUBE_CLIENT_SECRETS_FILE` in `.env`. Key realities:
- Default quota is 10,000 units/day; `videos.insert` costs ~1,600 units → **~6 uploads/day**.
- **Unaudited API projects *may* have public uploads force-locked to private** — but this
  mainly affects uploads to *other* people's channels. First-party uploads (your own project +
  channel + account) generally publish public/scheduled fine without the audit. The tool reports
  the actual returned privacy. "Testing"-mode refresh tokens expire after ~7 days (re-run `--auth`).
- Cached tokens live in `_internal/.youtube/` (gitignored — they grant channel-upload access).

**How it did:** `uv run --extra youtube tools/youtube_stats.py --account personal [--github]` reads views, likes and
comments with the same token, adds watch time, retention and traffic sources once the token has the Analytics scope
(`--auth`), and snapshots each run to `_internal/.youtube/stats/` so the next one shows the change. Reading the
numbers, and what has worked on Shorts, is in `.claude/skills/youtube-stats/`.

## Public copy: the copy lint is the first filter

Everything this toolkit makes is public: narration and captions, on-screen text, titles, descriptions, thumbnails,
the site, the README, docs, skills, PR bodies and commit subjects (they become the release notes). Run new words
through `tools/copy_lint.py` before anyone sees them: it refuses dashes and stock AI or marketing phrasing (the
list is `COPY_RULES` in the tool), and says why.

```bash
uv run tools/copy_lint.py site/*.html DESCRIPTION.md --text "The title"   # files and strings, exit 1 on a hit
uv run tools/copy_lint.py --diff origin/main --commits origin/main        # a branch's new lines + commit subjects
```

It is wired in where text leaves: video jobs refuse it before spend, `youtube_upload.py` refuses it (`--allow-copy`
to override), `voiceover.py` warns before voicing, and the `Copy lint` CI check fails a PR on new lines or commit
subjects. Old text is fixed when it's next touched. Web pages also go through taste-skill's `lint.mjs` and
`inspect.mjs` when it's installed.


1. **Create/resume project** - Run `/video`, choose template and brand (or resume existing); it sets the design
   direction (`design.md`) before any scene is built
2. **Review script** - Edit `VOICEOVER-SCRIPT.md` to plan content
3. **Gather assets** - Record demos with `/record-demo` or add external videos
4. **Scene review** - Run `/scene-review` to verify visuals in Remotion Studio
5. **Design refinement** - `/design` refines a scene within `design.md` (or the "Refine" option in scene-review),
   changes the direction, or critiques the render
6. **Generate audio** - Use `/generate-voiceover` for AI narration
7. **Sync timing** - Run `uv run tools/sync_timing.py --apply` to update config durations
8. **Preview** - `npm run studio` in project directory
9. **Iterate** - Adjust timing, content, styling with Claude Code
10. **QC** - `uv run tools/qc.py --project projects/my-video` before rendering
11. **Render** - `npm run render`, then `qc.py --render out/video.mp4 [--fix]` for loudness
12. **Thumbnail** - Run `/thumbnail` for a 1280×720 Ideogram 4 thumbnail (hook phrase + brand colours)
13. **Publish** - Run `/publish` to upload the render to YouTube (metadata and thumbnail auto-filled from `project.json`)

## Project Lifecycle

Projects move through phases tracked in `project.json`:

```
planning → assets → review → audio → editing → rendering → complete
```

| Phase | Description |
|-------|-------------|
| `planning` | Defining scenes, writing script |
| `assets` | Recording demos, gathering materials |
| `review` | Scene-by-scene review in Remotion Studio (`/scene-review`) |
| `audio` | Generating voiceover, music |
| `editing` | Adjusting timing, previewing |
| `rendering` | Final render in progress |
| `complete` | Done |

See `lib/project/README.md` for details on the project system.

## Video Timing

Timing is critical. Keep these guidelines in mind:

### Pacing Rules
- **Voiceover drives timing** — Narration length determines scene duration
- **Reading pace** — ~150 words/minute (2.5 words/second) for standard narration
- **Demo pacing** — Real-time demos often need 1.5-2x speedup (`playbackRate`)
- **Transitions** — Add 1-2s padding between scenes
- **FPS** — All videos use 30fps (frames = seconds × 30)

### Speaking Rate Tiers

| Pace | WPM | Use When |
|------|-----|----------|
| Slow | 120-130 | Technical explanations, complex concepts |
| Standard | 140-160 | General narration, demos, overviews |
| Fast | 160-180 | Energetic intros, recaps, CTAs |

### Narration Density by Scene Type

| Scene Type | Duration | Narration Density | Notes |
|------------|----------|-------------------|-------|
| Title | 3-5s | 0-10% | Logo + headline, let visuals breathe |
| Overview | 10-20s | 70-90% | 3-5 bullet points, narration-heavy |
| Demo | 10-30s | 30-50% | Let the demo speak, narrate key moments only |
| Stats | 8-12s | 70-90% | Read out highlights, skip obvious numbers |
| Credits | 5-10s | 0-20% | Quick fade, maybe a closing line |
| Problem/Solution | 10-15s | 80-90% | Narration drives the story |
| CTA | 5-10s | 60-80% | Clear call to action, leave a beat at end |

### Word Count Budgeting

Before writing scripts, budget words per scene:

```
Target duration × 2.5 = word budget (at standard pace)
Pause seconds × 2.5 = words to subtract from budget

Example: 15s scene with a 1s pause
  15 × 2.5 = 37 words budget
  1 × 2.5 = 3 words for pause
  Available: ~34 words of narration
```

Use `[pause 1.0s]` markers in scripts. Each second of pause costs ~2-3 words from the budget.

### Timing Calculations
```
Script words ÷ 150 = voiceover minutes (estimate)
Raw demo length ÷ playbackRate = demo duration
Sum of scenes + transitions = total video
```

### When to Check Timing
- **During scene planning** — Budget word counts per scene before writing
- **After writing script** — Count words per scene, compare to budget
- **After generating audio** — Run `sync_timing.py` to compare actual vs estimated
- **Before rendering** — Ensure `durationInFrames` matches actual audio for each scene

### TTS Duration Drift (The Real Timing Problem)

TTS engines do NOT consistently produce 150 WPM output. In practice:
- **ElevenLabs** tends to compress pauses and speed through short sentences. A 50s script may produce 40-45s of audio.
- **Qwen3-TTS** varies by speaker and tone preset. Ryan at "professional" tone speaks ~10% faster than "warm."
- **Short scenes drift more** — a 5-second scene might be off by 30%, while a 30-second scene is off by 10%.

**The feedback loop after TTS generation:**

1. Generate per-scene audio files
2. Run `uv run tools/sync_timing.py` to compare actual vs config durations
3. Run `uv run tools/sync_timing.py --apply` to update config automatically
4. For demo scenes: recalculate `playbackRate = rawDemoDuration / actualNarrationDuration`
5. Re-preview in Remotion Studio before rendering

**Common drift patterns and fixes:**

| Problem | Symptom | Fix |
|---------|---------|-----|
| Audio shorter than scene | Dead air / awkward silence at end | Reduce `durationInFrames` to match audio |
| Audio longer than scene | Narration cut off | Increase `durationInFrames` or trim script |
| Demo too fast for narration | Viewer can't follow | Decrease `playbackRate` or cut narration |
| Demo too slow for narration | Waiting for demo to catch up | Increase `playbackRate` (1.5-2x typical) |
| Pauses lost in TTS | Script felt spacious, audio feels rushed | Add explicit `<break time="1s"/>` in SSML or extend scene padding |
| Cloned voice rushes | Scenes land >170 wpm; retakes also fast (temperature doesn't help) | Pace is inherited from the reference audio. Use `--max-wpm 165` (pitch-preserving atempo clamp, both voiceover.py and qwen3_tts.py; if a take still lands fast it hit the 0.85x floor, so add `--min-tempo 0.8`), and record clone references as 12-25s of varied sentences *at narration pace* |

**Pacing QC:** voiceover.py and qwen3_tts.py report `wpm` and a `pacing` label
(fast/slow/ok) per generated file — check these before composition. Comfortable
narration is 140-160 wpm; flag anything over ~170. `--max-wpm` turns the check
into an automatic fix.

### Fixing Mismatches
- **Voiceover too long**: Speed up demos, trim pauses, cut content
- **Voiceover too short**: Slow demos, add scenes, expand narration
- **Demo too long**: Increase `playbackRate` (1.5x-2x typical)
- **Demo too short**: Decrease `playbackRate`, or loop/extend

### Audio-Anchored Timelines (the prevention approach)

`sync_timing.py` is reactive — it fixes drift after the fact. You can prevent drift entirely by **generating the audio first, then anchoring visuals to known timestamps** instead of estimating durations upfront.

**The pattern:**

1. Write the script and split into per-scene segments
2. Generate per-scene VO files: `voiceover.py --scene-dir public/audio/scenes --json`
3. Read the actual durations from the JSON output
4. Anchor every visual element to absolute timestamps in the timeline

This is especially clean for Python/moviepy builds where each clip carries its own `start=` parameter:

```python
# Audio-anchored scene timeline (25s total):
#   Scene 1 tired      0.3 → 3.74  (audio 3.44s)
#   Scene 2 worries    4.0 → 8.88  (audio 4.88s)
#   Scene 3 introduce  9.1 → 11.90 (audio 2.80s)

text_clip("TIRED OF",     start=0.5,  duration=1.2)
text_clip("THIRD-PARTY",  start=1.0,  duration=1.8)
vo_clip("01_tired.mp3",   start=0.3)
vo_clip("02_worries.mp3", start=4.0)
```

The comment block at the top is the source of truth. Every `start=` references it. Drift is impossible because durations aren't being estimated — they're being read from the rendered audio.

**Trade-off vs. `<Series>`-style auto-chaining:**

| Approach | Best for | Downside |
|----------|----------|----------|
| Audio-anchored absolute starts | Tight ad-style edits, sub-30s spots, anything with exact timing | Manual bookkeeping when re-timing a scene |
| `<Series>` / auto-chained durations | Long-form sprint reviews where adjacent scenes flex | Drift compounds across the timeline; needs `sync_timing.py` to recover |

For Remotion projects you can mix the two: use `<Sequence from={...}>` with absolute frames for tight sections and let `<Series>` handle the rest. For pure-Python builds (`build.py` + moviepy), audio-anchored is the natural default.

## Key Patterns

### Animations (Remotion)
```tsx
const frame = useCurrentFrame();
const opacity = interpolate(frame, [0, 20], [0, 1], { extrapolateRight: 'clamp' });
```

### Sequencing
```tsx
<Series>
  <Series.Sequence durationInFrames={150}><TitleSlide /></Series.Sequence>
  <Series.Sequence durationInFrames={900}><DemoClip /></Series.Sequence>
</Series>
```

### Media

**Never use a raw HTML `<video>` tag** — Remotion requires its own component for frame-accurate rendering.

The toolkit's templates use `<OffthreadVideo>` and `<Audio>` from the `remotion` package. Stick with these for consistency with existing templates. Upstream Remotion now also documents `<Video>` and `<Audio>` from the newer `@remotion/media` package (see `.claude/skills/remotion-official/`) — those add trim props, pitch shifting via `toneFrequency`, and richer loop behavior. Migrate per-template only if you need those features; both APIs render correctly.

```tsx
<OffthreadVideo src={staticFile('demo.mp4')} />
<Audio src={staticFile('voiceover.mp3')} volume={1} />
<Audio src={staticFile('music.mp3')} volume={0.15} />
```

## Scene Transitions

The toolkit includes a transitions library at `lib/transitions/`. See registry `transitions` section for the full list with options and best-use descriptions.

### Using TransitionSeries

```tsx
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { glitch, lightLeak, zoomBlur } from '../../../lib/transitions';

<TransitionSeries>
  <TransitionSeries.Sequence durationInFrames={90}>
    <TitleSlide />
  </TransitionSeries.Sequence>
  <TransitionSeries.Transition
    presentation={glitch({ intensity: 0.8 })}
    timing={linearTiming({ durationInFrames: 20 })}
  />
  <TransitionSeries.Sequence durationInFrames={120}>
    <ContentSlide />
  </TransitionSeries.Sequence>
</TransitionSeries>
```

### Transition Options Examples

```tsx
glitch({ intensity: 0.8, slices: 8, rgbShift: true })      // Tech/cyberpunk
lightLeak({ temperature: 'warm', direction: 'right' })       // Warm celebration
zoomBlur({ direction: 'in', blurAmount: 20 })                // High energy
rgbSplit({ direction: 'diagonal', displacement: 30 })        // Chromatic aberration
```

### Timing Functions

```tsx
linearTiming({ durationInFrames: 30 })                                      // Constant speed
springTiming({ config: { damping: 200 }, durationInFrames: 45 })            // Physics bounce
```

### Transition Duration Guidelines

| Type | Frames | Notes |
|------|--------|-------|
| Quick cut | 10-15 | Fast, punchy |
| Standard | 20-30 | Most common |
| Dramatic | 40-60 | Slow reveals |
| Glitch effects | 15-25 | Should feel sudden |
| Light leak | 30-45 | Needs time to sweep |

Preview all transitions: `cd showcase/transitions && npm run studio`

See `lib/transitions/README.md` for full documentation.

## Design: video-design and design sources

How a video looks is split in two. A **design source** gives the direction (palette, type, an idea): any design
skill the user trusts (plug in your own), or ours. **`video-design`** (`.claude/skills/video-design/`) is the toolkit's
own video layer: it turns that direction into scenes and critiques the render. The project records its source in
`project.json` (`design.source`) and the decisions in `design.md` (format in `video-design/handoff.md`), so a
different source changes the file, not the code. Known sources are in the registry's `designSources`; taste-skill
is the recommended external one today, Anthropic's frontend-design is another, and "ours" (video-design's act
palettes with the brand) is the default that needs nothing installed. Plug in your own design skill, or use ours.

**Turning a scene in place:** `PageTurns` (`lib/components/PageTurns.tsx`) turns one scene into new pages, each on
its narration cue: a hard edge sweeps across with the next page behind it, and nothing fades. Use it when the same
thing is shown again in a new way and the change is the point (one chart in three design sources, a screen before
and after a fix). Between scenes, use `wipe()` from `@remotion/transitions`: a hard-edged cut is the default for an
act change in `video-design`, and the effect transitions below are for when the brand calls for them. Sources are installed by the user and never copied into the toolkit.

### Usage
- **During scene review** (`/scene-review`): choose "Refine"
- **Focused sessions** (`/design`): `/design title`, `/design cta`
- **Whole video:** `uv run tools/contact_sheet.py out/video.mp4` tiles one settled frame per scene for the
  critique in `video-design/critique.md`

### The rules that hold whatever the source says
- **Colour follows the story:** two or three fields, changing only at a narrative turn (a new act, not a new
  scene), neighbours differing in lightness, hard cuts between them.
- **One data accent:** the brand colour on the narrated bar, line or number; other data one neutral step, never a
  tint of the ground.
- **Type:** five sizes per video, titles on two lines at most (broken by hand), one title edge for every scene.
- **No** rows of identical tiles, gradients, glows, shadows at rest, eyebrow labels.

Learned on Lugh's work (2026-11): a colour per scene read as noise ("not 100% sure about the different colors per
scene"); three fields by act (brand orange, bone, near-black) made each turn in the story visible.

## Toolkit vs Project Work

**Toolkit work** (evolves the toolkit itself):
- Skills, commands, templates, tools
- Tracked in `_internal/ROADMAP.md`

**Project work** (creates videos):
- Lives in `projects/`
- Each project has `project.json` (machine-readable state) and auto-generated `CLAUDE.md`

Keep these separate. Don't mix toolkit improvements with video production.

## Documentation

- `docs/getting-started.md` - First video walkthrough
- `docs/creating-templates.md` - Build new templates
- `docs/creating-brands.md` - Create brand profiles
- `docs/optional-components.md` - Setup for optional ML-based tools (ProPainter, etc.)
