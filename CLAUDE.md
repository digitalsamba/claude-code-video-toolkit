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

### lecture
A presenter at a chalkboard (experimental, needs Blender): a fitted character rendered as one transparent clip a
segment, standing at a board Remotion draws, with chalk written on the narration's words and windows the camera goes
into. A lecture is its script, `lecture.json` and the timeline `tools/lecture.py timeline` writes; the code is
`lib/lecture` and `tools/lecture.py`, so a new lecture copies none. `"opening": {}` starts it on the whole hall, a
segment's `"closes"` cut to a closer second camera on the presenter, and `"shorts"` cut vertical Shorts from
segments. The README is the walkthrough; `docs/blender.md` "A lecture" is the method.

### howto
Tutorial episodes about a tool, in chapters, with a 9:16 Short cut from each. One mark carries the series: a
terminal prompt whose chevron fills into a play triangle and whose caret becomes the playhead on a track with a
stop per chapter (the opening draws it, chapter cards move the playhead, the close rolls it back into a blinking
prompt). Scenes are `ledger`, `figure`, `bars`, `statement`, `session` (`TerminalSession` filling the frame),
`command` and `split` (who does what, with a count under each column), and every prop is a function of `cue`, so nothing is timed by hand. Numbers live in `facts.json` with
their source and the day they were checked. A `session` shows only what a real run printed: capture the
commands first, and mark in `facts.json` what is captured and what is quoted. `scripts/chapters.py` writes YouTube
chapters from the timeline (not `publish_kit.py`, which makes one per narration file). Ships as the toolkit's own
Getting started 1, "What it costs".

### concept-explainer-short
9:16 vertical concept-explainer shorts (TikTok/Reels/YouTube Shorts). **Python/moviepy, not Remotion** — the whole video derives from `scenes.json` (per-scene narration + visual asset). Pipeline: `gen_vo.py` (per-scene TTS via voiceover.py, clone or built-in, `--max-wpm` pacing clamp) → `gen_captions.py` (whisper word timing force-aligned to script text, needs `uv sync --extra whisper`) → `build.py` (audio-anchored composite: Ken Burns on stills, boomerang-looped clips, burned karaoke caption pills, ducked music). Renders at every stage — placeholder cards before assets, silent before audio. Visuals follow the FLUX/Ideogram/LTX split: Ideogram cards at `1440x2560` for anything text-bearing, LTX b-roll at `576x1024` for motion.

## Blender (optional)

`tools/blender_render.py check|render spec.json` makes 3D scenes, character acting, product shots and a logo ident
from a JSON shot spec. It needs Blender 4.2 or later and nothing else in the toolkit depends on it.

- Suggest it while planning a video only when it is installed (`uv run tools/blender_render.py doctor --no-bench`
  exits 0). Without Blender, don't propose it unless the user asks for 3D.
- Before proposing a character film, tell the person it is hours of rendering over a few sessions.
- The clay cast, the desk sets, the studio stage and clips in Remotion are settled. Characters fitted from a
  generated 3D model, the kitchen, the lecture hall, the living room and the pub are experimental.
- Characters are caricatures in one cartoon style, not likenesses. Reference photos of a real person stay on the
  machine and out of the repo.

Format, verbs and shot grammar are in the `blender` skill. Setup, timings and a summary of every use are in
`docs/blender.md`.

## Manim (optional, experimental)

`tools/manim_render.py doctor|render scene.py Scene --out clip.mp4` renders a Manim scene file as a clip: a
diagram that builds on the narration's words, a code change, a 2D ident from a brand's SVG, a looping background,
simple 3D geometry. `--alpha` gives a transparent WebM for a Remotion scene. It needs Manim on the machine and
nothing else depends on it; suggest it only when `doctor` exits 0 or the user asks. Scenes import `toolkit_manim`
(`lib/manim`) for cues, brand colours, loops and sound events. Read `docs/manim.md` before writing a scene: it
holds what the first tests found. Characters that act are Blender's job, not Manim's.

## Brand Profiles

Brands live in `brands/<name>/` (`brand.json`, `voice.json`, `assets/` and an optional `ident/`). See
`docs/creating-brands.md`.

A brand's **ident** is its mark in motion with one sound. Templates that read the slot (`templates/howto`, through
its `scripts/ident.py`) play it and keep their own 2D opening when it is empty, so no video depends on it. The
format is in `docs/creating-brands.md` ("Mark and ident"); `tools/mark_sheet.py` checks a mark, and
`examples/blender-ident` makes an ident from the mark's SVG (Blender, optional).


## Shared Components

Reusable Remotion components live in `lib/components/` and the registry lists them. For a product walkthrough or
release-notes video, prefer a synthetic UI demo (`Stage` and `Cursor`, screens rebuilt from the app's source) to a
Playwright recording when the source is readable. Inside Remotion, draw charts with `LineChart` and `BarChart`
instead of embedding a Matplotlib MP4. `Stage`, `Cursor`, `Terminal`, `TerminalSession`, the data charts and `cue()`
are described in `.claude/skills/remotion/components.md`. Without Node, `tools/data_video.py` renders a line or bar
chart video from a CSV or JSON spec in a brand's colours (Matplotlib and moviepy); the spec format is in
`examples/data-viz-chart/README.md`.

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

### Where each tool is documented

Read the file before using a tool in its row. Each holds the commands, the choices between tools and the gotchas.

| Task | Read |
|------|------|
| Voiceover, 60db, a cloned voice that is flat or rushed, pronunciation | `docs/voiceover.md` |
| Scene durations, word budgets, `sync_timing.py`, TTS drift, audio-anchored timelines | `docs/timing.md` |
| `qc.py` (project, `--render`, `--layout`, loudness), sound effects on generated movement | `docs/qc.md` |
| Burned captions, SRT and VTT | `docs/captions.md` |
| Cloud GPU setup, FLUX.2 or Ideogram 4, image editing, upscaling, watermark removal | `docs/image-tools.md` |
| Music, choosing a take, ducking, stock sound effects | `docs/music-and-sound.md` |
| Cut-downs to 9:16 or 1:1, highlights, redub, NotebookLM branding | `docs/repurposing.md` |
| Talking heads (SoulX is the default, SadTalker when Modal is unavailable) | `docs/soulx.md`, `docs/sadtalker.md` |
| Agent video jobs (`tools/video_job.py`, brief in, MP4 out) | `docs/agent-video-jobs.md` |
| YouTube upload, publish kit, stats | `docs/youtube-upload.md`, the `youtube-stats` skill |
| Channel art (banner, avatar, watermark) and what a real channel set taught | `docs/youtube-channel-art.md` |
| Running a series across sessions (a hub with state and approvals, a bible, a shared library) | `docs/series.md` |
| Diagrams, code changes, 2D idents and looping backgrounds drawn by Manim (optional) | `docs/manim.md` |
| Transitions | `lib/transitions/README.md` |

Rules that hold across them:
- Run `uv run tools/qc.py --project <project>` before every render and `qc.py --render <file>` after it. The
  templates do not normalise loudness, so `--fix` is what brings a render to -14 LUFS.
- An image that needs legible text in it goes to Ideogram 4. A background that Remotion puts text over goes to
  FLUX.2.
- Check a render before presenting it: freezes, black gaps, sound sync.
- When you add a tool, template, component, skill or design source to the registry, mention it in
  `skills/video-toolkit/` too, if an agent outside Claude Code could use it.

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

## Video Workflow

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

The speaking rate tiers, narration density by scene type, word budgets, the TTS drift table and audio-anchored
timelines are in `docs/timing.md`. Read it before writing a script and after generating audio.

## Remotion Media

**Never use a raw HTML `<video>` tag** — Remotion requires its own component for frame-accurate rendering.

The toolkit's templates use `<OffthreadVideo>` and `<Audio>` from the `remotion` package. Stick with these for consistency with existing templates. Upstream Remotion now also documents `<Video>` and `<Audio>` from the newer `@remotion/media` package (see `.claude/skills/remotion-official/`) — those add trim props, pitch shifting via `toneFrequency`, and richer loop behavior. Migrate per-template only if you need those features; both APIs render correctly.

```tsx
<OffthreadVideo src={staticFile('demo.mp4')} />
<Audio src={staticFile('voiceover.mp3')} volume={1} />
<Audio src={staticFile('music.mp3')} volume={0.15} />
```

## Design

A design source gives the direction (palette, type, an idea) and the `video-design` skill turns it into scenes and
critiques the render. The project records its source in `project.json` (`design.source`) and the decisions in
`design.md`. Sources, how to plug one in, and marks and idents are in `.claude/skills/video-design/sources.md`.
Between scenes, a hard-edged `wipe()` is the default for an act change. Judge a logo small with
`tools/mark_sheet.py` before building on it.

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
