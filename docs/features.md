# Everything in the toolkit

The reference the README points to: skills, commands, templates and the showcase table, transitions, brands,
the project system, the Python tools, cloud GPU, and how the folder is laid out. The always-current catalog is
[`_internal/toolkit-registry.json`](../_internal/toolkit-registry.json).

### Skills

Claude Code has deep knowledge in:

| Skill | Description |
|-------|-------------|
| **remotion** | React-based video framework: compositions, animations, rendering |
| **elevenlabs** | AI audio: text-to-speech, voice cloning, music, sound effects |
| **ffmpeg** | Media processing: format conversion, compression, resizing |
| **playwright-recording** | Browser automation: record demos as video |
| **video-design** | Visual design for video: colour by act, one data accent, type steps, a contact-sheet critique; takes direction from any design skill ([design sources](../README.md#design-sources)) |
| **qwen-edit** | AI image editing: prompting patterns and best practices |
| **ideogram4** | AI image generation with legible text in the image: title cards, thumbnails, exact brand colors |
| **acestep** | AI music generation: prompts, lyrics, scene presets, video integration |
| **ltx2** | AI video generation: text-to-video, image-to-video clips, prompting guide |
| **parody-short** | Comedy shorts about a group's own life or a shared topic, in a borrowed genre: in-jokes from the group's own chat and docs, costumed and animated photos or an invented cast, sound placed on the movement, vertical cuts |
| **blender** *(optional)* | 3D from a JSON shot spec: a rigged cartoon cast acting on cue, a scene that turns out to be on a laptop, an app on a phone or a 3D model on a studio stage; every camera checked before the render; as a film or as clips in any video. Needs [Blender](blender.md) and renders in minutes with a GPU. Characters fitted from a generated 3D model are experimental |
| **moviepy** | Python video composition: overlay text on LTX-2/SadTalker output, build.py-style projects |
| **runpod** | Cloud GPU: setup, Docker images, endpoint management, costs |

> The always-current catalog of skills, commands, tools, and templates lives in [`_internal/toolkit-registry.json`](../_internal/toolkit-registry.json).

### Commands

| Command | Description |
|---------|-------------|
| `/setup` | First-time setup: cloud GPU, file transfer, voice, prerequisites |
| `/video` | Video projects: list, resume, or create new |
| `/scene-review` | Scene-by-scene review in Remotion Studio |
| `/design` | Set a video's design direction (design.md) from a design source, refine a scene, or critique the render |
| `/brand` | Brand profiles: list, edit, or create new |
| `/template` | List available templates or create new ones |
| `/skills` | List installed skills or create new ones |
| `/contribute` | Share improvements: issues, PRs, examples |
| `/record-demo` | Record browser interactions with Playwright |
| `/generate-voiceover` | Generate AI voiceover from a script |
| `/redub` | Redub existing video with a different voice |
| `/voice-clone` | Record, test, and save a cloned voice to a brand |
| `/thumbnail` | Generate a YouTube thumbnail with Ideogram 4 (hook phrase + brand colours) |
| `/publish` | Publish a finished video to YouTube (metadata auto-filled from `project.json`) |
| `/versions` | Check dependency versions and toolkit updates |

> **Note:** After creating or modifying commands/skills, restart Claude Code to load changes.

### Templates

Pre-built video structures in `templates/`:

- **sprint-review**: Sprint review videos with demos, stats, and voiceover
- **sprint-review-v2**: Composable scene-based sprint review with modular architecture
- **product-demo**: Marketing videos with dark tech aesthetic, stats, CTA
- **concept-explainer-short**: 9:16 vertical TikTok/Reels/Shorts explainers (Python/moviepy, no Remotion)
- **ds-release-notes**: Synthetic release-notes videos: product screens rebuilt from source, scripted cursor and camera, sentence-level cloned narration (ships as the Digital Samba 650 build)
- **howto**: tutorial episodes about a tool, told as one conversation with an agent, in chapters, with a 9:16 Short cut from each
- **lecture** *(experimental, needs Blender)*: a presenter at a chalkboard as data, a fitted 3D character at a board Remotion draws, with chalk on the narration's words, an opening on the hall, closer shots and vertical Shorts

See `examples/` for finished projects you can learn from (newest first, scroll down to watch the toolkit evolve in reverse):

| Date | Demo | Description |
|------|------|-------------|
| 2026-11-20 | [blender-support](https://www.youtube.com/watch?v=DnsaPJOhHT8) | 4:43 introduction to Blender support, presented by a 3D model of the author at a chalkboard: the `lecture` template, with an opening on the hall, closer shots from a second 3D camera and narration-timed chalk (Blender, optional) |
| 2026-11-16 | [what-it-costs](https://www.youtube.com/watch?v=5Xp0GG_4qKU) | 2:35, the first Getting started episode: what the toolkit costs to run, in four chapters, each scene drawn on the words of its narration and every figure from a facts file with its source; the opening is the toolkit's own ident, made from its SVG mark in Blender, and a 9:16 Short is cut from each chapter (`templates/howto`) |
| 2026-11-14 | [just-a-quick-one](https://www.youtube.com/watch?v=CgxzhKQ0Afg) | 1:44 trailer for the toolkit: a 3D character in a kitchen acting from a JSON shot spec (the walk, the mug, the laptop lid and the sip are verbs in a file), every camera checked before a frame was rendered (Blender, optional) |
| 2026-11-12 | [made-with-itself](https://www.youtube.com/watch?v=3HjPatdeal4) | 93s video about the toolkit, made in the toolkit from one sentence typed into Claude Code: each stage (narrate, score, generate, compose, check) is the real command in the new `Terminal` component, every line its saved output, then what it made; narrated by Marilyn from the ElevenLabs voice library through the new `brands/video-toolkit`; the bill is measured ($0.33) and the Check scene shows what `qc.py` really found in the draft (`examples/toolkit-made-itself`) |
| 2026-11-09 | [design-sources](https://www.youtube.com/watch?v=AQLzseZ2aAY) | 46s: one chart designed three ways (the toolkit's own palettes, taste-skill, and a house style of your own), the page turning between them on the narration with `PageTurns`, then the render checked one frame per scene with `contact_sheet.py` |
| 2026-11-07 | [the-great-return](https://www.youtube.com/watch?v=YUqkQydQDh4) | 98s wildlife documentary about the return to the office, made with the `parody-short` skill and the same invented cast: mock-Latin lower thirds, a designed narrator voice, LTX clips lengthened with `ltx_extend.py` instead of frozen, every sound effect placed on its movement with `sync_check.py` ([vertical cut](https://youtube.com/shorts/e4HW-tvi8-8)) |
| 2026-11-07 | [youre-on-mute](https://www.youtube.com/watch?v=73JNiOJgx5k) | 76s horror trailer about the video call that never ends, from the `parody-short` skill: an invented cast of four (FLUX portraits animated with LTX through `LivingPhoto`), a call UI built in Remotion that misbehaves, a stinger after the title ([vertical cut](https://youtube.com/shorts/1hyDgC5Qg_o)) |
| 2026-11-04 | [data-chart-stars](https://demos.digitalsamba.com/video/data-chart-stars.mp4) | 31s chart video of the toolkit's GitHub star history, drawn as SVG by the new `LineChart` and `BarChart` components: the line is drawn while Lugh narrates each stretch, and the March spike's band and the monthly bars light on the words that name them, timed with `cue()` (`examples/data-chart-stars`) |
| 2026-11-01 | [the-agent-handoff](https://demos.digitalsamba.com/video/the-agent-handoff.mp4) | *An experiment in agent-first, one-prompt video, kept for the record rather than as a showcase.* 57s an agent made unattended from one Discord message: the agent wrote the script and the shot list, `video_job.py` voiced it with a cloned voice, generated four LTX-2 clips and a diagram card, built and QC'd it, and the Discord bot posted the link when it went live; $0.47, the fourth take after three pipeline fixes (`docs/agent-video-jobs.md`) |
| 2026-10-28 | [ds-release-650](https://demos.digitalsamba.com/video/ds-release-650.mp4) | 87s release-notes video built from the `ds-release-notes` template: six items across the room and account center, Lugh's cloned voice made livelier with `voice_prosody.py`, checked with `qc.py --render --layout` (`templates/ds-release-notes`) |
| 2026-10-21 | [ds-release-649](https://demos.digitalsamba.com/video/ds-release-649.mp4) | Release-notes video with no screen recording: three product UIs rebuilt from their source code, driven by a scripted cursor and camera, narrated by Lugh; re-rendered 2026-11-09 in the DS act palette from `video-design` (`examples/ds-release-649`) |
| 2026-10-21 | [ds-free-schedule-demo](https://demos.digitalsamba.com/video/ds-free-schedule-demo.mp4) | 26s synthetic product demo of scheduling a call, with a SoulX talking-head narrator tile (`examples/ds-free-schedule-demo`) |
| 2026-06-09 | [hallucinations-short](https://demos.digitalsamba.com/video/hallucinations_short.mp4) | 3:41 vertical explainer on AI hallucinations: Ideogram 4 cards, LTX-2 b-roll, burned karaoke captions, Qwen3-TTS voice *clone* (reference sample from [Pixabay](https://pixabay.com/sound-effects/search/voice%20overs/)) |
| 2026-06-09 | [sky-blue-short](https://demos.digitalsamba.com/video/short-blue-sky.mp4) | 52s "Why is the sky blue?": concept-explainer-short template showcase (`examples/sky-blue-short`), stock Qwen3-TTS voice |
| 2026-04-08 | [q2-townhall-stars](https://demos.digitalsamba.com/video/q2-townhall-stars.mp4) | GitHub star history time-lapse with animated chart and deadpan-to-excited commentary |
| 2026-04-08 | [q2-townhall-longarm-ad](https://demos.digitalsamba.com/video/q2-townhall-longarm-ad.mp4) | Super Bowl-style launch ad with dramatic Qwen3-TTS announcer and LTX-2 animated Lugh cameo |
| 2026-03-15 | [the-space-between](https://demos.digitalsamba.com/video/the-space-between.mp4) | AI-generated video essay: flux2 avatar, Qwen3-TTS voice, SadTalker animation |
| 2026-02-23 | [cortina](https://demos.digitalsamba.com/video/sprint-review.mp4) | Mobile platforms sprint review |
| 2026-01-25 | [schlumbergera](https://demos.digitalsamba.com/video/schlumbergera.mp4) | Android sprint review video |
| 2026-01-22 | [ds-remote-mcp](https://demos.digitalsamba.com/video/ds-remote-mcp.mp4) | Remote MCP server demo *(the jazz background music is a joke)* |
| 2025-12-10 | [digital-samba-skill-demo](https://demos.digitalsamba.com/video/digital-samba-skill-demo.mp4) | Product demo showcasing Claude Code skill |
| 2025-12-05 | [sprint-review-cho-oyu](https://demos.digitalsamba.com/sprint-review-cho-oyu.mp4) | iOS sprint review with demos |

Videos other people have made with the toolkit are in [Show and tell](https://github.com/digitalsamba/claude-code-video-toolkit/discussions/categories/show-and-tell). Share yours there: a link, the template or tools you used, and what it cost.

### Scene Transitions

The toolkit includes a transitions library for scene-to-scene effects:

| Transition | Description |
|------------|-------------|
| `glitch()` | Digital distortion with RGB shift |
| `rgbSplit()` | Chromatic aberration effect |
| `zoomBlur()` | Radial motion blur |
| `lightLeak()` | Cinematic lens flare |
| `clockWipe()` | Radial sweep reveal |
| `pixelate()` | Digital mosaic dissolution |
| `checkerboard()` | Grid-based reveal (9 patterns) |

Plus official Remotion transitions: `slide()`, `fade()`, `wipe()`, `flip()`

Preview all transitions:
```bash
cd showcase/transitions && npm install && npm run studio
```

See [lib/transitions/README.md](../lib/transitions/README.md) for full documentation.

### Brand Profiles

Define visual identity in `brands/`. When you create a project with `/video`, the brand's colors, fonts, and styling are automatically applied.

```
brands/my-brand/
├── brand.json    # Colors, fonts, typography
├── voice.json    # ElevenLabs voice settings
└── assets/       # Logo, backgrounds
```

Included brands: `default`, `digital-samba`

Create your own with `/brand`.

### Project Management System

Video projects are tracked through a multi-session lifecycle:

```
planning → assets → review → audio → editing → rendering → complete
```

Each project has a `project.json` that tracks:
- **Scenes**: What to show, asset status, visual types
- **Audio**: Voiceover and music status
- **Sessions**: Work history across Claude Code sessions
- **Phase**: Current stage in the workflow

The system automatically reconciles intent (what you planned) with reality (what files exist), and generates a `CLAUDE.md` per project for instant context when resuming.

See [lib/project/README.md](../lib/project/README.md) for schema details, scene status tracking, and filesystem reconciliation logic.

### Python Tools

Audio, video, and image tools in `tools/`:

```bash
# AI voiceover, ElevenLabs or self-hosted Qwen3-TTS (9 voices + cloning)
uv run tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json

# AI music (ACE-Step, free cloud API)
uv run tools/music_gen.py --preset corporate-bg --duration 120 --output music.mp3

# AI image generation (FLUX.2) and editing (Qwen-Image)
uv run tools/flux2.py --preset title-bg --brand digital-samba --cloud modal
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal

# AI video generation (LTX-2.3, text-to-video, image-to-video)
uv run tools/ltx2.py --prompt "A sunset over the ocean, cinematic" --cloud modal

# Talking head from a portrait + audio (SoulX-FlashHead)
uv run tools/soulx.py --image portrait.png --audio voiceover.mp3 --output talking.mp4
```

<details>
<summary><strong>All tools</strong>, sound effects, redub, upscaling, watermark removal, NotebookLM rebranding, in-image text…</summary>

```bash
# Generate voiceover (ElevenLabs)
uv run tools/voiceover.py --script script.md --output voiceover.mp3

# Generate voiceover (Qwen3-TTS, self-hosted, cheaper alternative)
uv run tools/voiceover.py --provider qwen3 --speaker Ryan --scene-dir public/audio/scenes --json
uv run tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3

# Generate background music (ElevenLabs)
uv run tools/music.py --prompt "Upbeat corporate" --duration 120 --output music.mp3

# Generate background music (ACE-Step, free cloud API, XL Turbo 4B model)
uv run tools/music_gen.py --preset corporate-bg --duration 120 --output music.mp3
uv run tools/music_gen.py --prompt "Dramatic cinematic" --duration 30 --bpm 90 --key "D Minor" --output reveal.mp3
uv run tools/music_gen.py --prompt "Upbeat indie rock" --duration 60 --variations 4 --output intro.mp3

# Generate sound effects
uv run tools/sfx.py --preset whoosh --output sfx.mp3

# Find real recorded sound effects (Creative Commons, free, no API key)
uv run tools/sfx_search.py --query "whoosh" --output sfx.mp3

# Redub video with different voice
uv run tools/redub.py --input video.mp4 --voice-id VOICE_ID --output dubbed.mp4

# Add background music to existing video
uv run tools/addmusic.py --input video.mp4 --prompt "Subtle ambient" --output output.mp4

# Rebrand NotebookLM videos (trim outro, add your logo/URL)
uv run tools/notebooklm_brand.py --input video.mp4 --logo logo.png --url "mysite.com" --output branded.mp4

# AI image editing (style transfer, backgrounds, custom prompts)
uv run tools/image_edit.py --input photo.jpg --style cyberpunk --cloud modal
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal

# AI image upscaling (2x/4x)
uv run tools/upscale.py --input photo.jpg --output photo_4x.png --cloud modal

# Remove watermarks (requires cloud GPU)
uv run tools/dewatermark.py --input video.mp4 --preset sora --output clean.mp4 --cloud modal

# Locate watermark coordinates
uv run tools/locate_watermark.py --input video.mp4 --grid --output-dir ./review/

# Generate talking head video from image + audio (SoulX-FlashHead)
uv run tools/soulx.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

# AI image generation (FLUX.2 klein 4B; klein-9B is a non-commercial Modal opt-in)
uv run tools/flux2.py --prompt "A sunset over mountains" --cloud modal
uv run tools/flux2.py --preset title-bg --brand digital-samba --cloud modal
uv run tools/flux2.py --list-presets

# AI video generation (LTX-2.3 22B, text-to-video + image-to-video)
uv run tools/ltx2.py --prompt "A sunset over the ocean, cinematic" --cloud modal
uv run tools/ltx2.py --prompt "Gentle camera drift" --input photo.jpg --cloud modal

# 3D shots from a JSON spec (optional; needs Blender 4.2+, and a GPU to render in minutes: see docs/blender.md)
uv run tools/blender_render.py doctor                                    # is Blender here, and how fast is it?
uv run tools/blender_render.py check examples/blender-disk-panic/spec.json    # cameras verified, contact sheet, ~5 s
uv run tools/blender_render.py render spec.json --clips --alpha --to projects/my-video/public/blender

# Publish a finished render to YouTube (OAuth 2.0 + Data API v3), contributed by @dascope (#29)
uv run tools/youtube_upload.py --auth                                   # one-time browser login
uv run tools/youtube_upload.py --video out/video.mp4 --title "My video" --privacy private --json-out
```
</details>


**Tool Categories:**

| Type | Tools | Purpose |
|------|-------|---------|
| **Project** | voiceover, music, music_gen, sfx, sfx_search, sync_timing, captions, align_captions, pick_takes, pick_music, audition, voice_prosody | Used during video creation workflow |
| **Utility** | redub, addmusic, cutdown, highlights, data_video, notebooklm_brand, locate_watermark | Quick transformations, no project needed |
| **Cloud GPU** | image_edit, upscale, dewatermark, sadtalker, soulx, qwen3_tts, flux2, music_gen, ltx2 | AI processing via Modal or RunPod |
| **Hosted API** | ideogram4, sixtydb_tts | Images with legible text, and a second hosted voice provider |
| **Checks** | qc, sync_check, contact_sheet, copy_lint | Before showing a render: freezes and motion that stops dead, loudness, layout, sound effects on their movement; and public words before they leave |
| **Publishing** | youtube_upload, publish_kit, youtube_stats | Upload a finished render to YouTube (or use `/publish`), write its title, description and chapters, and see how it did |
| **3D (optional)** | blender_render, lecture, reference_capture | Character acting, 3D reveals, product shots and logo idents from a shot spec, a lecture's presenter clips, and reference photos for a fitted character; `blender_render` does nothing unless Blender is installed |
| **Brand** | mark_sheet | A logo at every size, on every ground, in a circle and as a watermark, with warnings, before it is used |
| **Agents** | video_job, video_mcp | A brief in and an MP4 out, for an agent outside Claude Code (experimental) |

### Cloud GPU (Modal + RunPod)

8 AI tools run on cloud GPUs. Use `--cloud modal` (recommended) or `--cloud runpod` on any tool.

| Tool | What It Does | Est. Cost |
|------|--------------|-----------|
| `qwen3_tts` | AI text-to-speech (9 speakers, voice cloning) | ~$0.01 |
| `flux2` | AI image generation & editing | ~$0.02 |
| `image_edit` | AI image editing & style transfer | ~$0.03 |
| `upscale` | AI image upscaling (2x/4x) | ~$0.01 |
| `music_gen` | AI music generation (8 scene presets) | Free (acemusic) / ~$0.05 (self-hosted) |
| `soulx` | Talking head video from portrait + audio: holds identity over long takes | ~$0.0024/sec |
| `sadtalker` | Talking head video, warp-based: fast, cheap drafts | ~$0.10 |
| `ltx2` | AI video generation (text-to-video, image-to-video) | ~$0.23 |
| `ltx_extend` | Lengthen an LTX clip by continuing it from its last good frame, instead of holding or looping it | ~$0.08 per extension |
| `dewatermark` | Video watermark removal | ~$0.10 |

**Modal (recommended):** Each tool deploys from `docker/modal-*/app.py`, Modal builds and hosts the containers. $30/month free compute on the Starter plan, typical usage is $1-2/month. Run `/setup` to deploy all tools automatically.

**RunPod (alternative):** Uses pre-built Docker images from `ghcr.io/conalmullan/video-toolkit-*`. Pay-per-second, no minimums. Run `uv run tools/<tool>.py --setup` to create endpoints.

See [docs/modal-setup.md](modal-setup.md) and [docs/runpod-setup.md](runpod-setup.md) for details.

## Project Structure

```
claude-code-video-toolkit/
├── .claude/
│   ├── skills/          # Domain knowledge for Claude
│   └── commands/        # Slash commands (/video, /brand, etc.)
├── lib/                 # Shared components, theme system, utilities
│   ├── components/      # Reusable video components (11 components)
│   ├── transitions/     # Scene transition effects (7 custom + 4 official)
│   ├── theme/           # ThemeProvider, useTheme
│   └── project/         # Multi-session project system
├── tools/               # Python CLI tools
├── templates/           # Video templates
├── brands/              # Brand profiles
├── projects/            # Your video projects (gitignored)
├── examples/            # Curated showcase projects with finished videos
├── assets/              # Shared assets
├── playwright/          # Recording infrastructure
├── docs/                # Documentation
└── _internal/           # Toolkit metadata & roadmap
```
