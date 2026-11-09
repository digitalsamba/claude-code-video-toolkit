---
name: video_toolkit
description: Make short explainer videos from a brief with claude-code-video-toolkit — submit a scene spec, poll a job, get a QC-checked MP4. Also covers the individual tools (AI voiceover, images, clips, music, talking heads) for custom work.
metadata:
  openclaw:
    emoji: "🎬"
    skillKey: "video-toolkit"
    os: ["darwin", "linux", "win32"]
    requires:
      bins: ["uv", "ffmpeg"]
---

# Video Toolkit

Turn a brief into a finished MP4 with [claude-code-video-toolkit](https://github.com/digitalsamba/claude-code-video-toolkit).
This skill is written for **any agent runtime**: Claude Code, OpenClaw, Codex, Hermes, a bare API loop with a
`run_tool` call, or an MCP client. Nothing here depends on how your runtime backgrounds work, because the long
work runs in a detached job you poll.

## Who this is for

You are an agent: someone asks you for a video through a channel (a chat message, a Discord or Slack thread, an
MCP client) and you make it without a person at your terminal. You report back in that channel. The person asking
sees your messages and the finished video, never the toolkit.

## What the toolkit can make: pick a route first

Every route has one of two support levels. `catalog` gives it per template as `agent_support`.

- **`job`: one prompt.** `video_job.py` runs it unattended from one spec. You write the scenes; it voices,
  illustrates, captions, builds and QCs them; you post the MP4. Any runtime can do this, through the CLI or MCP.
- **`session`: a working session.** Copy a template or example into `projects/<name>/`, edit its code with the
  skill it names, render with Node, look at the result, fix it, render again. That needs a Claude Code (or
  Codex-style) session in the toolkit, which is what the toolkit is built around. The session can be a person's, or
  yours, if your runtime can run one headless with a shell and Node. Either way, the skill's review points (script,
  a contact sheet, the render) go back to the person who asked, in their channel, before you spend more. Tell them
  up front what the route needs from them (for a synthetic UI video: the app's source, the click path, the
  narration).

A route moves from `session` to `job` once its inputs fit in a message, a spec drives the build with no code
written, QC can judge the result, and there's a cost cap. Offer a `job` when one would do.

| The user asks for | Route | Support | Read first | See it |
|---|---|---|---|---|
| A short explainer on a topic (Shorts, Reels, a 16:9 explainer) from a brief | `concept-explainer-short` via `video_job.py` | job | this file | `examples/sky-blue-short` |
| A post or short video for someone who has a kit (a small business, a coach, a product team) | `video_job.py` with `"kit"`: their photos as `file` scenes, `text` scenes in their colours | job | `kits/README.md` | |
| A walkthrough, feature demo or release-notes video of an app whose source you can read | synthetic UI: `templates/ds-release-notes`, components `Stage`, `Cursor`, `cues` | session | `.claude/skills/synthetic-ui-demo/SKILL.md` | `examples/ds-release-649`, `examples/ds-free-schedule-demo` |
| The same, but you can't read the source | Playwright recording into `product-demo` | session | `.claude/skills/playwright-recording/` | |
| A marketing or product launch video | `templates/product-demo` | session | the template's README | `examples/digital-samba-skill-demo` |
| A sprint or monthly review | `templates/sprint-review`, `templates/sprint-review-v2` | session | the template's README | `examples/hello-world` |
| A data story: a number or a trend drawn as the narration talks | `DataChart` (`LineChart`, `BarChart`, `Counter`, `SplitBar`) in Remotion; Matplotlib + moviepy without Node | session | `lib/components/DataChart.tsx`, `.claude/skills/moviepy/` | `examples/data-chart-stars`, `examples/data-viz-chart` |
| A fun or holiday video about the user's own group (a release night as an action-show intro, a Christmas ad) starring their photos, or a spoof of a shared topic with an invented cast | Remotion shots + `LivingPhoto`, cast from `image_edit` then `ltx2.py --input` (`ltx_extend.py` for long shots), in-jokes from the group's own chat and docs | session | `.claude/skills/parody-short/SKILL.md` | |
| A sub-30 s ad spot or a brand stinger | moviepy, audio-anchored | session | `.claude/skills/moviepy/` | `examples/quick-spot`, `examples/ds-crt-stinger` |
| A presenter talking to camera | `soulx.py` (below), placed in a template with `NarratorPiP` | tool, or session | Talking heads, below | |
| Change an existing video: new voice, music, vertical cut, highlight reel, subtitles, watermark | `redub.py`, `addmusic.py`, `cutdown.py`, `highlights.py`, `captions.py --srt`, `dewatermark.py` | tool | QC, captions, cut-downs, below | |

`ds-release-notes` is built for one product's screens, but the method isn't: keep its `Stage` and `Cursor`
wiring and rebuild your own app's screens the way `.claude/skills/synthetic-ui-demo/` describes.

**The catalogue lives on disk.** `uv run tools/video_job.py catalog` (MCP: `video_catalog`) lists every template
with its demo videos and the skill that drives it, the example builds, components, brands, design sources and voice
presets, read from `_internal/toolkit-registry.json`. When a user asks "what can you make?", answer from it, with
the demo links, rather than from memory: the list grows each release. Skills under `.claude/skills/` are written
for Claude Code, but they are plain markdown: open `SKILL.md` in the folder and follow it.

**When asked "what can you make?"**, answer with the `job` routes first (what you can do from this message), then
the `session` routes and what each needs from them, each with its demo link from `catalog`.

### How it looks: brand, design source, transitions

- **Brand** (`brands/<name>/brand.json` colours and fonts, `voice.json` a voice): pass `"brand": "<name>"` in a job
  spec, or `--brand` to the image tools. `catalog` lists the brands on this host.
- **Design direction** for a session: `.claude/skills/video-design/` turns a direction into scenes (colour by act,
  type scale, one data accent) and critiques the render. The direction comes from a design source: `ours` (the
  default, video-design's own palettes, nothing to install), `taste-skill` (recommended, installed separately) or
  `frontend-design`. A project records its choice in `design.md`. After a render, `contact_sheet.py` tiles one
  frame per scene for the critique.
- **Turning a scene in place** (the same thing shown again in a new way): `PageTurns`. Between scenes, a hard
  `wipe()`; the effect transitions in `lib/transitions/` are for brands that call for them.
- **Other components** (`lib/components/`): `AnimatedBackground`, `SlideTransition`, `Label`, `Vignette`,
  `LogoWatermark`, `SplitScreen`, `Envelope`, `PointingHand`, `MazeDecoration`, `FilmGrain`, `Captions`. Remotion
  itself is documented in `.claude/skills/remotion-official/` (synced from upstream) and the toolkit's patterns in
  `.claude/skills/remotion/`.

The rest of this file is the job route, then the individual tools.

## Where the toolkit is

Every command below runs **from the toolkit root**. Find it, in this order:

1. `$VIDEO_TOOLKIT_ROOT` if set (also honoured by the toolkit's own scripts);
2. `$VIDEO_TOOLKIT_DIR` (an alias some runtimes set);
3. a `claude-code-video-toolkit/` directory under your workspace or home.

```bash
cd "${VIDEO_TOOLKIT_ROOT:-${VIDEO_TOOLKIT_DIR:-$HOME/claude-code-video-toolkit}}"
uv run tools/verify_setup.py --json     # prerequisites, keys and endpoints; [x] means ready
```

Never run a tool from inside a project directory: tools resolve paths relative to the toolkit root. If your
runtime can't `cd`, pass `--directory`: `uv run --directory /path/to/toolkit tools/video_job.py ...`.

**No shell at all?** Use the MCP server instead: `uv run --directory /path/to/toolkit --extra mcp tools/video_mcp.py`
exposes `video_estimate`, `video_submit`, `video_status`, `video_result` and `video_list_presets` over stdio.
Same flow as below, same JSON.

## The quick path: estimate → submit → poll → result

`tools/video_job.py` is one async entrypoint. `submit` returns in seconds with a job id; a detached worker runs
the whole pipeline (voiceover → visuals → captions → build → QC); every later call is short and reads state
from disk. That is how it fits a per-call time limit, and it survives you going away between turns.

### 1. Write the spec (you are the LLM)

A spec is a list of scenes, each with narration text and one visual. Budget about 2.4 words a second of
narration; a 45 s video is ~100 words in 3 to 5 scenes.

```json
{
  "title": "Why do we have leap years?",
  "format": "9:16",
  "voice": {"provider": "qwen3", "speaker": "Ryan", "tone": "warm", "maxWpm": 165},
  "scenes": [
    {"slug": "hook",
     "text": "Every four years, February gets an extra day. Why? Because a year isn't 365 days long.",
     "visual": {"kind": "ltx", "prompt": "Slow orbital shot of Earth turning in space, sunlight sweeping the oceans, cinematic, no text"}},
    {"slug": "quarter",
     "text": "Earth takes about 365 and a quarter days to orbit the sun. Ignore the quarter and the calendar drifts.",
     "visual": {"kind": "ideogram", "caption": { "...filled from the caption template below..." }}},
    {"slug": "cta",
     "text": "So every fourth year we add one day back. Follow for one concept a week.",
     "visual": {"kind": "ideogram", "caption": { "...filled from the caption template below..." }}}
  ]
}
```

- `format`: `9:16` (Shorts, Reels, TikTok) or `16:9` (YouTube, decks).
- `visual.kind`: `ltx` (5 s motion b-roll, prompt the motion and atmosphere, never on-screen text),
  `text` (constructed: a generated background that drifts, with `headline` and optional `subline`
  animated in; about a cent), `ideogram` (a designed card with legible text: **always a `caption`
  object**, filled from the template below), `flux` (text-free generated background), `file` (the
  person's own photo or clip: `"path"` in their kit's assets or your workspace, optional `headline`;
  free), `none` (placeholder, drafts only). A scene with no visual gets a background. Alternate motion
  and constructed scenes so there is a pattern interrupt every ~15 s.
- `backgrounds`: where `text` scenes and scenes with no visual get their background. `auto` (default) generates
  one with FLUX when this host has it, else draws a flat brand-colour field with grain here (free, no GPU);
  `flux` or `local` to choose.
- `kit`: a kit name. A kit is set up once in a working session for one person or product (`kits/README.md`): a
  brand, a voice, a format and a folder of their own photos and clips. The spec fills those in from it, and
  `file` paths resolve in its assets. `catalog` lists the kits on this host, each with its assets and what each one
  shows; for someone with a kit, a request in a chat becomes a few `file` and `text` scenes, each `file` chosen
  from that list to fit its sentence, with no code written. The kit's `focus` keeps an off-centre subject in a
  crop (a landscape clip in 9:16 keeps only its middle third); a scene can set `"focus": 0-1` itself.
  Prefer assets whose `usedRecently` is 0, so two videos for one person don't look the same.
- **Copy is checked before any spend.** Dashes and stock AI phrasing ("it's worth noting", "delve",
  "in today's world") in the narration, title, headlines or sublines are refused with the line named, and so is
  narration copied word for word from an earlier finished video on this host. Rewrite the named lines and submit
  again; `"allowCopy": true` only when the person asked for the same words. The same rules
  (`tools/copy_lint.py`) apply to every other word you send out: the message with the video, a title, a
  description. Run `uv run tools/copy_lint.py --text "..."` on them first.
- `voice`: an object as above, or a preset name: `assets/voices/<name>.json` (a cloned voice; its reference
  audio is gitignored, so the host needs its own copy) or a brand with a `voice.json`. `catalog` lists them. The title animates
  over the first scene and a soft music bed is mixed in by default; `"titleCard": false` and
  `"music": false` turn them off.
- **No JSON at all:** `--scene "narration :: ltx: motion prompt"` per scene, `--scene "narration ::
  text: HEADLINE ; subline"`, `--scene "narration :: file: candle.mp4 ; HEADLINE"`, `--title`, `--voice <preset>`,
  `--kit <name>` (MCP: `kit`; it works with `--brief` too, and the expander is given the kit's assets). See the
  docs page for the full flat form.
- Scene 1 must hook in one sentence. Scene N is the call to action. Captions are burned from the script text,
  so write every scene exactly as it should appear.
- Optional: `"music": {"enabled": true, "prompt": "..."}` (off by default), `"brand": "digital-samba"`,
  `"ideogram": {"speed": "TURBO"}` to halve image cost, `"captions": {"enabled": false}`.
- `uv run tools/video_job.py presets` prints the full price table, formats and an example spec.

#### The caption template (every `ideogram` scene)

Ideogram renders text reliably only from a structured JSON caption. A plain `prompt` works but adds small
filler text to the card ("A REMINDER", a fake tagline), so fill this in instead. Replace the `<...>`
placeholders and leave everything else. The headline is under 6 words, all caps. `bbox` is
`[top, left, bottom, right]` on a 0 to 1000 grid, so the same numbers work for 9:16 and 16:9.
`uv run tools/video_job.py presets --json` prints it as `ideogram_caption_template`.

```json
{
  "high_level_description": "A <dark-navy poster> with the headline <HEADLINE>, showing <one simple diagram or icon>, with clear space in the bottom fifth of the frame.",
  "style_description": {
    "aesthetics": "clean, diagrammatic, modern explainer",
    "lighting": "flat even lighting, soft glow on the main object",
    "medium": "flat vector graphic design",
    "art_style": "minimalist infographic poster, generous negative space",
    "color_palette": ["#0A1428", "#4DA8FF", "#FFB347", "#FFFFFF"]
  },
  "compositional_deconstruction": {
    "background": "Deep navy gradient, faint tiny stars.",
    "elements": [
      {"type": "text", "bbox": [80, 80, 200, 920], "text": "<HEADLINE>",
       "desc": "Bold condensed sans-serif headline, all caps, white, centered at the top.",
       "color_palette": ["#FFFFFF"]},
      {"type": "obj", "bbox": [280, 150, 760, 850],
       "desc": "<One object: what it is, its shape, its colour. No text on it.>",
       "color_palette": ["#4DA8FF", "#FFB347"]}
    ]
  }
}
```

Rules of thumb: one headline and one object per card; describe the object as shapes and colours, not as a
concept (for a diagram, write the object as a chain, `"object": "prompt -> agent -> video"`, and the toolkit
lays out labelled boxes and arrows); keep the bottom fifth empty because the burned captions sit there; change `color_palette` only
when the brand asks for it. The full caption format, for cards with more elements, is in the toolkit's
`.claude/skills/ideogram4/`.

Have only a brief and want the toolkit to write the spec? Set `VIDEO_JOB_EXPANDER` to any command that reads
`{"brief","format","duration_s",...}` on stdin and prints a spec JSON (`claude -p`, a curl to any model, your own
script). Then `--brief "..."` works everywhere `--spec` does. Without it, `estimate --brief` still answers with a
skeleton estimate; `submit --brief` is refused.

Runtimes that pass an argv list and can't write files (an allowlisted `run_tool`, most MCP
bridges) take the spec inline: `--spec-json '<json>'`, with the short caption form
`{"headline": "...", "object": "...", "description": "..."}` per ideogram scene so it fits in one
argument. Add `--output files/<dated-name>.mp4` when your runtime serves a `files/` directory at
a public URL; the file lands there once the job is done.

### What this host can make

Not every host has every provider. `catalog` and `presets` carry `host`: the visual kinds, voices and music
providers this host has keys or endpoints for (`host.available_visual_kinds`), the local tools it has
(`host.local`: ffmpeg, Node, the Python packages), and which route levels it can run (`host.routes.job`,
`host.routes.session`: a session also needs Node 18+). Offer only those. `estimate`
lists anything the spec uses that's missing as `host_missing`, and `submit` refuses it (exit 2, `error_code:
unavailable`, with `missing`: what, which env vars, and an alternative) before anything is written or spent. An
entry marked `"soft": true` (a music bed with no music provider) isn't refused: the video is made without it.

With no GPU provider at all, a job still runs: `text` scenes and scenes with no visual get local backgrounds,
`file` scenes use the person's own photos, and `ideogram` cards need only an Ideogram key. A voice needs ElevenLabs
or 60db (a key), or qwen3 on Modal or RunPod. `ltx` needs Modal; a scene that names `flux` needs FLUX. Music is
optional: without a provider the job runs without a bed and says so. Setting a host up: `uv run tools/verify_setup.py`, then `modal_setup.py --all`.

### 2. Estimate, then submit under a cap

```bash
uv run tools/video_job.py estimate --spec spec.json --json
# {"estimate_usd": 0.35, "breakdown": [...], "scenes": 3, "video_s_approx": 31.0, ...}

uv run tools/video_job.py submit --spec spec.json --max-usd 1.00 --json
# {"id": "20261031-230304-f34a", "phase": "queued", "estimate_usd": 0.353, "poll": "..."}
```

`submit` is refused (exit 2, `error_code: budget_exceeded`) when the estimate is over `--max-usd`, before any
spend. Only one job runs at a time: a second `submit` while one is running exits 3 with `error_code: lock_held`
and the running job's id. The worker also stops itself before the next paid step if actual spend would pass the
cap.

### 3. Poll status

```bash
uv run tools/video_job.py status JOB_ID --json
```

Poll every 20 to 30 s. `phase` walks `queued → voiceover → visuals → captions → build → qc → publish → done`,
or ends in `failed` (with `error` and `error_code`) or `cancelled`. `progress` is 0 to 100, `eta_s` is a rough
seconds-left figure (say "about 3 minutes to go", not a percentage), `actual_usd` is spend so far,
`step_detail` names the asset being generated, `events` is a timestamped list of what started and finished
(each scene voiced with its wpm, each clip or image made with its seconds and cost, QC, the render size), and
`log_tail` is the last few log lines. Use the last two or three `events` to say what is happening ("two images
done, the clip for scene 3 is rendering") rather than naming a phase. `eta_s` is calibrated from this host's
recent jobs. A typical 30 s video takes 5 to 10 minutes; the LTX clip is the slow step (2 to 4 minutes each,
more on a cold start).

Tell the user the id and the estimate as soon as `submit` returns. Do not promise to keep watching from inside
one turn: the job carries on without you, and `status` works from any fresh process. If your runtime has a
scheduler or a poller, point it at `status`; otherwise check when you are next invoked.

### 4. Collect the result

```bash
uv run tools/video_job.py result JOB_ID --json
# {"ready": true, "mp4": "/.../jobs/20261031-230304-f34a/why-do-we-have-leap-years-9x16.mp4",
#  "url": "https://...r2...", "duration_s": 31.4, "qc": {"passed": true, "errors": 0, "warnings": 1}, "actual_usd": 0.39}
```

`url` is a 2 h presigned download link when Cloudflare R2 is configured, otherwise null and you deliver the
file. `qc` is `tools/qc.py --render`: loudness was normalised to -14 LUFS, and it reports still stretches,
black gaps and a silent track. Pass `qc.warnings` on to the user in a line; `qc.errors > 0` means look before
you ship. `warnings` (a list of sentences) is for the user too: captions or music skipped, a scene voiced too
fast, a file over the size cap. Exit 2 means not ready yet. The MP4 is encoded under 9.5 MB so it attaches in
chat (Discord's bot limit is 10 MB); `maxMB: 0` in the spec turns that off for a full-quality YouTube upload.

If a job ends `failed` or `cancelled` with `resumable: true`, run `resume JOB_ID` (add `--max-usd` after a
`budget_exceeded` stop) rather than submitting again: it keeps the voice, visuals, captions and music already
paid for and redoes only what is missing.

Also: `list --json` (recent jobs), `cancel JOB_ID`, `prune` (old jobs' project trees; `submit` does this
itself), and every job keeps `jobs/<id>/job.log`, `receipts.jsonl`
(one JSON line per paid call: `ts, tool, cost_usd, elapsed_s, exit, args, job_id`) and the project copy it
rendered from, so a human can open `jobs/<id>/project/` and re-render after edits.

### Spend

Receipts also append to `MEDIA_SPEND_LOG` (default `logs/media-spend.log` in the toolkit) as JSON lines, one
per paid call, so a runtime that keeps a spend ledger reads single assets and jobs from one file. If your runtime
sets a workspace, export `VIDEO_JOB_WORKSPACE` to it: the ledger goes to its `logs/`, relative spec paths resolve
there, and `submit` books the whole estimate at once so a daily cap counts it before the work starts. Prices (`presets`, checked
against receipts on 2026-11-01): LTX clip ~$0.10, Ideogram image $0.03 to $0.10, FLUX image under a cent,
Qwen3 voice ~$0.01 a scene, captions ~$0.0001 a second. A 7-scene, 45 s video with two clips and two cards
lands around $0.40 to $0.55; the first clip after idle pays a Modal cold start on top. Under `--progress json`
every call prints one `cost` event, failures included ($0).

---

## Deeper: the individual tools

Everything below is what `video_job.py` drives for you. Reach for it directly when the user wants one asset
(an image, a clip, a song), a different template, or a talking head. Add `--progress json` to any cloud tool
for JSON-lines progress on stderr, and `--json` for a machine-readable result on stdout.

### Defaults that are current (2026-11-06)

| Need | Use | Not |
|---|---|---|
| Legible text in an image (title card, thumbnail, quote card) | `ideogram4.py --json caption.json` | FLUX or LTX (both garble text) |
| Text-free background for an overlay | `flux2.py --preset title-bg` | |
| Motion b-roll, animate a still | `ltx2.py` (`--input` for image-to-video) | |
| Voiceover | `voiceover.py --provider qwen3 --cloud modal --max-wpm 165` (free, self-hosted), `--provider elevenlabs`, or `--provider 60db` (`sixtydb_tts.py` standalone) | |
| Talking head anyone will watch | `soulx.py` (SoulX-FlashHead Pro, Modal only) | `sadtalker.py` is drafts, or when Modal is down |
| Music | `music_gen.py --variations 3`, then `pick_music.py` to rule takes out and `audition.py` to choose by watching | picking a take by ear alone |
| Sound effects | `sfx.py` (ElevenLabs, generated) or `sfx_search.py` (real recordings from Freesound via Openverse, free, no key) | |
| Before delivering any render | `qc.py --render out.mp4 --fix` | shipping unchecked |
| Captions for a Remotion template | `captions.py public/audio/scenes` (`--srt`/`--vtt` for subtitle files too) | burning the recogniser's own transcript |
| Social cut-downs of a 16:9 render | `cutdown.py --input out.mp4 --platform shorts` (`--mode track` keeps a talking head in frame) | |
| A highlight reel from a long recording | `highlights.py` (picks final takes from a word-timed transcript), then `cutdown.py --segment` | cutting by eye |
| Upload to YouTube | `youtube_upload.py` (OAuth, private by default) | |

### Voiceover

```bash
uv run tools/voiceover.py --provider qwen3 --cloud modal --scene-dir public/audio/scenes --max-wpm 165 --json
uv run tools/qwen3_tts.py --text "Hello" --speaker Ryan --tone warm --output hello.mp3 --cloud modal
uv run tools/qwen3_tts.py --text "Hello" --ref-audio ref.wav --ref-text "exact transcript" --output clone.mp3 --cloud modal
```

One `.txt` per scene in `--scene-dir`, one MP3 back per scene with `duration_seconds`, `wpm` and a `pacing`
label. A brand's voice: `--brand <name>` reads `brands/<name>/voice.json`. `--max-wpm 165` slows rushed takes with pitch-preserving atempo; a cloned voice copies its reference
clip's pace, so a flat or fast clone needs a better reference (`voice_prosody.py expand`), not a higher
temperature. For a cloned voice, generate one `.txt` per sentence three times and let `pick_takes.py` choose each
sentence by pace and pitch spread (the toolkit's CLAUDE.md, "Clone Prosody", has the loop). Speakers: Ryan, Aiden, Vivian, Serena, Uncle_Fu, Dylan, Eric, Ono_Anna, Sohee. Tones: neutral,
warm, professional, excited, calm, serious, storyteller, tutorial.

### Images

```bash
# Text in the image: author the caption JSON yourself (see .claude/skills/ideogram4/), 9:16 = 1440x2560, 16:9 = 1280x720
uv run tools/ideogram4.py --json caption.json --resolution 1440x2560 --output card.png
# Text-free background
uv run tools/flux2.py --preset title-bg --brand digital-samba --output bg.png --cloud modal --progress json
# Edit a photo, upscale
uv run tools/image_edit.py --input photo.jpg --prompt "Add sunglasses" --cloud modal
uv run tools/upscale.py --input small.png --scale 4 --cloud modal
```

### Video clips

```bash
uv run tools/ltx2.py --prompt "Aerial over a city at golden hour, cinematic" --width 1024 --height 576 \
    --num-frames 121 --output broll.mp4 --no-open --progress json --json
uv run tools/ltx2.py --prompt "Gentle camera drift, soft light shifts" --input still.png --output animated.mp4 --no-open
```

Sides must be multiples of 64; frames satisfy (n-1) % 8 == 0 (121 ≈ 5 s, 161 ≈ 7 s). About $0.07 to $0.20 a
clip and 2 to 4 minutes; the first clip after idle adds a cold start. Prompt motion and atmosphere, never text.
Add `--negative-prompt "anime, cartoon, watermark, text, logo"` when style drifts. `chain_video.py` chains
clips for continuity (one command, per-scene prompts in a JSON file).

### Talking heads

```bash
uv run tools/soulx.py --image presenter_16x9.png --audio scene01.mp3 --size 768 --output narrator01.mp4
uv run tools/soulx.py --image presenter_16x9.png --audio scene01.mp3 --model lite --width 512 --height 288 --output draft.mp4
```

SoulX follows the image's aspect ratio and holds identity over long takes (97% of frame-zero sharpness at
70 s). Pick one resolution per project: `torch.compile` recompiles on every size change (~10 min cold). Pro
sides are multiples of 16, Lite multiples of 32. SadTalker (`--preprocess full --still --expression-scale 0.8`)
is the RunPod fallback and square by default. Don't grade talking heads with a similarity metric; look at a
contact sheet.

### Music

```bash
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3 --progress json
uv run tools/music_gen.py --prompt "lofi, warm keys, no vocals" --duration 45 --variations 3 --output bg.mp3
uv run tools/pick_music.py bg_*.mp3 --target 45 --under voice --pick bg.mp3 --json   # takes the best, says why
```

Presets: corporate-bg, upbeat-tech, ambient, dramatic, tension, hopeful, cta, lofi. Default provider is the
acemusic API (free key); `--cloud modal` is the self-hosted fallback. Prompting and lyrics: `.claude/skills/acestep/`.

`pick_music.py` measures and can't listen, and good takes often tie. With a person in the loop, render the video
once without music and let them watch each take under it:

```bash
uv run tools/audition.py --video out/video.nomusic.mp4 --music bg_*.mp3 --volume 0.15 --fade-out 1.5 \
    --recommend --under voice    # -> out/audition/video.bg_1.mp4 ..., each normalised to -14 LUFS
```

Sound effects: `uv run tools/sfx.py --prompt "soft whoosh" --output whoosh.mp3` generates one;
`uv run tools/sfx_search.py --query "whoosh" --output whoosh.mp3` finds a real recording (CC0 and CC BY only by
default, with a `<name>.attribution.txt` to credit; Openverse doesn't guarantee its licence data, so check the
Freesound page before commercial use). To lay them on a finished video at set times, under ducked music:
`uv run tools/addmusic.py -i in.mp4 -m bed.mp3 --duck --events effects.json -o mixed.mp4`, with
`[{"path": "whoosh.mp3", "atSeconds": 1.5, "gain": 0.5}]`.

### QC, captions, cut-downs

```bash
uv run tools/qc.py --project projects/my-video                  # assets, timing, pacing, loudness
uv run tools/qc.py --render out/video.mp4 --fix                  # -14 LUFS, writes video.loudnorm.mp4; frame checks
uv run tools/qc.py --project projects/my-video --layout         # Remotion: text off canvas, too small, low contrast
uv run tools/contact_sheet.py out/video.mp4                      # one frame per scene, to look at before delivering
uv run tools/captions.py projects/my-video/public/audio/scenes  # word captions aligned to the script .txt
uv run tools/captions.py public/audio/scenes --srt --vtt        # also subtitle files, timed per scene audio
uv run tools/cutdown.py --input out/video.mp4 --platform shorts  # 16:9 -> 9:16, blur-fit
uv run tools/cutdown.py --input talk.mp4 --platform shorts --mode track   # follow one face (uv sync --extra vision)
uv run tools/highlights.py --video talk.mp4 --transcript words.json --seconds 45 --output reel.mp4 --save-plan edit.json
                                                                 # final takes; the plan's cutdownArgs feed cutdown.py
```

Run `qc.py --render` before every delivery: the templates mix voice and music without normalising, and
YouTube turns quiet audio down, never up. It exits 1 on errors (cut-off narration, silent track); a video that is
silent on purpose passes with `--allow-silent`. In sprint-review configs, `align_captions.py` re-times hand-written
`cap()` phrases to the audio.

Fixing existing videos: `redub.py --input v.mp4 --voice-id ID --sync` (new voice, word-level re-timing),
`addmusic.py` (a bed under a finished video; `--duck` lowers it under speech, `--events` adds timed effects), `locate_watermark.py` then `dewatermark.py` (ProPainter inpainting),
`notebooklm_brand.py` (a NotebookLM video with your logo and outro). Publishing: `youtube_upload.py --video
out.mp4 --title "..." --json-out` (one-time `--auth`, private unless told otherwise; `docs/youtube-upload.md`).

### Remotion templates (when a job's compositor isn't enough)

The routes table at the top says which template fits. Copy to `projects/<name>`, `npm install`, write the config,
generate per-scene audio, update each scene's `durationSeconds` to `ceil(audio + 2)` (or run
`uv run tools/sync_timing.py --apply`), `npm run render` from the project directory, then `qc.py --render`.
Always `<OffthreadVideo>` and `<Audio>` from `remotion`, never a raw `<video>` tag. The toolkit's CLAUDE.md
holds the full workflow. Cloud GPU setup and endpoint trouble: `.claude/skills/runpod/`, `docs/modal-setup.md`.
Image edit prompting: `.claude/skills/qwen-edit/`.

### Progress JSON (stderr, one object per line)

```json
{"ts":"14:23:15","stage":"submit","msg":"Sending to Modal...","pct":null,"elapsed":0.0}
{"ts":"14:24:02","stage":"complete","msg":"Saved: clip.mp4","pct":100,"elapsed":47.3}
{"ts":"14:24:02","stage":"cost","msg":"Est. cost: $0.0712 (103s on modal)","pct":null,"elapsed":103.0}
```

Stages: `submit`, `queue`, `processing`, `waiting`, `item` (n of m, `pct` set), `cost`, `complete`, `error`.
`video_job.py` reads the `cost` events for its receipts; a runtime that wraps single tool runs can do the same.

## Error recovery

| Problem | Fix |
|---|---|
| `No module named ...` | `uv sync` from the toolkit root, run tools with `uv run` |
| `MODAL_*_ENDPOINT_URL not configured` | `uv run tools/verify_setup.py`; `uv run tools/modal_setup.py --list` |
| `submit` exits 3, `lock_held` | Another job is running; `status` that id, or `cancel` it if it is stuck |
| `submit` exits 2, `unavailable` | This host lacks a provider the spec uses. `missing` says which and suggests an alternative: switch the scene kind or voice, or set the host up |
| `submit` exits 2, `budget_exceeded` | Raise `--max-usd`, drop a scene, use `flux` or `"ideogram": {"speed": "TURBO"}` |
| `submit --brief` exits 2, `no_expander` | Write the spec yourself and pass `--spec`, or set `VIDEO_JOB_EXPANDER` |
| Job `failed`, `error_code: step_failed` | `status --json` has the error; `jobs/<id>/job.log` has the tool output. Modal cold starts can time out once: `resume JOB_ID` |
| Job `failed`, `worker_died` or `worker_never_started` | The worker was killed (a container restart, say). `resume JOB_ID`: it reuses everything already made |
| Cold start on Modal | First call after idle takes 1 to 3 minutes; the next is warm |
| Captions missing from a job's video | No `ELEVENLABS_API_KEY` and no local whisper (`uv sync --extra whisper`); the job renders without and says so in `warnings` |
