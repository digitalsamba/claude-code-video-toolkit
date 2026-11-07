# Agent video jobs: brief in, MP4 out

`tools/video_job.py` lets any agent say "make a 45-second video about X" and get a finished, QC-checked MP4
back without writing Remotion code and without holding a connection open. It was built for agents that call
toolkit CLIs through an allowlisted `run_tool` with a hard per-call time limit (ours had 660 s), and
for MCP clients with no shell at all (`tools/video_mcp.py`). The agent-facing skill is `skills/video-toolkit/`.

A job is one route of several. Product walkthroughs rebuilt from an app's source (synthetic UI), data charts,
sprint reviews and marketing demos are **sessions**: a Claude Code-style working session copies a template, edits
it with the skill it names and renders with Node, with the person who asked reviewing the drafts. The session can
be theirs, or an agent's, if its runtime runs one headless. `video_job.py catalog` (MCP: `video_catalog`) lists every route from the registry, with
demo videos, so an agent can answer "what can you make?" and pick one; the routes table at the top of the skill
says which fits what a user asks for.

## How it works

```
submit ──► jobs/<id>/job.json (queued)        returns in seconds with the id and an estimate
              │
   detached worker (subprocess.Popen, new session / DETACHED_PROCESS on Windows)
              │
   spec ─► voiceover ─► visuals ─► captions ─► [music] ─► build ─► qc --fix ─► publish ─► done
           voiceover.py  ideogram4.py / ltx2.py / flux2.py   captions.py   moviepy   qc.py   copy + R2
              │
status ◄── job.json, job.log, receipts.jsonl         works from any fresh process
result ◄── jobs/<id>/<title>-<format>.mp4 (+ presigned R2 URL when configured)
```

The compositor is `templates/concept-explainer-short`, copied into `jobs/<id>/project/` and driven by the
spec: per-scene narration, Ken Burns on stills, boomerang-looped clips, burned karaoke captions, audio-anchored
timing. 16:9 is the same build with a landscape `config.json` (the template was 9:16 only before this).

## Commands

```bash
uv run tools/video_job.py estimate --spec spec.json --json          # pure function of the spec, no keys needed
uv run tools/video_job.py estimate --brief "..." --format 16:9 --duration 30
uv run tools/video_job.py submit   --spec spec.json --max-usd 1.00 --json
uv run tools/video_job.py status   JOB_ID --json
uv run tools/video_job.py result   JOB_ID --json                    # exit 2 until done
uv run tools/video_job.py list     --json
uv run tools/video_job.py cancel   JOB_ID                           # stops the worker and the tool it runs
uv run tools/video_job.py resume   JOB_ID [--max-usd 3.00]          # carry on a failed or cancelled job
uv run tools/video_job.py prune    [--keep 10] [--older-than 7] [--dry-run]
uv run tools/video_job.py presets  --json                           # prices, formats, kinds, example spec
uv run tools/video_job.py spec     --brief "..."                    # run the expander only
```

Exit codes: 0 ok, 1 bad request (`bad_request`, `not_found`, `job_unreadable`, `internal`), 2 refused or
not ready (`budget_exceeded`, `no_expander`, job not done, or a usage error), 3 another job holds the lock
(`lock_held`). Errors under `--json` are a JSON object with `error` and `error_code`, never a traceback.

### Resume

A job that stops part-way (the worker or its container died, a tool failed, the cap tripped, or it was
cancelled) keeps what it already bought. `status` says `resumable: true`, and `resume JOB_ID` restarts the worker
from the first missing output: every paid step skips files already on disk (scene MP3s, each visual, captions,
music), and the build, QC and publish rerun because they are free. The cap is checked against only what is still
missing; pass `--max-usd` to raise it after a `budget_exceeded` stop. A pruned job can't be resumed.

### Prune

Each job keeps a full template copy with its assets (40 to 60 MB). `prune` deletes the `project/` tree of old
finished jobs and keeps `job.json`, the receipts, the logs, the final MP4 and its contact sheet, so `status` and
`result` still answer. It never touches a running job, the `--keep` newest, or a job that published its MP4 in the
last 7 days. `submit` runs it with those defaults every time (`VIDEO_JOB_KEEP`, default 10; `0` turns it off).

### File size

The delivered MP4 is encoded under **9.5 MB** so it attaches in Discord, whose bot upload limit is 10 MB: CRF 23
under a VBV ceiling at the bitrate the duration allows (`config.json → encode.maxMB` in the template). A 50 s
16:9 render went from 11.7 MB to 8.3 MB at SSIM 0.993 against the uncapped file, with no visible difference on
the text cards. `spec.maxMB` or `VIDEO_JOB_MAX_MB` changes the cap and `0` turns it off (for a YouTube upload,
say). `result` carries `size_bytes`, and a file still over the cap shows up in `warnings`.

## The spec

```json
{
  "title": "What is an MCP server?",
  "format": "16:9",
  "voice": {"provider": "qwen3", "speaker": "Ryan", "tone": "professional", "maxWpm": 165},
  "scenes": [
    {"slug": "hook", "text": "Your AI agent can reason all day. But it can't read your files. Not without a door.",
     "visual": {"kind": "ltx", "prompt": "Slow dolly through a dark server room, blue status lights, cinematic, no text"}},
    {"slug": "door", "text": "An MCP server is that door: tools and data over one standard protocol.",
     "visual": {"kind": "ideogram", "caption": {"high_level_description": "...", "style_description": {}, "compositional_deconstruction": {}}}},
    {"slug": "cta", "text": "Write the server once, and every agent gets the same tools.",
     "visual": {"kind": "ideogram", "prompt": "Bold wide card, headline WRITE IT ONCE, deep navy, clean poster"}}
  ]
}
```

| Key | Meaning |
|---|---|
| `format` | `9:16` (1080x1920) or `16:9` (1920x1080). Default 9:16. |
| `scenes[].text` | Narration, verbatim. Captions burn this text. ~2.4 words a second. |
| `scenes[].visual.kind` | `ltx` (5 s motion clip, looped), `text` (constructed: a generated background that drifts plus the scene's `headline` and optional `subline` animated in), `ideogram` (designed card with legible text), `flux` (text-free generated background), `none` (placeholder card). A scene with no `visual` gets `flux` with a prompt built from its narration, or `none` when the spec says `"draft": true`. Any non-ideogram kind may carry `headline`/`subline` too. |
| `scenes[].visual.prompt` / `caption` | `prompt` for `ltx` and `flux`. For `ideogram`, always a JSON `caption`: `presets --json` prints `ideogram_caption_template` to fill in (a plain `prompt` works but adds filler text to the card). Optional per-scene `speed`, `numFrames`, `negativePrompt` (LTX gets a text-and-logo-excluding default), `seed`. An `ltx` prompt that names no style gets a live-action footage suffix; a top-level `ltxStyle` replaces it (`""` disables). |
| `voice` | An object merged over the template's voice block (`provider`, `speaker`, `tone`, `refAudio` + `refText` for a clone, `maxWpm`, `brand`, `voiceId`), or a preset name: `assets/voices/<name>.json` (a clone; the audio file beside it is gitignored, copy it) or a brand with a `voice.json`. `--voice <name>` on the CLI. |
| `captions` | `{"enabled": true, "engine": "auto"}`. `auto` is Scribe when `ELEVENLABS_API_KEY` is set, else local whisper; neither available means the job renders without captions and says so. |
| `music` | On by default (a subtle ambient bed from the free acemusic API, looped and ducked under the narration); `false`, a prompt string, or `{"enabled", "prompt", "provider"}`. Skipped with a log line when the key is missing. |
| `titleCard` | On by default: the title animates in over the first scene for 2.8 s (so frame 0, the Shorts cover, is never empty). `false` to skip, or `{"text", "seconds"}`. |
| `ideogram.speed` | `TURBO` $0.03, `DEFAULT` $0.06, `QUALITY` $0.10 per image. |
| `brand`, `palette` | A `brands/<name>` for voice and Ideogram colours; palette overrides for the placeholder cards and caption pills. |
| `kit` | A kit name (`kits/README.md`): fills `brand`, `voice`, `format`, `backgrounds` and `music` when the spec doesn't set them, sets the palette from the kit's colours, and is where `file` paths are looked up first. Searched in `$VIDEO_JOB_WORKSPACE/kits/`, then the toolkit's `kits/`. A `file` path must lead into the kit, the workspace, the submitting directory or the toolkit. |
| `allowCopy` | `true` skips the copy check (dashes and stock AI phrases in the narration, title, headlines and sublines, and narration repeated from an earlier finished job). Only when the person asked for the same words. |
| `backgrounds` | `auto` (default): `text` scenes and scenes with no visual get a FLUX background when the host has FLUX, else a local one. `local`: a flat field in the kit's or brand's ground colour with grain, drawn by the job (free, no GPU). `flux`: always FLUX. A scene that names kind `flux` always uses FLUX. |

Normalisation adds `id` (`01`, `02`...) and `asset` paths; `jobs/<id>/spec.json` is the normalised copy.

### The flat form (no JSON)

For callers that can't reliably write nested JSON (an open model with no editor, or a person at a
terminal), scenes are plain strings, one per `--scene`:

```bash
uv run tools/video_job.py submit --format 16:9 --title "The Real Bottleneck" --max-usd 2 \
  --scene "AI gets cheaper every month. Your PlayStation just doubled in price. :: ltx: slow push-in through a dark server room, blue lights pulsing" \
  --scene "Compute is the new scarce thing. :: text: COMPUTE IS THE SCARCE THING ; the price of intelligence falls, the price of silicon climbs" \
  --scene "One prompt. One agent. A finished video. :: text: ONE PROMPT ; made by an agent, start to finish"
```

After the narration and ` :: `: `ltx: <motion prompt>`, `text: <HEADLINE> ; <subline>`,
`card: <HEADLINE> ; <object>` (Ideogram), `flux: <prompt>`, or nothing (a generated background).
An `ltx:` prompt that asks for a screen, interface, dashboard, terminal or timeline is refused at
submit, before any spend, with a message naming the scene: the video model draws fake software covered
in garbled text. Film a physical thing instead, or make it a `text:` scene; `--allow-screens` overrides.
An object written as a chain, `card: THE LOOP ; prompt -> agent -> video`, becomes a labelled flow
diagram (2 to 4 boxes with arrows, across the frame in 16:9 and down it in 9:16). Asking for "a
diagram" in prose gets a silhouette; the chain form is how a diagram is spelled.
`file: <path> ; HEADLINE ; subline` is the person's own photo or clip. `--kit <name>` makes the video for someone
with a kit: its format, voice, colours and music, and `file:` paths in its assets. It works with `--spec`,
`--spec-json` and `--brief` too (the expander is then sent the kit's description and asset list).
`--music "..."`, `--no-music`, `--draft` and `--no-title-card` set the rest.

## What this host can make

`host_capabilities()` reads the environment (`.env` included) and reports which visual kinds, voices and music
providers this host can run, the local tools a run needs (ffmpeg, ffprobe, moviepy, numpy for a job; Node 18+
as well for a session route) and so which route levels it can run (`routes`); `catalog` and `presets` carry it as
`host`. A missing local tool is reported with `install` instead of `env`. `estimate` adds `host_missing`, and
`submit` raises `Unavailable` (CLI exit 2, `error_code: unavailable`, MCP the same with `missing`) before anything is
written or spent. Each entry names what needs it, every env var it needs, and an alternative (the other cloud
for a qwen3 voice when that one is set up, an ideogram card instead of a clip). The music bed, captions and R2
links are soft: the job runs without them and says so in `warnings`, and `estimate` lists a missing music provider
in `host_missing` marked `"soft": true`. `resume` runs the same check. `VIDEO_JOB_SKIP_HOST_CHECK=1` turns it off
(for `estimate` too) for a host whose tools find their keys some other way. A provider or cloud name the toolkit
doesn't know is a `bad_request`.

A host with no GPU provider can still make a job: `text` scenes and scenes with no visual get local backgrounds,
`file` scenes use the person's own photos and clips, and `ideogram` cards need only an Ideogram key. The voice
needs ElevenLabs or 60db (a key), or qwen3 on Modal or RunPod. Music is soft like captions: no provider, no bed,
and a warning.

## Brief → spec

That step needs an LLM, and the toolkit doesn't hardwire one. Two ways:

1. **The agent writes the spec** (preferred: the agent is the LLM, and it can show the user the scenes
   before spending anything). `presets` prints the shape and prices.
2. **An expander command.** Set `VIDEO_JOB_EXPANDER` to any command that reads a JSON request on stdin
   (`brief`, `format`, `duration_s`, `words_budget`, `spec_example`, `ideogram_caption_template`,
   `instructions`) and prints a spec JSON on stdout. Code fences are tolerated. For example, with the Claude Code CLI installed:

   ```bash
   export VIDEO_JOB_EXPANDER='claude -p --output-format text'
   uv run tools/video_job.py submit --brief "45 seconds on why prompt caching makes agents cheaper" --format 16:9
   ```

   Any script that calls any model works the same way.

Without an expander, `estimate --brief` still answers, with a skeleton (N scenes alternating motion and card
for the duration, marked `spec_source: skeleton`), and `submit --brief` is refused with `no_expander`.

## Cost

- `estimate` is a pure function of the spec and the `PRICES` table at the top of `tools/video_job.py`
  (`presets --json` prints it). It prices `captions.engine: auto` as Scribe, the paid engine, so it is an upper
  bound on that line.
- `submit` refuses when the estimate is over `--max-usd` (default 2.00, env `VIDEO_JOB_MAX_USD`), before
  anything is written.
- The worker checks before each paid step and stops with `budget_exceeded` if actual spend plus that step's
  estimate would pass the cap. Visuals are checked per asset.
- Actual cost comes from the tool's `cost` progress event when it emits one (LTX, FLUX and Qwen3 via
  `cloud_gpu.py`; `voiceover.py --progress json` relays Qwen3's), otherwise the table price is booked
  (ElevenLabs voice, Ideogram).
- Receipts: `jobs/<id>/receipts.jsonl` and `MEDIA_SPEND_LOG` (default `logs/media-spend.log`), one JSON line
  per tool call, plus `job_id`:
  `{"ts": "...Z", "tool": "ltx2.py", "cost_usd": 0.0712, "elapsed_s": 103.4, "exit": 0, "args": [...], "job_id": "..."}`.

The table was checked on 2026-11-01 against the receipts of the last ten jobs on a production agent host: LTX clips cost a
median $0.0825 (the table said $0.20, now $0.10) and FLUX backgrounds $0.0058 ($0.01, now $0.0075). Estimates
before that date were roughly double the actual spend.

## Job state

`jobs/<id>/job.json` is the source of truth (`VIDEO_JOBS_DIR` moves the directory):

| Field | Meaning |
|---|---|
| `phase` | `queued, voiceover, visuals, captions, music, build, qc, publish` then `done`, `failed` or `cancelled` |
| `progress` | 0 to 100 |
| `step_detail` | the asset being generated, e.g. `01_hook via ltx2.py` |
| `estimate_usd`, `actual_usd`, `max_usd` | money |
| `steps` | per-step summaries: VO durations and pacing, assets made, caption engine, QC verdict |
| `events` | timestamped list (last 200): each phase, tool start and finish (seconds, cost), each scene voiced (wpm), each asset made, captions engine or skip, music, QC findings, render size, publish, resume |
| `timings`, `host` | per-step and per-asset seconds and the build rate, which this host's ETAs are calibrated from |
| `resumes`, `pruned` | how many times the job was resumed; when its project tree was pruned |
| `made` | outputs whose tool succeeded; resume reuses only these, so a half-written file is bought again |
| `ledger_usd` | reserve mode: what the shared ledger holds for this job right now (settles net to `actual_usd`) |
| `output` | `mp4`, `url`, `r2_key`, `duration_s`, `size_bytes`, `qc` |
| `error`, `error_code` | `budget_exceeded`, `step_failed`, `worker_died`, `worker_never_started`, `worker_crash`, `cancelled` |
| `worker_pid` | the detached worker; `status` marks the job `worker_died` if it is gone before a terminal phase |

`status` adds `eta_s`, `done`, `resumable`, `warnings` and `log_tail`; `result` adds `warnings` and
`resumable`. `warnings` are sentences to pass on to the user: captions or music skipped (and how to get them),
scenes voiced too fast, an MP4 over the size cap.

Beside it: `job.log` (phases and the tail of every tool's output), `worker.out` (the worker's own stdout),
`receipts.jsonl`, `spec.json`, `qc.json`, `project/` (the rendered template copy, re-renderable by hand) and
the final MP4.

The lock `jobs/.lock` names the running job. It is considered stale when that job is finished, its worker is
dead, or no worker reported in within two minutes, so a crash never wedges the queue. `submit` writes the
worker's pid into the lock before it returns, and `status` or `list` clear a stale lock and fail a job whose
worker never started (`worker_never_started`, resumable). A runtime's daily
cap can assume serial spend, which the lock guarantees.

## Running under an agent runtime's tool call

The strictest runtime we've run it under spawns toolkit tools through a spend wrapper with an argv list, no
shell, no files the model can write, every argument capped at 4 KB, `--progress json` injected, and a daily
spend cap summed from a shared ledger. `video_job.py` fits that:

- `--progress json` is accepted on every command. `submit` then prints one `cost` event on stderr
  with the whole estimate, which the wrapper turns into a receipt, so the runtime's daily cap books the
  job as a **reservation** the moment it starts. In that mode (`VIDEO_JOB_WORKSPACE` set, or
  `VIDEO_JOB_RESERVE_AT_SUBMIT=1`) the worker keeps its actual per-tool receipts in
  `jobs/<id>/receipts.jsonl` only, so nothing is counted twice. `status`, `result`, `list` and
  `cancel` print a `$0.0000` cost event so they are not charged the wrapper's unknown-cost default.
  **Every** call prints exactly one: a refused or failed call (bad spec, lock held, a usage error, a
  crash) prints `$0.0000` too. Before 2026-11-01 failures printed none, and a wrapper that books a
  flat $0.05 for a receipt with no cost charged 15 failed submits $0.75 that way. `resume` reserves the rest of the
  estimate again (the job had settled to its actual spend when it stopped) and settles again at the end.
- Ledger noise: every `status` poll still leaves a `$0` receipt line in the shared ledger. It costs the
  cap nothing; filtering it out belongs in the runtime's wrapper (skip a receipt whose only cost
  event is `$0.0000` with `no generation` in the message), not here.
  The ledger path follows `MEDIA_SPEND_LOG`, else `$VIDEO_JOB_WORKSPACE/logs/media-spend.log`.
  `VIDEO_JOB_WORKSPACE` is the agent's workspace: relative `@path` specs resolve there first, and it turns
  reservation on. (`LONGARM_WORKSPACE` is read as an older name for it.)
- `--spec-json '<json>'` takes the spec inline. To fit a 3-scene spec in 4 KB, an `ideogram` scene
  can give the short caption form `{"headline": "...", "object": "...", "description": "..."}`, which
  `normalize_spec` expands into the full template.
- `submit --output files/<name>.mp4` copies the finished MP4 there as well (a runtime that serves `files/`
  at a public URL and computes that URL itself from the `--output` argument; the file exists once
  `status` says `done`). `result` reports it as `published`.
- `status` carries `eta_s`, seconds left: each remaining step priced at the median of this host's
  last 5 finished jobs (build as seconds per second of video, visuals per kind), the current step
  less the time already spent in it, plus 90 s for a cold LTX container when the host's last clip
  ended over 10 minutes ago. With no history it falls back to a table measured on a Mac. A small
  container host built about 10x slower than the Mac did before the 2026-11-01 speed-up, which is why
  the table alone was wrong there. `events` says what is happening, so an agent can report "scene 3's clip is
  rendering, two images done" rather than a phase name.
- `--output` must name an `.mp4`. With `VIDEO_JOB_OUTPUT_ROOTS` set (directories separated by
  `os.pathsep`), it must also be inside one of them.
- A runtime's per-call cost bound becomes the job's `--max-usd` (default 2.00, env
  `VIDEO_JOB_MAX_USD`); set it at or under the daily cap's call reserve policy you want.

An agent's Python environment needs `moviepy`, `numpy` and `elevenlabs` on top of the client libraries
(the build and QC steps run locally, in the agent's container if it has one); `IDEOGRAM_API_KEY` goes in the
toolkit's own `.env` on the host. Give the agent `skills/video-toolkit/SKILL.md` as its skill, or paste its
quick path into the system prompt of a runtime that has no skills.

## MCP server

`tools/video_mcp.py` wraps the same module for agents without a shell. It holds no pipeline logic.

```bash
uv sync --extra mcp
uv run --extra mcp tools/video_mcp.py        # stdio
```

Claude Desktop or Claude Code config:

```json
{"mcpServers": {"video-toolkit": {"command": "uv",
  "args": ["run", "--directory", "/path/to/claude-code-video-toolkit", "--extra", "mcp", "tools/video_mcp.py"]}}}
```

```bash
claude mcp add video-toolkit -- uv run --directory /path/to/claude-code-video-toolkit --extra mcp tools/video_mcp.py
```

Any other client spawns that command and speaks MCP over stdin/stdout. Tools: `video_estimate`,
`video_submit`, `video_status`, `video_result`, `video_list_presets`, plus `video_list`, `video_cancel` and
`video_resume`. Every dict result carries `ok` and, when false, `error` and `error_code`; no tool raises.

## Tests

```bash
uv run python -m unittest tests/python/test_video_job.py tests/python/test_video_job_agents.py \
    tests/python/test_video_job_hardening.py tests/python/test_video_job_host.py -v
```

Covers the estimate, the state machine (with the paid steps faked), the lock and its two-minute window,
worker-death detection, the cap refusal, the mid-job budget stop, the receipt shape, the expander hook, resume
(with a fake `run_tool` that writes real files, so the skip-what-exists logic is what is tested), prune, the
cost event on every exit, the ETA arithmetic, events and warnings, the size cap, and `cancel` stopping the
worker's child processes. No cloud keys or network.

### Windows

The worker spawn (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`), `pid_alive` (`OpenProcess` +
`GetExitCodeProcess` with explicit ctypes types), `cancel` (`taskkill /T /F`), the `job.json` replace retry and
the expander's command splitting were reviewed against the Python and Win32 docs but have **not been run on
Windows**. Known caveats: a worker started from a console whose job object kills its children on close dies
with that console, and a process that exits with code 259 reads as alive.

## Wiring an agent in

What a runtime needs, whatever it is: allowlist `video_job.py` (or connect `video_mcp.py`), count the job's
estimate as a reservation against any spend cap (`VIDEO_JOB_WORKSPACE` does this), poll `status` and post the
MP4 back to wherever the request came from, and ship `skills/video-toolkit/SKILL.md` as the agent's skill. Put
an approval step in front of anything over the amount you're comfortable with (we use ~$1).
