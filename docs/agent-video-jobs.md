# Agent video jobs: brief in, MP4 out

`tools/video_job.py` lets any agent say "make a 45-second video about X" and get a finished, QC-checked MP4
back without writing Remotion code and without holding a connection open. It was built for agents that call
toolkit CLIs through an allowlisted `run_tool` with a hard per-call time limit (longarm's Lugh has 660 s), and
for MCP clients with no shell at all (`tools/video_mcp.py`). The agent-facing skill is `skills/video-toolkit/`.

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
uv run tools/video_job.py cancel   JOB_ID
uv run tools/video_job.py presets  --json                           # prices, formats, kinds, example spec
uv run tools/video_job.py spec     --brief "..."                    # run the expander only
```

Exit codes: 0 ok, 1 bad request, 2 refused or not ready (`budget_exceeded`, `no_expander`, job not done),
3 another job holds the lock (`lock_held`).

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
| `scenes[].visual.kind` | `ltx` (5 s motion clip), `ideogram` (designed card with legible text), `flux` (text-free still), `none` (placeholder card). A scene with no `visual` gets `none`. |
| `scenes[].visual.prompt` / `caption` | `prompt` for every kind; `ideogram` also takes a full JSON `caption` (see `.claude/skills/ideogram4/`). Optional per-scene `speed`, `numFrames`, `negativePrompt`, `seed`. |
| `voice` | Merged over the template's voice block: `provider` (qwen3, elevenlabs, 60db), `speaker`, `tone`, `refAudio` + `refText` for a clone, `maxWpm`, `brand`, `voiceId`. |
| `captions` | `{"enabled": true, "engine": "auto"}`. `auto` is Scribe when `ELEVENLABS_API_KEY` is set, else local whisper; neither available means the job renders without captions and says so. |
| `music` | `{"enabled": false}` by default. `{"enabled": true, "prompt": "...", "provider": "acemusic"}` adds a ducked, looped bed. |
| `ideogram.speed` | `TURBO` $0.03, `DEFAULT` $0.06, `QUALITY` $0.10 per image. |
| `brand`, `palette` | A `brands/<name>` for voice and Ideogram colours; palette overrides for the placeholder cards and caption pills. |

Normalisation adds `id` (`01`, `02`...) and `asset` paths; `jobs/<id>/spec.json` is the normalised copy.

## Brief → spec

That step needs an LLM, and the toolkit doesn't hardwire one. Two ways:

1. **The agent writes the spec** (preferred: the agent is the LLM, and it can show the user the scenes
   before spending anything). `presets` prints the shape and prices.
2. **An expander command.** Set `VIDEO_JOB_EXPANDER` to any command that reads a JSON request on stdin
   (`brief`, `format`, `duration_s`, `words_budget`, `spec_example`, `instructions`) and prints a spec JSON on
   stdout. Code fences are tolerated. For example, with the Claude Code CLI installed:

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
- Actual cost comes from the tool's `cost` progress event when it emits one (LTX, FLUX via `cloud_gpu.py`),
  otherwise the table price is booked. Qwen3 and Scribe calls therefore show the estimate as the actual.
- Receipts: `jobs/<id>/receipts.jsonl` and `MEDIA_SPEND_LOG` (default `logs/media-spend.log`), one JSON line
  per tool call in the shape longarm's `video-spend.py` writes, plus `job_id`:
  `{"ts": "...Z", "tool": "ltx2.py", "cost_usd": 0.0712, "elapsed_s": 103.4, "exit": 0, "args": [...], "job_id": "..."}`.

Measured on 2026-10-31, a 3-scene 30 s video (one LTX clip, two Ideogram cards, Qwen3 voice, Scribe
captions) estimates at $0.35; see the PR that added this for the actual receipts.

## Job state

`jobs/<id>/job.json` is the source of truth (`VIDEO_JOBS_DIR` moves the directory):

| Field | Meaning |
|---|---|
| `phase` | `queued, voiceover, visuals, captions, music, build, qc, publish` then `done`, `failed` or `cancelled` |
| `progress` | 0 to 100 |
| `step_detail` | the asset being generated, e.g. `01_hook via ltx2.py` |
| `estimate_usd`, `actual_usd`, `max_usd` | money |
| `steps` | per-step summaries: VO durations and pacing, assets made, caption engine, QC verdict |
| `output` | `mp4`, `url`, `r2_key`, `duration_s`, `size_bytes`, `qc` |
| `error`, `error_code` | `budget_exceeded`, `step_failed`, `worker_died`, `worker_crash`, `cancelled` |
| `worker_pid` | the detached worker; `status` marks the job `worker_died` if it is gone before a terminal phase |

Beside it: `job.log` (phases and the tail of every tool's output), `worker.out` (the worker's own stdout),
`receipts.jsonl`, `spec.json`, `qc.json`, `project/` (the rendered template copy, re-renderable by hand) and
the final MP4.

The lock `jobs/.lock` names the running job. It is considered stale when that job is finished, its worker is
dead, or no worker reported in within two minutes, so a crash never wedges the queue. Longarm's daily cap
assumes serial spend, which the lock guarantees.

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
`video_submit`, `video_status`, `video_result`, `video_list_presets`, plus `video_list` and `video_cancel`.
Every result carries `ok` and, when false, `error_code`.

## Tests

```bash
uv run python -m unittest tests/python/test_video_job.py -v
```

Covers the estimate, the state machine (with the paid steps faked), the lock, worker-death detection, the
cap refusal, the mid-job budget stop, the receipt shape and the expander hook. No cloud keys or network.

## Wiring an agent in

For longarm (separate work): allowlist `video_job.py` for `native__run_tool`, count the job's estimate as a
reservation against the daily media cap, have a poller post the MP4 back to the thread, and ship
`skills/video-toolkit/SKILL.md` as the workspace skill. Anything over ~$1 goes through an approval card.
