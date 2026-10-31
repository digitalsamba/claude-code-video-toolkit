---
name: video_toolkit
description: Create professional videos autonomously using claude-code-video-toolkit — AI voiceovers, image generation, music, talking heads, and Remotion rendering.
metadata:
  openclaw:
    emoji: "🎬"
    skillKey: "video-toolkit"
    os: ["darwin", "linux"]
    requires:
      bins: ["node", "python3", "ffmpeg", "npm"]
---

# Video Toolkit

Create professional explainer videos from a text brief. The toolkit uses open-source AI models on cloud GPUs (Modal or RunPod) for voiceover, image generation, music, and talking head animation. Remotion (React) handles composition and rendering.

## CRITICAL: Toolkit Path

The toolkit lives at a fixed path. **ALWAYS `cd` here before running any tool command.**

```bash
TOOLKIT=~/.openclaw/workspace/claude-code-video-toolkit
cd $TOOLKIT
```

**NEVER run tool commands from inside a project directory.** Tools resolve paths relative to the toolkit root.

## CRITICAL: Progress Reporting

**ALWAYS add `--progress json` to every cloud GPU tool command.** This gives you structured JSON Lines on stderr so you can monitor job status, detect stuck jobs, and report progress to the user in real-time.

```bash
# CORRECT — always include --progress json
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3 --progress json

# WRONG — no visibility into job status
uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
```

Tools that support `--progress json`: `music_gen.py`, `qwen3_tts.py`, `flux2.py`, `upscale.py`, `sadtalker.py`, `image_edit.py`, `dewatermark.py`, `ltx2.py`, `chain_video.py`.

See the **Progress Reporting** section below for output format and stage definitions.

## CRITICAL: Long-Running Tasks — Use yieldMs, Not background:true

**Any tool command that takes more than 30 seconds MUST use `exec` with `yieldMs` so you can report progress to the user live.** This includes: batch FLUX generation, chain_video, SadTalker, music generation, and any multi-scene pipeline.

```
exec command:"cd ~/.openclaw/workspace/claude-code-video-toolkit && uv run tools/chain_video.py --output-dir /path/ --progress json ..." yieldMs:10000
```

**The polling loop:**
1. `exec` with `yieldMs:10000` starts the command and returns control to you every 10 seconds
2. Read the `--progress json` output — look for `"stage":"item"` (scene complete) or `"stage":"complete"` (all done)
3. Report progress to the user ("Scene 05/30 complete, 17%")
4. Poll again: `process action:poll sessionId:<id>`
5. Repeat until `"stage":"complete"`

**Why:** Your agent run ends when you finish responding. If you use `bash background:true`, you lose the ability to report progress — the user sees silence until they nudge you. With `yieldMs`, you stay in the loop.

**NEVER do this:**
- `bash background:true command:"long running thing"` then promise to "monitor" — you can't, your run ends
- Break a batch into individual tool calls across separate messages — your run ends between each one
- Promise to "continue autonomously" — you literally cannot without an external trigger

## Setup

### Step 1: Check Current State

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/verify_setup.py
```

If everything shows `[x]`, skip to "Quick Test" below. Otherwise continue setup.

### Step 2: Install Python Dependencies

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv sync
```

Note: `uv sync` creates its own `.venv/` from the lockfile, so it sidesteps Debian/Ubuntu's managed-Python restrictions (PEP 668) — no `--break-system-packages` needed. If `uv` is missing, install it first: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

### Step 3: Configure Cloud GPU Endpoints

The toolkit needs cloud GPU endpoint URLs in `.env`. Check if `.env` exists and has Modal endpoints:

```bash
cat ~/.openclaw/workspace/claude-code-video-toolkit/.env | grep MODAL
```

If Modal endpoints are configured, you're ready. If not, **ask the user to provide Modal endpoint URLs** or set up Modal:

```bash
uv sync --extra modal
uv run modal setup   # Opens browser for authentication

# Deploy each tool — capture the endpoint URL from output
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run modal deploy docker/modal-qwen3-tts/app.py
uv run modal deploy docker/modal-flux2/app.py
uv run modal deploy docker/modal-music-gen/app.py
uv run modal deploy docker/modal-sadtalker/app.py
uv run modal deploy docker/modal-image-edit/app.py
uv run modal deploy docker/modal-upscale/app.py
uv run modal deploy docker/modal-propainter/app.py
uv run modal deploy docker/modal-ltx2/app.py      # Requires: uv run modal secret create huggingface-token HF_TOKEN=hf_...
```

**LTX-2 prerequisite:** Before deploying LTX-2, create a HuggingFace secret and accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized):
```bash
uv run modal secret create huggingface-token HF_TOKEN=hf_your_read_access_token
```

Add each URL to `.env`:
```
ACEMUSIC_API_KEY=...                          # Free key from acemusic.ai/api-key (best music quality)
MODAL_QWEN3_TTS_ENDPOINT_URL=https://...modal.run
MODAL_FLUX2_ENDPOINT_URL=https://...modal.run
MODAL_MUSIC_GEN_ENDPOINT_URL=https://...modal.run
MODAL_SADTALKER_ENDPOINT_URL=https://...modal.run
MODAL_IMAGE_EDIT_ENDPOINT_URL=https://...modal.run
MODAL_UPSCALE_ENDPOINT_URL=https://...modal.run
MODAL_DEWATERMARK_ENDPOINT_URL=https://...modal.run
MODAL_LTX2_ENDPOINT_URL=https://...modal.run
```

Optional but recommended — Cloudflare R2 for reliable file transfer:
```
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=video-toolkit
```

### Step 4: Verify and Quick Test

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/verify_setup.py
```

All tools should show `[x]`. Then run a quick test to confirm the GPU pipeline works:

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/qwen3_tts.py --text "Hello, this is a test." --speaker Ryan --tone warm --output /tmp/video-toolkit-test.mp3 --cloud modal
```

If you get a valid .mp3 file, setup is complete. If it fails, check:
- `.env` has the correct `MODAL_QWEN3_TTS_ENDPOINT_URL`
- Run `uv run tools/verify_setup.py --json` and check `modal_tools` for which endpoints are missing

**Cost:** Modal includes $30/month free compute. A typical 60s video costs $1-3.

---

## Creating a Video

### Step 1: Create Project

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
cp -r templates/product-demo projects/PROJECT_NAME
cd projects/PROJECT_NAME
npm install
```

Templates: `product-demo` (marketing/explainer), `sprint-review`, `sprint-review-v2` (composable scenes).

### Step 2: Write Config

Edit `projects/PROJECT_NAME/src/config/demo-config.ts`:

```typescript
export const demoConfig: ProductDemoConfig = {
  product: {
    name: 'My Product',
    tagline: 'What it does in one line',
    website: 'example.com',
  },
  scenes: [
    { type: 'title', durationSeconds: 9, content: { headline: '...', subheadline: '...' } },
    { type: 'problem', durationSeconds: 14, content: { headline: '...', problems: ['...', '...'] } },
    { type: 'solution', durationSeconds: 13, content: { headline: '...', highlights: ['...', '...'] } },
    { type: 'stats', durationSeconds: 12, content: { stats: [{value: '99%', label: '...'}, ...] } },
    { type: 'cta', durationSeconds: 10, content: { headline: '...', links: ['...'] } },
  ],
  audio: {
    backgroundMusicFile: 'audio/bg-music.mp3',
    backgroundMusicVolume: 0.12,
  },
};
```

Scene types: `title`, `problem`, `solution`, `demo`, `feature`, `stats`, `cta`.

**Duration rule:** Estimate `durationSeconds` as `ceil(word_count / 2.5) + 2`. You will adjust this after generating audio in Step 4.

### Step 3: Write Voiceover Script

Create `projects/PROJECT_NAME/VOICEOVER-SCRIPT.md`:

```markdown
## Scene 1: Title (9s, ~17 words)
Build videos with AI. The product name toolkit makes it easy.

## Scene 2: Problem (14s, ~30 words)
The problem statement goes here. Keep it punchy and relatable.
```

**Word budget per scene:** `(durationSeconds - 2) * 2.5` words. The -2 accounts for 1s audio delay + 1s padding.

### Step 4: Generate Assets

**CRITICAL: All commands below MUST be run from the toolkit root, not the project directory.**

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
```

#### 4a. Background Music

Default provider is **acemusic** (official cloud API, free key). No GPU required. Falls back to Modal/RunPod for self-hosted.

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit

# Using acemusic cloud API (default — best quality, XL Turbo 4B model)
uv run tools/music_gen.py \
  --preset corporate-bg \
  --duration 90 \
  --output projects/PROJECT_NAME/public/audio/bg-music.mp3 \
  --progress json

# Or with custom prompt and thinking mode
uv run tools/music_gen.py \
  --prompt "Subtle ambient tech, soft synth pads" \
  --duration 90 \
  --output projects/PROJECT_NAME/public/audio/bg-music.mp3 \
  --progress json

# Fall back to self-hosted Modal if no acemusic key
uv run tools/music_gen.py \
  --preset corporate-bg \
  --duration 90 \
  --output projects/PROJECT_NAME/public/audio/bg-music.mp3 \
  --cloud modal --progress json
```

Presets: `corporate-bg`, `upbeat-tech`, `ambient`, `dramatic`, `tension`, `hopeful`, `cta`, `lofi`.

Setup: `echo "ACEMUSIC_API_KEY=your_key" >> .env` (get free key at acemusic.ai/api-key).

#### 4b. Voiceover (per-scene)

Generate ONE .mp3 file PER SCENE. Do NOT generate a single voiceover file.

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit

# Scene 01
uv run tools/qwen3_tts.py \
  --text "The voiceover text for scene one." \
  --speaker Ryan --tone warm \
  --output projects/PROJECT_NAME/public/audio/scenes/01.mp3 \
  --cloud modal --progress json

# Scene 02
uv run tools/qwen3_tts.py \
  --text "The voiceover text for scene two." \
  --speaker Ryan --tone warm \
  --output projects/PROJECT_NAME/public/audio/scenes/02.mp3 \
  --cloud modal --progress json

# ... repeat for each scene
```

**Speakers:** `Ryan`, `Aiden`, `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna`, `Sohee`
**Tones:** `neutral`, `warm`, `professional`, `excited`, `calm`, `serious`, `storyteller`, `tutorial`

For voice cloning (needs a reference recording):
```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/qwen3_tts.py \
  --text "Text to speak" \
  --ref-audio assets/voices/reference.m4a \
  --ref-text "Exact transcript of the reference audio" \
  --output projects/PROJECT_NAME/public/audio/scenes/01.mp3 \
  --cloud modal --progress json
```

#### 4c. Scene Images

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/flux2.py \
  --prompt "Dark tech background with blue geometric grid, cinematic lighting" \
  --width 1920 --height 1080 \
  --output projects/PROJECT_NAME/public/images/title-bg.png \
  --cloud modal --progress json
```

Image presets (use `--preset` instead of `--prompt --width --height`):
`title-bg`, `problem`, `solution`, `demo-bg`, `stats-bg`, `cta`, `thumbnail`, `portrait-bg`

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/flux2.py \
  --preset title-bg \
  --output projects/PROJECT_NAME/public/images/title-bg.png \
  --cloud modal --progress json
```

#### 4d. Video Clips — B-Roll & Animated Backgrounds (optional)

Generate AI video clips for b-roll cutaways, animated slide backgrounds, or intro/outro sequences:

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit

# B-roll clip from text
uv run tools/ltx2.py \
  --prompt "Aerial drone shot over a European city at golden hour, cinematic wide angle" \
  --output projects/PROJECT_NAME/public/videos/broll-europe.mp4 \
  --cloud modal --progress json

# Animate a slide/screenshot (image-to-video)
uv run tools/ltx2.py \
  --prompt "Gentle particle effects, soft ambient light shifts, very slight camera drift" \
  --input projects/PROJECT_NAME/public/images/title-bg.png \
  --output projects/PROJECT_NAME/public/videos/animated-title.mp4 \
  --cloud modal --progress json

# Abstract intro/outro background
uv run tools/ltx2.py \
  --prompt "Dark moody abstract background with flowing blue light streaks, bokeh particles, cinematic" \
  --output projects/PROJECT_NAME/public/videos/intro-bg.mp4 \
  --cloud modal --progress json
```

Use in Remotion compositions with `<OffthreadVideo>`:
```tsx
<OffthreadVideo src={staticFile('videos/broll-europe.mp4')} />
```

**LTX-2 rules:**
- Max ~8 seconds per clip (193 frames at 24fps). Default is ~5s (121 frames).
- Width/height must be divisible by 64. Default: 768x512.
- ~$0.20-0.25 per clip, ~2.5 min generation time.
- Cold start ~60-90s. Subsequent clips on warm GPU are faster.
- Generated audio is ambient only — use voiceover/music tools for speech and music.
- ~30% of generations may have training data artifacts (logos/text). Re-run with `--seed` to vary.

#### 4d-chain. Chained Video Sequences (visual continuity)

Generate a sequence of video clips where each scene flows from the last frame of the previous one. **This runs as a single command** — no manual nudging between scenes.

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit

# Chain scenes 1-30 from a directory of FLUX images
uv run tools/chain_video.py \
  --scenes-dir projects/PROJECT_NAME/public/images/scenes/ \
  --output-dir projects/PROJECT_NAME/public/videos/chain/ \
  --prompt "Cinematic continuation, flowing transition" \
  --start 1 --end 30 \
  --progress json

# Resume from scene 10 (skips existing files automatically)
uv run tools/chain_video.py \
  --scenes-dir projects/PROJECT_NAME/public/images/scenes/ \
  --output-dir projects/PROJECT_NAME/public/videos/chain/ \
  --start 10 --end 30 \
  --progress json

# Per-scene prompts from JSON file
uv run tools/chain_video.py \
  --scenes-dir projects/PROJECT_NAME/public/images/scenes/ \
  --output-dir projects/PROJECT_NAME/public/videos/chain/ \
  --prompts-file projects/PROJECT_NAME/scenes.json \
  --progress json

# Chain from an existing clip (no scene images needed)
uv run tools/chain_video.py \
  --first-clip output/chain-04.mp4 \
  --output-dir output/ \
  --start 5 --end 30 \
  --prompt "Celtic mythology, flowing transition" \
  --progress json
```

**Prompts file format** (`scenes.json`):
```json
{"1": "Ancient stone circle at dawn", "2": "Celtic spirals emerge from stone", "3": "Portal opens with golden light"}
```

**Chain rules:**
- Extracts last frame from scene N, feeds as `--input` to scene N+1 via LTX-2
- Skips scenes that already exist on disk (safe to resume)
- Falls back to scene images from `--scenes-dir` if chaining fails
- Use `--prefix` to set output filename prefix (default: `chain`)
- ~2.5 min per scene, ~$0.20-0.25 per clip
- Extra args (e.g. `--negative-prompt`, `--seed`) are passed through to ltx2.py

**CRITICAL: Style drift in chained sequences.** LTX-2 has ~30% training data contamination (anime/Asian content). Generic prompts like "cinematic transition" will drift toward anime aesthetics within 5-10 chained scenes. To prevent this:

1. **ALWAYS use `--prompts-file`** with specific per-scene prompts — never a single generic prompt for the whole chain
2. **ALWAYS add `--negative-prompt`** to exclude unwanted styles:
   ```
   --negative-prompt "anime, manga, asian, cartoon, illustration, watermark, text, logo"
   ```
3. Each per-scene prompt should include **strong style anchors** (e.g. "Irish landscape, Celtic knotwork, oil painting style") not just subject descriptions

**CRITICAL: Run with `yieldMs` for live progress reporting.** Don't break it into per-scene tool calls — OpenClaw's agent run ends between calls, causing the sequence to stall. Instead, use `exec` with `yieldMs` so you stay in the loop and can relay progress to the user:

```
exec command:"cd ~/.openclaw/workspace/claude-code-video-toolkit && uv run tools/chain_video.py --scenes-dir /path/to/images/ --output-dir /path/to/output/ --prompts-file scenes.json --progress json" yieldMs:10000
```

**How this works:**
- `yieldMs:10000` returns control to you every 10 seconds
- You read the `--progress json` output (JSON Lines on stderr with stage/pct/msg)
- Report progress to the user ("Scene 05/30 complete, 17%")
- Then poll again: `process action:poll sessionId:<id>`
- Repeat until `"stage":"complete"` appears

**This is the correct pattern for ALL long-running tool commands** (chain_video, batch flux, batch sadtalker, etc.). Never use `bash background:true` and forget about it — use `exec` + `yieldMs` + `process poll` loop so you can report progress live.

#### 4e. Talking Head Narrator (optional)

Generate a presenter portrait, then animate per-scene clips:

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit

# 1. Generate portrait
uv run tools/flux2.py \
  --prompt "Professional presenter portrait, clean style, dark background, facing camera, upper body" \
  --width 1024 --height 576 \
  --output projects/PROJECT_NAME/public/images/presenter.png \
  --cloud modal --progress json

# 2. Generate per-scene narrator clips (one per scene, NOT one long video)
uv run tools/sadtalker.py \
  --image projects/PROJECT_NAME/public/images/presenter.png \
  --audio projects/PROJECT_NAME/public/audio/scenes/01.mp3 \
  --preprocess full --still --expression-scale 0.8 \
  --output projects/PROJECT_NAME/public/narrator-01.mp4 \
  --cloud modal --progress json

# Repeat for each scene that needs a narrator
```

**SadTalker rules — follow these exactly:**
- **ALWAYS** use `--preprocess full` (default `crop` outputs a square, wrong aspect ratio)
- **ALWAYS** use `--still` (reduces head movement, looks professional)
- **ALWAYS** generate per-scene clips (6-15s each), NEVER one long video
- Processing: ~3-4 min per 10s of audio on Modal A10G
- `--expression-scale 0.8` keeps expressions subtle (range 0.0-1.5)

#### 4e. Image Editing (optional)

Create scene variants from existing images:

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/image_edit.py \
  --input projects/PROJECT_NAME/public/images/title-bg.png \
  --prompt "Make it darker with red tones, more ominous" \
  --output projects/PROJECT_NAME/public/images/problem-bg.png \
  --cloud modal --progress json
```

#### 4f. Upscaling (optional)

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/upscale.py \
  --input projects/PROJECT_NAME/public/images/some-image.png \
  --output projects/PROJECT_NAME/public/images/some-image-4x.png \
  --scale 4 --cloud modal --progress json
```

### Step 5: Sync Timing

**ALWAYS do this after generating voiceover.** Audio duration differs from estimates.

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
for f in projects/PROJECT_NAME/public/audio/scenes/*.mp3; do
  echo "$(basename $f): $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")s"
done
```

Update each scene's `durationSeconds` in `demo-config.ts` to: `ceil(actual_audio_duration + 2)`.

Example: if `01.mp3` is 6.8s, set scene 1 `durationSeconds` to `9` (ceil(6.8 + 2) = 9).

### Step 6: Review Still Frames

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit/projects/PROJECT_NAME
npx remotion still src/index.ts ProductDemo --frame=100 --output=/tmp/review-scene1.png
npx remotion still src/index.ts ProductDemo --frame=400 --output=/tmp/review-scene2.png
```

Check: text truncation, animation timing, narrator PiP positioning, background contrast.

### Step 7: Render

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit/projects/PROJECT_NAME
npm run render
```

**Output:** `out/ProductDemo.mp4`

---

## Composition Patterns

### Per-Scene Audio

Use per-scene audio with a 1-second delay (`from={30}` = 30 frames = 1s at 30fps):

```tsx
<Sequence from={30}>
  <Audio src={staticFile('audio/scenes/01.mp3')} volume={1} />
</Sequence>
```

### Per-Scene Narrator PiP

```tsx
<Sequence from={30}>
  <OffthreadVideo
    src={staticFile('narrator-01.mp4')}
    style={{ width: 320, height: 180, objectFit: 'cover' }}
    muted
  />
</Sequence>
```

**ALWAYS use `<OffthreadVideo>`, NEVER `<video>`.** Remotion requires its own component for frame-accurate rendering.

### Transitions

```tsx
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { glitch } from '../../../lib/transitions/presentations/glitch';
import { lightLeak } from '../../../lib/transitions/presentations/light-leak';
```

**NEVER import from `lib/transitions` barrel** — import custom transitions from `lib/transitions/presentations/` directly.

---

## Progress Reporting

All cloud GPU tools support structured progress output for automated monitoring.

### Usage

Add `--progress json` to any tool command to get JSON Lines on stderr:

```bash
cd ~/.openclaw/workspace/claude-code-video-toolkit
uv run tools/music_gen.py \
  --preset corporate-bg --duration 60 \
  --output projects/PROJECT_NAME/public/audio/bg-music.mp3 \
  --progress json
```

### Output Format

Each line on stderr is a JSON object:

```json
{"ts":"14:23:15","stage":"submit","msg":"Sending to acemusic.ai (XL Turbo 4B, thinking: on)...","pct":null,"elapsed":0.0}
{"ts":"14:23:30","stage":"waiting","msg":"Waiting for acemusic.ai response... (15s)","pct":null,"elapsed":15.0}
{"ts":"14:23:45","stage":"waiting","msg":"Waiting for acemusic.ai response... (30s)","pct":null,"elapsed":30.0}
{"ts":"14:24:02","stage":"complete","msg":"Saved: bg-music.mp3 (245 KB, 60.1s)","pct":100,"elapsed":47.3}
```

### Stages

| Stage | Meaning |
|-------|---------|
| `submit` | Job sent to provider |
| `queue` | RunPod: waiting for GPU |
| `processing` | RunPod: GPU processing |
| `waiting` | Heartbeat during synchronous calls (acemusic, Modal) |
| `complete` | Job finished successfully |
| `error` | Something failed — check `msg` for details |
| `item` | Multi-item progress (e.g., scene 3/7) — `pct` is populated |
| `cost` | Estimated cost for the operation |

### Behaviour by Provider

- **acemusic**: Emits `submit` → periodic `waiting` heartbeats (every 15s) → `complete`
- **RunPod**: Emits `submit` → `queue` → `processing` → `complete` (on each poll)
- **Modal**: Emits `submit` → periodic `waiting` heartbeats → `complete`

Default mode (`--progress human`) shows the same events as colored terminal output — no change to existing behaviour.

---

## Error Recovery

| Problem | Solution |
|---------|----------|
| Tool command fails with "No module named..." | Run `uv sync` from toolkit root and invoke tools via `uv run` |
| "MODAL_*_ENDPOINT_URL not configured" | Check `.env` has the endpoint URL. Run `uv run tools/verify_setup.py` |
| SadTalker output is square/cropped | You forgot `--preprocess full`. Re-run with that flag |
| Audio too short/long for scene | Re-run Step 5 (sync timing) and update config |
| `npm run render` fails | Make sure you're in the project dir, not toolkit root. Run `npm install` first |
| "Cannot find module" in Remotion | Check import paths. Custom components use `../../../lib/` relative paths |
| Cold start timeout on Modal | First call after idle takes 30-120s. Retry once — second call uses warm GPU |
| SadTalker client timeout (long audio) | The client HTTP request can time out before Modal finishes. **Modal still uploads the result to R2.** Check `sadtalker/results/` in the `video-toolkit` R2 bucket for the output. Use `uv run python -c "import boto3; ..."` with the R2 creds from `.env` to list and generate a presigned URL |

---

## Cost Estimates (Modal)

| Tool | Typical Cost | Notes |
|------|-------------|-------|
| Qwen3-TTS | ~$0.01/scene | ~20s per scene on warm GPU |
| FLUX.2 | ~$0.01/image | ~3s warm, ~30s cold |
| ACE-Step | ~$0.02-0.05 | Depends on duration |
| SadTalker | ~$0.05-0.20/scene | ~3-4 min per 10s audio |
| Qwen-Edit | ~$0.03-0.15 | ~8 min cold start (25GB model) |
| RealESRGAN | ~$0.005/image | Very fast |
| LTX-2.3 | ~$0.20-0.25/clip | ~2.5 min per 5s clip, A100-80GB |

**Total for a 60s video:** ~$1-3 depending on scenes and narrator clips.

Modal Starter plan: $30/month free compute. Apps scale to zero when idle.
