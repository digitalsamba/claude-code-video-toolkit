# EchoMimicV3 (Modal)

Audio-driven talking head generation with [EchoMimicV3](https://github.com/antgroup/echomimic_v3)
(Ant Group, Apache 2.0, AAAI 2026). Runs alongside `docker/modal-sadtalker/`, which is built on a
model unmaintained since 2023 but remains much cheaper and faster.

**Status: shipping.** User-facing docs are `docs/echomimic3.md`; this file keeps the build
detail, the measurements, and the traps. Still not run at full narration length.

## Why this model

| | SadTalker | EchoMimicV3-Flash |
|---|---|---|
| Released | 2023, unmaintained | Flash variant Jan 2026 |
| Licence | Apache 2.0 | Apache 2.0 |
| Params | ~0.3B (warp-based) | 1.3B (Wan2.1-Fun diffusion) |
| VRAM | ~8GB | 12GB (Flash) / 16GB with offload |
| Aspect ratio | square crop unless `--preprocess full` | follows the input image |
| Motion | head + light expression | head, upper body, gestures |
| Speed | fast (~$0.04 / 30s) | **28.9x realtime measured** (~$0.11 / 12s) |

The alternatives considered and rejected for this slot: **LongCat-Video-Avatar-1.5** (MIT, newer,
probably better output, but an A100/H100 tier); **InfiniteTalk** (Apache 2.0, best for long-form,
Wan-14B-based); **OmniHuman-1.5** (best quality, closed weights, API-only); hosted fal.ai endpoints
($0.15–0.30 per video-second, i.e. $27+ for a 3-minute narrator).

## Measured results (2026-08-27)

12s of narration, `--steps 5 --size 640`, 1024x576 presenter crop, Modal A10 (22.06GB usable):

| Metric | Value |
|---|---|
| Output | **848x480 — 16:9 preserved**, vs SadTalker's 512x512 square crop from the same portrait |
| Wall clock | 347s for 12.0s of video = **28.9x realtime** |
| Segments | 4, at a steady 78-82s each (exactly what the loop dry-run predicted) |
| Resident VRAM | 0.0GB idle; fits 24GB only via `enable_model_cpu_offload` |
| Cost | ~$0.11 per 12s ≈ **$0.0089 per second of output** |

Extrapolated to the 199s `pluribus-sprint` narration: **~96 minutes and ~$1.76**, against roughly
$0.27 for SadTalker. Call it **~6.5x the cost and far longer wall clock** — real, but not
prohibitive for a handful of narrator tracks per video.

**Quality:** clear viseme articulation and natural head rotation where SadTalker with `--still`
is near-frozen. Identity holds against the source portrait (frame 0 is essentially the input) and
across all four segments. **No visible seam or identity pop** at the segment joins (frames 73,
146, 219) — the overlap cross-fade does its job at these settings.

## Tuning matrix (2026-08-27) — and the bug it exposed

Six variants on a 5.8s clip (Rob's SadTalker render: frame 0 as the still, its audio as the driver),
one factor changed each from a fixed baseline, seed pinned to 43.

| Variant | mouth motion | sync_r | Human verdict |
|---|---|---|---|
| A baseline (audio CFG 3.0, cn, 8 steps) | 3.488 | 0.194 | best-validated config |
| B audio CFG 1.8 | 3.326 | 0.291 | **rejected — eye artifact** |
| C `--wav2vec english` | 2.264 | 0.157 | clearly worst |
| D prompt = `"A person is speaking."` | 3.630 | 0.201 | fine |
| E steps 5 | 4.035 | 0.168 | fine, most motion |
| F rich descriptive prompt | 3.387 | 0.205 | marginally preferred |

**Settled:** `--wav2vec english` is worse despite English audio — under-articulates throughout
(motion 2.26 vs 3.5) and visibly sits half-open. Keep `chinese` regardless of language; the Flash
model was trained with it.

**Do NOT trust `sync_r` alone.** It scores a *mouth crop only* and is structurally blind to eye,
hair and background artifacts. It ranked B highest — and B is the variant with the visible defect.
Any future scoring needs a whole-face check, or just human eyes.

### Fixed: segment re-anchoring could latch a blink

Variant B holds the eyes closed across frames ~72-80. Segment 2 starts at frame 73. The loop
re-anchors each segment on the last `overlap` frames of the previous one, so **if those anchor
frames land mid-blink, the next segment starts from closed eyes and holds them.** A blinks at f80
and recovers; B latched.

This is a flaw in the chunking, not an audio-CFG property — low audio CFG probably worsens it
(less audio drive, more deference to the anchor pose) but does not cause it.

**Fixed by `--anchor-retreat` (default 6).** Detecting eyes would need the face-landmark stack
this image deliberately omits, so `_pick_anchor_retreat` instead scores candidate anchor windows
by how much motion they contain in the **upper half** of the frame — where blinks live and mouth
movement does not — and backs off up to N frames to anchor on the calmest one. A blink is the
largest short transient up there, so it scores worst and gets skipped. It only pays the cost of
regenerating those frames when there is a clear improvement (< 0.8x the score at retreat 0), so
clips with clean seams regenerate nothing. `--anchor-retreat 0` restores the old behaviour.

Anchoring on a settled pose is the better default regardless: a continuation segment has to
extrapolate from whatever frames it is handed.

The selection logic was verified offline against synthetic blinks (detects a blink in the anchor
window, leaves clean windows alone, chooses a window that excludes the blink frames, clamps on
short clips) — no GPU needed. Whether it *looks* fixed on a real latching clip is still unwatched.

Note the earlier "no visible seam" finding was about *identity* continuity, which did hold. It did
not rule out a pose getting stuck across the join.

**Retracted from earlier in this session:** the recommendation to default `--audio-guidance-scale`
to 2.0. It rested on `sync_r`, which the above invalidates as a selector.

**Still unverified:** behaviour at full narration length (drift over 60+ segments), and whether the
prompt matters (F edged it by eye, but the metric that called it inert is the one that failed).

## Is the gap visible at NarratorPiP size? (2026-08-30)

Controlled A/B, same still and same 12s of audio: `conal-narrator.png` (772x440) against the
existing SadTalker render made from it. Both downscaled to 240x135 — NarratorPiP `sm`, ~3% of a
1080p frame — and measured for how much motion survives the downscale.

| | whole frame | mouth band | eye band |
|---|---|---|---|
| SadTalker (`--preprocess full --still`) | 0.325 | 0.130 | 0.359 |
| EchoMimicV3 (`--steps 5 --size 640`) | 1.151 | 0.800 | 1.208 |
| ratio | **3.5x** | **6.2x** | **3.4x** |

The difference is not washed out by the downscale. This is a *motion* measure, not a quality
one — more motion is not automatically better, and the tuning matrix above is the standing
warning against reading it that way — but it settles the narrow question the test was for:
whatever gap exists is still there at PiP size rather than being invisible.

**Human verdict on the same pair: "echo is significantly better but sadtalker isn't terrible
either."** So both tools stay. EchoMimicV3 is the better picture at any size; SadTalker remains
good enough for a small overlay, and at ~1/6th the cost and a fraction of the wall clock it keeps
earning its place for PiP boxes, drafts, and generating several takes to choose between. Note the
SadTalker side was rendered `--preprocess full --still`, so it was already 16:9 and deliberately
steady — this compared articulation, not framing.

That run also re-measured throughput at 22.8x realtime (274s of GPU for 12.0s of output),
against 28.9x in the original measurement.

## Deploy

```bash
uv sync --extra modal && uv run modal setup

# One-off: fill the weights volume (~26GB, ~10 min). Must run before the first deploy.
uv run modal run docker/modal-echomimic3/app.py::populate_weights

uv run modal deploy docker/modal-echomimic3/app.py
```

Then put the printed URL in `.env`:

```
MODAL_ECHOMIMIC3_ENDPOINT_URL=https://....modal.run
```

Weights (Wan2.1-Fun-V1.1-1.3B-InP, the `echomimicv3-flash-pro` transformer, and both
wav2vec2 encoders) live in a **Modal Volume**, not in the image — the one place this app
diverges from the other six. Rationale and numbers in #76; the short version is that
rebuild after a dependency change is 1.8-8.2s instead of 79-385s, while cold start and
generation speed are unchanged. Because the weights are no longer pinned by the image,
the upstream repo ref and all four model revisions are pinned by SHA in `app.py` — bump
them deliberately and re-run `populate_weights`.

For the baked variant (deploys separately as `video-toolkit-echomimic3-baked`):

```bash
ECHOMIMIC_WEIGHTS=image uv run modal deploy docker/modal-echomimic3/app.py
```

## Use

```bash
# Plain generation
uv run tools/echomimic3.py --image portrait.png --audio vo.mp3 --output talking.mp4

# NarratorPiP settings — 16:9 in, 16:9 out, cheap 5-step pass
uv run tools/echomimic3.py \
    --image presenter_16x9.png --audio scene_01.mp3 \
    --steps 5 --size 640 --output narrator.mp4

# Side-by-side against the existing SadTalker clip for the same inputs
uv run tools/echomimic3.py \
    --image presenter_16x9.png --audio scene_01.mp3 \
    --output narrator_echo.mp4 --compare narrator_sadtalker.mp4
```

## Getting it working — four blockers, for anyone repeating this

1. **OOM on load.** `pipeline.to("cuda")` consumed 21.98 of 22.06GB before inference allocated
   anything, and died in `@modal.enter()`. Fixed with `enable_model_cpu_offload()`. Upstream's
   `infer_flash.py` parses a `GPU_memory_mode` flag but never applies it, so following upstream
   literally does not work on a 24GB card.
2. **`chinese-wav2vec2-base` ships `pytorch_model.bin`**, and transformers >=4.51.3 refuses
   `torch.load` below torch 2.6 (CVE-2025-32434) — a hard failure, not a warning. Converted to
   safetensors at build time rather than bumping torch, which would flip `torch.load`'s
   `weights_only` default and break EchoMimic's own `.pth` loads for VAE/text encoder/CLIP.
3. **`hidden_states=None` from wav2vec.** transformers 5.x drives `output_hidden_states` from
   config, not the kwarg EchoMimic's `Wav2Vec2Model` subclass passes — so the audio embeddings,
   which *are* the lip sync, came back empty. Setting `config.output_hidden_states` did **not**
   fix it; pinning `transformers==4.49.0` did.
4. **Version floors are a trap here.** The repo's `diffusers>=0.30.1` / `transformers>=4.46.2`
   resolve to releases its vendored code does not survive. Both are now pinned exactly.

## What to check next

1. **Segment seams and drift.** Upstream's `infer_flash.py` generates a single 81-frame (3.2s)
   clip and silently truncates longer audio; the Flash pipeline does not take the long-video kwargs
   the preview pipeline does. So the segment loop lives in `app.py` instead — re-anchoring each
   segment on the last `overlap` frames and cross-fading the seam, mirroring `infer_preview.py`.

   At the defaults each segment advances only `81 - 8 = 73` frames (2.9s), so a 30s narrator is
   **11 full diffusion passes** and a 3-minute one is 62. That, not the per-step cost, is what
   makes this expensive — and it is the number to attack first if the realtime factor is bad
   (raise `--video-length` until VRAM complains). Watch a 30s clip for identity drift and for pops
   at ~2.9s intervals; a larger `--overlap` softens seams at the cost of more segments.

   The loop's frame arithmetic was dry-run separately across audio durations 0.5s–180s and sweeps
   of both `--video-length` and `--overlap`: coverage is complete, the silence padding always
   covers the rounded-up final segment, and there is no runaway. That is bookkeeping only — it
   says nothing about whether the output *looks* right.
   `--anchor-retreat` adds a little to this: each retreat discards frames that then have to be
   regenerated. It only fires on a detected transient, so the common case costs nothing.
2. **`--audio-guidance-scale`.** Defaulted to 3.0 to match `run_flash.sh`, but the upstream README
   recommends 1.8–2.0 for lip sync. Try both — and judge by eye, since the metric that would
   otherwise decide it is the discredited one.
3. **Watch a real latching clip with `--anchor-retreat` on.** The selection logic is verified
   against synthetic blinks, but variant B (audio CFG 1.8) is the known reproducer and has not
   been re-run.

(`--wav2vec chinese` vs `english` was on this list and is now settled — see the tuning matrix.)

## Known deviations from upstream

- `_build_inputs` reimplements `src.utils.get_image_to_video_latent2` because that helper calls
  `.resize()` on its argument before checking whether it is a list, so it raises `AttributeError`
  on the multi-frame re-anchoring the segment loop needs.
- The image omits `tensorflow`, `retina-face`, `gradio`, `decord` and `moviepy` from upstream's
  `requirements.txt`. None are reachable from the Flash path — tensorflow and retina-face are only
  used by `src.face_detect` for the preview variant's `ip_mask`. Add them back if the preview
  variant is ever wired up.
- `REPO_REF` is pinned to a commit SHA, and so are all four model revisions. An unpinned ref
  silently re-resolves on rebuild, which is how #71/#74 happened to flux2. Pinning matters more
  here than in the baked apps, because weights in a Volume are not tied to the image at all.
  Fetching by SHA needs `git init` + `fetch --depth 1 <sha>`; `clone --branch` only takes a
  branch or tag name.
