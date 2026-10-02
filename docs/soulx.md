# SoulX-FlashHead — Talking Head Video Generation

Generates a talking head video from one portrait image plus an audio track.
[SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul AI Lab,
Apache 2.0, 1.3B params, arXiv 2602.07449) is the toolkit's default talking
head generator.

It preserves the input image's aspect ratio, so a 16:9 presenter image comes
back 16:9 with no `--preprocess` workaround, and it holds identity across long
renders — which is the reason it is the default.

It ships in two variants, both served by the same endpoint:

- **Pro** (default) — final quality. What everything below measures unless it
  says otherwise.
- **Lite** (`--model lite`) — the draft mode. Same transformer, a far more
  compressive VAE, ~7x faster per second of output and no compile wait.
  Supersedes SadTalker for drafts and for picking between takes.

## Demo

[![SoulX-FlashHead talking head demo](../assets/readme-thumbs/soulx-talking-head.jpg)](https://demos.digitalsamba.com/video/soulx-talking-head.mp4)

**[80 seconds from a single still](https://demos.digitalsamba.com/video/soulx-talking-head.mp4)**
— one 1024x1024 photo and an audio track, no video input, rendered at 640x640
in one pass.

The length is the point. Watch past 50s, where a segment-chained model would
normally start shedding detail: the face, the beard and the chisels on the back
wall are all still there at 80s. Measured at 97-100% of frame-zero sharpness
throughout.

## Quick Start

```bash
# Basic — aspect ratio follows the input image
uv run tools/soulx.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

# NarratorPiP (16:9 in, 16:9 out)
uv run tools/soulx.py \
    --image presenter_16x9.png --audio scene_01.mp3 \
    --size 768 --output narrator.mp4

# Exact dimensions rather than a long edge
uv run tools/soulx.py -i p.png -a vo.mp3 --width 768 --height 432 -o narrator.mp4

# Fast draft with Lite (its sizes sit on a 32 grid; 512x288 is exact 16:9)
uv run tools/soulx.py -i presenter_16x9.png -a scene_01.mp3 \
    --model lite --width 512 --height 288 -o draft.mp4

# Side by side against an existing render of the same inputs
uv run tools/soulx.py -i p.png -a vo.mp3 -o new.mp4 --compare old.mp4
```

## Choosing a variant

| | SadTalker | SoulX Lite | SoulX Pro |
|---|---|---|---|
| Flag | `tools/sadtalker.py` | `--model lite` | default |
| Method | warp-based | diffusion, 4 steps, LTX-Video VAE | diffusion, 4 steps, Wan2.1 VAE |
| Aspect ratio | square unless `--preprocess full` | follows the input (32 grid) | follows the input (16 grid) |
| Identity over a long take | stable (it barely moves) | unmeasured past 3s | **stable — 97% at 70s** |
| Motion | head + light expression | head, shoulders, natural expression | head, shoulders, natural expression |
| Cost per second of output | ~$0.0014 | **~$0.0004** | ~$0.0024 |
| Speed | near realtime | 0.5x realtime at 512x288, 1.2x at 640x640, no compile | ~7.9x realtime, plus a one-off compile |
| Provider | RunPod or Modal | Modal | Modal |

**Use Pro** for anything a viewer actually watches: a narrator large in frame,
a held shot, a finished video.

**Use Lite** for drafts and for generating many takes to choose between. It
replaces SadTalker in that role. It is cheaper per second, it keeps the input
aspect ratio, and it shares the Pro final's endpoint, crop logic and
conditioning path, so the move from draft to final changes one flag.

**Use SadTalker** when Modal is unavailable. It is the only talking head with a
RunPod path.

What "cheaper" means in practice: the per-second figures are steady state. A
call to a cold container also pays the container start, Pro's load (~27s,
always, since Pro is what the container starts with) and Lite's (~12s). After
that the container idles for its 10-minute `scaledown_window` before it shuts
down, and that idle time is billed. So one isolated 3s draft costs about as
much as the idle tail (~$0.18), whatever the variant. Batch drafts into one
session: once the container is warm, a 3s Lite draft took **6.3s end to end**.

## The drift problem, and why this model

Segment-chained talking heads re-anchor each segment on the previous segment's
output. That makes the failure *absorbing*: one bad segment poisons everything
after it, and quality does not degrade gracefully so much as fall off a cliff.

SoulX-FlashHead is trained with **Oracle-Guided Bidirectional Distillation** —
the student generates from its own history while a teacher sees ground-truth
motion — which targets exactly that failure.

It was verified rather than taken on trust. A controlled A/B against
EchoMimicV3, *same photo, same 80s audio, same 544x736 output*, measuring
high-frequency energy as a share of frame zero:

| t | SoulX-FlashHead | EchoMimicV3 |
|---|---|---|
| 30s | 97% | 87% |
| 50s | 97% | 75% |
| 70s | **97%** | **50%** |

At 70s SoulX is still a sharp, correctly framed, unmistakably identical face
with the subject's glasses intact. EchoMimicV3 at 70s on the same input is a
featureless smear with no eyes. Confirmed by contact sheet, not by the number
alone — see [the metrics warning](#judging-quality) below.

Repeated on a second subject at 640x640 — different colouring, no glasses, and
a busy workshop background — which held **97-100% across the full 80s**,
including the fine background detail that drifts first. So it is a property of
the model rather than of one photograph.

## Parameters

### Core settings

| Flag | Default | Notes |
|------|---------|-------|
| `--image` / `-i` | required | Portrait. 16:9 for NarratorPiP |
| `--audio` / `-a` | required | Any ffmpeg-readable audio |
| `--model` | pro | `pro` for finals, `lite` for fast drafts |
| `--size` | 768 | Target long edge; aspect follows the image |
| `--width` / `--height` | — | Exact dimensions instead of `--size`. Both or neither. Multiples of 16 (Pro) or 32 (Lite) |
| `--seed` | 42 | |
| `--face-crop` | off | Upstream's face detect + crop |
| `--compare` | — | Also write a labelled side-by-side against an existing render |

### Resolution rules

The grid is the VAE's spatial stride times the transformer's patch size, so it
differs by variant:

| Variant | VAE | Stride × patch | Grid |
|---|---|---|---|
| Pro | Wan2.1 | 8 × 2x2 | **16** |
| Lite | LTX-Video | 32 × 1x1 (no patching) | **32** |

This is checked by the tool and again by the endpoint, because **nothing
upstream validates it**. `target_size` flows straight into
`lat_h = target_h // vae_stride[1]`, so an off-grid size floors silently and
desyncs the latent grid from the pixel grid — a wrong render, not an error.

`--size` handles this for you: it snaps to the right grid for `--model` while
preserving aspect. That means the same `--size 768` on a 16:9 image gives
`768x432` on Pro and `768x448` on Lite (a slightly tighter crop).

Sizes that are legal and useful:

- **Pro:** `768x432` and `640x368` (16:9), `544x736` (3:4 portrait),
  `640x640` / `512x512` (square)
- **Lite:** `512x288` and `1024x576` (exact 16:9), `768x448`, `544x736`,
  `640x640` / `512x512`. `768x432` is *illegal* on Lite.

Lite's grid is 32, not the 64 these docs used to claim: `Model_Lite/config.json`
has `patch_size [1,1,1]`, so there is no 2x2 patch to double the stride.
Verified by round-tripping 512x288 and 768x416 through the LTX VAE and by a
clean 512x288 Lite render.

## Image guidelines

Same as any talking head model:

- Face 30–70% of the frame
- Front-facing, eyes open, neutral or slightly smiling
- 512px+ on the short edge
- 16:9 for NarratorPiP

A closed-mouth, neutral source generally beats a frame grab mid-sentence.

## Performance and cost

Measured on A10G (24GB), Pro, compile on, no flash-attn:

| Output size | Median per 28-frame chunk | Realtime factor |
|---|---|---|
| 768x432 | 7.14s | ~6.4x |
| 544x736 | 8.87s | ~7.9x |
| 640x640 | 8.93s | ~8.0x |

At Modal's A10G rate that is roughly **$0.0024 per second of output** — an 80s
render costs about $0.20 and takes ~10 minutes of GPU beyond the compile.

A 2026-10-02 re-run of Pro at 640x640 measured **9.48s** per chunk, against
8.93s above. That is +6% from a single run, so it cannot be told apart from
run-to-run noise. Lite was resident in that container at the time; it is now
evicted before Pro renders.

Lite, on the same A10G, eager (no compile), no flash-attn:

| Output size | Median per 24-frame chunk | Realtime factor | Per second of output |
|---|---|---|---|
| 512x288 | 0.46s | ~0.5x (faster than realtime) | ~$0.00015 |
| 640x640 | 1.14s | ~1.2x | ~$0.0004 |

Per denoise step that is 0.16s for Lite against 1.36s for compiled Pro at
640x640. Lite's VAE compresses 32x spatially and the transformer does not
patchify, so a chunk is ~7x fewer tokens through the same 1.3B transformer.
Upstream quotes 96 FPS on an RTX 4090; on the A10G this measures ~52 FPS at
512x288 and ~21 FPS at 640x640.

Peak VRAM, from `peak_vram_gb` in the endpoint's stats: Lite 8.6GB at
512x288 and 9.7GB at 640x640, both with Pro resident too. Pro 17.5GB at
640x640 with Lite also resident; Lite holds an estimated ~4.5GB of that.

**These numbers are a floor, not the model's ceiling.** `flash_attn` and
`sageattention` are optional (the model try/excepts both and falls back to
PyTorch SDPA) and neither is installed, while upstream's quoted 10.8 FPS on an
RTX 4090 assumes one of them. Installing flash-attn is the obvious next
optimisation if generation time ever becomes the bottleneck.

### torch.compile

Upstream enables `torch.compile` for both the model and the VAE. It costs
**~600s on the first call into a container** and then saves roughly 40% per
chunk.

The important part: **it recompiles whenever the resolution changes.** So

- a batch of per-scene narrator clips **at one size** pays it once, and the
  container's 10-minute idle window is set generously so that batch reuses it;
- switching resolution per scene pays it *per scene*, which is far worse than
  the generation itself.

Pick one narrator resolution per project. Deploy with `SOULX_COMPILE=0` for a
genuinely one-off render at an unusual size.

A cold Pro call on 2026-10-02 took **~1,240s** before its second chunk, well
above the 590-780s seen earlier. Roughly 250s of that, by subtraction, went
on compiling the VAE encode inside `get_base_data`. The first chunk took 982s,
including **two full model compiles** of ~293s each on denoise steps 1 and 2,
then the VAE decode and the motion-frame re-encode.
The second model compile is probably upstream's `self.freqs`: a plain CPU
attribute that `forward` moves to the GPU on its first call, which breaks the
guard and forces a recompile. Not fixed here and not verified.

The ~40% saving did not show up on re-measurement either. Same day, same
image and inputs, compile off (`SOULX_COMPILE=0`, in a different container),
Pro ran **10.15s** per chunk at 640x640 against **9.48s** compiled: a 7%
saving. At 7%, a ~1,240s compile pays back only after ~30 minutes of output.
That is one run each, in different containers, so treat it as a reason to
re-measure before trusting either figure, not as a verdict. Peak VRAM was
lower eager too: 8.2GB, against an estimated ~13GB compiled.

**Lite never compiles.** Its eager chunks are 0.5-1.1s, so a 600s+ compile
would take hours of output to pay back. Changing resolution between Lite
drafts costs nothing.

## Judging quality

**Do not score a talking head with an automated similarity metric.** Two have
now produced confidently wrong answers on this exact question:

1. A mouth-crop sync score ranked *highest* the one variant with a visible eye
   defect — it scores a mouth crop and is structurally blind to eyes and hair.
2. MAE-vs-frame-0 *plateaued* straight through a run where the face collapsed
   to a smear, because a smear scores about as far from frame 0 as a
   drifted-but-valid face does. It cannot tell "different person" from "no
   person".

Use **high-frequency energy over time** plus a **contact sheet reviewed by
eye**. When a metric and a contact sheet disagree, the contact sheet is right.

One reading note: sharpness *above* 100% is not a model winning. It means
contrast and detail are being added that were never in the source photo — a
stylisation signal, and usually an early drift warning.

## Setup

```bash
uv sync --extra modal && uv run modal setup

# One-off: fill the weights volume (14.7 GB, ~5 min)
uv run modal run docker/modal-soulx/app.py::populate_weights

uv run modal deploy docker/modal-soulx/app.py
# Add the printed generate_web URL to .env:
#   MODAL_SOULX_ENDPOINT_URL=https://....modal.run
```

Modal-only. There is no RunPod path.

### Weight storage

Weights live in a **Modal Volume** rather than baked into the image, the same
call as `modal-echomimic3` (#76) and for the same reason: a code change
redeploys in **~2.3s** instead of re-downloading 15GB. Volume weights are not
tied to the image, so the upstream repo ref and both model revisions are pinned
by SHA — nothing else stops image and weights drifting apart.

## Troubleshooting

**A long render dies with "cancelled by user or a failure".**
`modal run` keeps a client attached and cancels the call when that client dies
— including when the laptop sleeps, and `--detach` does not save the in-flight
call. The tool's normal path is an HTTP endpoint and is unaffected. For direct
`modal` invocations of a long render, use `.spawn()` against the deployed app;
renders are also persisted to the `soulx-out` volume for exactly this reason.

**"must be a multiple of 16" / "multiple of 32 for --model lite".**
See [Resolution rules](#resolution-rules). Use `--size` and let it snap. A size
that is legal for Pro, such as 768x432, can be illegal for Lite.

**"asked for model=lite but the endpoint rendered pro".**
The endpoint was deployed before Lite existed and ignored the field. Redeploy
(`uv run modal deploy docker/modal-soulx/app.py`). The weights are already on
the volume and need no re-fetch.

**The first Lite call into a container takes ~40s longer than later ones.**
It loads Pro (~27s), then Lite (~12s). A Pro request evicts Lite, so the next
draft after a final pays the ~12s again.

**First render takes 10+ minutes with no output.**
`torch.compile` on a cold container. Subsequent renders at the same resolution
in the same container are ~40% faster per chunk.

**"SoulX weights are missing from the soulx-weights volume".**
The app was deployed before `populate_weights` ran. Run it once (see
[Setup](#setup)) and retry — no redeploy needed, a warm container picks the
weights up on the next request. `uv run tools/verify_setup.py` reports this too.

**CUDA OOM.**
Lower `--size`, or redeploy the app on a larger GPU. Not seen at 768x432 or
544x736 on the default A10G; 1280x720 does OOM there, at the end of the render.
Lite does not affect this, because it is evicted before every Pro render.
The GPU tier is a deploy-time setting:

```bash
SOULX_GPU=L40S uv run modal deploy docker/modal-soulx/app.py
```

Reported on L40S (#95): a 19s take at 1280x720 in 807s, about $0.25. The cost
and speed figures above are for A10G and do not carry over.

**The face is cropped square when a 16:9 image went in.**
`--face-crop` is on. Upstream's crop sets `new_height = new_width`
unconditionally, whatever the target size says, so it discards a 16:9 framing.
Leave it off.

## Known deviations from upstream

- The chunk loop is reimplemented in `docker/modal-soulx/app.py` rather than
  shelling out to `generate_video.py`, so the pipeline stays warm across calls
  in one container. It follows upstream's `stream` mode exactly.
- Resolution is set by mutating `flash_head.inference.infer_params` before
  `get_base_data`. There is no CLI flag upstream and the config is read at
  import time from a *relative* path, so the app also has to `chdir` into the
  repo root before importing.
- `flash_attn`, `sageattention`, `gradio`, `flask`, `decord` and `xformers` are
  not installed; none is reachable on the single-GPU generate path.
- `mediapipe` is floored rather than pinned to upstream's `0.10.9`, whose
  `protobuf<4` cap makes the dependency set unresolvable. Nothing calls it — it
  is imported transitively via the face-crop path.
- Lite runs with `torch.compile` off. Upstream compiles both variants.
- Both variants share one process, so the per-variant values that
  `get_pipeline` writes into the module-global `infer_params`
  (`motion_frames_num`: 5 for Pro, 9 for Lite) are snapshotted at load and
  restored before each render. Upstream runs one variant per process and
  never needs this.

## Still unverified

- **Gesture and upper-body motion.** Tested at head-and-shoulders framing only.
  This model is head-focused by name and design; a wider shot is unexplored.
- **Multi-person.** The pipeline supports several conditioning images with
  `reset_person_name()` to switch speaker mid-stream. Not wired up in the tool.
- **Lite over a long take.** Tested on 2.6s clips only, so it is three chunks
  deep. The 97%-at-70s drift result is for Pro. Lite is a separately distilled
  checkpoint and has not been measured for drift. Keep it to drafts until it is.
- **Lite quality against Pro.** Judged by contact sheet on one subject at
  640x640 and one 16:9 crop at 512x288. Identity held, there were no grid
  artifacts, and lip motion was visible. No side-by-side review by a person yet.
