# SoulX-FlashHead — Talking Head Video Generation

Generates a talking head video from one portrait image plus an audio track.
[SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul AI Lab,
Apache 2.0, 1.3B params, arXiv 2602.07449) is the toolkit's default talking
head generator.

It preserves the input image's aspect ratio, so a 16:9 presenter image comes
back 16:9 with no `--preprocess` workaround, and it holds identity across long
renders — which is the reason it is the default.

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

# Side by side against an existing render of the same inputs
uv run tools/soulx.py -i p.png -a vo.mp3 -o new.mp4 --compare old.mp4
```

## Choosing between SoulX-FlashHead and SadTalker

| | SadTalker | SoulX-FlashHead |
|---|---|---|
| Method | warp-based | diffusion, distilled to 4 steps |
| Aspect ratio | square unless `--preprocess full` | follows the input |
| Identity over a long take | stable (it barely moves) | **stable — 97% at 70s** |
| Motion | head + light expression | head, shoulders, natural expression |
| Cost per second of output | ~$0.0014 | ~$0.0024 |
| Speed | near realtime | ~7.9x realtime (plus a one-off compile) |

**Use SoulX-FlashHead** for anything a viewer actually watches: a narrator
large in frame, a held shot, a finished video.

**Use SadTalker** for throwaway drafts and for generating many takes to choose
between, where its speed matters more than its quality. The cost gap is now
only ~1.7x, so cost alone is rarely the deciding factor.

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
| `--size` | 768 | Target long edge; aspect follows the image |
| `--width` / `--height` | — | Exact dimensions instead of `--size`. Both or neither |
| `--seed` | 42 | |
| `--face-crop` | off | Upstream's face detect + crop |
| `--compare` | — | Also write a labelled side-by-side against an existing render |

### Resolution rules

Width and height must both be **multiples of 16**. The Wan2.1 VAE has spatial
stride 8 and the transformer patchifies 2×2, so 16 is the real grid.

This is checked by the tool and again by the endpoint, because **nothing
upstream validates it**. `target_size` flows straight into
`lat_h = target_h // vae_stride[1]`, so an off-grid size floors silently and
desyncs the latent grid from the pixel grid — a wrong render, not an error.

`--size` handles this for you: it snaps to the grid while preserving aspect.

Sizes that are legal and useful: `768x432` and `640x368` (16:9), `544x736`
(3:4 portrait), `640x640` / `512x512` (square).

> Only relevant if you deploy the Lite variant: Lite uses the **LTX-Video** VAE
> with stride 32, so its grid is 64 and `768x432` is *illegal* there. The app
> deploys Pro.

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

**"must be a multiple of 16".**
See [Resolution rules](#resolution-rules). Use `--size` and let it snap.

**First render takes 10+ minutes with no output.**
`torch.compile` on a cold container. Subsequent renders at the same resolution
in the same container are ~40% faster per chunk.

**CUDA OOM.**
Lower `--size`, or redeploy the app on L40S. Not seen at 768x432 or 544x736 on
A10G.

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

## Still unverified

- **Gesture and upper-body motion.** Tested at head-and-shoulders framing only.
  This model is head-focused by name and design; a wider shot is unexplored.
- **Multi-person.** The pipeline supports several conditioning images with
  `reset_person_name()` to switch speaker mid-stream. Not wired up in the tool.
- **The Lite variant.** Deployed weights include it, but nothing here runs it.
  Its different VAE means different legal resolutions.
