# EchoMimicV3 - Talking Head Video Generation

Audio-driven talking head generation with [EchoMimicV3-Flash](https://github.com/antgroup/echomimic_v3)
(Ant Group, Apache 2.0, AAAI 2026). Runs on Modal.

The toolkit has two talking-head generators. The short version: **EchoMimicV3 for anything
the viewer looks at directly, SadTalker for small overlays and quick drafts.** See
[Choosing between EchoMimicV3 and SadTalker](#choosing-between-echomimicv3-and-sadtalker).

## Quick Start

```bash
# Basic usage
uv run tools/echomimic3.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

# NarratorPiP settings - 16:9 in, 16:9 out, cheap 5-step pass
uv run tools/echomimic3.py \
    --image presenter_16x9.png --audio scene_01.mp3 \
    --steps 5 --size 640 --output narrator.mp4

# Side by side against an existing SadTalker render of the same inputs
uv run tools/echomimic3.py \
    --image presenter_16x9.png --audio scene_01.mp3 \
    --output narrator_echo.mp4 --compare narrator_sadtalker.mp4
```

## Choosing between EchoMimicV3 and SadTalker

| | SadTalker | EchoMimicV3-Flash |
|---|---|---|
| Released | 2023, unmaintained | Flash variant Jan 2026 |
| Licence | Apache 2.0 | Apache 2.0 |
| Params | ~0.3B (warp-based) | 1.3B (Wan2.1-Fun diffusion) |
| Aspect ratio | square crop unless `--preprocess full` | follows the input image |
| Motion | head + light expression | head, upper body, gestures |
| Cost | ~$0.0014 per second of output | ~$0.009 per second (**~6.5x**) |
| Speed | faster than realtime | 22.8-47.8x realtime |
| Cloud | RunPod or Modal | Modal only |

**Use EchoMimicV3 when** the narrator is large in frame, the shot is held long enough to
watch, or the source image is not square and you do not want to fight the crop.

**Use SadTalker when** the output is a small overlay, you need a draft in a minute rather
than an hour, or you are generating many takes to choose between.

This was checked rather than assumed. A controlled A/B — same still, same audio, both downscaled
to NarratorPiP `sm` (240x135) — found EchoMimicV3 carries 3.5x the whole-frame motion and 6.2x in
the mouth band, so the gap is not washed out by the shrink. On review the verdict was that
EchoMimicV3 is clearly better while SadTalker is still perfectly usable at that size. Hence two
tools rather than a replacement.

The cost gap is real but bounded: a 3-minute narrator is roughly $1.76 against $0.27.
The **wall clock** is the sharper constraint — that same 3 minutes is 1.5-2.4 hours of
generation, so per-scene narrator clips generated in advance beat a single long render.

## Setup

EchoMimicV3 is Modal-only.

```bash
uv sync --extra modal && uv run modal setup

# One-off: fill the weights volume (~26GB, ~10 min). Needed before first deploy.
uv run modal run docker/modal-echomimic3/app.py::populate_weights

uv run modal deploy docker/modal-echomimic3/app.py
```

Then add the printed URL to `.env`:

```
MODAL_ECHOMIMIC3_ENDPOINT_URL=https://....modal.run
```

Unlike the other Modal apps, this one keeps its weights in a **Modal Volume** rather than
baking them into the image — which is why `populate_weights` exists and why it must run
first. See [Weight storage](#weight-storage) for why.

## Parameters

### Core settings

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | 8 | 5 is materially cheaper and holds up well; 8+ for hero shots |
| `--size` | 768 | Generation resolution; 640 pairs well with `--steps 5` |
| `--fps` | 25 | |
| `--seed` | 43 | |
| `--prompt` | "A person is speaking to the camera." | Effect is weak — see below |
| `--wav2vec` | chinese | **Leave it.** See [Settled findings](#settled-findings) |

### Tuning

| Flag | Default | Notes |
|------|---------|-------|
| `--video-length` | 81 | Frames per segment. Raising it cuts the segment count — the single biggest lever on cost — until VRAM complains |
| `--overlap` | 8 | Frames cross-faded between segments |
| `--anchor-retreat` | 6 | Max frames to back off when a seam lands on a blink; 0 restores the old behaviour |
| `--guidance-scale` | 6.0 | Text CFG, 3-6 |
| `--audio-guidance-scale` | 3.0 | Audio CFG. Upstream suggests 1.8-2.0 for lip sync, but see below |
| `--audio-scale` | 1.0 | Audio conditioning strength |

## Image guidelines

Same as SadTalker with one difference that matters: **EchoMimicV3 follows the input
aspect ratio**, so a 16:9 presenter image comes back 16:9. There is no `--preprocess`
equivalent and none is needed.

Good source images are front-facing, evenly lit, with the face 30-70% of the frame.
Avoid heavy backlighting, extreme angles, and faces that fill the entire frame.

## Performance and cost

Measured on Modal A10G (24GB, 22.06 usable):

| Config | Realtime factor | Cost per second of output |
|--------|-----------------|---------------------------|
| `--steps 5 --size 640` | 22.8-28.9x | ~$0.009 |
| `--steps 8 --size 768` | 47.8x | ~$0.015 |

The dominant cost is the **number of segments**, not the per-step cost. At the defaults
each segment advances only `81 - 8 = 73` frames (2.9s), so a 30s narrator is 11 full
diffusion passes and a 3-minute one is 62. Raise `--video-length` before reaching for
fewer steps.

The model needs `enable_model_cpu_offload()` to fit 24GB, which trades PCIe paging for
resident VRAM on every pipeline call.

## How long audio is handled

Upstream's `infer_flash.py` generates one 81-frame (3.2s) clip and **silently truncates
longer audio** — the Flash pipeline does not accept the long-video kwargs the preview
pipeline does. The segment loop therefore lives in `docker/modal-echomimic3/app.py`:
each segment re-anchors on the last `--overlap` frames of the previous one, and the seam
is cross-faded.

### Blinks at segment seams

Because each segment starts from the previous segment's final frames, whatever pose those
frames hold becomes the next segment's opening pose. When they landed mid-blink the model
started closed-eyed and **held it** — a prolonged closure straddling the boundary.

`--anchor-retreat` fixes this: before re-anchoring, the loop scores candidate anchor
windows by how much motion they contain in the **upper half** of the frame (where blinks
live and mouth movement does not) and backs off up to N frames to anchor on the calmest
one. It only pays that cost when there is a clear improvement to be had, so clips with no
transient near a seam regenerate nothing. `--anchor-retreat 0` restores the old behaviour.

## Settled findings

Things established by measurement, so they do not get re-litigated:

- **`--wav2vec english` is worse than `chinese`, even for English audio.** It
  under-articulates throughout (motion 2.26 vs 3.5) and visibly sits half-open. The Flash
  model was trained with the Chinese encoder; `run_flash.sh` uses it regardless of
  language. Keep the default.
- **Do not trust a mouth-crop sync metric.** It scores a mouth crop only and is
  structurally blind to eye, hair and background artifacts. In the tuning matrix it ranked
  *highest* the one variant with a visible eye defect. Any automated scoring needs
  whole-face coverage, or use human review.
- **The dependency pins are load-bearing.** `diffusers==0.32.2` and
  `transformers==4.49.0`. Newer transformers drives `output_hidden_states` from config
  rather than the kwarg EchoMimic's `Wav2Vec2Model` subclass passes, so audio embeddings
  come back empty and **all lip sync silently disappears**. Setting
  `config.output_hidden_states` does not fix it.
- **The prompt is close to inert.** A rich descriptive prompt was marginally preferred by
  eye over `"A person is speaking."`, but not decisively. Do not spend effort here.
- **`--audio-guidance-scale` is unsettled.** Upstream recommends 1.8-2.0; the default here
  is 3.0 to match `run_flash.sh`. An earlier recommendation to lower it was retracted
  because it rested on the mouth-crop metric above.

## Weight storage

This app keeps its ~26GB of weights in a Modal Volume. Every other `docker/modal-*` app
bakes them into the image. That split is deliberate and measured:

| | Baked image | Volume |
|---|---|---|
| Rebuild after a dependency change | 79-385s | 1.8-8.2s |
| Cold start | 57-94s | 60-67s |
| Generation speed | identical | identical |
| Storage cost | £0 | £0 (26GB against a 1TiB/month allowance) |

Cold start and generation are a wash; the rebuild difference is 20-100x. Getting this
model working took four separate dependency changes, each of which re-downloaded 26GB
under the baked scheme. The settled apps (`upscale`, `image-edit`) change rarely and gain
nothing by moving, so they stay baked.

**The volume is optional, and it is free at this scale.** Modal charges $0.09/GiB/month
for volume storage with **1 TiB/month included free**, so the 26.6GB this app stores is
about 2.6% of the free allowance — nothing is being spent to hold it, and nothing is saved
by not holding it. That is what makes the choice a pure engineering one rather than a cost
trade-off: the only real question is whether faster rebuilds are worth an extra step, and
for an app that needed four dependency changes to get working, they are.

If you would rather have one self-contained artifact — no `populate_weights` step, no
ordering requirement, weights pinned to the image — the baked variant is fully supported
and deploys as a separate app. Neither path is deprecated.

The one genuine cost of the volume is reproducibility: weights are no longer pinned to the
image, so nothing but explicit revisions stops image and weights drifting apart. That is
why the upstream repo ref and all four model revisions are pinned by SHA in `app.py`. Bump
them deliberately and re-run `populate_weights`.

To deploy the baked variant instead (as a separate app, `video-toolkit-echomimic3-baked`):

```bash
ECHOMIMIC_WEIGHTS=image uv run modal deploy docker/modal-echomimic3/app.py
```

## Troubleshooting

**Lip sync is completely absent, no error.** The `transformers` pin has drifted. It must
be exactly `4.49.0` — see [Settled findings](#settled-findings).

**OOM during `@modal.enter()`.** `enable_model_cpu_offload()` is not being applied.
Upstream's `infer_flash.py` parses a `GPU_memory_mode` flag but never applies it, so
following upstream literally does not work on a 24GB card.

**A `pytorch_model.bin` refuses to load.** `chinese-wav2vec2-base` ships one, and
transformers >= 4.51.3 refuses `torch.load` below torch 2.6 (CVE-2025-32434) as a hard
failure. The image converts it to safetensors at build time rather than bumping torch,
which would flip `torch.load`'s `weights_only` default and break EchoMimic's own `.pth`
loads for the VAE, text encoder and CLIP.

**A prolonged eye closure across a segment boundary.** See
[Blinks at segment seams](#blinks-at-segment-seams) — raise `--anchor-retreat`.

## Known deviations from upstream

- `_build_inputs` reimplements `src.utils.get_image_to_video_latent2`, which calls
  `.resize()` on its argument before checking whether it is a list and so raises
  `AttributeError` on the multi-frame re-anchoring the segment loop needs.
- The image omits `tensorflow`, `retina-face`, `gradio`, `decord` and `moviepy` from
  upstream's `requirements.txt`. None are reachable from the Flash path — tensorflow and
  retina-face are only used by `src.face_detect` for the preview variant's `ip_mask`. Add
  them back if the preview variant is ever wired up.

## Still unverified

- Behaviour at full narration length — drift across 60+ segments has not been watched.
- Whether the prompt matters at all.
