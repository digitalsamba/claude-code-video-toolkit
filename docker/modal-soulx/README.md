# modal-soulx

Modal app behind `tools/soulx.py`. SoulX-FlashHead (Soul AI Lab, Apache 2.0,
1.3B) is the toolkit's default talking head generator.

User-facing documentation is `docs/soulx.md`. This file covers the deployment
and the upstream quirks the app works around.

## Deploy

```bash
uv run modal run docker/modal-soulx/app.py::populate_weights   # one-off, 14.7 GB
uv run modal deploy docker/modal-soulx/app.py
# Add the printed generate_web URL to .env as MODAL_SOULX_ENDPOINT_URL
```

Endpoints: `generate_web` (POST) and `health` (GET). `SOULX_COMPILE=0` at
deploy time disables `torch.compile` for one-off renders at unusual sizes.

One class serves both variants. **Pro** loads at container start, exactly as
before. **Lite** (`"model": "lite"` in the request) is loaded into the same
container on the first request that asks for it (~12s), and always runs
eager — compile costs Pro 10-20 minutes per resolution, and Lite's 0.5-1.1s
chunks would never pay that back. Lite is **evicted before every Pro render**,
so Pro keeps the VRAM headroom it had before Lite existed (Pro peaked at
17.5GB at 640x640 with Lite resident). A container that only serves Pro never
loads it.

## Why this model

It replaced EchoMimicV3, which was never released. Identity drift made that one
unusable for real narration, and the failure is *absorbing* rather than gradual
— each segment re-anchors on the previous segment's output, so one bad segment
poisons everything after it and no amount of tuning recovers it.

SoulX is trained with Oracle-Guided Bidirectional Distillation, which targets
exactly that. Verified rather than taken on trust — same photo, same 80s audio,
same 544x736 output, as a share of frame-zero sharpness:

| t | SoulX | EchoMimicV3 |
|---|---|---|
| 30s | 97% | 87% |
| 50s | 97% | 75% |
| 70s | **97%** | **50%** |

Repeated on a second subject at 640x640 — different colouring, no glasses, busy
workshop background — which held **97-100% across the full 80s**, including the
fine background detail that drifts first. So it is a property of the model, not
of one photograph.

Speed on A10G, compile on, no flash-attn: 7.14s per 28-frame chunk at 768x432,
8.87s at 544x736, 8.93s at 640x640 — ~6.4-7.9x realtime, roughly $0.0024 per
second of output. **A floor, not a ceiling:** upstream's quoted 10.8 FPS on a
4090 assumes flash-attn or sageattention, and neither is installed.

## What the code works around

- **There is no resolution argument anywhere upstream.** `generate_video.py` has
  no flag; `flash_head/inference.py` reads `height`/`width` from a module-global
  dict loaded at import from `flash_head/configs/infer_params.yaml`, by relative
  path. So the app `chdir`s into the repo root before importing, then mutates
  that global. That is the entire mechanism for non-square output.
- **Legal resolutions differ by variant, and nothing validates them.**
  `target_size` flows into `lat_h = target_h // vae_stride[1]`, so a
  non-divisible size floors silently and desyncs the latent grid from the pixel
  grid — a wrong render rather than an error. Pro uses the **Wan2.1** VAE
  (stride 8) with 2x2 patches: grid 16. Lite uses the **LTX-Video** VAE
  (stride 32) and does not patchify (`Model_Lite/config.json`:
  `patch_size [1,1,1]`): grid **32**, not the 64 this file once said. 768x432
  is legal for Pro and illegal for Lite. `_check_size` raises instead.
- **Both variants write the same module-global.** `get_pipeline` stores
  `motion_frames_num` (5 for Pro, 9 for Lite — it follows the VAE's temporal
  stride) into `flash_head.inference.infer_params`, so with both loaded the
  last load would win and the other variant would render with the wrong chunk
  overlap. Each load snapshots its values and each render restores them.
- **`--use_face_crop` is square-only** whatever the config says:
  `utils/facecrop.py` sets `new_height = new_width` unconditionally. Off for
  16:9 work.
- **`torch.compile` is on upstream** (`COMPILE_MODEL`/`COMPILE_VAE`) and costs
  ~600s on a container's first call, recompiling on every resolution change.
  `scaledown_window` is set to 600s so a batch at one resolution reuses the
  warm container and pays it once.
- **`flash_attn` and `sageattention` are optional** — `flash_head_model`
  try/excepts both and falls back to `F.scaled_dot_product_attention`. Neither
  is installed, which keeps the image CPU-buildable.
- **`xfuser` is imported unconditionally** even single-GPU, so it must be
  installed. Its `flash-attn` dependency is an extra and `yunchang` guards that
  import behind `HAS_FLASH_ATTN`, so it costs nothing.
- **`mediapipe` is imported unconditionally too**, via
  `pipeline -> facecrop -> cpu_face_handler`, even with face crop off. Upstream
  pins `0.10.9`, whose `protobuf<4` cap helped push pip into
  `resolution-too-deep`; floored instead, and the pip install is split into
  layers with real lower bounds throughout.
- **Weights live in a Modal Volume** (#76), not baked: redeploy after a code
  change is ~2.3s. The repo ref and both model revisions are pinned by SHA,
  since volume weights are not tied to the image.

## Long renders and dropped clients

`modal run` keeps a client attached and **cancels the call when that client
dies** — including when a laptop sleeps — and `--detach` keeps the app alive
but does not save the in-flight call. Two 80s renders were lost that way, one
six minutes into compile.

The normal path (`tools/soulx.py` over HTTP) is unaffected. For direct
invocations, use `.spawn()` against the *deployed* app rather than `modal run`.
Renders are also written to the `soulx-out` volume before being returned, so a
lost connection costs the download and not the GPU time.
