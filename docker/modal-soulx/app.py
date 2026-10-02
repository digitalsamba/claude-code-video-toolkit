"""
Modal deployment for SoulX-FlashHead talking head generation.

Deploy:
    modal deploy docker/modal-soulx/app.py
    modal run docker/modal-soulx/app.py::populate_weights   # one-off, 14.7 GB

SoulX-FlashHead (Soul AI Lab, Apache 2.0, 1.3B) is the toolkit's default
talking head. It is trained with Oracle-Guided Bidirectional Distillation --
the student generates from its own history against a teacher that sees
ground-truth motion -- which targets the failure mode that limits EchoMimicV3:
each segment re-anchors on the previous segment's output, so one bad segment
poisons everything after it.

Measured on identical input (same photo, same 80s audio, same 544x736) against
the EchoMimicV3 render that collapsed:

    t=30s   SoulX 97%   EchoMimicV3 87%
    t=50s   SoulX 97%   EchoMimicV3 75%
    t=70s   SoulX 97%   EchoMimicV3 50%   <- smear, no eyes

...as a share of frame-zero sharpness. SoulX holds across all 72 segments, so
there is no short-render ceiling to design around. It is also ~3-6x faster in
wall clock and ~3.7x cheaper per second of output. See docs/soulx.md.

Two variants, differing in more than size:
    Pro  -- Wan2.1 VAE, stride (4,8,8), 2x2 patches, 28 new frames per 33-frame chunk
    Lite -- LTX-Video VAE, stride (8,32,32), unpatched, 24 new frames per chunk
Same 1.3B transformer; Lite's speed comes from seeing ~7x fewer tokens per
chunk. Pro is loaded at container start and is the default. Lite is the draft
mode: loaded into the same container on the first request that asks for it,
run without torch.compile (see _load_lite), and evicted again before a Pro
render so Pro keeps all its VRAM (see _evict_lite). Stride x patch decides which
resolutions are legal (see _check_size): 768x432 is fine for Pro and illegal
for Lite.

Input format (POST JSON to the web endpoint):
{
    "image_url" | "image_base64": str,
    "audio_url" | "audio_base64": str,
    "model": str,           # "pro" (default) or "lite" -- the fast draft variant
    "height": int,          # optional, with "width"; else derived from "size"
    "width": int,
    "size": int,            # target long edge, default 768; aspect follows image
    "seed": int,            # default 42
    "use_face_crop": bool,  # default false -- upstream's crop is square-only
    "r2": dict              # optional R2 upload config
}
"""

import os

import modal

REPO_URL = "https://github.com/Soul-AILab/SoulX-FlashHead.git"
# Pinned: this app reaches into flash_head.* internals, so a new upstream main
# could move them under us. Bump deliberately, then re-run a smoke render.
REPO_REF = "9bc03de06bb0de82cd6bc477804512ae06144bf2"

MODEL_REPO = "Soul-AILab/SoulX-FlashHead-1_3B"
MODEL_REV = "59119b6c681230c3eeee157e224ae1941746711e"
WAV2VEC_REPO = "facebook/wav2vec2-base-960h"
WAV2VEC_REV = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"

APP_DIR = "/app/SoulX-FlashHead"
MODELS_DIR = "/models"
OUT_DIR = "/out"

# torch.compile is on upstream (COMPILE_MODEL/COMPILE_VAE in
# flash_head_pipeline.py) and costs 590-780s on the first call of a container,
# recompiling whenever the resolution changes. It then saves ~40% per chunk, so
# it pays for itself across a batch at one resolution and never on a single
# odd-sized one-off. Deploy with SOULX_COMPILE=0 for the latter.
#
# Read at import, and baked into the image below: Modal re-imports this module
# in-container where the deploy-time shell env does NOT exist, so anything
# gated on a bare getenv reads its default there instead (same trap as
# ECHOMIMIC_WEIGHTS in #76).
USE_COMPILE = os.environ.get("SOULX_COMPILE", "1") not in ("0", "false", "False")

app = modal.App("video-toolkit-soulx")

# Weights in a Volume rather than baked, per #76: measured rebuild after a code
# change is ~2.3s against 79-385s for a baked image, and this model is new
# enough that the rebuild cost is the one that bites.
volume = modal.Volume.from_name("soulx-weights", create_if_missing=True)
# Renders land here as well as being returned. A long render is too expensive
# to lose to a dropped connection -- see the spawn() path in tools/soulx.py.
out_volume = modal.Volume.from_name("soulx-out", create_if_missing=True)

_WEIGHT_FETCH = [
    (MODEL_REPO, MODEL_REV, f"{MODELS_DIR}/SoulX-FlashHead-1_3B", None, ["assets/*"]),
    (WAV2VEC_REPO, WAV2VEC_REV, f"{MODELS_DIR}/wav2vec2-base-960h",
     ["*.json", "*.txt", "model.safetensors", "vocab*"], None),
]

image = (
    # 3.11 rather than 3.12: torch 2.7.1 cu128 and mediapipe both ship 3.11
    # wheels, and nothing on the generate path needs newer.
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    # cu128 wheels, as upstream's README specifies.
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # Split into layers, all lower-bounded. One flat unbounded install of this
    # set drives pip to `resolution-too-deep` -- it was offering imageio 1.6
    # (2011) before giving up. Smaller graphs and real floors both matter.
    .pip_install(
        # Pinned as upstream pins it, not floored: EchoMimicV3 taught us that a
        # wav2vec subclass is exactly what a transformers bump breaks silently,
        # with no error and no lip sync (#77). flash_head has one.
        "transformers==4.57.3",
        "tokenizers>=0.20.3",
        "diffusers>=0.34.0",
        "accelerate>=1.8.1",
        "safetensors>=0.4.3",
        "huggingface_hub>=0.25.0",
        "numpy>=1.26.4,<2",
    )
    .pip_install(
        "opencv-python-headless>=4.12.0.88",
        "scikit-image>=0.24.0",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.5.1",
        "librosa>=0.10.2",
        "pyloudnorm>=0.1.1",
        "easydict>=1.13",
        "ftfy>=6.2.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.2",
        "Pillow>=10.3.0",
    )
    .pip_install(
        # Imported unconditionally by flash_head.src.modules.flash_head_model,
        # even single-GPU. Its flash-attn dependency is an extra, and yunchang
        # guards that import behind HAS_FLASH_ATTN, so this stays CPU-buildable.
        "xfuser==0.4.5",
        # Only reachable via facecrop -> cpu_face_handler, which the pipeline
        # imports unconditionally even when use_face_crop is False. Upstream
        # pins 0.10.9, whose protobuf<4 cap is a large part of why the flat
        # resolve failed; floored instead, since nothing here calls it.
        "mediapipe>=0.10.21",
        "boto3",
        "requests",
        "fastapi[standard]",
    )
    # Deliberately NOT installed:
    #   flash-attn / sageattention -- optional, flash_head_model try/excepts
    #     both and falls back to F.scaled_dot_product_attention. Building
    #     flash-attn costs more than it returns here, but it is the reason our
    #     measured speed is a floor rather than the model's ceiling: upstream's
    #     10.8 FPS on a 4090 assumes one of them.
    #   gradio, flask -- demo UIs only.
    #   decord, xformers -- not imported anywhere on the generate path.
    # Fetch-by-SHA rather than `clone --branch`, which only takes a ref name.
    .run_commands(
        f"git init {APP_DIR}",
        f"git -C {APP_DIR} remote add origin {REPO_URL}",
        f"git -C {APP_DIR} fetch --depth 1 origin {REPO_REF}",
        f"git -C {APP_DIR} checkout FETCH_HEAD",
    )
    .env({
        "PYTHONPATH": APP_DIR,
        "TOKENIZERS_PARALLELISM": "false",
        "SOULX_COMPILE": "1" if USE_COMPILE else "0",
    })
)

download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.25.0"
)


@app.function(image=download_image, volumes={MODELS_DIR: volume}, timeout=3600)
def populate_weights():
    """One-off, idempotent: fill the volume before first use.

    modal run docker/modal-soulx/app.py::populate_weights
    """
    import os

    from huggingface_hub import snapshot_download

    for repo, rev, dest, allow, ignore in _WEIGHT_FETCH:
        print(f"fetching {repo}@{rev[:8]} -> {dest}")
        snapshot_download(
            repo, revision=rev, local_dir=dest,
            allow_patterns=allow, ignore_patterns=ignore,
        )

    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(MODELS_DIR) for f in fs
    )
    print(f"weights ready: {total / 1e9:.1f} GB")
    volume.commit()


def _missing_weights():
    """Readable error if the weights volume was never populated, else None."""
    import os

    for _, _, dest, _, _ in _WEIGHT_FETCH:
        if not os.path.isdir(dest) or not os.listdir(dest):
            return (
                f"SoulX weights are missing from the soulx-weights volume ({dest} "
                "is empty). Run this once, then retry: "
                "modal run docker/modal-soulx/app.py::populate_weights"
            )
    return None


# Pixel dimensions must divide by the VAE's spatial stride times the
# transformer's spatial patch size. Nothing upstream validates this:
# target_size flows straight into `lat_h = target_h // vae_stride[1]`, so a bad
# size floors silently and desyncs the latent grid from the pixel grid rather
# than raising. From each variant's config.json at MODEL_REV:
#   Pro:  Wan2.1 VAE stride 8     x patch_size 2x2 = 16
#   Lite: LTX-Video VAE stride 32 x patch_size 1x1 = 32   (not 2x2 -- Lite
#         does not patchify, so its grid is 32 and not the 64 once assumed)
_GRID = {"pro": 16, "lite": 32}
_GRID_WHY = {
    "pro": "Wan2.1 VAE stride 8, 2x2 patches",
    "lite": "LTX-Video VAE stride 32, no patching",
}


def _check_size(height, width, model_type):
    grid = _GRID[model_type]
    for name, v in (("height", height), ("width", width)):
        if v % grid:
            raise ValueError(
                f"{name}={v} must be a multiple of {grid} for model="
                f"{model_type!r} ({_GRID_WHY[model_type]}). "
                f"Nearest legal: {v // grid * grid} or {(v // grid + 1) * grid}."
            )


# get_pipeline writes per-variant values into flash_head.inference's
# module-global infer_params: motion_frames_num follows the VAE's temporal
# stride (5 for Pro, 9 for Lite), which is why Pro emits 28 new frames a chunk
# and Lite 24. With both variants in one process the last load would win and
# the other would render with the wrong chunk overlap -- no error, just wrong
# frames. So each load snapshots these and each render puts its own back.
_PER_MODEL_PARAMS = ("motion_frames_num", "sample_steps")


def _fit_size(img_w, img_h, target, model_type):
    """Preserve the input aspect ratio, snapped to the model's latent grid.

    Matches EchoMimicV3's behaviour: a 16:9 presenter image comes back 16:9,
    with no square-crop workaround. `target` is the long edge.
    """
    grid = _GRID[model_type]
    if img_w >= img_h:
        w, h = target, target * img_h / img_w
    else:
        w, h = target * img_w / img_h, target
    snap = lambda v: max(grid, int(round(v / grid)) * grid)
    return snap(h), snap(w)


@app.cls(
    image=image,
    # A10G (24GB) is the toolkit's existing tier, and upstream quotes its FPS on
    # a 24GB 4090. Measured here: 8.87s per 28-frame chunk at 544x736 with
    # compile on, i.e. ~7.9x realtime, no OOM. 1280x720 does OOM on it (reported
    # in #95, fine on L40S), so the tier is a deploy-time setting:
    #   SOULX_GPU=L40S modal deploy docker/modal-soulx/app.py
    # Only the deploy-time value matters -- the in-container re-import falls
    # back to the default, but the GPU is already allocated by then.
    gpu=os.environ.get("SOULX_GPU", "A10G"),
    volumes={MODELS_DIR: volume, OUT_DIR: out_volume},
    timeout=7200,
    # Generous on purpose. torch.compile is per-container and per-resolution, so
    # a batch of per-scene narrator clips at one size should reuse one warm
    # container and pay that cost once rather than once each.
    scaledown_window=600,
)
@modal.concurrent(max_inputs=1)
class SoulXFlashHead:
    @modal.enter()
    def load_pipeline(self):
        # An unpopulated volume makes get_pipeline die with a bare
        # FileNotFoundError, and a container that dies here is restarted forever
        # while the client sees only a hang until its timeout (#95). Start
        # anyway and let the request say what is wrong.
        self.pipeline = None
        self.lite = None    # loaded on first use -- see _load_lite
        self._infer_overrides = {}
        self.load_error = _missing_weights()
        if self.load_error:
            print(self.load_error)
            return
        self._load()

    def _load(self):
        import os
        import time

        import torch

        # flash_head/inference.py opens "flash_head/configs/infer_params.yaml"
        # by RELATIVE path at import time, so the process must be sitting in the
        # repo root before that import happens. Not optional.
        os.chdir(APP_DIR)

        started = time.time()
        self.model_type = "pro"
        self.use_compile = os.environ.get("SOULX_COMPILE", "1") not in ("0", "false", "False")

        # Set before get_pipeline, which reads these in FlashHeadPipeline.__init__.
        import flash_head.src.pipeline.flash_head_pipeline as fhp
        fhp.COMPILE_MODEL = fhp.COMPILE_VAE = self.use_compile

        from flash_head.inference import get_pipeline

        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.pipeline = get_pipeline(
            world_size=1,
            ckpt_dir=f"{MODELS_DIR}/SoulX-FlashHead-1_3B",
            wav2vec_dir=f"{MODELS_DIR}/wav2vec2-base-960h",
            model_type=self.model_type,
        )
        self._snapshot_infer_params("pro")
        self.load_seconds = time.time() - started
        print(f"pipeline loaded in {self.load_seconds:.1f}s "
              f"(model_type={self.model_type}, compile={self.use_compile})")

    def _snapshot_infer_params(self, model_type):
        import flash_head.inference as fh

        self._infer_overrides[model_type] = {k: fh.infer_params[k] for k in _PER_MODEL_PARAMS}

    def _load_lite(self):
        """Load the Lite variant into this container, on first use.

        Lazy, so a container that only ever serves Pro is exactly what it was
        before Lite existed: same VRAM, same compiled graphs. Measured on A10G:
        Pro loads in ~27s, Lite in ~12s, and Lite adds ~4.5GB resident.

        Eager, not compiled. On Pro, torch.compile costs 10-20 minutes per
        container per resolution and saves 7-40% a chunk. Lite's eager chunks
        are 0.46s at 512x288 and 1.14s at 640x640, so compile would take hours
        of output to pay back -- and a draft is a minute. It also keeps
        Lite out of dynamo's cache: both variants are the same
        WanModelAudioProject class, so a compiled Lite would share a code
        object, and its guard and recompile bookkeeping, with the compiled Pro.
        """
        import time

        import flash_head.src.pipeline.flash_head_pipeline as fhp
        from flash_head.inference import get_pipeline

        started = time.time()
        # Read in FlashHeadPipeline.__init__, so only this construction sees it.
        fhp.COMPILE_MODEL = fhp.COMPILE_VAE = False
        try:
            self.lite = get_pipeline(
                world_size=1,
                ckpt_dir=f"{MODELS_DIR}/SoulX-FlashHead-1_3B",
                wav2vec_dir=f"{MODELS_DIR}/wav2vec2-base-960h",
                model_type="lite",
            )
        finally:
            fhp.COMPILE_MODEL = fhp.COMPILE_VAE = self.use_compile
        self._snapshot_infer_params("lite")
        print(f"lite pipeline loaded in {time.time() - started:.1f}s (compile=False)")

    def _evict_lite(self):
        """Free Lite before a Pro render.

        Pro peaked at 17.5GB at 640x640 with Lite resident -- an estimated
        ~13GB without it -- and 1280x720 already OOMs on the A10G with Pro alone. Kept
        resident, Lite's ~4.5GB would turn sizes in between into OOMs that did
        not happen before Lite existed. Reloading it costs ~12s and no compile,
        so a draft after a final pays that rather than Pro paying in headroom.
        """
        if self.lite is None:
            return
        import gc

        import torch

        self.lite = None
        gc.collect()
        torch.cuda.empty_cache()
        print("lite evicted to give pro its full VRAM")

    def _not_ready(self, model_type="pro"):
        """Make `model_type` ready to render; error string if it cannot, else None.

        Re-checks the volume first, so a container that started before
        populate_weights ran picks the weights up instead of refusing for the
        rest of its warm window. Pro always loads first: it is what the
        container starts with, and Lite is added to it -- then evicted again
        whenever Pro is asked for.
        """
        if self.pipeline is None:
            volume.reload()
            self.load_error = _missing_weights()
            if self.load_error is None:
                self._load()
            if self.load_error:
                return self.load_error
        if model_type == "pro":
            self._evict_lite()
        elif self.lite is None:
            ckpt = f"{MODELS_DIR}/SoulX-FlashHead-1_3B"
            missing = [d for d in ("Model_Lite", "VAE_LTX") if not os.path.isdir(f"{ckpt}/{d}")]
            if missing:
                return (f"Lite weights are missing from the soulx-weights volume "
                        f"({', '.join(missing)}). Re-run: "
                        "modal run docker/modal-soulx/app.py::populate_weights")
            try:
                self._load_lite()
            except Exception as e:
                import traceback

                print(traceback.format_exc())
                return f"Lite failed to load: {type(e).__name__}: {e}"
        return None

    # -- core ---------------------------------------------------------------

    def _render(self, image_path, audio_path, work, height, width, seed,
                use_face_crop, model_type="pro"):
        """Upstream's `stream` chunk loop, inlined.

        Inlined rather than shelled out to generate_video.py so the pipeline
        stays warm across calls in one container -- torch.compile is on, and
        paying it per render would dominate everything else.
        """
        import subprocess
        import time
        from collections import deque

        import imageio
        import librosa
        import numpy as np
        import torch

        import flash_head.inference as fh
        from flash_head.inference import (
            get_audio_embedding, get_base_data, get_infer_params, run_pipeline,
        )

        _check_size(height, width, model_type)
        pipeline = self.lite if model_type == "lite" else self.pipeline

        # There is no resolution argument anywhere upstream -- generate_video.py
        # has no flag and get_base_data reads the module-global dict. Mutating
        # that global is the entire mechanism for non-square output.
        fh.infer_params["height"] = height
        fh.infer_params["width"] = width
        fh.infer_params.update(self._infer_overrides[model_type])

        torch.cuda.reset_peak_memory_stats()
        get_base_data(
            pipeline,
            cond_image_path_or_dir=str(image_path),
            base_seed=seed,
            # Unconditionally square: facecrop.py sets new_height = new_width
            # whatever target_size says. Leave off for 16:9.
            use_face_crop=use_face_crop,
        )
        p = get_infer_params()
        sr, fps = p["sample_rate"], p["tgt_fps"]
        frame_num, motion = p["frame_num"], p["motion_frames_num"]
        slice_len = frame_num - motion

        speech, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        audio_seconds = len(speech) / sr
        slice_samples = slice_len * sr // fps
        remainder = len(speech) % slice_samples
        if remainder:
            speech = np.concatenate(
                [speech, np.zeros(slice_samples - remainder, dtype=speech.dtype)]
            )
        slices = speech.reshape(-1, slice_samples)

        cached = sr * p["cached_audio_duration"]
        end_idx = p["cached_audio_duration"] * fps
        start_idx = end_idx - frame_num
        dq = deque([0.0] * cached, maxlen=cached)

        frames, chunk_times = [], []
        started = time.time()
        for i, chunk in enumerate(slices):
            torch.cuda.synchronize()
            t0 = time.time()

            dq.extend(chunk.tolist())
            emb = get_audio_embedding(pipeline, np.array(dq), start_idx, end_idx)
            video = run_pipeline(pipeline, emb)
            frames.append(video[motion:].cpu())

            torch.cuda.synchronize()
            chunk_times.append(time.time() - t0)
            if i % 10 == 0 or i == len(slices) - 1:
                print(f"chunk {i + 1}/{len(slices)}  {chunk_times[-1]:.2f}s")

        elapsed = time.time() - started

        raw, out = work / "raw.mp4", work / "out.mp4"
        with imageio.get_writer(str(raw), format="mp4", mode="I", fps=fps,
                                codec="h264", ffmpeg_params=["-bf", "0"]) as w:
            for f in frames:
                arr = f.numpy().astype(np.uint8)
                for j in range(arr.shape[0]):
                    w.append_data(arr[j])
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(raw), "-i", str(audio_path),
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(out), "-y"],
            check=True,
        )

        n_frames = sum(f.shape[0] for f in frames)
        stats = {
            "model_type": model_type,
            "compile": self.use_compile and model_type == "pro",
            "width": width,
            "height": height,
            "chunks": len(slices),
            "frames": n_frames,
            "duration_seconds": round(n_frames / fps, 2),
            "audio_seconds": round(audio_seconds, 2),
            "processing_time_seconds": round(elapsed, 1),
            # The number that decides cost. EchoMimicV3 measured 22.8-47.8x.
            "realtime_factor": round(elapsed / (n_frames / fps), 1),
            "first_chunk_seconds": round(chunk_times[0], 1),
            "median_chunk_seconds": round(float(np.median(chunk_times)), 2),
            # Everything resident (both variants, if Lite has been loaded) plus
            # this render's activations: the headroom number behind any OOM.
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        }
        return out, stats

    def _resolve_size(self, image_path, request, model_type):
        from PIL import Image

        height, width = request.get("height"), request.get("width")
        if height and width:
            return int(height), int(width)
        with Image.open(image_path) as im:
            iw, ih = im.size
        return _fit_size(iw, ih, int(request.get("size", 768)), model_type)

    @staticmethod
    def _model_type(value):
        model_type = str(value or "pro").lower()
        if model_type not in _GRID:
            raise ValueError(f"model={value!r}: must be 'pro' or 'lite'")
        return model_type

    # -- entry points -------------------------------------------------------

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        height: int = 0,
        width: int = 0,
        size: int = 768,
        seed: int = 42,
        use_face_crop: bool = False,
        label: str = "render",
        model: str = "pro",
    ) -> dict:
        """Direct call, used by tools/soulx.py's spawn path.

        Persists to the soulx-out volume before returning, so a dropped client
        costs the download rather than the render.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        model_type = self._model_type(model)
        err = self._not_ready(model_type)
        if err:
            raise RuntimeError(err)

        work = Path(tempfile.mkdtemp())
        img_path, aud_path = work / "cond.png", work / "audio.wav"
        img_path.write_bytes(image_bytes)
        aud_path.write_bytes(audio_bytes)

        h, w = self._resolve_size(
            img_path, {"height": height, "width": width, "size": size}, model_type
        )
        out, stats = self._render(img_path, aud_path, work, h, w, seed,
                                  use_face_crop, model_type)
        video_bytes = out.read_bytes()

        os.makedirs(OUT_DIR, exist_ok=True)
        Path(f"{OUT_DIR}/{label}.mp4").write_bytes(video_bytes)
        Path(f"{OUT_DIR}/{label}.json").write_text(json.dumps(stats, indent=2))
        out_volume.commit()

        print("stats:", stats)
        print(f"persisted to volume soulx-out as {label}.mp4")
        return {"success": True, "video": video_bytes, "stats": stats}

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        return {
            "ok": self.pipeline is not None,
            "error": self.load_error,
            "model_type": getattr(self, "model_type", "pro"),
            "lite_loaded": getattr(self, "lite", None) is not None,
            "compile": getattr(self, "use_compile", USE_COMPILE),
            "load_seconds": round(getattr(self, "load_seconds", -1), 1),
        }

    @modal.fastapi_endpoint(method="POST")
    def generate_web(self, request: dict) -> dict:
        import base64
        import subprocess
        import tempfile
        import time
        import uuid
        from pathlib import Path

        import requests as req
        import torch

        start_time = time.time()

        image_url, image_base64 = request.get("image_url"), request.get("image_base64")
        audio_url, audio_base64 = request.get("audio_url"), request.get("audio_base64")
        if not image_url and not image_base64:
            return {"error": "Missing image_url or image_base64"}
        if not audio_url and not audio_base64:
            return {"error": "Missing audio_url or audio_base64"}
        try:
            # "model_type" too, as the stats call it: ignoring it would quietly
            # render Pro for a caller who asked for Lite.
            model_type = self._model_type(request.get("model") or request.get("model_type"))
        except ValueError as e:
            return {"error": str(e)}

        err = self._not_ready(model_type)
        if err:
            return {"error": err}

        work = Path(tempfile.mkdtemp(prefix="modal_soulx_"))
        try:
            def fetch(url, b64, dest):
                if url:
                    resp = req.get(url, stream=True, timeout=300)
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                else:
                    dest.write_bytes(base64.b64decode(b64.split(",", 1)[-1]))
                return dest

            img_path = fetch(image_url, image_base64, work / "input_image.png")
            aud_path = fetch(audio_url, audio_base64, work / "input_audio.wav")

            height, width = self._resolve_size(img_path, request, model_type)
            out, stats = self._render(
                img_path, aud_path, work, height, width,
                int(request.get("seed", 42)),
                bool(request.get("use_face_crop", False)),
                model_type,
            )

            result = {"success": True, **stats}
            result["processing_time_seconds"] = round(time.time() - start_time, 2)

            r2_config = request.get("r2")
            if r2_config:
                import boto3
                from botocore.config import Config

                client = boto3.client(
                    "s3",
                    endpoint_url=r2_config["endpoint_url"],
                    aws_access_key_id=r2_config["access_key_id"],
                    aws_secret_access_key=r2_config["secret_access_key"],
                    config=Config(signature_version="s3v4"),
                )
                object_key = f"soulx/results/{uuid.uuid4().hex[:12]}.mp4"
                client.upload_file(
                    str(out), r2_config["bucket_name"], object_key,
                    ExtraArgs={"ContentType": "video/mp4"},
                )
                result["video_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                    ExpiresIn=7200,
                )
                result["r2_key"] = object_key
            else:
                result["video_base64"] = base64.b64encode(out.read_bytes()).decode("utf-8")
                print("Warning: returning video as base64 (use R2 for large files)")

            return result

        except ValueError as e:
            # _check_size and friends: a caller error, not a server fault.
            return {"error": str(e)}
        except torch.cuda.OutOfMemoryError:
            return {"error": "CUDA OOM. Lower `size` (e.g. 640), or redeploy on a larger "
                             "GPU: SOULX_GPU=L40S modal deploy docker/modal-soulx/app.py"}
        except subprocess.CalledProcessError as e:
            return {"error": f"ffmpeg mux failed: {e.stderr[-300:] if e.stderr else e}"}
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return {"error": f"{type(e).__name__}: {e}"}
        finally:
            import shutil

            shutil.rmtree(work, ignore_errors=True)


@app.local_entrypoint()
def main(
    image: str,
    audio: str,
    out: str = "soulx_out.mp4",
    height: int = 0,
    width: int = 0,
    size: int = 768,
    seed: int = 42,
    label: str = "render",
    model: str = "pro",
):
    import json
    from pathlib import Path

    result = SoulXFlashHead().generate.remote(
        image_bytes=Path(image).read_bytes(),
        audio_bytes=Path(audio).read_bytes(),
        height=height, width=width, size=size, seed=seed, label=label,
        model=model,
    )
    Path(out).write_bytes(result["video"])
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {out}")
