"""
Modal deployment for LTX-2.5 video generation.

Text-to-video and image-to-video generation using LTX-2.5 22B DiT model.
Generates ~5s video clips at up to 1024x1536 resolution with audio.

Deploy:
    modal deploy docker/modal-ltx2/app.py

Input format (POST JSON to web endpoint):
{
    "prompt": str,                     # Required: text description
    "negative_prompt": str,            # Optional (sensible default provided)
    "image_url": str,                  # Optional: URL for image-to-video
    "image_base64": str,              # Optional: base64 for image-to-video
    "width": int,                      # Default: 768 (must be divisible by 64)
    "height": int,                     # Default: 512 (must be divisible by 64)
    "num_frames": int,                 # Default: 121 (must satisfy (n-1)%8==0)
    "fps": int,                        # Default: 24
    "num_inference_steps": int,        # Default: 30
    "seed": int,                       # Optional: random if not set
    "quality": "standard" | "fast",    # Default: "standard"
    "lora": str,                       # Optional: style LoRA key (e.g. "crt-terminal")
    "r2": dict                         # Optional: R2 upload config
}

Output format:
{
    "success": true,
    "seed": int,
    "duration": float,
    "width": int,
    "height": int,
    "num_frames": int,
    "fps": int,
    "inference_time_ms": int,
    "video_base64": str,               # If no R2
    "output_url": str,                 # If R2 configured
    "r2_key": str                      # If R2 configured
}
"""

import modal

app = modal.App("video-toolkit-ltx2")

LTX2_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
# Pinned to the v1.4.2 release (2026-10-02). LTX-2.5 support landed in v1.2.0,
# along with Gemma 4, split checkpoints and transformers 5.x. Upstream tests on
# torch 2.13 (its natten extra pins it), which is what the image installs
# below. An unpinned clone broke this app once already (#94). Bump
# deliberately, together with torch, then re-run a smoke render.
LTX2_REPO_REF = "9ec55f9f22798a3198d9c923856824821bc3317e"

# HuggingFace model repo. 2.5 ships as a split pack, one file per component,
# and bundles its own Gemma 4 text encoder (no separate Gemma repo).
HF_REPO = "Lightricks/LTX-2.5"
# Local path inside the container where weights are stored. Files keep the
# repo's folder layout under it.
MODEL_DIR = "/models/ltx2"
# The dev transformer + distilled LoRA keeps the guided two-stage pipeline
# (CFG/STG, negative prompt, adjustable steps) that "quality" and
# "num_inference_steps" control. The distilled transformer would be faster but
# runs a fixed 8+3 step schedule with no guidance, so those options would do
# nothing.
TRANSFORMER_FILE = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
DISTILLED_LORA_FILE = "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
SPATIAL_UPSAMPLER_FILE = (
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)
# Diffusion video decoder (DiffVAE), not the lighter -conv- variant: it's the
# 2.5 decoder upgrade (sharper faces/textures), and NATTEN makes it fast.
VIDEO_VAE_FILE = "vae/ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE_FILE = "vae/ltx-2.5-audio-vae-bf16.safetensors"
TEXT_ENCODER_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
# Style LoRAs baked into the image. Map of CLI key → {repo, filename, strength}.
# Add new LoRAs here to ship them in the container. Lightricks reports most
# 2.3 LoRAs run on 2.5 unchanged, with a few exceptions; crt-terminal was
# trained on 2.3 and hasn't been validated on 2.5 yet.
LORA_DIR = "/models/loras"
AVAILABLE_LORAS = {
    "crt-terminal": {
        "repo": "lovis93/crt-animation-terminal-ltx-2.3-lora",
        "filename": "crtanim_10000.safetensors",
        "strength": 1.0,
    },
}

# Build the container image with all dependencies + baked weights
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    # PyTorch 2.13 with CUDA 13.0 (separate index). Upstream uses cu132; cu130
    # matches the driver on Modal hosts (580 / CUDA 13.0). torchaudio stopped
    # at 2.11, which is what upstream pairs with torch 2.13 too.
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    # Other dependencies (from PyPI). ltx-core needs transformers >=5.8,<5.15
    # for Gemma 4; 5.14.1 is the newest it documents as verified.
    .pip_install(
        "einops",
        "numpy>=1.26",
        "transformers==5.14.1",
        "safetensors",
        "accelerate",
        "scipy>=1.14",
        "av",
        "tqdm",
        "Pillow",
        "colour-science",
        "openimageio",
        "cloudpickle>=3.1",
        "boto3",
        "requests",
        "fastapi[standard]",
        "huggingface_hub[hf_xet]",
    )
    # NATTEN accelerates the diffusion video decoder (falls back to Triton
    # without it). Same version upstream's natten extra pins, cu130 build.
    .pip_install(
        "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.7/"
        "natten-0.21.7%2Btorch2130cu130-cp312-cp312-linux_x86_64.whl"
    )
    # Fetch LTX-2 at the pinned ref and install its packages. Fetch-by-SHA
    # rather than `clone --branch`, which only takes a ref name.
    .run_commands(
        "git init /app/ltx2",
        f"git -C /app/ltx2 remote add origin {LTX2_REPO_URL}",
        f"git -C /app/ltx2 fetch --depth 1 origin {LTX2_REPO_REF}",
        "git -C /app/ltx2 checkout FETCH_HEAD",
        "pip install -e /app/ltx2/packages/ltx-core",
        "pip install -e /app/ltx2/packages/ltx-pipelines",
    )
    # torch 2.13 pins cuDNN 9.20, which lacks libcudnn_engines_tensor_ir.
    # Upstream overrides it to 9.24.0.43 (see LTX-2's root pyproject.toml).
    # Last pip step so nothing downgrades it again.
    .run_commands("pip install nvidia-cudnn-cu13==9.24.0.43")
    # Bake LTX-2.5 weights — gated repo, needs HF_TOKEN at build time.
    # One layer per large component so a failed download doesn't redo the rest.
    # Dev transformer (42GB)
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF_REPO}', local_dir='{MODEL_DIR}', "
        f"allow_patterns=['{TRANSFORMER_FILE}'])"
        "\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    # Gemma 4 12B text encoder + projections (26GB). Tokenizer and config are
    # embedded in the safetensors file.
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF_REPO}', local_dir='{MODEL_DIR}', "
        f"allow_patterns=['{TEXT_ENCODER_FILE}'])"
        "\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    # Distilled LoRA (8.9GB) + spatial upsampler (1GB) + video/audio VAEs (1.8GB)
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF_REPO}', local_dir='{MODEL_DIR}', "
        f"allow_patterns=['{DISTILLED_LORA_FILE}', '{SPATIAL_UPSAMPLER_FILE}', "
        f"'{VIDEO_VAE_FILE}', '{AUDIO_VAE_FILE}'])"
        "\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    # Bake style LoRAs (~500MB each). One hf_hub_download per entry so each
    # LoRA lives at /models/loras/<key>/<filename> and swaps are cheap.
    .run_commands(
        *[
            "python -c \""
            "from huggingface_hub import hf_hub_download; "
            f"hf_hub_download(repo_id='{meta['repo']}', "
            f"filename='{meta['filename']}', "
            f"local_dir='{LORA_DIR}/{key}')"
            "\""
            for key, meta in AVAILABLE_LORAS.items()
        ],
    )
)


@app.cls(
    image=image.env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}),
    gpu="A100-80GB",
    timeout=900,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("huggingface-token")],
)
@modal.concurrent(max_inputs=1)
class LTX2:
    """LTX-2.5 video generation."""

    @modal.enter()
    def load_pipeline(self):
        """Load the base LTX-2 pipeline when the container starts."""
        import os

        import torch

        from ltx_pipelines.utils.constants import detect_params
        from ltx_pipelines.utils.model_paths import ModelPaths

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}, VRAM: {props.total_memory // (1024**3)}GB")

        paths = {
            name: os.path.join(MODEL_DIR, rel)
            for name, rel in {
                "transformer": TRANSFORMER_FILE,
                "distilled_lora": DISTILLED_LORA_FILE,
                "spatial_upsampler": SPATIAL_UPSAMPLER_FILE,
                "video_vae": VIDEO_VAE_FILE,
                "audio_vae": AUDIO_VAE_FILE,
                "text_encoder": TEXT_ENCODER_FILE,
            }.items()
        }
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"LTX-2.5 weights missing from {MODEL_DIR}: {missing}")

        for name, path in paths.items():
            print(f"  {name}: {path}")

        # No duration head: requests always pass num_frames explicitly.
        self._model_paths = ModelPaths.from_split(
            transformer_path=paths["transformer"],
            text_encoder_path=paths["text_encoder"],
            video_vae_path=paths["video_vae"],
            audio_vae_path=paths["audio_vae"],
        )
        self._distilled_lora_path = paths["distilled_lora"]
        self._spatial_upsampler = paths["spatial_upsampler"]
        # Guidance defaults for this checkpoint's generation (CFG/STG scales,
        # STG block), read from its metadata the same way the upstream CLI does.
        self._params = detect_params(paths["transformer"])

        self.pipeline = None
        self._current_style_lora = None
        self._build_pipeline(style_lora=None)

    def _build_pipeline(self, style_lora):
        """Construct the pipeline with an optional style LoRA key.

        Called at cold start (no style LoRA) and on per-request style swaps.
        Construction is cheap: weights load per stage during each request and
        are freed after it, so a rebuild just swaps the LoRA set.
        """
        import gc
        import os
        import time

        import torch

        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        if self.pipeline is not None:
            print(f"Releasing current pipeline (style_lora={self._current_style_lora})")
            del self.pipeline
            self.pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        distilled = [
            LoraPathStrengthAndSDOps(
                self._distilled_lora_path, 0.8, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ]

        style_loras = []
        if style_lora:
            meta = AVAILABLE_LORAS[style_lora]
            lora_path = os.path.join(LORA_DIR, style_lora, meta["filename"])
            if not os.path.exists(lora_path):
                raise RuntimeError(f"Style LoRA file missing: {lora_path}")
            style_loras = [
                LoraPathStrengthAndSDOps(
                    lora_path, meta["strength"], LTXV_LORA_COMFY_RENAMING_MAP
                )
            ]
            print(f"  Style LoRA: {style_lora} @ strength {meta['strength']}")

        print(f"Building pipeline (style_lora={style_lora})...")
        start = time.time()
        # bf16 weights, no offload: on A100-80GB the stages load one at a time
        # (Gemma 4 ~26GB, then the 42GB transformer per stage), so peak VRAM
        # is one transformer plus activations, not the sum of the components.
        self.pipeline = TI2VidTwoStagesPipeline(
            model_paths=self._model_paths,
            distilled_lora=distilled,
            spatial_upsampler_path=self._spatial_upsampler,
            loras=style_loras,
        )
        self._current_style_lora = style_lora
        print(f"Pipeline built in {time.time() - start:.1f}s")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """Generate video from text prompt (and optional image)."""
        import base64
        import io
        import random
        import shutil
        import tempfile
        import time
        import uuid

        import torch
        from PIL import Image

        prompt = request.get("prompt")
        if not prompt:
            return {"error": "Missing required 'prompt' field"}

        negative_prompt = request.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, text, logo",
        )

        # Video parameters
        width = request.get("width", 768)
        height = request.get("height", 512)
        num_frames = request.get("num_frames", 121)
        fps = request.get("fps", 24)
        num_inference_steps = request.get("num_inference_steps", 30)
        seed = request.get("seed")
        quality = request.get("quality", "standard")
        r2_config = request.get("r2")

        # Optional style LoRA. Rebuild the pipeline only when the requested
        # LoRA differs from what's currently loaded.
        style_lora = request.get("lora")
        if style_lora and style_lora not in AVAILABLE_LORAS:
            return {
                "error": f"Unknown LoRA '{style_lora}'. "
                f"Available: {list(AVAILABLE_LORAS) or '(none)'}"
            }
        if style_lora != self._current_style_lora:
            self._build_pipeline(style_lora=style_lora)

        # Enforce dimension constraints. The two-stage pipeline generates at
        # half size first, so both sides must be multiples of 64.
        width = (width // 64) * 64
        height = (height // 64) * 64
        if width < 64 or height < 64:
            return {"error": "width and height must each be at least 64"}

        # Enforce frame count constraint: (num_frames - 1) % 8 == 0
        if (num_frames - 1) % 8 != 0:
            # Round to nearest valid frame count
            num_frames = ((num_frames - 1 + 4) // 8) * 8 + 1

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Fast mode: fewer steps
        if quality == "fast":
            num_inference_steps = min(num_inference_steps, 15)

        # Decode input image for I2V — pipeline expects file paths on disk
        images = []
        image_base64 = request.get("image_base64")
        image_url = request.get("image_url")

        work_dir = tempfile.mkdtemp(prefix="ltx2_")

        if image_base64 or image_url:
            try:
                if image_url:
                    import requests as req

                    resp = req.get(image_url, timeout=60)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                else:
                    b64 = image_base64
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

                # Save to disk — pipeline loads images from file paths
                import os

                img_path = os.path.join(work_dir, "input.png")
                img.save(img_path)

                from ltx_pipelines.utils.types import ImageConditioningInput

                # crf left unset: the pipeline re-compresses the image at the
                # CRF this checkpoint was trained with (from its metadata).
                images = [
                    ImageConditioningInput(
                        path=img_path,
                        frame_idx=0,
                        strength=0.8,
                    )
                ]
            except Exception as e:
                return {"error": f"Failed to decode input image: {e}"}
        start_time = time.time()

        try:
            print(
                f"Generating: {width}x{height}, {num_frames} frames, "
                f"{num_inference_steps} steps, seed={seed}"
            )

            # CRITICAL: torch.inference_mode() prevents PyTorch from retaining
            # the autograd graph. Without it, the text encoder's activations
            # stay in VRAM even after del, causing OOM when the transformer
            # loads (GitHub #152). The pipeline's __call__ doesn't set
            # inference_mode; the upstream CLI wraps main() in it.
            # The scope must include encode_video() because the pipeline returns
            # a lazy iterator — frames are decoded when the iterator is consumed.
            import os

            output_path = os.path.join(work_dir, "output.mp4")

            from ltx_core.model.video_vae import get_video_chunks_number
            from ltx_pipelines.utils.media_io import encode_video

            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    num_inference_steps=num_inference_steps,
                    video_guider_params=self._params.video_guider_params,
                    audio_guider_params=self._params.audio_guider_params,
                    images=images,
                )

                encode_video(
                    video=result.video,
                    fps=fps,
                    audio=result.audio,
                    output_path=output_path,
                    video_chunks_number=get_video_chunks_number(
                        result.num_frames, result.tiling_config
                    ),
                )

            elapsed_ms = int((time.time() - start_time) * 1000)
            num_frames = result.num_frames
            duration = num_frames / fps

            result = {
                "success": True,
                "seed": seed,
                "duration": round(duration, 2),
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "inference_time_ms": elapsed_ms,
            }

            # Upload to R2 or return as base64
            if r2_config:
                try:
                    import boto3
                    from botocore.config import Config

                    client = boto3.client(
                        "s3",
                        endpoint_url=r2_config["endpoint_url"],
                        aws_access_key_id=r2_config["access_key_id"],
                        aws_secret_access_key=r2_config["secret_access_key"],
                        config=Config(signature_version="s3v4"),
                    )
                    object_key = f"ltx2/results/{uuid.uuid4().hex[:12]}.mp4"
                    client.upload_file(
                        output_path,
                        r2_config["bucket_name"],
                        object_key,
                        ExtraArgs={"ContentType": "video/mp4"},
                    )
                    presigned_url = client.generate_presigned_url(
                        "get_object",
                        Params={
                            "Bucket": r2_config["bucket_name"],
                            "Key": object_key,
                        },
                        ExpiresIn=7200,
                    )
                    result["output_url"] = presigned_url
                    result["r2_key"] = object_key
                except Exception as e:
                    print(f"R2 upload error: {e}")
                    # Fall back to base64
                    with open(output_path, "rb") as f:
                        result["video_base64"] = base64.b64encode(f.read()).decode(
                            "utf-8"
                        )
            else:
                print(
                    "Warning: Returning video as base64 (use R2 for large files)"
                )
                with open(output_path, "rb") as f:
                    result["video_base64"] = base64.b64encode(f.read()).decode(
                        "utf-8"
                    )

            return result

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return {
                "error": f"GPU out of memory: {e}. Try smaller dimensions or fewer frames."
            }
        except Exception as e:
            import traceback

            print(f"Error: {e}")
            print(traceback.format_exc())
            return {"error": f"Internal error: {str(e)}"}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
