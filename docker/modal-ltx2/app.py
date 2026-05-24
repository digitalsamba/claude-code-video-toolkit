"""
Modal deployment for LTX-2.3 video generation.

Text-to-video and image-to-video generation using LTX-2.3 22B DiT model.
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

# HuggingFace model repos (2.3 weights are split across repos)
HF_REPO = "Lightricks/LTX-2.3"
HF_REPO_FP8 = "Lightricks/LTX-2.3-fp8"
# Local path inside the container where weights are stored
MODEL_DIR = "/models/ltx2"
# Gemma 3 text encoder (quantized variant — smaller, faster)
GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"
GEMMA_DIR = "/models/gemma3"
# Style LoRAs baked into the image. Map of CLI key → {repo, filename, strength}.
# Add new LoRAs here to ship them in the container.
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
    # PyTorch with CUDA 12.6 (separate index)
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    # Other dependencies (from PyPI)
    .pip_install(
        "einops",
        "numpy>=1.26",
        "transformers==4.57.6",
        "safetensors",
        "accelerate",
        "scipy>=1.14",
        "av",
        "tqdm",
        "Pillow",
        "boto3",
        "requests",
        "fastapi[standard]",
        "huggingface_hub>=0.25.0",
    )
    # Install flash-attn for optimized attention (optional, best-effort)
    .pip_install("packaging")
    .run_commands(
        "pip install --no-cache-dir flash-attn --no-build-isolation "
        "|| echo 'flash-attn not available, using SDPA fallback'"
    )
    # Clone LTX-2 and install its packages
    .run_commands(
        "git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/ltx2",
        "pip install -e /app/ltx2/packages/ltx-core",
        "pip install -e /app/ltx2/packages/ltx-pipelines",
    )
    # Bake LTX-2.3 model weights — full quality bf16 dev checkpoint + distilled LoRA + upsampler
    # Dev checkpoint (46.1GB) + distilled LoRA (7.6GB) + spatial upsampler (1GB) = ~55GB
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{HF_REPO}', local_dir='{MODEL_DIR}', "
        "allow_patterns=['ltx-2.3-22b-dev.safetensors', "
        "'ltx-2.3-22b-distilled-lora-384.safetensors', "
        "'ltx-2.3-spatial-upscaler-x2-1.1.safetensors'])"
        "\"",
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    # Bake Gemma 3 12B text encoder (~7GB quantized)
    # Gemma is a gated model — needs HF_TOKEN at build time
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{GEMMA_REPO}', local_dir='{GEMMA_DIR}', "
        ")"
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
    """LTX-2.3 video generation."""

    @modal.enter()
    def load_pipeline(self):
        """Load the base LTX-2 pipeline when the container starts."""
        import glob
        import os

        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}, VRAM: {props.total_memory // (1024**3)}GB")

        def find_file(pattern):
            matches = glob.glob(os.path.join(MODEL_DIR, pattern))
            return matches[0] if matches else None

        self._checkpoint = find_file("ltx-2.3-22b-dev.safetensors")
        self._distilled_lora_path = find_file("ltx-2.3-22b-distilled-lora-*.safetensors")
        self._spatial_upsampler = find_file("ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

        if not self._checkpoint:
            raise RuntimeError(
                f"LTX-2.3 checkpoint not found in {MODEL_DIR}. "
                f"Files: {os.listdir(MODEL_DIR)}"
            )

        print(f"  Checkpoint: {self._checkpoint}")
        print(f"  Distilled LoRA: {self._distilled_lora_path}")
        print(f"  Spatial upsampler: {self._spatial_upsampler}")
        print(f"  Gemma: {GEMMA_DIR}")

        self.pipeline = None
        self._current_style_lora = None
        self._build_pipeline(style_lora=None)

    def _build_pipeline(self, style_lora):
        """Construct the pipeline with an optional style LoRA key.

        Called at cold start (no style LoRA) and on per-request style swaps.
        Rebuilding costs ~30–60s of weight loading; frees the previous
        pipeline's VRAM first to avoid OOM on A100-80GB.
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

        distilled = []
        if self._distilled_lora_path:
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
        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=self._checkpoint,
            distilled_lora=distilled,
            spatial_upsampler_path=self._spatial_upsampler,
            gemma_root=GEMMA_DIR,
            loras=style_loras,
        )
        self._current_style_lora = style_lora
        print(f"Pipeline loaded in {time.time() - start:.1f}s")

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
        # LoRA differs from what's currently loaded — same-LoRA back-to-back
        # calls skip the ~60s reload.
        style_lora = request.get("lora")
        if style_lora and style_lora not in AVAILABLE_LORAS:
            return {
                "error": f"Unknown LoRA '{style_lora}'. "
                f"Available: {list(AVAILABLE_LORAS) or '(none)'}"
            }
        if style_lora != self._current_style_lora:
            self._build_pipeline(style_lora=style_lora)

        # Enforce dimension constraints
        width = (width // 64) * 64
        height = (height // 64) * 64

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

                from ltx_pipelines.utils.args import ImageConditioningInput

                images = [
                    ImageConditioningInput(
                        path=img_path,
                        frame_idx=0,
                        strength=0.8,
                        crf=0,  # lossless — no H.264 preprocessing
                    )
                ]
            except Exception as e:
                return {"error": f"Failed to decode input image: {e}"}
        start_time = time.time()

        try:
            from ltx_core.components.guiders import MultiModalGuiderParams

            video_guider = MultiModalGuiderParams(
                cfg_scale=3.0,
                stg_scale=1.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            )
            audio_guider = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=1.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            )

            print(
                f"Generating: {width}x{height}, {num_frames} frames, "
                f"{num_inference_steps} steps, seed={seed}"
            )

            # CRITICAL: torch.inference_mode() prevents PyTorch from retaining
            # the autograd graph. Without it, the Gemma text encoder's ~37GB of
            # activations stay in VRAM even after del, causing OOM when the
            # transformer loads. This is a known issue (GitHub #152) — the
            # pipeline's __call__ doesn't set inference_mode, but the CLI does.
            # The scope must include encode_video() because the pipeline returns
            # a lazy iterator — frames are decoded when the iterator is consumed.
            import os

            output_path = os.path.join(work_dir, "output.mp4")

            from ltx_pipelines.utils.media_io import encode_video

            with torch.inference_mode():
                video_iter, audio = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=float(fps),
                    num_inference_steps=num_inference_steps,
                    video_guider_params=video_guider,
                    audio_guider_params=audio_guider,
                    images=images if images else [],
                )

                encode_video(
                    video=video_iter,
                    fps=fps,
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=1,
                )

            elapsed_ms = int((time.time() - start_time) * 1000)
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
