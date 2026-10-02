"""
Modal deployment for FLUX.2 [klein].

Text-to-image generation and image editing.
Equivalent to docker/runpod-flux2/handler.py but deployed on Modal.

Deploy (the checkpoint is a deploy-time choice):
    modal deploy docker/modal-flux2/app.py                       # klein-4B
    FLUX2_MODEL=klein-9b modal deploy docker/modal-flux2/app.py  # klein-9B

LICENSES -- they differ, which is why 4B stays the default:
- klein-4B: Apache-2.0, commercial OK, ungated, fits an A10G.
- klein-9B: FLUX Non-Commercial License, which also requires filters or manual
  review of outputs. Sharper, but opt in only for personal/non-commercial work.
  Gated on Hugging Face: accept the terms at
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and put a token in the
  `huggingface-token` Modal secret (key HF_TOKEN) before deploying. Needs an L40S.

Input format (POST JSON to web endpoint):
{
    "operation": "generate" | "edit",
    "prompt": str,
    "image_base64": str,           # Required for edit
    "images_base64": [str],        # Optional additional reference images
    "width": int,                  # Default: 1024
    "height": int,                 # Default: 1024
    "num_inference_steps": int,    # Default: 4 (klein is step-distilled to 4)
    "guidance_scale": float,       # Default: 1.0 (ignored by distilled klein)
    "seed": int,
    "r2": dict                     # Optional R2 upload config
}
"""

import os

import modal

app = modal.App("video-toolkit-flux2")

# 9B is plain 9B, not FLUX.2-klein-9b-kv. The kv variant caches reference-image
# keys/values after step 0, which only pays off for multi-reference *edits*;
# this endpoint is mostly text-to-image, where it gains nothing, and its
# pipeline (Flux2KleinKVPipeline) has a different call signature.
VARIANTS = {
    "klein-4b": {
        "repo": "black-forest-labs/FLUX.2-klein-4B",
        "single_file": "flux-2-klein-4b.safetensors",
        "gated": False,
        "gpu": "A10G",
    },
    "klein-9b": {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "single_file": "flux-2-klein-9b.safetensors",
        "gated": True,
        # The 9B transformer (18.2GB) + Qwen3-8B text encoder (16.4GB) sit at
        # ~35GB resident in bf16 -- past the A10G's 24GB.
        "gpu": "L40S",
    },
}

# Read at deploy time, and again inside the container when it re-imports this
# file -- the image env below carries the deploy-time value into the container.
FLUX2_MODEL = os.environ.get("FLUX2_MODEL", "klein-4b")
if FLUX2_MODEL not in VARIANTS:
    raise ValueError(f"FLUX2_MODEL must be one of {sorted(VARIANTS)}, got {FLUX2_MODEL!r}")
VARIANT = VARIANTS[FLUX2_MODEL]
MODEL_ID = VARIANT["repo"]

# Distilled VAE decoder (Apache-2.0): ~1.4x faster decode and ~1.4x less decode
# VRAM, encoder unchanged. A drop-in for every open FLUX.2 model. Needs
# AutoencoderKLFlux2's decoder_block_out_channels, i.e. diffusers>=0.38.
VAE_ID = "black-forest-labs/FLUX.2-small-decoder"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # cu128 is the newest CUDA line that doesn't need a CUDA 13 driver on the host.
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # A tagged release at last: Flux2KleinPipeline landed in 0.37.0. Pinned
        # exactly so a rebuild can't drift onto a release the cached layers
        # below don't satisfy (#71, #74). Bump deliberately, smoke-test after.
        "diffusers==0.40.0",
        # diffusers 0.40 needs huggingface_hub>=1.23, which transformers 4.x
        # can't share (it caps hub <1.0) -- so transformers 5.
        "transformers==5.17.0",
        "huggingface_hub>=1.32.0,<2.0",
        "accelerate>=1.1.0",
        "safetensors>=0.8.0",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "boto3",
        "requests",
        "fastapi[standard]",
    )
    # Bake model weights into the image. The 9B repo is gated, so its download
    # needs HF_TOKEN. Skip the single-file checkpoint at the repo root --
    # from_pretrained reads the diffusers-format subfolders.
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}', "
        f"ignore_patterns=['{VARIANT['single_file']}', '*.jpg'])"
        '"',
        secrets=[modal.Secret.from_name("huggingface-token")] if VARIANT["gated"] else [],
    )
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{VAE_ID}', "
        "allow_patterns=['config.json', 'diffusion_pytorch_model.safetensors'])"
        '"'
    )
    # Everything is baked in, so never call the Hub at load time -- a gated
    # repo would otherwise need the token in the running container too.
    .env({"HF_HUB_OFFLINE": "1", "FLUX2_MODEL": FLUX2_MODEL})
)


@app.cls(
    image=image,
    gpu=VARIANT["gpu"],
    timeout=600,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class Flux2:
    """FLUX.2 Klein inference class."""

    @modal.enter()
    def load_pipeline(self):
        """Load the Flux2 Klein pipeline when the container starts."""
        import time
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}, VRAM: {props.total_memory // (1024**3)}GB")

        print(f"Loading Flux2 Klein pipeline from {MODEL_ID} (VAE: {VAE_ID})...")
        start = time.time()

        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline

        vae = AutoencoderKLFlux2.from_pretrained(VAE_ID, torch_dtype=torch.bfloat16)
        self.pipeline = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

        print(f"Pipeline loaded in {time.time() - start:.1f}s")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """Web endpoint — accepts same payload format as RunPod handler."""
        import base64
        import io
        import random
        import time
        import uuid

        import torch
        from PIL import Image

        operation = request.get("operation", "generate")
        prompt = request.get("prompt")

        if not prompt:
            return {"error": "Missing required 'prompt' in input"}

        seed = request.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        r2_config = request.get("r2")
        start_time = time.time()

        # klein is step-distilled to 4 steps for both generation and editing,
        # and the pipeline ignores guidance_scale for distilled checkpoints.
        num_inference_steps = request.get("num_inference_steps", 4)
        guidance_scale = request.get("guidance_scale", 1.0)

        try:
            generator = torch.Generator(device="cuda").manual_seed(seed)

            if operation == "edit":
                # Image editing
                image_base64 = request.get("image_base64")
                if not image_base64:
                    return {"error": "Missing required 'image_base64' for edit operation"}

                # Decode images
                def decode_b64_image(b64: str) -> Image.Image:
                    if "," in b64:
                        b64 = b64.split(",", 1)[1]
                    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

                all_images = [decode_b64_image(image_base64)]
                for ref_b64 in request.get("images_base64", [])[:2]:
                    all_images.append(decode_b64_image(ref_b64))

                image_input = all_images if len(all_images) > 1 else all_images[0]

                output = self.pipeline(
                    prompt=prompt,
                    image=image_input,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            else:
                # Text-to-image generation
                width = request.get("width", 1024)
                height = request.get("height", 1024)

                output = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

            output_image = output.images[0]

            # Encode to base64
            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            output_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            elapsed_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "image_base64": output_base64,
                "seed": seed,
                "inference_time_ms": elapsed_ms,
                "image_size": list(output_image.size),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "model": MODEL_ID,
            }

            # Upload to R2 if configured
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
                    object_key = f"flux2/results/{uuid.uuid4().hex[:12]}.png"
                    client.put_object(
                        Bucket=r2_config["bucket_name"],
                        Key=object_key,
                        Body=base64.b64decode(output_base64),
                        ContentType="image/png",
                    )
                    presigned_url = client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                        ExpiresIn=7200,
                    )
                    result["output_url"] = presigned_url
                    result["r2_key"] = object_key
                except Exception as e:
                    print(f"R2 upload error: {e}")

            return result

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return {"error": f"GPU out of memory: {e}. Try smaller dimensions."}
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            return {"error": f"Internal error: {str(e)}"}
