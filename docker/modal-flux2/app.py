"""
Modal deployment for FLUX.2 Klein 4B.

Text-to-image generation and image editing.
Equivalent to docker/runpod-flux2/handler.py but deployed on Modal.

Deploy:
    modal deploy docker/modal-flux2/app.py

Input format (POST JSON to web endpoint):
{
    "operation": "generate" | "edit",
    "prompt": str,
    "image_base64": str,           # Required for edit
    "images_base64": [str],        # Optional additional reference images
    "width": int,                  # Default: 1024
    "height": int,                 # Default: 1024
    "num_inference_steps": int,    # Default: 4 (generate), 50 (edit)
    "guidance_scale": float,       # Default: 1.0 (generate), 4.0 (edit)
    "seed": int,
    "r2": dict                     # Optional R2 upload config
}
"""

import modal

app = modal.App("video-toolkit-flux2")

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# Container image — mirrors docker/runpod-flux2/Dockerfile
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        # Flux2KleinPipeline (diffusers git main) imports Qwen3ForCausalLM,
        # which landed in transformers 4.51. The old >=4.45 floor let a
        # cached pre-Qwen3 layer sit under a fresh diffusers, crash-looping
        # the container at load_pipeline with ModuleNotFoundError.
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "boto3",
        "requests",
        "fastapi[standard]",
        "huggingface_hub>=0.25.0",
    )
    # Flux2KleinPipeline is not in any tagged diffusers release yet, so this has
    # to come from git. Pin the commit: an unpinned ref silently re-resolves on
    # every rebuild, and when the new main needs a dependency the cached layer
    # below it does not satisfy, the container dies in @modal.enter() -- which
    # reads as a slow cold start rather than an error (#71, #74).
    # Bump deliberately, and smoke-test the endpoint after.
    .run_commands(
        "pip install --no-cache-dir "
        "git+https://github.com/huggingface/diffusers@119c339551f68ea523b9f204120b929e56342421"
    )
    # Bake model weights into the image
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}')"
        '"'
    )
)


@app.cls(
    image=image,
    gpu="A10G",
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

        print(f"Loading Flux2 Klein pipeline from {MODEL_ID}...")
        start = time.time()

        from diffusers import Flux2KleinPipeline

        self.pipeline = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
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

                num_inference_steps = request.get("num_inference_steps", 50)
                guidance_scale = request.get("guidance_scale", 4.0)

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
                num_inference_steps = request.get("num_inference_steps", 4)
                guidance_scale = request.get("guidance_scale", 1.0)

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
