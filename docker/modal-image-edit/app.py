"""
Modal deployment for Qwen image editing.

Deploy (the model is a deploy-time choice):
    modal deploy docker/modal-image-edit/app.py                                  # Qwen-Image-Edit-2511
    IMAGE_EDIT_MODEL=qwen-image-2.1 modal deploy docker/modal-image-edit/app.py  # Qwen-Image-2.1

Capabilities: background replacement, style transfer, custom edits, multi-image
merge (3 images on 2511, up to 10 on 2.1), RGBA in/out on 2.1.

LICENSES -- they differ, which is why 2511 stays the default:
- Qwen-Image-Edit-2511: Apache-2.0, commercial OK.
- Qwen-Image-2.1: Qwen Research License, non-commercial ("research or evaluation
  purposes only"). Newer and stronger -- one model for generation and editing --
  but opt in only for personal/non-commercial work.

Both variants share one request/response contract, so tools/image_edit.py works
against either. Defaults the caller leaves out (steps, image count) come from
the deployed variant; the response's "model" says which one ran.

Input format (POST JSON to web endpoint):
{
    "image_base64": str,           # Required -- primary image ("Picture 1" in the prompt)
    "images_base64": [str],        # Optional -- more images (masks, references)
    "prompt": str,                 # Required
    "num_inference_steps": int,    # Default: 8 (2511) / 40 (2.1)
    "guidance_scale": float,       # Default: 1.0. On 2.1, >1 enables true CFG
    "negative_prompt": str,        # Optional
    "seed": int,
    "output_resolution": int,      # 2.1 only. Default: 1024 (output area ~= this squared)
    "width": int, "height": int,   # 2.1 only. Optional explicit output size
    "r2": dict                     # Optional R2 upload config
}
"""

import os

import modal

app = modal.App("video-toolkit-image-edit")

VARIANTS = {
    "qwen-image-edit-2511": {
        "repo": "Qwen/Qwen-Image-Edit-2511",
        "pipeline": "QwenImageEditPlusPipeline",
        "steps": 8,
        "max_images": 3,
    },
    "qwen-image-2.1": {
        "repo": "Qwen/Qwen-Image-2.1",
        # Not step-distilled: 40 is the model card's default.
        "pipeline": "QwenImage21Pipeline",
        "steps": 40,
        "max_images": 10,
    },
}

# Read at deploy time, and again inside the container when it re-imports this
# file -- the image env below carries the deploy-time value into the container.
IMAGE_EDIT_MODEL = os.environ.get("IMAGE_EDIT_MODEL", "qwen-image-edit-2511")
if IMAGE_EDIT_MODEL not in VARIANTS:
    raise ValueError(f"IMAGE_EDIT_MODEL must be one of {sorted(VARIANTS)}, got {IMAGE_EDIT_MODEL!r}")
VARIANT = VARIANTS[IMAGE_EDIT_MODEL]
MODEL_ID = VARIANT["repo"]

# QwenImage21Pipeline is only on diffusers main (added in #14804); this is the
# last commit that touched the 2.1 pipeline/model files, 2026-09-30. It also
# carries QwenImageEditPlusPipeline, so both variants share one pin. An unpinned
# ref re-resolves on every rebuild, and a main that needs something the cached
# layers below it don't have kills the container in @modal.enter() (#71, #74).
# Switch to a tagged release once one ships QwenImage21Pipeline.
DIFFUSERS_COMMIT = "c60830ee365d520ab52b110dda562dd26f7b4d7f"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    # diffusers main wants torch>=2.6. cu128 is the newest CUDA line that
    # doesn't need a CUDA 13 driver on the host.
    .pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # The 2.1 model card asks for transformers>=5.17 (Qwen3VLForConditionalGeneration
        # + the processor the checkpoint was saved with).
        "transformers==5.17.0",
        "accelerate>=1.1.0",
        "safetensors>=0.8.0",
        "einops>=0.7.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.9.0",
        "sentencepiece",
        "protobuf",
        "boto3",
        "requests",
        "fastapi[standard]",
        # diffusers main needs >=1.32,<2.0; transformers 5.17 needs <2.0.
        "huggingface_hub>=1.32.0,<2.0",
    )
    .run_commands(
        "pip install --no-cache-dir "
        f"git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}"
    )
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}')"
        '"'
    )
    .env({"IMAGE_EDIT_MODEL": IMAGE_EDIT_MODEL})
)


@app.cls(
    image=image,
    # 2511's 20B DiT needs the 80GB card. 2.1 is ~33GB (17.5GB Qwen3-VL text
    # encoder + 14.2GB DiT + VAE), and 80GB leaves room for 10 condition images.
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class ImageEditor:
    @modal.enter()
    def load_pipeline(self):
        import diffusers
        import torch

        print(f"Loading {MODEL_ID} ({VARIANT['pipeline']})...")
        pipeline_cls = getattr(diffusers, VARIANT["pipeline"])
        self.pipeline = pipeline_cls.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")
        print("Pipeline ready")

    @modal.fastapi_endpoint(method="POST")
    def edit(self, request: dict) -> dict:
        import base64
        import io
        import math
        import random
        import time
        import uuid

        import torch
        from PIL import Image

        image_base64 = request.get("image_base64")
        prompt = request.get("prompt")

        if not image_base64:
            return {"error": "Missing required 'image_base64'"}
        if not prompt:
            return {"error": "Missing required 'prompt'"}

        is_21 = IMAGE_EDIT_MODEL == "qwen-image-2.1"
        num_inference_steps = request.get("num_inference_steps") or VARIANT["steps"]
        guidance_scale = request.get("guidance_scale", 1.0)
        negative_prompt = request.get("negative_prompt")
        seed = request.get("seed")
        r2_config = request.get("r2")

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        start_time = time.time()

        try:
            def decode_b64_image(b64: str) -> Image.Image:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                # Keep alpha on 2.1: it edits transparent layers natively.
                has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
                return img.convert("RGBA" if is_21 and has_alpha else "RGB")

            # Primary image + optional references
            all_images = [decode_b64_image(image_base64)]
            for ref_b64 in request.get("images_base64", [])[: VARIANT["max_images"] - 1]:
                all_images.append(decode_b64_image(ref_b64))

            image_input = all_images if len(all_images) > 1 else all_images[0]

            generator = torch.Generator(device="cuda").manual_seed(seed)

            kwargs = {
                "prompt": prompt,
                "image": image_input,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }

            if is_21:
                # The 2.1 pipeline has no guidance_scale -- it is meant to be
                # sampled without guidance. Map ours onto true CFG, which only
                # switches on with a negative prompt, so supply a blank one if
                # the caller didn't.
                kwargs["true_cfg_scale"] = guidance_scale
                if guidance_scale > 1.0:
                    kwargs["negative_prompt"] = negative_prompt or " "

                output_resolution = request.get("output_resolution", 1024)
                width = request.get("width")
                height = request.get("height")
                # The pipeline sizes the output from the *last* condition image.
                # With references attached, size it from the primary instead.
                if len(all_images) > 1 and not (width and height):
                    w0, h0 = all_images[0].size
                    ratio = w0 / h0
                    area = output_resolution * output_resolution
                    width = round(math.sqrt(area * ratio) / 32) * 32
                    height = round(math.sqrt(area * ratio) / ratio / 32) * 32
                kwargs["output_resolution"] = output_resolution
                if width and height:
                    kwargs["width"] = width
                    kwargs["height"] = height
            else:
                kwargs["guidance_scale"] = guidance_scale
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt

            output = self.pipeline(**kwargs)
            output_image = output.images[0]

            # The 2.1 VAE is RGBA. Only hand back an alpha channel when the model
            # actually made something transparent; keep normal edits plain RGB.
            if output_image.mode == "RGBA" and output_image.getchannel("A").getextrema() == (255, 255):
                output_image = output_image.convert("RGB")

            buffer = io.BytesIO()
            output_image.save(buffer, format="PNG")
            output_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            elapsed_ms = int((time.time() - start_time) * 1000)

            result = {
                "success": True,
                "edited_image_base64": output_base64,
                "seed": seed,
                "inference_time_ms": elapsed_ms,
                "image_size": list(output_image.size),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "images_used": len(all_images),
                "model": MODEL_ID,
            }

            # R2 upload
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
                object_key = f"image-edit/results/{uuid.uuid4().hex[:12]}.png"
                client.put_object(
                    Bucket=r2_config["bucket_name"],
                    Key=object_key,
                    Body=base64.b64decode(output_base64),
                    ContentType="image/png",
                )
                result["output_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                    ExpiresIn=7200,
                )
                result["r2_key"] = object_key

            return result

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return {"error": f"GPU out of memory: {e}. Try smaller image."}
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            return {"error": f"Internal error: {str(e)}"}
