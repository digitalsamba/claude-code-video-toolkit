"""
Modal deployment for Qwen-Image-Edit (AI image editing).

Deploy:
    modal deploy docker/modal-image-edit/app.py

Capabilities: background replacement, style transfer, custom edits, multi-image merge.
"""

import modal

app = modal.App("video-toolkit-image-edit")

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.45.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "einops>=0.7.0",
        "Pillow>=10.0.0",
        "numpy>=1.26.0",
        "opencv-python-headless>=4.9.0",
        "sentencepiece",
        "protobuf",
        "boto3",
        "requests",
        "fastapi[standard]",
        "huggingface_hub>=0.25.0",
    )
    .run_commands("pip install --no-cache-dir git+https://github.com/huggingface/diffusers")
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL_ID}')"
        '"'
    )
)


@app.cls(
    image=image,
    # A100-80GB, not plain "A100" (= 40GB): the bfloat16 pipeline lands at ~38.8GiB
    # resident, so a 40GB card OOMs on the first allocation after load. The RunPod
    # deployment of this same model runs on AMPERE_80 for the same reason.
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class ImageEditor:
    @modal.enter()
    def load_pipeline(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        print(f"Loading {MODEL_ID}...")
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")
        print("Pipeline ready")

    @modal.fastapi_endpoint(method="POST")
    def edit(self, request: dict) -> dict:
        import base64
        import io
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

        num_inference_steps = request.get("num_inference_steps", 4)
        guidance_scale = request.get("guidance_scale", 1.0)
        seed = request.get("seed")
        negative_prompt = request.get("negative_prompt")
        r2_config = request.get("r2")

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        start_time = time.time()

        try:
            def decode_b64_image(b64: str) -> Image.Image:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

            # Primary image + optional references
            all_images = [decode_b64_image(image_base64)]
            for ref_b64 in request.get("images_base64", [])[:2]:
                all_images.append(decode_b64_image(ref_b64))

            image_input = all_images if len(all_images) > 1 else all_images[0]

            generator = torch.Generator(device="cuda").manual_seed(seed)

            kwargs = {
                "prompt": prompt,
                "image": image_input,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt

            output = self.pipeline(**kwargs)
            output_image = output.images[0]

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
