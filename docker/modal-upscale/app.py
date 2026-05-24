"""
Modal deployment for Real-ESRGAN image upscaling.

Deploy:
    modal deploy docker/modal-upscale/app.py

Models:
    - general: RealESRGAN_x4plus (default, most images)
    - anime: RealESRGAN_x4plus_anime_6B (illustrations)
    - photo: realesr-general-x4v3 (alternative general)
"""

import modal

app = modal.App("video-toolkit-upscale")

WEIGHT_URLS = {
    "general": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "photo": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}

GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    # Pre-install torch + pinned setuptools to avoid basicsr setup_requires CUDA conflict
    .pip_install(
        "setuptools<70",
        "wheel",
        "numpy>=1.24.0,<2.0",
        "torch==2.1.2",
        "torchvision==0.16.2",
    )
    # basicsr with --no-build-isolation so its setup.py sees the already-installed torch
    .pip_install(
        "basicsr>=1.4.2",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "realesrgan>=0.3.0",
        "facexlib>=0.3.0",
        "gfpgan>=1.3.8",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
        "boto3",
        "requests",
        "fastapi[standard]",
    )
    .run_commands(
        "mkdir -p /app/weights",
        f"python -c \"import requests; "
        f"[open(f'/app/weights/{{k}}.pth', 'wb').write(requests.get(v).content) "
        f"for k, v in {WEIGHT_URLS!r}.items()]\"",
    )
)


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class Upscaler:
    @modal.enter()
    def load_default_model(self):
        """Pre-load the default general model on container start."""
        import torch
        self._upscalers = {}
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Pre-load default model
        self._get_upscaler("general", 4, False)

    def _get_upscaler(self, model: str = "general", scale: int = 4, face_enhance: bool = False):
        """Get or create cached upscaler instance."""
        cache_key = f"{model}_{scale}_{face_enhance}"
        if cache_key in self._upscalers:
            return self._upscalers[cache_key]

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        weight_files = {
            "general": "/app/weights/general.pth",
            "anime": "/app/weights/anime.pth",
            "photo": "/app/weights/photo.pth",
        }

        num_block = 6 if model == "anime" else 23
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=4)

        print(f"Loading model: {model}")
        upscaler = RealESRGANer(
            scale=4,
            model_path=weight_files[model],
            dni_weight=None,
            model=net,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )

        if face_enhance:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path=GFPGAN_URL,
                upscale=scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upscaler,
            )
            self._upscalers[cache_key] = face_enhancer
            return face_enhancer

        self._upscalers[cache_key] = upscaler
        return upscaler

    @modal.fastapi_endpoint(method="POST")
    def upscale(self, request: dict) -> dict:
        import shutil
        import tempfile
        import time
        import uuid
        from pathlib import Path

        import cv2
        import numpy as np
        import requests as req

        image_url = request.get("image_url")
        if not image_url:
            return {"error": "Missing required 'image_url'"}

        scale = request.get("scale", 4)
        model = request.get("model", "general")
        face_enhance = request.get("face_enhance", False)
        output_format = request.get("output_format", "png").lower()
        r2_config = request.get("r2")

        if scale not in [2, 4]:
            return {"error": f"Invalid scale: {scale}. Must be 2 or 4"}
        if model not in ["general", "anime", "photo"]:
            return {"error": f"Invalid model: {model}"}
        if output_format not in ["png", "jpg", "jpeg", "webp"]:
            return {"error": f"Invalid output_format: {output_format}"}

        work_dir = Path(tempfile.mkdtemp(prefix="modal_upscale_"))
        start_time = time.time()

        try:
            # Download image
            print(f"Downloading image from {image_url[:80]}...")
            resp = req.get(image_url, stream=True, timeout=300)
            resp.raise_for_status()

            url_path = image_url.split("?")[0]
            input_ext = Path(url_path).suffix.lower() or ".png"
            input_path = str(work_dir / f"input{input_ext}")

            with open(input_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return {"error": "Failed to read image file"}

            input_height, input_width = img.shape[:2]
            print(f"Input: {input_width}x{input_height}")

            # Upscale
            upscaler = self._get_upscaler(model=model, scale=scale, face_enhance=face_enhance)

            print("Upscaling...")
            inference_start = time.time()

            if face_enhance:
                _, _, output = upscaler.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upscaler.enhance(img, outscale=scale)

            inference_time = time.time() - inference_start
            output_height, output_width = output.shape[:2]
            print(f"Output: {output_width}x{output_height} ({inference_time:.2f}s)")

            # Save output
            output_ext = "jpg" if output_format == "jpeg" else output_format
            output_path = str(work_dir / f"output.{output_ext}")

            if output_format in ["jpg", "jpeg"]:
                cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif output_format == "webp":
                cv2.imwrite(output_path, output, [cv2.IMWRITE_WEBP_QUALITY, 95])
            else:
                cv2.imwrite(output_path, output, [cv2.IMWRITE_PNG_COMPRESSION, 6])

            elapsed = time.time() - start_time

            result = {
                "success": True,
                "input_dimensions": f"{input_width}x{input_height}",
                "output_dimensions": f"{output_width}x{output_height}",
                "scale": scale,
                "model_used": model,
                "face_enhance": face_enhance,
                "output_format": output_format,
                "inference_time_seconds": round(inference_time, 2),
                "processing_time_seconds": round(elapsed, 2),
            }

            # Upload to R2 or return base64
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
                object_key = f"upscale/results/{uuid.uuid4().hex[:12]}.{output_ext}"
                content_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
                client.upload_file(
                    output_path, r2_config["bucket_name"], object_key,
                    ExtraArgs={"ContentType": content_types.get(output_ext, "application/octet-stream")},
                )
                result["output_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                    ExpiresIn=7200,
                )
                result["r2_key"] = object_key
            else:
                import base64
                with open(output_path, "rb") as f:
                    result["image_base64"] = base64.b64encode(f.read()).decode("utf-8")

            return result

        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            return {"error": f"Internal error: {str(e)}"}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
