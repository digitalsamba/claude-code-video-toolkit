#!/usr/bin/env python3
"""
Upscale images using AI (Real-ESRGAN).

Supports cloud processing via RunPod serverless GPUs.

Usage:
    # Cloud processing via RunPod (works from any machine)
    python tools/upscale.py --input image.jpg --output upscaled.png --runpod

    # Specify model and scale
    python tools/upscale.py --input image.jpg --output upscaled.png --model anime --scale 4 --runpod

    # With face enhancement
    python tools/upscale.py --input image.jpg --output upscaled.png --face-enhance --runpod

RunPod Setup:
    1. Create account at runpod.io
    2. Deploy the realesrgan Docker image (see docker/runpod-realesrgan/)
    3. Add to .env:
       RUNPOD_API_KEY=your_key
       RUNPOD_UPSCALE_ENDPOINT_ID=your_endpoint

Models:
    - general: RealESRGAN_x4plus (default, good for most images)
    - anime: RealESRGAN_x4plus_anime_6B (optimized for anime/illustration)
    - photo: realesr-general-x4v3 (alternative general model)

Cost (RunPod):
    - ~$0.01-0.05 per image depending on size
    - Uses RTX 3090 (~$0.34/hr) by default
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests

# Docker image for RunPod endpoint
REALESRGAN_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-realesrgan:v2"
REALESRGAN_TEMPLATE_NAME = "video-toolkit-realesrgan-v2"
REALESRGAN_ENDPOINT_NAME = "video-toolkit-upscale"


def get_runpod_config() -> dict:
    """Get RunPod configuration from environment."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_runpod_api_key
        api_key = get_runpod_api_key()
    except ImportError:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("RUNPOD_API_KEY")

    # Try specific endpoint first, then fall back to general
    from dotenv import load_dotenv
    load_dotenv()
    endpoint_id = os.getenv("RUNPOD_UPSCALE_ENDPOINT_ID")

    return {
        "api_key": api_key,
        "endpoint_id": endpoint_id,
    }


def _get_r2_client():
    """Get boto3 S3 client configured for Cloudflare R2."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if not r2_config:
        return None, None

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
        return client, r2_config
    except ImportError:
        print("  boto3 not installed, skipping R2", file=sys.stderr)
        return None, None


def _upload_to_r2(file_path: str, file_name: str) -> tuple[str | None, str | None]:
    """Upload to Cloudflare R2 and return presigned download URL."""
    client, config = _get_r2_client()
    if not client:
        return None, None

    import uuid
    object_key = f"upscale/{uuid.uuid4().hex[:8]}_{file_name}"

    try:
        client.upload_file(file_path, config["bucket_name"], object_key)

        # Generate presigned URL (valid for 2 hours)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return url, object_key
    except Exception as e:
        print(f"  R2 upload error: {e}", file=sys.stderr)
        return None, None


def _delete_from_r2(object_key: str) -> bool:
    """Delete object from R2 after job completion."""
    client, config = _get_r2_client()
    if not client or not object_key:
        return False

    try:
        client.delete_object(Bucket=config["bucket_name"], Key=object_key)
        return True
    except Exception:
        return False


def _download_from_r2(object_key: str, output_path: str) -> bool:
    """Download object from R2 to local path."""
    client, config = _get_r2_client()
    if not client:
        return False

    try:
        client.download_file(config["bucket_name"], object_key, output_path)
        return True
    except Exception as e:
        print(f"  R2 download error: {e}", file=sys.stderr)
        return False


def upload_to_storage(file_path: str, api_key: str) -> tuple[str | None, str | None]:
    """Upload a file to temporary storage for job input."""
    file_size = Path(file_path).stat().st_size
    file_name = Path(file_path).name

    print(f"Uploading {file_name} ({file_size // 1024}KB)...", file=sys.stderr)

    # Try R2 first if configured
    url, r2_key = _upload_to_r2(file_path, file_name)
    if url:
        print(f"  Upload complete (R2)", file=sys.stderr)
        return url, r2_key

    # Fall back to free services
    upload_services = [
        ("litterbox", _upload_to_litterbox),
        ("0x0.st", _upload_to_0x0),
    ]

    for service_name, upload_func in upload_services:
        try:
            url = upload_func(file_path, file_name)
            if url:
                print(f"  Upload complete ({service_name})", file=sys.stderr)
                return url, None
        except Exception as e:
            print(f"  {service_name} failed: {e}", file=sys.stderr)
            continue

    print("All upload services failed", file=sys.stderr)
    return None, None


def _upload_to_litterbox(file_path: str, file_name: str) -> str | None:
    """Upload to litterbox.catbox.moe (200MB limit, 24h retention)."""
    import subprocess
    result = subprocess.run(
        [
            "curl", "-s",
            "-F", "reqtype=fileupload",
            "-F", "time=24h",
            "-F", f"fileToUpload=@{file_path}",
            "https://litterbox.catbox.moe/resources/internals/api.php",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
    return None


def _upload_to_0x0(file_path: str, file_name: str) -> str | None:
    """Upload to 0x0.st (512MB limit, 30 day retention)."""
    import subprocess
    result = subprocess.run(
        ["curl", "-s", "-F", f"file=@{file_path}", "https://0x0.st"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        url = result.stdout.strip()
        if url.startswith("http"):
            return url
    return None


def submit_runpod_job(
    endpoint_id: str,
    api_key: str,
    image_url: str,
    scale: int = 4,
    model: str = "general",
    face_enhance: bool = False,
    output_format: str = "png",
    r2_config: dict | None = None,
) -> dict | None:
    """Submit an upscale job to RunPod serverless endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    payload = {
        "input": {
            "operation": "upscale",
            "image_url": image_url,
            "scale": scale,
            "model": model,
            "face_enhance": face_enhance,
            "output_format": output_format,
        }
    }

    # Pass R2 credentials for result upload (if configured)
    if r2_config:
        payload["input"]["r2"] = {
            "endpoint_url": r2_config["endpoint_url"],
            "access_key_id": r2_config["access_key_id"],
            "secret_access_key": r2_config["secret_access_key"],
            "bucket_name": r2_config["bucket_name"],
        }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Job submission failed: HTTP {response.status_code}", file=sys.stderr)
            print(f"  Response: {response.text[:500]}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Job submission error: {e}", file=sys.stderr)
        return None


def poll_runpod_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    timeout: int = 300,
    poll_interval: int = 2,
    verbose: bool = True,
) -> dict | None:
    """Poll RunPod job until completion or timeout."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    start_time = time.time()
    last_status = None

    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )

            if response.status_code != 200:
                print(f"Status check failed: HTTP {response.status_code}", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            data = response.json()
            status = data.get("status")

            if verbose and status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed}s] Status: {status}", file=sys.stderr)
                last_status = status

            if status == "COMPLETED":
                return data
            elif status == "FAILED":
                print(f"Job failed: {data.get('error', 'Unknown error')}", file=sys.stderr)
                return data
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                time.sleep(poll_interval)
            else:
                time.sleep(poll_interval)

        except Exception as e:
            print(f"Status check error: {e}", file=sys.stderr)
            time.sleep(poll_interval)

    print(f"Job timed out after {timeout}s", file=sys.stderr)
    return None


def download_from_url(url: str, output_path: str, verbose: bool = True) -> bool:
    """Download file from URL to local path."""
    try:
        if verbose:
            print(f"Downloading result...", file=sys.stderr)

        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if verbose:
            size_kb = Path(output_path).stat().st_size // 1024
            print(f"  Downloaded: {output_path} ({size_kb}KB)", file=sys.stderr)

        return True

    except Exception as e:
        print(f"Download error: {e}", file=sys.stderr)
        return False


def process_with_runpod(
    input_path: str,
    output_path: str,
    scale: int = 4,
    model: str = "general",
    face_enhance: bool = False,
    output_format: str = "png",
    timeout: int = 300,
    verbose: bool = True,
) -> dict:
    """Process image using RunPod serverless endpoint."""
    start_time = time.time()
    r2_keys_to_cleanup = []

    # Get RunPod config
    config = get_runpod_config()
    api_key = config.get("api_key")
    endpoint_id = config.get("endpoint_id")

    if not api_key:
        return {"error": "RUNPOD_API_KEY not set. Add to .env file."}
    if not endpoint_id:
        return {"error": "RUNPOD_UPSCALE_ENDPOINT_ID not set. Run with --setup first."}

    # Get R2 config (optional)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if verbose:
        print(f"Using RunPod endpoint: {endpoint_id}", file=sys.stderr)

    # Upload image
    image_url, image_r2_key = upload_to_storage(input_path, api_key)
    if not image_url:
        return {"error": "Failed to upload image"}
    if image_r2_key:
        r2_keys_to_cleanup.append(image_r2_key)

    # Submit job
    if verbose:
        print(f"Submitting job (scale={scale}, model={model})...", file=sys.stderr)

    job_response = submit_runpod_job(
        endpoint_id=endpoint_id,
        api_key=api_key,
        image_url=image_url,
        scale=scale,
        model=model,
        face_enhance=face_enhance,
        output_format=output_format,
        r2_config=r2_config,
    )

    if not job_response:
        return {"error": "Failed to submit job"}

    job_id = job_response.get("id")
    if not job_id:
        return {"error": f"No job ID in response: {job_response}"}

    if verbose:
        print(f"Job submitted: {job_id}", file=sys.stderr)

    # Poll for completion
    result = poll_runpod_job(
        endpoint_id=endpoint_id,
        api_key=api_key,
        job_id=job_id,
        timeout=timeout,
        verbose=verbose,
    )

    if not result:
        return {"error": "Job timed out or failed to get status"}

    status = result.get("status")
    if status != "COMPLETED":
        error = result.get("error") or result.get("output", {}).get("error") or "Unknown error"
        return {"error": f"Job failed: {error}"}

    # Get output from result
    output = result.get("output", {})
    if isinstance(output, dict) and output.get("error"):
        return {"error": output["error"]}

    # Download result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    output_r2_key = output.get("r2_key") if isinstance(output, dict) else None
    output_url = output.get("output_url") if isinstance(output, dict) else None

    if output_r2_key:
        if verbose:
            print(f"Downloading result from R2...", file=sys.stderr)
        downloaded = _download_from_r2(output_r2_key, output_path)
        if downloaded:
            r2_keys_to_cleanup.append(output_r2_key)
            if verbose:
                size_kb = Path(output_path).stat().st_size // 1024
                print(f"  Downloaded: {output_path} ({size_kb}KB)", file=sys.stderr)

    if not downloaded and output_url:
        downloaded = download_from_url(output_url, output_path, verbose=verbose)

    if not downloaded:
        return {"error": f"No output_url or r2_key in result: {output}"}

    # Cleanup R2 objects
    if r2_keys_to_cleanup:
        for key in r2_keys_to_cleanup:
            _delete_from_r2(key)

    elapsed = time.time() - start_time

    return {
        "success": True,
        "output": output_path,
        "job_id": job_id,
        "processing_time_seconds": round(elapsed, 2),
        "runpod_output": output,
    }


# =============================================================================
# RunPod Setup (GraphQL API)
# =============================================================================

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


def runpod_graphql_query(api_key: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against RunPod API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(
        RUNPOD_GRAPHQL_URL,
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        raise Exception(f"GraphQL request failed: HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")

    return data.get("data", {})


def list_runpod_templates(api_key: str) -> list[dict]:
    """List all user templates."""
    query = """
    query {
        myself {
            podTemplates {
                id
                name
                imageName
                isServerless
            }
        }
    }
    """
    data = runpod_graphql_query(api_key, query)
    templates = data.get("myself", {}).get("podTemplates", [])
    return [t for t in templates if t.get("isServerless")]


def find_realesrgan_template(api_key: str) -> dict | None:
    """Find existing Real-ESRGAN template."""
    templates = list_runpod_templates(api_key)
    for t in templates:
        if t.get("name") == REALESRGAN_TEMPLATE_NAME:
            return t
        if t.get("imageName") == REALESRGAN_DOCKER_IMAGE:
            return t
    return None


def create_runpod_template(api_key: str, verbose: bool = True) -> dict:
    """Create a serverless template for Real-ESRGAN."""
    if verbose:
        print(f"Creating template '{REALESRGAN_TEMPLATE_NAME}'...")

    mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
            isServerless
        }
    }
    """

    variables = {
        "input": {
            "name": REALESRGAN_TEMPLATE_NAME,
            "imageName": REALESRGAN_DOCKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 15,
            "volumeInGb": 0,
            "dockerArgs": "",
            "env": [],
        }
    }

    data = runpod_graphql_query(api_key, mutation, variables)
    template = data.get("saveTemplate")

    if not template or not template.get("id"):
        raise Exception(f"Failed to create template: {data}")

    if verbose:
        print(f"  Template created: {template['id']}")

    return template


def list_runpod_endpoints(api_key: str) -> list[dict]:
    """List all user endpoints."""
    query = """
    query {
        myself {
            endpoints {
                id
                name
                templateId
                gpuIds
                workersMin
                workersMax
                idleTimeout
            }
        }
    }
    """
    data = runpod_graphql_query(api_key, query)
    return data.get("myself", {}).get("endpoints", [])


def find_realesrgan_endpoint(api_key: str, template_id: str) -> dict | None:
    """Find existing Real-ESRGAN endpoint."""
    endpoints = list_runpod_endpoints(api_key)
    for e in endpoints:
        if e.get("name") == REALESRGAN_ENDPOINT_NAME:
            return e
        if e.get("templateId") == template_id:
            return e
    return None


def create_runpod_endpoint(
    api_key: str,
    template_id: str,
    gpu_id: str = "AMPERE_24",
    verbose: bool = True,
) -> dict:
    """Create a serverless endpoint for Real-ESRGAN."""
    if verbose:
        print(f"Creating endpoint '{REALESRGAN_ENDPOINT_NAME}'...")

    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
            gpuIds
            workersMin
            workersMax
            idleTimeout
        }
    }
    """

    variables = {
        "input": {
            "name": REALESRGAN_ENDPOINT_NAME,
            "templateId": template_id,
            "gpuIds": gpu_id,
            "workersMin": 0,
            "workersMax": 1,
            "idleTimeout": 5,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }

    data = runpod_graphql_query(api_key, mutation, variables)
    endpoint = data.get("saveEndpoint")

    if not endpoint or not endpoint.get("id"):
        raise Exception(f"Failed to create endpoint: {data}")

    if verbose:
        print(f"  Endpoint created: {endpoint['id']}")

    return endpoint


def save_endpoint_to_env(endpoint_id: str, verbose: bool = True) -> bool:
    """Save endpoint ID to .env file."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import find_workspace_root
        env_path = find_workspace_root() / ".env"
    except ImportError:
        env_path = Path(__file__).parent.parent / ".env"

    if verbose:
        print(f"Saving endpoint ID to {env_path}...")

    env_content = ""
    if env_path.exists():
        env_content = env_path.read_text()

    lines = env_content.split("\n")
    updated = False
    new_lines = []

    for line in lines:
        if line.startswith("RUNPOD_UPSCALE_ENDPOINT_ID="):
            new_lines.append(f"RUNPOD_UPSCALE_ENDPOINT_ID={endpoint_id}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"RUNPOD_UPSCALE_ENDPOINT_ID={endpoint_id}")

    env_path.write_text("\n".join(new_lines))

    if verbose:
        print(f"  Saved: RUNPOD_UPSCALE_ENDPOINT_ID={endpoint_id}")

    return True


def setup_runpod(gpu_id: str = "AMPERE_24", verbose: bool = True) -> dict:
    """Set up RunPod endpoint for upscale tool."""
    result = {
        "success": False,
        "template_id": None,
        "endpoint_id": None,
        "created_template": False,
        "created_endpoint": False,
    }

    config = get_runpod_config()
    api_key = config.get("api_key")

    if not api_key:
        result["error"] = "RUNPOD_API_KEY not set. Add to .env file first."
        return result

    if verbose:
        print("=" * 60)
        print("RunPod Setup (Real-ESRGAN Upscaler)")
        print("=" * 60)
        print(f"Docker Image: {REALESRGAN_DOCKER_IMAGE}")
        print(f"GPU Type: {gpu_id}")
        print()

    try:
        if verbose:
            print("[1/3] Checking for existing template...")

        template = find_realesrgan_template(api_key)
        if template:
            if verbose:
                print(f"  Found existing template: {template['id']}")
            result["template_id"] = template["id"]
        else:
            template = create_runpod_template(api_key, verbose=verbose)
            result["template_id"] = template["id"]
            result["created_template"] = True

        if verbose:
            print("[2/3] Checking for existing endpoint...")

        endpoint = find_realesrgan_endpoint(api_key, result["template_id"])
        if endpoint:
            if verbose:
                print(f"  Found existing endpoint: {endpoint['id']}")
            result["endpoint_id"] = endpoint["id"]
        else:
            endpoint = create_runpod_endpoint(
                api_key,
                result["template_id"],
                gpu_id=gpu_id,
                verbose=verbose,
            )
            result["endpoint_id"] = endpoint["id"]
            result["created_endpoint"] = True

        if verbose:
            print("[3/3] Saving configuration...")

        save_endpoint_to_env(result["endpoint_id"], verbose=verbose)

        result["success"] = True

        if verbose:
            print()
            print("=" * 60)
            print("Setup Complete!")
            print("=" * 60)
            print(f"Template ID:  {result['template_id']}")
            print(f"Endpoint ID:  {result['endpoint_id']}")
            print()
            print("You can now run:")
            print("  python tools/upscale.py --input image.jpg --output upscaled.png --runpod")
            print()

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale images using AI (Real-ESRGAN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale image 4x using RunPod
  python tools/upscale.py --input photo.jpg --output photo_4x.png --runpod

  # Use anime model for illustrations
  python tools/upscale.py --input art.png --output art_4x.png --model anime --runpod

  # With face enhancement
  python tools/upscale.py --input portrait.jpg --output portrait_4x.png --face-enhance --runpod

  # Setup RunPod endpoint (first-time)
  python tools/upscale.py --setup
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input image file path",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output image file path",
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=4,
        choices=[2, 4],
        help="Upscale factor (default: 4)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="general",
        choices=["general", "anime", "photo"],
        help="Model to use: general (default), anime, or photo",
    )
    parser.add_argument(
        "--face-enhance",
        action="store_true",
        help="Use GFPGAN for face enhancement",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="png",
        choices=["png", "jpg", "webp"],
        help="Output format (default: png)",
    )

    # RunPod options
    parser.add_argument(
        "--runpod",
        action="store_true",
        help="Process on RunPod serverless GPU",
    )
    parser.add_argument(
        "--runpod-timeout",
        type=int,
        default=300,
        help="RunPod job timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up RunPod endpoint automatically",
    )
    parser.add_argument(
        "--setup-gpu",
        type=str,
        default="AMPERE_24",
        choices=["AMPERE_16", "AMPERE_24", "ADA_24", "AMPERE_48"],
        help="GPU type for RunPod endpoint (default: AMPERE_24)",
    )

    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without processing",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.json

    # Handle --setup
    if args.setup:
        result = setup_runpod(gpu_id=args.setup_gpu, verbose=verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)
        sys.exit(0)

    # Validate required arguments
    if not args.input:
        print("Error: --input is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Dry run
    if args.dry_run:
        config = get_runpod_config()
        result = {
            "dry_run": True,
            "input": args.input,
            "output": args.output,
            "scale": args.scale,
            "model": args.model,
            "face_enhance": args.face_enhance,
            "output_format": args.format,
            "runpod": args.runpod,
            "endpoint_configured": bool(config.get("endpoint_id")),
            "api_key_configured": bool(config.get("api_key")),
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would process:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        return

    # RunPod processing
    if args.runpod:
        if verbose:
            print("Processing with RunPod cloud GPU...")

        result = process_with_runpod(
            input_path=args.input,
            output_path=args.output,
            scale=args.scale,
            model=args.model,
            face_enhance=args.face_enhance,
            output_format=args.format,
            timeout=args.runpod_timeout,
            verbose=verbose,
        )

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            output_info = result.get("runpod_output", {})
            input_dims = output_info.get("input_dimensions", "?")
            output_dims = output_info.get("output_dimensions", "?")
            print(f"Upscaled: {result['output']}")
            print(f"  {input_dims} -> {output_dims}")
            print(f"  Processing time: {result.get('processing_time_seconds', 0):.1f}s")

        return

    # Local processing not implemented yet
    print("Error: Local processing not implemented. Use --runpod for cloud processing.", file=sys.stderr)
    print("       Or run --setup first to configure RunPod endpoint.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
