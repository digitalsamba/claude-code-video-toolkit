#!/usr/bin/env python3
"""
Generate talking head videos using SadTalker.

Animates a static portrait image to match audio, creating a realistic
talking head video suitable for explainer videos, avatars, and narration.

Usage:
    # Basic usage
    python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --output talking.mp4

    # With options
    python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --still --output talking.mp4

    # Higher resolution (512px)
    python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --size 512 --output talking.mp4

    # Setup endpoint
    python tools/sadtalker.py --setup

Setup:
    1. Create account at runpod.io
    2. Run: python tools/sadtalker.py --setup
    3. Or manually deploy docker/runpod-sadtalker/ and add endpoint ID to .env

Cost:
    - ~$0.04 per 30 seconds of video
    - ~$0.27 per 3 minutes of video
    - Uses RTX 4090 ($0.00074/sec) by default
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# Docker image for RunPod endpoint
SADTALKER_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-sadtalker:latest"
SADTALKER_TEMPLATE_NAME = "video-toolkit-sadtalker"
SADTALKER_ENDPOINT_NAME = "video-toolkit-sadtalker"


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

    from dotenv import load_dotenv
    load_dotenv()
    endpoint_id = os.getenv("RUNPOD_SADTALKER_ENDPOINT_ID")

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


def _upload_to_r2(file_path: str, prefix: str) -> tuple[str | None, str | None]:
    """Upload to Cloudflare R2 and return presigned download URL."""
    client, config = _get_r2_client()
    if not client:
        return None, None

    import uuid
    file_name = Path(file_path).name
    object_key = f"{prefix}/{uuid.uuid4().hex[:8]}_{file_name}"

    try:
        client.upload_file(file_path, config["bucket_name"], object_key)

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


def upload_to_storage(file_path: str, prefix: str) -> tuple[str | None, str | None]:
    """Upload a file to temporary storage for job input."""
    file_size = Path(file_path).stat().st_size
    file_name = Path(file_path).name

    print(f"Uploading {file_name} ({file_size // 1024}KB)...", file=sys.stderr)

    # Try R2 first if configured
    url, r2_key = _upload_to_r2(file_path, prefix)
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
    audio_url: str,
    still_mode: bool = False,
    enhancer: str = "gfpgan",
    preprocess: str = "crop",
    size: int = 256,
    expression_scale: float = 1.0,
    pose_style: int = 0,
    r2_config: dict | None = None,
) -> dict | None:
    """Submit a SadTalker job to RunPod serverless endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"

    payload = {
        "input": {
            "image_url": image_url,
            "audio_url": audio_url,
            "still_mode": still_mode,
            "enhancer": enhancer,
            "preprocess": preprocess,
            "size": size,
            "expression_scale": expression_scale,
            "pose_style": pose_style,
        }
    }

    # Pass R2 credentials for result upload
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
    timeout: int = 600,
    poll_interval: int = 5,
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
    image_path: str,
    audio_path: str,
    output_path: str,
    still_mode: bool = False,
    enhancer: str = "gfpgan",
    preprocess: str = "crop",
    size: int = 256,
    expression_scale: float = 1.0,
    pose_style: int = 0,
    timeout: int = 600,
    verbose: bool = True,
) -> dict:
    """Process image+audio using RunPod serverless endpoint."""
    start_time = time.time()
    r2_keys_to_cleanup = []

    # Get RunPod config
    config = get_runpod_config()
    api_key = config.get("api_key")
    endpoint_id = config.get("endpoint_id")

    if not api_key:
        return {"error": "RUNPOD_API_KEY not set. Add to .env file."}
    if not endpoint_id:
        return {"error": "RUNPOD_SADTALKER_ENDPOINT_ID not set. Run with --setup first."}

    # Get R2 config (optional but recommended)
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import get_r2_config
        r2_config = get_r2_config()
    except ImportError:
        r2_config = None

    if not r2_config:
        print("Warning: R2 not configured. Video will be returned as base64.", file=sys.stderr)

    if verbose:
        print(f"Using RunPod endpoint: {endpoint_id}", file=sys.stderr)

    # Upload image
    image_url, image_r2_key = upload_to_storage(image_path, "sadtalker/input")
    if not image_url:
        return {"error": "Failed to upload image"}
    if image_r2_key:
        r2_keys_to_cleanup.append(image_r2_key)

    # Upload audio
    audio_url, audio_r2_key = upload_to_storage(audio_path, "sadtalker/input")
    if not audio_url:
        return {"error": "Failed to upload audio"}
    if audio_r2_key:
        r2_keys_to_cleanup.append(audio_r2_key)

    # Submit job
    if verbose:
        print(f"Submitting job (size={size}, enhancer={enhancer})...", file=sys.stderr)

    job_response = submit_runpod_job(
        endpoint_id=endpoint_id,
        api_key=api_key,
        image_url=image_url,
        audio_url=audio_url,
        still_mode=still_mode,
        enhancer=enhancer,
        preprocess=preprocess,
        size=size,
        expression_scale=expression_scale,
        pose_style=pose_style,
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
    output_url = output.get("video_url") if isinstance(output, dict) else None

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
        # Try base64 fallback
        video_base64 = output.get("video_base64")
        if video_base64:
            import base64
            Path(output_path).write_bytes(base64.b64decode(video_base64))
            downloaded = True
            if verbose:
                size_kb = Path(output_path).stat().st_size // 1024
                print(f"  Decoded from base64: {output_path} ({size_kb}KB)", file=sys.stderr)

    if not downloaded:
        return {"error": f"No video in result: {list(output.keys()) if isinstance(output, dict) else output}"}

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
        "duration_seconds": output.get("duration_seconds"),
        "chunks_processed": output.get("chunks_processed"),
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


def find_sadtalker_template(api_key: str) -> dict | None:
    """Find existing SadTalker template."""
    templates = list_runpod_templates(api_key)
    for t in templates:
        if t.get("name") == SADTALKER_TEMPLATE_NAME:
            return t
        if t.get("imageName") == SADTALKER_DOCKER_IMAGE:
            return t
    return None


def create_runpod_template(api_key: str, verbose: bool = True) -> dict:
    """Create a serverless template for SadTalker."""
    if verbose:
        print(f"Creating template '{SADTALKER_TEMPLATE_NAME}'...")

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
            "name": SADTALKER_TEMPLATE_NAME,
            "imageName": SADTALKER_DOCKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 20,
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


def find_sadtalker_endpoint(api_key: str, template_id: str) -> dict | None:
    """Find existing SadTalker endpoint."""
    endpoints = list_runpod_endpoints(api_key)
    for e in endpoints:
        if e.get("name") == SADTALKER_ENDPOINT_NAME:
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
    """Create a serverless endpoint for SadTalker."""
    if verbose:
        print(f"Creating endpoint '{SADTALKER_ENDPOINT_NAME}'...")

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
            "name": SADTALKER_ENDPOINT_NAME,
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
        if line.startswith("RUNPOD_SADTALKER_ENDPOINT_ID="):
            new_lines.append(f"RUNPOD_SADTALKER_ENDPOINT_ID={endpoint_id}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"RUNPOD_SADTALKER_ENDPOINT_ID={endpoint_id}")

    env_path.write_text("\n".join(new_lines))

    if verbose:
        print(f"  Saved: RUNPOD_SADTALKER_ENDPOINT_ID={endpoint_id}")

    return True


def setup_runpod(gpu_id: str = "AMPERE_24", verbose: bool = True) -> dict:
    """Set up RunPod endpoint for SadTalker."""
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
        print("RunPod Setup (SadTalker Talking Head Generator)")
        print("=" * 60)
        print(f"Docker Image: {SADTALKER_DOCKER_IMAGE}")
        print(f"GPU Type: {gpu_id}")
        print()

    try:
        if verbose:
            print("[1/3] Checking for existing template...")

        template = find_sadtalker_template(api_key)
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

        endpoint = find_sadtalker_endpoint(api_key, result["template_id"])
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
            print("  python tools/sadtalker.py --image avatar.png --audio voice.mp3 --output talking.mp4")
            print()

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate talking head videos using SadTalker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --output talking.mp4

  # Less head movement (good for serious content)
  python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --still --output talking.mp4

  # Higher resolution (512px)
  python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --size 512 --output talking.mp4

  # No face enhancement (faster)
  python tools/sadtalker.py --image avatar.png --audio voiceover.mp3 --no-enhance --output talking.mp4

  # Setup RunPod endpoint (first-time)
  python tools/sadtalker.py --setup
        """,
    )

    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Input portrait image (face should be centered, 30-70%% of frame)",
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        help="Input audio file (speech to animate)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output video file path",
    )
    parser.add_argument(
        "--still",
        action="store_true",
        help="Less head movement (good for serious/professional content)",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip GFPGAN face enhancement (faster but lower quality)",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="crop",
        choices=["crop", "resize", "full"],
        help="Face preprocessing: crop (default), resize, or full",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        choices=[256, 512],
        help="Output resolution: 256 (default) or 512",
    )
    parser.add_argument(
        "--expression-scale",
        type=float,
        default=1.0,
        help="Expression intensity (default: 1.0)",
    )
    parser.add_argument(
        "--pose-style",
        type=int,
        default=0,
        help="Pose variation 0-45 (default: 0, try 45 for natural movement)",
    )

    # Presets for common use cases
    parser.add_argument(
        "--preset",
        type=str,
        choices=["default", "natural", "expressive", "professional", "fullbody"],
        help="Use a preset configuration (overrides other settings)",
    )

    # RunPod options
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="RunPod job timeout in seconds (default: 600)",
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
    if not args.image:
        print("Error: --image is required", file=sys.stderr)
        sys.exit(1)
    if not args.audio:
        print("Error: --audio is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    # Check input files exist
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.audio).exists():
        print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    # Apply presets
    presets = {
        "default": {},  # Use CLI defaults
        "natural": {"pose_style": 45, "expression_scale": 1.0},
        "expressive": {"pose_style": 45, "expression_scale": 1.3},
        "professional": {"still": True, "expression_scale": 0.8},
        "fullbody": {"still": True, "preprocess": "full"},
    }

    # Start with CLI args
    still_mode = args.still
    enhancer = "none" if args.no_enhance else "gfpgan"
    preprocess = args.preprocess
    size = args.size
    expression_scale = args.expression_scale
    pose_style = args.pose_style

    # Override with preset if specified
    if args.preset and args.preset in presets:
        preset = presets[args.preset]
        if "still" in preset:
            still_mode = preset["still"]
        if "expression_scale" in preset:
            expression_scale = preset["expression_scale"]
        if "pose_style" in preset:
            pose_style = preset["pose_style"]
        if "preprocess" in preset:
            preprocess = preset["preprocess"]
        if verbose:
            print(f"Using preset '{args.preset}': {preset}")

    if verbose:
        print("Generating talking head video with SadTalker...")

    result = process_with_runpod(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        still_mode=still_mode,
        enhancer=enhancer,
        preprocess=preprocess,
        size=size,
        expression_scale=expression_scale,
        pose_style=pose_style,
        timeout=args.timeout,
        verbose=verbose,
    )

    if result.get("error"):
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        duration = result.get("duration_seconds", 0)
        chunks = result.get("chunks_processed", 1)
        print(f"Generated: {result['output']}")
        print(f"  Duration: {duration:.1f}s ({chunks} chunk{'s' if chunks > 1 else ''})")
        print(f"  Processing time: {result.get('processing_time_seconds', 0):.1f}s")


if __name__ == "__main__":
    main()
