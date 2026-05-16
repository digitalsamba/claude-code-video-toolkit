#!/usr/bin/env python3
"""
Upscale images using AI (Real-ESRGAN).

Cloud providers: RunPod (default), Modal.

Usage:
    # Cloud processing (RunPod, default)
    python tools/upscale.py --input image.jpg --output upscaled.png --cloud runpod

    # Using Modal
    python tools/upscale.py --input image.jpg --output upscaled.png --cloud modal

    # Specify model and scale
    python tools/upscale.py --input image.jpg --output upscaled.png --model anime --scale 4 --cloud runpod

    # With face enhancement
    python tools/upscale.py --input image.jpg --output upscaled.png --face-enhance --cloud runpod

    # Legacy flag (deprecated, use --cloud runpod)
    python tools/upscale.py --input image.jpg --output upscaled.png --runpod

Models:
    - general: RealESRGAN_x4plus (default, good for most images)
    - anime: RealESRGAN_x4plus_anime_6B (optimized for anime/illustration)
    - photo: realesr-general-x4v3 (alternative general model)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from file_transfer import (
    upload_to_storage, download_from_r2, delete_from_r2,
    download_from_url, get_r2_payload_config,
)

# Docker image for RunPod endpoint
REALESRGAN_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-realesrgan:v2"
REALESRGAN_TEMPLATE_NAME = "video-toolkit-realesrgan-v2"
REALESRGAN_ENDPOINT_NAME = "video-toolkit-upscale"


def process_with_cloud(
    input_path: str,
    output_path: str,
    scale: int = 4,
    model: str = "general",
    face_enhance: bool = False,
    output_format: str = "png",
    timeout: int = 300,
    verbose: bool = True,
    cloud: str = "runpod",
    progress=None,
) -> dict:
    """Process image using cloud GPU endpoint."""
    r2_keys_to_cleanup = []

    if verbose:
        print(f"Cloud provider: {cloud}", file=sys.stderr)

    # Upload image
    image_url, image_r2_key = upload_to_storage(input_path, "upscale/input")
    if not image_url:
        return {"error": "Failed to upload image"}
    if image_r2_key:
        r2_keys_to_cleanup.append(image_r2_key)

    # Build payload
    if verbose:
        print(f"Submitting job (scale={scale}, model={model})...", file=sys.stderr)

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

    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload

    # Call cloud GPU endpoint
    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="upscale",
        timeout=timeout,
        progress_label="Upscaling image",
        verbose=verbose,
        progress=progress,
    )

    if isinstance(result, dict) and result.get("error"):
        return {"error": result["error"]}

    # Download result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    output_r2_key = result.get("r2_key") if isinstance(result, dict) else None
    output_url = result.get("output_url") if isinstance(result, dict) else None

    if output_r2_key:
        if verbose:
            print(f"Downloading result from R2...", file=sys.stderr)
        downloaded = download_from_r2(output_r2_key, output_path)
        if downloaded:
            r2_keys_to_cleanup.append(output_r2_key)
            if verbose:
                size_kb = Path(output_path).stat().st_size // 1024
                print(f"  Downloaded: {output_path} ({size_kb}KB)", file=sys.stderr)

    if not downloaded and output_url:
        downloaded = download_from_url(output_url, output_path, verbose=verbose)

    if not downloaded:
        return {"error": f"No output_url or r2_key in result: {result}"}

    # Cleanup R2 objects
    for key in r2_keys_to_cleanup:
        delete_from_r2(key)

    return {
        "success": True,
        "output": output_path,
        "processing_time_seconds": round(elapsed, 2),
        "cloud_output": result,
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

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("RUNPOD_API_KEY")

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
  # Upscale image 4x using cloud GPU
  python tools/upscale.py --input photo.jpg --output photo_4x.png --cloud runpod

  # Use anime model for illustrations
  python tools/upscale.py --input art.png --output art_4x.png --model anime --cloud runpod

  # With face enhancement
  python tools/upscale.py --input portrait.jpg --output portrait_4x.png --face-enhance --cloud runpod

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

    # Cloud GPU options
    parser.add_argument(
        "--cloud",
        type=str,
        default=None,
        choices=["runpod", "modal"],
        help="Cloud GPU provider (default: runpod)",
    )
    parser.add_argument(
        "--runpod",
        action="store_true",
        help="[Deprecated] Use --cloud runpod instead",
    )
    parser.add_argument(
        "--timeout", "--runpod-timeout",
        type=int,
        default=300,
        dest="timeout",
        help="Job timeout in seconds (default: 300)",
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
    parser.add_argument(
        "--progress",
        choices=["human", "json"],
        default="human",
        help="Progress output mode: human (colored stderr, default) "
             "or json (JSON Lines to stderr for bots/agents)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.json

    from cloud_gpu import ProgressReporter
    reporter = ProgressReporter(mode=args.progress)

    # Handle deprecated --runpod flag
    if args.runpod:
        print("Note: --runpod is deprecated, use --cloud runpod instead", file=sys.stderr)
        if not args.cloud:
            args.cloud = "runpod"

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
        result = {
            "dry_run": True,
            "input": args.input,
            "output": args.output,
            "scale": args.scale,
            "model": args.model,
            "face_enhance": args.face_enhance,
            "output_format": args.format,
            "cloud": args.cloud,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would process:")
            for k, v in result.items():
                print(f"  {k}: {v}")
        return

    # Cloud processing
    if args.cloud:
        result = process_with_cloud(
            input_path=args.input,
            output_path=args.output,
            scale=args.scale,
            model=args.model,
            face_enhance=args.face_enhance,
            output_format=args.format,
            timeout=args.timeout,
            verbose=verbose,
            cloud=args.cloud,
            progress=reporter,
        )

        if result.get("error"):
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            output_info = result.get("cloud_output", {})
            input_dims = output_info.get("input_dimensions", "?")
            output_dims = output_info.get("output_dimensions", "?")
            print(f"Upscaled: {result['output']}")
            print(f"  {input_dims} -> {output_dims}")
            print(f"  Processing time: {result.get('processing_time_seconds', 0):.1f}s")

        return

    # No cloud provider specified
    print("Error: Specify --cloud runpod or --cloud modal for cloud processing.", file=sys.stderr)
    print("       Or run --setup first to configure a RunPod endpoint.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
