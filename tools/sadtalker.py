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
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from file_transfer import (
    upload_to_storage, download_from_r2, delete_from_r2,
    download_from_url, get_r2_payload_config,
)

# Docker image for RunPod endpoint
SADTALKER_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-sadtalker:latest"
SADTALKER_TEMPLATE_NAME = "video-toolkit-sadtalker"
SADTALKER_ENDPOINT_NAME = "video-toolkit-sadtalker"

# Processing time estimate: ~4 minutes per minute of audio + buffer
PROCESSING_TIME_MULTIPLIER = 20  # generous: size=512 + gfpgan on A10G can be slow
PROCESSING_TIME_BUFFER = 360  # 6 min buffer for cold start, upload, model loading


def get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def calculate_timeout(audio_duration: float) -> int:
    """Calculate appropriate timeout based on audio duration.

    ~4 minutes processing per minute of audio, plus buffer for cold start.
    """
    processing_time = audio_duration * PROCESSING_TIME_MULTIPLIER
    return int(processing_time + PROCESSING_TIME_BUFFER)


def retrieve_job_result(
    job_id: str,
    output_path: str,
    verbose: bool = True,
) -> dict:
    """Retrieve results from a completed RunPod job.

    Use this if a previous run timed out but the job completed on the server.
    RunPod-specific (job retrieval is a RunPod concept).
    """
    from cloud_gpu import get_provider_config

    config = get_provider_config("runpod", "sadtalker")
    api_key = config.get("api_key")
    endpoint_id = config.get("endpoint_id")

    if not api_key:
        return {"error": "RUNPOD_API_KEY not set"}
    if not endpoint_id:
        return {"error": "RUNPOD_SADTALKER_ENDPOINT_ID not set"}

    if verbose:
        print(f"Retrieving job: {job_id}", file=sys.stderr)

    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

        if response.status_code != 200:
            return {"error": f"Failed to get job status: HTTP {response.status_code}"}

        data = response.json()
        status = data.get("status")

        if verbose:
            print(f"  Status: {status}", file=sys.stderr)

        if status != "COMPLETED":
            return {"error": f"Job not completed. Status: {status}"}

        output = data.get("output", {})

        if output.get("error"):
            return {"error": output["error"]}

        video_url = output.get("video_url") or output.get("r2_url")
        if not video_url:
            return {"error": "No video_url in job output. Job may have failed."}

        if verbose:
            print(f"  Downloading from: {video_url[:80]}...", file=sys.stderr)

        if not download_from_url(video_url, output_path, verbose=verbose):
            return {"error": "Failed to download video"}

        return {
            "output": output_path,
            "job_id": job_id,
            "duration_seconds": output.get("duration_seconds"),
            "chunks_processed": output.get("chunks_processed"),
        }

    except Exception as e:
        return {"error": f"Failed to retrieve job: {e}"}


def process_with_cloud(
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
    cloud: str = "runpod",
    progress=None,
) -> dict:
    """Process image+audio using cloud GPU endpoint."""
    r2_keys_to_cleanup = []

    if verbose:
        print(f"Cloud provider: {cloud}", file=sys.stderr)

    # Auto-calculate timeout if not specified
    audio_duration = get_audio_duration(audio_path)
    if timeout <= 0:
        if audio_duration:
            timeout = calculate_timeout(audio_duration)
            if verbose:
                print(f"Audio duration: {audio_duration:.1f}s, timeout: {timeout}s", file=sys.stderr)
        else:
            timeout = 900  # Default 15 minutes
            if verbose:
                print(f"Could not determine audio duration, using default timeout: {timeout}s", file=sys.stderr)

    # Pre-flight estimate
    if verbose and audio_duration:
        chunk_duration = 45  # matches CHUNK_DURATION in Modal app
        num_chunks = max(1, int(audio_duration / chunk_duration) + (1 if audio_duration % chunk_duration > 0 else 0))
        est_minutes = (audio_duration * PROCESSING_TIME_MULTIPLIER) / 60
        # A10G cost: ~$0.000362/sec
        est_cost = (audio_duration * PROCESSING_TIME_MULTIPLIER + PROCESSING_TIME_BUFFER) * 0.000362
        print(f"Estimate: {num_chunks} chunk{'s' if num_chunks > 1 else ''}, "
              f"~{est_minutes:.0f} min processing, ~${est_cost:.2f} GPU cost",
              file=sys.stderr)

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

    # Build payload
    if verbose:
        print(f"Submitting job (size={size}, enhancer={enhancer})...", file=sys.stderr)

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

    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload
    else:
        print("Warning: R2 not configured. Video will be returned as base64.", file=sys.stderr)

    # Call cloud GPU endpoint
    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="sadtalker",
        timeout=timeout,
        progress_label="Generating talking head",
        verbose=verbose,
        progress=progress,
    )

    if isinstance(result, dict) and result.get("error"):
        return {"error": result["error"]}

    # Download result
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    output_r2_key = result.get("r2_key") if isinstance(result, dict) else None
    output_url = result.get("video_url") if isinstance(result, dict) else None

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
        # Try base64 fallback
        video_base64 = result.get("video_base64")
        if video_base64:
            Path(output_path).write_bytes(base64.b64decode(video_base64))
            downloaded = True
            if verbose:
                size_kb = Path(output_path).stat().st_size // 1024
                print(f"  Decoded from base64: {output_path} ({size_kb}KB)", file=sys.stderr)

    if not downloaded:
        return {"error": f"No video in result: {list(result.keys()) if isinstance(result, dict) else result}"}

    # Cleanup R2 objects
    for key in r2_keys_to_cleanup:
        delete_from_r2(key)

    return {
        "success": True,
        "output": output_path,
        "processing_time_seconds": round(elapsed, 2),
        "duration_seconds": result.get("duration_seconds"),
        "chunks_processed": result.get("chunks_processed"),
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

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("RUNPOD_API_KEY")

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

    # Cloud GPU options
    parser.add_argument(
        "--cloud",
        type=str,
        default="modal",
        choices=["runpod", "modal"],
        help="Cloud GPU provider (default: modal)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Job timeout in seconds (default: auto-calculated from audio duration)",
    )
    parser.add_argument(
        "--retrieve",
        type=str,
        metavar="JOB_ID",
        help="Retrieve results from a completed RunPod job (e.g., if previous run timed out)",
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

    # Handle --setup
    if args.setup:
        result = setup_runpod(gpu_id=args.setup_gpu, verbose=verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)
        sys.exit(0)

    # Handle --retrieve (download results from a previous job)
    if args.retrieve:
        if not args.output:
            print("Error: --output is required with --retrieve", file=sys.stderr)
            sys.exit(1)
        result = retrieve_job_result(args.retrieve, args.output, verbose=verbose)
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

    result = process_with_cloud(
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
        cloud=args.cloud,
        progress=reporter,
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
