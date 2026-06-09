#!/usr/bin/env python3
"""
Generate speech using Qwen3-TTS (via cloud GPU).

Supports built-in speakers with emotion control and voice cloning from reference audio.
Cloud providers: RunPod (default), Modal.

Usage:
    # Built-in speaker (RunPod, default)
    python tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3

    # Using Modal instead of RunPod
    python tools/qwen3_tts.py --text "Hello world" --cloud modal --output hello.mp3

    # With tone preset
    python tools/qwen3_tts.py --text "Hello world" --tone warm --output hello.mp3

    # With custom emotion/style instruction (overrides --tone)
    python tools/qwen3_tts.py --text "I'm so excited!" --instruct "Speak enthusiastically" --output excited.mp3

    # List tone presets
    python tools/qwen3_tts.py --list-tones

    # Voice cloning
    python tools/qwen3_tts.py --text "Hello" --ref-audio sample.wav --ref-text "transcript" --output cloned.mp3

    # List built-in voices
    python tools/qwen3_tts.py --list-voices

    # Setup endpoint (RunPod)
    python tools/qwen3_tts.py --setup

    # Setup endpoint (Modal)
    python tools/qwen3_tts.py --setup --cloud modal

Setup:
    RunPod:
        1. Create account at runpod.io
        2. Run: python tools/qwen3_tts.py --setup
        3. Or manually deploy docker/runpod-qwen3-tts/ and add endpoint ID to .env

    Modal:
        1. pip install modal && python3 -m modal setup
        2. Run: python tools/qwen3_tts.py --setup --cloud modal
        3. Or manually: modal deploy docker/modal-qwen3-tts/app.py
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))

# Docker image for RunPod endpoint
QWEN3_TTS_DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-qwen3-tts:latest"
QWEN3_TTS_TEMPLATE_NAME = "video-toolkit-qwen3-tts"
QWEN3_TTS_ENDPOINT_NAME = "video-toolkit-qwen3-tts"

# Built-in speakers
BUILTIN_SPEAKERS = {
    "Ryan": "English",
    "Aiden": "English",
    "Vivian": "Chinese",
    "Serena": "Chinese",
    "Uncle_Fu": "Chinese",
    "Dylan": "Chinese",
    "Eric": "Chinese",
    "Ono_Anna": "Japanese",
    "Sohee": "Korean",
}

SUPPORTED_LANGUAGES = [
    "Auto", "English", "Chinese", "French", "German",
    "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
]

# Named instruct presets for common voice tones
INSTRUCT_PRESETS = {
    "neutral": "",
    "warm": "Speak warmly and naturally, like talking to a friend.",
    "professional": "Speak in a clear, professional tone with measured pacing.",
    "excited": "Speak with enthusiasm and energy, conveying excitement.",
    "calm": "Speak slowly and calmly, with a soothing, relaxed delivery.",
    "serious": "Speak in a serious, authoritative tone with gravitas.",
    "storyteller": "Speak like a narrator telling a captivating story, with varied pacing and emphasis.",
    "tutorial": "Speak clearly and patiently, like explaining something step by step.",
}


def resolve_tone(tone: str | None, instruct: str) -> str:
    """Resolve a tone preset name and/or raw instruct text into final instruct string.

    Priority: explicit instruct > tone preset > empty string.
    Unknown tone names are treated as raw instruct text.
    """
    if instruct:
        return instruct
    if tone:
        if tone in INSTRUCT_PRESETS:
            return INSTRUCT_PRESETS[tone]
        return tone  # treat unknown tone as raw instruct text
    return ""


from file_transfer import (
    upload_to_storage, download_from_r2, delete_from_r2,
    download_from_url, get_r2_payload_config,
)


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (FileNotFoundError, ValueError):
        pass  # ffprobe not installed or invalid output
    return None



def generate_audio(
    text,                       # str OR list[str] for batch
    output_path,                # str OR list[str] (same shape as text)
    speaker: str = "Ryan",
    language="Auto",            # str OR list[str] (broadcasts if scalar)
    instruct: str = "",
    ref_audio: str | None = None,
    ref_text: str | None = None,
    design_instruct: str | None = None,   # if set → mode=voice_design
    output_format: str = "mp3",
    timeout: int = 300,
    verbose: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    cloud: str = "modal",
    progress=None,
    max_wpm: float | None = None,
) -> dict:
    """Generate audio using Qwen3-TTS via cloud GPU.

    Supports three modes:
      - custom_voice (default): built-in speaker + instruct
      - clone: set ref_audio + ref_text
      - voice_design: set design_instruct (brief) + text (seed sentence)

    Batch mode: pass `text` as a list of strings and `output_path` as a
    matching list. The handler reuses a single voice_clone_prompt across
    all utterances for consistent voice (proper Qwen3-TTS pattern).

    Pacing QC: every result includes `wpm` (words per minute) and a `pacing`
    label ('fast'/'slow'/'ok'/None). If `max_wpm` is set, takes that exceed
    it are slowed in place with pitch-preserving ffmpeg atempo (floor 0.85x)
    — cloned voices inherit pace from their reference and resist temperature,
    so deterministic correction beats regenerating. See tools/pacing.py.

    Returns dict with:
      single call: {success, output, duration_seconds, duration_frames_30fps, wpm, pacing}
      batch call:  {success, outputs: [{output, duration_seconds, ...}, ...]}
    """
    start_time = time.time()
    r2_keys_to_cleanup = []

    # Normalize inputs
    is_batch = isinstance(text, list)
    texts = text if is_batch else [text]
    out_paths = output_path if is_batch else [output_path]
    if len(out_paths) != len(texts):
        return {"success": False, "error": "text and output_path must have matching lengths"}
    if any(not t for t in texts):
        return {"success": False, "error": "all texts must be non-empty strings"}

    # Get R2 config (used for file transfer regardless of cloud provider)
    r2_payload = get_r2_payload_config()

    # Determine mode
    if design_instruct is not None:
        if is_batch:
            return {"success": False, "error": "voice_design mode expects a single seed text, not a batch"}
        mode = "voice_design"
    elif ref_audio:
        mode = "clone"
    else:
        mode = "custom_voice"

    # Upload reference audio for clone mode
    ref_audio_url = None
    if mode == "clone":
        if not Path(ref_audio).exists():
            return {"success": False, "error": f"Reference audio not found: {ref_audio}"}
        if not ref_text:
            return {"success": False, "error": "ref_text is required for voice cloning"}

        ref_audio_url, ref_r2_key = upload_to_storage(ref_audio, "qwen3-tts/input")
        if not ref_audio_url:
            return {"success": False, "error": "Failed to upload reference audio"}
        if ref_r2_key:
            r2_keys_to_cleanup.append(ref_r2_key)

    if verbose:
        print(f"Cloud provider: {cloud}", file=sys.stderr)
        if mode == "clone":
            print(f"Mode: voice clone ({len(texts)} utterance{'s' if len(texts) > 1 else ''}, shared prompt)", file=sys.stderr)
        elif mode == "voice_design":
            print(f"Mode: voice design (instruct={design_instruct[:60]!r}...)", file=sys.stderr)
        else:
            print(f"Speaker: {speaker}, Language: {language}, mode: custom_voice", file=sys.stderr)

    # Build payload (same format for both providers)
    payload = {
        "input": {
            "text": texts if is_batch else texts[0],
            "mode": mode,
            "language": language,
            "output_format": output_format,
        }
    }

    if mode == "clone":
        payload["input"]["ref_audio_url"] = ref_audio_url
        payload["input"]["ref_text"] = ref_text
    elif mode == "voice_design":
        payload["input"]["instruct"] = design_instruct
    else:
        payload["input"]["speaker"] = speaker
        if instruct:
            payload["input"]["instruct"] = instruct

    if temperature is not None:
        payload["input"]["temperature"] = temperature
    if top_p is not None:
        payload["input"]["top_p"] = top_p

    if r2_payload:
        payload["input"]["r2"] = r2_payload

    # Submit job via cloud_gpu abstraction
    try:
        from cloud_gpu import call_cloud_endpoint
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from cloud_gpu import call_cloud_endpoint

    progress_label = (
        f"Designing voice" if mode == "voice_design"
        else (f"Generating {len(texts)} utterances" if is_batch else "Generating speech")
    )
    # Batch calls can be longer-running; scale timeout by count
    effective_timeout = timeout * max(1, len(texts) // 3 + 1)

    output, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="qwen3_tts",
        timeout=effective_timeout,
        poll_interval=3,
        progress_label=progress_label,
        verbose=verbose,
        progress=progress,
    )

    if isinstance(output, dict) and output.get("error"):
        return {"success": False, "error": output["error"]}

    # Parse response — handler returns `outputs: [...]` always; legacy top-level
    # keys also present when input was single-text. Use `outputs` uniformly.
    outputs_list = None
    if isinstance(output, dict):
        if isinstance(output.get("outputs"), list):
            outputs_list = output["outputs"]
        else:
            # Pre-v0.2 handler or single-shot legacy response — synthesize a
            # one-item list from top-level keys.
            legacy_item = {
                k: output[k] for k in ("audio_url", "r2_key", "audio_base64", "duration_seconds")
                if k in output
            }
            if legacy_item:
                outputs_list = [legacy_item]

    if not outputs_list or len(outputs_list) != len(texts):
        return {
            "success": False,
            "error": f"Expected {len(texts)} output(s), got {len(outputs_list) if outputs_list else 0}: "
                     f"{list(output.keys()) if isinstance(output, dict) else output}",
        }

    # Download all outputs
    downloaded_items = []
    for i, (item, dest_path) in enumerate(zip(outputs_list, out_paths)):
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        downloaded = False

        r2_key = item.get("r2_key")
        url = item.get("audio_url")

        if r2_key:
            if verbose:
                print(f"Downloading output {i+1}/{len(outputs_list)} from R2...", file=sys.stderr)
            downloaded = download_from_r2(r2_key, dest_path)
            if downloaded:
                r2_keys_to_cleanup.append(r2_key)
                if verbose:
                    size_kb = Path(dest_path).stat().st_size // 1024
                    print(f"  Downloaded: {dest_path} ({size_kb}KB)", file=sys.stderr)

        if not downloaded and url:
            downloaded = download_from_url(url, dest_path, verbose=verbose)

        if not downloaded:
            audio_base64 = item.get("audio_base64")
            if audio_base64:
                Path(dest_path).write_bytes(base64.b64decode(audio_base64))
                downloaded = True
                if verbose:
                    size_kb = Path(dest_path).stat().st_size // 1024
                    print(f"  Decoded from base64: {dest_path} ({size_kb}KB)", file=sys.stderr)

        if not downloaded:
            # Cleanup before returning
            for key in r2_keys_to_cleanup:
                delete_from_r2(key)
            return {"success": False, "error": f"No audio in output[{i}]: {list(item.keys())}"}

        duration = get_audio_duration(dest_path)
        item_result = {
            "output": dest_path,
            "duration_seconds": round(duration, 2) if duration else None,
            "duration_frames_30fps": int(duration * 30) if duration else None,
            "script_chars": len(texts[i]),
        }

        # Pacing QC (voice_design seed sentences are auditions, not narration)
        if mode != "voice_design":
            from pacing import clamp_pace, pace_label
            if max_wpm:
                clamp = clamp_pace(dest_path, texts[i], max_wpm, verbose=verbose)
                if clamp.get("applied"):
                    new_dur = clamp["duration_seconds"]
                    item_result["duration_seconds"] = new_dur
                    item_result["duration_frames_30fps"] = (
                        int(new_dur * 30) if new_dur else None
                    )
                    item_result["pace_adjusted"] = {
                        "original_wpm": clamp["original_wpm"],
                        "atempo": clamp["atempo"],
                    }
                elif clamp.get("error") and verbose:
                    print(f"  Pace clamp skipped: {clamp['error']}", file=sys.stderr)
            wpm, label = pace_label(texts[i], item_result["duration_seconds"])
            item_result["wpm"] = wpm
            item_result["pacing"] = label
            if verbose and label in ("fast", "slow"):
                print(
                    f"  Pacing warning: {Path(dest_path).name} is {wpm:.0f} wpm "
                    f"({label}; comfortable narration is 140-160). "
                    "Consider --max-wpm to auto-correct.",
                    file=sys.stderr,
                )

        downloaded_items.append(item_result)

    # Cleanup R2 objects
    for key in r2_keys_to_cleanup:
        delete_from_r2(key)

    if is_batch:
        return {
            "success": True,
            "outputs": downloaded_items,
            "mode": mode,
        }
    else:
        return {
            "success": True,
            **downloaded_items[0],
            "mode": mode,
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


def find_template(api_key: str) -> dict | None:
    """Find existing Qwen3-TTS template."""
    templates = list_runpod_templates(api_key)
    for t in templates:
        if t.get("name") == QWEN3_TTS_TEMPLATE_NAME:
            return t
        if t.get("imageName") == QWEN3_TTS_DOCKER_IMAGE:
            return t
    return None


def create_runpod_template(api_key: str, verbose: bool = True) -> dict:
    """Create a serverless template for Qwen3-TTS."""
    if verbose:
        print(f"Creating template '{QWEN3_TTS_TEMPLATE_NAME}'...")

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
            "name": QWEN3_TTS_TEMPLATE_NAME,
            "imageName": QWEN3_TTS_DOCKER_IMAGE,
            "isServerless": True,
            "containerDiskInGb": 30,
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


def find_endpoint(api_key: str, template_id: str) -> dict | None:
    """Find existing Qwen3-TTS endpoint."""
    endpoints = list_runpod_endpoints(api_key)
    for e in endpoints:
        if e.get("name") == QWEN3_TTS_ENDPOINT_NAME:
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
    """Create a serverless endpoint for Qwen3-TTS."""
    if verbose:
        print(f"Creating endpoint '{QWEN3_TTS_ENDPOINT_NAME}'...")

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
            "name": QWEN3_TTS_ENDPOINT_NAME,
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
        if line.startswith("RUNPOD_QWEN3_TTS_ENDPOINT_ID="):
            new_lines.append(f"RUNPOD_QWEN3_TTS_ENDPOINT_ID={endpoint_id}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"RUNPOD_QWEN3_TTS_ENDPOINT_ID={endpoint_id}")

    env_path.write_text("\n".join(new_lines))

    if verbose:
        print(f"  Saved: RUNPOD_QWEN3_TTS_ENDPOINT_ID={endpoint_id}")

    return True


def setup_runpod(gpu_id: str = "AMPERE_24", verbose: bool = True) -> dict:
    """Set up RunPod endpoint for Qwen3-TTS."""
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
        print("RunPod Setup (Qwen3-TTS Speech Generation)")
        print("=" * 60)
        print(f"Docker Image: {QWEN3_TTS_DOCKER_IMAGE}")
        print(f"GPU Type: {gpu_id}")
        print()

    try:
        if verbose:
            print("[1/3] Checking for existing template...")

        template = find_template(api_key)
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

        endpoint = find_endpoint(api_key, result["template_id"])
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
            print('  python tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3')
            print()

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


# =============================================================================
# Modal Setup
# =============================================================================

def setup_modal(verbose: bool = True) -> dict:
    """Set up Modal endpoint for Qwen3-TTS.

    Deploys the Modal app and saves the endpoint URL to .env.
    Requires: pip install modal && python3 -m modal setup
    """
    import shutil
    import subprocess

    result = {
        "success": False,
        "endpoint_url": None,
    }

    if verbose:
        print("=" * 60)
        print("Modal Setup (Qwen3-TTS Speech Generation)")
        print("=" * 60)
        print()

    # Check modal CLI is installed
    if not shutil.which("modal"):
        result["error"] = (
            "Modal CLI not found. Install with:\n"
            "  pip install modal\n"
            "  python3 -m modal setup"
        )
        if verbose:
            print(f"Error: {result['error']}", file=sys.stderr)
        return result

    # Find the app file
    app_file = Path(__file__).parent.parent / "docker" / "modal-qwen3-tts" / "app.py"
    if not app_file.exists():
        result["error"] = f"Modal app file not found: {app_file}"
        if verbose:
            print(f"Error: {result['error']}", file=sys.stderr)
        return result

    if verbose:
        print(f"[1/3] Deploying Modal app: {app_file}")
        print("  This will build the container image and create the web endpoint.")
        print("  First deploy may take 5-10 minutes (downloading model weights)...")
        print()

    try:
        deploy_result = subprocess.run(
            ["modal", "deploy", str(app_file)],
            capture_output=True,
            text=True,
            timeout=900,  # 15 min for first deploy with model download
        )

        if deploy_result.returncode != 0:
            result["error"] = f"Modal deploy failed:\n{deploy_result.stderr}"
            if verbose:
                print(f"Error: {result['error']}", file=sys.stderr)
            return result

        if verbose:
            print(deploy_result.stdout)

        # Parse endpoint URL from deploy output
        # Modal prints lines like: Created web endpoint ... => https://workspace--app-name-fn.modal.run
        endpoint_url = None
        for line in deploy_result.stdout.splitlines():
            if "modal.run" in line:
                # Extract URL from the line
                import re
                urls = re.findall(r'https://[^\s"\']+modal\.run[^\s"\']*', line)
                if urls:
                    endpoint_url = urls[0]
                    break

        if not endpoint_url:
            # Try stderr too (some modal versions output there)
            for line in deploy_result.stderr.splitlines():
                if "modal.run" in line:
                    import re
                    urls = re.findall(r'https://[^\s"\']+modal\.run[^\s"\']*', line)
                    if urls:
                        endpoint_url = urls[0]
                        break

        if not endpoint_url:
            result["error"] = (
                "Deploy succeeded but could not parse endpoint URL from output.\n"
                "Check `modal app list` for the URL and add to .env manually:\n"
                "  MODAL_QWEN3_TTS_ENDPOINT_URL=https://your-workspace--video-toolkit-qwen3-tts-qwen3tts-generate.modal.run"
            )
            if verbose:
                print(f"Warning: {result['error']}", file=sys.stderr)
            return result

        result["endpoint_url"] = endpoint_url

        if verbose:
            print(f"[2/3] Endpoint URL: {endpoint_url}")
            print("[3/3] Saving configuration...")

        # Save to .env
        env_path = Path(__file__).parent.parent / ".env"
        env_var = "MODAL_QWEN3_TTS_ENDPOINT_URL"

        if env_path.exists():
            env_content = env_path.read_text()
            if env_var in env_content:
                # Update existing
                lines = env_content.splitlines()
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{env_var}="):
                        lines[i] = f"{env_var}={endpoint_url}"
                        updated = True
                        break
                if updated:
                    env_path.write_text("\n".join(lines) + "\n")
            else:
                with open(env_path, "a") as f:
                    f.write(f"\n{env_var}={endpoint_url}\n")
        else:
            env_path.write_text(f"{env_var}={endpoint_url}\n")

        if verbose:
            print(f"  Saved: {env_var}={endpoint_url}")

        result["success"] = True

        if verbose:
            print()
            print("=" * 60)
            print("Setup Complete!")
            print("=" * 60)
            print(f"Endpoint: {endpoint_url}")
            print()
            print("You can now run:")
            print('  python tools/qwen3_tts.py --text "Hello world" --cloud modal --output hello.mp3')
            print()

    except subprocess.TimeoutExpired:
        result["error"] = "Modal deploy timed out after 15 minutes"
        if verbose:
            print(f"Error: {result['error']}", file=sys.stderr)
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using Qwen3-TTS (via cloud GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Built-in speaker (RunPod, default)
  python tools/qwen3_tts.py --text "Hello world" --speaker Ryan --output hello.mp3

  # Using Modal instead
  python tools/qwen3_tts.py --text "Hello world" --cloud modal --output hello.mp3

  # With emotion control
  python tools/qwen3_tts.py --text "Great news!" --instruct "Speak enthusiastically" --output excited.mp3

  # Voice cloning
  python tools/qwen3_tts.py --text "Hello" --ref-audio sample.wav --ref-text "transcript" --output cloned.mp3

  # List voices
  python tools/qwen3_tts.py --list-voices

  # Setup endpoint (RunPod)
  python tools/qwen3_tts.py --setup

  # Setup endpoint (Modal)
  python tools/qwen3_tts.py --setup --cloud modal
        """,
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output audio file path (.mp3 or .wav)",
    )
    parser.add_argument(
        "--speaker", "-s",
        type=str,
        default="Ryan",
        help="Built-in speaker name (default: Ryan). Use --list-voices to see options.",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="Auto",
        choices=[l.lower() for l in SUPPORTED_LANGUAGES],
        help="Language hint (default: auto)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default="",
        help="Natural-language emotion/style instruction (e.g., 'Speak warmly'). Overrides --tone.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        help="Named tone preset (e.g., 'warm', 'professional'). Use --list-tones to see options.",
    )

    # Voice cloning
    parser.add_argument(
        "--ref-audio",
        type=str,
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        help="Transcript of reference audio (required with --ref-audio)",
    )

    # Voice design — generate a new character voice from a natural-language brief.
    # Produces a designed .wav that can be re-used as a clone reference.
    parser.add_argument(
        "--design-instruct",
        type=str,
        help="Character brief for voice design mode (e.g. 'Unhurried Irish "
             "craftsman, late 50s, baritone'). With --text as seed sentence, "
             "writes a designed-voice WAV to --output.",
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        default="mp3",
        choices=["mp3", "wav"],
        help="Output format (default: mp3)",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        help="Expressiveness (default: model default ~0.7, range: 0.3-1.5)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling (default: model default ~0.8, range: 0.1-1.0)",
    )

    # Cloud GPU options
    parser.add_argument(
        "--max-wpm",
        type=float,
        default=None,
        help="Pace clamp: if the take exceeds this words-per-minute, slow it "
             "with pitch-preserving atempo (floor 0.85x). Try 165. Cloned "
             "voices inherit pace from the reference and resist temperature.",
    )
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
        default=300,
        help="Job timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up cloud endpoint automatically",
    )
    parser.add_argument(
        "--setup-gpu",
        type=str,
        default="AMPERE_24",
        choices=["AMPERE_16", "AMPERE_24", "ADA_24", "AMPERE_48"],
        help="GPU type for RunPod endpoint (default: AMPERE_24)",
    )

    # Utility
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List built-in speakers and exit",
    )
    parser.add_argument(
        "--list-tones",
        action="store_true",
        help="List available tone presets and exit",
    )
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

    # Handle --list-voices
    if args.list_voices:
        print("Built-in speakers:")
        print()
        print(f"  {'Speaker':<12} {'Language'}")
        print(f"  {'-'*12} {'-'*10}")
        for speaker, lang in BUILTIN_SPEAKERS.items():
            print(f"  {speaker:<12} {lang}")
        print()
        print("Supported languages: " + ", ".join(SUPPORTED_LANGUAGES))
        print()
        print("Tip: Use --instruct to control emotion/style:")
        print('  --instruct "Speak warmly and calmly"')
        print('  --instruct "Whisper mysteriously"')
        print('  --instruct "Sound excited and energetic"')
        sys.exit(0)

    # Handle --list-tones
    if args.list_tones:
        print("Tone presets:")
        print()
        print(f"  {'Tone':<15} {'Instruct text'}")
        print(f"  {'-'*15} {'-'*50}")
        for name, text in INSTRUCT_PRESETS.items():
            display = text if text else "(no instruction)"
            print(f"  {name:<15} {display}")
        print()
        print("Usage: --tone warm")
        print("Override with: --instruct \"your custom instruction\"")
        print("Unknown --tone values are treated as raw instruct text.")
        sys.exit(0)

    # Handle --setup
    if args.setup:
        if args.cloud == "modal":
            result = setup_modal(verbose=verbose)
        else:
            result = setup_runpod(gpu_id=args.setup_gpu, verbose=verbose)
        if args.json:
            print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)
        sys.exit(0)

    # Validate required arguments
    if not args.text:
        print("Error: --text is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    # Validate clone mode
    if args.ref_audio and not args.ref_text:
        print("Error: --ref-text is required with --ref-audio", file=sys.stderr)
        sys.exit(1)
    if args.ref_audio and not Path(args.ref_audio).exists():
        print(f"Error: Reference audio not found: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)

    # Mode conflict checks
    if args.design_instruct and args.ref_audio:
        print("Error: --design-instruct and --ref-audio are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    # Capitalize language for API
    language = args.language.capitalize()

    # Resolve tone preset → instruct text
    instruct = resolve_tone(args.tone, args.instruct)

    # Warn if tone/instruct used with clone (clone mode ignores instruct)
    if args.ref_audio and instruct:
        print(
            "Note: --tone/--instruct is ignored when using a cloned voice.\n"
            "  The clone's tone comes from your reference recording.\n"
            "  Tip: Use --design-instruct to create a designed reference instead.",
            file=sys.stderr,
        )
        instruct = ""

    if verbose:
        if args.design_instruct:
            print("Designing voice with Qwen3-TTS VoiceDesign...")
        else:
            print("Generating speech with Qwen3-TTS...")
            if instruct:
                print(f"  Tone: {instruct}")

    result = generate_audio(
        text=args.text,
        output_path=args.output,
        speaker=args.speaker,
        language=language,
        instruct=instruct,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        design_instruct=args.design_instruct,
        output_format=args.format,
        timeout=args.timeout,
        verbose=verbose,
        temperature=args.temperature,
        top_p=args.top_p,
        cloud=args.cloud,
        progress=reporter,
        max_wpm=args.max_wpm,
    )

    if not result.get("success"):
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        duration = result.get("duration_seconds", 0)
        print(f"Generated: {result['output']}")
        if duration:
            wpm_note = f", {result['wpm']:.0f} wpm" if result.get("wpm") else ""
            print(f"  Duration: {duration:.1f}s ({int(duration * 30)} frames @ 30fps{wpm_note})")


if __name__ == "__main__":
    main()
