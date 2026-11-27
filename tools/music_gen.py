#!/usr/bin/env python3
"""
AI music generation using ACE-Step 1.5.

Capabilities:
- Text-to-music generation (--prompt)
- Vocal music with lyrics (--lyrics)
- Music control: BPM, key, time signature, duration
- Cover/style transfer from reference audio (--cover)
- Stem extraction (--extract)
- Batch variations — generate multiple takes (--variations)
- Scene presets for video production (--preset)
- Brand-aware generation (--brand)

Cloud providers (your own endpoint; the weights are MIT, so commercial use is free and you pay only GPU time):
  modal   — Self-hosted via Modal (ACE-Step 1.5 XL Turbo 4B, no LM), about $0.01-0.03 a take
  runpod  — Self-hosted via RunPod (ACE-Step 2B Turbo)

Examples:
  # Basic background music (on your Modal endpoint when it is set up)
  uv run tools/music_gen.py --prompt "Subtle corporate tech" --duration 60 --output bg.mp3

  # Generate 4 variations, pick the best
  uv run tools/music_gen.py --prompt "Upbeat electronic" --duration 30 --variations 4 --output intro.mp3


  # With musical control
  uv run tools/music_gen.py --prompt "Upbeat electronic" --duration 30 --bpm 128 --key "G Major" --output intro.mp3

  # Vocal music with lyrics
  uv run tools/music_gen.py --prompt "Indie pop" --lyrics "Hello world\\nWe build together" --duration 30 --output jingle.mp3

  # Scene presets for video production
  uv run tools/music_gen.py --preset corporate-bg --duration 60 --output bg.mp3
  uv run tools/music_gen.py --preset tension --duration 20 --output problem.mp3
  uv run tools/music_gen.py --preset cta --duration 15 --brand digital-samba --output cta.mp3

  # Cover / style transfer
  uv run tools/music_gen.py --cover --reference theme.mp3 --prompt "Same style, longer" --duration 90 --output extended.mp3



  # Stem extraction
  uv run tools/music_gen.py --extract vocals --input mixed.mp3 --output vocals.mp3

  # List presets
  uv run tools/music_gen.py --list-presets

  # Pick the cloud yourself
  uv run tools/music_gen.py --cloud runpod --prompt "Background music" --duration 60 --output bg.mp3

  # Set up the RunPod endpoint (first time; Modal is set up with tools/modal_setup.py music_gen)
  uv run tools/music_gen.py --setup
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv sync")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from file_transfer import delete_from_r2, download_from_url, get_r2_payload_config

# --- Constants ---

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DOCKER_IMAGE = "ghcr.io/conalmullan/video-toolkit-acestep:latest"
TEMPLATE_NAME = "video-toolkit-acestep"
ENDPOINT_NAME = "video-toolkit-acestep"
ENV_VAR_NAME = "RUNPOD_ACESTEP_ENDPOINT_ID"

# --- Scene presets for video production ---
# Musical equivalents of flux2.py visual presets

SCENE_PRESETS = {
    "corporate-bg": {
        "description": "Subtle corporate tech background — professional, non-distracting",
        "prompt": "Subtle corporate technology background music, soft synth pads, minimal beats, professional presentation, modern and clean",
        "bpm": 110,
        "key_scale": "C Major",
        "instrumental": True,
    },
    "upbeat-tech": {
        "description": "Energetic tech demo — driving beats, inspiring",
        "prompt": "Upbeat energetic technology demo music, driving synth bass, electronic beats, inspiring and forward-looking, product launch feel",
        "bpm": 128,
        "key_scale": "G Major",
        "instrumental": True,
    },
    "ambient": {
        "description": "Calm ambient — dreamy, reflective, overview slides",
        "prompt": "Calm ambient atmospheric music, soft piano with gentle reverb, warm pad layers, dreamy and meditative, suitable for reflective content",
        "bpm": 72,
        "key_scale": "D Major",
        "instrumental": True,
    },
    "dramatic": {
        "description": "Cinematic dramatic reveal — building tension, epic",
        "prompt": "Dramatic cinematic reveal music, building tension with orchestral strings, epic brass hits, powerful crescendo, trailer-style impact",
        "bpm": 90,
        "key_scale": "D Minor",
        "instrumental": True,
    },
    "tension": {
        "description": "Dark tension — problem statement, ominous",
        "prompt": "Dark moody tension music, minor key, subtle pulsing bass, ominous atmosphere, unsettling but not aggressive, problem statement mood",
        "bpm": 85,
        "key_scale": "A Minor",
        "instrumental": True,
    },
    "hopeful": {
        "description": "Hopeful resolution — bright, optimistic, solution reveal",
        "prompt": "Hopeful uplifting resolution music, bright synths, gentle acoustic guitar, optimistic mood, solution reveal, sunrise feeling",
        "bpm": 120,
        "key_scale": "C Major",
        "instrumental": True,
    },
    "cta": {
        "description": "Energetic call to action — punchy, motivating",
        "prompt": "Energetic call to action music, punchy drums, driving bass, exciting and motivating, short burst of energy, electronic pop",
        "bpm": 135,
        "key_scale": "E Major",
        "instrumental": True,
    },
    "lofi": {
        "description": "Lo-fi chill — relaxed background for screen recordings",
        "prompt": "Lo-fi chill hip hop beats, vinyl crackle, mellow jazz piano chords, warm bass, relaxed and focused mood, coding music",
        "bpm": 85,
        "key_scale": "F Major",
        "instrumental": True,
    },
}


# --- Logging ---

def log(msg: str, level: str = "info"):
    """Print formatted log message."""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "error": "\033[91m",
        "warn": "\033[93m",
        "dim": "\033[90m",
    }
    reset = "\033[0m"
    prefix = {"info": "->", "success": "OK", "error": "!!", "warn": "??", "dim": "  "}
    color = colors.get(level, "")
    # stderr, so --json leaves stdout holding only the JSON
    print(f"{color}{prefix.get(level, '->')} {msg}{reset}", file=sys.stderr)


# --- Brand integration ---

def load_brand_music_hints(brand_name: str) -> dict:
    """Load brand.json and extract music-relevant hints."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from config import find_workspace_root
        workspace = find_workspace_root()
    except ImportError:
        workspace = Path(__file__).parent.parent

    brand_path = workspace / "brands" / brand_name / "brand.json"
    if not brand_path.exists():
        log(f"Brand not found: {brand_path}", "warn")
        return {}

    try:
        brand = json.loads(brand_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log(f"Error reading brand: {e}", "warn")
        return {}

    hints = {}
    # Extract tone/mood from brand personality if available
    personality = brand.get("personality", {})
    if personality:
        hints["personality"] = personality

    # Check for music-specific config
    music_config = brand.get("music", {})
    if music_config:
        hints.update(music_config)

    return hints


def build_preset_prompt(
    preset_name: str,
    user_prompt: Optional[str] = None,
    brand_name: Optional[str] = None,
) -> dict:
    """Build generation params from a preset, with optional user overrides and brand hints."""
    preset = SCENE_PRESETS.get(preset_name)
    if not preset:
        raise ValueError(f"Unknown preset: {preset_name}. Use --list-presets to see options.")

    params = {
        "prompt": preset["prompt"],
        "bpm": preset["bpm"],
        "key_scale": preset["key_scale"],
    }

    # Append user prompt context if provided
    if user_prompt:
        params["prompt"] = f"{preset['prompt']}, {user_prompt}"

    # Apply brand hints
    if brand_name:
        hints = load_brand_music_hints(brand_name)
        if hints.get("genre"):
            params["prompt"] += f", {hints['genre']} style"
        if hints.get("mood"):
            params["prompt"] += f", {hints['mood']} mood"

    return params


def list_presets():
    """Print available scene presets."""
    print("\nScene Music Presets (for video production)")
    print("=" * 65)
    for name, preset in SCENE_PRESETS.items():
        bpm_key = f"{preset['bpm']} BPM, {preset['key_scale']}"
        print(f"  {name:<16} {bpm_key:>20}  {preset['description']}")
    print()
    print("Usage:")
    print("  music_gen.py --preset corporate-bg --duration 60 --output bg.mp3")
    print("  music_gen.py --preset tension --duration 20 --output problem.mp3")
    print("  music_gen.py --preset cta --brand digital-samba --duration 15 --output cta.mp3")
    print()


# --- Audio helpers ---

def get_audio_duration(file_path: str) -> Optional[float]:
    """Get audio duration using ffprobe if available."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
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
        pass
    return None


def encode_audio(path: str) -> str:
    """Encode audio file to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# --- RunPod API ---

# --- Generation functions (Modal/RunPod) ---

def default_cloud(env=None) -> Optional[str]:
    """The cloud to use when --cloud is not given: a self-hosted endpoint that is set up, or None."""
    env = os.environ if env is None else env
    if env.get("MODAL_MUSIC_GEN_ENDPOINT_URL"):
        return "modal"
    if env.get("RUNPOD_ACESTEP_ENDPOINT_ID") and env.get("RUNPOD_API_KEY"):
        return "runpod"
    return None


NO_ENDPOINT = ("No music endpoint is set up. Deploy one on Modal with `uv run tools/modal_setup.py music_gen` "
               "(about $0.01-0.03 a take), or on RunPod with `uv run tools/music_gen.py --setup`.")


def take_paths(output_path: str, variations: int) -> list[str]:
    """Where N takes are saved: <stem>_1<ext> to <stem>_N<ext>, or the path itself for one."""
    if variations <= 1:
        return [output_path]
    out = Path(output_path)
    return [str(out.parent / f"{out.stem}_{i + 1}{out.suffix}") for i in range(variations)]


def generate_music(
    prompt: str,
    output_path: str,
    duration: int = 30,
    bpm: Optional[int] = None,
    key_scale: Optional[str] = None,
    time_signature: Optional[int] = None,
    lyrics: Optional[str] = None,
    vocal_language: str = "en",
    steps: int = 8,
    audio_format: str = "mp3",
    seed: Optional[int] = None,
    json_output: bool = False,
    cloud: str = "runpod",
    progress=None,
) -> Optional[dict]:
    """Generate music from text prompt via cloud GPU."""
    log("ACE-Step 1.5 — Music Generation", "info")
    log("=" * 40, "dim")
    log(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}", "info")
    log(f"Duration: {duration}s | Steps: {steps}", "dim")
    if bpm:
        log(f"BPM: {bpm}", "dim")
    if key_scale:
        log(f"Key: {key_scale}", "dim")

    payload = {
        "input": {
            "task_type": "text2music",
            "prompt": prompt,
            "audio_duration": duration,
            "inference_steps": steps,
            "audio_format": audio_format,
        }
    }

    if lyrics:
        payload["input"]["lyrics"] = lyrics
        payload["input"]["vocal_language"] = vocal_language
    if bpm:
        payload["input"]["bpm"] = bpm
    if key_scale:
        payload["input"]["key_scale"] = key_scale
    if time_signature:
        payload["input"]["time_signature"] = time_signature
    if seed is not None:
        payload["input"]["seed"] = seed

    # R2 for file transfer
    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload

    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="music_gen",
        timeout=600,
        progress_label="Generating music",
        progress=progress,
    )

    if "error" in result:
        log(f"Generation failed: {result['error']}", "error")
        return None

    # Download the audio
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if "output_url" in result:
        log("Downloading from R2...", "dim")
        if download_from_url(result["output_url"], output_path, verbose=False):
            # The GPU side leaves the result object in R2. Drop it once the
            # download succeeded, so the bucket does not grow without bound.
            if result.get("r2_key"):
                delete_from_r2(result["r2_key"])
    elif "audio_base64" in result:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["audio_base64"]))
    else:
        log(f"Unexpected result format: {list(result.keys())}", "error")
        return None

    # Get actual duration
    actual_duration = get_audio_duration(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024

    log(f"Saved: {output_path} ({file_size_kb:.0f} KB)", "success")
    log(f"Time: {elapsed:.1f}s total", "dim")
    if actual_duration:
        log(f"Audio duration: {actual_duration:.1f}s (requested {duration}s)", "dim")

    metas = result.get("metas", {})
    if metas:
        log(f"Detected: BPM={metas.get('bpm', '?')}, Key={metas.get('keyscale', '?')}", "dim")

    output = {
        "success": True,
        "output": output_path,
        "prompt": prompt,
        "requested_duration": duration,
        "elapsed_seconds": round(elapsed, 2),
        "file_size_kb": round(file_size_kb, 1),
    }
    if actual_duration:
        output["actual_duration_seconds"] = round(actual_duration, 2)
    if metas:
        output["metas"] = metas
    if result.get("seed_value"):
        output["seed"] = result["seed_value"]

    return output


def generate_cover(
    reference_path: str,
    prompt: str,
    output_path: str,
    duration: int = 30,
    cover_strength: float = 0.7,
    steps: int = 8,
    audio_format: str = "mp3",
    json_output: bool = False,
    cloud: str = "runpod",
    progress=None,
) -> Optional[dict]:
    """Generate a cover/style transfer from reference audio."""
    if not Path(reference_path).exists():
        log(f"Reference file not found: {reference_path}", "error")
        return None

    log("ACE-Step 1.5 — Cover / Style Transfer", "info")
    log("=" * 40, "dim")
    log(f"Reference: {reference_path}", "info")
    log(f"Prompt: {prompt[:80]}", "info")
    log(f"Strength: {cover_strength} | Duration: {duration}s", "dim")

    payload = {
        "input": {
            "task_type": "cover",
            "prompt": prompt,
            "reference_audio_base64": encode_audio(reference_path),
            "audio_cover_strength": cover_strength,
            "audio_duration": duration,
            "inference_steps": steps,
            "audio_format": audio_format,
        }
    }

    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload

    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="music_gen",
        timeout=600,
        progress_label="Creating cover",
        progress=progress,
    )

    if "error" in result:
        log(f"Cover failed: {result['error']}", "error")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if "output_url" in result:
        if download_from_url(result["output_url"], output_path, verbose=False):
            # The GPU side leaves the result object in R2. Drop it once the
            # download succeeded, so the bucket does not grow without bound.
            if result.get("r2_key"):
                delete_from_r2(result["r2_key"])
    elif "audio_base64" in result:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["audio_base64"]))
    else:
        log(f"Unexpected result format: {list(result.keys())}", "error")
        return None

    actual_duration = get_audio_duration(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024

    log(f"Saved: {output_path} ({file_size_kb:.0f} KB)", "success")
    log(f"Time: {elapsed:.1f}s", "dim")

    return {
        "success": True,
        "output": output_path,
        "reference": reference_path,
        "elapsed_seconds": round(elapsed, 2),
        "actual_duration_seconds": round(actual_duration, 2) if actual_duration else None,
    }


def extract_stem(
    input_path: str,
    track: str,
    output_path: str,
    steps: int = 8,
    audio_format: str = "mp3",
    json_output: bool = False,
    cloud: str = "runpod",
    progress=None,
) -> Optional[dict]:
    """Extract a stem (vocals, drums, bass, etc.) from mixed audio."""
    if not Path(input_path).exists():
        log(f"Input file not found: {input_path}", "error")
        return None

    log("ACE-Step 1.5 — Stem Extraction", "info")
    log("=" * 40, "dim")
    log(f"Input: {input_path}", "info")
    log(f"Track: {track}", "info")

    payload = {
        "input": {
            "task_type": "extract",
            "src_audio_base64": encode_audio(input_path),
            "prompt": track,
            "inference_steps": steps,
            "audio_format": audio_format,
        }
    }

    r2_payload = get_r2_payload_config()
    if r2_payload:
        payload["input"]["r2"] = r2_payload

    from cloud_gpu import call_cloud_endpoint

    result, elapsed = call_cloud_endpoint(
        provider=cloud,
        payload=payload,
        tool_name="music_gen",
        timeout=600,
        progress_label="Extracting stem",
        progress=progress,
    )

    if "error" in result:
        log(f"Extraction failed: {result['error']}", "error")
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if "output_url" in result:
        if download_from_url(result["output_url"], output_path, verbose=False):
            # The GPU side leaves the result object in R2. Drop it once the
            # download succeeded, so the bucket does not grow without bound.
            if result.get("r2_key"):
                delete_from_r2(result["r2_key"])
    elif "audio_base64" in result:
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(result["audio_base64"]))
    else:
        log(f"Unexpected result format: {list(result.keys())}", "error")
        return None

    actual_duration = get_audio_duration(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024

    log(f"Saved: {output_path} ({file_size_kb:.0f} KB)", "success")
    log(f"Time: {elapsed:.1f}s", "dim")

    return {
        "success": True,
        "output": output_path,
        "track": track,
        "elapsed_seconds": round(elapsed, 2),
        "actual_duration_seconds": round(actual_duration, 2) if actual_duration else None,
    }


# --- RunPod setup ---

def runpod_graphql_query(api_key: str, query: str, variables: Optional[dict] = None) -> dict:
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


def find_template(api_key: str) -> Optional[dict]:
    """Find existing ACE-Step template."""
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
    for t in templates:
        if not t.get("isServerless"):
            continue
        if t.get("name") == TEMPLATE_NAME or t.get("imageName") == DOCKER_IMAGE:
            return t
    return None


def create_template(api_key: str, verbose: bool = True) -> dict:
    """Create a serverless template for ACE-Step."""
    if verbose:
        print(f"Creating template '{TEMPLATE_NAME}'...")

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
            "name": TEMPLATE_NAME,
            "imageName": DOCKER_IMAGE,
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


def find_endpoint(api_key: str, template_id: str) -> Optional[dict]:
    """Find existing ACE-Step endpoint."""
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
    endpoints = data.get("myself", {}).get("endpoints", [])
    for e in endpoints:
        if e.get("name") == ENDPOINT_NAME or e.get("templateId") == template_id:
            return e
    return None


def create_endpoint(
    api_key: str,
    template_id: str,
    gpu_id: str = "AMPERE_24,ADA_24",
    verbose: bool = True,
) -> dict:
    """Create a serverless endpoint for ACE-Step."""
    if verbose:
        print(f"Creating endpoint '{ENDPOINT_NAME}'...")

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
            "name": ENDPOINT_NAME,
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
        if line.startswith(f"{ENV_VAR_NAME}="):
            new_lines.append(f"{ENV_VAR_NAME}={endpoint_id}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"{ENV_VAR_NAME}={endpoint_id}")

    env_path.write_text("\n".join(new_lines))

    if verbose:
        print(f"  Saved: {ENV_VAR_NAME}={endpoint_id}")

    return True


def setup_runpod(gpu_id: str = "AMPERE_24,ADA_24", verbose: bool = True) -> dict:
    """Set up RunPod endpoint for ACE-Step."""
    result = {
        "success": False,
        "template_id": None,
        "endpoint_id": None,
        "created_template": False,
        "created_endpoint": False,
    }

    api_key = os.getenv("RUNPOD_API_KEY")

    if not api_key:
        result["error"] = "RUNPOD_API_KEY not set. Add to .env file first."
        return result

    if verbose:
        print("=" * 60)
        print("RunPod Setup (ACE-Step 1.5 Music Generation)")
        print("=" * 60)
        print(f"Docker Image: {DOCKER_IMAGE}")
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
            template = create_template(api_key, verbose=verbose)
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
            endpoint = create_endpoint(
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
            print('  uv run tools/music_gen.py --prompt "Upbeat tech" --duration 60 --output music.mp3')
            print('  uv run tools/music_gen.py --preset corporate-bg --duration 120 --output bg.mp3')
            print()

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)

    return result


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="AI music generation using ACE-Step 1.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "Subtle corporate tech" --duration 60 --output bg.mp3
  %(prog)s --prompt "Upbeat electronic" --duration 30 --bpm 128 --key "G Major" --output intro.mp3
  %(prog)s --preset corporate-bg --duration 60 --output bg.mp3
  %(prog)s --preset cta --brand digital-samba --duration 15 --output cta.mp3
  %(prog)s --cover --reference theme.mp3 --prompt "Same style" --duration 90 --output ext.mp3
  %(prog)s --extract vocals --input mixed.mp3 --output vocals.mp3
  %(prog)s --list-presets
  %(prog)s --setup
        """
    )

    # Generation mode
    mode_group = parser.add_argument_group("Generation")
    mode_group.add_argument("--prompt", "-p", help="Music description prompt")
    mode_group.add_argument("--lyrics", "-l", help="Song lyrics (enables vocals)")
    mode_group.add_argument("--vocal-language", default="en",
                            help="Vocal language code (default: en)")

    # Musical control
    music_group = parser.add_argument_group("Musical Control")
    music_group.add_argument("--duration", "-d", type=int, default=None,
                             help="Duration in seconds (10-600, default: 30, or the input's length for --cover)")
    music_group.add_argument("--bpm", type=int, help="Tempo in BPM (30-300)")
    music_group.add_argument("--key", dest="key_scale", help="Musical key (e.g., 'C Major', 'Am', 'F# minor')")
    music_group.add_argument("--time-sig", type=int, choices=[2, 3, 4, 6],
                             help="Time signature (2, 3, 4, or 6)")

    # Presets
    preset_group = parser.add_argument_group("Presets")
    preset_group.add_argument("--preset", choices=list(SCENE_PRESETS.keys()),
                              help="Scene music preset for video production")
    preset_group.add_argument("--brand", help="Brand name (loads hints from brands/<name>/brand.json)")
    preset_group.add_argument("--list-presets", action="store_true", help="List available scene presets")

    # Cover / extract modes
    edit_group = parser.add_argument_group("Cover, Extract & Edit")
    edit_group.add_argument("--cover", action="store_true", help="Cover/style transfer mode")
    edit_group.add_argument("--reference", help="Reference audio for cover mode")
    edit_group.add_argument("--cover-strength", type=float, default=0.7,
                            help="Cover strength 0.0-1.0 (default: 0.7)")
    edit_group.add_argument("--extract", metavar="TRACK",
                            help="Extract stem: vocals, drums, bass, guitar, piano, other")
    edit_group.add_argument("--input", "-i", help="Input audio for extract/cover")

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output", "-o", required=False,
                              help="Output audio file path")
    output_group.add_argument("--format", dest="audio_format", default="mp3",
                              choices=["mp3", "wav", "flac"],
                              help="Audio format (default: mp3)")

    # Advanced
    adv_group = parser.add_argument_group("Advanced")
    adv_group.add_argument("--steps", type=int, default=8,
                           help="Inference steps (default: 8 for turbo, use 32-64 for base model)")
    adv_group.add_argument("--seed", type=int, help="Random seed for reproducibility")
    adv_group.add_argument("--variations", type=int, default=1,
                           help="Generate N takes (1-8, default: 1), saved as <output>_1 to <output>_N")

    # Cloud provider
    cloud_group = parser.add_argument_group("Cloud")
    cloud_group.add_argument("--cloud", type=str, default=None,
                             choices=["modal", "runpod"],
                             help="Cloud provider (default: modal when its endpoint is in .env, then runpod)")

    # Setup
    setup_group = parser.add_argument_group("Setup")
    setup_group.add_argument("--setup", action="store_true", help="Set up RunPod endpoint")
    setup_group.add_argument("--setup-gpu", default="AMPERE_24,ADA_24",
                             help="GPU type(s) for endpoint (default: AMPERE_24,ADA_24)")

    # Output format
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--progress", choices=["human", "json"], default="human",
                        help="Progress output mode: human (colored stderr, default) "
                             "or json (JSON Lines to stderr for bots/agents)")

    args = parser.parse_args()

    # A cover reworks a track, so by default it keeps its length; a fixed 30s default cut a
    # 46s bed to 30s without saying so. Generation keeps 30s.
    if args.duration is None:
        src = args.input or args.reference
        src_len = get_audio_duration(src) if args.cover and src and Path(src).exists() else None
        args.duration = int(-(-src_len // 1)) if src_len else 30

    # Handle --list-presets
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # Handle --setup
    if args.setup:
        result = setup_runpod(gpu_id=args.setup_gpu, verbose=not args.json)
        if args.json:
            print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)
        sys.exit(0)

    if args.cloud is None:
        args.cloud = default_cloud()
    if args.cloud is None and not args.dry_run:
        parser.error(NO_ENDPOINT)

    # Create progress reporter
    from cloud_gpu import ProgressReporter
    reporter = ProgressReporter(mode=args.progress)

    # Handle --extract mode
    if args.extract:
        if not args.input:
            parser.error("--extract requires --input")
        if not args.output:
            stem = Path(args.input).stem
            args.output = f"{stem}_{args.extract}.{args.audio_format}"

        if args.dry_run:
            result = {
                "dry_run": True,
                "mode": "extract",
                "input": args.input,
                "track": args.extract,
                "output": args.output,
            }
            print(json.dumps(result, indent=2) if args.json else f"Would extract '{args.extract}' from {args.input}")
            sys.exit(0)

        result = extract_stem(
            input_path=args.input,
            track=args.extract,
            output_path=args.output,
            steps=args.steps,
            audio_format=args.audio_format,
            json_output=args.json,
            cloud=args.cloud,
            progress=reporter,
        )
        if args.json and result:
            print(json.dumps(result, indent=2))
        sys.exit(0 if result else 1)

    # Handle --cover mode
    if args.cover:
        ref = args.reference or args.input
        if not ref:
            parser.error("--cover requires --reference or --input")
        if not args.prompt:
            parser.error("--cover requires --prompt")
        if not args.output:
            stem = Path(ref).stem
            args.output = f"{stem}_cover.{args.audio_format}"

        if args.dry_run:
            result = {
                "dry_run": True,
                "mode": "cover",
                "reference": ref,
                "prompt": args.prompt,
                "duration": args.duration,
                "output": args.output,
            }
            print(json.dumps(result, indent=2) if args.json else f"Would create cover of {ref}")
            sys.exit(0)

        result = generate_cover(
            reference_path=ref,
            prompt=args.prompt,
            output_path=args.output,
            duration=args.duration,
            cover_strength=args.cover_strength,
            steps=args.steps,
            audio_format=args.audio_format,
            json_output=args.json,
            cloud=args.cloud,
            progress=reporter,
        )
        if args.json and result:
            print(json.dumps(result, indent=2))
        sys.exit(0 if result else 1)

    # Resolve prompt from preset or direct input
    if args.preset:
        preset_params = build_preset_prompt(args.preset, args.prompt, args.brand)
        prompt = preset_params["prompt"]
        bpm = args.bpm or preset_params.get("bpm")
        key_scale = args.key_scale or preset_params.get("key_scale")
        log(f"Preset: {args.preset}", "info")
        if args.brand:
            log(f"Brand: {args.brand}", "dim")
    elif args.prompt:
        prompt = args.prompt
        bpm = args.bpm
        key_scale = args.key_scale
    else:
        parser.print_help()
        print("\n\033[91m!! --prompt, --preset, --cover, or --extract is required\033[0m")
        sys.exit(1)

    # Default output path
    if not args.output:
        if args.preset:
            args.output = f"{args.preset}.{args.audio_format}"
        else:
            slug = prompt[:30].strip().replace(" ", "_").lower()
            slug = "".join(c for c in slug if c.isalnum() or c == "_")
            args.output = f"{slug}.{args.audio_format}"

    # Validate duration
    if args.duration < 10 or args.duration > 600:
        print("Error: Duration must be between 10 and 600 seconds", file=sys.stderr)
        sys.exit(1)

    # Dry run
    if args.dry_run:
        result = {
            "dry_run": True,
            "mode": "text2music",
            "prompt": prompt,
            "duration": args.duration,
            "bpm": bpm,
            "key_scale": key_scale,
            "lyrics": args.lyrics,
            "output": args.output,
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would generate music:")
            print(f"  Prompt: {prompt}")
            print(f"  Duration: {args.duration}s")
            if bpm:
                print(f"  BPM: {bpm}")
            if key_scale:
                print(f"  Key: {key_scale}")
            if args.lyrics:
                print(f"  Lyrics: {args.lyrics[:50]}...")
            print(f"  Output: {args.output}")
        sys.exit(0)

    print()
    # Takes on a self-hosted endpoint are one request each, saved as <stem>_1<ext> to <stem>_N<ext>. A given
    # seed moves on by one per take, so the takes differ.
    takes = []
    paths = take_paths(args.output, args.variations)
    for i, path in enumerate(paths):
        if len(paths) > 1:
            log(f"Take {i + 1}/{len(paths)}", "info")
        take = generate_music(
            prompt=prompt,
            output_path=path,
            duration=args.duration,
            bpm=bpm,
            key_scale=key_scale,
            time_signature=args.time_sig,
            lyrics=args.lyrics,
            vocal_language=args.vocal_language,
            steps=args.steps,
            audio_format=args.audio_format,
            seed=None if args.seed is None else args.seed + i,
            json_output=args.json,
            cloud=args.cloud,
            progress=reporter,
        )
        if take:
            takes.append(take)
        elif len(paths) > 1:
            log(f"Take {i + 1} failed; keeping the others", "warn")
    result = None
    if takes and len(paths) > 1:
        result = {**takes[0], "outputs": [t.get("output") for t in takes], "variations": len(takes),
                  "requested_variations": len(paths)}
    elif takes:
        result = takes[0]

    if args.json and result:
        print(json.dumps(result, indent=2))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
