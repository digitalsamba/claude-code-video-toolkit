#!/usr/bin/env python3
"""Generate videos with the MiniMax hosted video API.

Supports text-to-video and image-to-video generation in the global and China
regions. Video generation is asynchronous: the tool creates a task, polls it,
retrieves the generated file, and downloads the final MP4.

Examples:
  python3 tools/minimax_video.py \
      --prompt "A cinematic aerial shot over a coastal city" \
      --output coast.mp4

  python3 tools/minimax_video.py \
      --prompt "Slow camera push-in, natural wind in the trees" \
      --input first-frame.png \
      --output animated.mp4

  python3 tools/minimax_video.py \
      --region cn_zh \
      --prompt "A product reveal with soft studio lighting" \
      --output reveal.mp4
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any, Callable

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install requests python-dotenv")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

API_BASES = {
    "global_en": "https://api.minimax.io",
    "cn_zh": "https://api.minimaxi.com",
}
DEFAULT_MODEL = "MiniMax-H3"
V2_MODELS = {DEFAULT_MODEL}
HAILUO_23_MODEL = "MiniMax-Hailuo-2.3"
FAST_MODEL = "MiniMax-Hailuo-2.3-Fast"
HAILUO_02_MODEL = "MiniMax-Hailuo-02"
V1_MODELS = [
    HAILUO_23_MODEL,
    FAST_MODEL,
    HAILUO_02_MODEL,
    "T2V-01-Director",
    "T2V-01",
    "I2V-01-Director",
    "I2V-01-live",
    "I2V-01",
]
SUPPORTED_MODELS = [DEFAULT_MODEL, *V1_MODELS]
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
V2_RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]


class MiniMaxVideoError(RuntimeError):
    """Raised when the video API returns an error or invalid response."""


def log(message: str, level: str = "info") -> None:
    """Print a formatted status message to stderr."""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "error": "\033[91m",
        "warn": "\033[93m",
        "dim": "\033[90m",
    }
    reset = "\033[0m"
    prefix = {"info": "->", "success": "OK", "error": "!!", "warn": "??", "dim": "  "}
    print(f"{colors.get(level, '')}{prefix.get(level, '->')} {message}{reset}", file=sys.stderr)


def api_url(region: str, path: str) -> str:
    """Build an API URL for an official MiniMax region."""
    try:
        base = API_BASES[region]
    except KeyError as e:
        raise ValueError(f"Unknown region: {region}") from e
    return f"{base}{path}"


def image_source(source: str) -> str:
    """Return a URL/data URL, encoding a local image as a data URL when needed."""
    if source.startswith(("https://", "http://")):
        return source
    if source.startswith("data:image/"):
        media_type = source.split(";", 1)[0].removeprefix("data:")
        if media_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image data URL type: {media_type}")
        return source

    path = Path(source)
    if not path.is_file():
        raise ValueError(f"Input image not found: {source}")
    if path.stat().st_size >= MAX_IMAGE_BYTES:
        raise ValueError("Input image must be smaller than 20 MB")

    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in SUPPORTED_IMAGE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_TYPES))
        raise ValueError(f"Unsupported image type: {media_type or 'unknown'} (use {supported})")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def build_payload(
    *,
    prompt: str,
    model: str,
    duration: int,
    resolution: str,
    first_frame_image: str | None = None,
    prompt_optimizer: bool | None = None,
    fast_pretreatment: bool = False,
    ratio: str = "adaptive",
) -> dict[str, Any]:
    """Build and validate a video generation request payload."""
    if not prompt.strip():
        raise ValueError("Prompt must not be empty")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if model in V2_MODELS:
        if not 4 <= duration <= 15:
            raise ValueError(f"{model} duration must be between 4 and 15 seconds")
        if resolution != "2K":
            raise ValueError(f"{model} supports 2K resolution")
        if ratio not in V2_RATIOS:
            raise ValueError(f"Unsupported ratio: {ratio}")

        content: list[dict[str, str]] = [{"type": "text", "text": prompt}]
        if first_frame_image is not None:
            content.append({"type": "image_url", "image_url": first_frame_image, "role": "first_frame"})
        payload: dict[str, Any] = {
            "model": model,
            "content": content,
            "duration": duration,
            "resolution": resolution,
        }
        if ratio != "adaptive":
            payload["ratio"] = ratio
        return payload

    if model == FAST_MODEL and first_frame_image is None:
        raise ValueError(f"{FAST_MODEL} requires --input for image-to-video generation")
    if resolution == "2K":
        raise ValueError(f"{model} does not support 2K resolution")
    if duration == 10 and resolution == "1080P":
        raise ValueError("10-second generation supports 768P resolution")
    if model in {HAILUO_23_MODEL, FAST_MODEL} and resolution in {"512P", "720P"}:
        raise ValueError(f"{model} supports 768P or 1080P resolution")
    if model == HAILUO_02_MODEL and first_frame_image is None and resolution == "512P":
        raise ValueError(f"{HAILUO_02_MODEL} supports 512P only for image-to-video generation")

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
    }
    if first_frame_image is not None:
        payload["first_frame_image"] = first_frame_image
    if prompt_optimizer is not None:
        payload["prompt_optimizer"] = prompt_optimizer
    if fast_pretreatment:
        payload["fast_pretreatment"] = True
    return payload


def api_version_for_model(model: str) -> str:
    """Return the video API version required by a supported model."""
    if model in V2_MODELS:
        return "v2"
    if model in V1_MODELS:
        return "v1"
    raise ValueError(f"Unsupported model: {model}")


def request_json(
    session: Any,
    method: str,
    url: str,
    *,
    api_key: str,
    timeout: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Send an authenticated request and validate the API response envelope."""
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = session.request(method, url, headers=headers, timeout=timeout, **kwargs)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise MiniMaxVideoError(f"Request failed: {e}") from e

    try:
        data = response.json()
    except ValueError as e:
        raise MiniMaxVideoError("API returned invalid JSON") from e
    if not isinstance(data, dict):
        raise MiniMaxVideoError("API returned a non-object response")

    base_response = data.get("base_resp") or {}
    status_code = base_response.get("status_code")
    if status_code not in (None, 0):
        message = base_response.get("status_msg") or "unknown API error"
        raise MiniMaxVideoError(f"API error {status_code}: {message}")
    return data


def create_video_task(
    api_key: str,
    payload: dict[str, Any],
    *,
    region: str,
    request_timeout: int,
    session: Any = requests,
) -> str:
    """Create an asynchronous video task and return its ID."""
    api_version = api_version_for_model(str(payload.get("model", "")))
    path = "/v2/video_generation" if api_version == "v2" else "/v1/video_generation"
    data = request_json(
        session,
        "POST",
        api_url(region, path),
        api_key=api_key,
        timeout=request_timeout,
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    task_id = data.get("task_id")
    if not task_id:
        raise MiniMaxVideoError("Task creation response did not include task_id")
    return str(task_id)


def poll_video_task(
    api_key: str,
    task_id: str,
    *,
    region: str,
    request_timeout: int,
    generation_timeout: int,
    poll_interval: float,
    session: Any = requests,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Poll a video task until it succeeds, fails, or times out."""
    deadline = monotonic() + generation_timeout
    while True:
        data = request_json(
            session,
            "GET",
            api_url(region, "/v1/query/video_generation"),
            api_key=api_key,
            timeout=request_timeout,
            params={"task_id": task_id},
        )
        status = data.get("status")
        if status == "Success":
            file_id = data.get("file_id")
            if not file_id:
                raise MiniMaxVideoError("Successful task did not include file_id")
            return str(file_id)
        if status == "Fail":
            message = (data.get("base_resp") or {}).get("status_msg") or "unknown generation error"
            raise MiniMaxVideoError(f"Video generation failed: {message}")

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise MiniMaxVideoError(f"Video generation timed out after {generation_timeout}s")
        log(f"Task {task_id}: {status or 'unknown'}", "dim")
        sleep(min(poll_interval, remaining))


def poll_video_task_v2(
    api_key: str,
    task_id: str,
    *,
    region: str,
    request_timeout: int,
    generation_timeout: int,
    poll_interval: float,
    session: Any = requests,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """Poll a v2 video task until it returns a downloadable content URL."""
    deadline = monotonic() + generation_timeout
    while True:
        data = request_json(
            session,
            "GET",
            api_url(region, f"/v2/query/video_generation/{task_id}"),
            api_key=api_key,
            timeout=request_timeout,
        )
        task = data.get("task") or {}
        status = task.get("status")
        content = task.get("content") or {}
        if status in {"Success", "Succeeded", "Completed", "success", "succeeded", "completed"}:
            url = content.get("url")
            if not url:
                raise MiniMaxVideoError("Successful task did not include task.content.url")
            return str(url)
        if status in {"Fail", "Failed", "Error", "fail", "failed", "error"}:
            error = task.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else error
            raise MiniMaxVideoError(f"Video generation failed: {message or 'unknown generation error'}")

        remaining = deadline - monotonic()
        if remaining <= 0:
            raise MiniMaxVideoError(f"Video generation timed out after {generation_timeout}s")
        log(f"Task {task_id}: {status or 'unknown'}", "dim")
        sleep(min(poll_interval, remaining))


def retrieve_download_url(
    api_key: str,
    file_id: str,
    *,
    region: str,
    request_timeout: int,
    session: Any = requests,
) -> str:
    """Retrieve the temporary download URL for a generated video file."""
    data = request_json(
        session,
        "GET",
        api_url(region, "/v1/files/retrieve"),
        api_key=api_key,
        timeout=request_timeout,
        params={"file_id": file_id},
    )
    download_url = (data.get("file") or {}).get("download_url")
    if not download_url:
        raise MiniMaxVideoError("File response did not include download_url")
    return str(download_url)


def download_video(
    download_url: str,
    output_path: str,
    *,
    request_timeout: int,
    session: Any = requests,
) -> str:
    """Download a generated video to disk."""
    try:
        response = session.request("GET", download_url, timeout=request_timeout, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise MiniMaxVideoError(f"Video download failed: {e}") from e

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return str(output)


def generate_video(
    api_key: str,
    *,
    prompt: str,
    output_path: str,
    input_image: str | None = None,
    model: str = DEFAULT_MODEL,
    duration: int = 6,
    resolution: str = "2K",
    region: str = "global_en",
    prompt_optimizer: bool | None = None,
    fast_pretreatment: bool = False,
    ratio: str = "adaptive",
    request_timeout: int = 60,
    generation_timeout: int = 900,
    poll_interval: float = 10,
    session: Any = requests,
) -> dict[str, str]:
    """Generate and download a video, returning task and file metadata."""
    if request_timeout <= 0:
        raise ValueError("Request timeout must be greater than zero")
    if generation_timeout <= 0:
        raise ValueError("Generation timeout must be greater than zero")
    if poll_interval <= 0:
        raise ValueError("Poll interval must be greater than zero")

    first_frame = image_source(input_image) if input_image else None
    payload = build_payload(
        prompt=prompt,
        model=model,
        duration=duration,
        resolution=resolution,
        first_frame_image=first_frame,
        prompt_optimizer=prompt_optimizer,
        fast_pretreatment=fast_pretreatment,
        ratio=ratio,
    )

    mode = "image-to-video" if first_frame else "text-to-video"
    log(f"Mode: {mode}  Model: {model}  Region: {region}", "info")
    log(f"Duration: {duration}s  Resolution: {resolution}", "dim")

    task_id = create_video_task(
        api_key,
        payload,
        region=region,
        request_timeout=request_timeout,
        session=session,
    )
    log(f"Task submitted: {task_id}", "success")
    api_version = api_version_for_model(model)
    if api_version == "v2":
        download_url = poll_video_task_v2(
            api_key,
            task_id,
            region=region,
            request_timeout=request_timeout,
            generation_timeout=generation_timeout,
            poll_interval=poll_interval,
            session=session,
        )
        file_id = ""
    else:
        file_id = poll_video_task(
            api_key,
            task_id,
            region=region,
            request_timeout=request_timeout,
            generation_timeout=generation_timeout,
            poll_interval=poll_interval,
            session=session,
        )
        download_url = retrieve_download_url(
            api_key,
            file_id,
            region=region,
            request_timeout=request_timeout,
            session=session,
        )
    output = download_video(
        download_url,
        output_path,
        request_timeout=request_timeout,
        session=session,
    )
    log(f"Saved: {output}", "success")
    return {"output": output, "task_id": task_id, "file_id": file_id}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text-to-video or image-to-video clips with the MiniMax hosted API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "A cinematic ocean sunrise" --output sunrise.mp4
  %(prog)s --prompt "Slow camera push-in" --input still.png --output animated.mp4
  %(prog)s --region cn_zh --prompt "A studio product reveal" --output reveal.mp4
        """,
    )
    parser.add_argument("--prompt", "-p", required=True, help="Video content and motion prompt")
    parser.add_argument("--input", "-i", help="First-frame image path, public URL, or image data URL")
    parser.add_argument("--output", "-o", required=True, help="Output MP4 path")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=DEFAULT_MODEL,
                        help=f"Video model (default: {DEFAULT_MODEL})")
    parser.add_argument("--duration", type=int, default=6,
                        help="Video duration in seconds (default: 6)")
    parser.add_argument("--resolution", choices=["2K", "512P", "768P", "1080P"], default="2K",
                        help="Output resolution (default: 2K)")
    parser.add_argument("--region", choices=sorted(API_BASES), default="global_en",
                        help="API region: global_en or cn_zh (default: global_en)")
    parser.add_argument("--ratio", choices=V2_RATIOS, default="adaptive",
                        help="Aspect ratio for MiniMax-H3 (default: adaptive)")
    parser.add_argument("--prompt-optimizer", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable or disable server-side prompt optimization")
    parser.add_argument("--fast-pretreatment", action="store_true",
                        help="Reduce prompt optimization time on supported models")
    parser.add_argument("--request-timeout", type=int, default=60,
                        help="Per-request timeout seconds (default: 60)")
    parser.add_argument("--generation-timeout", type=int, default=900,
                        help="Overall generation timeout seconds (default: 900)")
    parser.add_argument("--poll-interval", type=float, default=10,
                        help="Task polling interval seconds (default: 10)")
    parser.add_argument("--no-open", action="store_true", help="Do not open the generated video")
    parser.add_argument("--json-out", action="store_true", help="Emit a machine-readable result line")
    args = parser.parse_args()

    from config import get_minimax_api_key

    api_key = get_minimax_api_key()
    if not api_key:
        log("MINIMAX_API_KEY not set.", "error")
        log("Add MINIMAX_API_KEY to .env before using this tool.", "info")
        sys.exit(1)

    try:
        result = generate_video(
            api_key,
            prompt=args.prompt,
            output_path=args.output,
            input_image=args.input,
            model=args.model,
            duration=args.duration,
            resolution=args.resolution,
            region=args.region,
            prompt_optimizer=args.prompt_optimizer,
            fast_pretreatment=args.fast_pretreatment,
            ratio=args.ratio,
            request_timeout=args.request_timeout,
            generation_timeout=args.generation_timeout,
            poll_interval=args.poll_interval,
        )
    except (MiniMaxVideoError, OSError, ValueError) as e:
        log(str(e), "error")
        if args.json_out:
            print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

    if args.json_out:
        print(json.dumps({"success": True, **result}))
    if not args.no_open and sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", result["output"]], check=False)


if __name__ == "__main__":
    main()
