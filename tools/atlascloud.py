#!/usr/bin/env python3
"""Generate and download images with Atlas Cloud's asynchronous media API.

Examples:
  python3 tools/atlascloud.py --prompt "Editorial title card, no text" --output title.jpg
  python3 tools/atlascloud.py --prompt "Square poster" --size "2048*2048" \
      --output poster.png --output-format png
  python3 tools/atlascloud.py --model bytedance/seedream-v5.0-lite \
      --prompt "Product backdrop" --output backdrop.jpg

Set ATLASCLOUD_API_KEY in the environment or a local .env file before running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
    from dotenv import load_dotenv
except ImportError as error:
    print(f"Missing dependency: {error}", file=sys.stderr)
    print("Install with: pip install requests python-dotenv", file=sys.stderr)
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

API_BASE_URL = "https://api.atlascloud.ai/api/v1"
DEFAULT_MODEL = "bytedance/seedream-v5.0-lite"
SUCCESS_STATUSES = {"completed", "succeeded"}
FAILURE_STATUSES = {"failed", "canceled", "cancelled"}


def log(message: str, level: str = "info") -> None:
    """Write a compact status message to stderr."""
    prefix = {"info": "->", "success": "OK", "error": "!!", "dim": "  "}
    print(f"{prefix.get(level, '->')} {message}", file=sys.stderr)


def _response_data(response: Any, action: str) -> dict[str, Any]:
    """Validate an Atlas Cloud response and return its data object."""
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(f"{action} returned invalid JSON") from error

    if not isinstance(payload, dict):
        raise TypeError(f"{action} returned an invalid response object")
    if payload.get("code") not in (None, 200):
        message = payload.get("message") or "unknown API error"
        raise RuntimeError(f"{action} failed: {message}")

    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise TypeError(f"{action} returned no data object")
    return data


def _output_urls(data: dict[str, Any]) -> list[str]:
    """Extract generated asset URLs from a completed prediction."""
    outputs = data.get("outputs") or data.get("output") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    if not isinstance(outputs, list):
        return []

    urls: list[str] = []
    for output in outputs:
        if isinstance(output, str) and output.startswith(("https://", "http://")):
            urls.append(output)
        elif isinstance(output, dict):
            url = output.get("url")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                urls.append(url)
    return urls


def generate_image(
    api_key: str,
    *,
    prompt: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    size: str | None = None,
    output_format: str | None = None,
    extra_params: dict[str, Any] | None = None,
    poll_interval: float = 3,
    timeout: float = 300,
    request_timeout: float = 60,
    http: Any = requests,
) -> dict[str, Any] | None:
    """Submit, poll, and download one Atlas Cloud image generation."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = dict(extra_params or {})
    payload.update({"model": model, "prompt": prompt})
    if size:
        payload["size"] = size
    if output_format:
        payload["output_format"] = output_format
    if model == DEFAULT_MODEL:
        payload["enable_base64_output"] = False

    try:
        submit = http.post(
            f"{API_BASE_URL}/model/generateImage",
            headers=headers,
            json=payload,
            timeout=request_timeout,
        )
        data = _response_data(submit, "Image submission")
        prediction_id = data.get("id")
        if not isinstance(prediction_id, str) or not prediction_id:
            raise RuntimeError("Image submission returned no prediction ID")

        urls = data.get("urls")
        poll_url = urls.get("get") if isinstance(urls, dict) else None
        if not isinstance(poll_url, str) or not poll_url:
            poll_url = f"{API_BASE_URL}/model/prediction/{prediction_id}"

        log(f"Prediction: {prediction_id}", "dim")
        deadline = time.monotonic() + timeout
        status = str(data.get("status") or "created").lower()

        while status not in SUCCESS_STATUSES:
            if status in FAILURE_STATUSES:
                error = data.get("error") or data.get("message") or status
                raise RuntimeError(f"Generation {status}: {error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Generation did not finish within {timeout:g}s")

            time.sleep(poll_interval)
            poll = http.get(poll_url, headers=headers, timeout=request_timeout)
            data = _response_data(poll, "Prediction polling")
            status = str(data.get("status") or "processing").lower()
            log(f"Status: {status}", "dim")

        output_urls = _output_urls(data)
        if not output_urls:
            raise RuntimeError("Completed prediction returned no output URL")

        download = http.get(output_urls[0], timeout=request_timeout)
        download.raise_for_status()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(download.content)
        log(f"Saved: {destination} ({len(download.content) // 1024} KB)", "success")
        return {
            "prediction_id": prediction_id,
            "model": model,
            "output": str(destination),
            "source_url": output_urls[0],
        }
    except (
        requests.RequestException,
        RuntimeError,
        TypeError,
        TimeoutError,
        OSError,
    ) as error:
        log(str(error), "error")
        return None


def _extra_params(value: str | None) -> dict[str, Any]:
    """Parse optional model-specific parameters from a JSON object."""
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("--extra-params must be a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and download an image with Atlas Cloud.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --prompt "Editorial title card, no text" --output title.jpg
  %(prog)s --prompt "Square poster" --size "2048*2048" \
      --output poster.png --output-format png
  %(prog)s --model bytedance/seedream-v5.0-lite \
      --prompt "Product backdrop" --output backdrop.jpg
        """,
    )
    parser.add_argument("--prompt", "-p", required=True, help="Image description")
    parser.add_argument("--output", "-o", required=True, help="Downloaded image path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Atlas Cloud model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--size", help="Model-specific image size, e.g. 2048*2048")
    parser.add_argument(
        "--output-format",
        choices=("jpeg", "png"),
        help="Requested model output format",
    )
    parser.add_argument(
        "--extra-params", help="Additional model parameters as a JSON object"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3,
        help="Polling interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Overall generation timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--json-out",
        action="store_true",
        help="Emit one machine-readable result line",
    )
    args = parser.parse_args()

    try:
        extra_params = _extra_params(args.extra_params)
    except (json.JSONDecodeError, TypeError) as error:
        parser.error(str(error))

    from config import get_atlascloud_api_key

    api_key = get_atlascloud_api_key()
    if not api_key:
        log("ATLASCLOUD_API_KEY not set.", "error")
        log(
            "Create a key at https://www.atlascloud.ai/console/api-keys "
            "and add it to .env.",
            "info",
        )
        sys.exit(1)

    result = generate_image(
        api_key,
        prompt=args.prompt,
        output_path=args.output,
        model=args.model,
        size=args.size,
        output_format=args.output_format,
        extra_params=extra_params,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    if args.json_out:
        print(json.dumps({"success": result is not None, **(result or {})}))
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
