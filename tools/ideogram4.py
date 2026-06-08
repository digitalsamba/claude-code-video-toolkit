#!/usr/bin/env python3
"""
AI text-to-image generation using Ideogram 4 (hosted v4 API).

Ideogram 4's strength is best-in-class *in-image text rendering* plus exact color and
layout control — ideal for title cards, thumbnails, and CTAs with baked-in text. That
advantage lives in the model's structured JSON caption format. See the `ideogram4` skill
(.claude/skills/ideogram4/) for how to author captions; Claude is the recommended
"magic prompt" expander.

Backend: Ideogram's hosted v4 API (POST /v1/ideogram-v4/generate). Paid plans include a
commercial license. Needs IDEOGRAM_API_KEY in .env (get a key at developer.ideogram.ai).

Two prompt modes (mutually exclusive, mirroring the API's text_prompt vs json_prompt):
  --json    Post a structured JSON caption as json_prompt (recommended for text/layout).
            Claude authors the caption via the ideogram4 skill. Pass a file or '-' for stdin.
  --prompt  Post plain text as text_prompt (Ideogram's server-side magic prompt expands it).

Examples:
  # Structured caption (recommended) — Claude writes caption.json via the skill
  python3 tools/ideogram4.py --json caption.json --output title.png

  # Caption from stdin
  cat caption.json | python3 tools/ideogram4.py --json - --output title.png

  # Plain prompt (server-side magic prompt)
  python3 tools/ideogram4.py --prompt "Title card: 'AI ENGINEERING REVIEW' bold white on dark" --output title.png

  # Inject brand palette into a JSON caption's style_description.color_palette
  python3 tools/ideogram4.py --json caption.json --brand digital-samba --output cta.png

  # Quality tier + resolution
  python3 tools/ideogram4.py --json caption.json --speed QUALITY --resolution 2048x2048 --output slide.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install requests python-dotenv")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

API_URL = "https://api.ideogram.ai/v1/ideogram-v4/generate"
RENDERING_SPEEDS = ["FLASH", "TURBO", "DEFAULT", "QUALITY"]
# Rough per-image cost by tier (USD) for cost-awareness logging.
SPEED_COST = {"FLASH": 0.02, "TURBO": 0.03, "DEFAULT": 0.06, "QUALITY": 0.09}


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
    print(f"{color}{prefix.get(level, '->')} {msg}{reset}", file=sys.stderr)


def load_brand_palette(brand_name: str) -> list[str]:
    """Load brand.json and return its colors as uppercase #RRGGBB hex strings."""
    workspace = Path(__file__).parent.parent
    brand_path = workspace / "brands" / brand_name / "brand.json"
    if not brand_path.exists():
        log(f"Brand not found: {brand_path}", "warn")
        return []
    try:
        brand = json.loads(brand_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log(f"Error reading brand: {e}", "warn")
        return []

    palette: list[str] = []
    for value in (brand.get("colors") or {}).values():
        if isinstance(value, str) and value.startswith("#") and len(value) in (4, 7):
            hx = value.upper()
            if hx not in palette:
                palette.append(hx)
    return palette[:16]  # API caps style palette at 16


def read_caption(source: str) -> dict:
    """Read a JSON caption from a file path or '-' for stdin. Returns the parsed object."""
    raw = sys.stdin.read() if source == "-" else Path(source).read_text()
    caption = json.loads(raw)
    if not isinstance(caption, dict):
        raise ValueError("JSON caption must be an object (got a non-object top level)")
    return caption


def inject_brand_palette(caption: dict, palette: list[str]) -> dict:
    """Merge brand hex colors into the caption's style_description.color_palette.

    Brand colors are prepended (they take priority) and de-duplicated; existing palette
    colors are preserved after them. Capped at the API's 16-color limit.
    """
    if not palette:
        return caption
    style = caption.setdefault("style_description", {})
    existing = style.get("color_palette") or []
    merged = palette + [c for c in existing if c.upper() not in {p.upper() for p in palette}]
    style["color_palette"] = merged[:16]
    return caption


def generate(
    api_key: str,
    *,
    text_prompt: Optional[str] = None,
    json_prompt: Optional[dict] = None,
    output_path: str,
    rendering_speed: str = "DEFAULT",
    resolution: Optional[str] = None,
    copyright_detection: bool = False,
    timeout: int = 300,
) -> Optional[str]:
    """Call the Ideogram v4 generate endpoint and download the result.

    Exactly one of text_prompt / json_prompt must be provided. Returns the output path
    on success, None on failure.
    """
    if (text_prompt is None) == (json_prompt is None):
        log("Provide exactly one of text_prompt / json_prompt.", "error")
        return None

    # Multipart/form-data: send each field as (None, value). json_prompt is serialized
    # compactly with non-ASCII preserved, per the model's caption format.
    fields: dict[str, tuple] = {"rendering_speed": (None, rendering_speed)}
    if text_prompt is not None:
        fields["text_prompt"] = (None, text_prompt)
        log(f"text_prompt: {text_prompt}", "info")
    else:
        caption_str = json.dumps(json_prompt, separators=(",", ":"), ensure_ascii=False)
        fields["json_prompt"] = (None, caption_str)
        hld = json_prompt.get("high_level_description", "")
        log(f"json_prompt: {hld or '(structured caption)'}", "info")
    if resolution:
        fields["resolution"] = (None, resolution)
    if copyright_detection:
        fields["enable_copyright_detection"] = (None, "true")

    cost = SPEED_COST.get(rendering_speed, 0.06)
    log(f"Speed: {rendering_speed} (~${cost:.2f}/image)  Resolution: {resolution or 'API default'}", "dim")

    try:
        response = requests.post(
            API_URL,
            headers={"Api-Key": api_key},
            files=fields,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        log(f"Request timed out ({timeout}s)", "error")
        return None
    except requests.exceptions.RequestException as e:
        log(f"Request failed: {e}", "error")
        return None

    if response.status_code != 200:
        log(f"API returned HTTP {response.status_code}: {response.text[:500]}", "error")
        return None

    try:
        result = response.json()
    except json.JSONDecodeError:
        log("Invalid JSON response from API", "error")
        return None

    images = result.get("data") or []
    urls = [img.get("url") for img in images if isinstance(img, dict) and img.get("url")]
    if not urls:
        log(f"No image URL in response: {json.dumps(result)[:500]}", "error")
        return None

    # Download. If multiple images came back, suffix _2, _3, ...
    out = Path(output_path)
    saved: list[str] = []
    for i, url in enumerate(urls):
        target = out if i == 0 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        try:
            img_resp = requests.get(url, timeout=timeout)
            img_resp.raise_for_status()
            target.write_bytes(img_resp.content)
        except requests.exceptions.RequestException as e:
            log(f"Download failed for {url}: {e}", "error")
            continue
        saved.append(str(target))
        meta = images[i]
        log(
            f"Saved: {target} ({len(img_resp.content) // 1024} KB)"
            f"{'  seed=' + str(meta.get('seed')) if meta.get('seed') is not None else ''}"
            f"{'  safe=' + str(meta.get('is_image_safe')) if 'is_image_safe' in meta else ''}",
            "success",
        )

    if not saved:
        return None

    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["open", saved[0]], check=False)

    return saved[0]


def main():
    parser = argparse.ArgumentParser(
        description="Ideogram 4 text-to-image (hosted v4 API) — best-in-class in-image text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --json caption.json --output title.png
  cat caption.json | %(prog)s --json - --output title.png
  %(prog)s --prompt "Title card: 'SHIP FASTER' bold" --output thumb.png
  %(prog)s --json caption.json --brand digital-samba --speed QUALITY --output cta.png

Authoring captions: see the `ideogram4` skill (.claude/skills/ideogram4/). The --json path
posts the caption as the API's json_prompt (no server-side magic prompt — Claude is the expander).
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--json", dest="json_src", metavar="FILE",
        help="Structured JSON caption file (or '-' for stdin) — posted as json_prompt",
    )
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Plain text prompt — posted as text_prompt (server-side magic prompt expands it)",
    )

    parser.add_argument("--output", "-o", required=True, help="Output PNG path")
    parser.add_argument("--brand", help="Brand name — inject brands/<name>/brand.json colors into the caption palette (JSON mode)")
    parser.add_argument("--speed", choices=RENDERING_SPEEDS, default="DEFAULT",
                        help="Rendering speed / quality tier (default: DEFAULT)")
    parser.add_argument("--resolution", help="Resolution e.g. 2048x2048 (omit for API default; see Ideogram docs for valid values)")
    parser.add_argument("--copyright-detection", action="store_true",
                        help="Enable Ideogram's copyright detection")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds (default: 300)")
    parser.add_argument("--json-out", action="store_true", help="Emit a machine-readable result line to stdout")

    args = parser.parse_args()

    from config import get_ideogram_api_key
    api_key = get_ideogram_api_key()
    if not api_key:
        log("IDEOGRAM_API_KEY not set.", "error")
        log("Get a key at https://developer.ideogram.ai/ then: echo 'IDEOGRAM_API_KEY=your_key' >> .env", "info")
        sys.exit(1)

    text_prompt = None
    json_prompt = None
    if args.json_src is not None:
        try:
            json_prompt = read_caption(args.json_src)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            log(f"Could not read JSON caption: {e}", "error")
            sys.exit(1)
        if args.brand:
            palette = load_brand_palette(args.brand)
            if palette:
                json_prompt = inject_brand_palette(json_prompt, palette)
                log(f"Brand palette: {', '.join(palette)}", "dim")
    else:
        text_prompt = args.prompt
        if args.brand:
            log("--brand only applies in --json mode (no palette field in plain text). Ignoring.", "warn")

    print(file=sys.stderr)
    log("Ideogram 4 (hosted v4 API)", "info")

    result = generate(
        api_key,
        text_prompt=text_prompt,
        json_prompt=json_prompt,
        output_path=args.output,
        rendering_speed=args.speed,
        resolution=args.resolution,
        copyright_detection=args.copyright_detection,
        timeout=args.timeout,
    )

    if args.json_out:
        print(json.dumps({"success": result is not None, "output": result}))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
