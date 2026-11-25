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

Ideogram 4.5 is an opt-in (--model 4.5). The same flags go to its v2 endpoint
(POST /v2/image/generate/ideogram-4-5), also as multipart/form-data. Run live, it renders
JSON captions near-perfectly but adds garbled filler text to plain prompts. Ideogram 4 stays
the default because 4.5 sizes must be multiples of 32 (no exact 1280x720). What differs on 4.5:
  - Both prompt modes fill its single `prompt` field (a --json caption as compact JSON).
  - --speed sets `quality`: FLASH and TURBO -> low, DEFAULT -> medium, QUALITY -> high.
  - --resolution is sent as `size`: 'auto', or WIDTHxHEIGHT with both sides multiples of
    32 and at least 256px, at most 2048x2048 pixels in total, aspect ratio at most 6:1.

Two prompt modes (mutually exclusive, mirroring the API's text_prompt vs json_prompt):
  --json    Post a structured JSON caption as json_prompt (recommended for text/layout);
            with --model 4.5 it goes in `prompt`. Claude authors the caption via the
            ideogram4 skill. Pass a file or '-' for stdin.
  --prompt  Post plain text as text_prompt, or `prompt` with --model 4.5 (Ideogram's
            server-side magic prompt expands it).

Examples:
  # Structured caption (recommended) — Claude writes caption.json via the skill
  uv run tools/ideogram4.py --json caption.json --output title.png

  # Caption from stdin
  cat caption.json | uv run tools/ideogram4.py --json - --output title.png

  # Plain prompt (server-side magic prompt)
  uv run tools/ideogram4.py --prompt "Title card: 'AI ENGINEERING REVIEW' bold white on dark" --output title.png

  # Inject brand palette into a JSON caption's style_description.color_palette
  uv run tools/ideogram4.py --json caption.json --brand digital-samba --output cta.png

  # Quality tier + resolution
  uv run tools/ideogram4.py --json caption.json --speed QUALITY --resolution 2048x2048 --output slide.png

  # Ideogram 4.5 (opt-in)
  uv run tools/ideogram4.py --model 4.5 --json caption.json --resolution 1440x2560 --output card.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from media_spend import record_spend

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: uv sync")
    sys.exit(1)

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

API_URL = "https://api.ideogram.ai/v1/ideogram-v4/generate"
RENDERING_SPEEDS = ["FLASH", "TURBO", "DEFAULT", "QUALITY"]
# Per-image list price by tier (USD) for cost-awareness logging, from ideogram.ai/pricing
# (API tab, checked 2026-10-12). FLASH has no published price: Ideogram's docs say it is
# "coming soon" for Ideogram 4 and currently returns HTTP 400. It stays selectable.
SPEED_COST = {"TURBO": 0.03, "DEFAULT": 0.06, "QUALITY": 0.10}

# Ideogram 4.5, opt-in via --model 4.5. Sent as multipart/form-data like the v1 call.
MODELS = ["4", "4.5"]
DEFAULT_MODEL = "4"
API_URL_45 = "https://api.ideogram.ai/v2/image/generate/ideogram-4-5"
# --speed -> 4.5 `quality`. TURBO, DEFAULT and QUALITY map to low, medium and high at
# the same price as on Ideogram 4. FLASH has no Ideogram 4 price; it maps to low too,
# the cheapest tier text-to-image accepts (4.5's very_low needs source images).
SPEED_TO_QUALITY_45 = {"FLASH": "low", "TURBO": "low", "DEFAULT": "medium", "QUALITY": "high"}
# 4.5 per-image list price by quality (USD, same source); 1K and 2K sizes cost the same.
QUALITY_COST_45 = {"low": 0.03, "medium": 0.06, "high": 0.10}
# v1 resolution presets that break 4.5's size rules, each with a size that passes. The
# first four keep the exact aspect ratio at twice the size.
SIZE_SUGGESTIONS_45 = {
    "720x1280": "1440x2560", "1280x720": "2560x1440",
    "720x1440": "1440x2880", "1440x720": "2880x1440",
    "1296x3168": "1280x3072", "3168x1296": "3072x1280",
}


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


def build_fields_4(
    text_prompt: Optional[str],
    json_prompt: Optional[dict],
    *,
    rendering_speed: str,
    resolution: Optional[str],
    copyright_detection: bool,
) -> dict[str, tuple]:
    """Multipart fields for Ideogram 4 on POST /v1/ideogram-v4/generate (the default)."""
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

    if rendering_speed == "FLASH":
        log("Ideogram's docs say FLASH currently returns HTTP 400 for Ideogram 4 ('coming soon'); "
            "TURBO is the cheapest tier that works. Sending FLASH as asked.", "warn")
    cost = SPEED_COST.get(rendering_speed)
    price = f"~${cost:.2f}/image" if cost is not None else "no published price"
    log(f"Speed: {rendering_speed} ({price})  Resolution: {resolution or 'API default'}", "dim")
    return fields


def size_error_45(size: str) -> Optional[str]:
    """Check --resolution against Ideogram 4.5's documented size rules.

    Returns an error message, or None when the size passes. Passing is necessary but not
    sufficient: a text-to-image size must also be one of Ideogram's supported 1K/2K
    presets, which aren't published as a list, so the API can still answer HTTP 400.
    """
    if size == "auto":
        return None
    if size == "source":
        return ("--resolution source needs source images, which this tool doesn't send. "
                "Use 'auto' or WIDTHxHEIGHT.")
    m = re.fullmatch(r"(\d+)x(\d+)", size)
    if not m:
        return f"--resolution must be 'auto' or WIDTHxHEIGHT, e.g. 1440x2560 (got {size!r})."
    w, h = int(m.group(1)), int(m.group(2))
    problems = []
    if w % 32 or h % 32:
        problems.append("both sides must be multiples of 32")
    if min(w, h) < 256:
        problems.append("both sides must be at least 256px")
    if w * h > 2048 * 2048:
        problems.append("at most 2048x2048 pixels in total")
    if max(w, h) > 6 * min(w, h):
        problems.append("aspect ratio at most 6:1")
    if not problems:
        return None
    hint = SIZE_SUGGESTIONS_45.get(size)
    try_size = f"--resolution {hint}" if hint else "a preset such as 1024x1024, 2048x2048 or 1440x2560"
    return f"Ideogram 4.5 can't use --resolution {size} ({'; '.join(problems)}). Try {try_size}, or 'auto'."


def build_fields_45(
    text_prompt: Optional[str],
    json_prompt: Optional[dict],
    *,
    rendering_speed: str,
    resolution: Optional[str],
    copyright_detection: bool,
) -> dict[str, tuple]:
    """Multipart fields for Ideogram 4.5 on POST /v2/image/generate/ideogram-4-5 (opt-in).

    v2 has one `prompt` field for plain text or a JSON caption (4.5 reads the Ideogram 4
    caption format). Values go as (None, str) `files=` entries, as in the v1 call, because
    requests sends `data=` on its own as form-urlencoded rather than multipart.
    """
    if text_prompt is not None:
        prompt = text_prompt
        log(f"prompt: {text_prompt}", "info")
    else:
        prompt = json.dumps(json_prompt, separators=(",", ":"), ensure_ascii=False)
        hld = json_prompt.get("high_level_description", "")
        log(f"prompt (JSON caption): {hld or '(structured caption)'}", "info")
    quality = SPEED_TO_QUALITY_45[rendering_speed]
    fields: dict[str, tuple] = {"prompt": (None, prompt), "quality": (None, quality)}
    if resolution:
        fields["size"] = (None, resolution)
    if copyright_detection:
        fields["enable_copyright_detection"] = (None, "true")

    cost = QUALITY_COST_45[quality]
    log(f"Quality: {quality} (from --speed {rendering_speed}, ~${cost:.2f}/image)"
        f"  Size: {resolution or 'API default (auto)'}", "dim")
    return fields


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
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """Call the Ideogram generate endpoint for `model` and download the result.

    Exactly one of text_prompt / json_prompt must be provided. Returns the output path
    on success, None on failure.
    """
    if (text_prompt is None) == (json_prompt is None):
        log("Provide exactly one of text_prompt / json_prompt.", "error")
        return None

    if model == "4.5":
        api_url, build_fields = API_URL_45, build_fields_45
    else:
        api_url, build_fields = API_URL, build_fields_4
    fields = build_fields(
        text_prompt,
        json_prompt,
        rendering_speed=rendering_speed,
        resolution=resolution,
        copyright_detection=copyright_detection,
    )

    try:
        response = requests.post(
            api_url,
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

    rate = QUALITY_COST_45[SPEED_TO_QUALITY_45[rendering_speed]] if model == "4.5" else SPEED_COST.get(rendering_speed)
    record_spend("ideogram4.py", "ideogram", "images", len(urls), model=model,
                 cost_usd=len(urls) * rate if rate is not None else None)

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
  %(prog)s --model 4.5 --json caption.json --resolution 1440x2560 --output card.png

Authoring captions: see the `ideogram4` skill (.claude/skills/ideogram4/). The --json path
posts the caption as the API's json_prompt (no server-side magic prompt — Claude is the expander).
With --model 4.5 it goes in `prompt`: a valid caption skips magic prompt, an invalid one
is rewritten by it.

--model 4.5 is an opt-in (use it with --json; plain prompts gain filler text). It posts multipart to
/v2/image/generate/ideogram-4-5 with the caption or text in `prompt`, --speed as `quality`
(FLASH and TURBO -> low, DEFAULT -> medium, QUALITY -> high) and --resolution as `size`.
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--json", dest="json_src", metavar="FILE",
        help="Structured JSON caption file (or '-' for stdin), posted as json_prompt "
             "(Ideogram 4) or as prompt (--model 4.5)",
    )
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Plain text prompt, posted as text_prompt (Ideogram 4) or prompt "
             "(--model 4.5); server-side magic prompt expands it",
    )

    parser.add_argument("--output", "-o", required=True, help="Output PNG path")
    parser.add_argument("--brand", help="Brand name — inject brands/<name>/brand.json colors into the caption palette (JSON mode)")
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL,
                        help="Ideogram model: 4 (default) or 4.5 (opt-in, v2 API; use with --json)")
    parser.add_argument("--speed", choices=RENDERING_SPEEDS, default="DEFAULT",
                        help="Rendering speed / quality tier (default: DEFAULT). With --model 4.5 it sets "
                             "quality: FLASH and TURBO -> low, DEFAULT -> medium, QUALITY -> high")
    parser.add_argument("--resolution", help="Resolution e.g. 2048x2048 (omit for API default; see Ideogram docs for valid values). "
                        "With --model 4.5: 'auto' or WIDTHxHEIGHT, both sides multiples of 32 and at least 256, "
                        "at most 2048x2048 pixels in total, aspect ratio at most 6:1")
    parser.add_argument("--copyright-detection", action="store_true",
                        help="Enable Ideogram's copyright detection")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds (default: 300)")
    parser.add_argument("--json-out", action="store_true", help="Emit a machine-readable result line to stdout")

    args = parser.parse_args()

    if args.model == "4.5" and args.resolution:
        size_err = size_error_45(args.resolution)
        if size_err:
            parser.error(size_err)

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
    if args.model == "4.5":
        log("Ideogram 4.5 (hosted v2 API, opt-in)", "info")
    else:
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
        model=args.model,
    )

    if args.json_out:
        print(json.dumps({"success": result is not None, "output": result}))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
