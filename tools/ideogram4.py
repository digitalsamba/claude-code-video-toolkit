#!/usr/bin/env python3
"""
AI text-to-image generation (and precise editing) with Ideogram 4.5 / 4.0 (hosted v2 API).

Ideogram's strength is best-in-class *in-image text rendering* plus exact color and
layout control — ideal for title cards, thumbnails, and CTAs with baked-in text. That
advantage lives in the structured JSON caption format introduced with Ideogram 4.0 (4.5
still accepts it). See the `ideogram4` skill (.claude/skills/ideogram4/) for how to author
captions; Claude is the recommended "magic prompt" expander.

Backend: Ideogram's hosted v2 API — the model is in the path (POST /v2/{content}/{action}/{model}):
  --model 4.5 (default)  POST /v2/image/generate/ideogram-4-5
  --model 4              POST /v2/image/generate/ideogram-4
  --edit IMAGE           POST /v2/image/precise-edit/ideogram-4-5   (4.5 only)
Paid plans include a commercial license. Needs IDEOGRAM_API_KEY in .env (get a key at
developer.ideogram.ai).

Two prompt modes (mutually exclusive). Both fill the API's single `prompt` field:
  --json    A structured JSON caption, serialized into `prompt` (recommended for text/layout).
            A valid caption skips server-side magic prompt, so Claude stays the expander.
            Claude authors the caption via the ideogram4 skill. Pass a file or '-' for stdin.
  --prompt  Plain text. Ideogram's server-side magic prompt expands it (see --magic-prompt).

Per-model knobs (the tool rejects a knob the chosen model does not have):
  4.5   --quality low|medium|high (default high; very_low is edit-only)
        --resolution auto | WxH (multiples of 32, <= 2048x2048 pixels total, <= 6:1)
  4.0   --speed turbo|default|quality (no FLASH tier on v2)
        --resolution one of the 4.0 presets (e.g. 1440x2560, 2048x2048)

Examples:
  # Structured caption (recommended) — Claude writes caption.json via the skill
  uv run tools/ideogram4.py --json caption.json --output title.png

  # Caption from stdin
  cat caption.json | uv run tools/ideogram4.py --json - --output title.png

  # Plain prompt (server-side magic prompt)
  uv run tools/ideogram4.py --prompt "Title card: 'AI ENGINEERING REVIEW' bold white on dark" --output title.png

  # Inject brand palette into a JSON caption's style_description.color_palette
  uv run tools/ideogram4.py --json caption.json --brand digital-samba --output cta.png

  # Quality tier + size (Ideogram 4.5)
  uv run tools/ideogram4.py --json caption.json --quality medium --resolution 2048x2048 --output slide.png

  # Ideogram 4.0 with a speed tier
  uv run tools/ideogram4.py --model 4 --json caption.json --speed quality --output slide.png

  # Price quote without generating (nothing is billed)
  uv run tools/ideogram4.py --json caption.json --resolution 1440x2560 --dry-run

  # Precise edit (4.5): only the described change; output keeps the image's own size
  uv run tools/ideogram4.py --edit title.png --prompt "Change the headline colour to #FF6B35" --output title_v2.png
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
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

API_BASE = "https://api.ideogram.ai/v2/image"
# CLI model name -> v2 path slug.
MODELS = {"4.5": "ideogram-4-5", "4": "ideogram-4"}
MODEL_LABELS = {"4.5": "Ideogram 4.5", "4": "Ideogram 4.0"}
DEFAULT_MODEL = "4.5"

# Ideogram 4.5 `quality`. very_low requires source images, so it is edit-only here.
QUALITIES = ["very_low", "low", "medium", "high"]
# Ideogram 4.0 `rendering_speed`. v2 is lowercase and has no FLASH tier (v1 did).
RENDERING_SPEEDS = ["turbo", "default", "quality"]
# Per-image list price (USD) for the 4.0 tiers, for cost-awareness logging. 4.5 is priced by
# quality and output size (1K/2K) — use --dry-run for an exact quote.
SPEED_COST = {"turbo": 0.03, "default": 0.06, "quality": 0.10}

# Ideogram 4.0 `resolution` is an enum in the v2 API reference.
V4_RESOLUTIONS = [
    "2048x2048", "1440x2880", "2880x1440", "1664x2496", "2496x1664", "1792x2240", "2240x1792",
    "1440x2560", "2560x1440", "1600x2560", "2560x1600", "1728x2304", "2304x1728", "1296x3168",
    "3168x1296", "1152x2944", "2944x1152", "1248x3328", "3328x1248", "1280x3072", "3072x1280",
    "1024x3072", "3072x1024", "1024x1024", "896x1120", "1120x896", "864x1152", "1152x864",
    "832x1248", "1248x832", "800x1280", "1280x800", "720x1280", "1280x720", "720x1440",
    "1440x720", "512x1536", "1536x512",
]

MAX_PROMPT_CHARS = 10_000   # 4.5 generate + precise-edit `prompt` maxLength
MAX_NUM_IMAGES = 8
MAX_SEED = 2_147_483_647
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Explicit map: mimetypes doesn't know .webp on every platform (e.g. Windows' registry).
UPLOAD_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
MAX_REFERENCES = 4          # precise-edit reference_images; the mask takes one slot


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
    return palette[:16]  # Caption format caps the style palette at 16


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
    colors are preserved after them. Capped at the caption format's 16-color limit.
    """
    if not palette:
        return caption
    style = caption.setdefault("style_description", {})
    existing = style.get("color_palette") or []
    merged = palette + [c for c in existing if c.upper() not in {p.upper() for p in palette}]
    style["color_palette"] = merged[:16]
    return caption


def caption_to_prompt(caption: dict) -> str:
    """Serialize a JSON caption into the v2 `prompt` string (compact, non-ASCII preserved)."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def size_error_45(size: str) -> Optional[str]:
    """Check an Ideogram 4.5 `size` against the documented rules. Returns an error or None.

    Passing this check does not guarantee acceptance: text-to-image only takes Ideogram's
    supported 1K/2K presets (not published as a list). --dry-run validates for free.
    """
    if size == "auto":
        return None
    if size == "source":
        return "size 'source' needs source images (use --edit, which always keeps the image's size)"
    m = re.fullmatch(r"(\d+)x(\d+)", size)
    if not m:
        return f"--resolution must be 'auto' or WIDTHxHEIGHT, e.g. 2048x2048 (got {size!r})"
    w, h = int(m.group(1)), int(m.group(2))
    if w % 32 or h % 32:
        return f"Ideogram 4.5 sizes need both sides to be multiples of 32 (got {size})"
    if min(w, h) < 256:
        return f"Ideogram 4.5 sizes need both sides >= 256px (got {size})"
    if w * h > 2048 * 2048:
        return f"Ideogram 4.5 sizes are capped at 2048x2048 pixels total (got {size} = {w * h:,} px)"
    if max(w, h) > 6 * min(w, h):
        return f"Ideogram 4.5 sizes are capped at a 6:1 aspect ratio (got {size})"
    return None


def validate_options(
    *,
    model: str,
    edit: bool,
    prompt: str,
    quality: Optional[str],
    rendering_speed: Optional[str],
    resolution: Optional[str],
    magic_prompt: Optional[str],
    seed: Optional[int],
    num_images: Optional[int],
    mask: Optional[str] = None,
    references: Optional[list[str]] = None,
) -> list[str]:
    """Return a list of human-readable errors for options the chosen model/endpoint lacks."""
    errors: list[str] = []
    label = MODEL_LABELS[model]

    if edit and model != "4.5":
        errors.append("--edit (precise edit) is Ideogram 4.5 only — drop --model 4")

    if model == "4.5" or edit:
        if rendering_speed is not None:
            errors.append(
                f"--speed is an Ideogram 4.0 option (rendering_speed). {label} uses "
                "--quality low|medium|high instead, or pass --model 4"
            )
        if quality == "very_low" and not edit:
            errors.append("--quality very_low requires source images on Ideogram 4.5 (use --edit), "
                          "or pick low|medium|high")
        if len(prompt) > MAX_PROMPT_CHARS:
            errors.append(f"prompt is {len(prompt):,} chars; Ideogram 4.5 caps it at {MAX_PROMPT_CHARS:,}")
    else:  # Ideogram 4.0 generate
        if quality is not None:
            errors.append("--quality is an Ideogram 4.5 option. Ideogram 4.0 uses "
                          "--speed turbo|default|quality instead, or drop --model 4")
        if rendering_speed == "flash":
            errors.append("FLASH was a v1-only tier; v2 Ideogram 4.0 offers --speed "
                          "turbo|default|quality (turbo is the cheapest)")

    if edit:
        if resolution is not None:
            errors.append("--resolution can't be set with --edit — precise edit always returns "
                          "the input image's own width and height")
        if magic_prompt is not None:
            errors.append("--magic-prompt doesn't apply to --edit (precise edit always converts "
                          "the instruction into a structured edit prompt)")
        refs = references or []
        limit = MAX_REFERENCES - (1 if mask else 0)
        if len(refs) > limit:
            errors.append(f"precise edit takes at most {limit} --reference images"
                          f"{' with a --mask' if mask else ''} (got {len(refs)})")
    else:
        if mask or references:
            errors.append("--mask / --reference only apply with --edit")
        if resolution is not None and resolution != "auto":
            if model == "4.5":
                err = size_error_45(resolution)
                if err:
                    errors.append(err)
            elif resolution not in V4_RESOLUTIONS:
                errors.append(f"Ideogram 4.0 has no {resolution} preset. Use one of: "
                              f"{', '.join(V4_RESOLUTIONS[:12])}, ... (or 'auto' / omit)")

    if seed is not None and not 0 <= seed <= MAX_SEED:
        errors.append(f"--seed must be between 0 and {MAX_SEED}")
    if num_images is not None and not 1 <= num_images <= MAX_NUM_IMAGES:
        errors.append(f"--num-images must be between 1 and {MAX_NUM_IMAGES}")
    return errors


def build_generate_body(
    prompt: str,
    *,
    model: str,
    quality: Optional[str] = None,
    rendering_speed: Optional[str] = None,
    resolution: Optional[str] = None,
    magic_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    num_images: Optional[int] = None,
    copyright_detection: bool = False,
) -> dict:
    """Build the application/json body for POST /v2/image/generate/{model}.

    Unset options are omitted so the API applies its own defaults (4.5: quality high,
    size auto; 4.0: rendering_speed default, aspect picked from the prompt).
    """
    body: dict = {"prompt": prompt}
    if model == "4.5":
        if quality:
            body["quality"] = quality
        if resolution:
            body["size"] = resolution
    else:
        if rendering_speed:
            body["rendering_speed"] = rendering_speed
        if resolution and resolution != "auto":  # 4.0: omitting resolution == auto
            body["resolution"] = resolution
    if magic_prompt:
        body["magic_prompt"] = magic_prompt
    if seed is not None:
        body["seed"] = seed
    if num_images is not None:
        body["num_images"] = num_images
    if copyright_detection:
        body["enable_copyright_detection"] = True
    return body


def build_edit_form(
    prompt: str,
    *,
    quality: Optional[str] = None,
    seed: Optional[int] = None,
    num_images: Optional[int] = None,
    copyright_detection: bool = False,
) -> dict[str, str]:
    """Build the non-file multipart fields for POST /v2/image/precise-edit/ideogram-4-5."""
    form = {"prompt": prompt}
    if quality:
        form["quality"] = quality
    if seed is not None:
        form["seed"] = str(seed)
    if num_images is not None:
        form["num_images"] = str(num_images)
    if copyright_detection:
        form["enable_copyright_detection"] = "true"
    return form


def check_upload(path: str, what: str) -> Optional[str]:
    """Return an error if an upload is missing, oversized, or not JPEG/PNG/WEBP."""
    p = Path(path)
    if not p.is_file():
        return f"{what} not found: {path}"
    if p.suffix.lower() not in UPLOAD_MIME:
        return f"{what} must be JPEG, PNG, or WEBP (got {p.suffix or 'no extension'}): {path}"
    if p.stat().st_size > MAX_UPLOAD_BYTES:
        return f"{what} is over Ideogram's 25MB upload limit: {path}"
    return None


def describe_http_error(response: requests.Response) -> str:
    """Summarise an error response, surfacing v2's error/reject_reason when present."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(body, dict) and body.get("reject_reason"):
        return f"{body.get('error', '')} (reject_reason={body['reject_reason']})".strip()
    return json.dumps(body)[:500]


def post(
    api_key: str,
    url: str,
    *,
    json_body: Optional[dict] = None,
    form: Optional[dict] = None,
    files: Optional[list] = None,
    dry_run: bool = False,
    timeout: int = 300,
) -> Optional[dict]:
    """POST to a v2 endpoint. Returns the parsed JSON (a PriceQuote when dry_run), or None."""
    params = {"dry_run": "true"} if dry_run else None
    try:
        response = requests.post(
            url,
            headers={"Api-Key": api_key},
            params=params,
            json=json_body,
            data=form,
            files=files,
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        log(f"Request timed out ({timeout}s)", "error")
        return None
    except requests.exceptions.RequestException as e:
        log(f"Request failed: {e}", "error")
        return None

    if response.status_code != 200:
        log(f"API returned HTTP {response.status_code}: {describe_http_error(response)}", "error")
        return None

    try:
        result = response.json()
    except ValueError:
        log("Invalid JSON response from API", "error")
        return None
    if not isinstance(result, dict):
        log(f"Unexpected response shape: {json.dumps(result)[:500]}", "error")
        return None
    return result


def generate(
    api_key: str,
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    quality: Optional[str] = None,
    rendering_speed: Optional[str] = None,
    resolution: Optional[str] = None,
    magic_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    num_images: Optional[int] = None,
    copyright_detection: bool = False,
    dry_run: bool = False,
    timeout: int = 300,
) -> Optional[dict]:
    """Call POST /v2/image/generate/{model}. Returns the parsed response, or None on failure."""
    url = f"{API_BASE}/generate/{MODELS[model]}"
    body = build_generate_body(
        prompt,
        model=model,
        quality=quality,
        rendering_speed=rendering_speed,
        resolution=resolution,
        magic_prompt=magic_prompt,
        seed=seed,
        num_images=num_images,
        copyright_detection=copyright_detection,
    )
    if model == "4.5":
        log(f"Quality: {quality or 'high (API default)'}  Size: {resolution or 'auto'}"
            f"  (price: --dry-run for a quote)", "dim")
    else:
        speed = rendering_speed or "default"
        log(f"Speed: {speed} (~${SPEED_COST[speed]:.2f}/image)  "
            f"Resolution: {resolution or 'API default'}", "dim")
    return post(api_key, url, json_body=body, dry_run=dry_run, timeout=timeout)


def precise_edit(
    api_key: str,
    *,
    prompt: str,
    image: str,
    mask: Optional[str] = None,
    references: Optional[list[str]] = None,
    quality: Optional[str] = None,
    seed: Optional[int] = None,
    num_images: Optional[int] = None,
    copyright_detection: bool = False,
    dry_run: bool = False,
    timeout: int = 300,
) -> Optional[dict]:
    """Call POST /v2/image/precise-edit/ideogram-4-5 (multipart). Returns the parsed response.

    Pixels the edit doesn't touch are copied from `image`; the output keeps its size. The
    mask is black = edit, white = keep, and must match `image`'s width and height.
    """
    url = f"{API_BASE}/precise-edit/{MODELS['4.5']}"
    form = build_edit_form(
        prompt, quality=quality, seed=seed, num_images=num_images,
        copyright_detection=copyright_detection,
    )
    log(f"Image: {image}{'  Mask: ' + mask if mask else ''}"
        f"{'  References: ' + str(len(references)) if references else ''}", "dim")
    log(f"Quality: {quality or 'medium (API default)'}  (price: --dry-run for a quote)", "dim")

    with contextlib.ExitStack() as stack:
        def part(field: str, path: str) -> tuple:
            mime = UPLOAD_MIME.get(Path(path).suffix.lower(), "application/octet-stream")
            return (field, (Path(path).name, stack.enter_context(open(path, "rb")), mime))

        files = [part("image", image)]
        files += [part("reference_images", ref) for ref in (references or [])]
        if mask:
            files.append(part("mask", mask))
        return post(api_key, url, form=form, files=files, dry_run=dry_run, timeout=timeout)


def save_images(result: dict, output_path: str, timeout: int = 300) -> Optional[str]:
    """Download every image in a v2 response's `data`. Returns the first saved path, or None.

    If multiple images came back, later ones get _2, _3, ... suffixes.
    """
    images = [img for img in (result.get("data") or []) if isinstance(img, dict)]
    for i, img in enumerate(images):
        if not img.get("url") and img.get("is_image_safe") is False:
            log(f"Image {i + 1} failed Ideogram's safety checks (no URL returned)", "warn")
    usable = [img for img in images if img.get("url")]
    if not usable:
        log(f"No image URL in response: {json.dumps(result)[:500]}", "error")
        return None
    if result.get("generation_id"):
        log(f"generation_id={result['generation_id']}", "dim")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, meta in enumerate(usable):
        target = out if i == 0 else out.with_name(f"{out.stem}_{i + 1}{out.suffix}")
        try:
            img_resp = requests.get(meta["url"], timeout=timeout)
            img_resp.raise_for_status()
            target.write_bytes(img_resp.content)
        except requests.exceptions.RequestException as e:
            log(f"Download failed for {meta['url']}: {e}", "error")
            continue
        saved.append(str(target))
        log(
            f"Saved: {target} ({len(img_resp.content) // 1024} KB)"
            f"{'  ' + meta['resolution'] if meta.get('resolution') else ''}"
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
        description="Ideogram 4.5 / 4.0 text-to-image + 4.5 precise edit (hosted v2 API) — best-in-class in-image text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --json caption.json --output title.png
  cat caption.json | %(prog)s --json - --output title.png
  %(prog)s --prompt "Title card: 'SHIP FASTER' bold" --output thumb.png
  %(prog)s --json caption.json --brand digital-samba --quality high --output cta.png
  %(prog)s --model 4 --json caption.json --speed quality --resolution 1440x2560 --output card.png
  %(prog)s --json caption.json --resolution 1440x2560 --dry-run          # price quote only
  %(prog)s --edit title.png --prompt "Make the headline teal" --output title_v2.png
  %(prog)s --edit shot.png --mask mask.png --reference logo.png --prompt "Put this logo on the mug" -o out.png

Models: --model 4.5 (default) -> /v2/image/generate/ideogram-4-5 (--quality, --resolution auto|WxH)
        --model 4             -> /v2/image/generate/ideogram-4   (--speed, --resolution 4.0 presets)
        --edit                -> /v2/image/precise-edit/ideogram-4-5 (4.5 only; keeps the image's size;
                                 mask is black = edit, white = keep)

Authoring captions: see the `ideogram4` skill (.claude/skills/ideogram4/). The --json caption is
posted as the API's `prompt`; a valid caption skips server-side magic prompt (Claude is the expander).
        """,
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--json", dest="json_src", metavar="FILE",
        help="Structured JSON caption file (or '-' for stdin) — posted as the prompt, used as is",
    )
    prompt_group.add_argument(
        "--prompt", "-p",
        help="Plain text prompt (or edit instruction with --edit) — server-side magic prompt expands it",
    )

    parser.add_argument("--output", "-o", help="Output PNG path (required unless --dry-run)")
    parser.add_argument("--model", "-m", choices=list(MODELS), default=DEFAULT_MODEL,
                        help=f"Ideogram model (default: {DEFAULT_MODEL})")
    parser.add_argument("--brand", help="Brand name — inject brands/<name>/brand.json colors into the caption palette (JSON mode)")
    parser.add_argument("--quality", type=lambda s: s.lower().replace("-", "_"), choices=QUALITIES,
                        help="Ideogram 4.5 quality tier (default: high; edit default: medium; very_low is edit-only)")
    parser.add_argument("--speed", type=str.lower, choices=RENDERING_SPEEDS + ["flash"],
                        metavar="{turbo,default,quality}",
                        help="Ideogram 4.0 rendering speed (--model 4 only; default: default)")
    parser.add_argument("--resolution", "--size", dest="resolution", type=str.lower,
                        help="Output size: 'auto' or WxH, e.g. 1440x2560 (4.5: multiples of 32, "
                             "<= 2048x2048 px total; 4.0: one of its presets). Omit for auto")
    parser.add_argument("--magic-prompt", choices=["auto", "on", "off"],
                        help="Server-side prompt expansion (default: auto). 'on' also rewrites JSON captions")
    parser.add_argument("--seed", type=int, help=f"Random seed for reproducible output (0-{MAX_SEED})")
    parser.add_argument("--num-images", type=int,
                        help=f"Images per request (1-{MAX_NUM_IMAGES}); extras saved as <name>_2, <name>_3, ...")
    parser.add_argument("--copyright-detection", action="store_true",
                        help="Enable Ideogram's copyright detection")

    edit_group = parser.add_argument_group("precise edit (Ideogram 4.5)")
    edit_group.add_argument("--edit", metavar="IMAGE",
                            help="Image to edit (JPEG/PNG/WEBP, <= 25MB). The prompt becomes the edit instruction")
    edit_group.add_argument("--mask", metavar="FILE",
                            help="Optional mask, same size as IMAGE: black = edit, white = keep")
    edit_group.add_argument("--reference", action="append", metavar="FILE", default=None,
                            help=f"Reference image to guide the edit (repeatable, max {MAX_REFERENCES}; "
                                 f"{MAX_REFERENCES - 1} with --mask)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and price the exact request via ?dry_run=true — nothing generated or billed")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout seconds (default: 300)")
    parser.add_argument("--json-out", action="store_true", help="Emit a machine-readable result line to stdout")

    args = parser.parse_args()

    if not args.output and not args.dry_run:
        parser.error("--output is required (unless --dry-run)")

    text_prompt = None
    if args.json_src is not None:
        try:
            caption = read_caption(args.json_src)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            log(f"Could not read JSON caption: {e}", "error")
            sys.exit(1)
        if args.brand:
            palette = load_brand_palette(args.brand)
            if palette:
                caption = inject_brand_palette(caption, palette)
                log(f"Brand palette: {', '.join(palette)}", "dim")
        prompt = caption_to_prompt(caption)
        if args.magic_prompt == "on":
            log("--magic-prompt on rewrites your JSON caption server-side (default skips it).", "warn")
    else:
        prompt = text_prompt = args.prompt
        if args.brand:
            log("--brand only applies in --json mode (no palette field in plain text). Ignoring.", "warn")

    edit = args.edit is not None
    errors = validate_options(
        model=args.model,
        edit=edit,
        prompt=prompt,
        quality=args.quality,
        rendering_speed=args.speed,
        resolution=args.resolution,
        magic_prompt=args.magic_prompt,
        seed=args.seed,
        num_images=args.num_images,
        mask=args.mask,
        references=args.reference,
    )
    if edit:
        uploads = [(args.edit, "--edit image")] + ([(args.mask, "--mask")] if args.mask else [])
        uploads += [(ref, "--reference") for ref in (args.reference or [])]
        errors += [err for path, what in uploads if (err := check_upload(path, what))]
    if errors:
        for err in errors:
            log(err, "error")
        sys.exit(2)

    from config import get_ideogram_api_key
    api_key = get_ideogram_api_key()
    if not api_key:
        log("IDEOGRAM_API_KEY not set.", "error")
        log("Get a key at https://developer.ideogram.ai/ then: echo 'IDEOGRAM_API_KEY=your_key' >> .env", "info")
        sys.exit(1)

    print(file=sys.stderr)
    action = "precise edit" if edit else "generate"
    log(f"{MODEL_LABELS[args.model]} {action} (hosted v2 API){'  [dry run]' if args.dry_run else ''}", "info")
    if text_prompt is not None:
        log(f"prompt: {text_prompt}", "info")
    else:
        log(f"prompt (JSON caption): {caption.get('high_level_description') or '(structured caption)'}", "info")

    common = dict(
        prompt=prompt,
        seed=args.seed,
        num_images=args.num_images,
        copyright_detection=args.copyright_detection,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )
    if edit:
        response = precise_edit(
            api_key, image=args.edit, mask=args.mask, references=args.reference,
            quality=args.quality, **common,
        )
    else:
        response = generate(
            api_key, model=args.model, quality=args.quality, rendering_speed=args.speed,
            resolution=args.resolution, magic_prompt=args.magic_prompt, **common,
        )

    if args.dry_run:
        # The PriceQuote shape isn't pinned down in Ideogram's docs, so pass it through verbatim.
        if response is not None:
            log("Dry run: request validated and priced — nothing generated or billed.", "success")
        if args.json_out:
            print(json.dumps({"success": response is not None, "output": None,
                              "dry_run": True, "quote": response}))
        elif response is not None:
            print(json.dumps(response, indent=2))
        sys.exit(0 if response is not None else 1)

    result = save_images(response, args.output, timeout=args.timeout) if response else None

    if args.json_out:
        print(json.dumps({"success": result is not None, "output": result}))

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
