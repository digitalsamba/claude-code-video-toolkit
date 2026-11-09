#!/usr/bin/env python3
"""
Find and download Creative Commons sound effects via the Openverse API.

Searches Freesound through Openverse. No API key is needed (anonymous limit:
20 requests/minute, 200/day). Only licenses that allow commercial use without
share-alike are searched unless --license says otherwise.

Usage:
    # List matches
    uv run tools/sfx_search.py --query "whoosh"

    # Download the top match, with a credit file beside it
    uv run tools/sfx_search.py --query "whoosh" --output public/audio/whoosh.mp3

    # Download a specific result from the list
    uv run tools/sfx_search.py --id 1b2c3d4e-... --output public/audio/whoosh.mp3

Downloads are Freesound's high-quality MP3 previews. A `<name>.attribution.txt`
is written beside each one: CC BY sounds must be credited in the video.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from win_encoding import ensure_utf8_output

API_URL = "https://api.openverse.org/v1/audio/"
HEADERS = {"User-Agent": "claude-code-video-toolkit"}
TIMEOUT = 30
MAX_RETRY_WAIT = 60

# Openverse leaves `category` empty on Freesound records, so filtering on
# category=sound_effect returns nothing. Freesound is the sound-effect source.
SOURCE = "freesound"
DEFAULT_LICENSES = "cc0,by"


class OpenverseError(Exception):
    """An Openverse request that failed in a way the user should hear about."""


def get(url: str, **kwargs) -> requests.Response:
    """GET with one retry when rate limited, if the wait is short."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", MAX_RETRY_WAIT + 1))
        if wait > MAX_RETRY_WAIT:
            raise OpenverseError(f"Openverse rate limit reached, retry in {wait}s")
        time.sleep(wait)
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kwargs)
    if not response.ok:
        raise OpenverseError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")
    return response


def summarize(record: dict) -> dict:
    """Keep the fields a video project needs. Openverse durations are in ms."""
    license_name = record["license"]
    if record.get("license_version"):
        license_name = f"{license_name} {record['license_version']}"
    return {
        "id": record["id"],
        "title": record.get("title") or "",
        "creator": record.get("creator") or "",
        "duration_seconds": round((record.get("duration") or 0) / 1000, 2),
        "license": license_name,
        "license_url": record.get("license_url") or "",
        "attribution": record.get("attribution") or "",
        "url": record["url"],
        "landing_url": record.get("foreign_landing_url") or "",
    }


def search(query: str, licenses: str, max_duration: float, limit: int) -> list[dict]:
    """Search Freesound, dropping sounds with no duration or over max_duration."""
    params = {
        "q": query,
        "source": SOURCE,
        "license": licenses,
        "page_size": 20,  # the anonymous maximum
    }
    results = [summarize(r) for r in get(API_URL, params=params).json()["results"]]
    fitting = [r for r in results if 0 < r["duration_seconds"] <= max_duration]
    return fitting[:limit]


def fetch(identifier: str) -> dict:
    """Look up one sound by its Openverse id."""
    return summarize(get(f"{API_URL}{identifier}/").json())


def download(sound: dict, output: Path) -> Path:
    """Save the audio and its credit file. Returns the credit file's path."""
    response = get(sound["url"])
    if not response.headers.get("Content-Type", "").startswith("audio/"):
        raise OpenverseError(f"Not an audio file: {sound['url']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.content)

    credit = output.with_name(f"{output.stem}.attribution.txt")
    credit.write_text(
        f"{sound['attribution']}\nSource: {sound['landing_url']}\n", encoding="utf-8"
    )
    return credit


def print_results(results: list[dict]) -> None:
    for r in results:
        print(f"{r['id']}  {r['duration_seconds']:>6.2f}s  {r['license']:<9}  {r['title']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find and download Creative Commons sound effects via Openverse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/sfx_search.py --query "camera shutter" --max-duration 3
  uv run tools/sfx_search.py --query "whoosh" --output public/audio/whoosh.mp3
  uv run tools/sfx_search.py --id ID --output public/audio/whoosh.mp3
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", "-q", help="What to search for (English works best)")
    group.add_argument("--id", help="Openverse id of a specific sound (needs --output)")
    parser.add_argument(
        "--output", "-o", help="Download here. With --query, takes the top match"
    )
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSES,
        help=f"Comma-separated licenses (default: {DEFAULT_LICENSES}, both fine commercially)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10,
        help="Longest sound to accept, in seconds (default: 10)",
    )
    parser.add_argument("--limit", type=int, default=10, help="Results to list (default: 10)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    if args.id and not args.output:
        parser.error("--id needs --output")
    return args


def main():
    ensure_utf8_output()
    args = parse_args()

    try:
        if args.id:
            sound = fetch(args.id)
        else:
            results = search(args.query, args.license, args.max_duration, args.limit)
            if not results:
                print(f"No sounds found for: {args.query}", file=sys.stderr)
                sys.exit(1)
            if not args.output:
                if args.json:
                    print(json.dumps(results, indent=2, ensure_ascii=False))
                else:
                    print_results(results)
                return
            sound = results[0]

        output = Path(args.output)
        credit = download(sound, output)
    except (OpenverseError, requests.RequestException) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps({**sound, "output": str(output), "credit": str(credit)}, indent=2, ensure_ascii=False))
    else:
        print(f"Saved: {output} ({sound['duration_seconds']}s, {sound['license']})")
        print(f"Credit: {credit}")
    if sound["license"].startswith(("nc", "by-nc")) or "-nd" in sound["license"]:
        print(f"Warning: {sound['license']} restricts commercial use or edits", file=sys.stderr)


if __name__ == "__main__":
    main()
