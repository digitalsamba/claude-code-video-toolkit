#!/usr/bin/env python3
"""Generate music with MiniMax's global or mainland China API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ENDPOINTS = {
    "global_en": "https://api.minimax.io/v1/music_generation",
    "cn_zh": "https://api.minimaxi.com/v1/music_generation",
}
MODELS = ("music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free")
OUTPUT_FORMATS = ("url", "hex")
AUDIO_FORMATS = ("mp3", "wav", "pcm")
SAMPLE_RATES = (16000, 24000, 32000, 44100)
BITRATES = (32000, 64000, 128000, 256000)


class MiniMaxMusicError(RuntimeError):
    """Raised when a MiniMax request or response is invalid."""


def build_payload(
    *,
    model: str,
    prompt: str | None,
    lyrics: str | None,
    stream: bool,
    output_format: str,
    sample_rate: int,
    bitrate: int,
    audio_format: str,
    lyrics_optimizer: bool,
    is_instrumental: bool,
    region: str,
    aigc_watermark: bool | None = None,
) -> dict[str, Any]:
    """Validate inputs and build the documented music-generation payload."""
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if region not in ENDPOINTS:
        raise ValueError(f"Unsupported region: {region}")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    if audio_format not in AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {audio_format}")
    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"Unsupported sample rate: {sample_rate}")
    if bitrate not in BITRATES:
        raise ValueError(f"Unsupported bitrate: {bitrate}")
    if stream and output_format != "hex":
        raise ValueError("Streaming output only supports hex")
    if aigc_watermark is not None and region != "cn_zh":
        raise ValueError("aigc_watermark is only available for the cn_zh region")
    if stream and aigc_watermark:
        raise ValueError("aigc_watermark is only available for non-streaming requests")

    prompt = prompt.strip() if prompt else None
    lyrics = lyrics.strip() if lyrics else None
    if prompt and len(prompt) > 2000:
        raise ValueError("Prompt must be no more than 2000 characters")
    if lyrics and len(lyrics) > 3500:
        raise ValueError("Lyrics must be no more than 3500 characters")
    if is_instrumental and not prompt:
        raise ValueError("Instrumental generation requires a prompt")
    if not is_instrumental and not lyrics and not lyrics_optimizer:
        raise ValueError("Vocal generation requires lyrics or --lyrics-optimizer")
    if lyrics_optimizer and not lyrics and not prompt:
        raise ValueError("Lyrics optimization without lyrics requires a prompt")

    payload: dict[str, Any] = {
        "model": model,
        "stream": stream,
        "output_format": output_format,
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": audio_format,
        },
        "lyrics_optimizer": lyrics_optimizer,
        "is_instrumental": is_instrumental,
    }
    if prompt:
        payload["prompt"] = prompt
    if lyrics:
        payload["lyrics"] = lyrics
    if aigc_watermark is not None:
        payload["aigc_watermark"] = aigc_watermark
    return payload


def _check_event(event: dict[str, Any]) -> tuple[int | None, str | None]:
    base_resp = event.get("base_resp") or {}
    status_code = base_resp.get("status_code")
    if status_code not in (None, 0):
        message = base_resp.get("status_msg") or "unknown API error"
        raise MiniMaxMusicError(f"MiniMax API error {status_code}: {message}")

    data = event.get("data") or {}
    return data.get("status"), data.get("audio")


def _parse_json_response(response: requests.Response) -> tuple[str, dict[str, Any]]:
    try:
        event = response.json()
    except ValueError as exc:
        raise MiniMaxMusicError("MiniMax returned invalid JSON") from exc
    if not isinstance(event, dict):
        raise MiniMaxMusicError("MiniMax returned an invalid response object")
    status, audio = _check_event(event)
    if status != 2:
        raise MiniMaxMusicError(f"Music generation did not complete (status={status!r})")
    if not isinstance(audio, str) or not audio:
        raise MiniMaxMusicError("Completed response did not include audio")
    return audio, event


def _parse_stream_response(response: requests.Response) -> tuple[bytes, dict[str, Any]]:
    chunks: list[bytes] = []
    last_event: dict[str, Any] = {}
    completed = False

    for raw_line in response.iter_lines(decode_unicode=True):
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MiniMaxMusicError("MiniMax returned an invalid streaming event") from exc
        if not isinstance(event, dict):
            raise MiniMaxMusicError("MiniMax returned an invalid streaming event object")
        status, audio = _check_event(event)
        if audio:
            try:
                chunks.append(bytes.fromhex(audio))
            except ValueError as exc:
                raise MiniMaxMusicError("MiniMax returned invalid hex audio") from exc
        if status == 2:
            completed = True
        last_event = event

    if not completed:
        raise MiniMaxMusicError("Music generation stream ended before completion")
    if not chunks:
        raise MiniMaxMusicError("Completed stream did not include audio")
    return b"".join(chunks), last_event


def generate_music(
    *,
    api_key: str,
    output: str | Path,
    region: str = "global_en",
    timeout: float = 600,
    session: requests.Session | None = None,
    **payload_args: Any,
) -> dict[str, Any]:
    """Call MiniMax, save the returned audio, and return machine-readable metadata."""
    if not api_key:
        raise ValueError("A MiniMax API key is required")
    payload = build_payload(region=region, **payload_args)
    client = session or requests.Session()
    try:
        response = client.post(
            ENDPOINTS[region],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            stream=payload["stream"],
            timeout=timeout,
        )
        response.raise_for_status()
        if payload["stream"]:
            audio_bytes, raw_response = _parse_stream_response(response)
        else:
            audio, raw_response = _parse_json_response(response)
            if payload["output_format"] == "hex":
                try:
                    audio_bytes = bytes.fromhex(audio)
                except ValueError as exc:
                    raise MiniMaxMusicError("MiniMax returned invalid hex audio") from exc
            else:
                if not audio.startswith("https://"):
                    raise MiniMaxMusicError("MiniMax returned an invalid audio URL")
                download = client.get(audio, timeout=timeout)
                download.raise_for_status()
                audio_bytes = download.content
    except requests.RequestException as exc:
        raise MiniMaxMusicError(f"MiniMax request failed: {exc}") from exc

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)
    return {
        "success": True,
        "output": str(output_path),
        "bytes": len(audio_bytes),
        "region": region,
        "model": payload["model"],
        "audio_format": payload["audio_setting"]["format"],
        "output_format": payload["output_format"],
        "stream": payload["stream"],
        "status": (raw_response.get("data") or {}).get("status"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate music with MiniMax")
    parser.add_argument("--prompt", help="Style, mood, and scenario description")
    parser.add_argument("--lyrics", help="Song lyrics, including optional section tags")
    parser.add_argument("--model", choices=MODELS, default="music-3.0")
    parser.add_argument("--region", choices=tuple(ENDPOINTS), default="global_en")
    parser.add_argument("--stream", action="store_true", help="Stream hex audio chunks")
    parser.add_argument("--output-format", choices=OUTPUT_FORMATS, default="hex")
    parser.add_argument("--format", dest="audio_format", choices=AUDIO_FORMATS, default="mp3")
    parser.add_argument("--sample-rate", type=int, choices=SAMPLE_RATES, default=44100)
    parser.add_argument("--bitrate", type=int, choices=BITRATES, default=256000)
    parser.add_argument("--lyrics-optimizer", action="store_true")
    parser.add_argument("--instrumental", action="store_true")
    parser.add_argument(
        "--aigc-watermark",
        action="store_true",
        default=None,
        help="Add the mainland China AIGC watermark (cn_zh, non-streaming only)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output audio path")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("MINIMAX_API_KEY", "")
    try:
        result = generate_music(
            api_key=api_key,
            output=args.output,
            region=args.region,
            timeout=args.timeout,
            model=args.model,
            prompt=args.prompt,
            lyrics=args.lyrics,
            stream=args.stream,
            output_format=args.output_format,
            sample_rate=args.sample_rate,
            bitrate=args.bitrate,
            audio_format=args.audio_format,
            lyrics_optimizer=args.lyrics_optimizer,
            is_instrumental=args.instrumental,
            aigc_watermark=args.aigc_watermark,
        )
    except (MiniMaxMusicError, ValueError) as exc:
        if args.json:
            print(json.dumps({"success": False, "error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2) if args.json else f"Music saved to: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
