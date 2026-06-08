#!/usr/bin/env python3
"""
Generate speech using 60db (https://60db.ai) — a premium cloud TTS provider.

This is the 60db counterpart to ElevenLabs (`voiceover.py`) and Qwen3-TTS
(`qwen3_tts.py`). It exposes a `generate_audio()` function used by
`voiceover.py` (as `--provider 60db`) and `redub.py`, plus a standalone CLI.

Three transports are supported (all produce a finished audio file):
  - synthesize (default): POST /tts-synthesize  -> JSON {audio_base64}
  - stream:               POST /tts-stream       -> NDJSON audio chunks
  - websocket:            wss://api.60db.ai/ws/tts -> context protocol

Voice settings use a UNIFIED 0-1 scale (same as ElevenLabs in this toolkit).
They are converted to 60db's native 0-100 scale internally.

Usage:
    # Quick generation (REST, default voice)
    python tools/sixtydb_tts.py --text "Hello world" --output hello.mp3

    # Pick a voice and tune settings (0-1 scale)
    python tools/sixtydb_tts.py --text "Hello" --voice-id <uuid> \
        --stability 0.6 --similarity 0.9 --speed 1.0 --output hello.mp3

    # Streaming transport
    python tools/sixtydb_tts.py --text "Hello" --transport stream --output hello.mp3

    # Realtime websocket transport (writes a WAV, transcoded to --output format)
    python tools/sixtydb_tts.py --text "Hello" --transport websocket --output hello.mp3

    # List your voices (GET /myvoices)
    python tools/sixtydb_tts.py --list-voices

Setup:
    echo "SIXTYDB_API_KEY=sk_live_your_key" >> .env
    # Optional default voice:
    echo "SIXTYDB_VOICE_ID=<uuid>" >> .env
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import get_sixtydb_api_key, get_sixtydb_voice_id

# --- 60db endpoints ---
SIXTYDB_BASE_URL = "https://api.60db.ai"
SIXTYDB_SYNTHESIZE_URL = f"{SIXTYDB_BASE_URL}/tts-synthesize"
SIXTYDB_STREAM_URL = f"{SIXTYDB_BASE_URL}/tts-stream"
SIXTYDB_VOICES_URL = f"{SIXTYDB_BASE_URL}/myvoices"
SIXTYDB_WS_URL = "wss://api.60db.ai/ws/tts"

# Documented default voice (used when no voice id is configured anywhere).
DEFAULT_VOICE_ID = "fbb75ed2-975a-40c7-9e06-38e30524a9a1"

SUPPORTED_FORMATS = ["mp3", "wav", "ogg", "flac"]
SUPPORTED_TRANSPORTS = ["synthesize", "stream", "websocket"]

# Hard limits from the 60db docs.
MAX_TEXT_CHARS = 5000          # per REST request
MAX_WS_BUFFER_CHARS = 50000    # cumulative per websocket context


def get_audio_duration(file_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe (if available)."""
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


def _unit_to_100(value: float | None, default_100: int) -> int:
    """Convert a unified 0-1 setting to 60db's native 0-100 scale.

    Values are clamped to [0, 100]. A None value yields the 60db default.
    Values already > 1 are assumed to be on the native 0-100 scale and are
    passed through (clamped) — this keeps the tool forgiving if a caller
    hands us 75 instead of 0.75.
    """
    if value is None:
        return default_100
    scaled = value * 100 if value <= 1 else value
    return max(0, min(100, int(round(scaled))))


def list_voices(api_key: str, timeout: int = 30) -> dict:
    """Fetch the caller's voices from GET /myvoices.

    Returns {"success": True, "voices": [...]} or {"success": False, "error": ...}.
    Each voice dict has at least: voice_id, name, category, model, labels.
    """
    try:
        resp = requests.get(
            SIXTYDB_VOICES_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    try:
        data = resp.json()
    except ValueError:
        return {"success": False, "error": f"Invalid JSON response: {resp.text[:300]}"}

    if not data.get("success", True):
        return {"success": False, "error": data.get("message", "Request unsuccessful")}

    return {"success": True, "voices": data.get("data", [])}


def _write_pcm_as_audio(pcm_bytes: bytes, sample_rate: int, output_path: str) -> bool:
    """Write raw 16-bit mono PCM to a WAV, transcoding to the target format.

    The websocket transport returns LINEAR16 PCM chunks. We wrap them in a WAV
    header. If the requested output is not .wav, we transcode with ffmpeg.
    """
    import wave

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".wav":
        wav_path = str(out)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    if wav_path == str(out):
        return True

    # Transcode WAV -> requested format
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, str(out)],
        capture_output=True,
        text=True,
    )
    Path(wav_path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"ffmpeg transcode error: {result.stderr[-400:]}", file=sys.stderr)
        return False
    return True


def _synthesize_rest(
    text: str, voice_id: str, stability: int, similarity: int,
    speed: float, enhance: bool, output_format: str, api_key: str,
    output_path: str, timeout: int, verbose: bool,
) -> dict:
    """POST /tts-synthesize — single JSON response with base64 audio."""
    body = {
        "text": text,
        "voice_id": voice_id,
        "enhance": enhance,
        "speed": speed,
        "stability": stability,
        "similarity": similarity,
        "output_format": output_format,
    }
    try:
        resp = requests.post(
            SIXTYDB_SYNTHESIZE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    try:
        data = resp.json()
    except ValueError:
        return {"success": False, "error": f"Invalid JSON response: {resp.text[:300]}"}

    if not data.get("success", True):
        return {"success": False, "error": data.get("message", "Synthesis unsuccessful")}

    audio_b64 = data.get("audio_base64")
    if not audio_b64:
        return {"success": False, "error": f"No audio_base64 in response: {list(data.keys())}"}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(audio_b64))
    return {"success": True}


def _synthesize_stream(
    text: str, voice_id: str, stability: int, similarity: int,
    speed: float, enhance: bool, api_key: str,
    output_path: str, timeout: int, verbose: bool,
) -> dict:
    """POST /tts-stream — NDJSON chunks, each carrying a base64 audio slice."""
    body = {
        "text": text,
        "voice_id": voice_id,
        "enhance": enhance,
        "speed": speed,
        "stability": stability,
        "similarity": similarity,
    }
    try:
        resp = requests.post(
            SIXTYDB_STREAM_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
            stream=True,
        )
    except requests.RequestException as e:
        return {"success": False, "error": f"Request failed: {e}"}

    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio = bytearray()
    chunk_count = 0

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue  # skip malformed line
        mtype = msg.get("type")
        if mtype == "error":
            return {"success": False, "error": msg.get("message", "stream error")}
        if mtype in ("chunk", "complete"):
            b64 = (msg.get("result") or {}).get("audioContent")
            if b64:
                audio.extend(base64.b64decode(b64))
                chunk_count += 1
        if mtype == "complete":
            break

    if not audio:
        return {"success": False, "error": "No audio received from stream"}

    out.write_bytes(bytes(audio))
    if verbose:
        print(f"  Received {chunk_count} audio chunk(s)", file=sys.stderr)
    return {"success": True}


def _synthesize_websocket(
    text: str, voice_id: str, stability: int, similarity: int,
    speed: float, sample_rate: int, api_key: str,
    output_path: str, timeout: int, verbose: bool,
) -> dict:
    """Realtime websocket transport. Collects LINEAR16 PCM, writes an audio file.

    Uses the `websocket-client` package (lazy import).
    """
    try:
        from websocket import create_connection  # websocket-client
    except ImportError:
        return {
            "success": False,
            "error": (
                "websocket transport requires the 'websocket-client' package.\n"
                "  Install it with: pip install websocket-client\n"
                "  Or use --transport synthesize (no extra dependency)."
            ),
        }

    context_id = "voiceover"
    url = f"{SIXTYDB_WS_URL}?apiKey={api_key}"
    try:
        ws = create_connection(url, timeout=timeout)
    except Exception as e:
        return {"success": False, "error": f"WebSocket connect failed: {e}"}

    pcm = bytearray()
    try:
        ws.send(json.dumps({
            "create_context": {
                "context_id": context_id,
                "voice_id": voice_id,
                "audio_config": {
                    "audio_encoding": "LINEAR16",
                    "sample_rate_hertz": sample_rate,
                },
                "speed": speed,
                "stability": stability,
                "similarity": similarity,
            }
        }))
        ws.send(json.dumps({"send_text": {"context_id": context_id, "text": text}}))
        ws.send(json.dumps({"close_context": {"context_id": context_id}}))

        while True:
            raw = ws.recv()
            if not raw:
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if "audio_chunk" in msg:
                b64 = msg["audio_chunk"].get("audioContent")
                if b64:
                    pcm.extend(base64.b64decode(b64))
            elif "error" in msg:
                return {"success": False, "error": msg["error"].get("message", "ws error")}
            elif "context_closed" in msg:
                break
    except Exception as e:
        return {"success": False, "error": f"WebSocket error: {e}"}
    finally:
        try:
            ws.close()
        except Exception:
            pass

    if not pcm:
        return {"success": False, "error": "No audio received from websocket"}

    if not _write_pcm_as_audio(bytes(pcm), sample_rate, output_path):
        return {"success": False, "error": "Failed to write/transcode websocket audio"}
    return {"success": True}


def generate_audio(
    text: str,
    output_path: str,
    voice_id: str | None = None,
    stability: float | None = 0.85,
    similarity: float | None = 0.95,
    speed: float = 1.0,
    enhance: bool = True,
    output_format: str = "mp3",
    transport: str = "synthesize",
    sample_rate: int = 24000,
    api_key: str | None = None,
    timeout: int = 120,
    verbose: bool = True,
) -> dict:
    """Generate a single audio file with 60db.

    `stability` / `similarity` are on a UNIFIED 0-1 scale and converted to
    60db's native 0-100 internally. `speed` (0.5-2.0) is the same on both.

    Returns dict with the same shape the other providers use:
      {success, output, duration_seconds, duration_frames_30fps, script_chars}
    or {success: False, error: ...}.
    """
    if not text:
        return {"success": False, "error": "text must be a non-empty string"}

    api_key = api_key or get_sixtydb_api_key()
    if not api_key:
        return {"success": False, "error": "No 60db API key (set SIXTYDB_API_KEY)"}

    voice_id = voice_id or get_sixtydb_voice_id() or DEFAULT_VOICE_ID

    if transport not in SUPPORTED_TRANSPORTS:
        return {"success": False, "error": f"Unknown transport: {transport}"}

    limit = MAX_WS_BUFFER_CHARS if transport == "websocket" else MAX_TEXT_CHARS
    if len(text) > limit:
        return {
            "success": False,
            "error": f"text exceeds {limit} character limit for transport '{transport}' "
                     f"({len(text)} chars). Split into smaller scenes.",
        }

    stab = _unit_to_100(stability, 50)
    sim = _unit_to_100(similarity, 75)

    if verbose:
        print(
            f"60db: voice={voice_id} transport={transport} "
            f"stability={stab} similarity={sim} speed={speed}",
            file=sys.stderr,
        )

    if transport == "synthesize":
        result = _synthesize_rest(
            text, voice_id, stab, sim, speed, enhance, output_format,
            api_key, output_path, timeout, verbose,
        )
    elif transport == "stream":
        result = _synthesize_stream(
            text, voice_id, stab, sim, speed, enhance,
            api_key, output_path, timeout, verbose,
        )
    else:  # websocket
        result = _synthesize_websocket(
            text, voice_id, stab, sim, speed, sample_rate,
            api_key, output_path, timeout, verbose,
        )

    if not result.get("success"):
        return result

    duration = get_audio_duration(output_path)
    out = {
        "success": True,
        "output": output_path,
        "script_chars": len(text),
        "voice_id": voice_id,
        "transport": transport,
    }
    if duration:
        out["duration_seconds"] = round(duration, 2)
        out["duration_frames_30fps"] = int(duration * 30)
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate speech using 60db (cloud TTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/sixtydb_tts.py --text "Hello world" --output hello.mp3
  python tools/sixtydb_tts.py --text "Hello" --voice-id <uuid> --stability 0.6 --output hello.mp3
  python tools/sixtydb_tts.py --text "Hello" --transport stream --output hello.mp3
  python tools/sixtydb_tts.py --list-voices
        """,
    )
    parser.add_argument("--text", "-t", type=str, help="Text to synthesize (max 5000 chars for REST)")
    parser.add_argument("--output", "-o", type=str, help="Output audio file path")
    parser.add_argument("--voice-id", "-v", type=str, help="60db voice ID (defaults to SIXTYDB_VOICE_ID or the 60db default voice)")
    parser.add_argument("--stability", type=float, default=0.85, help="Voice stability 0-1 (default: 0.85). Lower = more expressive.")
    parser.add_argument("--similarity", type=float, default=0.95, help="Similarity 0-1 (default: 0.95). Higher = closer to source voice.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed 0.5-2.0 (default: 1.0)")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false", help="Disable 60db audio enhancement (on by default)")
    parser.set_defaults(enhance=True)
    parser.add_argument("--output-format", type=str, default="mp3", choices=SUPPORTED_FORMATS, help="Audio format (default: mp3). REST/synthesize only.")
    parser.add_argument("--transport", type=str, default="synthesize", choices=SUPPORTED_TRANSPORTS, help="API transport (default: synthesize)")
    parser.add_argument("--sample-rate", type=int, default=24000, choices=[8000, 16000, 24000, 48000], help="Sample rate for websocket transport (default: 24000)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--list-voices", action="store_true", help="List your 60db voices and exit")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without calling the API")
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.json

    api_key = get_sixtydb_api_key()

    if args.list_voices:
        if not api_key:
            print("Error: No 60db API key found. Add SIXTYDB_API_KEY to .env", file=sys.stderr)
            sys.exit(1)
        res = list_voices(api_key)
        if not res.get("success"):
            print(f"Error: {res['error']}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            voices = res["voices"]
            print(f"Your 60db voices ({len(voices)}):\n")
            print(f"  {'voice_id':<38} {'name':<22} {'category':<12} {'lang'}")
            print(f"  {'-'*38} {'-'*22} {'-'*12} {'-'*6}")
            for v in voices:
                labels = v.get("labels") or {}
                lang = labels.get("language", "")
                print(f"  {v.get('voice_id',''):<38} {(v.get('name') or '')[:22]:<22} "
                      f"{(v.get('category') or '')[:12]:<12} {lang}")
        sys.exit(0)

    if not args.text:
        print("Error: --text is required", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: --output is required", file=sys.stderr)
        sys.exit(1)

    voice_id = args.voice_id or get_sixtydb_voice_id() or DEFAULT_VOICE_ID

    if args.dry_run:
        result = {
            "dry_run": True,
            "provider": "60db",
            "text_chars": len(args.text),
            "output": args.output,
            "voice_id": voice_id,
            "transport": args.transport,
            "settings": {
                "stability": args.stability,
                "similarity": args.similarity,
                "speed": args.speed,
                "enhance": args.enhance,
                "output_format": args.output_format,
            },
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Would generate speech with 60db:")
            print(f"  Voice ID: {voice_id}")
            print(f"  Transport: {args.transport}")
            print(f"  Text: {len(args.text)} characters")
            print(f"  Output: {args.output}")
        sys.exit(0)

    if not api_key:
        print(
            "Error: No 60db API key found.\n"
            "  echo \"SIXTYDB_API_KEY=sk_live_your_key\" >> .env",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"Generating speech with 60db ({len(args.text)} chars)...", file=sys.stderr)

    result = generate_audio(
        text=args.text,
        output_path=args.output,
        voice_id=voice_id,
        stability=args.stability,
        similarity=args.similarity,
        speed=args.speed,
        enhance=args.enhance,
        output_format=args.output_format,
        transport=args.transport,
        sample_rate=args.sample_rate,
        api_key=api_key,
        timeout=args.timeout,
        verbose=verbose,
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
        print(f"Generated: {result['output']}", file=sys.stderr)
        duration = result.get("duration_seconds")
        if duration:
            print(f"  Duration: {duration:.1f}s ({int(duration * 30)} frames @ 30fps)", file=sys.stderr)


if __name__ == "__main__":
    main()
