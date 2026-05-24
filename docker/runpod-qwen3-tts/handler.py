#!/usr/bin/env python3
"""
RunPod serverless handler for Qwen3-TTS.

Supports three modes:
  custom_voice  — built-in speaker + natural-language `instruct` direction
  clone         — ICL voice clone from ref_audio + ref_text
  voice_design  — generate a new character voice from a natural-language
                  `instruct` brief (returns a designed .wav usable as a
                  clone reference)

All three modes accept either a scalar `text` (single generation) or a
list of strings (batch). Batch clone is the important one: the
`voice_clone_prompt` is extracted once from the ref_audio and reused
across every utterance, which gives consistent voice across many
scenes — the "proper" way to use Qwen3-TTS for long-form narration.

Input format:
{
    "input": {
        # Universal
        "text":        str | list[str],   # required; list → batch
        "language":    str | list[str],   # default "Auto"; scalar broadcasts
        "output_format": "mp3" | "wav",   # default "mp3"
        "temperature": float,             # default 0.7
        "top_p":       float,             # default 0.8

        # Mode selection
        "mode": "custom_voice" | "clone" | "voice_design",   # default "custom_voice"

        # custom_voice
        "speaker":  str,                  # e.g. "Ryan", "Aiden"
        "instruct": str,                  # style direction (optional)

        # clone
        "ref_audio_url":    str,          # OR
        "ref_audio_base64": str,
        "ref_text":         str,          # required for ICL clone

        # voice_design
        "instruct": str,                  # character brief, required
        # text is reused as the seed sentence to speak for the designed voice

        # R2 (optional; else base64 returned)
        "r2": {
            "endpoint_url":      str,
            "access_key_id":     str,
            "secret_access_key": str,
            "bucket_name":       str
        }
    }
}

Output format:
{
    "success": true,
    "mode": str,                         # echoed
    "processing_time_seconds": float,
    "outputs": [
        {
            "audio_url":          str,   # if R2 configured
            "r2_key":             str,
            "audio_base64":       str,   # if no R2
            "duration_seconds":   float
        },
        ...
    ],
    # Legacy top-level keys also populated when the request was single-text
    # (text was a string, not a list). Facilitates backward compat.
    "audio_url":        str,
    "r2_key":           str,
    "audio_base64":     str,
    "duration_seconds": float
}
"""

import base64
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import runpod
import requests
import soundfile as sf

# Lazy-loaded global models (kept in GPU memory between requests)
_custom_voice_model = None
_base_model = None
_voice_design_model = None


def log(message: str) -> None:
    """Log message to stderr (visible in RunPod logs)."""
    print(message, file=sys.stderr, flush=True)


# ─── Model loaders ──────────────────────────────────────────

def get_custom_voice_model():
    """Lazy-load CustomVoice model (built-in speakers + instruct)."""
    global _custom_voice_model
    if _custom_voice_model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        log("Loading CustomVoice model...")
        _custom_voice_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        log("CustomVoice model loaded")
    return _custom_voice_model


def get_base_model():
    """Lazy-load Base model (voice cloning)."""
    global _base_model
    if _base_model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        log("Loading Base model...")
        _base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        log("Base model loaded")
    return _base_model


def get_voice_design_model():
    """Lazy-load VoiceDesign model (generate voice from natural-language brief)."""
    global _voice_design_model
    if _voice_design_model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        log("Loading VoiceDesign model...")
        _voice_design_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        log("VoiceDesign model loaded")
    return _voice_design_model


# ─── I/O helpers ────────────────────────────────────────────

def download_file(url: str, output_path: Path, timeout: int = 300) -> bool:
    try:
        log(f"Downloading from {url[:80]}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        log(f"  Downloaded: {output_path.name} ({output_path.stat().st_size // 1024}KB)")
        return True
    except Exception as e:
        log(f"Download error: {e}")
        return False


def decode_base64_file(data: str, output_path: Path) -> bool:
    try:
        if "," in data:
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        output_path.write_bytes(decoded)
        log(f"Decoded base64 to {output_path.name} ({len(decoded) // 1024}KB)")
        return True
    except Exception as e:
        log(f"Base64 decode error: {e}")
        return False


def encode_file_base64(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def get_audio_duration(audio_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        log(f"Error getting audio duration: {e}")
        return 0.0


def wav_to_mp3(wav_path: Path, mp3_path: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", "192k",
                str(mp3_path),
            ],
            capture_output=True, timeout=120, check=True,
        )
        return mp3_path.exists()
    except Exception as e:
        log(f"WAV to MP3 conversion error: {e}")
        return False


def upload_to_r2(file_path: Path, job_id: str, r2_config: dict, content_type: str) -> tuple[Optional[str], Optional[str]]:
    """Upload audio to Cloudflare R2 and return (presigned_url, object_key)."""
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
        )
        ext = file_path.suffix
        object_key = f"qwen3-tts/results/{job_id}_{uuid.uuid4().hex[:8]}{ext}"
        client.upload_file(
            str(file_path),
            r2_config["bucket_name"],
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
        presigned_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
            ExpiresIn=7200,
        )
        return presigned_url, object_key
    except Exception as e:
        log(f"R2 upload error: {e}")
        return None, None


# ─── Generation helpers ─────────────────────────────────────

def _as_list(value, length: int):
    """Return value as a list of `length` — scalar broadcast, or verified same length."""
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"list length mismatch: got {len(value)}, need {length}")
        return value
    return [value] * length


def generate_custom_voice_batch(texts: list[str], speaker: str, languages: list[str], instruct: str = "", **kwargs):
    """Generate one or more utterances with built-in speaker. Returns (list_of_wavs, sr)."""
    model = get_custom_voice_model()

    gen_kwargs = {
        "text": texts if len(texts) > 1 else texts[0],
        "language": languages if len(texts) > 1 else languages[0],
        "speaker": speaker,
    }
    if instruct:
        gen_kwargs["instruct"] = instruct
    if kwargs:
        gen_kwargs.update(kwargs)

    wavs, sr = model.generate_custom_voice(**gen_kwargs)
    # Normalize: wavs is list for batch, single array for scalar
    if len(texts) == 1 and not isinstance(wavs, list):
        wavs = [wavs]
    return list(wavs), sr


def generate_clone_voice_batch(texts: list[str], languages: list[str], ref_audio_path: Path, ref_text: str, **kwargs):
    """Generate one or more utterances with a single shared voice_clone_prompt.

    The prompt is extracted ONCE from the reference and reused across all
    texts. Note: qwen-tts's `generate_voice_clone(text=[...])` batch path
    hits a PyTorch tensor-aliasing bug ("written-to tensor refers to a
    single memory location"), so we serialize the calls while still
    sharing the pre-computed prompt — which is what gives us consistency.
    """
    model = get_base_model()

    log(f"Extracting voice_clone_prompt from reference...")
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
    )

    log(f"Generating {len(texts)} utterance(s) with shared clone prompt...")
    wavs = []
    sr = None
    for i, (t, lang) in enumerate(zip(texts, languages)):
        single_gen_kwargs = {
            "text": t,
            "language": lang,
            "voice_clone_prompt": prompt,
        }
        if kwargs:
            single_gen_kwargs.update(kwargs)
        single_wavs, single_sr = model.generate_voice_clone(**single_gen_kwargs)
        wav = single_wavs[0] if isinstance(single_wavs, list) else single_wavs
        wavs.append(wav)
        sr = single_sr
        log(f"  [{i + 1}/{len(texts)}] generated")

    return wavs, sr


def generate_voice_design(text: str, language: str, instruct: str, **kwargs):
    """Generate a designed character voice from a natural-language brief.

    Returns a single ([wav], sr). The designed wav is typically used as a
    reference for subsequent clone calls to produce consistent character
    voice across many utterances.
    """
    model = get_voice_design_model()

    if not instruct:
        raise ValueError("voice_design mode requires an 'instruct' brief")

    gen_kwargs = {
        "text": text,
        "language": language,
        "instruct": instruct,
    }
    if kwargs:
        gen_kwargs.update(kwargs)

    log(f"Designing voice (instruct={instruct[:60]}...)")
    wavs, sr = model.generate_voice_design(**gen_kwargs)
    if not isinstance(wavs, list):
        wavs = [wavs]
    return list(wavs), sr


# ─── Handler ────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """Main RunPod handler for Qwen3-TTS."""
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    log(f"Job {job_id}: Starting Qwen3-TTS")

    # Required
    text_in = job_input.get("text")
    if not text_in:
        return {"error": "Missing required field: text"}

    # Normalize text to list — remember whether original was scalar
    is_batch = isinstance(text_in, list)
    texts = text_in if is_batch else [text_in]
    if not texts or any(not t for t in texts):
        return {"error": "text list must contain non-empty strings"}

    # Options
    mode = job_input.get("mode", "custom_voice")
    language = job_input.get("language", "Auto")
    output_format = job_input.get("output_format", "mp3")
    r2_config = job_input.get("r2")

    try:
        languages = _as_list(language, len(texts))
    except ValueError as e:
        return {"error": str(e)}

    # Generation kwargs (optional)
    gen_kwargs = {}
    if "temperature" in job_input:
        gen_kwargs["temperature"] = float(job_input["temperature"])
    if "top_p" in job_input:
        gen_kwargs["top_p"] = float(job_input["top_p"])

    work_dir = Path(tempfile.mkdtemp(prefix=f"qwen3tts_{job_id}_"))
    log(f"Working directory: {work_dir}")

    try:
        if mode == "clone":
            ref_audio_path = work_dir / "ref_audio.wav"
            ref_text = job_input.get("ref_text")
            if not ref_text:
                return {"error": "ref_text is required for clone mode"}

            if job_input.get("ref_audio_url"):
                if not download_file(job_input["ref_audio_url"], ref_audio_path):
                    return {"error": "Failed to download reference audio"}
            elif job_input.get("ref_audio_base64"):
                if not decode_base64_file(job_input["ref_audio_base64"], ref_audio_path):
                    return {"error": "Failed to decode reference audio"}
            else:
                return {"error": "ref_audio_url or ref_audio_base64 required for clone mode"}

            wavs, sr = generate_clone_voice_batch(
                texts=texts,
                languages=languages,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                **gen_kwargs,
            )

        elif mode == "voice_design":
            if is_batch:
                return {"error": "voice_design mode expects a single seed text, not a list"}
            instruct = job_input.get("instruct")
            if not instruct:
                return {"error": "voice_design mode requires 'instruct'"}

            wavs, sr = generate_voice_design(
                text=texts[0],
                language=languages[0],
                instruct=instruct,
                **gen_kwargs,
            )

        else:
            # custom_voice (default)
            speaker = job_input.get("speaker", "Ryan")
            instruct = job_input.get("instruct", "")
            wavs, sr = generate_custom_voice_batch(
                texts=texts,
                speaker=speaker,
                languages=languages,
                instruct=instruct,
                **gen_kwargs,
            )

        # Write, convert, (upload|encode) per-utterance
        outputs = []
        for i, wav_data in enumerate(wavs):
            base_name = f"output_{i}" if len(wavs) > 1 else "output"
            wav_path = work_dir / f"{base_name}.wav"
            sf.write(str(wav_path), wav_data, sr)

            if output_format == "mp3":
                out_path = work_dir / f"{base_name}.mp3"
                if not wav_to_mp3(wav_path, out_path):
                    return {"error": f"Failed to convert WAV to MP3 (index {i})"}
                content_type = "audio/mpeg"
            else:
                out_path = wav_path
                content_type = "audio/wav"

            duration = get_audio_duration(out_path)
            item = {"duration_seconds": round(duration, 2)}

            if r2_config:
                url, r2_key = upload_to_r2(out_path, f"{job_id}_{i}", r2_config, content_type)
                if not url:
                    return {"error": f"Failed to upload to R2 (index {i})"}
                item["audio_url"] = url
                item["r2_key"] = r2_key
            else:
                item["audio_base64"] = encode_file_base64(out_path)

            outputs.append(item)

        elapsed = time.time() - start_time
        log(f"Generated {len(outputs)} output(s) in {elapsed:.1f}s")

        result = {
            "success": True,
            "mode": mode,
            "processing_time_seconds": round(elapsed, 2),
            "outputs": outputs,
        }
        # Legacy single-output compat: surface first output's fields at top level
        # for non-batch calls. Batch callers should use `outputs`.
        if not is_batch and outputs:
            for k in ("audio_url", "r2_key", "audio_base64", "duration_seconds"):
                if k in outputs[0]:
                    result[k] = outputs[0][k]

        return result

    except Exception as e:
        import traceback
        log(f"Handler exception: {e}")
        log(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            log("Cleaned up working directory")
        except Exception:
            pass


# RunPod serverless entry point
if __name__ == "__main__":
    log("Starting RunPod Qwen3-TTS handler (v2: voice_design + batch)...")

    try:
        import torch
        if torch.cuda.is_available():
            log(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            log("WARNING: CUDA not available!")
    except ImportError:
        log("Warning: torch not imported for CUDA check")

    runpod.serverless.start({"handler": handler})
