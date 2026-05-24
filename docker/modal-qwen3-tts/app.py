"""
Modal deployment for Qwen3-TTS.

Mirrors docker/runpod-qwen3-tts/handler.py — supports the same three modes
(custom_voice, clone, voice_design) and the same batch protocol (pass a
list of strings as `text` to share one voice_clone_prompt across many
utterances).

Deploy:
    modal deploy docker/modal-qwen3-tts/app.py

See the RunPod handler for full payload/response documentation.
"""

import modal

app = modal.App("video-toolkit-qwen3-tts")

# Container image — mirrors docker/runpod-qwen3-tts/Dockerfile.
# Three models baked: CustomVoice + Base + VoiceDesign (~10GB total weights)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "qwen-tts",
        "soundfile",
        "boto3",
        "requests",
        "fastapi[standard]",
    )
    .run_commands(
        'python -c "'
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'); "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base'); "
        "snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')"
        '"'
    )
)


def _as_list(value, length: int):
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"list length mismatch: got {len(value)}, need {length}")
        return value
    return [value] * length


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
class Qwen3TTS:
    """Three models, lazy-loaded. Persisted across requests on warm containers."""

    @modal.enter()
    def load_base_models(self):
        """Load the two most-used models eagerly (CustomVoice + Base).

        VoiceDesign is lazy-loaded on first use to keep cold-start fast for
        the common case (narration batches in clone mode).
        """
        import torch
        from qwen_tts import Qwen3TTSModel

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("Loading CustomVoice model...")
        self.custom_voice_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("CustomVoice model loaded")

        print("Loading Base model...")
        self.base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("Base model loaded")

        self._voice_design_model = None

    def _get_voice_design(self):
        """Lazy-load VoiceDesign model on first use."""
        if self._voice_design_model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            print("Loading VoiceDesign model...")
            self._voice_design_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print("VoiceDesign model loaded")
        return self._voice_design_model

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: dict) -> dict:
        """Web endpoint — accepts same payload format as RunPod handler.

        See docker/runpod-qwen3-tts/handler.py for full protocol docs.
        """
        import base64
        import shutil
        import subprocess
        import tempfile
        import time
        import traceback
        import uuid
        from pathlib import Path

        import requests as req
        import soundfile as sf

        start_time = time.time()

        # Required
        text_in = request.get("text")
        if not text_in:
            return {"error": "Missing required field: text"}

        is_batch = isinstance(text_in, list)
        texts = text_in if is_batch else [text_in]
        if not texts or any(not t for t in texts):
            return {"error": "text list must contain non-empty strings"}

        mode = request.get("mode", "custom_voice")
        language = request.get("language", "Auto")
        output_format = request.get("output_format", "mp3")
        r2_config = request.get("r2")

        try:
            languages = _as_list(language, len(texts))
        except ValueError as e:
            return {"error": str(e)}

        gen_kwargs = {}
        if "temperature" in request:
            gen_kwargs["temperature"] = float(request["temperature"])
        if "top_p" in request:
            gen_kwargs["top_p"] = float(request["top_p"])

        job_id = uuid.uuid4().hex[:12]
        work_dir = Path(tempfile.mkdtemp(prefix=f"qwen3tts_{job_id}_"))

        try:
            # --- Dispatch by mode ---
            if mode == "clone":
                ref_audio_path = work_dir / "ref_audio.wav"
                ref_text = request.get("ref_text")
                if not ref_text:
                    return {"error": "ref_text is required for clone mode"}

                if request.get("ref_audio_url"):
                    resp = req.get(request["ref_audio_url"], timeout=300)
                    resp.raise_for_status()
                    ref_audio_path.write_bytes(resp.content)
                elif request.get("ref_audio_base64"):
                    data = request["ref_audio_base64"]
                    if "," in data:
                        data = data.split(",", 1)[1]
                    ref_audio_path.write_bytes(base64.b64decode(data))
                else:
                    return {"error": "ref_audio_url or ref_audio_base64 required for clone mode"}

                # Extract voice_clone_prompt ONCE — reused across all texts.
                # Note: batching via a list in `generate_voice_clone(text=[...])`
                # hits a tensor-aliasing bug in qwen-tts (PyTorch "written-to
                # tensor refers to a single memory location"). We work around
                # it by iterating one text at a time while still sharing the
                # pre-computed prompt — which is what gives us consistency.
                print(f"Extracting voice_clone_prompt from reference...")
                prompt = self.base_model.create_voice_clone_prompt(
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                )
                print(f"Generating {len(texts)} utterance(s) with shared clone prompt...")
                wavs = []
                sr = None
                for i, (t, lang) in enumerate(zip(texts, languages)):
                    single_wavs, single_sr = self.base_model.generate_voice_clone(
                        text=t,
                        language=lang,
                        voice_clone_prompt=prompt,
                        **gen_kwargs,
                    )
                    if isinstance(single_wavs, list):
                        wav = single_wavs[0]
                    else:
                        wav = single_wavs
                    wavs.append(wav)
                    sr = single_sr
                    print(f"  [{i + 1}/{len(texts)}] generated")

            elif mode == "voice_design":
                if is_batch:
                    return {"error": "voice_design mode expects a single seed text, not a list"}
                instruct = request.get("instruct")
                if not instruct:
                    return {"error": "voice_design mode requires 'instruct'"}

                design_model = self._get_voice_design()
                print(f"Designing voice (instruct={instruct[:60]}...)")
                wavs, sr = design_model.generate_voice_design(
                    text=texts[0],
                    language=languages[0],
                    instruct=instruct,
                    **gen_kwargs,
                )
                if not isinstance(wavs, list):
                    wavs = [wavs]
                wavs = list(wavs)

            else:
                # custom_voice (default)
                speaker = request.get("speaker", "Ryan")
                instruct = request.get("instruct", "")

                cv_kwargs = {
                    "text": texts if len(texts) > 1 else texts[0],
                    "language": languages if len(texts) > 1 else languages[0],
                    "speaker": speaker,
                }
                if instruct:
                    cv_kwargs["instruct"] = instruct
                cv_kwargs.update(gen_kwargs)

                wavs, sr = self.custom_voice_model.generate_custom_voice(**cv_kwargs)
                if len(texts) == 1 and not isinstance(wavs, list):
                    wavs = [wavs]
                wavs = list(wavs)

            # --- Encode, upload (or base64), per utterance ---
            outputs = []
            for i, wav_data in enumerate(wavs):
                base_name = f"output_{i}" if len(wavs) > 1 else "output"
                wav_path = work_dir / f"{base_name}.wav"
                sf.write(str(wav_path), wav_data, sr)

                if output_format == "mp3":
                    out_path = work_dir / f"{base_name}.mp3"
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(wav_path),
                         "-codec:a", "libmp3lame", "-b:a", "192k",
                         str(out_path)],
                        capture_output=True, timeout=120, check=True,
                    )
                    content_type = "audio/mpeg"
                else:
                    out_path = wav_path
                    content_type = "audio/wav"

                try:
                    dur_result = subprocess.run(
                        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
                        capture_output=True, text=True, timeout=30,
                    )
                    duration = float(dur_result.stdout.strip())
                except Exception:
                    duration = 0.0

                item = {"duration_seconds": round(duration, 2)}

                if r2_config:
                    import boto3
                    from botocore.config import Config

                    client = boto3.client(
                        "s3",
                        endpoint_url=r2_config["endpoint_url"],
                        aws_access_key_id=r2_config["access_key_id"],
                        aws_secret_access_key=r2_config["secret_access_key"],
                        config=Config(signature_version="s3v4"),
                    )
                    ext = out_path.suffix
                    object_key = f"qwen3-tts/results/{job_id}_{i}_{uuid.uuid4().hex[:8]}{ext}"
                    client.upload_file(
                        str(out_path), r2_config["bucket_name"], object_key,
                        ExtraArgs={"ContentType": content_type},
                    )
                    presigned_url = client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": r2_config["bucket_name"], "Key": object_key},
                        ExpiresIn=7200,
                    )
                    item["audio_url"] = presigned_url
                    item["r2_key"] = object_key
                else:
                    item["audio_base64"] = base64.b64encode(out_path.read_bytes()).decode("utf-8")

                outputs.append(item)

            elapsed = time.time() - start_time
            result = {
                "success": True,
                "mode": mode,
                "processing_time_seconds": round(elapsed, 2),
                "outputs": outputs,
            }
            if not is_batch and outputs:
                for k in ("audio_url", "r2_key", "audio_base64", "duration_seconds"):
                    if k in outputs[0]:
                        result[k] = outputs[0][k]
            return result

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return {"error": f"Internal error: {str(e)}"}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
