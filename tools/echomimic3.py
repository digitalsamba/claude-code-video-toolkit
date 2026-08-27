#!/usr/bin/env python3
"""
Generate talking head videos using EchoMimicV3-Flash.

Candidate replacement for tools/sadtalker.py. EchoMimicV3 (Ant Group, Apache 2.0,
1.3B params) is a Wan2.1-Fun-based audio-driven human animation model. Unlike
SadTalker it preserves the input image's aspect ratio, so 16:9 presenter images
come back 16:9 with no --preprocess workaround.

Usage:
    # Basic
    uv run tools/echomimic3.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

    # NarratorPiP settings (16:9 input, cheaper 5-step pass)
    uv run tools/echomimic3.py \
        --image presenter_16x9.png --audio scene_01.mp3 \
        --steps 5 --size 640 --output narrator.mp4

    # A/B against the current SadTalker narrator for the same inputs
    uv run tools/echomimic3.py --image p.png --audio vo.mp3 --output new.mp4 --compare old.mp4

Setup:
    uv sync --extra modal && uv run modal setup
    uv run modal deploy docker/modal-echomimic3/app.py
    # then add the printed URL to .env:
    MODAL_ECHOMIMIC3_ENDPOINT_URL=https://....modal.run

Cost:
    Diffusion video generation, not SadTalker's warp-based animation -- expect
    roughly an order of magnitude more GPU time per second of output. The tool
    prints the measured realtime factor so the real number replaces this guess.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from file_transfer import (
    upload_to_storage, download_from_r2, r2_cleanup,
    download_from_url, get_r2_payload_config,
)

# Wall-clock budget per second of audio. Generous: an A10G cold start has to
# page ~20GB of weights into VRAM before the first segment starts.
PROCESSING_TIME_MULTIPLIER = 90
PROCESSING_TIME_BUFFER = 420

DEFAULT_PROMPT = "A person is speaking to the camera."


def get_audio_duration(audio_path: str) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def calculate_timeout(audio_duration: float) -> int:
    return int(audio_duration * PROCESSING_TIME_MULTIPLIER + PROCESSING_TIME_BUFFER)


def build_comparison(new_video: str, old_video: str, output_path: str,
                     verbose: bool = True) -> str | None:
    """Stack two talking head renders side by side for eyeballing.

    Labels each half so the pair stays readable once it is out of context.
    Heights are matched to the taller input; the audio comes from the new clip.
    """
    if verbose:
        print(f"Building comparison: {old_video} | {new_video}", file=sys.stderr)

    labelled = (
        "[0:v]scale=-2:720,pad=iw:ih+40:0:40:black,"
        "drawtext=text='SadTalker':x=10:y=8:fontsize=24:fontcolor=white[a];"
        "[1:v]scale=-2:720,pad=iw:ih+40:0:40:black,"
        "drawtext=text='EchoMimicV3':x=10:y=8:fontsize=24:fontcolor=white[b];"
        "[a][b]hstack=inputs=2[v]"
    )
    # drawtext needs a fontconfig default that not every ffmpeg build ships, so
    # fall back to an unlabelled stack rather than losing the comparison.
    plain = "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2[v]"

    for filt in (labelled, plain):
        cmd = [
            "ffmpeg", "-y", "-i", old_video, "-i", new_video,
            "-filter_complex", filt, "-map", "[v]", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            if verbose:
                note = "" if filt is labelled else " (unlabelled: drawtext unavailable)"
                print(f"  Comparison: {output_path}{note}", file=sys.stderr)
            return output_path

    print(f"Comparison render failed: {proc.stderr[-400:]}", file=sys.stderr)
    return None


def process_with_cloud(
    image_path: str,
    audio_path: str,
    output_path: str,
    prompt: str = DEFAULT_PROMPT,
    steps: int = 8,
    size: int = 768,
    video_length: int = 81,
    overlap: int = 8,
    guidance_scale: float = 6.0,
    audio_guidance_scale: float = 3.0,
    audio_scale: float = 1.0,
    seed: int = 43,
    fps: int = 25,
    wav2vec: str = "chinese",
    timeout: int = 0,
    verbose: bool = True,
    cloud: str = "modal",
    progress=None,
) -> dict:
    """Generate a talking head via the EchoMimicV3 cloud endpoint."""
    with r2_cleanup() as r2_keys_to_cleanup:
        if verbose:
            print(f"Cloud provider: {cloud}", file=sys.stderr)

        audio_duration = get_audio_duration(audio_path)
        if timeout <= 0:
            timeout = calculate_timeout(audio_duration) if audio_duration else 1800

        if verbose and audio_duration:
            total_frames = int(audio_duration * fps)
            stride = max(1, video_length - overlap)
            segments = max(1, -(-total_frames // stride))
            print(f"Audio: {audio_duration:.1f}s -> ~{total_frames} frames, "
                  f"~{segments} segment{'s' if segments > 1 else ''}, timeout {timeout}s",
                  file=sys.stderr)

        image_url, image_r2_key = upload_to_storage(image_path, "echomimic3/input")
        if not image_url:
            return {"error": "Failed to upload image"}
        if image_r2_key:
            r2_keys_to_cleanup.append(image_r2_key)

        audio_url, audio_r2_key = upload_to_storage(audio_path, "echomimic3/input")
        if not audio_url:
            return {"error": "Failed to upload audio"}
        if audio_r2_key:
            r2_keys_to_cleanup.append(audio_r2_key)

        payload = {
            "input": {
                "image_url": image_url,
                "audio_url": audio_url,
                "prompt": prompt,
                "steps": steps,
                "sample_size": [size, size],
                "video_length": video_length,
                "overlap": overlap,
                "guidance_scale": guidance_scale,
                "audio_guidance_scale": audio_guidance_scale,
                "audio_scale": audio_scale,
                "seed": seed,
                "fps": fps,
                "wav2vec": wav2vec,
            }
        }

        r2_payload = get_r2_payload_config()
        if r2_payload:
            payload["input"]["r2"] = r2_payload
        else:
            print("Warning: R2 not configured. Video will be returned as base64.", file=sys.stderr)

        from cloud_gpu import call_cloud_endpoint

        result, elapsed = call_cloud_endpoint(
            provider=cloud,
            payload=payload,
            tool_name="echomimic3",
            timeout=timeout,
            progress_label="Generating talking head",
            verbose=verbose,
            progress=progress,
        )

        if isinstance(result, dict) and result.get("error"):
            return {"error": result["error"]}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        downloaded = False

        output_r2_key = result.get("r2_key") if isinstance(result, dict) else None
        output_url = result.get("video_url") if isinstance(result, dict) else None

        if output_r2_key:
            downloaded = download_from_r2(output_r2_key, output_path)
            if downloaded:
                r2_keys_to_cleanup.append(output_r2_key)

        if not downloaded and output_url:
            downloaded = download_from_url(output_url, output_path, verbose=verbose)
            if downloaded and output_r2_key:
                r2_keys_to_cleanup.append(output_r2_key)

        if not downloaded:
            video_base64 = result.get("video_base64") if isinstance(result, dict) else None
            if video_base64:
                Path(output_path).write_bytes(base64.b64decode(video_base64))
                downloaded = True

        if not downloaded:
            return {"error": f"No video in result: {list(result.keys()) if isinstance(result, dict) else result}"}

        if verbose:
            size_kb = Path(output_path).stat().st_size // 1024
            rtf = result.get("realtime_factor")
            print(f"  Downloaded: {output_path} ({size_kb}KB, "
                  f"{result.get('width')}x{result.get('height')}"
                  f"{f', {rtf}x realtime' if rtf else ''})", file=sys.stderr)

        return {
            "success": True,
            "output": output_path,
            "processing_time_seconds": round(elapsed, 2),
            "duration_seconds": result.get("duration_seconds"),
            "segments": result.get("segments"),
            "width": result.get("width"),
            "height": result.get("height"),
            "realtime_factor": result.get("realtime_factor"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate talking head videos with EchoMimicV3-Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_group = parser.add_argument_group("Input/output")
    io_group.add_argument("--image", "-i", required=True, help="Portrait image (16:9 for NarratorPiP)")
    io_group.add_argument("--audio", "-a", required=True, help="Driving audio file")
    io_group.add_argument("--output", "-o", default="talking.mp4", help="Output video path")
    io_group.add_argument("--compare", metavar="OLD_VIDEO",
                          help="Also render a side-by-side against an existing (e.g. SadTalker) clip")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--prompt", "-p", default=DEFAULT_PROMPT,
                           help=f"Motion prompt (default: {DEFAULT_PROMPT!r})")
    gen_group.add_argument("--steps", type=int, default=8,
                           help="Denoise steps: 5 for talking head, 15-25 for talking body (default: 8)")
    gen_group.add_argument("--size", type=int, default=768,
                           help="Target size; output keeps the image's aspect ratio (default: 768)")
    gen_group.add_argument("--seed", type=int, default=43, help="Random seed")
    gen_group.add_argument("--fps", type=int, default=25, help="Output frame rate (default: 25)")
    gen_group.add_argument("--wav2vec", choices=["chinese", "english"], default="chinese",
                           help="Audio encoder. 'chinese' is what upstream run_flash.sh uses "
                                "for both languages; 'english' is worth A/B-ing (default: chinese)")

    tune_group = parser.add_argument_group("Tuning")
    tune_group.add_argument("--video-length", type=int, default=81,
                            help="Frames per segment; lower to cut VRAM (default: 81)")
    tune_group.add_argument("--overlap", type=int, default=8,
                            help="Frames cross-faded between segments (default: 8)")
    tune_group.add_argument("--guidance-scale", type=float, default=6.0, help="Text CFG, 3-6")
    tune_group.add_argument("--audio-guidance-scale", type=float, default=3.0,
                            help="Audio CFG; upstream suggests 1.8-2.0 for tightest lip sync")
    tune_group.add_argument("--audio-scale", type=float, default=1.0, help="Audio conditioning strength")

    out_group = parser.add_argument_group("Output control")
    out_group.add_argument("--timeout", type=int, default=0, help="Override auto-calculated timeout")
    out_group.add_argument("--json", action="store_true", help="Output result as JSON")
    out_group.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    out_group.add_argument("--cloud", default="modal", choices=["modal"],
                           help="Cloud provider (RunPod not deployed for this tool yet)")

    args = parser.parse_args()
    verbose = not args.quiet and not args.json

    for label, path in (("image", args.image), ("audio", args.audio)):
        if not Path(path).exists():
            msg = f"{label.capitalize()} not found: {path}"
            print(json.dumps({"error": msg}) if args.json else f"Error: {msg}", file=sys.stderr)
            return 1

    if args.compare and not Path(args.compare).exists():
        msg = f"Comparison video not found: {args.compare}"
        print(json.dumps({"error": msg}) if args.json else f"Error: {msg}", file=sys.stderr)
        return 1

    result = process_with_cloud(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        prompt=args.prompt,
        steps=args.steps,
        size=args.size,
        video_length=args.video_length,
        overlap=args.overlap,
        guidance_scale=args.guidance_scale,
        audio_guidance_scale=args.audio_guidance_scale,
        audio_scale=args.audio_scale,
        seed=args.seed,
        fps=args.fps,
        wav2vec=args.wav2vec,
        timeout=args.timeout,
        verbose=verbose,
    )

    if result.get("error"):
        print(json.dumps(result) if args.json else f"Error: {result['error']}", file=sys.stderr)
        return 1

    if args.compare:
        compare_path = str(Path(args.output).with_name(Path(args.output).stem + "_vs_sadtalker.mp4"))
        built = build_comparison(args.output, args.compare, compare_path, verbose=verbose)
        if built:
            result["comparison"] = built

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nGenerated: {result['output']}")
        if result.get("realtime_factor"):
            print(f"  {result['duration_seconds']}s of video in "
                  f"{result['processing_time_seconds']}s ({result['realtime_factor']}x realtime)")
        if result.get("comparison"):
            print(f"  Side-by-side: {result['comparison']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
