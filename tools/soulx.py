#!/usr/bin/env python3
"""
Generate talking head videos using SoulX-FlashHead.

The toolkit's default talking head. SoulX-FlashHead (Soul AI Lab, Apache 2.0,
1.3B) is trained with Oracle-Guided Bidirectional Distillation, which targets
the identity drift that limits segment-chained talking head models: measured at
97% of frame-zero sharpness at 70s where EchoMimicV3 fell to 50% on identical
input. There is no short-render ceiling to design around.

It preserves the input image's aspect ratio, so 16:9 presenter images come back
16:9 with no --preprocess workaround.

Usage:
    # Basic -- aspect ratio follows the input image
    uv run tools/soulx.py --image portrait.png --audio voiceover.mp3 --output talking.mp4

    # NarratorPiP (16:9 in, 16:9 out)
    uv run tools/soulx.py \
        --image presenter_16x9.png --audio scene_01.mp3 \
        --size 768 --output narrator.mp4

    # Exact dimensions instead of a long edge
    uv run tools/soulx.py -i p.png -a vo.mp3 --width 768 --height 432 -o n.mp4

    # A/B against an existing SadTalker render of the same inputs
    uv run tools/soulx.py -i p.png -a vo.mp3 -o new.mp4 --compare old.mp4

Setup:
    uv sync --extra modal && uv run modal setup
    uv run modal run docker/modal-soulx/app.py::populate_weights   # one-off, 14.7GB
    uv run modal deploy docker/modal-soulx/app.py
    # then add the printed URL to .env:
    MODAL_SOULX_ENDPOINT_URL=https://....modal.run

Cost:
    Measured ~$0.0024 per second of output on A10G (~7.9x realtime steady
    state), against SadTalker's ~$0.0014. The first call into a cold container
    also pays ~600s of torch.compile, which is per-resolution -- so a batch of
    per-scene clips at ONE size amortises it, and switching sizes per scene
    does not. The tool prints the measured realtime factor.
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

# Wall-clock budget per second of audio. Generous because a cold container pays
# torch.compile (~600s) before the first chunk, on top of paging weights in.
PROCESSING_TIME_MULTIPLIER = 30
PROCESSING_TIME_BUFFER = 900

# Pixel dimensions must land on the model's latent grid: the Wan2.1 VAE has
# spatial stride 8 and the transformer patchifies 2x2. Nothing upstream
# validates this -- a bad size floors silently and desyncs the latent grid from
# the pixel grid -- so it is checked here as well as server-side.
GRID = 16


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


def check_size(height: int, width: int) -> str | None:
    """Return an error string if the requested size is off the latent grid."""
    for name, v in (("height", height), ("width", width)):
        if v % GRID:
            return (f"--{name} {v} must be a multiple of {GRID}. "
                    f"Nearest: {v // GRID * GRID} or {(v // GRID + 1) * GRID}.")
    return None


def build_comparison(new_video: str, old_video: str, output_path: str,
                     old_label: str = "previous", verbose: bool = True) -> str | None:
    """Stack two talking head renders side by side for eyeballing.

    Labels each half so the pair stays readable once it is out of context.
    Audio comes from the new clip.
    """
    if verbose:
        print(f"Building comparison: {old_video} | {new_video}", file=sys.stderr)

    labelled = (
        f"[0:v]scale=-2:720,pad=iw:ih+40:0:40:black,"
        f"drawtext=text='{old_label}':x=10:y=8:fontsize=24:fontcolor=white[a];"
        f"[1:v]scale=-2:720,pad=iw:ih+40:0:40:black,"
        f"drawtext=text='SoulX-FlashHead':x=10:y=8:fontsize=24:fontcolor=white[b];"
        f"[a][b]hstack=inputs=2[v]"
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
    size: int = 768,
    height: int = 0,
    width: int = 0,
    seed: int = 42,
    use_face_crop: bool = False,
    timeout: int = 0,
    verbose: bool = True,
    cloud: str = "modal",
    progress=None,
) -> dict:
    """Generate a talking head via the SoulX-FlashHead cloud endpoint."""
    with r2_cleanup() as r2_keys_to_cleanup:
        if verbose:
            print(f"Cloud provider: {cloud}", file=sys.stderr)

        if height and width:
            err = check_size(height, width)
            if err:
                return {"error": err}

        audio_duration = get_audio_duration(audio_path)
        if timeout <= 0:
            timeout = calculate_timeout(audio_duration) if audio_duration else 2400

        if verbose and audio_duration:
            # Pro emits 28 new frames per 33-frame chunk at 25fps.
            chunks = max(1, -(-int(audio_duration * 25) // 28))
            print(f"Audio: {audio_duration:.1f}s -> ~{chunks} chunk"
                  f"{'s' if chunks > 1 else ''}, timeout {timeout}s", file=sys.stderr)

        image_url, image_r2_key = upload_to_storage(image_path, "soulx/input")
        if not image_url:
            return {"error": "Failed to upload image"}
        if image_r2_key:
            r2_keys_to_cleanup.append(image_r2_key)

        audio_url, audio_r2_key = upload_to_storage(audio_path, "soulx/input")
        if not audio_url:
            return {"error": "Failed to upload audio"}
        if audio_r2_key:
            r2_keys_to_cleanup.append(audio_r2_key)

        payload = {
            "input": {
                "image_url": image_url,
                "audio_url": audio_url,
                "seed": seed,
                "use_face_crop": use_face_crop,
            }
        }
        if height and width:
            payload["input"]["height"] = height
            payload["input"]["width"] = width
        else:
            payload["input"]["size"] = size

        r2_payload = get_r2_payload_config()
        if r2_payload:
            payload["input"]["r2"] = r2_payload
        else:
            print("Warning: R2 not configured. Video will be returned as base64.",
                  file=sys.stderr)

        from cloud_gpu import call_cloud_endpoint

        result, elapsed = call_cloud_endpoint(
            provider=cloud,
            payload=payload,
            tool_name="soulx",
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
            keys = list(result.keys()) if isinstance(result, dict) else result
            return {"error": f"No video in result: {keys}"}

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
            "chunks": result.get("chunks"),
            "width": result.get("width"),
            "height": result.get("height"),
            "realtime_factor": result.get("realtime_factor"),
            "median_chunk_seconds": result.get("median_chunk_seconds"),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate talking head videos with SoulX-FlashHead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io_group = parser.add_argument_group("Input/output")
    io_group.add_argument("--image", "-i", required=True,
                          help="Portrait image (16:9 for NarratorPiP)")
    io_group.add_argument("--audio", "-a", required=True, help="Driving audio file")
    io_group.add_argument("--output", "-o", default="talking.mp4", help="Output video path")
    io_group.add_argument("--compare", metavar="OLD_VIDEO",
                          help="Also write a side-by-side against an existing render")
    io_group.add_argument("--compare-label", default="previous",
                          help="Label for the left half of --compare (default: previous)")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--size", type=int, default=768,
                           help="Target long edge; aspect follows the image (default: 768)")
    gen_group.add_argument("--height", type=int, default=0,
                           help=f"Exact height, multiple of {GRID}. Use with --width")
    gen_group.add_argument("--width", type=int, default=0,
                           help=f"Exact width, multiple of {GRID}. Use with --height")
    gen_group.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_group.add_argument("--face-crop", action="store_true",
                           help="Upstream face detect+crop. Square-only, so it "
                                "discards a 16:9 framing -- rarely what you want")

    out_group = parser.add_argument_group("Output control")
    out_group.add_argument("--timeout", type=int, default=0,
                           help="Override auto-calculated timeout")
    out_group.add_argument("--json", action="store_true", help="Output result as JSON")
    out_group.add_argument("--quiet", "-q", action="store_true",
                           help="Suppress progress output")
    out_group.add_argument("--cloud", default="modal", choices=["modal"],
                           help="Cloud provider (Modal only)")

    args = parser.parse_args()
    verbose = not args.quiet

    if bool(args.height) != bool(args.width):
        parser.error("--height and --width must be given together (or use --size)")

    for path, what in ((args.image, "Image"), (args.audio, "Audio")):
        if not Path(path).exists():
            msg = f"{what} not found: {path}"
            print(json.dumps({"error": msg}) if args.json else f"Error: {msg}",
                  file=sys.stderr)
            sys.exit(1)

    result = process_with_cloud(
        image_path=args.image,
        audio_path=args.audio,
        output_path=args.output,
        size=args.size,
        height=args.height,
        width=args.width,
        seed=args.seed,
        use_face_crop=args.face_crop,
        timeout=args.timeout,
        verbose=verbose,
        cloud=args.cloud,
    )

    if result.get("success") and args.compare:
        if Path(args.compare).exists():
            comparison = build_comparison(
                args.output, args.compare,
                str(Path(args.output).with_name(Path(args.output).stem + "_compare.mp4")),
                old_label=args.compare_label, verbose=verbose,
            )
            if comparison:
                result["comparison"] = comparison
        else:
            print(f"Warning: --compare file not found: {args.compare}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, indent=2))
    elif result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
    else:
        print(f"Generated: {result['output']}")

    sys.exit(1 if result.get("error") else 0)


if __name__ == "__main__":
    main()
