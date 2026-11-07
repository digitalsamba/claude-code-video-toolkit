#!/usr/bin/env python3
"""Put each music take under the finished picture, so the takes are chosen against the video, not alone.

    uv run tools/audition.py --video out/video.nomusic.mp4 --music public/audio/music_*.mp3 \\
        --volume 0.15 --fade-out 1.5
    uv run tools/audition.py --video out/video.mp4 --drop-audio --music public/audio/takes/*.mp3 \\
        --volume 0.9 --fade-out 1.5 --recommend --cuts 11.5,17.5,37.5 --under text

The picture doesn't depend on the music, so one render serves every take: each one is mixed in with ffmpeg in
seconds, at the level and fades the composition uses, trimmed to the video, and normalised the way
`qc.py --fix` does (-14 LUFS). Outputs land in <video dir>/audition/<video>.<take>.mp4.

The video's own audio is kept and the take mixed under it (the narration stays). For a video whose render
already carries a music bed and nothing else, pass --drop-audio. A narrated video needs one render without
music: the templates skip a music file that isn't there (getStaticFiles), so move it aside and render once.

--recommend adds pick_music.py's measured recommendation (pass its --cuts / --under / --target), so a command
can say which take it would choose and why, and then ask the user to watch them all and choose.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def probe(path: Path) -> tuple[float, bool]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    d = json.loads(out)
    return float(d["format"]["duration"]), any(s.get("codec_type") == "audio" for s in d.get("streams", []))


def mix(video: Path, take: Path, out: Path, *, duration: float, keep: bool, volume: float, start: float,
        fade_in: float, fade_out: float, lufs: float, tp: float) -> None:
    length = max(0.0, duration - start)
    chain = [f"atrim=0:{length:.3f}", "asetpts=PTS-STARTPTS", f"volume={volume}"]
    if fade_in > 0:
        chain.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        chain.append(f"afade=t=out:st={max(0.0, length - fade_out):.3f}:d={fade_out}")
    if start > 0:
        ms = int(start * 1000)
        chain.append(f"adelay={ms}|{ms}")
    chain.append(f"apad=whole_dur={duration:.3f}")
    music = f"[1:a]aformat=channel_layouts=stereo,{','.join(chain)}[m]"
    if keep:
        graph = f"{music};[0:a]aformat=channel_layouts=stereo[v];[v][m]amix=inputs=2:normalize=0:duration=first[a0]"
    else:
        graph = f"{music};[m]anull[a0]"
    graph += f";[a0]loudnorm=I={lufs}:TP={tp}:LRA=11[a]"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(take), "-filter_complex", graph,
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-t", f"{duration:.3f}", str(out)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="the finished picture (its audio is kept unless --drop-audio)")
    ap.add_argument("--music", nargs="+", required=True, help="the takes to audition")
    ap.add_argument("--drop-audio", action="store_true", help="discard the video's own audio (an old music bed)")
    ap.add_argument("--volume", type=float, default=1.0, help="music gain, as in the composition (default 1.0)")
    ap.add_argument("--start", type=float, default=0.0, help="second the music starts in the video")
    ap.add_argument("--fade-in", type=float, default=0.0)
    ap.add_argument("--fade-out", type=float, default=0.0, help="the composition's music fade at its end, seconds")
    ap.add_argument("--lufs", type=float, default=-14.0)
    ap.add_argument("--true-peak", type=float, default=-1.5, help="dBTP; -1.5 keeps -1 after AAC encoding")
    ap.add_argument("--out-dir", help="default: <video dir>/audition")
    ap.add_argument("--recommend", action="store_true", help="add pick_music.py's recommendation")
    ap.add_argument("--cuts", default="", help="for --recommend: the video's act cuts, comma-separated seconds")
    ap.add_argument("--under", choices=["voice", "text", "none"], default="none", help="for --recommend")
    ap.add_argument("--json", action="store_true", help="print the result as JSON on stdout")
    a = ap.parse_args()

    video = Path(a.video)
    takes = [Path(m) for m in a.music]
    missing = [str(p) for p in [video, *takes] if not p.exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    duration, has_audio = probe(video)
    keep = has_audio and not a.drop_audio
    out_dir = Path(a.out_dir) if a.out_dir else video.parent / "audition"
    out_dir.mkdir(parents=True, exist_ok=True)

    renders = []
    for take in takes:
        out = out_dir / f"{video.stem}.{take.stem}.mp4"
        mix(video, take, out, duration=duration, keep=keep, volume=a.volume, start=a.start,
            fade_in=a.fade_in, fade_out=a.fade_out, lufs=a.lufs, tp=a.true_peak)
        renders.append({"take": str(take), "render": str(out)})
        log(f"  {take.name:<28} -> {out}")

    result: dict = {"video": str(video), "kept_video_audio": keep, "renders": renders}
    if a.recommend:
        import pick_music as pm

        cuts = [float(c) for c in a.cuts.split(",") if c.strip()]
        rows = pm.rank(takes, duration, cuts, a.under, a.fade_out)
        best = rows[0]
        render = next(r["render"] for r in renders if r["take"] == best["file"])
        result["recommended"] = {"take": best["file"], "render": render, "eligible": best["eligible"],
                                 "good": best["good"], "bad": best["bad"], "defects": best["defects"]}
        log(f"\nMeasured pick: {Path(best['file']).name}"
            + (f" ({'; '.join(best['good'][:3])})" if best["good"] else "")
            + ("" if best["eligible"] else f"  [defects: {'; '.join(best['defects'])}]"))
        if len(rows) > 1 and rows[1]["score"] == best["score"] and rows[1]["eligible"]:
            log(f"  Tied on score with {Path(rows[1]['file']).name}: the measure can't separate them, so listen.")
    if not keep and has_audio:
        log("(the video's own audio was dropped)")
    elif keep:
        log("(mixed under the video's own audio)")
    log("Watch each render through before choosing: the measures rule takes out, they don't hear them.")
    if a.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
