#!/usr/bin/env python3
"""Step 1 — per-scene voiceover from scenes.json.

Writes one .txt per scene into audio/scenes/, drives tools/voiceover.py over
the directory (clone or built-in speaker, pacing-clamped via --max-wpm), and
records the actual durations in vo_durations.json — the timeline anchor that
build.py reads. Run from this project directory:

    python3 gen_vo.py            # generate all scenes
    python3 gen_vo.py --force    # regenerate even if MP3s exist
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_toolkit_root() -> Path:
    import os
    env = os.environ.get("VIDEO_TOOLKIT_ROOT")
    if env and (Path(env) / "tools" / "voiceover.py").exists():
        return Path(env)
    p = HERE
    for _ in range(5):
        if (p / "tools" / "voiceover.py").exists():
            return p
        p = p.parent
    sys.exit("Could not find toolkit root (tools/voiceover.py) above this directory.\n"
             "Projects normally live in <toolkit>/projects/; for copies elsewhere "
             "set VIDEO_TOOLKIT_ROOT=/path/to/claude-code-video-toolkit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Regenerate existing MP3s")
    args = ap.parse_args()

    root = find_toolkit_root()
    config = json.loads((HERE / "config.json").read_text())
    scenes = json.loads((HERE / "scenes.json").read_text())["scenes"]
    voice = config.get("voice", {})

    scene_dir = HERE / "audio" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for s in scenes:
        txt = scene_dir / f"{s['id']}_{s['slug']}.txt"
        mp3 = txt.with_suffix(".mp3")
        txt.write_text(s["text"])
        if args.force or not mp3.exists():
            pending.append(s["id"])
    if not pending:
        print("All scene MP3s exist — use --force to regenerate.")
        return
    if args.force:
        for s in scenes:
            (scene_dir / f"{s['id']}_{s['slug']}.mp3").unlink(missing_ok=True)

    cmd = [
        sys.executable, str(root / "tools" / "voiceover.py"),
        "--provider", voice.get("provider", "qwen3"),
        "--scene-dir", str(scene_dir),
        "--json",
    ]
    if voice.get("maxWpm"):
        cmd += ["--max-wpm", str(voice["maxWpm"])]
    if voice.get("provider", "qwen3") == "qwen3":
        cmd += ["--cloud", voice.get("cloud", "modal")]
        ref_audio = voice.get("refAudio", "")
        if ref_audio:
            ref_path = Path(ref_audio)
            if not ref_path.is_absolute():
                ref_path = HERE / ref_path
            if not ref_path.exists():
                sys.exit(f"Clone reference audio not found: {ref_path}")
            if not voice.get("refText"):
                sys.exit("voice.refText is required with voice.refAudio (exact transcript).")
            cmd += ["--ref-audio", str(ref_path), "--ref-text", voice["refText"]]
        else:
            cmd += ["--speaker", voice.get("speaker", "Ryan")]
            if voice.get("tone"):
                cmd += ["--tone", voice["tone"]]

    print(f"Generating {len(scenes)} scene VOs "
          f"({voice.get('provider', 'qwen3')}, max {voice.get('maxWpm', '—')} wpm)...",
          file=sys.stderr)
    r = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        sys.exit(f"voiceover.py failed:\n{r.stderr[-1000:]}")

    result = json.loads(r.stdout)
    failed = [Path(i["output"]).stem for i in result["scenes"] if not i.get("success")]
    if failed:
        err = next(i.get("error", "?") for i in result["scenes"] if not i.get("success"))
        sys.exit(f"{len(failed)} scene(s) failed ({', '.join(failed)}): {err}")
    durations, problems = {}, []
    for item in result["scenes"]:
        name = Path(item["output"]).stem            # e.g. 01_hook
        sid = name.split("_", 1)[0]
        durations[sid] = item["duration_seconds"]
        note = ""
        if item.get("pace_adjusted"):
            note = f" (clamped from {item['pace_adjusted']['original_wpm']:.0f} wpm)"
        elif item.get("pacing") in ("fast", "slow"):
            note = f" [{item['pacing'].upper()} {item['wpm']:.0f} wpm]"
            problems.append(name)
        print(f"  {name}: {item['duration_seconds']}s, {item.get('wpm', '?')} wpm{note}")

    (HERE / "vo_durations.json").write_text(json.dumps(durations, indent=2))
    total = sum(durations.values())
    print(f"Total narration: {total:.1f}s → video ≈ {total + len(scenes) * 1.2:.0f}s")
    if total + len(scenes) * 1.2 > 178:
        print("Note: over ~3 min — fine for TikTok, too long for YouTube Shorts/Reels.")
    if problems:
        print(f"Pacing flags on: {', '.join(problems)} — consider voice.maxWpm in config.json")
    print("Next: python3 gen_captions.py")


if __name__ == "__main__":
    main()
