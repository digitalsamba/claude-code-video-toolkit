#!/usr/bin/env python3
"""
Render a Manim scene as a clip for a video (optional: needs Manim, and nothing else in the toolkit depends on it).

Manim Community (https://github.com/ManimCommunity/manim) animates shapes, text, code, graphs and SVGs from a
Python scene file. This wrapper finds it, renders at the toolkit's 30 fps, converts a transparent render to the
WebM that Remotion plays, and keeps the scene's event notes beside the clip.

    uv run tools/manim_render.py doctor
    uv run tools/manim_render.py render examples/manim-ident/idents.py Typed --out out/typed.mp4
    uv run tools/manim_render.py render scene.py Flow --alpha --out public/manim/flow.webm --captions public/audio/flow.captions.json
    uv run tools/manim_render.py render scene.py Fix --size 1080x1920 --out out/fix-9x16.mp4

Scenes import `toolkit_manim` (lib/manim) for narration cues, a brand's colours, loops and event notes.
Experimental. What works, what bit and how to install it: docs/manim.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "lib" / "manim"

INSTALL = {
    "Darwin": "brew install manim   (a prebuilt bottle; it upgrades ffmpeg, see docs/manim.md)",
    "Linux": "see https://docs.manim.community/en/stable/installation.html (or the manimcommunity/manim Docker image)",
    "Windows": "uv tool install manim   (or see https://docs.manim.community/en/stable/installation.html)",
}


def find_manim() -> str | None:
    return os.environ.get("MANIM_BIN") or shutil.which("manim")


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


def _ffmpeg_has(kind: str, name: str) -> bool:
    out = _run(["ffmpeg", "-hide_banner", f"-{kind}"]).stdout
    return bool(re.search(rf"\s{re.escape(name)}\s", out))


def doctor(as_json: bool = False) -> int:
    checks = []

    def add(name, ok, detail, required=True):
        checks.append({"check": name, "ok": bool(ok), "required": required, "detail": detail})

    manim = find_manim()
    version = ""
    if manim:
        res = _run([manim, "--version"], env={**os.environ, "PYTHONWARNINGS": "ignore"})
        found = re.search(r"v(\d+\.\d+\.\d+)", res.stdout + res.stderr)
        version = found.group(1) if found else ""
    add("manim", manim and version, f"{version} at {manim}" if version else
        f"not found. Install: {INSTALL.get(platform.system(), INSTALL['Linux'])}")
    ffmpeg = shutil.which("ffmpeg")
    add("ffmpeg", ffmpeg, ffmpeg or "not found")
    if ffmpeg:
        add("ffmpeg libvpx-vp9", _ffmpeg_has("encoders", "libvpx-vp9"),
            "transparent clips for Remotion" if _ffmpeg_has("encoders", "libvpx-vp9") else "missing: --alpha cannot make a WebM")
        text = _ffmpeg_has("filters", "drawtext")
        add("ffmpeg drawtext", text, "present" if text else
            "missing. Not needed here, but tools/notebooklm_brand.py uses it; Homebrew's ffmpeg 9 moved it to ffmpeg-full",
            required=False)
    latex = shutil.which("latex") and shutil.which("dvisvgm")
    add("latex", latex, "formulas (MathTex, Tex) available" if latex else
        "not found: Text, shapes, code and SVGs work; formulas (MathTex, Tex) do not", required=False)

    ok = all(c["ok"] for c in checks if c["required"])
    if as_json:
        print(json.dumps({"ok": ok, "checks": checks}, indent=1))
    else:
        for c in checks:
            mark = "ok  " if c["ok"] else ("FAIL" if c["required"] else "note")
            print(f"{mark} {c['check']}: {c['detail']}")
    return 0 if ok else 1


def parse_size(text: str) -> tuple[int, int]:
    found = re.fullmatch(r"(\d+)x(\d+)", text.strip().lower())
    if not found:
        raise argparse.ArgumentTypeError(f"size is WIDTHxHEIGHT, e.g. 1920x1080 (got {text!r})")
    w, h = int(found.group(1)), int(found.group(2))
    if w % 2 or h % 2:
        raise argparse.ArgumentTypeError("width and height must be even")
    return w, h


def output_plan(out: Path, alpha: bool) -> dict:
    """What gets written where. A transparent clip is a WebM (Remotion) unless .mov is asked for (ProRes 4444)."""
    suffix = out.suffix.lower()
    if alpha and suffix not in (".webm", ".mov"):
        raise SystemExit(f"--alpha writes .webm (for Remotion) or .mov (ProRes 4444), not {suffix or 'no extension'}")
    if not alpha and suffix != ".mp4":
        raise SystemExit(f"an opaque clip is an .mp4, not {suffix or 'no extension'} (add --alpha for a transparent one)")
    return {"clip": out, "events": out.with_suffix(".events.json"), "kind": suffix.lstrip(".")}


def manim_command(manim: str, scene_file: Path, scene: str, size: tuple[int, int], fps: int, alpha: bool, media: Path) -> list[str]:
    cmd = [manim, "render", "-qh", "--fps", str(fps), "-r", f"{size[0]},{size[1]}", "--media_dir", str(media),
           "--disable_caching", "-o", scene]
    if alpha:
        cmd.append("-t")
    return cmd + [str(scene_file), scene]


def render(args) -> int:
    manim = find_manim()
    if not manim:
        print(f"Manim is not installed. {INSTALL.get(platform.system(), INSTALL['Linux'])}", file=sys.stderr)
        return 2
    scene_file = Path(args.file).resolve()
    if not scene_file.exists():
        print(f"no scene file at {scene_file}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    plan = output_plan(out, args.alpha)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="manim-") as tmp:
        media = Path(tmp)
        events_tmp = media / "events.json"
        env = {**os.environ, "PYTHONWARNINGS": "ignore", "MANIM_TOOLKIT_ROOT": str(ROOT),
               "MANIM_TOOLKIT_SIZE": f"{args.size[0]}x{args.size[1]}", "MANIM_TOOLKIT_EVENTS": str(events_tmp),
               "PYTHONPATH": os.pathsep.join([str(LIB), str(scene_file.parent), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
        if args.brand:
            env["MANIM_TOOLKIT_BRAND"] = args.brand
        if args.captions:
            env["MANIM_TOOLKIT_CAPTIONS"] = str(Path(args.captions).resolve())
        res = _run(manim_command(manim, scene_file, args.scene, args.size, args.fps, args.alpha, media),
                   cwd=scene_file.parent, env=env)
        made = sorted(media.glob(f"videos/**/{args.scene}.mov" if args.alpha else f"videos/**/{args.scene}.mp4"))
        made = [p for p in made if "partial_movie_files" not in p.parts]
        if res.returncode or not made:
            tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-25:])
            print(f"Manim failed on {args.scene}:\n{tail}", file=sys.stderr)
            return 1
        src = made[0]
        if not args.alpha:
            shutil.move(str(src), plan["clip"])
        else:
            codec = (["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28", "-auto-alt-ref", "0"]
                     if plan["kind"] == "webm" else ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"])
            conv = _run(["ffmpeg", "-v", "error", "-y", "-i", src, *codec, plan["clip"]])
            if conv.returncode:
                print(f"ffmpeg could not write {plan['clip'].name}:\n{conv.stderr.strip()}", file=sys.stderr)
                return 1
        events = None
        if events_tmp.exists():
            shutil.move(str(events_tmp), plan["events"])
            events = plan["events"]

    probe = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", plan["clip"]])
    seconds = float(probe.stdout.strip() or 0)
    result = {"clip": str(plan["clip"]), "seconds": round(seconds, 3), "frames": round(seconds * args.fps),
              "size": f"{args.size[0]}x{args.size[1]}", "fps": args.fps, "alpha": args.alpha,
              "events": str(events) if events else None}
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"{plan['clip']}  {seconds:.2f} s, {result['size']} at {args.fps} fps" + (", transparent" if args.alpha else ""))
        if events:
            print(f"{events}  when each audible thing happens")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    doc = sub.add_parser("doctor", help="is Manim here, and what will and won't work")
    doc.add_argument("--json", action="store_true")
    ren = sub.add_parser("render", help="render one scene to a clip")
    ren.add_argument("file", help="the scene file (.py)")
    ren.add_argument("scene", help="the Scene class to render")
    ren.add_argument("--out", required=True, help=".mp4, or with --alpha .webm (Remotion) or .mov (ProRes 4444)")
    ren.add_argument("--alpha", action="store_true", help="transparent background")
    ren.add_argument("--size", type=parse_size, default=(1920, 1080), help="WIDTHxHEIGHT (default 1920x1080); a portrait size gives a 9x16 frame")
    ren.add_argument("--fps", type=int, default=30)
    ren.add_argument("--brand", help="brands/NAME, for Brand() in the scene")
    ren.add_argument("--captions", help="a .captions.json, for Voice() in the scene (MANIM_TOOLKIT_CAPTIONS)")
    ren.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return doctor(args.json) if args.command == "doctor" else render(args)


if __name__ == "__main__":
    sys.exit(main())
