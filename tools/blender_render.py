#!/usr/bin/env python3
"""Shot spec in, Blender film out: build a 3D scene from a JSON spec, check every shot, then render.

The spec says what happens (cast, set, shots by type, acting beats in seconds); lib/blender compiles it into a
Blender scene. `check` is the loop to iterate in: it builds the scene, verifies each camera by geometry (subject
in frame, nothing blocking the view), measures each shot's exposure, and tiles one still per shot into a labelled
contact sheet to look at. `render` makes the film and its sound, placed on the frames the motion was keyed on.

    uv run tools/blender_render.py doctor                                                  # is Blender ready? how fast?
    uv run tools/blender_render.py check  examples/blender-disk-panic/spec.json            # ~5 s
    uv run tools/blender_render.py render examples/blender-disk-panic/spec.json            # ~15 min on an M4
    uv run tools/blender_render.py render spec.json --res 1280x720                         # a faster draft
    uv run tools/blender_render.py render spec.json --engine cycles                        # physical light, hours
    uv run tools/blender_render.py check  spec.json --json                                 # for agents

Elements for a Remotion video: one clip per shot, with a transparent background if asked, copied into the
project's public/ folder with a clips.json manifest (lib/components BlenderClip plays them):

    uv run tools/blender_render.py render spec.json --clips --alpha --to projects/my-video/public/blender
    uv run tools/blender_render.py render spec.json --clips --shots S2_reveal              # one shot again

    uv run tools/blender_render.py open spec.json                                          # the built scene in Blender

Output goes to <spec dir>/out/ (out/<name>/ for a spec not called spec.json) unless --out is given. Needs Blender 4.2 or later (made and tested on 5.3): on PATH
as `blender`, in the usual install folder, or named by BLENDER=/path/to/blender. Spec format:
.claude/skills/blender/SKILL.md. Setup and timings: docs/blender.md.
Exit codes: 0 passed, 1 a check failed (render still runs if asked with --force), 2 spec or Blender error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPILER = ROOT / "lib" / "blender" / "compile.py"
sys.path.insert(0, str(ROOT / "lib" / "blender"))

LUMA_RANGE = (0.08, 0.70)   # mean luma (0-1) a shot should sit in
CLIP_LIMIT = 0.02           # share of near-white pixels allowed
MIN_VERSION = (4, 2)        # EEVEE Next; older builds have a different engine and API
INSTALL = ("install it (https://www.blender.org/download/, `brew install --cask blender`, `winget install "
           "BlenderFoundation.Blender`, `snap install blender --classic`) or set BLENDER=/path/to/blender")


def blender_candidates() -> list[str]:
    """Where Blender usually is, per platform, newest first where there are several."""
    home = Path.home()
    found = [os.environ.get("BLENDER"), shutil.which("blender"),
             "/Applications/Blender.app/Contents/MacOS/Blender",
             str(home / "Applications/Blender.app/Contents/MacOS/Blender")]
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
        if base:
            found += sorted(glob.glob(os.path.join(base, "Blender Foundation", "Blender*", "blender.exe")), reverse=True)
    found += ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender", "/opt/blender/blender"]
    return [c for c in found if c]


def installed_blender() -> str | None:
    """The first candidate that is a file: BLENDER set to a folder is not a Blender."""
    return next((c for c in blender_candidates() if Path(c).is_file()), None)


def find_blender() -> str:
    return installed_blender() or sys.exit(f"Blender not found: {INSTALL}")


def write_json(path: Path, data) -> None:
    """Whole file or no file: a run that dies mid-write must not leave half a manifest for the next one to read."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def merge_clips(path: Path, made: list[dict], spec_name: str, whole_spec: bool) -> dict:
    """Add this run's clips to a clips.json that other specs may share. Each clip carries its own fps, resolution
    and spec, so two specs at different rates or shapes don't relabel each other's clips."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ids = {c["id"] for c in made}
    kept = []
    for c in old.get("clips", []):
        # clips written before specs were recorded take the file's fps and resolution
        c = {"fps": old.get("fps"), "resolution": old.get("resolution"), **c}
        mine = c.get("spec") == spec_name
        if c["id"] in ids:
            if not mine and c.get("spec"):
                print(f"  WARNING: shot {c['id']!r} in {path.name} was from {c['spec']} and is now from {spec_name}: "
                      f"give the two specs' shots different ids")
            continue
        if mine and whole_spec:
            continue  # a shot this spec no longer has (renamed or removed)
        kept.append(c)
    clips = kept + made
    # top-level fps and resolution stay for older readers: the values of the clips just made
    data = {"fps": made[-1]["fps"] if made else old.get("fps"),
            "resolution": made[-1]["resolution"] if made else old.get("resolution"), "clips": clips}
    write_json(path, data)
    return data


def blender_version(binary: str) -> tuple[tuple[int, ...], str]:
    out = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"Blender (\d+)\.(\d+)(?:\.(\d+))?", out)
    return (tuple(int(x) for x in m.groups() if x) if m else (0,)), (m.group(0) if m else out.strip()[:60] or "unknown")


def run_compiler(spec: Path, out: Path, mode: str, res: str | None, engine: str | None, *, alpha=False,
                 clips=False, shots: str | None = None, bench=0, total_frames=0) -> dict:
    binary = find_blender()
    version, label = blender_version(binary)
    if version < MIN_VERSION:
        raise RuntimeError(f"{label} at {binary} is too old: needs {MIN_VERSION[0]}.{MIN_VERSION[1]} or later")
    cmd = [binary, "-b", "-noaudio", "-P", str(COMPILER), "--", "--spec", str(spec), "--out", str(out),
           "--mode", mode]
    for flag, val in (("--res", res), ("--engine", engine), ("--shots", shots)):
        if val:
            cmd += [flag, val]
    if alpha:
        cmd.append("--alpha")
    if clips:
        cmd.append("--clips")
    if bench:
        cmd += ["--bench", str(bench)]
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").unlink(missing_ok=True)  # a crash must not be explained by the last run's verdict
    log = out / f"blender-{'bench' if bench else mode}.log"
    t = time.time()
    done, last = 0, t
    bench_result = None
    with open(log, "w", encoding="utf-8") as fh, \
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace") as proc:
        for line in proc.stdout:
            fh.write(line)
            if line.startswith("BENCH "):
                bench_result = json.loads(line[6:])
            # a frame is finished when Blender appends it to the video or saves its PNG
            if total_frames and ("Video append frame" in line or ("Saved:" in line and "frames" in line)):
                done += 1
                if time.time() - last > 30 and done < total_frames:
                    last = time.time()
                    spf = (last - t) / done
                    print(f"  {done}/{total_frames} frames, {spf:.2f} s/frame, about "
                          f"{(total_frames - done) * spf / 60:.1f} min left", flush=True)
    text = log.read_text(encoding="utf-8", errors="replace")
    checks_path = out / "checks.json"
    if proc.returncode != 0 or "COMPILE_DONE" not in text:
        err = json.loads(checks_path.read_text()).get("spec_error") if checks_path.exists() else None
        tail = "\n".join(ln for ln in text.splitlines() if "Error" in ln or "Traceback" in ln or ln.startswith("  File"))[-2000:]
        raise RuntimeError(err or f"Blender failed (exit {proc.returncode}); see {log}\n{tail}")
    checks = json.loads(checks_path.read_text())
    checks["seconds"] = round(time.time() - t, 1)
    if bench_result:
        checks["bench"] = bench_result
    return checks


def exposure(png: Path) -> tuple[float, float]:
    """Mean luma and the share of near-white pixels, over the pixels that are there (alpha weighted)."""
    from PIL import Image
    im = Image.open(png).convert("RGBA").resize((240, 135))
    luma, alpha = list(im.convert("L").getdata()), list(im.getchannel("A").getdata())
    weight = sum(alpha) or 1
    return (sum(lu * a for lu, a in zip(luma, alpha)) / weight / 255,
            sum(a for lu, a in zip(luma, alpha) if lu > 250) / weight)


def contact_sheet(stills: list[tuple[str, Path, list[str]]], dest: Path, cols: int = 2):
    from PIL import Image, ImageDraw
    first = Image.open(stills[0][1])
    w = 640
    h, pad, bar = round(w * first.height / first.width), 8, 26
    rows = (len(stills) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + bar + pad) + pad), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for i, (label, png, problems) in enumerate(stills):
        x, y = pad + (i % cols) * (w + pad), pad + (i // cols) * (h + bar + pad)
        im = Image.open(png).convert("RGBA").resize((w, h))
        ground = Image.new("RGBA", (w, h), (236, 238, 242, 255))  # what a transparent still sits on
        sheet.paste(Image.alpha_composite(ground, im).convert("RGB"), (x, y + bar))
        d.rectangle([x, y, x + w, y + bar - 2], fill=(150, 40, 40) if problems else (40, 110, 60))
        d.text((x + 8, y + 7), label + ("  -  " + "; ".join(problems) if problems else "  -  ok"), fill=(255, 255, 255))
    sheet.save(dest)


def describe_fix(shot) -> str:
    parts = []
    if shot["swing"]:
        parts.append(f"swung {shot['swing']:+d} deg")
    if shot["rise"]:
        parts.append(f"rose {shot['rise']:.1f} m")
    if shot.get("back", 1) != 1:
        parts.append(f"pulled back x{shot['back']}")
    return ", ".join(parts)


def out_dir(args) -> tuple[Path, Path]:
    spec = Path(args.spec).resolve()
    if not spec.exists():
        sys.exit(f"spec not found: {spec}")
    # spec.json -> out/; any other name gets its own folder, so two specs side by side don't share checks and stills
    default = spec.parent / "out" if spec.stem == "spec" else spec.parent / "out" / spec.stem
    return spec, Path(args.out or default).resolve()


def shot_ids(text: str | None) -> list[str]:
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def still_size(spec: Path, res: str | None, long_side: int = 960) -> str:
    """The size `check` builds at: the film's own shape (the spec's, or --res), scaled down. The solver's answer
    depends on the frame's shape, so a 9:16 film checked at 16:9 would be a different film."""
    try:
        text = res or str(json.loads(spec.read_text(encoding="utf-8")).get("resolution", "1920x1080"))
        w, h = (int(x) for x in text.lower().split("x"))
        k = long_side / max(w, h)
        return f"{max(2, round(w * k / 2) * 2)}x{max(2, round(h * k / 2) * 2)}"
    except (ValueError, OSError, AttributeError, ZeroDivisionError):
        return "960x540"  # the compiler names what is wrong with the spec


def shows_a_picture(spec: dict) -> bool:
    """A laptop, phone or card in the set shows a picture or a video. Its white (a document, a light UI) is the
    picture's own, so it is not counted as blown out."""
    st = spec.get("set") or {}
    screens = [st.get("screen")] + [p.get("screen") for p in st.get("props") or [] if isinstance(p, dict)]
    return any(isinstance(s, dict) and (s.get("image") or s.get("video")) for s in screens)


def check(args) -> int:
    spec, out = out_dir(args)
    try:
        own_white = shows_a_picture(json.loads(spec.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        own_white = False
    try:
        checks = run_compiler(spec, out, "check", still_size(spec, args.res), args.engine,
                              alpha=getattr(args, "alpha", False), shots=getattr(args, "shots", None))
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    wanted = set(shot_ids(getattr(args, "shots", None))) or None
    sheet_rows = []
    for shot in checks["shots"]:
        if wanted and shot["id"] not in wanted:
            shot["passed"], shot["problems"], shot["skipped"] = not shot["issues"], list(shot["issues"]), True
            continue
        png = out / "stills" / f"{shot['id']}.png"
        mean, clipped = exposure(png)
        shot["luma"], shot["clipped"] = round(mean, 3), round(clipped, 4)
        problems = list(shot["issues"])
        if not LUMA_RANGE[0] <= mean <= LUMA_RANGE[1]:
            problems.append(f"exposure {mean:.2f} outside {LUMA_RANGE}")
        if clipped > CLIP_LIMIT and shot["type"] != "screen_fill" and not own_white:  # a screen's picture may be white
            problems.append(f"{clipped:.1%} blown out")
        shot["passed"] = not problems
        shot["problems"] = problems
        note = describe_fix(shot) if shot["adjusted"] else ""
        sheet_rows.append((f"{shot['id']} ({shot['type']}) {note}".strip(), png, problems))
    checks["passed"] = all(s["passed"] for s in checks["shots"]) and checks["ik_max_error_m"] < 0.01
    checks["contact_sheet"] = str(out / "contact_sheet.png")
    contact_sheet(sheet_rows, out / "contact_sheet.png")
    write_json(out / "checks.json", checks)
    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        print(f"{spec.name}: {checks['duration']:.1f}s, {len(checks['shots'])} shots, {checks['frames']} frames at "
              f"{checks['fps']} fps, built in {checks['seconds']}s")
        for s in checks["shots"]:
            if s.get("skipped"):
                continue
            fix = ""
            if s["adjusted"]:
                why = "; ".join(s.get("nominal_issues", [])[:1])
                fix = f"  [camera {describe_fix(s)}: nominal had {why}]"
            print(f"  {'PASS' if s['passed'] else 'FAIL'}  {s['id']:<12} {s['type']:<14} luma {s['luma']:.2f}{fix}")
            for p in s["problems"]:
                print(f"        - {p}")
        print(f"  IK reach error: {checks['ik_max_error_m'] * 1000:.1f} mm (limit 10 mm)")
        print(f"  contact sheet: {checks['contact_sheet']}")
        print("PASSED" if checks["passed"] else "FAILED")
    return 0 if checks["passed"] else 1


def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found: it joins the picture and the sound, and encodes clips (https://ffmpeg.org/download.html)")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def encode_clips(out: Path, checks: dict, wanted: list[str], prores: bool, sound_on: bool, spec_name: str) -> list[dict]:
    """PNG sequences -> one clip per shot (+ its effects as a WAV); returns their manifest entries."""
    import sound
    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)
    fps, alpha = checks["fps"], checks["alpha"]
    events = json.loads((out / "events.json").read_text())["events"]
    made = []
    for shot in checks["shots"]:
        sid = shot["id"]
        if sid not in wanted:
            continue
        f0, f1 = shot["frames"]
        seq = ["-framerate", fps, "-start_number", f0, "-i", out / "frames" / sid / "f_%04d.png"]
        if alpha:
            # VP9 with an alpha plane: Chrome plays it, and Remotion's <OffthreadVideo transparent> reads it
            name = f"{sid}.webm"
            ffmpeg(*seq, "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", 18, "-b:v", 0, "-row-mt", 1,
                   "-auto-alt-ref", 0, clips_dir / name)
        else:
            name = f"{sid}.mp4"
            ffmpeg(*seq, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", 16, "-movflags", "+faststart", clips_dir / name)
        entry = {"id": sid, "file": name, "alpha": alpha, "frames": f1 - f0 + 1, "seconds": round((f1 - f0 + 1) / fps, 4),
                 "fps": fps, "resolution": checks["resolution"], "spec": spec_name,
                 "events": [{"frame": e["frame"] - f0, "kind": e["kind"]} for e in events if f0 <= e["frame"] <= f1]}
        if prores:  # for editors that don't read VP9 alpha (Premiere, Resolve, Final Cut)
            ffmpeg(*seq, "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le" if alpha else "yuv444p10le",
                   clips_dir / f"{sid}.mov")
            entry["prores"] = f"{sid}.mov"
        if sound_on and entry["events"]:
            sound.render(out / "events.json", clips_dir / f"{sid}.wav", start=shot["start"],
                         end=shot["start"] + shot["dur"], score=False, room=False)
            entry["sound"] = f"{sid}.wav"
        made.append(entry)
    return made


def render(args) -> int:
    spec, out = out_dir(args)
    need_ffmpeg()
    if not args.skip_check:
        code = check(argparse.Namespace(spec=args.spec, out=str(out), res=args.res, engine=None, json=False,
                                        alpha=args.alpha, shots=args.shots))
        if code == 2 or (code == 1 and not args.force):
            print("Not rendering: fix the failed checks, or pass --force.")
            return code
    if not (out / "checks.json").exists():
        print("ERROR: --skip-check needs an earlier check of this spec (its checks.json is the shot list): run "
              "`check` first, or drop --skip-check")
        return 2
    plan = json.loads((out / "checks.json").read_text())
    if plan.get("alpha") and not args.clips:
        print('ERROR: this spec is transparent ("transparent": true) and a film is one opaque MP4: render it with --clips')
        return 2
    wanted = shot_ids(args.shots) or [s["id"] for s in plan["shots"]]
    total = sum(s["frames"][1] - s["frames"][0] + 1 for s in plan["shots"] if s["id"] in wanted) if args.clips \
        else plan.get("frames", 0)
    if args.clips:
        for sid in wanted:
            shutil.rmtree(out / "frames" / sid, ignore_errors=True)
    try:
        checks = run_compiler(spec, out, "render", args.res, args.engine, alpha=args.alpha, clips=args.clips,
                              shots=args.shots, total_frames=total)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    rate = f"{checks['seconds']}s" + (f", {checks['seconds'] / total:.2f} s/frame" if total else "")
    if args.clips:
        source = f"{spec.parent.name}/{spec.name}"  # which spec a clip came from, in a manifest several specs share
        made = encode_clips(out, checks, wanted, args.prores, not args.no_sound, source)
        merge_clips(out / "clips" / "clips.json", made, source, whole_spec=not args.shots)
        if not args.keep_frames:
            for sid in wanted:
                shutil.rmtree(out / "frames" / sid, ignore_errors=True)
        dest = out / "clips"
        if args.to:
            dest = Path(args.to).resolve()
            dest.mkdir(parents=True, exist_ok=True)
            for c in made:
                for k in ("file", "sound", "prores"):
                    if c.get(k):
                        shutil.copy(out / "clips" / c[k], dest / c[k])
            # several specs can feed one folder (a room and a transparent desk): keep the other specs' clips
            merge_clips(dest / "clips.json", made, source, whole_spec=not args.shots)
        print(f"rendered {len(made)} clip(s) to {dest} ({total} frames in {rate})")
        for c in made:
            print(f"  {c['file']:<24} {c['frames']} frames, {c['seconds']}s" + (f", sound {c['sound']}" if c.get("sound") else ""))
        return 0
    final = out / f"{spec.stem if spec.stem != 'spec' else spec.parent.name}.mp4"  # a folder may be called v1.2
    heard = json.loads((out / "events.json").read_text(encoding="utf-8"))
    if args.no_sound or not (heard["events"] or heard["score"]):
        # nothing to hear (a studio set has no beats): a silent film, not room tone turned up to full level
        shutil.copy(out / "silent.mp4", final)
        if not args.no_sound:
            print("  no beats and no score: the film is silent (qc.py --render ... --allow-silent)")
    else:
        import sound
        sound.render(out / "events.json", out / "sound.wav")
        ffmpeg("-i", out / "silent.mp4", "-i", out / "sound.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-shortest", final)
    print(f"rendered {final} ({total} frames in {rate})")
    return 0


def open_scene(args) -> int:
    spec, out = out_dir(args)
    blend = out / "scene.blend"
    if not blend.exists() or blend.stat().st_mtime < spec.stat().st_mtime:
        code = check(argparse.Namespace(spec=args.spec, out=str(out), res=None, engine=None, json=False))
        if code == 2:
            return code
    subprocess.Popen([find_blender(), str(blend)])
    print(f"opened {blend} (a copy to look at: it is rebuilt from the spec on every run, so change the spec)")
    return 0


def doctor(args) -> int:
    """Is everything a render needs here, and how long will one take on this machine?"""
    ok = True
    binary = installed_blender()
    if not binary:
        print(f"FAIL  Blender not found: {INSTALL}")
        return 1
    version, label = blender_version(binary)
    if version < MIN_VERSION:
        ok = False
        print(f"FAIL  {label} at {binary}: needs {MIN_VERSION[0]}.{MIN_VERSION[1]} or later")
    else:
        note = "" if version >= (5, 0) else "  (works from 4.2; made and tested on 5.3)"
        print(f"ok    {label} at {binary}{note}")
    if shutil.which("ffmpeg"):
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        print("ok    ffmpeg" + ("" if "libvpx-vp9" in enc else
                                "  (no libvpx-vp9 encoder: --alpha clips need it; --prores still works)"))
    else:
        ok = False
        print("FAIL  ffmpeg not found (joins picture and sound, encodes clips)")
    try:
        import PIL  # noqa: F401
        print("ok    Pillow (contact sheets, exposure)")
    except ImportError:
        ok = False
        print("FAIL  Pillow missing: run through `uv run`, which installs the toolkit's dependencies")
    if not ok or args.no_bench:
        return 0 if ok else 1
    spec = Path(args.spec).resolve() if args.spec else ROOT / "examples" / "blender-disk-panic" / "spec.json"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        try:
            checks = run_compiler(spec, Path(tmp), "render", args.res, args.engine, bench=max(1, args.frames))
        except RuntimeError as e:
            print(f"FAIL  test render: {e}")
            return 1
    b = checks["bench"]
    w, h = checks["resolution"]
    film = b["seconds_per_frame"] * checks["frames"]
    print(f"ok    test render: {b['seconds_per_frame']:.2f} s/frame at {w}x{h} ({checks['engine']}; the first frame "
          f"took {b['first_frame_s']:.1f} s)")
    name = spec.parent.name if spec.stem == "spec" else spec.name
    print(f"      {name}: {checks['frames']} frames, about {film / 60:.0f} min to render here")
    return 0


def main():
    for stream in (sys.stdout, sys.stderr):  # a path or a cue outside the console's code page (Windows cp1252)
        if hasattr(stream, "reconfigure"):   # must not turn an error message into a UnicodeEncodeError
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="build, verify every shot, write stills + contact sheet")
    c.add_argument("spec")
    c.add_argument("--out")
    c.add_argument("--res", help="still resolution (default: the spec's shape, 960 on its long side)")
    c.add_argument("--engine", choices=["eevee", "cycles"])
    c.add_argument("--alpha", action="store_true", help="transparent background (as --alpha on render)")
    c.add_argument("--shots", help="comma-separated shot ids: stills for these only")
    c.add_argument("--json", action="store_true")
    r = sub.add_parser("render", help="check, then render the film with sound, or one clip per shot")
    r.add_argument("spec")
    r.add_argument("--out")
    r.add_argument("--res", help="override the spec's resolution, e.g. 1280x720 for a quick draft")
    r.add_argument("--engine", choices=["eevee", "cycles"])
    r.add_argument("--no-sound", action="store_true")
    r.add_argument("--skip-check", action="store_true")
    r.add_argument("--force", action="store_true", help="render even if a check failed")
    r.add_argument("--clips", action="store_true", help="one clip per shot in out/clips/ with clips.json, not one film")
    r.add_argument("--alpha", action="store_true", help="transparent background: clips are WebM (VP9 + alpha)")
    r.add_argument("--shots", help="comma-separated shot ids: render these only (with --clips)")
    r.add_argument("--to", help="with --clips: also copy the clips and clips.json here (a project's public/blender)")
    r.add_argument("--prores", action="store_true", help="with --clips: also write ProRes 4444 .mov for other editors")
    r.add_argument("--keep-frames", action="store_true", help="with --clips: keep the PNG sequences in out/frames/")
    o = sub.add_parser("open", help="open the built scene in Blender to look round it")
    o.add_argument("spec")
    o.add_argument("--out")
    d = sub.add_parser("doctor", help="check Blender, ffmpeg and Pillow, and time a test render")
    d.add_argument("spec", nargs="?", help="spec to time (default: examples/blender-disk-panic)")
    d.add_argument("--res")
    d.add_argument("--engine", choices=["eevee", "cycles"])
    d.add_argument("--frames", type=int, default=4, help="frames to time (default 4)")
    d.add_argument("--no-bench", action="store_true", help="only check what is installed")
    args = ap.parse_args()
    if args.cmd == "render" and not args.clips and (args.shots or args.to or args.prores or args.alpha):
        ap.error("--alpha, --shots, --to and --prores need --clips (a film is one opaque MP4)")
    sys.exit({"check": check, "render": render, "open": open_scene, "doctor": doctor}[args.cmd](args))


if __name__ == "__main__":
    main()
