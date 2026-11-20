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


COLOUR_LIMIT = 10.0  # how far a part's face may read from its colour in the mark (CIE76 Delta E)


def lab(rgb) -> tuple[float, float, float]:
    """sRGB (0-255) as CIE L*a*b* under D65, for a colour distance that is roughly what the eye sees."""
    lin = [(c / 255 / 12.92) if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in rgb]
    x = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.95047
    y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.08883
    f = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in (x, y, z)]
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def mark_colours(png: Path, mark: dict) -> list[str]:
    """The last-frame test's picture half: does each part of a logo read as its own colour in the rendered
    still? `mark` is the compiler's report (each part's outline in the picture, and its colour). Fills in each
    part's "seen" and "colour_off", and returns the problems in words."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
    im = Image.open(png).convert("RGB")
    w, h = im.size
    problems = []
    for part in mark["parts"]:
        if part.get("off") is None:
            continue
        mask = Image.new("L", (w, h), 0)
        for ring in part["outline"]:  # a second outline inside the first is a hole
            one = Image.new("L", (w, h), 0)
            ImageDraw.Draw(one).polygon([(x * w, y * h) for x, y in ring], fill=255)
            mask = ImageChops.difference(mask, one)
        inner = mask.filter(ImageFilter.MinFilter(2 * max(2, round(min(w, h) * 0.008)) + 1))  # off the bevel
        px = [p for p, m in zip(im.getdata(), (inner if inner.getbbox() else mask).getdata()) if m]
        if not px:
            problems.append(f'{part["id"]} is too small in the frame to read its colour')
            continue
        seen = tuple(sorted(c[i] for c in px)[len(px) // 2] for i in range(3))
        want = tuple(int(part["color"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        a, b = lab(seen), lab(want)
        off = sum((p - q) ** 2 for p, q in zip(a, b)) ** 0.5
        part["seen"], part["colour_off"] = "#%02x%02x%02x" % seen, round(off, 1)
        if off > COLOUR_LIMIT:
            how = ("darker: more light on the stage, or more glow" if a[0] < b[0] - COLOUR_LIMIT / 2 else
                   "lighter or clipped: less light, or a glow of 1 or less" if a[0] > b[0] + COLOUR_LIMIT / 2 else
                   "a different hue: a tinted light, or a view transform other than Standard")
            problems.append(f'{part["id"]} reads {part["seen"]} where the mark has {part["color"]} (off by {off:.0f}, '
                            f'limit {COLOUR_LIMIT:.0f}): {how}')
    return problems


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
        if shot.get("built_only"):  # (built for the acting to run through, and never rendered: not judged)
            shot["passed"], shot["problems"] = not shot["issues"], list(shot["issues"])
            continue
        png = out / "stills" / f"{shot['id']}.png"
        mean, clipped = exposure(png)
        shot["luma"], shot["clipped"] = round(mean, 3), round(clipped, 4)
        problems = list(shot["issues"])
        if shot.get("mark"):
            problems += mark_colours(png, shot["mark"])
        # (a logo on a dark stage is mostly dark, and alone on a transparent ground it may be mostly white: its
        # colours have their own test, so the frame's average says nothing more)
        if not LUMA_RANGE[0] <= mean <= LUMA_RANGE[1] and not shot.get("mark"):
            problems.append(f"exposure {mean:.2f} outside {LUMA_RANGE}")
        # (a screen's picture may be white, and so may a logo: its colours have their own test)
        if clipped > CLIP_LIMIT and shot["type"] != "screen_fill" and not own_white and not shot.get("mark"):
            problems.append(f"{clipped:.1%} blown out")
        shot["passed"] = not problems
        shot["problems"] = problems
        note = describe_fix(shot) if shot["adjusted"] else ""
        sheet_rows.append((f"{shot['id']} ({shot['type']}) {note}".strip(), png, problems))
    checks["passed"] = (all(s["passed"] for s in checks["shots"]) and checks["ik_max_error_m"] < 0.01
                        and not checks.get("body_problems") and not checks.get("set_problems"))
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
            if s.get("built_only"):  # (built for the acting to run through, never rendered: no exposure to read)
                print(f"  ----  {s['id']:<12} {s['type']:<14} built, not rendered")
                continue
            print(f"  {'PASS' if s['passed'] else 'FAIL'}  {s['id']:<12} {s['type']:<14} luma {s['luma']:.2f}{fix}")
            if s.get("mark"):
                seen = ", ".join(f'{p["id"]} {p.get("seen", "not seen")} (off {p.get("colour_off", "-")})' for p in s["mark"]["parts"])
                print(f"        mark at frame {s['mark']['frame']}, {s['mark']['facing_deg']} deg off square: {seen}")
            for p in s["problems"]:
                print(f"        - {p}")
        if checks.get("ik_max_error_m") or any(not s.get("mark") and s["type"] not in ("product", "orbit", "reveal", "ident")
                                               for s in checks["shots"]):
            print(f"  IK reach error: {checks['ik_max_error_m'] * 1000:.1f} mm (limit 10 mm)")
        for note in checks.get("notes", []):
            print(f"  note  {note}")
        for p in checks.get("body_problems", []):  # contact: feet on the floor, hands on things, nothing passed through
            print(f"  FAIL  {p}")
        for p in checks.get("set_problems", []):  # the set itself: faces in one plane, which flicker
            print(f"  FAIL  {p}")
        print(f"  contact sheet: {checks['contact_sheet']}")
        print("PASSED" if checks["passed"] else "FAILED")
    return 0 if checks["passed"] else 1


def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found: it joins the picture and the sound, and encodes clips (https://ffmpeg.org/download.html)")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def encode_clips(out: Path, checks: dict, wanted: list[str], prores: bool, sound_on: bool, spec_name: str,
                 with_sound: bool = False) -> list[dict]:
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
            if with_sound:  # the clip plays by itself: its sound is in the file as well as beside it
                tmp = clips_dir / f"{sid}.tmp{Path(name).suffix}"
                ffmpeg("-i", clips_dir / name, "-i", clips_dir / f"{sid}.wav", "-c:v", "copy",
                       *(("-c:a", "libopus", "-b:a", "128k") if alpha else ("-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart")),
                       tmp)
                os.replace(tmp, clips_dir / name)
                entry["has_sound"] = True
        made.append(entry)
    return made


FLICKER_LEVELS = 4.0  # a still point brighter or darker than the frames either side of it by this (0-255)
FLICKER_COUNT = 5     # this many such frames on one object in a shot is flicker; fewer is noted, not failed (a
                      # hand's reflection crossing a laptop, a shadow passing, change a point for a frame or two)


def gray_frames(video: Path, size: tuple[int, int]):
    """The frames of a video as luma arrays, one at a time."""
    import numpy as np
    w, h = size
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "gray",
                             "-s", f"{w}x{h}", "-"], stdout=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(w * h)
            if len(buf) < w * h:
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w).astype(np.float32)
    finally:
        proc.stdout.close()
        proc.wait()


def flicker(out: Path, videos: dict, size: tuple[int, int]) -> list[dict]:
    """Flicker in rendered video: `check` knows which points of the set stand still and where each is in every
    frame (still_points.json, written at render); a still surface's brightness changes smoothly or not at all,
    so a point that jumps for one frame and comes back is flicker (two faces in one plane, light noise).
    The level is read at the point's exact place, not the nearest pixel's, so a surface with a picture on it
    (a board with writing) under a moving camera is not misread.
    `videos` maps a shot id to (file, the frame number of its first frame). Returns one finding per object and
    shot: {"object", "shot", "frames", "worst", "at"}.

    Lights that change on purpose (a `lights` beat: banks coming on, a tube's stutter) are listed with their frames
    in still_points.json ("light_changes"), and only those frames are left out, with one frame either side: a
    stutter is a jump that comes back, exactly what this looks for, everywhere else."""
    points = json.loads((out / "still_points.json").read_text())
    excused = {f for a, b in points.get("light_changes", []) for f in range(a - 1, b + 2)}
    w, h = size
    found = []
    for sid, (video, first) in videos.items():
        shot = points["shots"].get(sid)
        if not shot:
            continue
        if not Path(video).exists():  # (said, not skipped quietly: an empty result here read as a clean render)
            print(f"  not read for flicker: {sid}'s video {video} does not exist")
            continue
        f0, f1 = shot["frames"]
        by_frame = {}
        for row in shot["points"]:
            by_frame.setdefault(row[0], []).append(row)

        def level(im, x, y):
            # a 5x5 patch centred on the point itself, between pixels: on a whole-pixel patch a camera move
            # steps the patch a pixel at a time, and over lettering or a picture each step reads as a jump
            fx, fy = x * w - 0.5, y * h - 0.5
            px, py = min(w - 4, max(2, int(fx // 1))), min(h - 4, max(2, int(fy // 1)))
            ax, ay = min(1.0, max(0.0, fx - px)), min(1.0, max(0.0, fy - py))
            a = im[py - 2:py + 4, px - 2:px + 4]
            a = (1 - ay) * a[:-1] + ay * a[1:]
            return float(((1 - ax) * a[:, :-1] + ax * a[:, 1:]).mean())

        spikes, window = {}, {}
        for i, im in enumerate(gray_frames(Path(video), size)):
            f = first + i
            if f > f1 + 1:
                break
            window[f] = im
            window.pop(f - 3, None)
            mid = f - 1  # the frame with a neighbour on each side now
            if mid - 1 not in window or mid in excused:
                continue
            for _f, k, x, y, xa, ya, xb, yb in by_frame.get(mid, ()):
                a, b, c = level(window[mid - 1], xa, ya), level(window[mid], x, y), level(window[f], xb, yb)
                jump = min(abs(b - a), abs(b - c)) if (b - a) * (b - c) > 0 else 0.0
                if jump >= FLICKER_LEVELS:
                    spikes.setdefault(points["objects"][k], {}).setdefault(mid, 0.0)
                    spikes[points["objects"][k]][mid] = max(spikes[points["objects"][k]][mid], jump)
        for name, frames in spikes.items():
            if len(frames) >= 2:
                worst = max(frames, key=frames.get)
                found.append({"object": name, "shot": sid, "frames": sorted(frames), "worst": round(frames[worst], 1),
                              "at": worst, "fails": len(frames) >= FLICKER_COUNT})
    return sorted(found, key=lambda r: -len(r["frames"]))


def report_flicker(out: Path, videos: dict, size) -> bool:
    """Measure, print and save (render_checks.json); True when the render is clean."""
    if not (out / "still_points.json").exists():
        return True
    found = flicker(out, videos, size)
    failed = [r for r in found if r["fails"]]
    changes = json.loads((out / "still_points.json").read_text()).get("light_changes", [])
    result = {"passed": not failed, "flicker": failed, "flicker_notes": [r for r in found if not r["fails"]]}
    if changes:  # (said, so a reader knows which frames the check did not read)
        result["not_read"] = {"frames": changes, "why": "lights change on purpose here (a lights beat: banks coming "
                                                        "on, each tube's stutter); one frame either side is left out too"}
        print(f"  not read for flicker: {len(changes)} light changes on purpose, frames "
              + ", ".join(f"{a}-{b}" for a, b in changes))
    write_json(out / "render_checks.json", result)
    for r in failed:
        print(f"  FAIL  {r['object']} flickers in {r['shot']}: {len(r['frames'])} frames jump and come back (worst "
              f"{r['worst']:.0f} levels at frame {r['at']}). Two faces in one plane, or light noise: see docs/blender.md")
    for r in found:
        if not r["fails"]:
            print(f"  note  {r['object']} in {r['shot']}: {len(r['frames'])} frames jump (worst {r['worst']:.0f} levels at "
                  f"frame {r['at']}): look if it is not a reflection or a shadow passing")
    if not failed:
        print("  no flicker: every still surface the cameras see changes smoothly")
    return not failed


HIT_REACH = 8  # frames either side of a hit that are searched for the picture's biggest change


def hit_frames(video: Path, first: int, hits: list[int], size=(160, 90)) -> list[dict]:
    """For each hit (a frame number in the film), the frame nearby where the picture changes most from the one
    before it. A hit is seen on the frame of the biggest change, so its sound belongs on that frame; when the
    two differ the sound will be heard early or late. `first` is the frame number of the video's first frame."""
    import numpy as np
    if not hits:
        return []
    change, prev = {}, None
    for i, im in enumerate(gray_frames(video, size)):
        f = first + i
        if prev is not None:
            change[f] = float(np.abs(im - prev).mean())
        prev = im
    out = []
    for h in hits:
        near = {f: c for f, c in change.items() if abs(f - h) <= HIT_REACH}
        if not near:
            continue
        seen = max(near, key=near.get)
        out.append({"frame": h, "seen": seen, "change": round(near[seen], 2), "at_hit": round(near.get(h, 0.0), 2),
                    "passed": abs(seen - h) <= 1})
    return out


def report_hits(out: Path, videos: dict, events: list[dict], fps: int) -> bool:
    """Check every hit against its clip, print it and add it to render_checks.json; True when all are on time."""
    found = []
    for sid, (video, first) in videos.items():
        hits = [e["frame"] for e in events if e.get("hit") and e.get("shot") == sid]
        for r in hit_frames(Path(video), first, hits):
            found.append({"shot": sid, **r})
    if not found:
        return True
    path = out / "render_checks.json"
    data = json.loads(path.read_text()) if path.exists() else {"passed": True}
    bad = [r for r in found if not r["passed"]]
    data["hits"], data["passed"] = found, bool(data.get("passed", True)) and not bad
    write_json(path, data)
    for r in found:
        if r["passed"]:
            print(f"  hit in {r['shot']} at frame {r['frame']}: the picture's biggest change is on frame {r['seen']}")
        else:
            print(f"  FAIL  hit in {r['shot']} is on frame {r['frame']} and the picture changes most on frame {r['seen']} "
                  f"({(r['seen'] - r['frame']) / fps:+.2f} s): its sound will be heard "
                  f"{'early' if r['seen'] > r['frame'] else 'late'}. Move the hit, or what follows it")
    return not bad


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
        made = encode_clips(out, checks, wanted, args.prores, not args.no_sound, source, args.with_sound)
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
        starts = {s["id"]: s["frames"][0] for s in checks["shots"]}
        videos = {c["id"]: (out / "clips" / c["file"], starts[c["id"]]) for c in made}
        clean = report_flicker(out, videos, tuple(checks["resolution"]))
        heard = json.loads((out / "events.json").read_text(encoding="utf-8"))["events"]
        clean = report_hits(out, videos, heard, checks["fps"]) and clean
        return 0 if clean else 1
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
    clean = report_flicker(out, {s["id"]: (out / "silent.mp4", 1) for s in checks["shots"]}, tuple(checks["resolution"]))
    clean = report_hits(out, {"film": (out / "silent.mp4", 1)}, [{**e, "shot": "film"} for e in heard["events"]],
                        checks["fps"]) and clean
    return 0 if clean else 1


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


def fit(args) -> int:
    """Bind a generated character model to the cast's rig (lib/blender/fit_model.py), and say what it made."""
    model, fit_json, out = Path(args.model), Path(args.fit), Path(args.out)
    for path in (model, fit_json):
        if not path.exists():
            print(f"not found: {path}")
            return 2
    cmd = [find_blender(), "-b", "-noaudio", "-P", str(ROOT / "lib" / "blender" / "fit_model.py"), "--",
           "--model", str(model), "--fit", str(fit_json), "--out", str(out)]
    if args.preview:
        Path(args.preview).mkdir(parents=True, exist_ok=True)
        cmd += ["--preview", args.preview]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith(("fit:", "FIT "))]
    print("\n".join(lines))
    if not any(ln.startswith("FIT ") for ln in lines):
        print((done.stdout + done.stderr)[-1500:])
        return 2
    if any(" FAIL " in ln for ln in lines):
        return 1
    print(f'fitted: name it in a character file with "model": "{out.name}"'
          + (f", and look at {args.preview}/fit-front.png" if args.preview else ""))
    return 0


def reduce(args) -> int:
    """A smaller copy of a generated model, to keep in a repository (lib/blender/reduce_model.py)."""
    model = Path(args.model)
    if not model.exists():
        print(f"not found: {model}")
        return 2
    cmd = [find_blender(), "-b", "-noaudio", "-P", str(ROOT / "lib" / "blender" / "reduce_model.py"), "--",
           "--model", str(model), "--out", str(args.out), "--faces", str(args.faces), "--texture", str(args.texture)]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("REDUCED ")]
    if not lines:
        print((done.stdout + done.stderr)[-1500:])
        return 2
    print(lines[-1])
    return 0


SHEET_HEADS = (("front", 0), ("threeq-left", 40), ("threeq-right", -40), ("profile-left", 90), ("profile-right", -90),
               ("back", 180))
SHEET_FACES = (("rest", 0.5), ("smile", 1.7), ("blink", 2.6), ("glance", 3.8), ("wink", 4.5), ("grin", 5.8),
               ("shocked", 6.6))
SHEET_POSES = (("arms-up", 8.0, 0), ("arms-up-back", 8.0, 160), ("on-head", 9.2, 0), ("on-head-side", 9.2, 70),
               ("behind-head", 10.4, 0), ("behind-head-back", 10.4, 150))
SHEET_SEATED = (("sitting-down", 15.3, 100), ("sit", 16.9, 45), ("sit-side", 16.9, 90), ("type", 20.6, 45),
                ("type-side", 20.6, 90), ("type-front", 20.6, 10))


def sheet_spec(cast_file: Path, sit_at: float) -> dict:
    """The film a review sheet is cut from: every face, the poses that stretch a body most, then down to a desk."""
    return {
        "title": "Review sheet", "fps": 24, "resolution": "1280x720", "engine": "eevee",
        "set": {"kind": "desk", "chair_out": 0.75},
        "cast": [{"id": "lead", "from": str(cast_file), "posture": "stand", "at": [1.9, sit_at], "facing": 0}],
        "shots": [{"id": "A", "dur": 21.0, "type": "full", "follow": True}],
        "beats": [
            {"at": 1.0, "do": "expression", "face": "happy", "dur": 0.3},
            {"at": 2.0, "do": "expression", "face": "neutral", "dur": 0.3},
            {"at": 2.6, "do": "blink", "dur": 0.6},
            {"at": 3.4, "do": "glance", "yaw": 22, "pitch": 0, "dur": 0.6},
            {"at": 4.4, "do": "wink"},
            {"at": 5.2, "do": "expression", "face": "grin", "dur": 0.3},
            {"at": 6.0, "do": "expression", "face": "shocked", "dur": 0.3},
            {"at": 6.8, "do": "expression", "face": "neutral", "dur": 0.3},
            {"at": 7.0, "do": "pose", "pose": "arms_up", "dur": 0.5},
            {"at": 8.2, "do": "pose", "pose": "hands_on_head", "dur": 0.5},
            {"at": 9.4, "do": "pose", "pose": "hands_behind_head", "dur": 0.5},
            {"at": 10.6, "do": "pose", "pose": "down", "dur": 0.5},
            {"at": 11.4, "do": "walk", "to": [0, sit_at], "facing": 0, "until": 14.4},
            {"at": 14.6, "do": "sit_down", "dur": 1.8},
            {"at": 16.9, "do": "pull_in", "dur": 1.5},
            {"at": 18.6, "do": "pose", "pose": "type", "dur": 0.6},
            {"at": 19.3, "do": "type", "speed": "calm", "until": "end"},
        ]}


def sheet_views() -> list:
    f = lambda s: int(round(s * 24)) + 1  # noqa: E731
    views = []
    for name, az in SHEET_HEADS:
        views.append({"name": "head-" + name, "frame": f(0.5), "on": "head", "az": az, "dist": 3.0, "fill": 0.62,
                      "res": [720, 720]})
        views.append({"name": "body-" + name, "frame": f(0.5), "on": "body", "az": az, "dist": 1.7, "fill": 0.86,
                      "res": [540, 900]})
    views += [{"name": "face-" + name, "frame": f(t), "on": "head", "az": 0, "dist": 3.0, "fill": 0.7,
               "res": [560, 560]} for name, t in SHEET_FACES]
    views += [{"name": "pose-" + name, "frame": f(t), "on": "body", "az": az, "dist": 1.7, "fill": 0.8,
               "res": [640, 800]} for name, t, az in SHEET_POSES]
    views += [{"name": "pose-" + name, "frame": f(t), "on": "body", "az": az, "elev": 12, "dist": 2.2, "fill": 0.8,
               "res": [640, 800], "alone": False} for name, t, az in SHEET_SEATED]
    return views


def sheet_rows(folder: Path, names: list, cell: int, title: str):
    """One row of the sheet: the named stills side by side at one height, each with its name."""
    from PIL import Image, ImageDraw
    ims = []
    for n in names:
        im = Image.open(folder / f"{n}.png").convert("RGB")
        im = im.resize((round(im.width * cell / im.height), cell), Image.LANCZOS)
        ImageDraw.Draw(im).text((6, 4), n, fill=(20, 20, 20))
        ims.append(im)
    row = Image.new("RGB", (sum(i.width for i in ims), cell + 22), (245, 245, 242))
    ImageDraw.Draw(row).text((6, 5), title, fill=(20, 20, 20))
    x = 0
    for im in ims:
        row.paste(im, (x, 22))
        x += im.width
    return row


def sheet_pairs(folder: Path, marks: dict, photos_file: Path) -> list:
    """Each photo beside the still from its angle, and the two laid over each other, with the crown and the chin
    of both on the same lines. `photos.json` names the photo for a view and where its crown, chin and eyes are,
    in the photo's pixels (read off the photo with a grid over it)."""
    from PIL import Image, ImageDraw
    photos = json.loads(photos_file.read_text(encoding="utf-8"))
    made = []
    for view, ph in photos.items():
        m = marks.get(view)
        if not m or "chin" not in m:
            print(f"  skipped {view}: no still of that name with a chin mark (names: head-front, head-threeq-left...)")
            continue
        still = Image.open(folder / f"{view}.png").convert("RGB")
        photo = Image.open(photos_file.parent / ph["file"]).convert("RGB")
        crown, chin = m["head"]["top"], m["chin"][1]
        k = (chin - crown) / (ph["chin"] - ph["crown"])
        photo = photo.resize((round(photo.width * k), round(photo.height * k)), Image.LANCZOS)
        mid_still = sum(e[0] for e in m["eyes"]) / 2
        mid_photo = k * (sum(e[0] for e in ph["eyes"]) / 2 if "eyes" in ph else ph["centre"])
        placed = Image.new("RGB", still.size, (128, 128, 128))
        placed.paste(photo, (round(mid_still - mid_photo), round(crown - k * ph["crown"])))
        over = Image.blend(placed, still, 0.5)
        pair = Image.new("RGB", (still.width * 3, still.height), (0, 0, 0))
        for i, im in enumerate((placed, still, over)):
            d = ImageDraw.Draw(im)
            for y in (crown, chin):
                d.line([(0, y), (im.width, y)], fill=(255, 220, 0), width=1)
            pair.paste(im, (i * still.width, 0))
        out = folder / f"pair-{view}.jpg"
        pair.save(out, quality=90)
        made.append(out)
    return made


def sheet(args) -> int:
    """A review sheet of one character: every angle, every face, the poses that stretch it, and the numbers."""
    cast_file = Path(args.cast).resolve()
    if not cast_file.exists():
        print(f"not found: {cast_file}")
        return 2
    out = Path(args.out).resolve() if args.out else cast_file.parent / "sheet"
    out.mkdir(parents=True, exist_ok=True)
    spec = out / "sheet-spec.json"
    write_json(spec, sheet_spec(cast_file, args.sit_at))
    try:
        checks = run_compiler(spec, out, "check", None, None)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 2
    write_json(out / "views.json", sheet_views())
    done = subprocess.run([find_blender(), "-b", "-noaudio", str(out / "scene.blend"), "-P",
                           str(ROOT / "lib" / "blender" / "sheet.py"), "--", str(out / "views.json"), str(out / "stills")],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "SHEET_DONE" not in done.stdout:
        print((done.stdout + done.stderr)[-1500:])
        return 2
    from PIL import Image
    marks = json.loads((out / "stills" / "marks.json").read_text())
    rows = [sheet_rows(out / "stills", ["head-" + n for n, _ in SHEET_HEADS], 420, "Head, six angles"),
            sheet_rows(out / "stills", ["body-" + n for n, _ in SHEET_HEADS], 520, "Body, six angles"),
            sheet_rows(out / "stills", ["face-" + n for n, _ in SHEET_FACES], 360, "Faces"),
            sheet_rows(out / "stills", ["pose-" + n for n, _, _ in SHEET_POSES], 440, "Arms overhead"),
            sheet_rows(out / "stills", ["pose-" + n for n, _, _ in SHEET_SEATED], 440, "Sitting and typing")]
    page = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (245, 245, 242))
    y = 0
    for r in rows:
        page.paste(r, (0, y))
        y += r.height
    page.save(out / "sheet.jpg", quality=88)
    # the model's own proportions, from where its landmarks fall in the stills (each a share of the head's height)
    hf, bf = marks["head-front"], marks["body-front"]
    measures = {}
    if "chin" in hf:
        tall = hf["chin"][1] - hf["head"]["top"]
        measures["head"] = {
            "eye_line": round((hf["eyes"][0][1] - hf["head"]["top"]) / tall, 3),
            "eyes_apart": round((hf["eyes"][1][0] - hf["eyes"][0][0]) / tall, 3),
            "width": round((hf["head"]["right"] - hf["head"]["left"]) / tall, 3)}
        if "glasses" in hf:
            g = hf["glasses"]
            measures["glasses"] = {"width": round((g["right"] - g["left"]) / tall, 3),
                                   "height": round((g["bottom"] - g["top"]) / tall, 3),
                                   "eye_in_frame": round((hf["eyes"][0][1] - g["top"]) / (g["bottom"] - g["top"]), 3)}
    if "chin" in bf:
        j, tall = bf["joints"], bf["body"]["bottom"] - bf["body"]["top"]
        measures["body"] = {
            "heads_tall": round(tall / (bf["chin"][1] - bf["head"]["top"]), 2),
            "shoulders": round((j["shoulder_l"][0] - j["shoulder_r"][0]) / tall, 3),
            "arm": round((j["wrist_l"][1] - j["shoulder_l"][1]) / tall, 3),
            "leg": round((j["ankle_l"][1] - j["hip_l"][1]) / tall, 3)}
    problems = checks.get("body_problems") or [i for s in checks.get("shots", []) for i in s.get("issues", [])]
    write_json(out / "sheet.json", {"measures": measures, "skin": checks.get("skin"), "face": checks.get("face"),
                                    "ik_max_error_m": checks.get("ik_max_error_m"),
                                    "hand_in_body_m": checks.get("hand_in_body_m"), "problems": problems})
    print(f"sheet: {out / 'sheet.jpg'}")
    for part, vals in measures.items():
        print(f"  {part}: " + ", ".join(f"{k} {v}" for k, v in vals.items()))
    for line in problems:
        print(f"  {line}")
    if args.photos:
        for made in sheet_pairs(out / "stills", marks, Path(args.photos).resolve()):
            print(f"  pair: {made}")
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
    r.add_argument("--with-sound", action="store_true",
                   help="with --clips: put each clip's sound in its file too (an ident that plays by itself)")
    o = sub.add_parser("open", help="open the built scene in Blender to look round it")
    o.add_argument("spec")
    o.add_argument("--out")
    f = sub.add_parser("fit", help="experimental: bind a generated 3D character (GLB) to the cast's rig")
    f.add_argument("--model", required=True, help="one textured mesh, standing in an A-pose, facing -Y")
    f.add_argument("--fit", required=True, help="fit.json: its joints, clothes line and face (docs/blender.md)")
    f.add_argument("--out", required=True, help="the fitted .blend, named by a character file's \"model\"")
    f.add_argument("--preview", help="folder for three stills of the fitted model")
    m = sub.add_parser("reduce", help="experimental: a smaller copy of a generated model, to keep in a repository")
    m.add_argument("--model", required=True, help="the model as generated (GLB)")
    m.add_argument("--out", required=True, help="the smaller GLB")
    m.add_argument("--faces", type=int, default=60000, help="faces to keep (0: all of them)")
    m.add_argument("--texture", type=int, default=1024, help="the base colour texture's size in pixels")
    sh = sub.add_parser("sheet", help="a review sheet of one character: every angle, face and stretching pose")
    sh.add_argument("cast", help="the character file (a fitted model or a clay description)")
    sh.add_argument("--out", help="folder for the sheet (default: sheet/ beside the character file)")
    sh.add_argument("--photos", help="photos.json: reference photos by view, with crown, chin and eyes in pixels")
    sh.add_argument("--sit-at", type=float, default=0.41, help="how far in front of the desk they stand to sit (m)")
    d = sub.add_parser("doctor", help="check Blender, ffmpeg and Pillow, and time a test render")
    d.add_argument("spec", nargs="?", help="spec to time (default: examples/blender-disk-panic)")
    d.add_argument("--res")
    d.add_argument("--engine", choices=["eevee", "cycles"])
    d.add_argument("--frames", type=int, default=4, help="frames to time (default 4)")
    d.add_argument("--no-bench", action="store_true", help="only check what is installed")
    args = ap.parse_args()
    if args.cmd == "render" and not args.clips and (args.shots or args.to or args.prores or args.alpha or args.with_sound):
        ap.error("--alpha, --shots, --to, --prores and --with-sound need --clips (a film is one opaque MP4)")
    sys.exit({"check": check, "render": render, "open": open_scene, "doctor": doctor, "fit": fit, "reduce": reduce,
              "sheet": sheet}[args.cmd](args))


if __name__ == "__main__":
    main()
