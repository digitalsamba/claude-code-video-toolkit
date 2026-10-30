#!/usr/bin/env python3
"""
Pre-render QC: catch the problems that otherwise show up in the final render.

Run it from a project directory before `npm run render` (or before build.py).
It checks, without changing anything:

  * assets    — media files referenced by the config that don't exist
                (Remotion fails the render on a 404 staticFile)
  * timing    — per-scene narration longer than its scene (cut off) or much
                shorter (dead air). Reuses sync_timing.py's config parsing.
  * pacing    — words-per-minute per scene, from the .txt next to each .mp3
                (same labels voiceover.py prints; see pacing.py)
  * loudness  — per-scene integrated LUFS + true peak via ffmpeg `loudnorm`:
                clipping risk, and scenes noticeably louder/quieter than the rest
  * render    — with --render, the finished video's loudness against the
                YouTube target (-14 LUFS integrated, -1 dBTP). Nothing in the
                Remotion templates normalises loudness, so check it here.
                A missing or silent audio track is an error.
  * frames    — with --render, what the picture does: stretches over 3s
                where nothing on screen changes (dead air for the eye), black
                gaps mid-video, and an empty first frame (the cover image on
                Shorts and Reels). Thresholds from motion-craft.md.
  * layout    — with --layout, the project is bundled and opened in a browser
                (tools/qc_layout.mjs) once a second: text off the canvas or
                clipped by its box, text that never reaches 2.6% of the short
                side (28px at 1080p), and contrast under WCAG AA, grouped by
                colour pair with a passing colour suggested. Text inside
                data-qc="product" (Stage's screen) is reported separately, as
                info: it is the product's own UI, not the video's choice.

`--fix` normalises loudness (and only loudness): out-of-range scene files are
rewritten in place with originals kept in `.qc-backup/`, and a render is
written to a new `<name>.loudnorm.mp4` with the video stream copied.

Understands Remotion projects (src/config/*-config.ts, audio in
public/audio/scenes) and concept-explainer-short projects (scenes.json,
audio in audio/scenes).

Usage:
    uv run tools/qc.py --project projects/my-video
    uv run tools/qc.py --project projects/my-video --render projects/my-video/out/video.mp4
    uv run tools/qc.py --project projects/my-video --fix
    uv run tools/qc.py --project projects/my-video --layout
    uv run tools/qc.py --project projects/my-video --json

Exit code: 0 when there are no errors (warnings allowed, unless --strict),
1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from pacing import pace_label
from sync_timing import (
    detect_config_file,
    detect_template_type,
    get_audio_duration,
    match_audio_to_scenes,
    parse_scenes_from_config,
    scan_audio_files,
)
from win_encoding import ensure_utf8_output

MEDIA_EXTS = (
    ".mp3", ".wav", ".m4a", ".aac", ".ogg",
    ".mp4", ".mov", ".webm", ".mkv",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
)
# Remotion templates prefix config filenames with a folder at the call site
# (staticFile(`audio/${file}`), `demos/${videoFile}`, `images/${...}`), so a
# bare filename can legitimately live in any of these under public/.
PUBLIC_SUBDIRS = ("", "audio", "demos", "images", "videos", "video")
# Config keys the templates check with getStaticFiles() and skip when missing:
# the render succeeds, the file is just absent. Anything else 404s the render.
OPTIONAL_ASSET_KEYS = {"backgroundMusicFile", "chimeFile", "screenshotFile", "voiceoverFile"}

DEAD_AIR_SECONDS = 3.0      # scene runs this much longer than its narration
YOUTUBE_LUFS = -14.0
YOUTUBE_TRUE_PEAK = -1.0
SCENE_LUFS = -16.0          # leaves headroom for music + sfx in the mix
SCENE_TRUE_PEAK = 0.0       # stems are allowed near full scale; the render gets the -1 rule
SCENE_SPREAD_LU = 1.5       # scene this far from the median sounds uneven
FIX_TOLERANCE_LU = 1.0
MIN_LOUDNESS_SECONDS = 3.0  # loudnorm's integrated measure is unreliable below this

# Frame checks run on a small greyscale copy of the render, sampled at STILL_FPS.
FRAME_WIDTH = 320
STILL_FPS = 4
STILL_TILE = 20             # px at FRAME_WIDTH: a cursor or a talking-head PiP fills a tile
STILL_TILE_DIFF = 1.0       # mean abs change (0-255) in the busiest tile; encoder noise sits below
MAX_STILL_SECONDS = 3.0     # motion-craft.md: longer than this with nothing changing reads as a stall
STILL_END_GRACE = 3.0       # the closing card may hold
BLACK_MIN_SECONDS = 0.5     # a dip to black at a cut is a few frames; this is a gap
EMPTY_EDGE_FRACTION = 0.005 # under 0.5% edge pixels: background only, maybe a disclaimer line

# --layout: text measured in the page itself (tools/qc_layout.mjs)
LAYOUT_EVERY = 1.0          # seconds between samples
TEXT_FLOOR = 0.026          # of the short side: 28px at 1080p, the smallest text a phone viewer reads
LAYOUT_PERSIST = 2          # consecutive samples before overflow counts; entrances and exits pass through
LAYOUT_EDGE_PX = 2
IGNORED_REQUESTS = ("/events",)  # Remotion's studio event stream, absent from a bundle


# ─── Findings ────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.findings: list[dict] = []
        self.data: dict = {}

    def add(self, level: str, check: str, message: str, **extra):
        self.findings.append({"level": level, "check": check, "message": message, **extra})

    def count(self, level: str) -> int:
        return sum(1 for f in self.findings if f["level"] == level)


# ─── Loudness ────────────────────────────────────────────────

def _num(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v in (float("inf"), float("-inf")) else v


def measure_loudness(path: Path) -> dict | None:
    """Integrated LUFS, true peak and LRA via ffmpeg loudnorm's first pass."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
             "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr)
    if not m:
        return None
    raw = json.loads(m.group(0))
    return {
        "lufs": _num(raw.get("input_i")),
        "true_peak": _num(raw.get("input_tp")),
        "lra": _num(raw.get("input_lra")),
        "thresh": _num(raw.get("input_thresh")),
        "offset": _num(raw.get("target_offset")),
        "_raw": raw,
    }


def normalize_loudness(src: Path, dst: Path, target: float, true_peak: float,
                       measured: dict, video: bool = False) -> str | None:
    """Second loudnorm pass using the first pass's measurements. Returns an error or None."""
    raw = measured["_raw"]
    af = (
        f"loudnorm=I={target}:TP={true_peak}:LRA=11"
        f":measured_I={raw['input_i']}:measured_TP={raw['input_tp']}"
        f":measured_LRA={raw['input_lra']}:measured_thresh={raw['input_thresh']}"
        f":offset={raw['target_offset']}:linear=true"
    )
    # loudnorm resamples to 192 kHz internally; bring it back down.
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", "48000"]
    if video:
        cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    elif dst.suffix.lower() == ".mp3":
        cmd += ["-b:a", "192k"]
    cmd.append(str(dst))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)
    return r.stderr.strip()[-300:] if r.returncode != 0 else None


# ─── Project loading ─────────────────────────────────────────

def _strip_comments(text: str) -> str:
    """Drop // and /* */ comments from TS source, leaving string contents alone."""
    out, i, n = [], 0, len(text)
    quote = None
    while i < n:
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"`":
            quote = c
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def referenced_media(ts_text: str) -> list[tuple[str | None, str]]:
    """(key, path) for every string literal in the (uncommented) source naming a media file."""
    refs = []
    for m in re.finditer(r"(?:(\w+)\s*:\s*)?['\"`]([^'\"`\n$]+)['\"`]", _strip_comments(ts_text)):
        value = m.group(2).strip()
        if value.lower().endswith(MEDIA_EXTS) and not re.match(r"^[a-z]+://", value):
            refs.append((m.group(1), value))
    return list(dict.fromkeys(refs))


def resolve_public(public_dir: Path, ref: str) -> Path | None:
    ref = ref.lstrip("/")
    for sub in PUBLIC_SUBDIRS:
        p = public_dir / sub / ref
        if p.exists():
            return p
    return None


def load_remotion(project: Path, report: Report, config_override: str | None):
    """Returns (audio_dir, [{label, audio_path, config_seconds}]) for a Remotion project."""
    config_path = Path(config_override) if config_override else detect_config_file(project)
    if not config_path or not config_path.exists():
        return None
    config_text = config_path.read_text(encoding="utf-8")
    template_type = detect_template_type(config_text, config_path)
    report.data["config"] = str(config_path)
    report.data["template_type"] = template_type

    # Assets: the scene config plus brand.ts (logos)
    public_dir = project / "public"
    sources = [config_path] + [p for p in [config_path.parent / "brand.ts"] if p.exists()]
    checked = 0
    for src in sources:
        for key, ref in referenced_media(src.read_text(encoding="utf-8")):
            checked += 1
            if resolve_public(public_dir, ref):
                continue
            if key in OPTIONAL_ASSET_KEYS:
                report.add("warn", "assets", f"{ref} ({key}) — not found under public/; the template "
                           "skips it, so the render will be without it", file=ref)
            else:
                report.add("error", "assets", f"{ref} — referenced in {src.name}, not found under public/",
                           file=ref)
    report.data["assets_checked"] = checked

    audio_dir = public_dir / "audio" / "scenes"
    scenes = parse_scenes_from_config(config_text, template_type)
    if not audio_dir.is_dir() or not scenes:
        return audio_dir, []

    matches = match_audio_to_scenes(scan_audio_files(audio_dir), scenes)
    # When the config names its audio, trust only those pairings — index/name
    # guessing pairs a silent credits scene with whatever file shares its number.
    explicit = any(s.get("audioFile") for s in scenes)
    timed = []
    for m in matches:
        if not m["audio"] or (explicit and m["match_method"] != "audioFile"):
            continue
        timed.append({
            "label": f"{m['scene_index'] + 1:02d} {m['scene'].get('type', 'scene')}",
            "audio_path": Path(m["audio"]["path"]),
            "audio_seconds": m["audio"]["duration_seconds"],
            "config_seconds": m["scene"].get("durationSeconds"),
        })
    return audio_dir, timed


def load_scenes_json(project: Path, report: Report):
    """concept-explainer-short: scenes.json + config.json, audio-anchored (no timing check).

    Returns (audio_dir, timed, narration) where narration maps each scene's mp3 to its text.
    """
    data = json.loads((project / "scenes.json").read_text(encoding="utf-8"))
    report.data["config"] = str(project / "scenes.json")
    report.data["template_type"] = "concept-explainer-short"
    refs = [s["asset"] for s in data.get("scenes", []) if s.get("asset")]
    cfg_path = project / "config.json"
    if cfg_path.exists():
        music = json.loads(cfg_path.read_text(encoding="utf-8")).get("music", {}).get("file")
        if music:
            refs.append(music)
    for ref in refs:
        if not (project / ref).exists():
            # build.py renders a placeholder card for a missing visual, so it won't crash
            report.add("warn", "assets", f"{ref} — not found (build.py will render a placeholder)", file=ref)
    report.data["assets_checked"] = len(refs)
    # gen_vo.py passes the narration inline, so the text lives here, not in .txt files
    audio_dir = project / "audio" / "scenes"
    narration = {audio_dir / f"{s['id']}_{s['slug']}.mp3": s.get("text", "")
                 for s in data.get("scenes", []) if s.get("id") and s.get("slug")}
    return audio_dir, [], narration


# ─── Checks ──────────────────────────────────────────────────

def check_timing(timed: list[dict], report: Report):
    for t in timed:
        audio, scene = t["audio_seconds"], t["config_seconds"]
        if audio is None or scene is None:
            continue
        if audio > scene:
            report.add("error", "timing",
                       f"{t['label']}: narration {audio:.1f}s > scene {scene:.1f}s — will be cut off",
                       file=t["audio_path"].name)
        elif scene - audio > DEAD_AIR_SECONDS:
            report.add("warn", "timing",
                       f"{t['label']}: scene {scene:.1f}s, narration {audio:.1f}s — "
                       f"{scene - audio:.1f}s of trailing silence",
                       file=t["audio_path"].name)
    report.data["scenes_timed"] = len(timed)


def scene_audio_files(audio_dir: Path, timed: list[dict]) -> list[Path]:
    """The scene VO files in play: the config's own when known, else everything in the dir."""
    if timed:
        return [t["audio_path"] for t in timed]
    if not audio_dir.is_dir():
        return []
    return sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a"))


def check_pacing(files: list[Path], narration: dict[Path, str], report: Report):
    """wpm per scene from its narration text: scenes.json, else the .txt voiceover.py read."""
    measured = 0
    for f in files:
        text = narration.get(f)
        if text is None:
            txt = f.with_suffix(".txt")
            if not txt.exists():
                continue
            text = txt.read_text(encoding="utf-8")
        wpm, label = pace_label(text, get_audio_duration(str(f)))
        if wpm is None:
            continue
        measured += 1
        if label in ("fast", "slow"):
            report.add("warn", "pacing", f"{f.name}: {wpm:.0f} wpm ({label}; comfortable is 140-160)",
                       file=f.name, wpm=wpm)
    report.data["pacing_measured"] = measured


def check_scene_loudness(files: list[Path], report: Report) -> dict[Path, dict]:
    levels: dict[Path, dict] = {}
    for f in files:
        dur = get_audio_duration(str(f))
        if dur is not None and dur < MIN_LOUDNESS_SECONDS:
            continue
        m = measure_loudness(f)
        if m and m["lufs"] is not None:
            levels[f] = m
    report.data["scene_loudness"] = {
        f.name: {"lufs": m["lufs"], "true_peak": m["true_peak"]} for f, m in levels.items()
    }
    if not levels:
        return levels

    for f, m in levels.items():
        if m["true_peak"] is not None and m["true_peak"] > SCENE_TRUE_PEAK:
            report.add("warn", "loudness", f"{f.name}: true peak {m['true_peak']:+.1f} dBTP — clipped",
                       file=f.name)
    if len(levels) >= 3:
        median = statistics.median(m["lufs"] for m in levels.values())
        report.data["scene_loudness_median"] = median
        for f, m in levels.items():
            delta = m["lufs"] - median
            if abs(delta) > SCENE_SPREAD_LU:
                word = "louder" if delta > 0 else "quieter"
                report.add("warn", "loudness",
                           f"{f.name}: {m['lufs']:.1f} LUFS, {abs(delta):.1f} LU {word} than the other scenes "
                           f"({median:.1f})", file=f.name)
    return levels


def probe_video(path: Path) -> dict | None:
    """width, height, duration and whether there is an audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        data = json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        return None
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "duration": _num(data.get("format", {}).get("duration")),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def check_render(render: Path, report: Report, target: float, info: dict | None, *, allow_silent: bool = False) -> dict | None:
    if not render.exists():
        report.add("error", "render", f"{render} not found")
        return None
    if info and not info["has_audio"]:
        report.add("info" if allow_silent else "error", "render", f"{render.name}: no audio track" + (' (intentional silent output)' if allow_silent else ''))
        return None
    m = measure_loudness(render)
    if not m:
        report.add("error", "render", f"{render.name}: could not measure loudness")
        return None
    if m["lufs"] is None:
        # loudnorm reports -inf for digital silence
        report.add("info" if allow_silent else "error", "render", f"{render.name}: the audio track is silent")
        return None
    report.data["render"] = {"file": str(render), "lufs": m["lufs"], "true_peak": m["true_peak"], "lra": m["lra"]}
    if m["lufs"] < target - 2:
        # Platforms turn loud content down but never turn quiet content up.
        report.add("warn", "render", f"{render.name}: {m['lufs']:.1f} LUFS — quieter than the {target:.0f} target, "
                   "and YouTube won't raise it")
    elif m["lufs"] > target + 1:
        report.add("warn", "render", f"{render.name}: {m['lufs']:.1f} LUFS — YouTube will turn it down "
                   f"{m['lufs'] - target:.1f} dB")
    if m["true_peak"] is not None and m["true_peak"] > YOUTUBE_TRUE_PEAK:
        report.add("warn", "render", f"{render.name}: true peak {m['true_peak']:+.1f} dBTP — clipping after "
                   "lossy encoding")
    return m


def _fmt_t(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:04.1f}"


def _edge_fraction(frame) -> float:
    """Share of pixels on a sharp edge: text, UI and images have them, gradients and blurs don't."""
    import numpy as np
    gx = np.abs(np.diff(frame, axis=1))[:-1, :]
    gy = np.abs(np.diff(frame, axis=0))[:, :-1]
    return float((np.maximum(gx, gy) > 24).mean())


def check_frames(render: Path, info: dict, report: Report, max_still: float):
    """Still stretches, black gaps and an empty first frame, from a small greyscale copy."""
    import numpy as np

    w = FRAME_WIDTH
    h = max(STILL_TILE, round(w * info["height"] / info["width"] / 2) * 2)
    rows, cols = h // STILL_TILE, w // STILL_TILE
    frame_bytes = w * h
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(render), "-map", "0:v:0",
             "-vf", f"fps={STILL_FPS},scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE,
        )
    except OSError as e:
        report.add("error", "frames", f"{render.name}: could not decode frames: {e}")
        return

    busiest: list[float] = []   # per sample pair: mean change in the busiest tile
    dark: list[bool] = []
    prev = None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(h, w).astype(np.int16)
        if prev is not None:
            diff = np.abs(frame - prev)[: rows * STILL_TILE, : cols * STILL_TILE]
            busiest.append(float(diff.reshape(rows, STILL_TILE, cols, STILL_TILE).mean(axis=(1, 3)).max()))
        # blackdetect's defaults: 98% of pixels under ~10% luma
        dark.append(float((frame >= 26).mean()) < 0.02)
        prev = frame
    proc.wait()
    if prev is None:
        report.add("error", "frames", f"{render.name}: no video frames decoded")
        return

    duration = info["duration"] or len(dark) / STILL_FPS
    report.data["frames"] = {"sampled": len(dark), "fps": STILL_FPS}

    # Runs of samples where nothing changed. Sample i pairs frames i and i+1.
    stills = []
    start = None
    for i, change in enumerate(busiest + [float("inf")]):
        if change < STILL_TILE_DIFF:
            start = i if start is None else start
        elif start is not None:
            t0, t1 = start / STILL_FPS, min(i / STILL_FPS, duration - STILL_END_GRACE)
            if t1 - t0 > max_still:
                stills.append((t0, t1))
            start = None
    for t0, t1 in stills:
        report.add("warn", "frames",
                   f"{_fmt_t(t0)}-{_fmt_t(t1)}: {t1 - t0:.1f}s with nothing on screen changing. Give the hold "
                   "a next action (the cursor moving on, the camera's next move) or cut it shorter",
                   start=round(t0, 2), end=round(t1, 2), kind="still")
    report.data["frames"]["still_stretches"] = len(stills)

    # Black gaps away from the opening fade-in and the closing fade-out.
    start = None
    for i, is_dark in enumerate(dark + [False]):
        if is_dark:
            start = i if start is None else start
        elif start is not None:
            t0, t1 = start / STILL_FPS, i / STILL_FPS
            if t1 - t0 >= BLACK_MIN_SECONDS and start > 0 and i < len(dark):
                report.add("warn", "frames", f"{_fmt_t(t0)}-{_fmt_t(t1)}: black for {t1 - t0:.1f}s "
                           "(a gap between sequences, or media that failed to load?)",
                           start=round(t0, 2), end=round(t1, 2), kind="black")
            start = None

    # Frame 0 is the cover on Shorts and Reels, where a vertical or square upload can't
    # take a custom thumbnail; a landscape upload usually gets one from /thumbnail.
    # Decoded on its own: the fps filter's first sample can land a few frames in.
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(render), "-map", "0:v:0", "-frames:v", "1",
         "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=120,
    )
    if len(r.stdout) < frame_bytes:
        return
    edges = _edge_fraction(np.frombuffer(r.stdout[:frame_bytes], np.uint8).reshape(h, w).astype(np.int16))
    report.data["frames"]["first_frame_edges"] = round(edges, 4)
    if edges < EMPTY_EDGE_FRACTION:
        vertical = info["height"] >= info["width"]
        report.add("warn" if vertical else "info", "frames",
                   "first frame is background only" + (
                       ", and it is the cover on Shorts and Reels. Start with the title already on screen"
                       if vertical else " (fine with a custom thumbnail; it is what autoplay shows first)"),
                   kind="first_frame_empty")


def composition_id(project: Path) -> str | None:
    """The composition `npm run render` renders, from package.json."""
    try:
        scripts = json.loads((project / "package.json").read_text(encoding="utf-8")).get("scripts", {})
    except (OSError, json.JSONDecodeError):
        return None
    m = re.search(r"remotion render\s+(?:\S+\.[jt]sx?\s+)?([A-Za-z][\w-]*)", scripts.get("render", ""))
    return m.group(1) if m else None


def run_layout_audit(project: Path, composition: str | None, every: float, report: Report) -> dict | None:
    """Bundle the project and run tools/qc_layout.mjs on it. Returns its raw JSON."""
    node, npx = shutil.which("node"), shutil.which("npx")
    if not node or not npx:
        report.add("error", "layout", "--layout needs Node.js (node and npx on PATH)")
        return None
    if not (project / "package.json").exists():
        report.add("error", "layout", f"--layout needs a Remotion project; no package.json in {project}")
        return None
    toolkit = Path(__file__).resolve().parent.parent
    if not (toolkit / "playwright" / "node_modules" / "playwright").exists():
        report.add("error", "layout", "--layout needs Playwright: cd playwright && npm install")
        return None
    bundle = Path(tempfile.mkdtemp(prefix="qc-layout-"))
    try:
        r = subprocess.run([npx, "remotion", "bundle", "--out-dir", str(bundle)], cwd=project,
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            report.add("error", "layout", f"remotion bundle failed: {(r.stderr or r.stdout).strip()[-300:]}")
            return None
        cmd = [node, str(toolkit / "tools" / "qc_layout.mjs"), "--bundle", str(bundle), "--every", str(every)]
        if composition:
            cmd += ["--composition", composition]
        elif composition_id(project):
            cmd += ["--prefer", composition_id(project)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            report.add("error", "layout", f"layout audit failed: {r.stderr.strip()[-300:]}")
            return None
        return json.loads(r.stdout)
    except subprocess.TimeoutExpired as e:
        report.add("error", "layout", f"layout audit timed out ({e.cmd[1]})")
        return None
    finally:
        shutil.rmtree(bundle, ignore_errors=True)


def _contrast(a: list[int], b: list[int]) -> float:
    def lum(c):
        c = [x / 255 for x in c]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _suggest(fg: list[int], bg: list[int], need: float) -> str | None:
    """The nearest colour in the same direction (darker on light, lighter on dark) that passes."""
    toward = 0 if sum(bg) > sum(fg) else 255
    for i in range(1, 101):
        c = [round(x + (toward - x) * i / 100) for x in fg]
        if _contrast(c, bg) >= need + 0.05:
            return "#{:02x}{:02x}{:02x}".format(*c)
    return None


def _runs(frames: list[int], samples: list[int]) -> list[tuple[int, int]]:
    """Runs of consecutive samples, as (first frame, last frame), at least LAYOUT_PERSIST long."""
    position = {f: i for i, f in enumerate(samples)}
    idx = sorted(position[f] for f in set(frames))
    runs, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        if j - i + 1 >= LAYOUT_PERSIST:
            runs.append((samples[idx[i]], samples[idx[j]]))
        i = j + 1
    return runs


def check_layout(data: dict, report: Report):
    comp = data.get("composition")
    if not comp:
        return
    fps, samples = comp["fps"], data["samples"]
    floor = round(TEXT_FLOOR * min(comp["width"], comp["height"]))
    report.data["layout"] = {"composition": comp["id"], "samples": len(samples), "texts": len(data["texts"])}
    for note in data.get("notes", []):
        report.add("info", "layout", note)

    for req in {(r["url"], r["status"]) for r in data["failedRequests"] if r["url"] not in IGNORED_REQUESTS}:
        report.add("warn", "layout", f"the page requested {req[0]} and got {req[1]}")
    for err in dict.fromkeys(data["pageErrors"]):
        report.add("warn", "layout", f"page error: {err}")

    by_text: dict[tuple[str, bool], list[dict]] = {}
    for t in data["texts"]:
        by_text.setdefault((t["text"], t["product"]), []).append(t)

    for product in (False, True):
        level = "info" if product else "warn"
        where = "product UI: " if product else ""
        texts = {k[0]: v for k, v in by_text.items() if k[1] == product}

        # Overflow and clipping that persists (a moving element passes through the edge). Not for
        # product UI: Stage's camera crops it on purpose.
        for text, obs in ({} if product else texts).items():
            for key, what in (("offCanvas", "off the canvas"), ("clipped", "clipped by its box")):
                hits = [o for o in obs if o[key] > LAYOUT_EDGE_PX]
                for f0, f1 in _runs([o["frame"] for o in hits], samples):
                    px = max(o[key] for o in hits if f0 <= o["frame"] <= f1)
                    report.add(level, "layout", f"{where}{_fmt_t(f0 / fps)}-{_fmt_t(f1 / fps)} \"{text[:50]}\" "
                               f"{px}px {what}", kind=key, product=product, text=text)

        # Contrast: a text fails when its typical sample is under WCAG AA for the size it is shown at.
        # Grouped by colour pair, because one style usually fails everywhere it's used.
        pairs: dict[tuple, list] = {}
        for text, obs in texts.items():
            measured = [o for o in obs if o.get("ratio")]
            if not measured:
                continue
            for o in measured:
                large = o["fontPx"] >= 24 or (o["bold"] and o["fontPx"] >= 18.66)
                o["need"] = 3.0 if large else 4.5
            typical = sorted(measured, key=lambda o: o["ratio"] / o["need"])[len(measured) // 2]
            if typical["ratio"] >= typical["need"]:
                continue
            key = (tuple(round(x / 8) for x in typical["fg"]), tuple(round(x / 8) for x in typical["bg"]))
            pairs.setdefault(key, []).append((text, typical, min(o["frame"] for o in measured)))
        for items in sorted(pairs.values(), key=lambda v: min(x[1]["ratio"] for x in v)):
            text, o, first = min(items, key=lambda x: x[1]["ratio"])
            fg, bg = "#{:02x}{:02x}{:02x}".format(*o["fg"]), "#{:02x}{:02x}{:02x}".format(*o["bg"])
            fix = _suggest(o["fg"], o["bg"], o["need"])
            names = ", ".join(f'"{x[0][:30]}"' for x in items[:3]) + (f" +{len(items) - 3}" if len(items) > 3 else "")
            report.add(level, "layout", f"{where}contrast {o['ratio']:.2f}:1 (needs {o['need']:.1f}) {fg} on {bg}: "
                       f"{names}, from {_fmt_t(first / fps)}" + (f". {fix} would pass" if fix else ""),
                       kind="contrast", product=product, ratio=o["ratio"], need=o["need"], fg=fg, bg=bg,
                       texts=[x[0] for x in items])

        # Size: text that never reaches the floor, at any zoom, in any sample
        small = sorted(((max(o["fontPx"] for o in obs), text) for text, obs in texts.items()
                        if max(o["fontPx"] for o in obs) < floor - 0.5))
        if small:
            names = ", ".join(f'"{t[:30]}" {px:.0f}px' for px, t in small[:5])
            more = f" +{len(small) - 5}" if len(small) > 5 else ""
            report.add(level, "layout", f"{where}{len(small)} of {len(texts)} texts never reach {floor}px "
                       f"({TEXT_FLOOR:.1%} of the short side, the phone floor): {names}{more}",
                       kind="size", product=product, count=len(small))
        if product and texts:
            report.data["layout"]["product_texts"] = len(texts)


# ─── Fix ─────────────────────────────────────────────────────

def fix_scenes(levels: dict[Path, dict], target: float, report: Report) -> list[str]:
    fixed = []
    for f, m in levels.items():
        off_target = abs(m["lufs"] - target) > FIX_TOLERANCE_LU
        clipping = m["true_peak"] is not None and m["true_peak"] > SCENE_TRUE_PEAK
        if not (off_target or clipping):
            continue
        backup_dir = f.parent / ".qc-backup"
        backup_dir.mkdir(exist_ok=True)
        backup = backup_dir / f.name
        if not backup.exists():  # never overwrite the true original on a re-run
            shutil.copy2(f, backup)
        tmp = f.with_name(f".qc-tmp-{f.name}")
        err = normalize_loudness(f, tmp, target, YOUTUBE_TRUE_PEAK - 0.5, m)
        if err:
            tmp.unlink(missing_ok=True)
            report.add("error", "fix", f"{f.name}: normalisation failed: {err}", file=f.name)
            continue
        shutil.move(str(tmp), str(f))
        fixed.append(f.name)
        # The file's loudness warnings no longer describe it
        report.findings = [x for x in report.findings
                           if not (x["check"] == "loudness" and x.get("file") == f.name)]
        report.add("info", "fix", f"{f.name}: {m['lufs']:.1f} → {target:.0f} LUFS "
                   f"(original in {backup_dir.name}/)", file=f.name)
    return fixed


def fix_render(render: Path, m: dict, target: float, report: Report) -> Path | None:
    out = render.with_name(f"{render.stem}.loudnorm{render.suffix}")
    # Margin under the ceiling: the AAC re-encode adds a little true peak back
    err = normalize_loudness(render, out, target, YOUTUBE_TRUE_PEAK - 0.5, m, video=True)
    if err:
        report.add("error", "fix", f"{render.name}: normalisation failed: {err}")
        return None
    report.add("info", "fix", f"wrote {out.name} ({m['lufs']:.1f} → {target:.0f} LUFS); "
               f"{render.name} is unchanged — re-run with --render {out.name} to confirm")
    return out


# ─── Output ──────────────────────────────────────────────────

ICONS = {"error": "✗", "warn": "!", "info": "·"}


def print_report(report: Report, project: Path):
    d = report.data
    print(f"QC — {project} ({d.get('template_type', 'unknown')})", file=sys.stderr)
    print(f"  assets checked: {d.get('assets_checked', 0)} | scenes timed: {d.get('scenes_timed', 0)} | "
          f"pacing measured: {d.get('pacing_measured', 0)} | "
          f"loudness measured: {len(d.get('scene_loudness', {}))}", file=sys.stderr)
    if "scene_loudness_median" in d:
        print(f"  scene loudness median: {d['scene_loudness_median']:.1f} LUFS", file=sys.stderr)
    if "render" in d:
        r = d["render"]
        tp = f"{r['true_peak']:+.1f} dBTP" if r["true_peak"] is not None else "n/a"
        print(f"  render: {r['lufs']:.1f} LUFS, {tp}", file=sys.stderr)
    print(file=sys.stderr)
    for check in ("assets", "timing", "pacing", "loudness", "render", "frames", "layout", "fix"):
        items = [f for f in report.findings if f["check"] == check]
        if not items:
            continue
        print(f"{check}:", file=sys.stderr)
        for f in items:
            print(f"  {ICONS[f['level']]} {f['message']}", file=sys.stderr)
    errors, warns = report.count("error"), report.count("warn")
    verdict = "FAIL" if errors else ("PASS with warnings" if warns else "PASS")
    print(f"\n{verdict}: {errors} error(s), {warns} warning(s)", file=sys.stderr)


# ─── Main ────────────────────────────────────────────────────

def main():
    ensure_utf8_output()  # Windows: piped stdout defaults to cp1252
    parser = argparse.ArgumentParser(
        description="Pre-render QC: missing assets, scene timing, pacing, loudness; with --render, "
                    "the render's loudness and frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/qc.py --project projects/my-video                  # check before rendering
  uv run tools/qc.py --project projects/my-video --render projects/my-video/out/video.mp4
  uv run tools/qc.py --project projects/my-video --fix            # normalise scene VO loudness
  uv run tools/qc.py --render out/video.mp4 --fix                 # write video.loudnorm.mp4
  uv run tools/qc.py --project projects/my-video --layout         # text size, overflow, contrast
  uv run tools/qc.py --project projects/my-video --json           # machine-readable
""",
    )
    parser.add_argument("--project", default=".", help="Project directory (default: current directory)")
    parser.add_argument("--config", help="Remotion config file (auto-detected if omitted)")
    parser.add_argument("--render", help="Finished video to check against the loudness target")
    parser.add_argument("--target-lufs", type=float, default=YOUTUBE_LUFS,
                        help=f"Integrated loudness target for the render (default: {YOUTUBE_LUFS:.0f}, YouTube)")
    parser.add_argument("--scene-lufs", type=float, default=SCENE_LUFS,
                        help=f"Target for per-scene VO when fixing (default: {SCENE_LUFS:.0f}, leaves "
                             "headroom for music)")
    parser.add_argument("--fix", action="store_true",
                        help="Normalise loudness: scene files in place (originals in .qc-backup/), "
                             "render to <name>.loudnorm.mp4")
    parser.add_argument("--skip-loudness", action="store_true", help="Skip the (slower) loudness pass")
    parser.add_argument("--skip-frames", action="store_true", help="Skip the render's frame checks")
    parser.add_argument('--allow-silent', action='store_true', help='Explicitly accept intentional silent video; picture checks still run')
    parser.add_argument("--max-still", type=float, default=MAX_STILL_SECONDS,
                        help=f"Seconds with nothing changing before a render warns "
                             f"(default: {MAX_STILL_SECONDS:.0f})")
    parser.add_argument("--layout", action="store_true",
                        help="Bundle the project and measure its text in the browser: off canvas, clipped, "
                             "too small for a phone, contrast below WCAG AA (needs Node and playwright/)")
    parser.add_argument("--composition", help="Composition for --layout (default: the one `npm run render` uses)")
    parser.add_argument("--layout-every", type=float, default=LAYOUT_EVERY,
                        help=f"Seconds between --layout samples (default: {LAYOUT_EVERY:.0f})")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings too")
    parser.add_argument("--json", action="store_true", help="Output JSON for scripting")
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("Error: ffmpeg and ffprobe are required (see the ffmpeg skill for install).", file=sys.stderr)
        sys.exit(1)

    project = Path(args.project).resolve()
    report = Report()

    narration: dict[Path, str] = {}
    if (project / "scenes.json").exists() and not args.config:
        audio_dir, timed, narration = load_scenes_json(project, report)
    else:
        loaded = load_remotion(project, report, args.config)
        if loaded is None and not (args.render or args.layout):
            print(f"Error: no Remotion config or scenes.json found in {project}. "
                  "Use --project, --config, --render or --layout.", file=sys.stderr)
            sys.exit(1)
        audio_dir, timed = loaded or (None, [])

    files = scene_audio_files(audio_dir, timed) if audio_dir else []
    check_timing(timed, report)
    check_pacing(files, narration, report)

    levels = {} if args.skip_loudness else check_scene_loudness(files, report)
    render_m = None
    if args.render:
        render = Path(args.render)
        info = probe_video(render) if render.exists() else None
        render_m = check_render(render, report, args.target_lufs, info, allow_silent=args.allow_silent)
        if info and not args.skip_frames:
            check_frames(render, info, report, args.max_still)

    if args.layout:
        layout = run_layout_audit(project, args.composition, args.layout_every, report)
        if layout:
            check_layout(layout, report)

    if args.fix:
        report.data["fixed_scenes"] = fix_scenes(levels, args.scene_lufs, report)
        if render_m:
            out = fix_render(Path(args.render), render_m, args.target_lufs, report)
            report.data["fixed_render"] = str(out) if out else None

    failed = report.count("error") > 0 or (args.strict and report.count("warn") > 0)
    if args.json:
        print(json.dumps({**report.data, "findings": report.findings,
                          "errors": report.count("error"), "warnings": report.count("warn"),
                          "passed": not failed}, indent=2))
    else:
        print_report(report, project)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
