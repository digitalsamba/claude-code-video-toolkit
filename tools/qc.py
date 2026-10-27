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
    """concept-explainer-short / youtube-longform: scenes.json + config.json, audio-anchored
    (no timing check).

    Returns (audio_dir, timed, narration) where narration maps each scene's mp3 to its text.
    """
    data = json.loads((project / "scenes.json").read_text(encoding="utf-8"))
    report.data["config"] = str(project / "scenes.json")
    chaptered = any(s.get("chapter") for s in data.get("scenes", []))
    report.data["template_type"] = "youtube-longform" if chaptered else "concept-explainer-short"
    refs = []
    for s in data.get("scenes", []):
        asset = s.get("asset")  # youtube-longform allows a list of assets per scene
        refs += asset if isinstance(asset, list) else [asset] if asset else []
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


def check_render(render: Path, report: Report, target: float) -> dict | None:
    if not render.exists():
        report.add("error", "render", f"{render} not found")
        return None
    m = measure_loudness(render)
    if not m or m["lufs"] is None:
        report.add("error", "render", f"{render.name}: could not measure loudness (no audio stream?)")
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
    for check in ("assets", "timing", "pacing", "loudness", "render", "fix"):
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
        description="Pre-render QC: missing assets, scene timing, pacing, loudness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run tools/qc.py --project projects/my-video                  # check before rendering
  uv run tools/qc.py --project projects/my-video --render projects/my-video/out/video.mp4
  uv run tools/qc.py --project projects/my-video --fix            # normalise scene VO loudness
  uv run tools/qc.py --render out/video.mp4 --fix                 # write video.loudnorm.mp4
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
        if loaded is None and not args.render:
            print(f"Error: no Remotion config or scenes.json found in {project}. "
                  "Use --project, --config, or --render.", file=sys.stderr)
            sys.exit(1)
        audio_dir, timed = loaded or (None, [])

    files = scene_audio_files(audio_dir, timed) if audio_dir else []
    check_timing(timed, report)
    check_pacing(files, narration, report)

    levels = {} if args.skip_loudness else check_scene_loudness(files, report)
    render_m = check_render(Path(args.render), report, args.target_lufs) if args.render else None

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
