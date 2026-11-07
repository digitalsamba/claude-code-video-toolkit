#!/usr/bin/env python3
"""Recommend one of several music takes (music_gen.py --variations N), with the reasons, for a person to confirm.

    uv run tools/music_gen.py --preset ambient --duration 46 --variations 3 --output public/audio/music.mp3
    uv run tools/pick_music.py public/audio/music_*.mp3 --target 44.5 --cuts 11.5,17.5,37.5 --under text \\
        --fade-out 1.5 --pick public/audio/music.mp3

A tool can't hear whether a take sounds right, so this measures what can be measured and leaves the taste to the
listener. It reports, per take:

- defects that rule a take out: shorter than the video (--target), clipping, a dropout (0.6s or more, 30 dB under
  the track) away from the ends;
- how busy it is: note onsets a second. Under narration or on-screen text (--under voice|text) a calm bed is
  better, so a busy one loses points;
- its arc: the spread of its loudness (a flat bed under 2 dB loses points) and whether it builds, holds or fades;
- its ending: a fade, or a cut-off the video must cover (--fade-out says how long the video's own music fade is);
- fit to the video's turns (--cuts, in seconds): whether a note or a change in level lands within 0.35s of each.

Then it recommends the best eligible take and says why, against the runner-up. Interactive commands should show
this to the user and ask them to listen before using it; an agent with nobody to ask (video_job.py) uses the
recommendation and records the reasons. --pick copies the winner to a path; --json prints the report on stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 22050
HOP = 512
WIN = 2048
FPS = SR / HOP  # analysis frames a second


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def lufs(path: Path) -> float | None:
    err = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    return float(m[-1]) if m else None


def frames(x: np.ndarray) -> np.ndarray:
    n = 1 + max(0, (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    pad = np.pad(x, (0, max(0, idx.max() + 1 - len(x))))
    return pad[idx]


def smooth(v: np.ndarray, seconds: float) -> np.ndarray:
    k = max(1, int(seconds * FPS))
    return np.convolve(v, np.ones(k) / k, mode="same")


def analyse(path: Path, target: float | None, cuts: list[float], fade_out: float) -> dict:
    x = decode(path)
    dur = len(x) / SR
    fr = frames(x)
    rms = np.sqrt(np.mean(fr**2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    t = np.arange(len(db)) / FPS

    # onset strength: positive spectral flux of log magnitude
    mag = np.abs(np.fft.rfft(fr * np.hanning(WIN), axis=1))
    # log1p, not log: log of near-empty bins turns quantisation noise into flux, and a held pad reads as notes
    logmag = np.log1p(100 * mag)
    flux = np.concatenate([[0.0], np.maximum(0, np.diff(logmag, axis=0)).sum(axis=1)])
    # local threshold, with a floor at a fifth of the track's strong notes so a held sound has no onsets
    thr = np.maximum(smooth(flux, 1.5) + flux.std(), 0.2 * np.percentile(flux, 99.5))
    peaks = [i for i in range(1, len(flux) - 1) if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]]
    onsets: list[int] = []
    for i in peaks:  # at most one onset per 0.1s
        if not onsets or i - onsets[-1] >= 0.1 * FPS:
            onsets.append(i)

    body = (t >= 1.0) & (t <= dur - 2.0)
    if body.sum() < FPS:  # very short take: judge the whole of it
        body = np.ones_like(t, dtype=bool)
    level = 10 * np.log10(smooth(rms**2, 2.0) + 1e-18)
    median_db = float(np.median(db[body]))
    spread = float(np.percentile(level[body], 90) - np.percentile(level[body], 10))
    thirds = np.array_split(level[body], 3)
    drift = float(thirds[2].mean() - thirds[0].mean())
    arc = "builds" if drift > 3 else "fades" if drift < -3 else "holds"

    defects: list[str] = []
    if target and dur < target - 0.25:
        defects.append(f"{target - dur:.1f}s shorter than the video")
    clip = float(np.mean(np.abs(x) >= 0.999))
    if clip > 1e-4:
        defects.append(f"clips ({clip * 100:.2f}% of samples at full scale)")
    # a dropout is a gap the music comes back from; going quiet and staying quiet is its ending
    quiet = (db < median_db - 30) & body
    back = db > median_db - 12
    run = 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run == int(0.6 * FPS) and back[i:].any():
            defects.append(f"drops out at {t[i - run + 1]:.1f}s")
    # where the video stops using it
    end = min(dur, target) if target else dur
    if target and dur > target + 0.5:
        tail = db[(t >= end - 0.5) & (t < end)]
        held = tail.size and tail.mean() > median_db - 12
        ending = "playing at the end" if held else "quiet at the end"
        ending_note = f"still playing at {end:.1f}s, where the video stops it" if held else f"quiet by {end:.1f}s, the video's end"
        cuts_off = bool(held)
    else:
        tail = db[t >= dur - 0.5]
        cuts_off = bool(tail.size and tail.mean() > median_db - 12)
        ending = "cuts off" if cuts_off else "fades"
        ending_note = "ends on a fade" if not cuts_off else "stops while still playing"

    hits = []
    onset_t = np.array(onsets) / FPS
    for c in cuts:
        # a note lands on the cut: a detected onset within 0.35s of it (not raw flux, which in a held pad is noise)
        near = onset_t[np.abs(onset_t - c) <= 0.35] if onset_t.size else onset_t
        before = level[(t >= c - 1.0) & (t < c)]
        after = level[(t >= c) & (t < c + 1.0)]
        change = abs(float(after.mean() - before.mean())) if before.size and after.size else 0.0
        hits.append({"at": c, "note": bool(near.size),
                     "note_offset_s": round(float(near[np.argmin(np.abs(near - c))] - c), 2) if near.size else None,
                     "level_change_db": round(change, 1), "lands": bool(near.size) or change >= 2.0})

    return {
        "file": str(path),
        "duration": round(dur, 2),
        "lufs": lufs(path),
        "onsets_per_s": round(len(onsets) / dur, 2) if dur else 0.0,
        "spread_db": round(spread, 1),
        "arc": arc,
        "ending": ending,
        "ending_note": ending_note,
        "cuts_off": cuts_off,
        "fade_covers_it": cuts_off and fade_out >= 1.0,
        "cuts": hits,
        "defects": defects,
    }


CALM = {"voice": 2.5, "text": 3.0}


def score(a: dict, under: str) -> tuple[float, list[str], list[str]]:
    good, bad = [], []
    s = 0.0
    if a["cuts"]:
        landed = sum(h["lands"] for h in a["cuts"])
        s += 2.0 * landed / len(a["cuts"])
        (good if landed else bad).append(f"something lands on {landed} of {len(a['cuts'])} cuts")
    if under in CALM:
        over = a["onsets_per_s"] - CALM[under]
        if over > 0:
            s -= 0.5 * over
            bad.append(f"busy under {under} ({a['onsets_per_s']} notes/s)")
        else:
            good.append(f"calm enough to sit under {under} ({a['onsets_per_s']} notes/s)")
    if a["spread_db"] < 2:
        s -= 0.5
        bad.append(f"flat ({a['spread_db']} dB of movement)")
    elif a["spread_db"] >= 4:
        good.append(f"moves ({a['spread_db']} dB, {a['arc']})")
    if a["cuts_off"] and not a["fade_covers_it"]:
        s -= 1.0
        bad.append(a["ending_note"] + " and the video doesn't fade it (--fade-out)")
    elif not a["cuts_off"]:
        good.append(a["ending_note"])
    return round(s, 2), good, bad


def rank(files: list[Path], target: float | None, cuts: list[float], under: str, fade_out: float) -> list[dict]:
    """Every take measured and scored, best first: eligible ones by score, a tie to the calmer take under voice
    or text (else the one that moves more); ineligible ones after, by fewest defects."""
    rows = []
    for f in files:
        m = analyse(Path(f), target, cuts, fade_out)
        m["score"], m["good"], m["bad"] = score(m, under)
        m["eligible"] = not m["defects"]
        rows.append(m)
    calm = under in CALM
    rows.sort(key=lambda r: (not r["eligible"], len(r["defects"]), -r["score"],
                             r["onsets_per_s"] if calm else -r["spread_db"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+", help="the take files (e.g. music_1.mp3 music_2.mp3)")
    ap.add_argument("--target", type=float, help="the video's length in seconds (a shorter take is ruled out)")
    ap.add_argument("--cuts", default="", help="comma-separated seconds where the video turns (act cuts)")
    ap.add_argument("--under", choices=["voice", "text", "none"], default="none",
                    help="what the music sits under; voice or text prefer a calm bed")
    ap.add_argument("--fade-out", type=float, default=0.0, help="seconds of music fade the video applies at its end")
    ap.add_argument("--pick", help="copy the recommended take to this path")
    ap.add_argument("--json", action="store_true", help="print the report as JSON on stdout")
    a = ap.parse_args()

    files = [Path(p) for p in a.takes]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        sys.exit(f"not found: {', '.join(missing)}")
    cuts = [float(c) for c in a.cuts.split(",") if c.strip()]

    rows = rank(files, a.target, cuts, a.under, a.fade_out)
    calm = a.under in CALM
    best = rows[0]
    runner = rows[1] if len(rows) > 1 else None

    why = best["good"][:3]
    if runner:
        name = Path(runner["file"]).name
        if not runner["eligible"]:
            why.append(f"{name} {', '.join(runner['defects'])}")
        elif runner["score"] == best["score"]:
            why.append(f"ties with {name} and is calmer ({best['onsets_per_s']} vs {runner['onsets_per_s']} notes/s)"
                       if calm else f"ties with {name} and moves more ({best['spread_db']} vs {runner['spread_db']} dB)")
        else:
            extra = [b for b in runner["bad"] if b not in best["bad"]]
            if extra:
                why.append(f"{name}: {extra[0]}")
    report = {
        "recommended": best["file"],
        "eligible": best["eligible"],
        "why": why,
        "takes": rows,
        "note": "Measured, not heard: listen before you use it.",
    }

    if a.pick:
        Path(a.pick).parent.mkdir(parents=True, exist_ok=True)
        if Path(a.pick).resolve() != Path(best["file"]).resolve():
            shutil.copyfile(best["file"], a.pick)
        report["picked"] = a.pick

    if a.json:
        print(json.dumps(report, indent=2))
    else:
        log(f"{'take':<28} {'dur':>6} {'LUFS':>6} {'notes/s':>8} {'spread':>7} {'arc':>7}  {'ending':<18} {'cuts':>5} {'score':>6}")
        for r in rows:
            landed = f"{sum(h['lands'] for h in r['cuts'])}/{len(r['cuts'])}" if r["cuts"] else "-"
            lu = f"{r['lufs']:.1f}" if r["lufs"] is not None else "?"
            flag = "" if r["eligible"] else "  OUT: " + "; ".join(r["defects"])
            log(f"{Path(r['file']).name:<28} {r['duration']:>6.1f} {lu:>6} {r['onsets_per_s']:>8.2f} {r['spread_db']:>6.1f}dB"
                f" {r['arc']:>7}  {r['ending']:<18} {landed:>5} {r['score']:>6.2f}{flag}")
        verdict = "Recommended" if best["eligible"] else "Every take has a defect; least bad"
        log(f"\n{verdict}: {Path(best['file']).name}" + (f" ({'; '.join(why)})" if why else ""))
        if a.pick:
            log(f"Copied to {a.pick}")
        log(report["note"])
    return 0 if best["eligible"] else 1


if __name__ == "__main__":
    sys.exit(main())
