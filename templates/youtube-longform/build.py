#!/usr/bin/env python3
"""Step 3 — compose the long-form video. Audio-anchored, chapter-by-chapter.

Long videos (10–60 min) are too big to composite in one moviepy pass, so the
build is split:

  1. Timeline   scene starts derive from real VO durations (vo_durations.json),
                scenes group into chapters by their "chapter" field.
  2. Chapters   each chapter's *video* renders to out/chapters/NN.mp4 and is
                cached by a content hash — editing one chapter re-renders only
                that chapter.
  3. Audio      one continuous track: every scene VO at its anchored start,
                plus the music bed looped across the whole video and ducked
                under the voice (sidechain), so music never restarts per chapter.
  4. Master     ffmpeg concatenates the chapter videos (stream copy) and muxes
                the audio, two-pass loudnorm'd to YouTube's -14 LUFS / -1 dBTP.

Per scene, the visual is picked by asset extension (asset may be a list — the
scene's time is split across them, which keeps long scenes moving):

  .png/.jpg/.jpeg/.webp  → slow Ken Burns zoom (alternating in/out)
  .mp4/.mov/.webm        → boomerang loop (forward + reversed), cut to fit
  missing                → gradient placeholder card

Also writes, beside out/video.mp4:
  youtube-description.txt   description + chapter timestamps, paste-ready
  captions.srt              upload as YouTube subtitles (needs gen_captions.py)
  timeline.json             every scene/chapter start, for review

Run from this project directory:

    uv run build.py                 # full render
    uv run build.py --preview       # half-res, fast encode → out/preview.mp4
    uv run build.py --chapter 3     # render just chapter 3 → out/chapters/03.mp4
    uv run build.py --force         # ignore the chapter cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from moviepy import (
        VideoClip,
        VideoFileClip,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("    uv sync   (from toolkit root)")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
TEXT_CACHE = HERE / ".text_cache"
BUILD_VERSION = "2"  # bump to invalidate every cached chapter

CONFIG = json.loads((HERE / "config.json").read_text())
FMT = CONFIG["format"]
PALETTE = CONFIG["palette"]
CAPTIONS = CONFIG.get("captions", {})
TIMING = CONFIG.get("timing", {})
CHAPTER_CARD = CONFIG.get("chapterCard", {})
AUDIO = CONFIG.get("audio", {})

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}

_FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"],
    "Windows": ["C:/Windows/Fonts/arialbd.ttf"],
}

# Set by main() once --preview is known; layout values in config.json are in
# full-resolution pixels and get multiplied by S.
S = 1.0
W, H, FPS = FMT["width"], FMT["height"], FMT["fps"]


def _load_font(size: int):
    for path in _FONT_CANDIDATES.get(platform.system(), []):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(hex_color: str) -> tuple:
    return tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def fmt_ts(seconds: float) -> str:
    """YouTube chapter timestamp: M:SS, or H:MM:SS past an hour."""
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── text & placeholder rendering (PIL — moviepy 2.x TextClip clips ascenders) ──

def pill_png(txt: str, size: int, fill: str, alpha: int = 185) -> str:
    """Text on a dark rounded pill, cached, width-capped to the safe area."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{txt}|{size}|{fill}|{alpha}|{W}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    font = _load_font(size)
    sw = max(2, size // 16)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), txt, font=font, stroke_width=sw)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = max(18, size // 2), max(12, size // 4)

    img = Image.new("RGBA", (tw + pad_x * 2, th + pad_y * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, img.width - 1, img.height - 1),
                           radius=img.height // 4, fill=(*_rgb(PALETTE["ink"]), alpha))
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), txt, font=font,
              fill=(*_rgb(fill), 255), stroke_width=sw,
              stroke_fill=(*_rgb(PALETTE["ink"]), 235))

    max_w = int(W * 0.9)
    if img.width > max_w:
        img = img.resize((max_w, max(1, int(img.height * max_w / img.width))),
                         Image.LANCZOS)
    img.save(path)
    return str(path)


def placeholder_png(scene: dict, label: str, idx: int) -> str:
    """Gradient card with the scene slug — renders before any asset exists."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    accent = PALETTE["accent1"] if idx % 2 == 0 else PALETTE["accent2"]
    key = hashlib.sha1(f"ph|{scene['slug']}|{label}|{accent}|{W}x{H}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    ink, acc = _rgb(PALETTE["ink"]), _rgb(accent)
    img = Image.new("RGB", (W, H), ink)
    draw = ImageDraw.Draw(img)
    for x in range(W):  # horizontal gradient, accent rising to the right
        t = max(0.0, x / W - 0.4) * 0.55
        draw.line([(x, 0), (x, H)],
                  fill=tuple(int(i + (a - i) * t) for i, a in zip(ink, acc)))
    font = _load_font(int(96 * S))
    title = scene["slug"].replace("_", " ").replace("-", " ").upper()
    bbox = draw.textbbox((0, 0), title, font=font)
    draw.text(((W - bbox[2] + bbox[0]) / 2, H * 0.40), title,
              font=font, fill=_rgb(PALETTE["white"]))
    small = _load_font(int(34 * S))
    hint = f"placeholder — add {label}"
    bbox = draw.textbbox((0, 0), hint, font=small)
    draw.text(((W - bbox[2] + bbox[0]) / 2, H * 0.53), hint,
              font=small, fill=_rgb(PALETTE["slate"]))
    img.save(path)
    return str(path)


# ── visuals ──

def ken_burns(img_path: str, duration: float, zoom_in: bool):
    """Slow zoom on a still, as a frame function t → HxWx3 uint8. The source
    is scaled once to cover the frame at peak zoom; each frame is a centred
    crop resized to W×H — far cheaper than resizing the whole image."""
    amp = 0.06
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        cover = max(W / im.width, H / im.height) * (1.0 + amp)
        big = im.resize((max(W, math.ceil(im.width * cover)),
                         max(H, math.ceil(im.height * cover))), Image.LANCZOS)
    bw, bh = big.size
    zmax = 1.0 + amp

    def frame(t: float) -> np.ndarray:
        p = min(1.0, max(0.0, t / duration))
        z = 1.0 + amp * (p if zoom_in else 1 - p)
        cw, ch = W * zmax / z, H * zmax / z  # crop size in big-image pixels
        x0, y0 = (bw - cw) / 2, (bh - ch) / 2
        return np.asarray(big.resize((W, H), Image.BILINEAR,
                                     box=(x0, y0, x0 + cw, y0 + ch)))

    return frame


def boomerang_fill(video_path: str):
    """Loop a clip forward/backward, as a frame function. ffmpeg pre-renders
    the cover-cropped forward+reverse loop once (cached), so playback is a
    plain sequential read — no per-frame resize, no backwards seeking."""
    src = Path(video_path)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    key = hashlib.sha1(f"boom|{src}|{st.st_size}|{st.st_mtime_ns}|{W}x{H}|{FPS}"
                       .encode()).hexdigest()[:16]
    loop = TEXT_CACHE / f"boom_{key}.mp4"
    if not loop.exists():
        vf = (f"[0:v]fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H},setsar=1,split[f][b];[b]reverse[r];[f][r]concat=n=2:v=1[v]")
        r = ffmpeg("-i", str(src), "-filter_complex", vf, "-map", "[v]", "-an",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-pix_fmt", "yuv420p", str(loop))
        if r.returncode:
            sys.exit(f"ffmpeg boomerang prep failed for {src}:\n{r.stderr[-800:]}")
    clip = VideoFileClip(str(loop))
    period = max(1.0 / FPS, clip.duration - 1.0 / FPS)

    def frame(t: float) -> np.ndarray:
        return clip.get_frame(t % period)

    return frame


def scene_assets(scene: dict) -> list[str]:
    a = scene.get("asset")
    if not a:
        return []
    return a if isinstance(a, list) else [a]


def estimate_duration(text: str) -> float:
    return max(2.0, len(text.split()) / 2.5)  # ~150 wpm fallback estimate


# ── timeline ──

def build_timeline(scenes: list[dict], durations: dict) -> tuple[list[dict], float]:
    """Group scenes into chapters and anchor every scene to real VO durations.

    Chapter lengths are rounded to whole frames so the concatenated chapters
    land exactly where the audio track expects them."""
    lead = TIMING.get("lead", 0.4)
    tail = TIMING.get("tail", 0.8)
    start_pad = TIMING.get("startPad", 0.3)
    end_pad = TIMING.get("endPad", 1.5)

    chapters: list[dict] = []
    for s in scenes:
        title = s.get("chapter") or (chapters[-1]["title"] if chapters else "Intro")
        if not chapters or title != chapters[-1]["title"]:
            chapters.append({"title": title, "scenes": []})
        chapters[-1]["scenes"].append(s)

    t = 0.0
    for ci, ch in enumerate(chapters):
        ch["index"] = ci + 1
        ch["start"] = t
        local = start_pad if ci == 0 else 0.0
        for s in ch["scenes"]:
            measured = s["id"] in durations
            vo = durations.get(s["id"]) or estimate_duration(s["text"])
            s["_local_start"] = local
            s["_vo_local"] = local + lead
            s["_vo"] = vo
            s["_measured"] = measured
            s["_dur"] = lead + vo + tail
            local += s["_dur"]
        if ci == len(chapters) - 1:
            local += end_pad
        ch["duration"] = math.ceil(local * FPS - 1e-6) / FPS
        t += ch["duration"]
    return chapters, t


def chunk_words(words: list[dict], max_words: int, max_chars: int) -> list[dict]:
    chunks, cur = [], []
    for w in words:
        cand = " ".join([c["w"] for c in cur] + [w["w"]])
        if cur and (len(cur) >= max_words or len(cand) > max_chars):
            chunks.append({"text": " ".join(c["w"] for c in cur),
                           "start": cur[0]["s"], "end": cur[-1]["e"]})
            cur = []
        cur.append(w)
        # break after sentence punctuation so subtitles follow the script's rhythm
        if w["w"][-1:] in ".?!":
            chunks.append({"text": " ".join(c["w"] for c in cur),
                           "start": cur[0]["s"], "end": cur[-1]["e"]})
            cur = []
    if cur:
        chunks.append({"text": " ".join(c["w"] for c in cur),
                       "start": cur[0]["s"], "end": cur[-1]["e"]})
    return chunks


def load_words(scene: dict) -> list[dict] | None:
    f = HERE / "captions" / f"words_{scene['id']}.json"
    return json.loads(f.read_text()) if f.exists() else None


def scene_mp3(scene: dict) -> Path:
    return HERE / "audio" / "scenes" / f"{scene['id']}_{scene['slug']}.mp3"


# ── chapter render ──

def chapter_key(ch: dict, preview: bool) -> str:
    """Content hash of everything that affects a chapter's pixels."""
    h = hashlib.sha256()
    h.update(BUILD_VERSION.encode())
    h.update(json.dumps({"fmt": FMT, "palette": PALETTE, "captions": CAPTIONS,
                         "timing": TIMING, "card": CHAPTER_CARD,
                         "preview": preview, "title": ch["title"],
                         "duration": ch["duration"]}, sort_keys=True).encode())
    for s in ch["scenes"]:
        h.update(json.dumps({k: v for k, v in s.items()}, sort_keys=True, default=str).encode())
        files = [HERE / a for a in scene_assets(s)]
        if CAPTIONS.get("burn"):
            files.append(HERE / "captions" / f"words_{s['id']}.json")
        for f in files:
            if f.exists():
                st = f.stat()
                h.update(f"{f.name}|{st.st_size}|{st.st_mtime_ns}".encode())
            else:
                h.update(f"{f.name}|missing".encode())
    return h.hexdigest()[:20]


class Overlay:
    """A pre-rendered RGBA PNG blended into a fixed region of the frame."""

    def __init__(self, png: str, start: float, dur: float, x, y,
                 fade_in: float = 0.0, fade_out: float = 0.0):
        with Image.open(png) as im:
            arr = np.asarray(im.convert("RGBA"), dtype=np.float32)
        h, w = arr.shape[:2]
        self.x = (W - w) // 2 if x == "center" else int(x)
        self.y = int(y)
        # clip to the frame
        x0, y0 = max(0, self.x), max(0, self.y)
        x1, y1 = min(W, self.x + w), min(H, self.y + h)
        arr = arr[y0 - self.y:y1 - self.y, x0 - self.x:x1 - self.x]
        self.box = (x0, y0, x1, y1)
        self.rgb, self.alpha = arr[..., :3], arr[..., 3:] / 255.0
        self.start, self.end = start, start + dur
        self.fade_in, self.fade_out = fade_in, fade_out

    def blend(self, frame: np.ndarray, t: float) -> None:
        f = fade_factor(t, self.start, self.end, self.fade_in, self.fade_out)
        if f <= 0:
            return
        x0, y0, x1, y1 = self.box
        a = self.alpha * f
        region = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (region * (1 - a) + self.rgb * a).astype(np.uint8)


def fade_factor(t: float, start: float, end: float, fi: float, fo: float) -> float:
    f = 1.0
    if fi > 0:
        f = min(f, (t - start) / fi)
    if fo > 0:
        f = min(f, (end - t) / fo)
    return max(0.0, min(1.0, f))


def render_chapter(ch: dict, out: Path, preview: bool, zoom_in: bool) -> bool:
    """Render one chapter's video with a small numpy compositor: one base
    visual at a time (fading to the ink colour at its edges) plus RGBA
    overlays blended only where they sit. moviepy's CompositeVideoClip costs
    ~4x more per frame, which adds up to hours on a long video."""
    xfade = TIMING.get("xfade", 0.45)
    ink = np.array(_rgb(PALETTE["ink"]), dtype=np.float32)
    bases: list[tuple] = []      # (start, end, frame_fn)
    overlays: list[Overlay] = []

    for idx, s in enumerate(ch["scenes"]):
        assets = scene_assets(s) or [None]
        part = s["_dur"] / len(assets)
        for ai, a in enumerate(assets):
            p = HERE / a if a else None
            start = s["_local_start"] + ai * part
            if p and p.exists() and p.suffix.lower() in VIDEO_EXTS:
                fn = boomerang_fill(str(p))
            elif p and p.exists() and p.suffix.lower() in IMAGE_EXTS:
                fn = ken_burns(str(p), part, zoom_in)
                zoom_in = not zoom_in
            else:
                fn = ken_burns(placeholder_png(s, a or "an asset", idx + ai), part, zoom_in)
                zoom_in = not zoom_in
            bases.append((start, start + part, fn))

        if CAPTIONS.get("burn"):
            words = load_words(s)
            if words:
                for c in chunk_words(words, CAPTIONS.get("maxWords", 6),
                                     CAPTIONS.get("maxChars", 40)):
                    overlays.append(Overlay(
                        pill_png(c["text"], int(CAPTIONS.get("size", 52) * S),
                                 CAPTIONS.get("fill", "#FFFFFF")),
                        s["_vo_local"] + c["start"], max(0.25, c["end"] - c["start"]),
                        "center", int(CAPTIONS.get("y", 930) * S)))

    if CHAPTER_CARD.get("enabled", True) and ch["title"]:
        label = (f"{ch['index']:02d}  {ch['title']}" if CHAPTER_CARD.get("numbered", True)
                 else ch["title"])
        start = 0.6 if ch["index"] == 1 else 0.3
        overlays.append(Overlay(
            pill_png(label, int(CHAPTER_CARD.get("size", 44) * S), PALETTE["white"], alpha=200),
            start, min(CHAPTER_CARD.get("seconds", 3.5), ch["duration"] - start),
            int(CHAPTER_CARD.get("x", 72) * S), int(CHAPTER_CARD.get("y", 64) * S),
            fade_in=0.4, fade_out=0.5))

    blank = np.empty((H, W, 3), dtype=np.uint8)
    blank[:] = ink.astype(np.uint8)

    def frame(t: float) -> np.ndarray:
        img = None
        for start, end, fn in bases:
            if start <= t < end:
                f = fade_factor(t, start, end, xfade, xfade)
                img = np.array(fn(t - start), dtype=np.uint8)
                if f < 1.0:
                    img = (ink + (img.astype(np.float32) - ink) * f).astype(np.uint8)
                break
        if img is None:
            img = blank.copy()
        for ov in overlays:
            if ov.start <= t < ov.end:
                ov.blend(img, t)
        return img

    clip = VideoClip(frame, duration=ch["duration"])
    tmp = out.with_suffix(".tmp.mp4")
    clip.write_videofile(str(tmp), fps=FPS, codec="libx264", audio=False,
                         preset="veryfast" if preview else "medium",
                         ffmpeg_params=["-crf", "26" if preview else "18",
                                        "-pix_fmt", "yuv420p"],
                         threads=8, logger=None)
    tmp.replace(out)
    return zoom_in


# ── audio ──

def build_audio(chapters: list[dict], total: float, wav: Path) -> bool:
    """One continuous VO track: every scene MP3 delayed to its anchored start
    (music is mixed in master). Plain ffmpeg adelay + amix — sample-accurate
    and independent of video rendering."""
    parts = []
    for ch in chapters:
        for s in ch["scenes"]:
            mp3 = scene_mp3(s)
            if mp3.exists() and s["_measured"]:
                parts.append((mp3, ch["start"] + s["_vo_local"]))
    if not parts:
        return False
    inputs, graph = [], []
    for i, (mp3, start) in enumerate(parts):
        inputs += ["-i", str(mp3)]
        ms = int(round(start * 1000))
        graph.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"adelay={ms}|{ms}[a{i}]")
    graph.append("".join(f"[a{i}]" for i in range(len(parts)))
                 + f"amix=inputs={len(parts)}:normalize=0:duration=longest,"
                   f"apad=whole_dur={total:.3f}[out]")
    script = wav.with_suffix(".filter")
    script.write_text(";\n".join(graph))
    r = ffmpeg(*inputs, "-filter_complex_script", str(script), "-map", "[out]",
               "-t", f"{total:.3f}", "-c:a", "pcm_s16le", str(wav))
    if r.returncode:
        sys.exit(f"ffmpeg VO track failed:\n{r.stderr[-1500:]}")
    return True


def ffmpeg(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ffmpeg", "-hide_banner", "-y", *args],
                          capture_output=True, text=True)


def master(chapter_files: list[Path], vo_wav: Path | None, total: float,
           out: Path, preview: bool) -> None:
    """Concat chapter videos, mix VO + ducked music, loudnorm to YouTube spec."""
    work = out.parent / ".work"
    work.mkdir(parents=True, exist_ok=True)
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in chapter_files))

    music_cfg = CONFIG.get("music", {})
    music = HERE / music_cfg.get("file", "audio/music.mp3")
    have_music = music.exists()
    if not vo_wav and not have_music:
        r = ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                   "-movflags", "+faststart", str(out))
        if r.returncode:
            sys.exit(f"ffmpeg concat failed:\n{r.stderr[-1500:]}")
        print("[no VO or music found — rendered silent]")
        return

    # 1) mix to a wav: VO as-is, music looped + ducked under the VO
    mix = work / "mix.wav"
    inputs, graph = [], ""
    if vo_wav:
        inputs += ["-i", str(vo_wav)]
    if have_music:
        inputs += ["-stream_loop", "-1", "-i", str(music)]
    fade = max(0.0, total - 3.0)
    vol = music_cfg.get("volume", 0.18)
    if vo_wav and have_music:
        duck = music_cfg.get("duck", True)
        graph = (f"[1:a]aresample=48000,volume={vol},atrim=0:{total:.3f},"
                 f"afade=t=out:st={fade:.3f}:d=3[m];"
                 "[0:a]asplit=2[vo][sc];")
        if duck:
            graph += ("[m][sc]sidechaincompress=threshold=0.02:ratio=6:"
                      "attack=30:release=600[md];[vo][md]")
        else:
            graph += "[vo][m]"
        graph += "amix=inputs=2:duration=first:normalize=0[a]"
    elif have_music:
        graph = (f"[0:a]aresample=48000,volume={vol},atrim=0:{total:.3f},"
                 f"afade=t=out:st={fade:.3f}:d=3[a]")
    else:
        graph = "[0:a]anull[a]"
    r = ffmpeg(*inputs, "-filter_complex", graph, "-map", "[a]",
               "-t", f"{total:.3f}", "-ar", "48000", "-c:a", "pcm_s16le", str(mix))
    if r.returncode:
        sys.exit(f"ffmpeg audio mix failed:\n{r.stderr[-1500:]}")

    # 2) loudnorm, two-pass (measure → apply linear) to the YouTube target
    target = AUDIO.get("lufs", -14.0)
    tp = AUDIO.get("truePeak", -1.0)
    af = "anull"
    if AUDIO.get("normalize", True):
        r = ffmpeg("-i", str(mix), "-af",
                   f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json",
                   "-f", "null", "-")
        try:
            js = r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1]
            m = json.loads(js)
            af = (f"loudnorm=I={target}:TP={tp}:LRA=11:"
                  f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
                  f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
                  f"offset={m['target_offset']}:linear=true,aresample=48000")
            print(f"  loudness: {m['input_i']} LUFS → {target} LUFS")
        except (ValueError, KeyError):
            print("  [loudnorm measurement failed — mixing without normalisation]")

    # 3) mux: chapter video stream-copied, audio normalised + AAC
    r = ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(mix),
               "-map", "0:v", "-map", "1:a", "-c:v", "copy",
               "-af", af, "-c:a", "aac", "-b:a", "192k" if not preview else "128k",
               "-t", f"{total:.3f}", "-movflags", "+faststart", str(out))
    if r.returncode:
        sys.exit(f"ffmpeg mux failed:\n{r.stderr[-1500:]}")


# ── deliverables ──

def write_sidecars(chapters: list[dict], total: float, out_dir: Path) -> None:
    meta = json.loads((HERE / "scenes.json").read_text())

    # chapter timestamps + YouTube's rules for them to activate
    lines = [f"{fmt_ts(ch['start'])} {ch['title']}" for ch in chapters]
    problems = []
    if len(chapters) < 3:
        problems.append(f"only {len(chapters)} chapter(s) — YouTube needs at least 3")
    short = [ch["title"] for ch in chapters if ch["duration"] < 10]
    if short:
        problems.append(f"chapters under 10s (YouTube ignores the list): {', '.join(short)}")
    desc = meta.get("description", "").strip()
    body = (desc + "\n\n" if desc else "") + "Chapters\n" + "\n".join(lines) + "\n"
    if meta.get("descriptionFooter"):
        body += "\n" + meta["descriptionFooter"].strip() + "\n"
    (out_dir / "youtube-description.txt").write_text(body)

    # SRT for YouTube subtitles — sentence-aware chunks, global timing
    srt_cfg = CONFIG.get("srt", {})
    entries, missing = [], 0
    for ch in chapters:
        for s in ch["scenes"]:
            words = load_words(s)
            if not words:
                missing += 1
                continue
            base = ch["start"] + s["_vo_local"]
            for c in chunk_words(words, srt_cfg.get("maxWords", 12),
                                 srt_cfg.get("maxChars", 84)):
                entries.append((base + c["start"], base + c["end"] + 0.05, c["text"]))
    srt_path = out_dir / "captions.srt"
    if entries:
        srt = []
        for i, (a, b, txt) in enumerate(entries, 1):
            if i < len(entries):
                b = min(b, entries[i][0])
            srt.append(f"{i}\n{fmt_srt(a)} --> {fmt_srt(b)}\n{txt}\n")
        srt_path.write_text("\n".join(srt))
    elif srt_path.exists():
        srt_path.unlink()

    (out_dir / "timeline.json").write_text(json.dumps({
        "total": round(total, 3),
        "chapters": [{
            "index": ch["index"], "title": ch["title"],
            "start": round(ch["start"], 3), "duration": round(ch["duration"], 3),
            "scenes": [{"id": s["id"], "slug": s["slug"],
                        "start": round(ch["start"] + s["_local_start"], 3),
                        "vo": round(s["_vo"], 3), "measured": s["_measured"]}
                       for s in ch["scenes"]],
        } for ch in chapters],
    }, indent=2))

    print("── YouTube chapters (in youtube-description.txt) ──")
    for ln in lines:
        print(f"  {ln}")
    for p in problems:
        print(f"  ⚠ {p}")
    if entries:
        print(f"  captions.srt: {len(entries)} cues"
              + (f" ({missing} scene(s) without timing — run gen_captions.py)" if missing else ""))
    else:
        print("  [no caption timing — run gen_captions.py for captions.srt]")


def main() -> None:
    global S, W, H
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview", action="store_true", help="Half resolution, fast encode")
    ap.add_argument("--chapter", type=int, help="Render only this chapter (1-based)")
    ap.add_argument("--force", action="store_true", help="Ignore the chapter cache")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required on PATH")
    if args.preview:
        S = 0.5
        W, H = int(FMT["width"] * S) // 2 * 2, int(FMT["height"] * S) // 2 * 2

    meta = json.loads((HERE / "scenes.json").read_text())
    scenes = meta["scenes"]
    dur_path = HERE / "vo_durations.json"
    durations = json.loads(dur_path.read_text()) if dur_path.exists() else {}
    if not durations:
        print("[no vo_durations.json — using word-count estimates, no audio. "
              "Run gen_vo.py for the real thing]")

    chapters, total = build_timeline(scenes, durations)
    print(f"── {meta.get('title', 'untitled')}: {len(chapters)} chapters, "
          f"{len(scenes)} scenes, {total / 60:.1f} min ──")
    for ch in chapters:
        print(f"  {ch['index']:02d} {fmt_ts(ch['start']):>8}  {ch['title']}  "
              f"({len(ch['scenes'])} scenes, {ch['duration']:.1f}s)")
        for s in ch["scenes"]:
            flag = "" if s["_measured"] else " (est.)"
            long = ""
            if s["_vo"] > 30 and len(scene_assets(s)) < 2:
                long = "  ← long scene on one visual: give it a list of assets"
            print(f"       {s['id']} {s['slug']:20s} vo {s['_vo']:6.1f}s{flag}{long}")

    out_dir = HERE / "out"
    ch_dir = out_dir / ("chapters-preview" if args.preview else "chapters")
    ch_dir.mkdir(parents=True, exist_ok=True)

    targets = chapters
    if args.chapter:
        targets = [c for c in chapters if c["index"] == args.chapter]
        if not targets:
            sys.exit(f"No chapter {args.chapter} (have 1–{len(chapters)})")

    zoom_in = True
    files = []
    for ch in chapters:
        f = ch_dir / f"{ch['index']:02d}.mp4"
        files.append(f)
        if ch not in targets:
            continue
        key = chapter_key(ch, args.preview)
        stamp = f.with_suffix(".key")
        if not args.force and f.exists() and stamp.exists() and stamp.read_text() == key:
            print(f"  chapter {ch['index']:02d}: cached")
            continue
        print(f"  chapter {ch['index']:02d}: rendering {ch['duration']:.1f}s …", flush=True)
        zoom_in = render_chapter(ch, f, args.preview, zoom_in)
        stamp.write_text(key)

    if args.chapter:
        print(f"wrote {files[args.chapter - 1]}  (video only — full build mixes audio)")
        return

    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    vo_wav = work / "vo.wav"
    has_vo = build_audio(chapters, total, vo_wav)
    out = out_dir / ("preview.mp4" if args.preview else "video.mp4")
    print("── mastering ──")
    master(files, vo_wav if has_vo else None, total, out, args.preview)
    write_sidecars(chapters, total, out_dir)
    print(f"wrote {out}  ({total / 60:.1f} min)")
    if total < 480:
        print("note: under 8 min — YouTube allows mid-roll ads only on 8 min+ videos")


if __name__ == "__main__":
    main()
