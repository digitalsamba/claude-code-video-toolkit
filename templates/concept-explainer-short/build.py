#!/usr/bin/env python3
"""Step 3 — compose the short. Audio-anchored moviepy build.

Scene starts derive from actual VO durations (vo_durations.json), so timing
drift is impossible. Per scene, the visual is picked by asset extension:

  .png/.jpg/.jpeg/.webp  → slow Ken Burns zoom (alternating in/out)
  .mp4/.mov/.webm        → boomerang loop (forward + reversed), cut to fit
  missing                → gradient placeholder card (render works
                           out-of-the-box before any assets exist)

Captions (captions/words_*.json, from gen_captions.py) are burned as karaoke
pills sized/positioned via config.json. VO + ducked looped music are mixed if
present. Run from this project directory:

    uv run build.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoClip,
        VideoFileClip,
        concatenate_videoclips,
        vfx,
    )
    from moviepy.config import FFMPEG_BINARY
    from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
    from moviepy.audio.fx.AudioLoop import AudioLoop
    from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("    uv sync   (from toolkit root)")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
TEXT_CACHE = HERE / ".text_cache"

CONFIG = json.loads((HERE / "config.json").read_text())
FMT = CONFIG["format"]
PALETTE = CONFIG["palette"]
CAPTIONS = CONFIG.get("captions", {})
TIMING = CONFIG.get("timing", {})
ENCODE = CONFIG.get("encode", {})   # {"preset": "medium", "maxMB": null}; maxMB caps the file size

W, H, FPS = FMT["width"], FMT["height"], FMT["fps"]
START_PAD = TIMING.get("startPad", 0.3)
LEAD = TIMING.get("lead", 0.4)     # visual on screen before VO starts
TAIL = TIMING.get("tail", 0.8)     # visual hold after VO ends
XFADE = TIMING.get("xfade", 0.45)  # fade to black per scene edge

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}

_FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"],
    "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "Windows": ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"],
}


def _load_font(size: int):
    for path in _FONT_CANDIDATES.get(platform.system(), []):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1: a scalable fallback
    except TypeError:
        return ImageFont.load_default()


def _rgb(hex_color: str) -> tuple:
    return tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))


def render_caption_png(txt: str) -> str:
    """Caption text on a dark rounded pill, cached, width-capped to the safe
    area. PIL, not TextClip — moviepy 2.x TextClip clips ascenders."""
    size = CAPTIONS.get("size", 80)
    fill = CAPTIONS.get("fill", "#FFFFFF")
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{txt}|{size}|{fill}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    font = _load_font(size)
    sw = max(4, size // 14)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = probe.textbbox((0, 0), txt, font=font, stroke_width=sw)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(24, size // 4)

    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, img.width - 1, img.height - 1),
                           radius=img.height // 4, fill=(*_rgb(PALETTE["ink"]), 185))
    draw.text((pad - bbox[0], pad - bbox[1]), txt, font=font,
              fill=(*_rgb(fill), 255), stroke_width=sw,
              stroke_fill=(*_rgb(PALETTE["ink"]), 235))

    max_w = int(W * 0.91)
    if img.width > max_w:
        img = img.resize((max_w, max(1, int(img.height * max_w / img.width))),
                         Image.LANCZOS)
    img.save(path)
    return str(path)


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cur and draw.textbbox((0, 0), cand, font=font)[2] > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def headline_png(text: str, size: int, fill: str, max_w: int, shadow: bool = True) -> str:
    """Wrapped headline with a soft shadow, transparent background, cached."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"hl|{text}|{size}|{fill}|{max_w}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)
    font = _load_font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = _wrap(probe, text, font, max_w)
    lh = int(size * 1.15)
    widths = [probe.textbbox((0, 0), ln, font=font)[2] for ln in lines]
    pad = size // 3
    img = Image.new("RGBA", (max(widths) + pad * 2, lh * len(lines) + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        x = (img.width - widths[i]) // 2
        y = pad + i * lh
        if shadow:
            draw.text((x + 3, y + 4), ln, font=font, fill=(0, 0, 0, 150))
        draw.text((x, y), ln, font=font, fill=(*_rgb(fill), 255),
                  stroke_width=max(1, size // 40), stroke_fill=(*_rgb(PALETTE["ink"]), 120))
    img.save(path)
    return str(path)


def _ease_out(t: float, d: float) -> float:
    x = max(0.0, min(1.0, t / d))
    return 1 - (1 - x) ** 3


def scrim_png(height: int) -> str:
    """A top-down dark gradient so headlines stay legible over a busy background."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    path = TEXT_CACHE / f"scrim_{W}x{height}.png"
    if path.exists():
        return str(path)
    img = Image.new("RGBA", (W, height), (0, 0, 0, 0))
    px = img.load()
    ink = _rgb(PALETTE["ink"])
    for y in range(height):
        t = y / height
        a = int(175 * (1 - t) ** 1.6)
        for x in range(W):
            px[x, y] = (*ink, a)
    img.save(path)
    return str(path)


def overlay_clips(headline: str, subline: str, start: float, duration: float, big: bool = False) -> list:
    """Animated headline block: headline slides up and fades in, an accent bar draws across
    beneath it, the subline follows. Sits in the upper half, clear of the caption band."""
    out = []
    if duration <= 0.8:
        return out
    vertical = H > W
    out.append(ImageClip(scrim_png(int(H * 0.62))).with_start(start).with_duration(duration)
               .with_position((0, 0)).with_effects([vfx.FadeIn(0.45), vfx.FadeOut(0.35)]))
    size = int(W * (0.085 if vertical else 0.052) * (1.25 if big else 1.0))
    max_w = int(W * 0.84)
    y_top = int(H * (0.14 if vertical else 0.16))
    hl = ImageClip(headline_png(headline, size, PALETTE["white"], max_w))
    rise = 36
    hl = (hl.with_start(start).with_duration(duration)
            .with_position(lambda t: ("center", y_top + int(rise * (1 - _ease_out(t, 0.7)))))
            .with_effects([vfx.FadeIn(0.45), vfx.FadeOut(0.35)]))
    out.append(hl)
    bar_w = int(W * 0.14)
    bar_h = max(4, size // 12)
    bar_y = y_top + hl.h + size // 5
    bar = (ColorClip((bar_w, bar_h), color=_rgb(PALETTE["accent2"]))
           .with_start(start + 0.25).with_duration(max(0.1, duration - 0.25))
           .resized(lambda t: (max(2, int(bar_w * _ease_out(t, 0.6))), bar_h))
           .with_position(lambda t: ((W - max(2, int(bar_w * _ease_out(t, 0.6)))) // 2, bar_y))
           .with_effects([vfx.FadeOut(0.35)]))
    out.append(bar)
    if subline:
        sub = ImageClip(headline_png(subline, int(size * 0.56), PALETTE["white"], max_w, shadow=True))
        sub = (sub.with_start(start + 0.45).with_duration(max(0.1, duration - 0.45))
                  .with_position(("center", bar_y + bar_h + size // 3))
                  .with_effects([vfx.FadeIn(0.5), vfx.FadeOut(0.35)]))
        out.append(sub)
    return out


def placeholder_png(scene: dict, idx: int) -> str:
    """Gradient card with the scene slug — used when the asset is missing so
    the project renders before any Ideogram/LTX generation has happened."""
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    accent = PALETTE["accent1"] if idx % 2 == 0 else PALETTE["accent2"]
    key = hashlib.sha1(f"ph|{scene['slug']}|{accent}|{W}x{H}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    ink, acc = _rgb(PALETTE["ink"]), _rgb(accent)
    img = Image.new("RGB", (W, H), ink)
    draw = ImageDraw.Draw(img)
    for y in range(H):  # vertical gradient, accent rising from the bottom
        t = max(0.0, y / H - 0.45) * 0.5
        draw.line([(0, y), (W, y)],
                  fill=tuple(int(i + (a - i) * t) for i, a in zip(ink, acc)))
    font = _load_font(110)
    label = scene["slug"].replace("_", " ").upper()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.text(((W - bbox[2] + bbox[0]) / 2, H * 0.42), label,
              font=font, fill=_rgb(PALETTE["white"]))
    small = _load_font(40)
    hint = f"placeholder — add {scene.get('asset', 'an asset')}"
    bbox = draw.textbbox((0, 0), hint, font=small)
    draw.text(((W - bbox[2] + bbox[0]) / 2, H * 0.52), hint,
              font=small, fill=_rgb(PALETTE["slate"]))
    img.save(path)
    return str(path)


KB_AMP = 0.07   # Ken Burns zoom range: the still drifts between 1.0x and 1.07x of a frame-covering fit


def ken_burns(img_path: str, duration: float, zoom_in: bool, focus: float = 0.5) -> VideoClip:
    """Slow zoom on a still. The source is scaled once to cover the frame at the deepest
    zoom; each frame is then a single bilinear resize of a sub-pixel box out of it. That is
    smoother than resizing the whole source per frame to a rounded size (no 1 px jitter)
    and several times cheaper, which matters on small CPUs (a LANCZOS resize of a
    1440x2560 card per frame was the build's biggest cost)."""
    with Image.open(img_path) as im:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            # a transparent card shows the palette's ink behind it, as it did when this was
            # an ImageClip with a mask over the ink background
            flat = Image.new("RGBA", im.size, (*_rgb(PALETTE["ink"]), 255))
            im = Image.alpha_composite(flat, im.convert("RGBA"))
        im = im.convert("RGB")
        base = max(W / im.width, H / im.height) * (1 + KB_AMP)
        big = im.resize((max(W, math.ceil(im.width * base)), max(H, math.ceil(im.height * base))),
                        Image.LANCZOS)
    bw, bh = big.size
    # the frame at zoom z (1..1+amp) is a window of the big image of size frame*(1+amp)/z
    # (centred, or at `focus` 0-1 along the side that is cropped), resized to the frame; at the deepest
    # zoom that window is the frame itself
    f = min(1.0, max(0.0, float(focus)))

    def frame(t: float):
        p = min(1.0, max(0.0, t / duration)) if duration > 0 else 0.0
        z = 1.0 + KB_AMP * (p if zoom_in else 1 - p)
        cw, ch = W * (1 + KB_AMP) / z, H * (1 + KB_AMP) / z
        x0, y0 = (bw - cw) * f, (bh - ch) * f
        return np.asarray(big.resize((W, H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch)))

    return VideoClip(frame_function=frame, duration=duration)


def _boomerang_file(video_path: str, focus: float = 0.5) -> str | None:
    """The clip played forward then backward, scaled and centre-cropped to the frame, made
    once with ffmpeg and cached. Playing that file forward is cheap; the old path reversed
    in moviepy, which re-seeks the decoder for every backward frame, and resized every
    frame in Python. None when ffmpeg fails (the caller falls back to the old path)."""
    src = Path(video_path)
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    f = min(1.0, max(0.0, float(focus)))
    key = hashlib.sha1(f"boom|{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{W}x{H}|{f:.3f}".encode()).hexdigest()[:16]
    out = TEXT_CACHE / f"clip_{key}.mp4"
    if out.exists():
        return str(out)
    # reverse at the source size (an LTX clip is ~140 MB of frames), scale after
    graph = (f"[0:v]split[f][b];[b]reverse[r];[f][r]concat=n=2:v=1:a=0,"
             f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,"
             f"crop={W}:{H}:(in_w-{W})*{f:.3f}:(in_h-{H})*{f:.3f},setsar=1[v]")
    cmd = [FFMPEG_BINARY, "-v", "error", "-y", "-i", str(src), "-filter_complex", graph, "-map", "[v]",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=900)
        return str(out)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"     [ffmpeg boomerang failed ({e}); building it in moviepy]")
        out.unlink(missing_ok=True)
        return None


def boomerang_fill(video_path: str, duration: float, focus: float = 0.5):
    """Loop a short clip forward/backward until it covers `duration`. `focus` (0-1) is where the crop
    sits along the side that doesn't fit: 0 left/top, 0.5 centre, 1 right/bottom."""
    pre = _boomerang_file(video_path, focus)
    if pre:
        loop = VideoFileClip(pre).without_audio()
        segs = [loop] * max(1, math.ceil((duration + 0.1) / loop.duration))
        return concatenate_videoclips(segs).subclipped(0, duration)
    fwd = VideoFileClip(video_path).without_audio()
    fwd = fwd.resized(max(W / fwd.w, H / fwd.h))
    f = min(1.0, max(0.0, float(focus)))
    fwd = fwd.cropped(x1=round((fwd.w - W) * f), y1=round((fwd.h - H) * f), width=W, height=H)
    rev = fwd.with_effects([vfx.TimeMirror()])
    segs, total = [], 0.0
    while total < duration + 0.5:
        nxt = fwd if len(segs) % 2 == 0 else rev
        segs.append(nxt)
        total += nxt.duration
    return concatenate_videoclips(segs).subclipped(0, duration)


def encode_params(total_s: float, max_mb: float | None, audio_kbps: int = 192) -> list[str]:
    """x264 settings that keep the file under max_mb: CRF 23 as usual, with a VBV ceiling at
    the bitrate the size allows (5% margin, audio set aside). Measured on a 50 s 16:9 render:
    11.7 MB uncapped, 8.3 MB capped at 9.5 MB, SSIM 0.993 against the original. Discord's
    bot upload limit is 10 MB, which is what jobs set this for."""
    if not max_mb or total_s <= 0:
        return []
    kbps = int(max_mb * 8_000 * 0.95 / total_s) - audio_kbps
    kbps = max(300, kbps)
    return ["-crf", "23", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k"]


def estimate_duration(text: str) -> float:
    return max(2.0, len(text.split()) / 2.4)  # ~145 wpm fallback estimate


def build() -> None:
    doc = json.loads((HERE / "scenes.json").read_text())
    scenes = doc["scenes"]
    title_card = doc.get("titleCard") or None
    dur_path = HERE / "vo_durations.json"
    durations = json.loads(dur_path.read_text()) if dur_path.exists() else {}
    if not durations:
        print("[no vo_durations.json — using word-count estimates, no audio. "
              "Run gen_vo.py for the real thing]")

    clips: list = [ColorClip((W, H), color=_rgb(PALETTE["ink"]))]
    audio: list = []
    cursor = START_PAD
    zoom_in = True

    print("── audio-anchored timeline ──")
    for idx, s in enumerate(scenes):
        vo_dur = durations.get(s["id"]) or estimate_duration(s["text"])
        # Scenes crossfade: from the second one on, the visual starts XFADE early and fades
        # in over the previous scene's fade-out, so a join never dips through black (qc.py
        # flagged a 0.5 s black gap at a dark-on-dark join when they ran back to back).
        overlap = XFADE if idx > 0 else 0.0
        content_start = cursor                     # where this scene's own time begins
        content_dur = LEAD + vo_dur + TAIL
        scene_start = content_start - overlap
        audio_start = content_start + LEAD
        scene_dur = overlap + content_dur
        print(f"  {s['id']} {s['slug']:16s} {content_start:7.2f} → "
              f"{content_start + content_dur:7.2f}  (vo {vo_dur:.2f}s)")

        asset = HERE / s["asset"] if s.get("asset") else None
        if asset and asset.exists() and asset.suffix.lower() in VIDEO_EXTS:
            vis = boomerang_fill(str(asset), scene_dur, s.get("focus", 0.5))
        elif asset and asset.exists() and asset.suffix.lower() in IMAGE_EXTS:
            vis = ken_burns(str(asset), scene_dur, zoom_in, s.get("focus", 0.5))
            zoom_in = not zoom_in
        else:
            if asset:
                print(f"     [missing {s['asset']} — placeholder card]")
            vis = ken_burns(placeholder_png(s, idx), scene_dur, zoom_in)
            zoom_in = not zoom_in
        # the first scene is on screen from frame 0 (that frame is the cover on Shorts/Reels)
        fx = [vfx.FadeOut(XFADE)] if idx == 0 and START_PAD == 0 else [vfx.FadeIn(XFADE), vfx.FadeOut(XFADE)]
        clips.append(vis.with_start(scene_start).with_effects(fx))

        ov = s.get("overlay") or {}
        if idx == 0 and title_card and title_card.get("text"):
            secs = min(float(title_card.get("seconds", 2.8)), content_dur - 0.4)
            clips += overlay_clips(title_card["text"], "", content_start, secs, big=True)
            if ov.get("headline"):
                clips += overlay_clips(ov["headline"], ov.get("subline", ""),
                                       content_start + secs + 0.1, content_dur - secs - 0.5)
        elif ov.get("headline"):
            clips += overlay_clips(ov["headline"], ov.get("subline", ""), content_start + 0.35, content_dur - 0.7)

        mp3 = HERE / "audio" / "scenes" / f"{s['id']}_{s['slug']}.mp3"
        if mp3.exists() and s["id"] in durations:
            audio.append(AudioFileClip(str(mp3))
                         .with_effects([MultiplyVolume(1.1)])
                         .with_start(audio_start))

        words_file = HERE / "captions" / f"words_{s['id']}.json"
        if CAPTIONS.get("enabled", True) and words_file.exists():
            for ch in json.loads(words_file.read_text()):
                clips.append(
                    ImageClip(render_caption_png(ch["text"].upper()))
                    .with_duration(max(0.18, ch["end"] - ch["start"]))
                    .with_start(audio_start + ch["start"])
                    .with_position(("center", CAPTIONS.get("y", 1640)))
                )

        cursor = content_start + content_dur

    total = cursor + 0.4
    clips[0] = clips[0].with_duration(total)
    print(f"  total: {total:.2f}s"
          + ("  (>3 min: fine for TikTok, too long for Shorts/Reels)" if total > 180 else ""))

    music_cfg = CONFIG.get("music", {})
    music_file = music_cfg.get("file", "audio/music.mp3")     # null: no bed
    music_path = HERE / music_file if music_file else None
    if music_path and music_path.exists():
        audio.insert(0, AudioFileClip(str(music_path)).with_effects([
            AudioLoop(duration=total),
            MultiplyVolume(music_cfg.get("volume", 0.16)),
            AudioFadeOut(2.0),
        ]))

    # use_bgclip: clips[0] is the opaque ink background, so moviepy skips building a
    # full-frame float mask for every frame (the overlays have masks, the output never does)
    final = CompositeVideoClip(clips, size=(W, H), use_bgclip=True).with_duration(total)
    if audio:
        final = final.with_audio(CompositeAudioClip(audio))
    else:
        print("[no audio found — rendering silent]")

    out = HERE / "out" / "short.mp4"
    out.parent.mkdir(exist_ok=True)
    final.write_videofile(str(out), fps=FPS, codec="libx264",
                          audio_codec="aac", audio_bitrate="192k",
                          preset=ENCODE.get("preset", "medium"), threads=os.cpu_count() or 4,
                          ffmpeg_params=encode_params(total, ENCODE.get("maxMB")) or None,
                          logger="bar" if sys.stderr.isatty() else None)  # no bar spam in job logs
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
