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

    python3 build.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        concatenate_videoclips,
        vfx,
    )
    from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
    from moviepy.audio.fx.AudioLoop import AudioLoop
    from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("    python3 -m pip install -r tools/requirements.txt   (from toolkit root)")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
TEXT_CACHE = HERE / ".text_cache"

CONFIG = json.loads((HERE / "config.json").read_text())
FMT = CONFIG["format"]
PALETTE = CONFIG["palette"]
CAPTIONS = CONFIG.get("captions", {})
TIMING = CONFIG.get("timing", {})

W, H, FPS = FMT["width"], FMT["height"], FMT["fps"]
START_PAD = TIMING.get("startPad", 0.3)
LEAD = TIMING.get("lead", 0.4)     # visual on screen before VO starts
TAIL = TIMING.get("tail", 0.8)     # visual hold after VO ends
XFADE = TIMING.get("xfade", 0.45)  # fade to black per scene edge

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}

_FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"],
    "Windows": ["C:/Windows/Fonts/arialbd.ttf"],
}


def _load_font(size: int):
    for path in _FONT_CANDIDATES.get(platform.system(), []):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
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


def ken_burns(img_path: str, duration: float, zoom_in: bool) -> ImageClip:
    """Slow zoom on a still; base scale covers the frame at any source size."""
    with Image.open(img_path) as im:
        base = max(W / im.width, H / im.height)
    amp = 0.07
    if zoom_in:
        scale = lambda t: base * (1.0 + amp * (t / duration))
    else:
        scale = lambda t: base * (1.0 + amp * (1 - t / duration))
    return (ImageClip(img_path).with_duration(duration)
            .resized(scale).with_position(("center", "center")))


def boomerang_fill(video_path: str, duration: float):
    """Loop a short clip forward/backward until it covers `duration`."""
    fwd = VideoFileClip(video_path).without_audio()
    fwd = fwd.resized(max(W / fwd.w, H / fwd.h)).with_position(("center", "center"))
    rev = fwd.with_effects([vfx.TimeMirror()])
    segs, total = [], 0.0
    while total < duration + 0.5:
        nxt = fwd if len(segs) % 2 == 0 else rev
        segs.append(nxt)
        total += nxt.duration
    return concatenate_videoclips(segs).subclipped(0, duration)


def estimate_duration(text: str) -> float:
    return max(2.0, len(text.split()) / 2.4)  # ~145 wpm fallback estimate


def build() -> None:
    scenes = json.loads((HERE / "scenes.json").read_text())["scenes"]
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
        scene_start = cursor
        audio_start = scene_start + LEAD
        scene_dur = LEAD + vo_dur + TAIL
        print(f"  {s['id']} {s['slug']:16s} {scene_start:7.2f} → "
              f"{scene_start + scene_dur:7.2f}  (vo {vo_dur:.2f}s)")

        asset = HERE / s["asset"] if s.get("asset") else None
        if asset and asset.exists() and asset.suffix.lower() in VIDEO_EXTS:
            vis = boomerang_fill(str(asset), scene_dur)
        elif asset and asset.exists() and asset.suffix.lower() in IMAGE_EXTS:
            vis = ken_burns(str(asset), scene_dur, zoom_in)
            zoom_in = not zoom_in
        else:
            if asset:
                print(f"     [missing {s['asset']} — placeholder card]")
            vis = ken_burns(placeholder_png(s, idx), scene_dur, zoom_in)
            zoom_in = not zoom_in
        clips.append(vis.with_start(scene_start)
                        .with_effects([vfx.FadeIn(XFADE), vfx.FadeOut(XFADE)]))

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

        cursor = scene_start + scene_dur

    total = cursor + 0.4
    clips[0] = clips[0].with_duration(total)
    print(f"  total: {total:.2f}s"
          + ("  (>3 min: fine for TikTok, too long for Shorts/Reels)" if total > 180 else ""))

    music_cfg = CONFIG.get("music", {})
    music_path = HERE / music_cfg.get("file", "audio/music.mp3")
    if music_path.exists():
        audio.insert(0, AudioFileClip(str(music_path)).with_effects([
            AudioLoop(duration=total),
            MultiplyVolume(music_cfg.get("volume", 0.16)),
            AudioFadeOut(2.0),
        ]))

    final = CompositeVideoClip(clips, size=(W, H)).with_duration(total)
    if audio:
        final = final.with_audio(CompositeAudioClip(audio))
    else:
        print("[no audio found — rendering silent]")

    out = HERE / "out" / "short.mp4"
    out.parent.mkdir(exist_ok=True)
    final.write_videofile(str(out), fps=FPS, codec="libx264",
                          audio_codec="aac", preset="medium", threads=8)
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
