"""
quick-spot — a 15-second ad-style example built with moviepy.

Demonstrates the toolkit's "single-file build.py" pattern for short-form
video without Remotion. Run as:

    python3 build.py

By default this produces a silent text-on-color spot using only Pillow +
moviepy. If you generate per-scene voiceover into ./public/audio/scenes/
the build will pick it up automatically — see README.md for the workflow.

Patterns demonstrated
---------------------
* PIL → ImageClip text rendering (workaround for moviepy 2.x TextClip
  ascender-clipping bug — see the moviepy skill).
* Audio-anchored timeline: every visual element has an absolute start
  time anchored to a comment block at the top of build(). No drift.
* Optional per-scene VO + ducked music mixing pattern.
* Solid colour backgrounds with fades — no external assets required.
"""
from __future__ import annotations

import hashlib
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
        vfx,
    )
    from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
    from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
    from moviepy.audio.fx.MultiplyVolume import MultiplyVolume
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install the toolkit's Python dependencies:")
    print("    python3 -m pip install -r ../../tools/requirements.txt")
    print("(run from this directory, or use an absolute path)")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
AUDIO = PUBLIC / "audio"
TEXT_CACHE = HERE / ".text_cache"
OUT = HERE / "out.mp4"

# 1080p 16:9, 30fps
W, H = 1920, 1080
FPS = 30
DURATION = 15.0

# Neutral palette — override these for your brand
NAVY_DARK = (0x14, 0x18, 0x26)
NAVY = (0x32, 0x3F, 0x66)
CORAL_HEX = "#F06859"
WHITE_HEX = "#FFFFFF"
SLATE_HEX = "#8B91A3"

# Cross-platform bold sans-serif fallback chain. PIL needs a real TTF;
# each OS has well-known defaults. If nothing matches, fall back to
# PIL's bitmap default so the example still runs (ugly, but never crashes).
_FONT_CANDIDATES = {
    "Darwin": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ],
    "Windows": [
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}


def _load_font(size: int):
    for path in _FONT_CANDIDATES.get(platform.system(), []):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_text_png(txt: str, size: int, hex_color: str) -> str:
    """Render text to a transparent PNG via PIL. Cached by content hash.

    Why PIL: moviepy 2.x TextClip(method='label') has a tight-bbox bug that
    clips letter ascenders/descenders. PIL doesn't.
    """
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{txt}|{size}|{hex_color}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    font = _load_font(size)
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), txt, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(20, size // 4)

    img = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    rgb = tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    ImageDraw.Draw(img).text(
        (pad - bbox[0], pad - bbox[1]), txt, font=font, fill=(*rgb, 255)
    )
    img.save(path)
    return str(path)


def text_clip(
    txt: str,
    *,
    size: int,
    color: str,
    duration: float,
    start: float,
    position=("center", "center"),
):
    return (
        ImageClip(render_text_png(txt, size, color))
        .with_duration(duration)
        .with_start(start)
        .with_position(position)
    )


def fade(clip, fin: float = 0.25, fout: float = 0.25):
    return clip.with_effects([vfx.FadeIn(fin), vfx.FadeOut(fout)])


def color_bg(rgb: tuple, *, duration: float, start: float = 0.0):
    return ColorClip((W, H), color=rgb).with_duration(duration).with_start(start)


def build():
    clips: list = []

    # ── Audio-anchored scene timeline (15s total) ──────────────────
    #   Scene 1 hook       0.0 → 4.0   "Tired of generic dashboards?"
    #   Scene 2 problem    4.0 → 8.0   3 pain points
    #   Scene 3 reveal     8.0 → 12.0  product name
    #   Scene 4 cta       12.0 → 15.0  call to action
    #
    # Every visual `start=` below references this block. Drift is
    # impossible — durations come from the audio, not estimates.

    # ── scene 1: dark hook (0-4s) ──────────────────────────────────
    clips.append(color_bg(NAVY_DARK, duration=4.0, start=0.0))
    clips.append(fade(text_clip(
        "TIRED OF",
        size=140, color=SLATE_HEX,
        duration=3.5, start=0.3,
        position=("center", 380),
    )))
    clips.append(fade(text_clip(
        "GENERIC DASHBOARDS?",
        size=110, color=CORAL_HEX,
        duration=3.2, start=0.6,
        position=("center", 560),
    )))

    # ── scene 2: pain points (4-8s) ────────────────────────────────
    clips.append(color_bg(NAVY_DARK, duration=4.0, start=4.0))
    pains = [("CONFUSING MENUS.", 4.2), ("SLOW LOADS.", 5.5), ("UGLY CHARTS.", 6.8)]
    for line, when in pains:
        clips.append(fade(text_clip(
            line,
            size=120, color=WHITE_HEX,
            duration=1.2, start=when,
        ), fin=0.15, fout=0.15))

    # ── scene 3: reveal (8-12s) ────────────────────────────────────
    clips.append(fade(color_bg(NAVY, duration=4.0, start=8.0), fin=0.4, fout=0.0))
    clips.append(fade(text_clip(
        "INTRODUCING",
        size=72, color=WHITE_HEX,
        duration=1.0, start=8.4,
        position=("center", 380),
    )))
    clips.append(fade(text_clip(
        "ACME ANALYTICS",
        size=180, color=CORAL_HEX,
        duration=3.0, start=8.8,
        position=("center", 480),
    )))

    # ── scene 4: cta (12-15s) ──────────────────────────────────────
    clips.append(fade(color_bg(NAVY, duration=3.0, start=12.0), fin=0.0, fout=0.4))
    clips.append(fade(text_clip(
        "TRY IT FREE",
        size=140, color=WHITE_HEX,
        duration=3.0, start=12.0,
        position=("center", 420),
    )))
    clips.append(fade(text_clip(
        "acme.example",
        size=72, color=CORAL_HEX,
        duration=3.0, start=12.0,
        position=("center", 620),
    )))

    # ── audio (optional — picked up if files exist) ────────────────
    scene_vo = [
        ("01_hook.mp3",     0.0),
        ("02_problem.mp3",  4.0),
        ("03_reveal.mp3",   8.0),
        ("04_cta.mp3",     12.0),
    ]
    audio = build_audio(scene_vo)

    final = CompositeVideoClip(clips, size=(W, H)).with_duration(DURATION)
    if audio is not None:
        final = final.with_audio(audio)

    final.write_videofile(
        str(OUT), fps=FPS, codec="libx264", audio_codec="aac",
        preset="medium", threads=4,
    )
    print(f"wrote {OUT}")


def build_audio(scene_vo):
    """Mix per-scene VO over ducked music if the assets exist, else None.

    To enable audio:
        python3 ../../tools/voiceover.py \\
            --script VOICEOVER-SCRIPT.md --scene-dir public/audio/scenes
        python3 ../../tools/music_gen.py \\
            --preset cta --duration 15 --output public/audio/music.mp3
    """
    scenes = AUDIO / "scenes"
    vo_files = [scenes / name for name, _ in scene_vo]
    music_path = AUDIO / "music.mp3"

    if not all(f.exists() for f in vo_files):
        print("[no audio] rendering silent — see README to add voiceover")
        return None

    vo_clips = [
        AudioFileClip(str(scenes / name))
        .with_effects([MultiplyVolume(1.15)])
        .with_start(start)
        for name, start in scene_vo
    ]

    if music_path.exists():
        music = AudioFileClip(str(music_path)).with_effects([
            MultiplyVolume(0.22),
            AudioFadeIn(0.5),
            AudioFadeOut(1.5),
        ])
        return CompositeAudioClip([music] + vo_clips)
    return CompositeAudioClip(vo_clips)


if __name__ == "__main__":
    build()
