"""
data-viz-chart — animated chart with deterministic text overlay.

Demonstrates the toolkit's "matplotlib for data, moviepy for trustworthy
text" pattern. Run as:

    python3 build.py

Produces a 15-second animated time-series chart from data/star_series.json
with a title, axis labels, and source attribution rendered via PIL +
moviepy on top of the matplotlib animation.

Why split the work this way
---------------------------
* matplotlib is great at drawing the data (lines, fills, axes, ticks)
  from a data file. Use it for that.
* matplotlib's text rendering across animation frames is fragile and
  the styling controls are clunky compared to design-grade text.
* Anything that needs to be *trustworthy* — chart titles, source
  attribution, headline figures, dates, brand marks — should be
  composited deterministically with moviepy on top.

This is exactly how news graphics are produced: the data layer comes
from a data tool, the text layer comes from a deterministic compositor.

Patterns demonstrated
---------------------
* matplotlib `FuncAnimation` → mp4 → moviepy `VideoFileClip`
* PIL → ImageClip text rendering (workaround for moviepy 2.x TextClip
  ascender-clipping bug — see the moviepy skill)
* Trustworthy text overlay (title, source attribution, headline number)
* Single-file `build.py` pattern, no Remotion required
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates  # noqa: E402
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.animation import FuncAnimation  # noqa: E402
    from PIL import Image, ImageDraw, ImageFont  # noqa: E402
    from moviepy import (  # noqa: E402
        ColorClip,
        CompositeVideoClip,
        ImageClip,
        VideoFileClip,
        vfx,
    )
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install the toolkit's Python dependencies:")
    print("    python3 -m pip install -r ../../tools/requirements.txt")
    print("(run from this directory, or use an absolute path)")
    sys.exit(1)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "star_series.json"
CHART_MP4 = HERE / "chart_anim.mp4"
TEXT_CACHE = HERE / ".text_cache"
OUT = HERE / "out.mp4"

W, H = 1920, 1080
FPS = 30
DURATION = 15.0

NAVY = "#323F66"
NAVY_RGB = (0x32, 0x3F, 0x66)
CORAL_HEX = "#F06859"
SLATE_HEX = "#5E667A"
WHITE_HEX = "#FFFFFF"

# Cross-platform sans-serif fallback chain. PIL needs a real TTF; each
# OS has well-known defaults for bold and regular. If nothing matches,
# fall back to PIL's bitmap default so the example still runs.
_FONT_CANDIDATES = {
    "Darwin": {
        "bold": [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ],
        "regular": [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ],
    },
    "Linux": {
        "bold": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ],
        "regular": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ],
    },
    "Windows": {
        "bold":    ["C:/Windows/Fonts/arialbd.ttf"],
        "regular": ["C:/Windows/Fonts/arial.ttf"],
    },
}


def _load_font(size: int, *, bold: bool = True):
    weight = "bold" if bold else "regular"
    for path in _FONT_CANDIDATES.get(platform.system(), {}).get(weight, []):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# ── PIL text rendering with content-hash cache ────────────────────────

def render_text_png(txt: str, size: int, hex_color: str, *, bold: bool = True) -> str:
    """Render text to a transparent PNG via PIL. Cached by content hash.

    Why PIL: moviepy 2.x TextClip(method='label') has a tight-bbox bug that
    clips letter ascenders/descenders. PIL doesn't.
    """
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{txt}|{size}|{hex_color}|{bold}".encode()).hexdigest()[:16]
    path = TEXT_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    font = _load_font(size, bold=bold)
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
    start: float = 0.0,
    position=("center", "center"),
    bold: bool = True,
):
    return (
        ImageClip(render_text_png(txt, size, color, bold=bold))
        .with_duration(duration)
        .with_start(start)
        .with_position(position)
    )


def fade(clip, fin: float = 0.4, fout: float = 0.4):
    return clip.with_effects([vfx.FadeIn(fin), vfx.FadeOut(fout)])


# ── matplotlib: render the data layer to mp4 ──────────────────────────

def render_chart_animation():
    """Animate the cumulative series to mp4. Data only — no titles."""
    series = json.loads(DATA.read_text())
    days = [datetime.strptime(d, "%Y-%m-%d") for d, _ in series]
    counts = [c for _, c in series]

    frame_count = int(DURATION * FPS)
    total = len(series)
    idx_per_frame = [
        min(total - 1, int((i / frame_count) * total)) for i in range(frame_count)
    ]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=120, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.78, bottom=0.16)
    ax.set_facecolor("white")

    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("Stars", color=NAVY, fontsize=24, fontweight="bold", labelpad=14)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="both", which="major", labelsize=18, colors=NAVY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SLATE_HEX)
    ax.grid(axis="y", color=SLATE_HEX, alpha=0.15)

    # Plot the line + filled area. Title and headline number come from
    # the moviepy text overlay layer — not matplotlib.
    (line,) = ax.plot([], [], color=NAVY, linewidth=5)
    ax.fill_between(days[:1], counts[:1], color=CORAL_HEX, alpha=0.22)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        k = idx_per_frame[i] + 1
        xs, ys = days[:k], counts[:k]
        line.set_data(xs, ys)
        for coll in list(ax.collections):
            coll.remove()
        ax.fill_between(xs, ys, color=CORAL_HEX, alpha=0.22)
        return (line,)

    anim = FuncAnimation(
        fig, update, init_func=init, frames=frame_count,
        interval=1000 / FPS, blit=False,
    )
    anim.save(
        str(CHART_MP4), fps=FPS, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )
    plt.close(fig)
    print(f"wrote {CHART_MP4}")


# ── moviepy: composite the trustworthy text overlay ───────────────────

def build():
    if not CHART_MP4.exists() or CHART_MP4.stat().st_mtime < DATA.stat().st_mtime:
        render_chart_animation()

    series = json.loads(DATA.read_text())
    final_count = series[-1][1]

    chart = VideoFileClip(str(CHART_MP4))
    bg = ColorClip((W, H), color=(255, 255, 255)).with_duration(DURATION)
    chart_clip = chart.resized(width=W).with_position(("center", "center"))

    clips: list = [bg, chart_clip]

    # Title — top of frame, brand colour
    clips.append(fade(text_clip(
        "github.com/digitalsamba/claude-code-video-toolkit",
        size=44, color=NAVY,
        duration=DURATION, start=0.0,
        position=(80, 50),
    ), fin=0.6, fout=0.6))

    # Headline number — large, animates in toward the end as the
    # payoff. (For a true counter you'd interpolate per-frame, which
    # is best done with a custom moviepy clip — left as an exercise.)
    clips.append(fade(text_clip(
        f"{final_count}",
        size=180, color=NAVY,
        duration=4.5, start=10.0,
        position=("right", 130),
    ), fin=0.5, fout=0.5))
    clips.append(fade(text_clip(
        "stars",
        size=44, color=SLATE_HEX,
        duration=4.5, start=10.0,
        position=(W - 240, 320),
    ), fin=0.5, fout=0.5))

    # Source attribution — bottom of frame.
    # This is the *trustworthy text* point of the example: the source
    # name must be readable and exact. moviepy guarantees both. An AI
    # video model cannot.
    clips.append(fade(text_clip(
        "Source: GitHub Stars API",
        size=28, color=SLATE_HEX,
        duration=DURATION, start=0.0,
        position=(80, H - 80),
        bold=False,
    ), fin=0.6, fout=0.6))

    final = CompositeVideoClip(clips, size=(W, H)).with_duration(DURATION)
    final.write_videofile(
        str(OUT), fps=FPS, codec="libx264", audio_codec="aac",
        preset="medium", threads=4,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
