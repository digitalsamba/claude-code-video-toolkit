"""DS CRT boot stinger — composite the real brand mark over the CRT reveal.

Timeline:
  0.0 – 3.5s   CRT boot clip plays normally (phosphor scroll → pixel DS)
  3.5 – 5.0s   Crossfade from CRT clip to the real DS wordmark
  5.0 – 6.0s   Hold on the real wordmark on a dark canvas
"""

import subprocess
import sys
from pathlib import Path

from moviepy import CompositeVideoClip, ImageClip, VideoFileClip, afx, vfx

HERE = Path(__file__).parent
# 1st arg = CRT clip variant (astro | amber | coral | green)
# 2nd arg = grungy-logo palette. Defaults to the variant name. Use when the
#          clip name doesn't match a palette, e.g. `astro amber`.
variant = sys.argv[1] if len(sys.argv) > 1 else "astro"
logo_palette = sys.argv[2] if len(sys.argv) > 2 else "amber"

CRT_CLIP = HERE / f"ds_boot_{variant}.mp4"
LOGO_IMG = HERE / f"ds_logo_grungy_{logo_palette}.png"
OUT = HERE / f"ds_stinger_{variant}.mp4"

if not CRT_CLIP.exists():
    sys.exit(
        f"Missing CRT clip: {CRT_CLIP.name}\n"
        "Generate one with:\n"
        "  python3 ../../tools/ltx2.py --lora crt-terminal "
        f'--prompt "..." --output examples/ds-crt-stinger/{CRT_CLIP.name}'
    )
if not LOGO_IMG.exists():
    print(f"Building grungy logo ({logo_palette})…")
    subprocess.check_call([sys.executable, str(HERE / "make_grungy_logo.py"), logo_palette])

print(f"variant={variant}  logo={LOGO_IMG.name}")

CRT_LEN = 5.0          # CRT source clip is ~5.04s; trim to 5.0 for clean math
CROSSFADE = 1.5        # final 1.5s of the CRT clip crossfades into the logo
LOGO_HOLD = 1.0        # hold the real logo after crossfade completes
FPS = 24

total = CRT_LEN + LOGO_HOLD  # 6.0s

crt = VideoFileClip(str(CRT_CLIP)).subclipped(0, CRT_LEN)
logo = ImageClip(str(LOGO_IMG), duration=CROSSFADE + LOGO_HOLD).with_fps(FPS)

logo = logo.with_start(CRT_LEN - CROSSFADE).with_effects([vfx.CrossFadeIn(CROSSFADE)])
crt_tail_fade = crt.with_effects([vfx.CrossFadeOut(CROSSFADE)])

composite = CompositeVideoClip(
    [crt_tail_fade, logo],
    size=(1024, 1024),
    bg_color=(10, 14, 20),
).with_duration(total)

if crt.audio is not None:
    audio = crt.audio.subclipped(0, CRT_LEN).with_effects([afx.AudioFadeOut(CROSSFADE)])
    composite = composite.with_audio(audio)

composite.write_videofile(
    str(OUT),
    fps=FPS,
    codec="libx264",
    audio_codec="aac",
    preset="medium",
)

print(f"wrote: {OUT}")
