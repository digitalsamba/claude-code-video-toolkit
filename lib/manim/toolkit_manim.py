"""Helpers for Manim scenes rendered through tools/manim_render.py (which puts this folder on the path).

    from toolkit_manim import *      # Manim, plus ToolkitScene, Loop, Voice, Brand

- ToolkitScene.at(seconds) waits until a narration cue and fails the render when the scene is already past it.
- ToolkitScene.ev("key") notes the second something audible happens; the notes are written beside the clip.
- Loop.run(build) plays build(phase) for phase 0..1, for a background that leads back into its first frame.
- A portrait size makes the frame 9 by 16 units, so one scene file can lay itself out for both shapes.
What a first spike found, and why each of these exists, is in docs/manim.md."""
import json
import os
from pathlib import Path

from manim import *  # noqa: F401,F403
from manim import Scene, ThreeDScene, ValueTracker, always_redraw, config, linear

from core import Brand, Voice, frame_size  # noqa: F401

_W, _H = frame_size()
if _H > _W:
    # asking Manim for 1080x1920 changes the pixels only; say what a portrait frame measures
    config.frame_width, config.frame_height = 9.0, 16.0
PORTRAIT = _H > _W


class _Toolkit:
    def at(self, seconds: float, slack: float = 1 / 30):
        """Wait until `seconds` on the scene's clock. Refuses to go back in time: an animation that overran its
        slot would push everything after it late without a word."""
        now = self.renderer.time
        if seconds < now - slack:
            raise RuntimeError(f"cue at {seconds:.2f}s is {now - seconds:.2f}s in the past: shorten the animation before it")
        if seconds > now:
            self.wait(seconds - now)

    def ev(self, what: str, after: float = 0.0):
        """Note that `what` happens now (or `after` seconds from now), for placing sound from the scene."""
        self.__dict__.setdefault("_events", []).append({"at": round(self.renderer.time + after, 4), "what": what})

    def tear_down(self):
        events = self.__dict__.get("_events")
        if events:
            path = Path(os.environ.get("MANIM_TOOLKIT_EVENTS") or f"events/{type(self).__name__}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(events, indent=1) + "\n")
        super().tear_down()


class ToolkitScene(_Toolkit, Scene):
    pass


class ToolkitScene3D(_Toolkit, ThreeDScene):
    pass


class Loop(ToolkitScene):
    """A loop that closes on itself. Give every motion a whole number of turns over the phase and the end meets the start."""
    seconds = 6.0

    def run(self, build):
        phase = ValueTracker(0)
        self.add(always_redraw(lambda: build(phase.get_value())))
        self.play(phase.animate.set_value(1), run_time=self.seconds, rate_func=linear)
