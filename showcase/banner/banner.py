"""The README banner: script, voice, picture and video fold into the mark, then the name arrives beside it.

    uv run tools/manim_render.py render showcase/banner/banner.py Banner --brand video-toolkit \
        --size 1920x888 --out showcase/banner/out/banner.mp4

The chain is the one in examples/manim-ident (Chain), on the brand's navy field with its mark-on-dark, so the
banner reads on GitHub's light and dark themes. It ends on a fade so the GIF loops.
The product names are in the README text, not here: a list of agents dates and leaves out the ones that call
the toolkit as a tool."""
from toolkit_manim import *

B = Brand()
ACCENT, INK, MUTED = B.c("primaryLight"), B.c("textOnDark"), B.c("textOnDarkMuted")
config.background_color = B.c("bgDark")
MONO, SANS = "Menlo", "Helvetica Neue"


def solid(w, h, color=INK):
    return Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=1)


def corners(shape):
    """A polygon's distinct corners."""
    seen = []
    for pt in shape.get_anchors():
        if not any(np.allclose(pt, q, atol=1e-4) for q in seen):
            seen.append(pt)
    return seen


def like_a_rectangle(shape, color):
    """The shape redrawn with a Rectangle's corner order (up right, up left, down left, down right), so a
    Rectangle morphs into it edge for edge. Without this the morph turns inside out and shows slivers."""
    pts = corners(shape)
    if len(pts) == 3:                                   # the play triangle: its right edge closes to the tip
        tip = max(pts, key=lambda q: q[0])
        top = max(pts, key=lambda q: q[1])
        low = min(pts, key=lambda q: q[1])
        order = [tip, top, low, tip]
    else:                                               # the slash
        by_y = sorted(pts, key=lambda q: q[1])
        low, high = sorted(by_y[:2], key=lambda q: q[0]), sorted(by_y[2:], key=lambda q: q[0])
        order = [high[1], high[0], low[0], low[1]]
    return Polygon(*order, stroke_width=0, fill_color=color, fill_opacity=1)


class Banner(ToolkitScene):
    def construct(self):
        mark = SVGMobject(str(B.mark("mark-on-dark"))).set(height=3.6)
        slash, play = mark[0], mark[1]
        centre = play.get_center() + RIGHT * 0.1
        bar = solid(slash.width * 0.36, slash.height, ACCENT).move_to(slash)
        lines = VGroup(*[solid(w, 0.16) for w in (2.3, 1.7, 2.1, 1.2, 1.9)]).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        wave = VGroup(*[solid(0.2, h) for h in (0.9, 2.1, 1.3, 2.5, 1.1)]).arrange(RIGHT, buff=0.26)
        frame = VGroup(*[solid(0.57, 1.6) for _ in range(5)]).arrange(RIGHT, buff=-0.01)
        for group in (lines, wave, frame):
            group.scale(1.05).move_to(centre)
        copies = [like_a_rectangle(play, INK) for _ in range(5)]
        slash_target = like_a_rectangle(slash, ACCENT)
        word = lambda s: Text(s, font=MONO, color=MUTED).set(height=0.4).move_to(DOWN * 2.75)
        label = word("script")

        self.wait(0.3)
        self.play(FadeIn(bar, shift=UP * 0.2), run_time=0.25)
        self.play(LaggedStart(*[GrowFromEdge(l, LEFT) for l in lines], lag_ratio=0.15), FadeIn(label), run_time=0.6)
        self.wait(0.45)
        for target, name in ((wave, "voice"), (frame, "picture")):
            nxt = word(name)
            self.play(*[Transform(a, b) for a, b in zip(lines, target)], FadeOut(label, run_time=0.15),
                      FadeIn(nxt, run_time=0.3), run_time=0.5)
            label = nxt
            self.wait(0.45)
        nxt = word("video")
        self.play(*[Transform(a, b) for a, b in zip(lines, copies)], Transform(bar, slash_target),
                  FadeOut(label, run_time=0.15), FadeIn(nxt, run_time=0.25), run_time=0.45, rate_func=smooth)
        parts = VGroup(bar, *lines)
        self.play(parts.animate(rate_func=there_and_back, run_time=0.22).scale(1.06))
        self.remove(*parts)
        self.add(mark)
        self.wait(0.35)

        name = Text("Agent Video Toolkit", font=SANS, weight=BOLD, color=INK).set(height=1.15)
        sub = Text("Your agent makes the whole video", font=SANS, color=MUTED).set(height=0.6)
        words = VGroup(name, sub).arrange(DOWN, buff=0.42, aligned_edge=LEFT)
        lockup = VGroup(mark.copy(), words).arrange(RIGHT, buff=0.7)
        lockup.set(width=config.frame_width * 0.86).move_to(ORIGIN)      # the frame is 14.2 wide at any aspect
        self.play(Transform(mark, lockup[0]), FadeOut(nxt, run_time=0.2),
                  FadeIn(name, shift=LEFT * 0.6), run_time=0.6, rate_func=smooth)
        self.play(FadeIn(sub, shift=UP * 0.15), run_time=0.35)
        self.wait(3.2)
        self.play(FadeOut(VGroup(mark, words)), run_time=0.35)
        self.wait(0.15)
