"""Two idents for a brand's mark, each saying what the toolkit does and ending on the flat mark.

    uv run tools/manim_render.py render examples/manim-ident/idents.py Typed --brand video-toolkit --out examples/manim-ident/out/typed.mp4

The mark is the brand's own SVG (brands/<name>/assets/mark.svg): its first shape is the slash, its second the
play triangle. For another brand's mark, change what the parts turn from, not how the scene ends."""
from toolkit_manim import *

B = Brand()
ACCENT, INK, MUTED = B.c("primary"), B.c("textDark"), B.c("textLight")
config.background_color = B.c("bgLight")
MONO = "Menlo"


def the_mark():
    mark = SVGMobject(str(B.mark())).set(height=4.2)
    return mark, mark[0], mark[1]


def solid(w, h, color=INK):
    return Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=1)


class Ident(ToolkitScene):
    def land(self, parts, mark, hold=0.7):
        """The hit: a small swell, then the untouched SVG takes over, so the last frame is the flat mark."""
        self.ev("land")
        self.play(VGroup(*parts).animate(rate_func=there_and_back, run_time=0.22).scale(1.06))
        self.remove(*parts)
        self.add(mark)
        self.wait(hold)


class Flat(Scene):
    """A still of the mark as the idents must end: `sound.py --check` compares each last frame against it."""
    def construct(self):
        self.add(the_mark()[0])


class Typed(Ident):
    """Type a command, get a video: `/video` is typed at a caret, its slash becomes the mark's slash and the
    word folds into the play triangle."""
    def construct(self):
        mark, slash, play = the_mark()
        text = Text("/video", font=MONO, color=INK, weight=BOLD).set(width=8.2)
        text[0].set_color(ACCENT)
        caret = solid(0.16, text.height, ACCENT).next_to(text, LEFT, buff=0.2).align_to(text, DOWN)
        self.play(FadeIn(caret), run_time=0.15)
        self.play(caret.animate(rate_func=there_and_back, run_time=0.3).set_opacity(0.15))
        for letter in text:                       # typed by hand: one letter, one click, 100 ms apart
            self.ev("key")
            self.add(letter)
            caret.next_to(letter, RIGHT, buff=0.12).align_to(text, DOWN)
            self.wait(0.1)
        self.wait(0.25)
        copies = [play.copy() for _ in text[1:]]  # many shapes into one: each to its own copy, swapped at the end
        self.ev("fold:0.55")
        self.play(ReplacementTransform(text[0], slash), FadeOut(caret, run_time=0.1),
                  LaggedStart(*[ReplacementTransform(a, b) for a, b in zip(text[1:], copies)], lag_ratio=0.08),
                  run_time=0.55, rate_func=rush_into)
        self.land([slash, *copies], mark)


class Chain(Ident):
    """Script, voice, picture, video: five lines of script become a waveform, the waveform a frame, the frame
    play, with one word under each. Five parts all the way through, so every morph is one to one."""
    def construct(self):
        mark, slash, play = the_mark()
        centre = play.get_center() + RIGHT * 0.1
        bar = solid(slash.width * 0.36, slash.height, ACCENT).move_to(slash)
        lines = VGroup(*[solid(w, 0.16) for w in (2.3, 1.7, 2.1, 1.2, 1.9)]).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        wave = VGroup(*[solid(0.2, h) for h in (0.9, 2.1, 1.3, 2.5, 1.1)]).arrange(RIGHT, buff=0.26)
        frame = VGroup(*[solid(0.57, 1.6) for _ in range(5)]).arrange(RIGHT, buff=-0.01)   # overlapped: no seams
        for group in (lines, wave, frame):
            group.scale(1.25).move_to(centre)
        copies = [play.copy() for _ in range(5)]
        word = lambda s: Text(s, font=MONO, color=MUTED).set(height=0.42).move_to(DOWN * 3.0)
        label = word("script")
        self.play(FadeIn(bar, shift=UP * 0.2), run_time=0.2)
        for i in range(5):
            self.ev("key", after=0.02 + i * 0.075)
        self.play(LaggedStart(*[GrowFromEdge(l, LEFT) for l in lines], lag_ratio=0.15), FadeIn(label), run_time=0.5)
        self.wait(0.2)
        for target, name in ((wave, "voice"), (frame, "picture")):
            nxt = word(name)
            self.ev("morph:0.4")
            self.play(*[Transform(a, b) for a, b in zip(lines, target)], FadeOut(label, run_time=0.12),
                      FadeIn(nxt, run_time=0.25), run_time=0.4)
            label = nxt
            self.wait(0.22)
        nxt = word("video")
        self.ev("fold:0.35")
        self.play(*[Transform(a, b) for a, b in zip(lines, copies)], Transform(bar, slash),
                  FadeOut(label, run_time=0.12), FadeIn(nxt, run_time=0.2), run_time=0.35, rate_func=rush_into)
        self.play(FadeOut(nxt), run_time=0.2)
        self.land([bar, *lines], mark, hold=0.6)
