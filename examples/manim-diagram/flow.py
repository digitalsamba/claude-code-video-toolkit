"""A diagram that builds on the narration's words: a request path, the cache that went in between, and the
number it moved. 19 s. A sprint review's backend change, which a screen recording can't show.

    uv run tools/manim_render.py render examples/manim-diagram/flow.py Flow --brand video-toolkit \\
        --captions examples/manim-diagram/audio/flow.captions.json --alpha --out examples/manim-diagram/out/flow.webm

Every step waits for its word with self.at(V.cue(...)). Rendered with --alpha it is the diagram alone, for a
Remotion scene that draws the ground, the title and the captions."""
from toolkit_manim import *

B = Brand()
INK, MUTED, ACCENT, NAVY, FILL, FAINT, WARN = (B.c("textDark"), B.c("textLight"), B.c("primary"), B.c("navy", "#323e66"),
                                              B.c("bgAlt", "#f1f2f4"), B.c("divider", "#c9cfe0"), "#f06859")
FONT = "Helvetica Neue"
config.background_color = B.c("bgLight")
V = Voice()


def box(label, w=2.9):
    rect = RoundedRectangle(corner_radius=0.12, width=w, height=1.45, stroke_color=NAVY, stroke_width=3, fill_color=FILL, fill_opacity=1)
    return VGroup(rect, Text(label, font=FONT, color=INK, weight=MEDIUM).scale(0.62).move_to(rect))


def arrow(a, b, color=NAVY):
    return Arrow(a.get_right(), b.get_left(), buff=0.1, color=color, stroke_width=4)


class Flow(ToolkitScene):
    def request(self, start, end, target, color=ACCENT, run_time=0.5):
        """A dot travels from one box to another and the box it reaches answers."""
        dot = Dot(start.get_right(), color=color, radius=0.12)
        self.play(MoveAlongPath(dot, Line(start.get_right(), end.get_left())), run_time=run_time, rate_func=linear)
        self.play(Indicate(target[0], color=color, scale_factor=1.04), FadeOut(dot), run_time=0.2)

    def construct(self):
        browser, api, db = box("Browser"), box("API"), box("Database")
        VGroup(browser, api, db).arrange(RIGHT, buff=1.7).shift(UP * 0.9)
        a1, a2 = arrow(browser, api), arrow(api, db)

        self.at(V.cue("browser", lead=0.25))
        self.play(FadeIn(browser, shift=UP * 0.2), run_time=0.4)
        self.at(V.cue("API", lead=0.25))
        self.play(GrowArrow(a1), FadeIn(api, shift=UP * 0.2), run_time=0.5)
        self.at(V.cue("database", lead=0.3))
        self.play(GrowArrow(a2), FadeIn(db, shift=UP * 0.2), run_time=0.5)

        self.at(V.cue("every page load"))                # every page load pays: three go the whole way
        for _ in range(3):
            self.request(browser, db, db)

        cache = box("Cache", w=2.4)                      # the cache goes in between
        cache[0].set_stroke(ACCENT, 4)
        b2, p2, c2, d2 = VGroup(browser.copy(), api.copy(), cache, db.copy()).arrange(RIGHT, buff=1.0).shift(UP * 0.9)
        to_cache = arrow(p2, c2, ACCENT)
        self.at(V.cue("cache", lead=0.35))
        self.play(Transform(browser, b2), Transform(api, p2), Transform(db, d2), Transform(a1, arrow(b2, p2)),
                  Transform(a2, arrow(c2, d2, FAINT)), FadeIn(cache, scale=0.8), GrowArrow(to_cache), run_time=0.8)

        self.at(V.cue("cache", nth=2, lead=0.5))         # asks the cache first: hits stop there
        for _ in range(2):
            self.request(browser, cache, cache, run_time=0.4)
        self.at(V.cue("miss", lead=0.1))                 # only a miss reaches the database
        self.request(browser, db, db, color=WARN, run_time=0.7)

        ms = ValueTracker(400)                           # the number
        spot = DOWN * 1.9
        number = always_redraw(lambda: Text(f"{int(ms.get_value())} ms", font=FONT, color=ACCENT, weight=BOLD).scale(1.5).move_to(spot))
        label = Text("median response", font=FONT, color=MUTED).scale(0.5).next_to(spot, UP, buff=0.7)
        self.at(V.cue("four hundred", lead=0.3))
        self.play(FadeIn(number), FadeIn(label), run_time=0.3)
        self.at(V.cue("ninety", lead=0.5))
        self.play(ms.animate.set_value(90), run_time=0.7, rate_func=rush_from)
        self.at(V.end + 0.6)
