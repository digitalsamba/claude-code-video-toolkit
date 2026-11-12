"""Sets and effects for shot specs (runs inside Blender).

DeskRoom: a small room, desk, chair and a laptop. The screen is one of two things:

  * a live storage-style UI (the default): a title, a usage bar, a "free" line, an alert popup, a "working" line.
    The verbs (alert, work, done) animate it and the screen light. Text and numbers come from the spec, so the
    same set shows any "thing is full / thing is fixed" story.
  * a picture or a video (`screen.video` / `screen.image`): a Remotion render playing on the laptop, 16:9, unlit
    so its colours survive. With it, a `screen_fill` or `pull_out` shot starts on the render edge to edge.

`walls=False` (set kind `desk`) leaves out the room, so the desk and whoever sits at it can be rendered over a
transparent background and laid into a Remotion scene.
"""
from __future__ import annotations

import random

import bpy
from mathutils import Vector

from kit import (R, area_light, box, color, cyl, emission_socket, empty, glow, key, link, mat, media, n, quad, show,
                 sphere, text, ticks, tween)

D = bpy.data
TR = (R(90), 0, R(180))  # text facing +Y (the screen's viewer)

SCREEN_DEFAULTS = {
    "title": "Storage", "subtitle": "Main disk  -  460 GB",
    "before": "20 GB free", "after": "92 GB free",
    "fill_before": 0.957, "fill_after": 0.80,
    "alert": ["Your disk is almost full", "Free up space now!"],
    "working": "Cleaning up...",
    "bar_bad": "#FF3B2F", "bar_mid": "#FFC21A", "bar_good": "#4CE08A",
}


SCREEN_W = 0.456  # the picture's width (m); its height follows the aspect


class DeskRoom:
    def __init__(self, palette: dict | None = None, screen: dict | None = None, walls: bool = True,
                 media_path=None, media_start: int = 1, media_offset: int = 0, media_strength: float = 1.0):
        p = {"wall": "#A9B3BD", "wall2": "#909DAA", "floor": "#8A6B55", "desk": "#9C7552", "accent1": "#2E73B8",
             "accent2": "#F2C230", "rug": "#B25570", **(palette or {})}
        self.cfg = {**SCREEN_DEFAULTS, **(screen or {})}
        self.events: list[dict] = []
        self.walls = walls
        self.media = (media_path, media_start, media_offset, media_strength) if media_path else None
        aw, ah = (float(x) for x in str(self.cfg.get("aspect", "16:9" if media_path else "38:25")).split(":"))
        self.sw, self.sh = SCREEN_W, SCREEN_W * ah / aw
        self.m = {
            "wood": mat("wood", p["desk"], 0.45), "metal": mat("metal", (0.2, 0.2, 0.22), 0.35, 0.8),
            "chair": mat("chair", (0.12, 0.12, 0.14), 0.6), "silver": mat("silver", (0.76, 0.77, 0.8), 0.28, 0.9),
            "keys": mat("keys", (0.06, 0.06, 0.07), 0.5), "floor": mat("floor", p["floor"], 0.45),
            "wall": mat("wall", p["wall"], 0.9), "wall2": mat("wall2", p["wall2"], 0.9),
            "window": mat("window", (0.6, 0.78, 1.0), 0.5, emit=1.3), "rug": mat("rug", p["rug"], 0.95),
            "pot": mat("pot", (0.82, 0.42, 0.3), 0.6), "leaf": mat("leaf", (0.18, 0.5, 0.22), 0.5),
            "mug": mat("mug", (0.92, 0.92, 0.86), 0.3), "coffee": mat("coffee", (0.15, 0.07, 0.03), 0.1),
            "poster1": mat("poster1", p["accent1"], 0.7), "poster2": mat("poster2", p["accent2"], 0.7),
            "sticker": mat("sticker", p["accent2"], 0.5),
        }
        self._room()
        self._laptop()
        self._lights()

    def _room(self):
        M = self.m
        if self.walls:
            box("floor", (10, 10, 0.02), M["floor"], loc=(0, 0, -0.01))
            box("wall_back", (10, 0.1, 4), M["wall"], loc=(0, 3, 2))
            box("wall_front", (10, 0.1, 4), M["wall2"], loc=(0, -4.6, 2))
            box("wall_left", (0.1, 10, 4), M["wall2"], loc=(-4.2, 0, 2))
            box("wall_right", (0.1, 10, 4), M["wall"], loc=(4.2, 0, 2))
            box("window", (1.8, 0.02, 1.2), M["window"], loc=(0.6, -4.54, 1.9))
            for size, at in (((1.9, 0.05, 0.05), (0.6, -4.52, 1.9)), ((0.05, 0.05, 1.3), (0.6, -4.52, 1.9)),
                             ((1.9, 0.05, 0.05), (0.6, -4.52, 2.52)), ((1.9, 0.05, 0.05), (0.6, -4.52, 1.28)),
                             ((0.05, 0.05, 1.3), (-0.32, -4.52, 1.9)), ((0.05, 0.05, 1.3), (1.52, -4.52, 1.9))):
                box("window_frame", size, M["wall"], loc=at)  # reads as a window, not a white card
            box("poster1", (0.8, 0.02, 1.0), M["poster1"], loc=(-1.3, 2.94, 1.9), bevel=0.01)
            box("poster2", (0.55, 0.02, 0.7), M["poster2"], loc=(1.25, 2.94, 2.0), bevel=0.01)
            cyl("rug", 1.4, 0.01, M["rug"], loc=(0, -0.3, 0.005), seg=48)
        box("desk_top", (1.7, 0.8, 0.05), M["wood"], loc=(0, -0.75, 0.875), bevel=0.01)
        for x in (-0.8, 0.8):
            for y in (-0.4, -1.1):
                cyl("desk_leg", 0.03, 0.85, M["metal"], loc=(x, y, 0.425))
        cyl("pot", 0.07, 0.12, M["pot"], loc=(0.66, -0.88, 0.96))
        for i, (dx, dy, dz, r) in enumerate([(0, 0, .1, .08), (.05, .03, .16, .06), (-.05, -.02, .15, .065), (0, .05, .21, .05)]):
            sphere(f"leaf{i}", r, M["leaf"], loc=(0.66 + dx, -0.88 + dy, 0.96 + dz), seg=16)
        cyl("mug", 0.045, 0.1, M["mug"], loc=(-0.5, -0.52, 0.95))
        cyl("coffee", 0.04, 0.002, M["coffee"], loc=(-0.5, -0.52, 0.997))
        box("seat", (0.52, 0.5, 0.07), M["chair"], loc=(0, 0.05, 0.5), bevel=0.02)
        box("chair_back", (0.5, 0.06, 0.6), M["chair"], loc=(0, 0.34, 0.86), rot=(R(-8), 0, 0), bevel=0.02)
        cyl("chair_post", 0.03, 0.45, M["metal"], loc=(0, 0.05, 0.24))
        box("chair_base", (0.5, 0.05, 0.03), M["metal"], loc=(0, 0.05, 0.03))
        box("chair_base2", (0.05, 0.5, 0.03), M["metal"], loc=(0, 0.05, 0.03))

    def _laptop(self):
        M = self.m
        box("lap_base", (0.5, 0.34, 0.022), M["silver"], loc=(0, -0.6, 0.911), bevel=0.005)
        box("lap_keys", (0.44, 0.15, 0.004), M["keys"], loc=(0, -0.665, 0.923))
        box("lap_pad", (0.14, 0.08, 0.002), M["keys"], loc=(0, -0.49, 0.923))
        self.hinge = h = empty("hinge", None, (0, -0.77, 0.922), (R(15), 0, 0))
        hw, z0, z1 = self.sw / 2, 0.015, 0.015 + self.sh
        self.lid = box("lid", (0.5, 0.014, self.sh + 0.03), M["silver"], h, (0, 0, (self.sh + 0.03) / 2), bevel=0.004)
        cyl("sticker", 0.04, 0.002, M["sticker"], h, (0.09, -0.0075, 0.2 * self.sh / 0.3), (R(90), 0, 0))
        if self.media:
            path, start, offset, strength = self.media
            self.screen_bg = quad("scr_media", -hw, hw, z0, z1, media("scr_media", path, strength, start, offset),
                                  h, (0, 0.008, 0))
            self.bar = self.free_a = self.free_b = self.working = self.popup = None
        else:
            self._screen_ui(h, hw, z0, z1)
        sl = D.lights.new("screen_light", "AREA")
        sl.size, sl.energy, sl.color = 0.45, 18, (0.75, 0.85, 1.0)
        self.screen_light = sl
        o = link(D.objects.new("screen_light", sl))
        o.parent, o.location, o.rotation_euler = self.hinge, (0, 0.12, 0.17), (R(90), 0, 0)

    def _screen_ui(self, h, hw, z0, z1):
        c = self.cfg
        g = self.g = {
            "bg": glow("scr_bg", (0.05, 0.06, 0.08), 1.0), "txt": glow("scr_txt", (0.95, 0.95, 1.0), 2.2),
            "dim": glow("scr_dim", (0.55, 0.58, 0.65), 1.2), "bar_bg": glow("bar_bg", (0.22, 0.23, 0.26), 1.0),
            "bar": glow("bar_fill", color(c["bar_bad"]), 3.0), "pop_bg": glow("pop_bg", (0.93, 0.93, 0.95), 1.7),
            "pop_red": glow("pop_red", (0.9, 0.04, 0.02), 4.0), "pop_txt": glow("pop_txt", (0.1, 0.1, 0.12), 0.6),
            "work": glow("work", (0.3, 0.85, 1.0), 2.5), "good": glow("good", color(c["bar_good"]), 2.5),
        }
        self.screen_bg = quad("scr_bg", -hw, hw, z0, z1, g["bg"], h, (0, 0.008, 0))
        text("scr_title", c["title"], 0.034, g["txt"], h, (0.2, 0.0095, 0.27), TR, "LEFT")
        text("scr_sub", c["subtitle"], 0.018, g["dim"], h, (0.2, 0.0095, 0.235), TR, "LEFT")
        quad("bar_bg", -0.2, 0.2, 0.165, 0.2, g["bar_bg"], h, (0, 0.009, 0))
        self.bar = quad("bar_fill", -0.4, 0.0, 0.165, 0.2, g["bar"], h, (0.2, 0.0098, 0))
        self.bar.scale = (c["fill_before"], 1, 1)
        self.free_a = text("free_a", c["before"], 0.026, g["txt"], h, (0.2, 0.0095, 0.135), TR, "LEFT")
        self.free_b = text("free_b", c["after"], 0.026, g["good"], h, (0.2, 0.0095, 0.135), TR, "LEFT")
        self.free_b.scale = (0, 0, 0)
        self.working = text("working", c["working"], 0.022, g["work"], h, (0.2, 0.0095, 0.085), TR, "LEFT")
        self.working.scale = (0, 0, 0)
        self.popup = empty("popup", h, (0, 0.0115, 0.2))
        self.popup.scale = (0, 0, 0)
        quad("pop_bg", -0.17, 0.17, -0.055, 0.055, g["pop_bg"], self.popup)
        quad("pop_stripe", -0.17, 0.17, 0.042, 0.055, g["pop_red"], self.popup, (0, 0.0004, 0))
        text("pop_t1", c["alert"][0], 0.021, g["pop_red"], self.popup, (0, 0.0008, 0.012), TR)
        text("pop_t2", c["alert"][1], 0.016, g["pop_txt"], self.popup, (0, 0.0008, -0.022), TR)

    def _lights(self):
        aim = empty("aim", None, (0, -0.2, 1.3))
        area_light("key", (-2.2, -2.4, 3.2), 160, 2.5, (1, 0.93, 0.85), aim)
        area_light("fill", (2.6, -1.8, 2.0), 50, 3, (0.8, 0.88, 1.0), aim)
        area_light("rim", (0.8, 2.4, 3.0), 180, 1.5, (1, 0.9, 0.8), aim)

    # ------------------------------------------------------------ anchors / checks
    def anchors(self) -> dict:
        return {"screen": tuple(self.screen_center()), "desk": (0, -0.75, 0.9)}

    def screen_center(self) -> Vector:
        return self.hinge.matrix_world @ Vector((0, 0.008, 0.015 + self.sh / 2))

    def screen_normal(self) -> Vector:
        return (self.hinge.matrix_world.to_3x3() @ Vector((0, 1, 0))).normalized()

    def screen_points(self) -> list:
        """Centre and four points a little in from the corners (on screen and unblocked = the screen reads)."""
        mw = self.hinge.matrix_world
        x, lo, hi = self.sw / 2 - 0.048, 0.015 + 0.035, 0.015 + self.sh - 0.025
        return [mw @ Vector(p) for p in ((0, 0.012, 0.015 + self.sh / 2), (x, 0.012, lo), (-x, 0.012, lo),
                                         (x, 0.012, hi), (-x, 0.012, hi))]

    def visible_objects(self) -> set:
        out, stack = set(), [self.hinge]
        while stack:
            o = stack.pop()
            out.add(o)
            stack.extend(o.children)
        return out

    # ------------------------------------------------------------ verbs
    def alert(self, f: int):
        if self.popup:
            show(self.popup, f + n(2), True, (1.15,) * 3, dur=4)
            key(self.popup, "scale", f + n(5), (1, 1, 1))
        tween(self.screen_light, "color", f - n(2), f + n(1), (1.0, 0.45, 0.4))
        tween(self.screen_light, "energy", f - n(2), f + n(1), 22)
        self.events.append({"frame": f, "kind": "alert"})

    def work(self, f0: int, f1: int):
        """Popup away, 'working' line on, the bar drains bad -> mid -> good."""
        c = self.cfg
        if self.bar:
            show(self.popup, f0 - n(2), False, dur=5)
            show(self.working, f0 - n(2), True)
            mid = (f0 + f1) // 2
            tween(self.bar, "scale", f0, f1, (c["fill_after"], 1, 1))
            sock = emission_socket(self.g["bar"])
            tween(sock, "default_value", f0, mid, (*color(c["bar_mid"]), 1))
            tween(sock, "default_value", mid, f1, (*color(c["bar_good"]), 1))
        tween(self.screen_light, "color", f0, f1, (0.6, 1.0, 0.7))
        steps = ticks(f0, f1, 3)
        for i, f in enumerate(steps):
            self.events.append({"frame": f, "kind": "progress", "step": i, "of": len(steps)})

    def done(self, f: int):
        if self.free_a:
            show(self.free_a, f, False)
            show(self.working, f, False)
            show(self.free_b, f + n(2), True, (1.2,) * 3)
            key(self.free_b, "scale", f + n(6), (1, 1, 1))
        self.events.append({"frame": f, "kind": "success"})


def confetti(f: int, count: int = 60, seed: int = 7, colors=None):
    rng = random.Random(seed)
    mats = [mat(f"conf{i}", c, 0.4, emit=0.8) for i, c in enumerate(
        colors or [(1, .3, .3), (.3, .8, 1), (1, .85, .2), (.4, 1, .5), (.9, .4, 1)])]
    for i in range(count):
        c = box(f"confetti{i}", (0.035, 0.004, 0.05), mats[i % len(mats)])
        x, y, f0 = rng.uniform(-1.3, 1.3), rng.uniform(-1.4, 0.6), f + n(rng.randint(0, 10)) - n(4)
        land = 0.905 if (-1.15 < y < -0.35 and abs(x) < 0.85) else 0.01
        key(c, "scale", f0 - 1, (0, 0, 0))
        key(c, "scale", f0, (1, 1, 1))
        key(c, "location", f0, (x, y, 3.3))
        key(c, "location", f0 + n(rng.randint(45, 70)), (x + rng.uniform(-.3, .3), y, land))
        key(c, "rotation_euler", f0, (0, 0, 0))
        key(c, "rotation_euler", f0 + n(60), tuple(rng.uniform(-8, 8) for _ in range(3)))


def title3d(f: int, body: str, end: int, loc=(0, -0.25, 2.27), c="#FFD24A"):
    m = mat("title", color(c), 0.25, 0.6, emit=1.5)
    t = text("title3d", body, 0.28, m, None, loc, (R(90), 0, 0), extrude=0.03)
    t.scale = (0, 0, 0)
    key(t, "scale", f - n(4))
    key(t, "scale", f + n(4), (1.2,) * 3)
    key(t, "scale", f + n(10), (1, 1, 1))
    for i, fr in enumerate(ticks(f + n(10), end + 1, 20)):
        key(t, "rotation_euler", fr, (R(90), R((0, 3, 0, -3)[i % 4]), 0))
    return t
