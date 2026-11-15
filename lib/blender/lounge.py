"""A living room on a bright afternoon for someone on a sofa (runs inside Blender).

LivingRoom: a sofa where the desk room's chair stands (its seat at the character's own knee height, as the chair's
is), with no desk and no laptop: whoever sits there has their hands in their lap. Round it: a wall close behind
with pictures, a floor lamp and a side table with a mug, a rug and a low table in front, a window at dusk in the
side wall, a plant. It is lit as a room is on a bright afternoon: sun through the side window as the key, a soft warm fill from the
front, the lamp behind as a warm edge.

`"sofa": {"x": 0.45, "width": 1.9, "color": "#..."}` puts the sofa's middle `x` to the sitter's left (so they sit
toward one end) and sizes it. The palette recolours the room: `wall`, `wall2`, `floor`, `sofa`, `cushion`, `rug`,
`wood`, `lamp` (the lamp's light), `window` (the daylight in the window).
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

from kit import R, area_light, box, color, cyl, empty, glow, link, mat, quad, sphere
from sets import DeskRoom

D = bpy.data

BACK_Y = 1.0     # the face of the wall behind the sofa
LEFT_X = -2.4    # the side walls (x is the sitter's left)
RIGHT_X = 3.0
FRONT_Y = -4.6


class LivingRoom(DeskRoom):
    rest_pose = "lap"  # no desk: hands rest on the thighs (compile lays them there on the first frame)

    def __init__(self, palette: dict | None = None, fit: dict | None = None, sofa: dict | None = None):
        self.lp = {"wall": "#C7B9A6", "wall2": "#B7A893", "floor": "#7A5B45", "sofa": "#4F6F73", "cushion": "#C9773F",
                   "rug": "#A9967C", "wood": "#7A5236", "lamp": "#FFC486", "window": "#FFF1D6", **(palette or {})}
        self.sofa_cfg = sofa or {}
        super().__init__({"wall": self.lp["wall"], "wall2": self.lp["wall2"], "floor": self.lp["floor"],
                          "desk": self.lp["wood"]}, None, walls=True, fit=fit)

    # ------------------------------------------------------------ the room
    def _room(self):
        M, P = self.m, self.lp
        M.update({
            "sofa": mat("sofa", self.sofa_cfg.get("color", P["sofa"]), 0.95),
            "sofa2": mat("sofa_base", tuple(0.8 * c for c in color(self.sofa_cfg.get("color", P["sofa"]))), 0.95),
            "cushion": mat("cushion", P["cushion"], 0.95), "cushion2": mat("cushion2", "#E3D7C2", 0.95),
            "rug2": mat("rug2", P["rug"], 1.0), "frame": mat("frame", (0.1, 0.08, 0.06), 0.5),
            "paper": mat("paper", (0.9, 0.87, 0.8), 0.8), "art1": mat("art1", "#B5562F", 0.8),
            "art2": mat("art2", "#3E6B66", 0.8), "art3": mat("art3", "#D9B45A", 0.8),
            "shade": mat("lamp_shade", (0.93, 0.85, 0.7), 0.7, emit=1.6, emit_color=color(P["lamp"])),
            "brass": mat("brass", (0.62, 0.48, 0.25), 0.35, 0.9), "dusk": mat("daylight", P["window"], 0.4, emit=4.0),
            "trim": mat("trim", (0.92, 0.9, 0.86), 0.5), "book": mat("book", "#8C3B3B", 0.7),
            "book2": mat("book2", "#2F4A6B", 0.7), "curtain": mat("curtain", "#8A7B6A", 0.95),
        })
        box("floor", (12, 12, 0.02), M["floor"], loc=(0, 0, -0.01))
        box("wall_back", (10, 0.1, 4), M["wall"], loc=(0, BACK_Y + 0.05, 2))
        box("wall_front", (10, 0.1, 4), M["wall2"], loc=(0, FRONT_Y, 2))
        box("wall_left", (0.1, 10, 4), M["wall2"], loc=(LEFT_X - 0.05, 0, 2))
        box("wall_right", (0.1, 10, 4), M["wall2"], loc=(RIGHT_X + 0.05, 0, 2))
        box("ceiling", (12, 12, 0.02), M["wall2"], loc=(0, 0, 2.75))
        box("skirting", (10, 0.02, 0.1), M["trim"], loc=(0, BACK_Y - 0.01, 0.05))
        self.desk_top = None
        self._sofa(M)
        self._dressing(M)

    def _sofa(self, M):
        """A two-seat sofa built round the chair's place: the sitter's cushion is the seat a body rests on, the
        cushion behind them its back. Its parts hang from the chair's empty, like a chair's."""
        cfg, ch, C = self.sofa_cfg, self.chair, self.chair_z
        W, sx = float(cfg.get("width", 1.9)), float(cfg.get("x", 0.45))
        top = 0.505 + C                  # the top of the seat cushions (a little under the desk chair's: a cushion gives)
        front, back = -0.3, 0.62         # the seat from its front edge to the back of the frame
        arm = 0.2
        box("sofa_base", (W, back - front - 0.04, top - 0.2), M["sofa2"], ch, (sx, (front + back) / 2 + 0.02, (top - 0.14 + 0.06) / 2),
            bevel=0.02)
        half = (W - 2 * arm) / 2
        seats = []
        for i in (0, 1):  # (two cushions, a gap between: the sitter is on the one at their end)
            x = sx - half / 2 + i * half
            seats.append(box("sofa_seat", (half - 0.012, 0.64, 0.15), M["sofa"], ch, (x, front + 0.32, top - 0.075), bevel=0.045))
        self.seat = min(seats, key=lambda o: abs(o.location.x))
        box("sofa_backrest", (W - 2 * arm + 0.03, 0.2, 0.62), M["sofa2"], ch, (sx, back - 0.115, top - 0.1 + 0.31), bevel=0.03)
        backs = []
        for i in (0, 1):
            x = sx - half / 2 + i * half
            backs.append(box("sofa_back", (half - 0.02, 0.17, 0.5), M["sofa"], ch, (x, 0.415, top + 0.27), (R(-10), 0, 0),
                             bevel=0.06))
        self.chair_back = min(backs, key=lambda o: abs(o.location.x))
        for s in (-1, 1):
            box("sofa_arm", (arm, back - front, 0.3), M["sofa"], ch, (sx + s * (W - arm) / 2, (front + back) / 2, top + 0.02),
                bevel=0.06)
            for y in (front + 0.08, back - 0.08):
                cyl("sofa_foot", 0.025, 0.07, M["wood"], ch, (sx + s * (W / 2 - 0.1), y, 0.035))
        # a scatter cushion at the far end and a folded throw over the arm beside it
        far = 1 if sx >= 0 else -1
        box("scatter", (0.42, 0.14, 0.4), M["cushion"], ch, (sx + far * (W / 2 - arm - 0.26), 0.3, top + 0.2),
            (R(-16), R(8 * far), R(-14 * far)), bevel=0.06)
        box("throw", (0.24, 0.6, 0.03), M["cushion2"], ch, (sx + far * (W - arm) / 2, 0.12, top + 0.185), bevel=0.012)
        self.sofa_x, self.sofa_w, self.seat_top = sx, W, top

    def _dressing(self, M):
        sx, W = self.sofa_x, self.sofa_w
        near = sx - W / 2  # the sofa's end nearer the sitter's right
        # pictures over the sofa: three frames with a mount and a block of colour
        for i, (x, w, h, art) in enumerate(((sx - 0.55, 0.5, 0.65, "art1"), (sx + 0.05, 0.42, 0.42, "art2"),
                                            (sx + 0.6, 0.5, 0.65, "art3"))):
            z = 1.72 + (0.08 if i == 1 else 0)
            box("picture", (w, 0.03, h), M["frame"], loc=(x, BACK_Y - 0.015, z), bevel=0.006)
            box("mount", (w - 0.06, 0.006, h - 0.06), M["paper"], loc=(x, BACK_Y - 0.032, z))
            box("art", (w - 0.2, 0.004, h - 0.24), M[art], loc=(x, BACK_Y - 0.037, z))
        # a floor lamp behind the near arm, and what it lights
        lx, ly = near - 0.38, 0.62
        cyl("lamp_foot", 0.14, 0.025, M["brass"], loc=(lx, ly, 0.0125), seg=32)
        cyl("lamp_pole", 0.012, 1.45, M["brass"], loc=(lx, ly, 0.75))
        cyl("lamp_shade", 0.2, 0.26, M["shade"], loc=(lx, ly, 1.58), seg=40, smooth=True)
        lamp = D.lights.new("lamp", "POINT")
        lamp.energy, lamp.color, lamp.shadow_soft_size = 90, color(self.lp["lamp"]), 0.18
        link(D.objects.new("lamp", lamp)).location = (lx, ly - 0.02, 1.5)
        # a side table by that arm: a mug, two books
        tx, ty = near - 0.34, 0.05
        cyl("side_top", 0.24, 0.03, M["wood"], loc=(tx, ty, 0.56), seg=40)
        cyl("side_leg", 0.02, 0.55, M["wood"], loc=(tx, ty, 0.275))
        cyl("side_foot", 0.15, 0.02, M["wood"], loc=(tx, ty, 0.01), seg=32)
        cyl("tea", 0.042, 0.095, mat("tea_mug", "#E8E1D2", 0.3), loc=(tx - 0.05, ty - 0.07, 0.6225), seg=32, smooth=True)
        box("book_a", (0.2, 0.14, 0.025), M["book"], loc=(tx + 0.06, ty + 0.06, 0.5875), rot=(0, 0, R(14)))
        box("book_b", (0.18, 0.13, 0.02), M["book2"], loc=(tx + 0.065, ty + 0.055, 0.61), rot=(0, 0, R(-6)))
        # a rug and a low table in front
        box("rug", (2.4, 1.7, 0.012), M["rug2"], loc=(sx, -1.0, 0.006), bevel=0.004)
        box("low_table", (1.05, 0.5, 0.04), M["wood"], loc=(sx, -1.25, 0.38), bevel=0.01)
        for x in (-0.46, 0.46):
            for y in (-0.19, 0.19):
                cyl("low_leg", 0.02, 0.36, M["wood"], loc=(sx + x, -1.25 + y, 0.18))
        box("magazine", (0.22, 0.3, 0.008), M["paper"], loc=(sx + 0.2, -1.22, 0.404), rot=(0, 0, R(18)))
        # a sunny window in the side wall past the far end, with a curtain each side, and a plant under it
        wy, wz0, wz1, ww = -0.6, 0.95, 2.2, 1.3
        side = RIGHT_X if sx >= 0 else LEFT_X
        inn = -1 if sx >= 0 else 1
        turn = empty("dusk_at", None, (side + inn * 0.004, wy, 0), (0, 0, R(90 * inn)))
        quad("dusk", -ww / 2, ww / 2, wz0, wz1, M["dusk"], turn)
        for size, at in (((0.05, ww + 0.1, 0.05), (side + inn * 0.03, wy, wz1 + 0.025)),
                         ((0.05, ww + 0.1, 0.05), (side + inn * 0.03, wy, wz0 - 0.025)),
                         ((0.03, 0.03, wz1 - wz0), (side + inn * 0.02, wy, (wz0 + wz1) / 2))):
            box("window_frame", size, M["trim"], loc=at)
        for dy in (-1, 1):
            box("curtain", (0.06, 0.3, 1.6), M["curtain"], loc=(side + inn * 0.07, wy + dy * (ww / 2 + 0.16), 1.5), bevel=0.02)
        px = side + inn * 0.4
        cyl("plant_pot", 0.16, 0.3, M["pot"], loc=(px, 0.55, 0.15), seg=32)
        for i, (dx, dy, dz, r) in enumerate(((0, 0, 0.5, 0.2), (0.1, 0.05, 0.75, 0.17), (-0.1, -0.04, 0.72, 0.16),
                                             (0.02, 0.08, 0.98, 0.14), (-0.03, -0.06, 1.15, 0.1))):
            sphere(f"plant{i}", r, M["leaf"], loc=(px + dx, 0.55 + dy, dz), scale=(1, 1, 1.25), seg=14)
        self.window_side, self.window_y = side, wy

    def _laptop(self):
        """No desk and no laptop: the screen's place is kept (some cameras and verbs ask for it) and nothing is drawn."""
        self.hinge = empty("hinge", None, (0, -0.8, 0.9))
        self.lid = self.lap_base = self.screen_bg = None
        self.bar = self.free_a = self.free_b = self.working = self.popup = None
        sl = D.lights.new("screen_light", "AREA")
        sl.energy = 0.0
        self.screen_light = sl

    def _lights(self):
        """A bright afternoon: sun through the side window as the key, a broad warm fill from in front on the
        lamp's side, a soft light off the ceiling, and a warm edge from behind (the floor lamp's own light is
        made with the lamp)."""
        aim = empty("aim", None, (0, -0.1, 1.15))
        k = -1 if self.sofa_x >= 0 else 1
        area_light("key", (self.window_side * 0.8, self.window_y - 0.9, 2.0), 260, 2.4, (1.0, 0.93, 0.8), aim)
        area_light("fill", (k * 1.6, -2.4, 1.8), 120, 3.0, (1.0, 0.9, 0.78), aim)
        area_light("top", (0.3, -0.8, 2.6), 60, 3.0, (1.0, 0.96, 0.9), empty("top_aim", None, (0.3, -0.6, 0)))
        area_light("rim", (k * 1.1, 0.75, 2.2), 45, 1.0, (1, 0.85, 0.65), aim)

    # ------------------------------------------------------------ anchors / checks
    def anchors(self) -> dict:
        return {"screen": tuple(self.hinge.location), "desk": (0, -0.8, 0.5)}

    def solids(self) -> list:
        return [(self.seat, 0.06), (self.chair_back, 0.06)]  # (sofa cushions give more than a chair's seat)

    def screen_center(self) -> Vector:
        return Vector(self.hinge.location)

    def visible_objects(self) -> set:
        return {self.hinge}
