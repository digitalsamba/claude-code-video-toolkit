"""A cosy pub in the afternoon for someone at a small table with a pint (runs inside Blender).

Pub: the kitchen's table and chair places (so a fitted character's furniture fit, the reach to the table and
`pick_up`, `sip` and `put_down` work as they do there), dressed as a snug: a dark wooden table on an iron
pedestal, a wooden chair, timber panelling to shoulder height under warm cream walls, a bar along one side with
taps and shelves of bottles, a fire in the wall behind, framed pictures, brass pendant lamps and a window with
the afternoon in it. Two other customers sit at a table at the back: plain shapes, meant to be out of focus
(give the shot a `dof`). There is no laptop.

The cup here is a pint of stout: a tulip glass with no handle, dark to the top with a cream head, standing on a
beer mat. A hand takes it round its middle (`pick_up` with `"lift": false` lays the hand round it and leaves
it standing; `let_go` takes the hand off it).

`"pint": {"at": [x, y], "side": "left"}` places it (by default in reach of the left hand). The palette
recolours the room: `wall`, `panel`, `floor`, `desk` (the table), `bar`, `accent1` (the seats), `lamp`, `window`.
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Vector

from kit import R, area_light, box, color, cyl, empty, glow, link, mat, mesh_obj, quad, sphere
from kitchen import Kitchen

D = bpy.data

BACK_Y = 2.9     # the face of the wall behind the sitter
LEFT_X = -2.3    # the bar's wall (x is the sitter's left... the room's left as the sitter looks at the table)
RIGHT_X = 2.6
FRONT_Y = -3.6


def lathe(name, profile, m, parent=None, loc=(0, 0, 0), seg=40):
    """A turned shape: `profile` is [(radius, z), ...] from the bottom up, spun round Z, closed top and bottom."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), z))
                      for i in range(seg)])
    for a, b in zip(rings, rings[1:]):
        for i in range(seg):
            bm.faces.new((a[i], a[(i + 1) % seg], b[(i + 1) % seg], b[i]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    return mesh_obj(name, bm, m, parent, loc, smooth=True)


class Pub(Kitchen):
    MUG_H, MUG_R = 0.152, 0.043   # a pint glass: tall, and its radius at the bulge

    def __init__(self, palette: dict | None = None, fit: dict | None = None, pint: dict | None = None,
                 hand: float = 0.085):
        self.pp = {"wall": "#E9DCC0", "panel": "#6E4428", "floor": "#7B5236", "desk": "#5A3620", "bar": "#4A2A18",
                   "accent1": "#2F5D4A", "lamp": "#FFCF8F", "window": "#FFF0D0", **(palette or {})}
        # (a cartoon hand is big: the glass is sized to the hand that takes it, never under a real pint's size)
        self.k = max(1.0, hand / 0.085) * float((pint or {}).get("scale", 1.0))
        self.MUG_H, self.MUG_R = Pub.MUG_H * self.k, Pub.MUG_R * self.k
        cfg = dict(pint or {})
        cfg.setdefault("on_table", True)
        self.pint_cfg = cfg
        super().__init__({"wall": self.pp["wall"], "wall2": self.pp["wall"], "floor": self.pp["floor"],
                          "desk": self.pp["desk"], "accent1": self.pp["accent1"], "lamp": self.pp["lamp"]},
                         None, fit=fit, mug=cfg)

    # ------------------------------------------------------------ the room
    def _room(self):
        M, P = self.m, self.pp
        M.update({
            "panel": mat("panel", P["panel"], 0.5), "panel2": mat("panel_rail", tuple(0.8 * c for c in color(P["panel"])), 0.45),
            "bar": mat("bar", P["bar"], 0.35), "brass": mat("brass", (0.72, 0.55, 0.26), 0.3, 0.9),
            "iron": mat("iron", (0.05, 0.05, 0.055), 0.5, 0.6), "seat": mat("seat_pad", P["accent1"], 0.9),
            "frame": mat("frame", (0.12, 0.08, 0.05), 0.45), "paper": mat("paper", (0.93, 0.9, 0.82), 0.8),
            "glassy": mat("bottle_glass", (0.1, 0.25, 0.12), 0.12), "amber": mat("bottle_amber", (0.45, 0.22, 0.05), 0.12),
            "clear": mat("bottle_clear", (0.8, 0.85, 0.85), 0.08), "label": mat("label", (0.9, 0.86, 0.7), 0.7),
            "daylight": mat("pub_daylight", P["window"], 0.4, emit=5.0), "trim": mat("pub_trim", (0.2, 0.12, 0.07), 0.45),
            "fire": glow("fire", (1.0, 0.45, 0.1), 7.0), "ember": glow("ember", (1.0, 0.2, 0.03), 3.0),
            "stone": mat("stone", (0.42, 0.4, 0.37), 0.9), "shade": mat("pub_shade", (0.62, 0.45, 0.2), 0.35, 0.8),
            "bulb": glow("pub_bulb", color(P["lamp"]), 3.0), "mat_": mat("beer_mat", (0.93, 0.9, 0.82), 0.9),
            "mat2": mat("beer_mat_ring", (0.1, 0.1, 0.1), 0.8),
        })
        box("floor", (12, 12, 0.02), M["floor"], loc=(0, 0, -0.01))
        for i in range(-9, 10):  # (floorboards: a dark line every 30 cm)
            box("board_gap", (0.008, 11, 0.004), M["trim"], loc=(i * 0.3 + 0.07, 0, 0.0005))
        for name, size, at in (("wall_back", (10, 0.1, 4), (0, BACK_Y + 0.05, 2)), ("wall_front", (10, 0.1, 4), (0, FRONT_Y - 0.05, 2)),
                               ("wall_left", (0.1, 10, 4), (LEFT_X - 0.05, 0, 2)), ("wall_right", (0.1, 10, 4), (RIGHT_X + 0.05, 0, 2))):
            box(name, size, M["wall"], loc=at)
        box("ceiling", (12, 12, 0.02), mat("pub_ceiling", (0.88, 0.82, 0.68), 0.9), loc=(0, 0, 2.7))
        for y in (-2.2, -0.6, 1.0, 2.4):  # beams
            box("beam", (10, 0.16, 0.14), M["panel2"], loc=(0, y, 2.62))
        self._panelling(M)
        self._back_wall(M)
        self._bar(M)
        self._window(M)
        self._others(M)
        self._table_and_chair(M)

    def _panelling(self, M):
        """Timber to shoulder height on the back and right walls: boards, a rail on top, a skirting."""
        h = 1.25
        for name, size, at in (("panel_back", (RIGHT_X - LEFT_X, 0.03, h), ((LEFT_X + RIGHT_X) / 2, BACK_Y - 0.015, h / 2)),
                               ("panel_right", (0.03, BACK_Y - FRONT_Y, h), (RIGHT_X - 0.015, (BACK_Y + FRONT_Y) / 2, h / 2))):
            box(name, size, M["panel"], loc=at)
        box("rail_back", (RIGHT_X - LEFT_X, 0.06, 0.05), M["panel2"], loc=((LEFT_X + RIGHT_X) / 2, BACK_Y - 0.04, h + 0.025))
        box("rail_right", (0.06, BACK_Y - FRONT_Y - 0.1, 0.05), M["panel2"], loc=(RIGHT_X - 0.04, (BACK_Y + FRONT_Y) / 2 - 0.05, h + 0.025))
        x = LEFT_X + 0.25
        while x < RIGHT_X - 0.1:
            box("batten", (0.03, 0.012, h - 0.12), M["panel2"], loc=(x, BACK_Y - 0.036, h / 2))
            x += 0.42
        y = FRONT_Y + 0.3
        while y < BACK_Y - 0.2:
            box("batten_r", (0.012, 0.03, h - 0.12), M["panel2"], loc=(RIGHT_X - 0.036, y, h / 2))
            y += 0.42

    def _back_wall(self, M):
        """Behind the sitter: a fireplace with a fire in it, a mantel, a mirror over it, pictures either side."""
        fx = -0.55
        box("chimney", (1.3, 0.16, 2.7), M["wall"], loc=(fx, BACK_Y - 0.08, 1.35))
        box("hearth_stone", (1.1, 0.1, 1.05), M["stone"], loc=(fx, BACK_Y - 0.19, 0.525), bevel=0.01)
        box("hearth_hole", (0.62, 0.06, 0.62), mat("soot", (0.02, 0.018, 0.016), 0.9), loc=(fx, BACK_Y - 0.225, 0.33))
        for i, (dx, w, hgt) in enumerate(((-0.14, 0.12, 0.26), (0.0, 0.16, 0.38), (0.13, 0.11, 0.22), (0.05, 0.08, 0.3))):
            sphere(f"flame{i}", 0.5, M["fire"], loc=(fx + dx, BACK_Y - 0.26, 0.1 + hgt / 2), scale=(w, 0.06, hgt), seg=14)
        box("embers", (0.5, 0.05, 0.06), M["ember"], loc=(fx, BACK_Y - 0.25, 0.05))
        box("mantel", (1.34, 0.2, 0.07), M["panel2"], loc=(fx, BACK_Y - 0.22, 1.1), bevel=0.012)
        fire = D.lights.new("fire_light", "POINT")
        fire.energy, fire.color, fire.shadow_soft_size = 60, (1.0, 0.55, 0.22), 0.25
        fire.use_shadow = False  # (a light inside the hearth, for the same reason as the pendants)
        link(D.objects.new("fire_light", fire)).location = (fx, BACK_Y - 0.6, 0.45)
        cyl("mirror_frame", 0.36, 0.03, M["brass"], loc=(fx, BACK_Y - 0.175, 1.72), rot=(R(90), 0, 0), seg=48, smooth=True)
        # (not a real mirror: a metal disc with nothing to reflect flickered 48 levels from frame to frame, and one
        # that glowed set the shelves and the bar beside it flickering)
        cyl("mirror", 0.32, 0.034, mat("mirror", (0.8, 0.85, 0.88), 0.3), loc=(fx, BACK_Y - 0.176, 1.72), rot=(R(90), 0, 0), seg=48)
        for i, (x, w, hgt, c) in enumerate(((0.75, 0.46, 0.6, "#3E6B66"), (1.35, 0.36, 0.46, "#B5562F"), (1.9, 0.46, 0.6, "#D9B45A"),
                                            (-1.65, 0.4, 0.52, "#7E9AA6"))):
            z = 1.85 - (0.07 if i == 1 else 0)
            box("picture", (w, 0.03, hgt), M["frame"], loc=(x, BACK_Y - 0.015, z), bevel=0.006)
            box("mount", (w - 0.06, 0.006, hgt - 0.06), M["paper"], loc=(x, BACK_Y - 0.032, z))
            box("art", (w - 0.18, 0.004, hgt - 0.2), mat(f"pub_art{i}", c, 0.8), loc=(x, BACK_Y - 0.037, z))

    def _bar(self, M):
        """Along the left wall: a counter with a brass foot rail and taps, shelves of bottles on the wall behind."""
        bx, y0, y1 = LEFT_X + 0.95, -2.4, 2.2
        L = y1 - y0
        box("bar_front", (0.08, L, 1.02), M["bar"], loc=(bx, (y0 + y1) / 2, 0.51))
        box("bar_top", (0.62, L + 0.06, 0.06), M["panel2"], loc=(bx - 0.22, (y0 + y1) / 2, 1.08), bevel=0.012)
        for z in (0.32, 0.72):
            box("bar_rail", (0.02, L - 0.1, 0.05), M["panel2"], loc=(bx + 0.045, (y0 + y1) / 2, z))
        cyl("foot_rail", 0.02, L - 0.2, M["brass"], loc=(bx + 0.14, (y0 + y1) / 2, 0.2), rot=(R(90), 0, 0), smooth=True)
        for i, y in enumerate((-0.7, -0.4, -0.1)):  # three taps
            cyl("tap_post", 0.018, 0.3, M["brass"], loc=(bx - 0.18, y, 1.26), smooth=True)
            box("tap_badge", (0.03, 0.07, 0.1), mat(f"badge{i}", ("#1B1B1B", "#B5562F", "#2F5D4A")[i], 0.4), loc=(bx - 0.16, y, 1.42), bevel=0.01)
        rng = random.Random(4)
        wx = LEFT_X + 0.09
        for z in (1.35, 1.75, 2.15):
            box("shelf", (0.2, L - 0.5, 0.03), M["panel2"], loc=(wx + 0.02, (y0 + y1) / 2, z))
            y = y0 + 0.4
            while y < y1 - 0.35:
                kind = rng.choice(("glassy", "amber", "clear", "amber"))
                hgt = rng.uniform(0.22, 0.3)
                cyl("bottle", 0.036, hgt, M[kind], loc=(wx + 0.03, y, z + 0.015 + hgt / 2), seg=14, smooth=True)
                cyl("bottle_neck", 0.014, 0.08, M[kind], loc=(wx + 0.03, y, z + 0.015 + hgt + 0.04), seg=10, smooth=True)
                box("bottle_label", (0.004, 0.05, 0.07), M["label"], loc=(wx + 0.066, y, z + 0.015 + hgt * 0.45))
                y += rng.uniform(0.11, 0.17)
        box("bar_mirror", (0.02, L - 0.6, 0.3), mat("bar_mirror", (0.66, 0.62, 0.54), 0.35), loc=(LEFT_X + 0.012, (y0 + y1) / 2, 1.56))
        for y in (-1.5, 0.3, 1.6):  # stools
            cyl("stool_seat", 0.17, 0.05, M["seat"], loc=(bx + 0.45, y, 0.72), seg=28, smooth=True)
            cyl("stool_post", 0.025, 0.7, M["iron"], loc=(bx + 0.45, y, 0.35))
            cyl("stool_foot", 0.16, 0.02, M["iron"], loc=(bx + 0.45, y, 0.01), seg=28)

    def _window(self, M):
        """The afternoon in a window on the right wall: small panes, a deep sill, and its light across the room."""
        wy, z0, z1, ww = 0.9, 1.4, 2.3, 1.5
        turn = empty("pub_window_at", None, (RIGHT_X - 0.036, wy, 0), (0, 0, R(-90)))
        quad("pub_window", -ww / 2, ww / 2, z0, z1, M["daylight"], turn)
        for u in (-ww / 2, -ww / 4, 0, ww / 4, ww / 2):
            box("glazing", (0.03, 0.04, z1 - z0 + 0.06), M["trim"], loc=(RIGHT_X - 0.05, wy + u, (z0 + z1) / 2))
        for z in (z0 - 0.02, (z0 + z1) / 2, z1 + 0.02):
            box("glazing_h", (0.03, ww + 0.08, 0.04), M["trim"], loc=(RIGHT_X - 0.05, wy, z))
        box("pub_sill", (0.16, ww + 0.2, 0.04), M["panel2"], loc=(RIGHT_X - 0.1, wy, z0 - 0.06))
        self.window_at = Vector((RIGHT_X - 0.3, wy, (z0 + z1) / 2))

    def _others(self, M):
        """Two customers at a table by the back wall, to the right of the fire: plain shapes in soft colours, for
        the background of a shot whose focus is on the sitter."""
        tx, ty = 1.35, 1.95
        cyl("far_table", 0.36, 0.04, M["wood"], loc=(tx, ty, 0.74), seg=36)
        cyl("far_table_post", 0.035, 0.72, M["iron"], loc=(tx, ty, 0.36))
        cyl("far_table_foot", 0.22, 0.02, M["iron"], loc=(tx, ty, 0.01), seg=28)
        for i, (dx, dy, top, skin, hair, turn) in enumerate(((-0.52, 0.12, "#7A8FA6", "#D9A98A", "#3B2A20", 80),
                                                             (0.5, 0.2, "#A65F4A", "#E0B79A", "#B9B2A6", -75))):
            g = empty(f"customer{i}", None, (tx + dx, ty + dy, 0), (0, 0, R(turn)))
            body = mat(f"customer{i}_top", top, 0.9)
            box("customer_legs", (0.34, 0.42, 0.16), mat(f"customer{i}_legs", "#2B2F3A", 0.9), g, (0, -0.16, 0.5), bevel=0.05)
            sphere("customer_torso", 0.5, body, g, (0, 0, 0.84), scale=(0.42, 0.28, 0.62), seg=18)
            sphere("customer_head", 0.115, mat(f"customer{i}_skin", skin, 0.6), g, (0, -0.02, 1.3), seg=18)
            sphere("customer_hair", 0.12, mat(f"customer{i}_hair", hair, 0.7), g, (0, 0.02, 1.325), scale=(1, 1, 0.9), seg=18)
            cyl("customer_stool", 0.16, 0.44, M["panel2"], g, (0, 0, 0.22), seg=20)
            cyl("customer_glass", 0.034, 0.13, mat(f"customer{i}_glass", ("#1A0F0A", "#C98A2B")[i], 0.1),
                loc=(tx + dx * 0.35, ty + dy * 0.3, 0.825), seg=16, smooth=True)

    def _table_and_chair(self, M):
        """A small pub table where the desk stands (its near edge where the desk's is), on an iron pedestal; a
        wooden chair with a padded seat; brass pendants over this table and down the room."""
        Y, Z, C = self.desk_y, self.desk_z, self.chair_z
        self.desk_top = box("desk_top", (1.15, 0.72, 0.045), M["wood"], loc=(0, -0.71 + Y, 0.8775 + Z), bevel=0.014)
        cyl("pub_table_post", 0.045, 0.83 + Z, M["iron"], loc=(0, -0.71 + Y, (0.83 + Z) / 2))
        box("pub_table_foot", (0.62, 0.09, 0.035), M["iron"], loc=(0, -0.71 + Y, 0.0175), bevel=0.008)
        box("pub_table_foot2", (0.09, 0.5, 0.035), M["iron"], loc=(0, -0.71 + Y, 0.0176), bevel=0.008)
        ch = self.chair
        seat_z = 0.5 + C
        self.seat = box("seat", (0.46, 0.44, 0.045), M["seat"], ch, (0, 0.05, seat_z), bevel=0.016)
        self.chair_back = box("chair_back", (0.44, 0.04, 0.2), M["wood"], ch, (0, 0.27, seat_z + 0.44), (R(-8), 0, 0), bevel=0.01)
        box("chair_rail", (0.4, 0.025, 0.04), M["wood"], ch, (0, 0.265, seat_z + 0.2), (R(-8), 0, 0), bevel=0.006)
        for x in (-0.2, 0.2):
            cyl("chair_leg", 0.018, seat_z, M["wood"], ch, (x, -0.14, seat_z / 2), seg=12, smooth=True)
            cyl("chair_leg_back", 0.02, seat_z + 0.56, M["wood"], ch, (x, 0.25, (seat_z + 0.56) / 2), (R(-4), 0, 0), seg=12, smooth=True)
        self.lamp_z = 2.0
        for i, (x, y) in enumerate(((0.0, -0.75 + Y), (1.5, 1.6), (-0.9, 1.3), (1.4, -1.8))):
            cyl("pendant_cord", 0.004, 2.7 - self.lamp_z, M["iron"], loc=(x, y, (2.7 + self.lamp_z) / 2))
            lathe(f"pendant_shade{i}", [(0.03, 0.09), (0.05, 0.06), (0.16, -0.04), (0.165, -0.06)], M["shade"], None, (x, y, self.lamp_z))
            sphere(f"pendant_bulb{i}", 0.04, M["bulb"], loc=(x, y, self.lamp_z - 0.045), seg=14)
            if i:
                lp = D.lights.new(f"pendant{i}", "POINT")
                lp.energy, lp.color, lp.shadow_soft_size = 45, color(self.pp["lamp"]), 0.2
                # (no shadows from a lamp that hangs inside its own shade: the shade's soft shadow on the ceiling
                # and the beams was drawn from a different handful of rays each frame, and the ceiling shimmered)
                lp.use_shadow = False
                link(D.objects.new(f"pendant{i}", lp)).location = (x, y, self.lamp_z - 0.12)

    def _laptop(self):
        """No laptop on a pub table: its place is kept (some verbs ask where the screen is) and nothing is drawn."""
        self.hinge = empty("hinge", None, (0, -0.8, 0.9))
        self.lid = self.lap_base = self.screen_bg = None
        self.bar = self.free_a = self.free_b = self.working = self.popup = None
        sl = D.lights.new("screen_light", "AREA")
        sl.energy = 0.0
        self.screen_light = sl
        self.sh = 0.3

    def _lights(self):
        """A bright pub afternoon: the pendant over the table as a warm key from above and in front, the window's
        daylight from the side, a broad warm fill off the room, the fire's glow behind."""
        P, Y = self.pp, self.desk_y
        lamp = D.lights.new("pendant", "SPOT")
        lamp.energy, lamp.color, lamp.spot_size, lamp.spot_blend = 300, color(P["lamp"]), R(125), 0.7
        lamp.shadow_soft_size = 0.16
        o = link(D.objects.new("pendant", lamp))
        o.location, o.rotation_euler = (0, -0.75 + Y, self.lamp_z - 0.12), (0, 0, 0)
        aim = empty("aim", None, (0, -0.2, 1.2))
        area_light("window_light", tuple(self.window_at), 230, 1.6, (1.0, 0.95, 0.85), aim)
        area_light("fill", (-1.2, -2.6, 2.0), 150, 3.2, (1.0, 0.88, 0.72), aim)
        area_light("rim", (0.9, 1.6, 2.3), 60, 1.2, (1, 0.85, 0.65), aim)

    def solids(self) -> list:
        return [(self.seat, 0.03), (self.chair_back, 0.02), (self.desk_top, 0.0)]

    def visible_objects(self) -> set:
        return {self.hinge}

    # ------------------------------------------------------------ the pint
    def _mug(self, name):
        """A pint of stout in a tulip glass: its middle at the object's origin, dark nearly to the top, a cream
        head, a rim of glass. No handle: a hand goes round it (`rim`, `tall` and `wide` say how big it is)."""
        h, r = self.MUG_H, self.MUG_R
        stout = mat(name + "_stout", (0.012, 0.008, 0.007), 0.06)
        head = mat(name + "_head", (0.93, 0.86, 0.7), 0.75)
        glass = mat(name + "_glass", (0.8, 0.84, 0.84), 0.04)
        top = empty(name)
        root = empty(name + "_pour", top)
        z0 = -h / 2
        k = self.k
        lathe(name + "_body", [(0.029 * k, z0), (0.031 * k, z0 + 0.012 * k), (0.036 * k, z0 + 0.06 * k), (r, z0 + 0.105 * k),
                               (r - 0.001, z0 + 0.128 * k)], stout, root)
        lathe(name + "_head", [(r - 0.001, z0 + 0.128 * k), (r - 0.002, z0 + 0.146 * k)], head, root)
        lathe(name + "_rim", [(r - 0.0005, z0 + 0.146 * k), (r - 0.001, h / 2)], glass, root)
        top["rim"], top["tall"], top["wide"] = r, h, r
        top["hold_drop"] = 0.0  # (a hand round its middle)
        return top

    def _mug_props(self):
        Y, Z = self.desk_y, self.desk_z
        self.cup_hand, self.cup_table = self._mug("cup_hand"), self._mug("cup_table")
        self.cup_hand.scale = (0, 0, 0)
        s = 1 if self.pint_cfg.get("side", "left") == "left" else -1
        at = self.pint_cfg.get("at", (0.3 * s, -0.58 + Y))
        table = 0.9 + Z
        self.cup_spot = Vector((at[0], at[1], table + self.MUG_H / 2 + 0.003))
        self.cup_table.location = self.cup_spot
        self.cup_table.scale = (1, 1, 1) if self.pint_cfg.get("on_table", True) else (0, 0, 0)
        box("beer_mat", (0.095 * self.k, 0.095 * self.k, 0.003), self.m["mat_"], loc=(at[0], at[1], table + 0.0015), rot=(0, 0, R(12)), bevel=0.001)

    def prop(self, name):
        if name in ("cup", "pint"):
            return self.cup_hand, self.cup_table, self.cup_spot
        raise ValueError(f'a pub has a "pint", not {name!r}')
