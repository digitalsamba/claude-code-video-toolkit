"""A kitchen at night for someone at a laptop (runs inside Blender).

Kitchen: the desk room's table, chair, laptop and door, dressed as a home kitchen: a run of units along the left
wall (cupboards, worktop, sink, hob, kettle, a window over the sink), a fridge in the corner, a shelf on the back
wall, a pendant lamp over the table, and night outside. The table is a kitchen table and the chair a wooden one,
in the same places as the desk room's (so a fitted character's furniture fit, `pull_in` and the laptop work as
they do there).

What it adds to the desk room:

  * a mug of tea that can be carried in and put down (`carry`, `put_down` with `"prop": "cup"`), the pair of
    objects the lecture hall uses for its cup;
  * the laptop's lid starts shut with `"lid": "closed"`, and the set verbs `open_lid` / `close_lid` swing it,
    with `"by"` putting a hand on its top edge for the move (the desk room has the same verbs);
  * `"claude"`: a beige CRT with a face on its screen (a picture or a video), a tinfoil hat, and two arms at a
    keyboard, standing on the worktop. `claude_type` sets its hands going on the keys (a click per key), and
    `claude_turn` turns it on the spot.

The palette recolours it: `wall`, `wall2`, `floor`, `desk` (the table), `units`, `worktop`, `accent1`, `night`
(the window), `lamp` (the pendant's light).
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Vector

from kit import (R, arc_tube, area_light, box, color, cyl, empty, glow, key, link, mat, media, mesh_obj, n, quad,
                 sphere, ticks, tween)
from sets import DeskRoom

D = bpy.data

LEFT_X = -2.6     # the left wall's middle (the units stand against it)
RIGHT_X = 3.4
FRONT_Y = -3.4
UNIT_D = 0.6      # depth of the units, front to back
WORKTOP_Z = 0.92  # top of the worktop


def cone(name, r0, r1, depth, m, parent=None, loc=(0, 0, 0), seg=28, rot=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r0, radius2=r1, depth=depth)
    return mesh_obj(name, bm, m, parent, loc, rot, smooth=True)


def crumpled_cone(name, r0, depth, m, parent, loc, seed=5):
    """A tinfoil hat: a cone whose surface is pushed in and out a little, so it catches the light in facets."""
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=18, radius1=r0, radius2=0.004, depth=depth)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
    for v in bm.verts:
        r = math.hypot(v.co.x, v.co.y)
        if r > 0.01:
            k = 1 + rng.uniform(-0.09, 0.09)
            v.co.x, v.co.y = v.co.x * k, v.co.y * k
            v.co.z += rng.uniform(-0.006, 0.006)
    return mesh_obj(name, bm, m, parent, loc, smooth=False)


class Kitchen(DeskRoom):
    MUG_H, MUG_R, MUG_LOOP = 0.118, 0.058, 0.043   # tall, radius, and the handle's loop out from its wall

    def __init__(self, palette: dict | None = None, screen: dict | None = None, media_path=None,
                 media_start: int = 1, media_offset: int = 0, media_strength: float = 1.0, fit: dict | None = None,
                 door: dict | None = None, chair_out: float = 0.0, lid: str = "open", mug: dict | None = None,
                 claude: dict | None = None, claude_media=None, claude_start: int = 1, claude_strength: float = 1.0):
        self.kp = {"wall": "#C9BBA3", "wall2": "#B4A68E", "floor": "#6B5443", "desk": "#9A7550",
                   "units": "#4E6B66", "worktop": "#D9D2C3", "accent1": "#B5562F", "night": "#0E1A2C",
                   "lamp": "#FFC98A", **(palette or {})}
        self.mug_cfg = mug or {}
        self.claude_cfg, self.claude = claude, None
        self.claude_media = (claude_media, claude_start, claude_strength)
        super().__init__({k: v for k, v in self.kp.items() if k in ("wall", "wall2", "floor", "desk", "accent1")},
                         screen, walls=True, media_path=media_path, media_start=media_start,
                         media_offset=media_offset, media_strength=media_strength, fit=fit, door=door,
                         chair_out=chair_out)
        self.lid_open = self.hinge.rotation_euler.copy()
        self.lid_shut = (R(-90), 0, 0)  # flat on the keys (the lid's screen side faces down into the base)
        if lid == "closed":
            self.hinge.rotation_euler = self.lid_shut
            self.screen_light.energy = 0.0
        self.lid_state = lid
        self._mug_props()
        if claude is not None:
            self._claude(claude)

    # ------------------------------------------------------------ the room
    def _room(self):
        M, P = self.m, self.kp
        M.update({
            "units": mat("units", P["units"], 0.55), "worktop": mat("worktop", P["worktop"], 0.35),
            "steel": mat("steel", (0.7, 0.71, 0.72), 0.25, 0.9), "hob": mat("hob", (0.03, 0.03, 0.035), 0.15),
            "ring": mat("hob_ring", (0.22, 0.22, 0.24), 0.4), "fridge": mat("fridge", (0.9, 0.9, 0.88), 0.3),
            "night": mat("night", P["night"], 0.4, emit=0.6), "tile": mat("tile", (0.93, 0.92, 0.88), 0.25),
            "jar": mat("jar", (0.8, 0.86, 0.85), 0.1), "kettle": mat("kettle", P["accent1"], 0.35),
            "shade": mat("shade", (0.12, 0.13, 0.12), 0.45), "bulb": glow("bulb", color(P["lamp"]), 6.0),
            "frame": mat("frame", (0.16, 0.12, 0.09), 0.5), "fruit": mat("fruit", (0.95, 0.62, 0.12), 0.5),
            "fruit2": mat("fruit2", (0.55, 0.75, 0.25), 0.5), "bowl": mat("bowl", (0.85, 0.83, 0.78), 0.4),
            "towel": mat("towel", (0.75, 0.3, 0.25), 0.9), "board": mat("chopping_board", (0.72, 0.55, 0.36), 0.6),
            "leaf": M["leaf"], "pot": M["pot"],
        })
        box("floor", (12, 12, 0.02), M["floor"], loc=(0, 0, -0.01))
        if self.door_cfg:
            self._door(M)
        else:
            box("wall_back", (10, 0.1, 4), M["wall"], loc=(0, 3, 2))
        box("wall_front", (10, 0.1, 4), M["wall2"], loc=(0, FRONT_Y, 2))
        box("wall_right", (0.1, 10, 4), M["wall"], loc=(RIGHT_X, 0, 2))
        box("wall_left", (0.1, 10, 4), M["wall2"], loc=(LEFT_X, 0, 2))
        self._dressing(M)
        self._table_and_chair(M)

    def _run(self, name, M, at, turn, length, cupboards=()):
        """A run of base units against a wall: carcass, plinth, door fronts with bar handles, a worktop and tiles
        over it, and wall cupboards over the stretches given (pairs of u along the run). Built in the run's own
        frame (an empty at the middle of the run on the wall's face, turned `turn` degrees): u is along the run
        and -y comes out into the room. Returns the empty, for what stands on the worktop."""
        run = empty(name, None, at, (0, 0, R(turn)))
        d, L = UNIT_D, length
        box(name + "_carcass", (L, d - 0.04, 0.82), M["units"], run, (0, -(d - 0.04) / 2, 0.47))
        # (the plinth ends well inside the carcass: an end near the carcass's own read as faces in one plane)
        box(name + "_plinth", (L - 0.12, d - 0.1, 0.1), M["hob"], run, (0, -(d - 0.1) / 2 - 0.004, 0.05))
        box(name + "_worktop", (L + 0.02, d + 0.03, 0.04), M["worktop"], run, (0, -(d + 0.03) / 2, WORKTOP_Z - 0.02),
            bevel=0.004)
        doors = max(1, int(round(L / 0.6)))
        dw = L / doors
        for i in range(doors):  # fronts stand proud of the carcass with a gap between, each with a bar handle
            u = -L / 2 + dw * (i + 0.5)
            box(name + "_door", (dw - 0.006, 0.02, 0.72), M["units"], run, (u, -d + 0.03, 0.5), bevel=0.003)
            box(name + "_handle", (0.12, 0.02, 0.012), M["steel"], run, (u, -d + 0.005, 0.8), bevel=0.003)
        box(name + "_tiles", (L, 0.01, 0.5), M["tile"], run, (0, -0.006, WORKTOP_Z + 0.25))
        for a, b in cupboards:
            box(name + "_cupboard", (b - a, 0.34, 0.7), M["units"], run, ((a + b) / 2, -0.17, 1.85), bevel=0.004)
            k = max(1, int(round((b - a) / 0.5)))
            for i in range(k):
                u = a + (b - a) / k * (i + 0.5)
                box(name + "_cb_handle", (0.012, 0.02, 0.1), M["steel"], run, (u + 0.08, -0.35, 1.58), bevel=0.003)
        return run

    def _dressing(self, M):
        """The back wall's run behind the table (a window at night over the sink, the kettle, the hob), a fridge
        in the back left corner, a short run on the left wall (where Claude stands, when he is in), a clock and a
        picture."""
        WY = 2.95  # the back wall's face
        x0, x1 = -1.85, 1.95  # the back run, from the fridge to the door
        back = self._run("back", M, ((x0 + x1) / 2, WY - 0.005, 0), 0, x1 - x0, cupboards=((-1.9, -0.95), (0.55, 1.9)))
        W = WORKTOP_Z
        # the window over the sink: the night in a frame on the wall, a cross of glazing bars, a sill
        wu, ww, wz0, wz1 = -0.2, 1.3, 1.2, 2.15
        quad("night", -ww / 2, ww / 2, wz0, wz1, M["night"], empty("night_at", back, (wu, -0.014, 0), (0, 0, R(180))))
        f = M["frame"]
        for size, at in (((ww + 0.1, 0.05, 0.05), (wu, -0.03, wz1 + 0.025)), ((ww + 0.1, 0.05, 0.05), (wu, -0.03, wz0 - 0.025)),
                         ((0.05, 0.05, wz1 - wz0), (wu - ww / 2 - 0.025, -0.03, (wz0 + wz1) / 2)),
                         ((0.05, 0.05, wz1 - wz0), (wu + ww / 2 + 0.025, -0.03, (wz0 + wz1) / 2)),
                         ((0.03, 0.03, wz1 - wz0), (wu, -0.02, (wz0 + wz1) / 2)), ((ww, 0.03, 0.03), (wu, -0.02, wz0 + 0.62))):
            box("window_frame", size, f, back, at)
        box("sill", (ww + 0.16, 0.12, 0.03), M["worktop"], back, (wu, -0.06, wz0 - 0.06))
        # a little moonlight through it, and a streetlamp's glow low in the corner of the glass
        area_light("window_light", (wu + (x0 + x1) / 2, WY - 0.3, 1.7), 18, 1.0, (0.55, 0.68, 1.0),
                   empty("window_aim", None, (0.0, 0.5, 0.9)))
        sphere("streetlamp", 0.03, glow("streetlamp", (1.0, 0.72, 0.35), 3.0), back, (wu + 0.42, -0.001, wz0 + 0.3), seg=12)
        cyl("herb_pot", 0.05, 0.09, M["pot"], back, (wu + 0.45, -0.06, wz0 - 0.0))
        for i, (du, dz, r) in enumerate(((0, 0.07, 0.045), (0.03, 0.1, 0.035), (-0.03, 0.09, 0.035))):
            sphere(f"herb{i}", r, M["leaf"], back, (wu + 0.45 + du, -0.06, wz0 + dz), seg=12)
        # the sink under the window, its tap, the kettle and a board, the hob and a tea towel on the oven door
        box("sink", (0.6, 0.45, 0.012), M["steel"], back, (wu, -0.32, W + 0.001), bevel=0.01)
        box("sink_hole", (0.5, 0.36, 0.004), M["hob"], back, (wu, -0.32, W + 0.006))
        tap = empty("tap", back, (wu, -0.08, W))
        cyl("tap_post", 0.014, 0.28, M["steel"], tap, (0, 0, 0.14))
        arc_tube("tap_spout", 0.07, 0.012, 0, math.pi, M["steel"], empty("spout_at", tap, (0, 0, 0), (0, 0, R(90))),
                 (0.07, 0, 0.28), seg=16, ring=8)
        k = empty("kettle", back, (0.75, -0.3, W))
        cone("kettle_body", 0.1, 0.075, 0.2, M["kettle"], k, (0, 0, 0.1))
        cyl("kettle_lid", 0.05, 0.02, M["kettle"], k, (0, 0, 0.21), smooth=True)
        arc_tube("kettle_handle", 0.07, 0.012, 0, math.pi, M["shade"], k, (0, 0, 0.2), seg=14, ring=8)
        box("chopping_board", (0.42, 0.3, 0.02), M["board"], back, (-1.2, -0.3, W + 0.01), bevel=0.004)
        for i, (du, h, m) in enumerate(((-1.55, 0.2, "jar"), (-1.43, 0.15, "jar"), (-1.31, 0.17, "kettle"))):
            cyl(f"jar{i}", 0.05, h, M[m], back, (du, -0.1, W + h / 2), smooth=True)
        hu = 1.35
        box("hob", (0.56, 0.5, 0.008), M["hob"], back, (hu, -0.3, W + 0.004))
        for du, dy, r in ((-0.13, -0.12, 0.08), (0.13, -0.12, 0.06), (-0.13, 0.12, 0.06), (0.13, 0.12, 0.08)):
            cyl("hob_ring", r, 0.002, M["ring"], back, (hu + du, -0.3 + dy, W + 0.009), seg=32)
        box("oven_front", (0.56, 0.02, 0.5), M["hob"], back, (hu, -UNIT_D + 0.015, 0.45))
        box("tea_towel", (0.18, 0.02, 0.3), M["towel"], back, (hu + 0.12, -UNIT_D - 0.005, 0.62))
        # the fridge in the corner, magnets and notes on its door
        fx = LEFT_X + 0.4
        box("fridge", (0.7, 0.68, 1.85), M["fridge"], loc=(fx, WY - 0.36, 0.925), bevel=0.01)
        box("fridge_gap", (0.66, 0.005, 0.006), M["hob"], loc=(fx, WY - 0.702, 1.2))
        box("fridge_handle", (0.02, 0.025, 0.4), M["steel"], loc=(fx + 0.3, WY - 0.71, 1.45), bevel=0.004)
        for i, c in enumerate(("#3771E0", "#F06859", "#F2C230")):
            box(f"note{i}", (0.09, 0.006, 0.11), mat(f"note{i}", c, 0.7), loc=(fx - 0.18 + 0.12 * i, WY - 0.706, 1.5 - 0.07 * i))
        # the short run on the left wall, a fruit bowl at its far end
        y0, y1 = -1.3, WY - 0.72
        side = self._run("side", M, (LEFT_X + 0.055, (y0 + y1) / 2, 0), 90, y1 - y0)
        cyl("bowl", 0.12, 0.06, M["bowl"], side, ((y1 - y0) / 2 - 0.3, -0.3, W + 0.03), seg=32, smooth=True)
        for i, (du, dy, m) in enumerate(((0, 0, "fruit"), (0.06, 0.04, "fruit2"), (-0.05, 0.05, "fruit"))):
            sphere(f"fruit{i}", 0.04, M[m], side, ((y1 - y0) / 2 - 0.3 + du, -0.3 + dy, W + 0.09), seg=16)
        # a clock and a picture on the right wall
        clock = empty("clock", None, (RIGHT_X - 0.05, 0.9, 2.2), (R(90), 0, R(90)))
        cyl("clock_face", 0.16, 0.03, M["worktop"], clock, seg=40, smooth=True)
        cyl("clock_rim", 0.17, 0.02, M["frame"], clock, (0, 0, 0.004), seg=40, smooth=True)
        box("picture", (0.03, 0.7, 0.5), M["frame"], loc=(RIGHT_X - 0.07, -0.6, 1.75), bevel=0.006)
        box("picture_art", (0.005, 0.6, 0.4), mat("art", "#7E9AA6", 0.8), loc=(RIGHT_X - 0.09, -0.6, 1.75))

    def _table_and_chair(self, M):
        """The kitchen table where the desk stands (the same top, so the laptop and the reach are the desk
        room's), on four turned legs; a wooden chair where the office chair was. A pendant lamp hangs over it."""
        Y, Z, C = self.desk_y, self.desk_z, self.chair_z
        # (deeper than the desk, away from them: its near edge is where the desk's is, 0.35 ahead of the clay
        # cast, which is what a fitted character's belly room is measured from)
        self.desk_top = box("desk_top", (1.6, 0.9, 0.05), M["wood"], loc=(0, -0.8 + Y, 0.875 + Z), bevel=0.012)
        box("table_apron", (1.45, 0.75, 0.08), M["wood"], loc=(0, -0.8 + Y, 0.81 + Z))
        for x in (-0.72, 0.72):
            for y in (-0.41, -1.19):
                cyl("table_leg", 0.035, 0.85 + Z, M["wood"], loc=(x, y + Y, 0.425 + Z / 2), seg=16, smooth=True)
        ch = self.chair
        seat_z = 0.5 + C
        self.seat = box("seat", (0.46, 0.44, 0.045), M["wood"], ch, (0, 0.05, seat_z), bevel=0.012)
        self.chair_back = box("chair_back", (0.44, 0.04, 0.16), M["wood"], ch, (0, 0.27, seat_z + 0.42),
                              (R(-8), 0, 0), bevel=0.01)
        box("chair_rail", (0.4, 0.025, 0.04), M["wood"], ch, (0, 0.265, seat_z + 0.2), (R(-8), 0, 0), bevel=0.006)
        for x in (-0.2, 0.2):
            cyl("chair_leg", 0.018, seat_z, M["wood"], ch, (x, -0.14, seat_z / 2), seg=12, smooth=True)
            cyl("chair_leg_back", 0.02, seat_z + 0.5, M["wood"], ch, (x, 0.25, (seat_z + 0.5) / 2), (R(-4), 0, 0),
                seg=12, smooth=True)
        # the lamp over the table: a cord, a dark shade, a warm bulb, and the room's main light from it
        top = 3.2
        lz = 2.2
        cyl("lamp_cord", 0.004, top - lz, M["shade"], loc=(0, -0.75 + Y, (top + lz) / 2))
        cone("lamp_shade", 0.17, 0.045, 0.16, M["shade"], None, (0, -0.75 + Y, lz))
        sphere("lamp_bulb", 0.035, M["bulb"], loc=(0, -0.75 + Y, lz - 0.07), seg=16)
        self.lamp_z = lz
        box("ceiling", (12, 12, 0.02), M["wall2"], loc=(0, 0, top + 0.01))

    def _lights(self):
        """Night: the pendant over the table is the key, warm and from above; a dim warm fill from the room; the
        window's cool light from the left (made with the window); a small lamp under the wall cupboards."""
        P = self.kp
        Y = self.desk_y
        lamp = D.lights.new("pendant", "SPOT")
        lamp.energy, lamp.color, lamp.spot_size, lamp.spot_blend = 260, color(P["lamp"]), R(120), 0.6
        lamp.shadow_soft_size = 0.12
        o = link(D.objects.new("pendant", lamp))
        o.location, o.rotation_euler = (0, -0.75 + Y, self.lamp_z - 0.1), (0, 0, 0)
        aim = empty("aim", None, (0, -0.2, 1.1))
        area_light("fill", (2.4, -2.4, 2.4), 40, 3, (1.0, 0.86, 0.7), aim)
        area_light("rim", (1.2, 2.5, 2.7), 70, 1.5, (1, 0.85, 0.68), aim)
        area_light("under_cupboard", (LEFT_X + 0.3, -1.0, 1.45), 25, 1.6, (1, 0.82, 0.6),
                   empty("uc_aim", None, (LEFT_X + 0.3, -1.0, 0.5)))

    # ------------------------------------------------------------ a mug of tea
    def _mug(self, name):
        """A mug of tea: its middle at the object's origin, a handle big enough for a hand on its +X side, a rim,
        a band of the room's accent round it and tea near the brim. (The first one was a small cream cylinder
        with its handle hidden in the fist, and read as a candle: a mug is told by its handle, its colour and
        being wider than the hand that holds it.) `handle` on the object is how far the handle's bar is from
        the mug's middle: a hand takes it there (acting.carry)."""
        M, h, r = self.m, self.MUG_H, self.MUG_R
        c = mat(name + "_glaze", self.mug_cfg.get("color", self.kp["accent1"]), 0.28)
        inner = mat(name + "_inside", self.mug_cfg.get("inside", "#EFE8DA"), 0.3)
        top = empty(name)
        root = empty(name + "_pour", top)  # (what tips when it is drunk from: the mug in the hand stays level)
        cyl(name + "_body", r, h, c, root, (0, 0, 0), seg=40, smooth=True)
        arc_tube(name + "_rim", r - 0.004, 0.005, 0, 2 * math.pi, inner, root, (0, 0, h / 2), seg=40, ring=8).rotation_euler = (R(90), 0, 0)
        cyl(name + "_inside", r - 0.006, 0.004, inner, root, (0, 0, h / 2 - 0.001), seg=40)
        cyl(name + "_tea", r - 0.009, 0.003, mat(name + "_tea", "#6B3A1C", 0.12), root, (0, 0, h / 2 + 0.0012), seg=40)
        arc_tube(name + "_handle", self.MUG_LOOP, 0.0085, -math.pi / 2, math.pi / 2, c, root, (r - 0.004, 0, 0), seg=18, ring=10)
        root = top
        root["handle"], root["rim"], root["tall"] = r - 0.004 + self.MUG_LOOP, r, h
        root["hold_drop"] = 0.024  # (the fist a little above the handle's middle: clear of the table as it is set down)
        return root

    def _mug_props(self):
        Y, Z = self.desk_y, self.desk_z
        self.cup_hand, self.cup_table = self._mug("cup_hand"), self._mug("cup_table")
        self.cup_hand.scale = (0, 0, 0)
        at = self.mug_cfg.get("at", (0.5, -0.47 + Y))  # (out past a hand that lies beside the laptop)
        self.cup_spot = Vector((at[0], at[1], 0.9 + Z + self.MUG_H / 2 + 0.001))
        self.cup_table.location = self.cup_spot
        self.cup_table.rotation_euler = (0, 0, R(self.mug_cfg.get("turn", 200)))
        self.cup_table.scale = (1, 1, 1) if self.mug_cfg.get("on_table") else (0, 0, 0)

    def prop(self, name):
        if name == "cup":
            return self.cup_hand, self.cup_table, self.cup_spot
        raise ValueError(f'a kitchen has a "cup" (the mug), not {name!r}')

    # ------------------------------------------------------------ Claude, the CRT
    def _claude(self, cfg):
        """A beige CRT on the worktop, facing the room, with a face on its screen, a tinfoil hat and two arms
        out of its sides to a keyboard. Its origin is the middle of its foot on the worktop."""
        x, y = cfg.get("at", (LEFT_X + 0.05 + UNIT_D / 2 + 0.02, 0.2))
        root = empty("claude", None, (x, y, cfg.get("z", WORKTOP_Z)), (0, 0, R(cfg.get("facing", 90))))
        self.claude = root
        beige = mat("crt_beige", cfg.get("color", "#D9CFB4"), 0.55)
        dark = mat("crt_dark", (0.07, 0.07, 0.075), 0.5)
        foil = mat("foil", (0.82, 0.83, 0.86), 0.3, 1.0)
        W, H, Dp = 0.4, 0.34, 0.36
        z0 = 0.045
        box("crt_foot", (0.24, 0.22, 0.03), beige, root, (0, 0.04, 0.015), bevel=0.006)
        box("crt_neck", (0.12, 0.1, 0.03), beige, root, (0, 0.04, 0.04))
        body = empty("crt_head", root, (0, 0, z0))
        self.claude_head = body
        box("crt_body", (W, Dp * 0.62, H), beige, body, (0, 0.0, H / 2), bevel=0.02)
        box("crt_back", (W * 0.7, Dp * 0.45, H * 0.72), beige, body, (0, Dp * 0.48, H * 0.46), bevel=0.02)
        # the screen: a dark bezel inset, and the face (picture or video) a hair in front of it
        box("crt_bezel", (W * 0.82, 0.01, H * 0.76), dark, body, (0, -Dp * 0.31, H * 0.53), bevel=0.01)
        sw, sh = W * 0.74, H * 0.66
        path, start, strength = self.claude_media
        face = media("crt_face", path, strength, start) if path else glow("crt_face", (0.12, 0.85, 0.35), 2.0)
        turn = empty("crt_face_at", body, (0, -Dp * 0.31 - 0.0056, 0), (0, 0, R(180)))
        self.claude_screen = quad("crt_face", -sw / 2, sw / 2, H * 0.53 - sh / 2, H * 0.53 + sh / 2, face, turn)
        if not path:  # a face made of bars, for a check before the video exists
            g = glow("crt_eyes", (0.02, 0.12, 0.05), 1.0)
            for ex in (-0.05, 0.05):
                box("crt_eye", (0.022, 0.003, 0.04), g, body, (ex, -Dp * 0.31 - 0.008, H * 0.6))
            box("crt_mouth", (0.1, 0.003, 0.016), g, body, (0, -Dp * 0.31 - 0.008, H * 0.43))
        box("crt_button", (0.03, 0.01, 0.012), dark, body, (W * 0.33, -Dp * 0.31, 0.03))
        sl = D.lights.new("crt_light", "AREA")
        sl.size, sl.energy, sl.color = 0.3, 6, (0.4, 1.0, 0.55)
        o = link(D.objects.new("crt_light", sl))
        o.parent, o.location, o.rotation_euler = body, (0, -Dp * 0.31 - 0.08, H * 0.55), (R(90), 0, 0)
        if cfg.get("hat", True):
            crumpled_cone("foil_hat", W * 0.42, 0.26, foil, body, (0, 0.03, H + 0.12))
        # the keyboard in front, and two arms from its sides to it
        kb = empty("crt_keyboard", root, (0, -0.33, 0.0))
        box("crt_kb", (0.36, 0.13, 0.025), beige, kb, (0, 0, 0.0125), bevel=0.005)
        box("crt_keys", (0.32, 0.09, 0.006), dark, kb, (0, 0.005, 0.027))
        self.claude_hands = []
        for s in (1, -1):
            sh_ = empty("crt_shoulder", body, (s * (W / 2 + 0.005), -0.04, H * 0.4))
            hand = Vector((s * 0.08 - s * (W / 2 + 0.005), -0.3, 0.05 - (z0 + H * 0.4)))  # (in the shoulder's space)
            mid = hand / 2
            arm = cyl("crt_arm", 0.012, hand.length, dark, sh_, mid, seg=10, smooth=True)
            arm.rotation_mode = "QUATERNION"
            arm.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(hand.normalized())
            sphere("crt_mitt", 0.028, mat("crt_glove", "#F2F0EA", 0.6), sh_, hand, scale=(1, 1.15, 0.7), seg=14)
            sh_.rotation_mode = "XYZ"
            self.claude_hands.append(sh_)

    def claude_type(self, f0: int, f1: int, speed: str = "frantic"):
        """The CRT's hands go up and down on the keys, one after the other, a click on each landing. Frantic
        is the madman's speed (ten keys a second); calm is half that."""
        if not self.claude:
            raise ValueError('this kitchen has no Claude: give the set "claude": {}')
        step = 2.4 if speed == "frantic" else 4.8  # frames at 24 fps between keys
        rng = random.Random(f0)
        for i, f in enumerate(ticks(f0, f1, step)):
            sh_ = self.claude_hands[i % 2]
            up = n(step * 0.45)
            key(sh_, "rotation_euler", f - up, (R(-7 - rng.uniform(0, 4)), 0, R(rng.uniform(-3, 3))))
            key(sh_, "rotation_euler", f, (R(1.5), 0, 0))
            self.events.append({"frame": f, "kind": "key_click"})
        for sh_ in self.claude_hands:
            key(sh_, "rotation_euler", f1 + n(3), (0, 0, 0))
        # the head bobs with the work, a little
        h = self.claude_head
        for j, f in enumerate(ticks(f0, f1, 9)):
            key(h, "rotation_euler", f, (R(1.2 if j % 2 else -0.6), 0, R(rng.uniform(-1.5, 1.5))))
        key(h, "rotation_euler", f1 + n(4), (0, 0, 0))

    def claude_turn(self, f0: int, f1: int, facing: float):
        """The CRT turns on its foot to face `facing` degrees (0 faces -Y), with a small swing past at the end."""
        if not self.claude:
            raise ValueError('this kitchen has no Claude: give the set "claude": {}')
        r = self.claude
        now = r.rotation_euler.z
        goal = R(facing)
        key(r, "rotation_euler", f0)
        past = goal + 0.08 * (goal - now)
        key(r, "rotation_euler", max(f0 + 1, f1 - n(4)), (0, 0, past))
        key(r, "rotation_euler", f1, (0, 0, goal))
        self.events.append({"frame": f0, "kind": "rustle"})

    # ------------------------------------------------------------ the lid
    FINGER = 0.02  # how far a fingertip's middle is from what its pad lies on (m)

    def lid_edge(self, x: float = 0.0):
        """Where fingertips go on the lid to open or shut it, in the room, now, `x` along its top edge from the
        middle. Open, they lie on the top edge from above; shut, on the lid's back a little in from that edge
        (the back is on top then); in between, in between: so the fingers are never through the lid, and
        never in the base under it."""
        bpy.context.view_layer.update()
        shut, opened = self.lid_shut[0], self.lid_open[0]
        u = max(0.0, min(1.0, (self.hinge.rotation_euler.x - shut) / (opened - shut)))
        top = self.sh + 0.03
        off = self.FINGER * (1 + 0.8 * math.sin(math.pi * u))  # (further off in mid swing, where the lid moves fastest)
        return self.hinge.matrix_world @ Vector((x, (u - 1) * (0.007 + off), top - 0.03 + u * (0.03 + off)))

    def beside_laptop(self, s) -> Vector:
        """A place on the table top for a hand to lie beside the laptop (s = 1 their left): clear of the lid as
        it swings, and of a hand's own breadth."""
        edge = self.desk_edge(s)
        return Vector((s * max(abs(edge.x), 0.37), edge.y, edge.z))

    def move_lid(self, f0: int, f1: int, shut: bool):
        tween(self.hinge, "rotation_euler", f0, f1, self.lid_shut if shut else self.lid_open)
        sl = self.screen_light
        sl.keyframe_insert("energy", frame=f0 if shut else f0)
        sl.energy = 0.0 if shut else 18.0
        sl.keyframe_insert("energy", frame=f1 if shut else f0 + max(1, (f1 - f0) // 2))
        self.events.append({"frame": f1 if shut else f0, "kind": "lid_close" if shut else "lid_open"})

