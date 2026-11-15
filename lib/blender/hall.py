"""A lecture hall for a presenter on their feet (runs inside Blender).

LectureHall: a wall with a board on it that shows a picture (a Remotion still of the board), a door beside the
board, a small table with a cup and a stick of chalk. No chair, laptop or rug. `walls=False` leaves the room out,
so the presenter and the table can be rendered over a transparent background and laid in front of a board that
Remotion draws.

The board is a rectangle of the back wall given as x0, x1, z0, z1 (metres): the same rectangle a `fixed` camera
frames square on, so a pixel of the Remotion frame is a point of the wall.
"""
from __future__ import annotations

import bmesh
import bpy
from mathutils import Vector

from kit import R, area_light, box, cyl, empty, mat, media, mesh_obj, quad
from sets import DeskRoom

D = bpy.data


def cone(name, r0, r1, depth, m, parent=None, loc=(0, 0, 0)):
    """A cylinder that widens from r0 at the bottom to r1 at the top."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=28, radius1=r0, radius2=r1, depth=depth)
    return mesh_obj(name, bm, m, parent, loc, smooth=True)


WALL_Y = 2.95  # the back wall's face on the room side (the wall is 0.1 thick about y = 3)


class LectureHall(DeskRoom):
    def __init__(self, palette: dict | None = None, walls: bool = True, door: dict | None = None,
                 board: dict | None = None, image=None, strength: float = 1.0, table: dict | None = None):
        self.door_cfg, self.door, self.handle = door, None, None
        self.chair, self.desk_y, self.desk_z, self.chair_z = None, 0.0, 0.0, 0.0
        p = {"wall": "#CFC6B2", "wall2": "#BDB29C", "floor": "#6E5540", "desk": "#8A6444", "board": "#2F4A3C",
             "trim": "#5B3F2A", **(palette or {})}
        self.events: list[dict] = []
        self.walls = walls
        b = {"x0": -2.3, "x1": 1.7, "z0": 0.55, "z1": 2.8, **(board or {})}
        self.board = b
        t = {"at": (1.6, 1.42), "size": (0.5, 0.5), "height": 0.97, **(table or {})}
        self.table = t
        self.m = M = {
            "wood": mat("wood", p["desk"], 0.45), "metal": mat("metal", (0.2, 0.2, 0.22), 0.35, 0.8),
            "floor": mat("floor", p["floor"], 0.5), "wall": mat("wall", p["wall"], 0.9),
            "wall2": mat("wall2", p["wall2"], 0.9), "trim": mat("board_trim", p["trim"], 0.5),
            "board": mat("board", p["board"], 0.85), "cup": mat("cup", (0.92, 0.9, 0.84), 0.45),
            "lid": mat("cup_lid", (0.07, 0.06, 0.06), 0.45), "sleeve": mat("cup_sleeve", (0.42, 0.24, 0.12), 0.8),
            "chalk": mat("chalk", (0.95, 0.95, 0.92), 0.9),
        }
        if walls:
            box("floor", (10, 10, 0.02), M["floor"], loc=(0, 0, -0.01))
            if door:
                self._door(M)
            else:
                box("wall_back", (10, 0.1, 4), M["wall"], loc=(0, 3, 2))
            box("wall_front", (10, 0.1, 4), M["wall2"], loc=(0, -4.6, 2))
            box("wall_left", (0.1, 10, 4), M["wall2"], loc=(-4.2, 0, 2))
            box("wall_right", (0.1, 10, 4), M["wall"], loc=(4.2, 0, 2))
            self._board(image, strength)
        self._table()
        self._props()
        aim = empty("aim", None, (1.0, 2.0, 1.3))
        area_light("key", (-1.6, -1.6, 3.3), 230, 2.5, (1, 0.93, 0.85), aim)
        area_light("fill", (3.4, -0.6, 2.0), 70, 3, (0.8, 0.88, 1.0), aim)
        area_light("top", (0.6, 2.2, 3.7), 90, 2.0, (1, 0.95, 0.88), aim)

    def _board(self, image, strength):
        """The board: the picture on a panel just off the wall, facing the room, and a wooden frame round it
        with a chalk tray under it. The frame stands outside the picture's rectangle, so the rectangle is the
        whole of what a camera framed on it sees."""
        b, M = self.board, self.m
        w, h = b["x1"] - b["x0"], b["z1"] - b["z0"]
        cx, cz = (b["x0"] + b["x1"]) / 2, (b["z0"] + b["z1"]) / 2
        y = WALL_Y - 0.03
        # (a panel made facing +Y, turned to face the room: its picture reads left to right from the room)
        turn = empty("board_at", None, (cx, y, 0), (0, 0, R(180)))
        face = media("board_picture", image, strength) if image else M["board"]
        self.board_face = quad("board_face", -w / 2, w / 2, b["z0"], b["z1"], face, turn)
        box("board_back", (w + 0.02, 0.02, h + 0.02), M["trim"], loc=(cx, WALL_Y - 0.012, cz))
        e = 0.06
        for size, at in (((w + 2 * e, 0.05, e), (cx, y - 0.005, b["z1"] + e / 2 + 0.001)),
                         ((w + 2 * e, 0.05, e), (cx, y - 0.005, b["z0"] - e / 2 - 0.001)),
                         ((e, 0.05, h - 0.002), (b["x0"] - e / 2 - 0.001, y - 0.005, cz)),
                         ((e, 0.05, h - 0.002), (b["x1"] + e / 2 + 0.001, y - 0.005, cz))):
            box("board_frame", size, M["trim"], loc=at, bevel=0.006)
        box("chalk_tray", (w * 0.9, 0.09, 0.02), M["trim"], loc=(cx, y - 0.06, b["z0"] - e - 0.012), bevel=0.004)

    def _table(self):
        t, M = self.table, self.m
        (x, y), (w, d), h = t["at"], t["size"], t["height"]
        self.table_top = box("table_top", (w, d, 0.04), M["wood"], loc=(x, y, h - 0.02), bevel=0.008)
        for sx in (-1, 1):
            for sy in (-1, 1):
                cyl("table_leg", 0.02, h - 0.04, M["metal"], loc=(x + sx * (w / 2 - 0.05), y + sy * (d / 2 - 0.05),
                                                                 (h - 0.04) / 2))

    CUP_H, CUP_R = 0.115, 0.045

    def _cup(self, name):
        """A takeaway cup: white, a brown sleeve round its middle and a dark lid, its middle at the object's
        origin. Wider at the top than the bottom, as a paper cup is."""
        M, h, r = self.m, self.CUP_H, self.CUP_R
        root = empty(name)
        cone(name + "_body", r * 0.78, r, h, M["cup"], root, (0, 0, 0))
        cone(name + "_sleeve", r * 0.86 + 0.0015, r * 0.97 + 0.0015, h * 0.5, M["sleeve"], root, (0, 0, 0.004))
        cyl(name + "_lid", r + 0.004, 0.016, M["lid"], root, (0, 0, h / 2 + 0.006), seg=28, smooth=True)
        cyl(name + "_lid_top", r * 0.8, 0.008, M["lid"], root, (0, 0, h / 2 + 0.016), seg=28, smooth=True)
        return root

    def _props(self):
        """A cup and a stick of chalk, each twice: one that rides in a hand and one that lies on the table. Only
        one of a pair shows at a time (the other is scaled to nothing), so putting a thing down is a swap."""
        t = self.table
        (x, y), h = t["at"], t["height"]
        self.cup_hand, self.cup_table = self._cup("cup_hand"), self._cup("cup_table")
        self.cup_hand.scale = self.cup_table.scale = (0, 0, 0)
        self.cup_spot = Vector(t.get("cup", (x - 0.1, y + 0.12))).to_3d() + Vector((0, 0, h + self.CUP_H / 2))
        self.cup_table.location = self.cup_spot
        if t.get("cup_on"):  # (the cup already stands on the table: a clip that starts after it was put down)
            self.cup_table.scale = (1, 1, 1)
        M = self.m
        self.chalk_hand = cyl("chalk_hand", 0.0065, 0.075, M["chalk"], seg=10, smooth=True)
        self.chalk_table = cyl("chalk_table", 0.0065, 0.075, M["chalk"], seg=10, smooth=True)
        self.chalk_spot = Vector(t.get("chalk", (x - 0.14, y - 0.06))).to_3d() + Vector((0, 0, h + 0.0065))
        self.chalk_table.location, self.chalk_table.rotation_euler = self.chalk_spot, (R(90), 0, R(35))
        self.chalk_hand.scale = (0, 0, 0)
        if t.get("chalk_on", True) is False:
            self.chalk_table.scale = (0, 0, 0)
        if t.get("hidden"):
            # (the table and what stands on it are there to be reached for, and are not drawn: a clip whose
            # window would show only a corner of the table)
            for ob in D.objects:
                if ob.name.split(".")[0].startswith(("table_", "cup_table", "chalk_table")):
                    ob.hide_render = True

    def prop(self, name):
        """(the one in a hand, the one on the table, where it stands on the table) for "cup" or "chalk"."""
        if name == "cup":
            return self.cup_hand, self.cup_table, self.cup_spot
        if name == "chalk":
            return self.chalk_hand, self.chalk_table, self.chalk_spot
        raise ValueError(f'this set has a "cup" and a "chalk", not {name!r}')

    # ------------------------------------------------------------ what the compiler and the cameras ask of a set
    def anchors(self) -> dict:
        return {"screen": tuple(self.screen_center())}

    def desk_edge(self, s):
        raise ValueError("a lecture hall has no desk to pull a chair in to")

    def solids(self) -> list:
        return [(self.table_top, 0.0)] + ([(self.leaf, 0.0)] if self.door else [])

    def screen_center(self) -> Vector:
        b = self.board
        return Vector(((b["x0"] + b["x1"]) / 2, WALL_Y - 0.03, (b["z0"] + b["z1"]) / 2))

    def screen_normal(self) -> Vector:
        return Vector((0, -1, 0))

    def screen_points(self) -> list:
        b, y = self.board, WALL_Y - 0.04
        return [self.screen_center()] + [Vector((x, y, z)) for x in (b["x0"] + 0.2, b["x1"] - 0.2)
                                         for z in (b["z0"] + 0.2, b["z1"] - 0.2)]

    def visible_objects(self) -> set:
        return {self.board_face} if self.walls else set()

    def move_chair(self, f0, f1, out):
        raise ValueError("a lecture hall has no chair")
