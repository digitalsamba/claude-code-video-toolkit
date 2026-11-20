"""A lecture hall for a presenter on their feet (runs inside Blender).

LectureHall: a wall with a board on it that shows a picture (a Remotion still of the board), a door beside the
board, a small table with a cup and a stick of chalk. No chair, laptop or rug. `walls=False` leaves the room out,
so the presenter and the table can be rendered over a transparent background and laid in front of a board that
Remotion draws.

The board is a rectangle of the back wall given as x0, x1, z0, z1 (metres): the same rectangle a `fixed` camera
frames square on, so a pixel of the Remotion frame is a point of the wall.

The hall takes its measures from the presenter, as the desk and chair do: `scale` is the presenter's height over
STATURE (1 for the clay cast, which the hall was built round), and the board and the table are that much bigger or
smaller unless the spec gives them in metres. So a fitted body of 2.25 m and one of 1.64 m both stand beside a
board that is as big against them as a lecture hall's is against a lecturer. A spec can also name places by the
board itself (`board_point`, `board_floor`, `board_camera`), which holds whatever the scale.
"""
from __future__ import annotations

import random

import bmesh
import bpy
from mathutils import Vector

from kit import R, Lamps, area_light, box, color, cyl, empty, glow, key, link, mat, media, mesh_obj, n, quad
from sets import DeskRoom

D = bpy.data


def cone(name, r0, r1, depth, m, parent=None, loc=(0, 0, 0)):
    """A cylinder that widens from r0 at the bottom to r1 at the top."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=28, radius1=r0, radius2=r1, depth=depth)
    return mesh_obj(name, bm, m, parent, loc, smooth=True)


WALL_Y = 2.95  # the back wall's face on the room side (the wall is 0.1 thick about y = 3)
STATURE = 1.75  # a lecturer's height: a fitted presenter this tall stands in the hall as it is built
BOARD = {"x0": -2.3, "x1": 1.7, "z0": 0.55, "z1": 2.8}  # the board for a presenter of STATURE (16:9, as the frame)
# A dressed hall (set "hall": {...}), in the hall's measure (metres for a presenter of STATURE): how deep it is from
# the board's wall to the back, how wide and high, the rows of benches on a rake with an aisle down the middle
# (in line with the board, so the board camera, which stands 12 m out, looks down it over the rows), the windows
# and the lights. Every number is an option of the set.
HALL = {"depth": 17.0, "width": 10.6, "height": 6.8, "seats": "benches",
        "rows": {"from": 4.6, "pitch": 0.95, "rise": 0.13, "aisle": 2.6},
        "windows": "left", "lights": True, "audience": 0, "seed": 7, "corridor": 6.0, "spill": 120.0, "board_dark": 0.06,
        "colors": {"seat": "#7A3B34", "bench": "#8A6444", "rake": "#5E4636", "ceiling": "#E4DDCC", "panel": "#9C7A57",
                   "sky": "#CFE0F2", "lamp": "#FFF1D6"}}


def door_x(rect: dict, door: dict) -> float:
    """Where a door with no "x" stands: beside the board, clear of its frame."""
    return door["x"] if "x" in door else max(2.6, rect["x1"] + 0.3 + door["width"] / 2)


def door_floor(rect: dict, scale: float, door: dict, out: float) -> list:
    """A spot on the floor on the door's line: `out` metres (the hall's measure) into the room from the wall, or
    behind it, in the corridor, when negative."""
    return [door_x(rect, door), WALL_Y - out * scale if out >= 0 else 3.05 - out * scale]


def boxes(name, parts, m, bevel=0.0):
    """Many boxes as one object (rows of benches, window bars): [(size, loc)], each a box at loc."""
    bm = bmesh.new()
    for size, loc in parts:
        made = bmesh.ops.create_cube(bm, size=1.0)["verts"]
        bmesh.ops.scale(bm, vec=size, verts=made)
        bmesh.ops.translate(bm, vec=loc, verts=made)
    ob = mesh_obj(name, bm, m, smooth=False)
    if bevel:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, 2, "ANGLE"
    return ob


def board_rect(board: dict | None, scale: float) -> dict:
    """The board's rectangle of the wall (x0, x1, z0, z1): as given, or the hall's own scaled by the presenter."""
    return {**{key: v * scale for key, v in BOARD.items()}, **{key: v for key, v in (board or {}).items() if key in BOARD}}


def board_floor(rect: dict, scale: float, u, out) -> list:
    """A spot on the floor in front of board pixel column u, `out` metres (hall measure) from the wall."""
    return [rect["x0"] + u / 1920 * (rect["x1"] - rect["x0"]), WALL_Y - out * scale]


class LectureHall(DeskRoom):
    def __init__(self, palette: dict | None = None, walls: bool = True, door: dict | None = None,
                 board: dict | None = None, image=None, strength: float = 1.0, table: dict | None = None,
                 scale: float = 1.0, hall: dict | None = None, lights: str = "on"):
        self.door_cfg, self.door, self.handle = door, None, None
        # "off": the hall starts dark, lit by its windows and the corridor, until a `lights` beat switches it on
        # (the ceiling lights and the hall's own fill in banks front to back, the presenter's lights with the
        # front bank). Built split into banks only then, so a hall that is lit from the start is as it was.
        self.dark = lights == "off"
        self.lamps = Lamps()
        self.switch = None
        self.chair, self.desk_y, self.desk_z, self.chair_z = None, 0.0, 0.0, 0.0
        p = {"wall": "#CFC6B2", "wall2": "#BDB29C", "floor": "#6E5540", "desk": "#8A6444", "board": "#2F4A3C",
             "trim": "#5B3F2A", **(palette or {})}
        self.events: list[dict] = []
        self.walls = walls
        self.glow = strength  # (emission that shows as its own colour under this film's exposure)
        self.scale = k = scale
        b = board_rect(board, k)
        self.board = b
        t = {"at": (1.6 * k, 1.42 * k), "size": (0.5 * k, 0.5 * k), "height": 0.97 * k, **(table or {})}
        if isinstance(t["at"], dict):  # {"board": [u, out]}: in front of a board column, as a presenter is placed,
            # and then its size and height are in the hall's measure too, as are the cup's and chalk's spots
            spot = lambda r: (b["x0"] + r["board"][0] / 1920 * (b["x1"] - b["x0"]), WALL_Y - r["board"][1] * k)  # noqa: E731
            t["at"] = spot(t["at"])
            if table and "size" in table:
                t["size"] = tuple(x * k for x in table["size"])
            if table and "height" in table:
                t["height"] = table["height"] * k
            for prop in ("cup", "chalk"):
                if isinstance(t.get(prop), dict):
                    t[prop] = spot(t[prop])
        self.table = t
        if door is not None and "x" not in door:  # (beside the board, clear of its frame)
            door = {**door, "x": door_x(b, door)}
            self.door_cfg = door
        self.hall = None
        if hall is not None and walls:  # (a dressed hall: deeper, wider and higher, with rows, windows and lights)
            self.hall = {**HALL, **hall, "rows": {**HALL["rows"], **(hall.get("rows") or {})},
                         "colors": {**HALL["colors"], **(hall.get("colors") or {})}}
            cx, hw = (b["x0"] + b["x1"]) / 2, self.hall["width"] / 2 * k
            self.back_wall = (cx - hw - 0.05, cx + hw + 0.05, self.hall["height"] * k)
        self.m = M = {
            "wood": mat("wood", p["desk"], 0.45), "metal": mat("metal", (0.2, 0.2, 0.22), 0.35, 0.8),
            "floor": mat("floor", p["floor"], 0.5), "wall": mat("wall", p["wall"], 0.9),
            "wall2": mat("wall2", p["wall2"], 0.9), "trim": mat("board_trim", p["trim"], 0.5),
            "board": mat("board", p["board"], 0.85), "cup": mat("cup", (0.92, 0.9, 0.84), 0.45),
            "lid": mat("cup_lid", (0.07, 0.06, 0.06), 0.45), "sleeve": mat("cup_sleeve", (0.42, 0.24, 0.12), 0.8),
            "chalk": mat("chalk", (0.95, 0.95, 0.92), 0.9),
        }
        if walls and self.hall:
            if door:
                self._door(M)
            else:
                x0, x1, top = self.back_wall
                box("wall_back", (x1 - x0, 0.1, top), M["wall"], loc=((x0 + x1) / 2, 3, top / 2))
            self._dress(M)
            self._board(image, strength)
        elif walls:
            box("floor", (10, 10, 0.02), M["floor"], loc=(0, 0, -0.01))
            if door:
                self._door(M)
            else:
                box("wall_back", (10, 0.1, 4), M["wall"], loc=(0, 3, 2))
            box("wall_front", (10, 0.1, 4), M["wall2"], loc=(0, -4.6, 2))
            box("wall_left", (0.1, 10, 4), M["wall2"], loc=(-4.2, 0, 2))
            box("wall_right", (0.1, 10, 4), M["wall"], loc=(4.2, 0, 2))
            self._board(image, strength)
        elif door:
            # (no room, but someone comes in through its door in a stretch that is built and not rendered: the
            # door is there to be taken hold of and opened, and is not drawn, nor is the light in the hall behind)
            self._door(M)
            for ob in D.objects:
                if ob.name.split(".")[0].startswith(("wall_back", "hall_", "door_", "handle_")):
                    ob.hide_render = True
        self.table_top = None
        if table is not False:  # (False: no table at all, as in a lecture that never puts its chalk down)
            self._table()
        self._props()
        if table is False:
            for ob in (self.cup_table, self.chalk_table, *self.cup_table.children):
                ob.scale, ob.hide_render = (0, 0, 0), True
        # (the lights stand where they would for a lecturer, as far from the presenter for any size, as bright)
        aim = empty("aim", None, (1.0 * k, 2.0, 1.3 * k))
        key_l = area_light("key", (-1.6 * k, -1.6 * k, 3.3 * k), 230 * k * k, 2.5 * k, (1, 0.93, 0.85), aim)
        fill = area_light("fill", (3.4 * k, -0.6 * k, 2.0 * k), 70 * k * k, 3 * k, (0.8, 0.88, 1.0), aim)
        top = area_light("top", (0.6 * k, 2.2, 3.7 * k), 90 * k * k, 2.0 * k, (1, 0.95, 0.88), aim)
        if self.dark:  # (the presenter's lights are the front bank's: on well before a walk to the board ends)
            for o in (key_l, fill, top):
                self.lamps.add(0, o)
            face = getattr(self, "board_face", None)
            if face is not None and face.active_material and face.active_material.name.startswith("board_picture"):
                # (a picture on the board shines by itself: in the dark it is as faint as a board by a window)
                self.lamps.add(0, face.active_material, dark=float((self.hall or {}).get("board_dark", HALL["board_dark"])))
            if self.door_cfg:
                self._switch()
                self._spill()
                corridor = D.objects.get("hall_light")
                if corridor:  # (the corridor, the one light in a dark hall, throws the doorway across the floor)
                    corridor.data.energy *= float((self.hall or {}).get("corridor", HALL["corridor"]))
            self.lamps.off(1)
            ee = bpy.context.scene.eevee
            if self.walls and hasattr(ee, "shadow_pool_size") and int(ee.shadow_pool_size) < 1024:
                # (a row of fill lights to a bank, each with its shadows, ran EEVEE's shadow pool out at 512 MB:
                # shadows popped between two states as the camera moved, the floor and the wall by the door 20 to
                # 40 levels from frame to frame. 1024 MB holds them all)
                ee.shadow_pool_size = "1024"
        if self.hall:  # (a dressed hall shows the wall over the board, where this light, close to it, makes a hot spot)
            wall = D.collections.new("top_light_skips")
            for ob in D.objects:
                if ob.name.split(".")[0] in ("wall_back", "wall_back_r", "wall_back_top"):
                    wall.objects.link(ob)
            top.light_linking.receiver_collection = wall
            for c in wall.collection_objects:
                c.light_linking.link_state = "EXCLUDE"

    # ------------------------------------------------------------ a dressed hall
    def out_y(self, d: float) -> float:
        """The y of a line `d` metres out from the board's wall, in the hall's measure."""
        return WALL_Y - d * self.scale

    def hall_point(self, x: float, out: float, z: float) -> list:
        """A point named in the hall's measure: x from the board's middle, `out` from its wall, z up from the floor."""
        b, k = self.board, self.scale
        return [(b["x0"] + b["x1"]) / 2 + x * k, self.out_y(out), z * k]

    def _dress(self, M):
        """A lecture hall worth seeing wide: a floor in front of the board, rows of desks and benches on a rake
        behind it with an aisle down the middle, a ceiling with rows of lights, tall windows down one side,
        panelling, slatted panels, a clock over the door. All in the hall's measure, so it is the same hall round any
        presenter. Its own lights light only the hall (light linking): the presenter is lit as in every clip."""
        h, k, b = self.hall, self.scale, self.board
        col = h["colors"]
        cx, hw, H, depth = (b["x0"] + b["x1"]) / 2, h["width"] / 2 * k, h["height"] * k, h["depth"]
        Y = self.out_y
        far = Y(depth)
        lit = []  # what the hall's own lights reach
        glow_k = getattr(self, "glow", 1.0)
        m = {"rake": mat("rake", col["rake"], 0.7), "bench": mat("bench", col["bench"], 0.5),
             "seat": mat("seat", col["seat"], 0.75), "ceiling": mat("ceiling", col["ceiling"], 0.9),
             "panel": mat("panel", col["panel"], 0.55), "sky": glow("window_sky", col["sky"], 1.0 * glow_k),
             "lamp": glow("lamp", col["lamp"], 1.6 * glow_k)}
        lit.append(box("floor", (2 * hw + 0.3, 5.0 - far, 0.02), M["floor"], loc=(cx, (5.0 + far) / 2, -0.01)))
        lit.append(box("wall_left", (0.1, 3.0 - far, H), M["wall2"], loc=(cx - hw - 0.05, (3.0 + far) / 2, H / 2)))
        lit.append(box("wall_right", (0.1, 3.0 - far, H), M["wall"], loc=(cx + hw + 0.05, (3.0 + far) / 2, H / 2)))
        lit.append(box("hall_end", (2 * hw + 0.2, 0.1, H), M["wall2"], loc=(cx, far - 0.05, H / 2)))
        lit.append(box("ceiling", (2 * hw + 0.2, 3.0 - far + 0.1, 0.1), m["ceiling"], loc=(cx, (3.0 + far) / 2, H + 0.05)))
        # the rake: a step up for each row, the last one running on to the back wall
        r = h["rows"]
        rows = 0 if h["seats"] in (None, "none", False) else max(0, int((depth - 0.7 - r["from"]) / r["pitch"]))
        steps, desks, seats, ends = [], [], [], []
        aisle = r["aisle"] / 2 * k
        for i in range(rows):
            d0 = r["from"] + i * r["pitch"]
            z = (i + 1) * r["rise"] * k
            y0, y1 = Y(d0), Y(depth)
            steps.append(((2 * hw + 0.04, y0 - y1, z), (cx, (y0 + y1) / 2, z / 2)))
            for side in (-1, 1):  # a block of desks and benches either side of the aisle
                xa, xb = cx + side * aisle, cx + side * (hw - 0.75 * k)
                w, mid = abs(xb - xa), (xa + xb) / 2
                desks.append(((w, 0.04 * k, 0.72 * k), (mid, Y(d0 + 0.04), z + 0.36 * k)))       # the front panel
                desks.append(((w + 0.04 * k, 0.32 * k, 0.04 * k), (mid, Y(d0 + 0.17), z + 0.74 * k)))  # its desk top
                seats.append(((w, 0.42 * k, 0.07 * k), (mid, Y(d0 + 0.62), z + 0.44 * k)))       # the bench
                seats.append(((w, 0.06 * k, 0.42 * k), (mid, Y(d0 + 0.86), z + 0.68 * k)))       # its back
                for x in (xa, xb):  # an end panel at each end of the block
                    ends.append(((0.05 * k, (r["pitch"] - 0.02) * k, 0.9 * k), (x + side * 0.0, Y(d0 + r["pitch"] / 2), z + 0.45 * k)))
        if rows:
            lit += [boxes("rake", steps, m["rake"]), boxes("desks", desks, m["bench"], 0.008 * k),
                    boxes("benches", seats, m["seat"], 0.01 * k), boxes("bench_ends", ends, m["bench"], 0.008 * k)]
        if rows and h["audience"]:  # plain seated shapes, not characters: a hall with people in it, kept quiet
            rng = random.Random(h["seed"])
            places = []
            for i in range(rows):
                for side in (-1, 1):
                    x, end = cx + side * (aisle + 0.45 * k), cx + side * (hw - 1.0 * k)
                    while side * (end - x) > 0:
                        places.append((i, x))
                        x += side * 0.62 * k
            tones = ("#5B6B7A", "#7A5B5B", "#6B7A5B", "#4E4E5E", "#8A7A62")
            skins = ("#C99A7A", "#A87458", "#7A5340", "#E0B896")
            bodies, heads = {t: [] for t in tones}, {t: [] for t in skins}
            for i, x in rng.sample(places, min(int(h["audience"]), len(places))):
                d0, z = r["from"] + i * r["pitch"], (i + 1) * r["rise"] * k
                y = Y(d0 + 0.66)
                bodies[rng.choice(tones)].append(((0.4 * k, 0.28 * k, 0.6 * k), (x, y, z + 0.47 * k + 0.3 * k)))
                heads[rng.choice(skins)].append((0.105 * k, (x, y + 0.01 * k, z + 1.19 * k)))
            for t, parts in bodies.items():
                if parts:
                    lit.append(boxes("audience", parts, mat("audience_" + t, t, 0.8), 0.09 * k))
            for t, parts in heads.items():
                if parts:
                    bm = bmesh.new()
                    for rad, at in parts:
                        made = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=rad)["verts"]
                        bmesh.ops.translate(bm, vec=at, verts=made)
                    lit.append(mesh_obj("audience_heads", bm, mat("audience_skin_" + t, t, 0.6)))
        # panelling along both side walls and the back, below the windows
        pz = 1.15 * k
        lit.append(boxes("panelling", [((0.03, 3.0 - far, pz), (cx - hw + 0.015, (3.0 + far) / 2, pz / 2)),
                                       ((0.03, 3.0 - far, pz), (cx + hw - 0.015, (3.0 + far) / 2, pz / 2)),
                                       ((2 * hw - 0.06, 0.03, pz), (cx, far + 0.015, pz / 2))], m["panel"]))
        # tall windows down one side (or both): a pane of sky, a frame, bars
        sides = {"left": (-1,), "right": (1,), "both": (-1, 1)}.get(h["windows"], ())
        frames, panes = [], []
        for side in sides:
            xw = cx + side * hw
            d = 3.4
            while d < depth - 1.4:
                yc, ww = Y(d), 1.5 * k
                z0, z1 = 2.7 * k, 5.9 * k
                t, dp = 0.07 * k, 0.05 * k
                # (the pane runs on under the frame and sits inside its depth: no face of it lies in one of the
                # frame's, which flickered at one scale of the hall and not at another)
                panes.append(((0.003 * k, ww + t, z1 - z0 + t), (xw - side * 0.31 * dp, yc, (z0 + z1) / 2)))
                xf = xw - side * dp / 2
                frames += [((dp, ww + 2 * t, t), (xf, yc, z1 + t / 2)), ((dp, ww + 2 * t, t), (xf, yc, z0 - t / 2)),
                           ((dp, t, z1 - z0), (xf, yc - ww / 2 - t / 2, (z0 + z1) / 2)),
                           ((dp, t, z1 - z0), (xf, yc + ww / 2 + t / 2, (z0 + z1) / 2)),
                           ((dp * 0.8, 0.035 * k, z1 - z0), (xf, yc, (z0 + z1) / 2))]
                for zb in (z0 + (z1 - z0) / 3, z0 + 2 * (z1 - z0) / 3):
                    frames.append(((dp * 0.8, ww, 0.035 * k), (xf, yc, zb)))
                frames.append(((dp * 2.2, ww + 2 * t + 0.06 * k, 0.05 * k), (xw - side * dp * 1.1, yc, z0 - t - 0.025 * k)))
                d += 3.1
        if panes:
            boxes("window_panes", panes, m["sky"])
            lit.append(boxes("window_frames", frames, M.get("trim2") or mat("window_frame", "#EDE8DC", 0.5), 0.004 * k))
        # on a wall without windows, tall panels of wooden slats at the windows' rhythm (the sound in a hall)
        slats, backs = [], []
        for side in [s_ for s_ in (-1, 1) if s_ not in sides]:
            d = 3.4
            while d < depth - 1.4:
                yc, ww, z0, z1 = Y(d), 1.5 * k, 2.4 * k, 5.9 * k
                xw = cx + side * hw
                backs.append(((0.03 * k, ww, z1 - z0), (xw - side * 0.016 * k, yc, (z0 + z1) / 2)))
                for j in range(7):
                    y = yc - ww / 2 + (j + 0.5) * ww / 7
                    slats.append(((0.05 * k, 0.09 * k, z1 - z0 + 0.1 * k), (xw - side * 0.055 * k, y, (z0 + z1) / 2)))
                d += 3.1
        if slats:
            lit += [boxes("wall_panels", backs, mat("panel_back", "#3E3128", 0.8)), boxes("wall_slats", slats, m["panel"], 0.006 * k)]
        # rows of lights in the ceiling
        rows_at = []  # how far out each row of ceiling lights is
        if h["lights"]:
            panels = []
            d = 2.2
            while d < depth - 0.8:
                rows_at.append(d)
                for x in (-3.2, 0.0, 3.2):
                    panels.append(((1.5 * k, 0.55 * k, 0.04), (cx + x * k, Y(d), H - 0.025)))
                d += 2.6
            if not self.dark:
                boxes("ceiling_lights", panels, m["lamp"])
            else:  # (a row to a bank, each its own glow, a diffuser that reads pale under the windows when off)
                for i in range(len(rows_at)):
                    lamp = mat(f"lamp_row{i}", "#BDB8AE", 0.6, emit=1.6 * glow_k, emit_color=col["lamp"])
                    boxes("ceiling_lights", panels[3 * i:3 * i + 3], lamp)
                    self.lamps.add(1 + i, lamp)
        # a clock over the door
        if self.door_cfg:
            dx, top = self.door_cfg["x"], self.door_cfg["height"]
            face = mat("clock_face", "#F4F1E8", 0.6)
            ring = cyl("clock_ring", 0.24 * k, 0.05, M["trim"], loc=(dx, WALL_Y - 0.025, top + 0.55 * k), rot=(R(90), 0, 0), seg=40, smooth=True)
            cyl("clock_face", 0.21 * k, 0.012, face, loc=(dx, WALL_Y - 0.052, top + 0.55 * k), rot=(R(90), 0, 0), seg=40)
            hand = mat("clock_hand", "#1E1E1E", 0.5)
            for length, ang in ((0.14, 62), (0.1, -118)):
                box("clock_hand", (0.012 * k, 0.008, length * k), hand, loc=(dx, WALL_Y - 0.062, top + 0.55 * k),
                    rot=(0, R(ang), 0)).data.transform(__import__("mathutils").Matrix.Translation((0, 0, length * k / 2)))
            lit.append(ring)
        # the hall's own light: from the ceiling over the rows and in at the windows, onto the hall only
        recv = D.collections.new("hall_lit")
        for ob in lit:
            recv.objects.link(ob)
        if not self.dark or not rows_at:
            aim = empty("hall_light_aim", None, (cx, Y(depth * 0.55), 0))
            fills = [area_light("hall_fill", (cx, Y(depth * 0.55), H - 0.3), 900 * k * k, depth * 0.6 * k, (1, 0.95, 0.88), aim)]
        else:  # (the same light in a strip under each row, so the hall lights up as its rows come on)
            fills = []
            for i, d in enumerate(rows_at):
                o = area_light("hall_fill", (cx, Y(d), H - 0.3), 900 * k * k / len(rows_at), 2.6 * k, (1, 0.95, 0.88),
                               empty("hall_light_aim", None, (cx, Y(d), 0)))
                o.data.shape, o.data.size, o.data.size_y = "RECTANGLE", 2 * hw * 0.9, 2.6 * k
                fills.append(o)
                self.lamps.add(1 + i, o)
        for side in sides:
            win_aim = empty("window_light_aim", None, (cx - side * hw, Y(depth * 0.55), 0))
            fills.append(area_light("window_light", (cx + side * (hw - 0.3), Y(depth * 0.55), 4.6 * k), 700 * k * k,
                                    depth * 0.5 * k, (0.82, 0.9, 1.0), win_aim))
        for f in fills:
            f.light_linking.receiver_collection = recv
        self.hall_objects = lit

    # ------------------------------------------------------------ the light switch and the lights coming on
    def _switch(self):
        """A switch plate on the wall beside the door, on the side away from its hinge (where a hand coming in
        finds it), at the height of a hand reaching out: the door handle's (sized for the presenter) and a
        little more. Named door_* so a room that is not drawn keeps it to be reached and does not draw it."""
        d, k = self.door_cfg, self.scale
        side = 1 if d["x"] > 0 else -1
        x = d["x"] - side * (d["width"] / 2 + 0.07 + 0.09 * k)
        z = d["handle"] + 0.2 * k
        plate = mat("switch_plate", "#ECE8DE", 0.4)
        self.switch = empty("door_switch", None, (x, WALL_Y, z))
        box("door_switch_plate", (0.085 * k, 0.012, 0.12 * k), plate, self.switch, (0, -0.006, 0), bevel=0.003)
        self.rocker = empty("door_switch_pivot", self.switch, (0, -0.013, 0))
        box("door_switch_rocker", (0.03 * k, 0.012, 0.05 * k), plate, self.rocker, (0, -0.004, 0), bevel=0.002)
        self.rocker.rotation_euler = (R(-9), 0, 0)  # (off: the top of the rocker in)

    SWITCH_ALONG = 0.25  # where someone stands to press the switch: this far from it toward the door (hall measure)

    def switch_floor(self, out: float, along: float = 0.0) -> list:
        """A spot on the floor in front of the switch, `out` metres into the room and `along` metres from it
        toward the door (both the hall's measure)."""
        if self.switch is None:
            raise ValueError('this hall has no light switch: a lecture_hall with a "door" and "lights": "off" has one')
        toward = 1 if self.door_cfg["x"] > self.switch.location.x else -1
        # (the plate hangs on nothing: its own place is where it is)
        return [self.switch.location.x + toward * along * self.scale, WALL_Y - out * self.scale]

    def _spill(self):
        """The corridor's light through the open doorway: a warm panel the size of the opening, facing into the
        room, off while the door is shut and up as it opens (open_door keys it), so the floor inside the door,
        the jamb and whoever stands in front of it catch the light. Hidden with the room when it is not drawn."""
        d, k = self.door_cfg, self.scale
        light = D.lights.new("door_spill", "AREA")
        light.shape, light.size, light.size_y = "RECTANGLE", d["width"] * 0.9, d["height"] * 0.8
        light.energy, light.color = float((self.hall or {}).get("spill", HALL["spill"])) * k * k, (1, 0.86, 0.7)
        self.spill = link(D.objects.new("door_spill", light))
        self.spill.location = (d["x"], WALL_Y + 0.07, d["height"] * 0.55)
        self.spill.constraints.new("TRACK_TO").target = empty("door_spill_aim", None, (d["x"], WALL_Y - 3.0 * k, 0.0))
        self.spill_full = light.energy
        light.energy = 0.0
        if not self.walls:
            self.spill.hide_render = True

    def open_door(self, f0: int, f1: int, angle: float = 100.0, latch: int | None = None):
        super().open_door(f0, f1, angle, latch)
        spill = getattr(self, "spill", None)
        if spill is not None:  # (the light through the doorway widens with the gap: a smooth change, not a step)
            spill.data.keyframe_insert("energy", frame=f0)
            spill.data.energy = self.spill_full if angle else 0.0
            spill.data.keyframe_insert("energy", frame=f0 + max(1, (f1 - f0) // 2))

    def switch_point(self) -> Vector:
        """Where a fingertip presses the switch: on the rocker's face, off the wall by the plate and the rocker."""
        if self.switch is None:
            raise ValueError('this hall has no light switch: a lecture_hall with a "door" and "lights": "off" has one')
        return self.switch.location + Vector((0, -0.028, 0))

    def lights(self, f: int, banks: int = 3, stutter: bool = True, gap: float = 11, pressed: bool = True) -> int:
        """The lights come on: the switch clicks on frame f (when someone presses it), and the rows of ceiling
        lights with the hall's fill under them come on in `banks` groups front to back, `gap` frames (at 24 fps)
        apart, the presenter's lights with the first; with `stutter` each group flashes first. Events: "switch",
        and for each group "lamp_stutter" as it starts and "lamp_on" as it holds full. Returns the frame the
        last group holds full from."""
        if not self.dark:
            raise ValueError('the lights are on already: give the set "lights": "off" for a hall that starts dark')
        if pressed and self.switch is not None:
            key(self.rocker, "rotation_euler", f - 1, (R(-9), 0, 0))
            key(self.rocker, "rotation_euler", f, (R(9), 0, 0))
            self.events.append({"frame": f, "kind": "switch"})
        rows = len(self.lamps.banks) - 1
        banks = max(1, min(int(banks), rows)) if rows else 1
        last = f
        for g in range(banks):
            at = f + n(5) + n(gap * g)
            members = ([0] if g == 0 else []) + [1 + r for r in range(rows) if r * banks // rows == g]
            held = max(self.lamps.on(b, at, stutter, seed=g) for b in members)
            if stutter:
                self.events.append({"frame": at, "kind": "lamp_stutter", "bank": g, "of": banks})
            self.events.append({"frame": held, "kind": "lamp_on", "bank": g, "of": banks})
            last = max(last, held)
        return last

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

    # ------------------------------------------------------------ places named by the board
    # A board pixel (u, v) is one of the Remotion frame's 1920x1080 when the board fills it: the board camera.
    def board_point(self, u, v) -> list:
        """The point of the board's face at pixel (u, v), a hand's breadth off it (where a point or a look goes)."""
        b = self.board
        return [b["x0"] + u / 1920 * (b["x1"] - b["x0"]), WALL_Y - 0.05, b["z1"] - v / 1080 * (b["z1"] - b["z0"])]

    def board_floor(self, u, out) -> list:
        """A spot on the floor in front of board column u, `out` metres from the wall in the hall's measure (so
        the same number puts any presenter as far off the board, against their size)."""
        return board_floor(self.board, self.scale, u, out)

    def door_floor(self, out) -> list:
        """A spot on the door's line, `out` metres into the room (the hall's measure), behind it when negative."""
        if not self.door_cfg:
            raise ValueError('this hall has no door: give the set "door": {}')
        return door_floor(self.board, self.scale, self.door_cfg, out)

    def board_camera(self, lens: float, sensor: float = 36.0) -> tuple[list, list]:
        """A camera square on to the board that frames it edge to edge with a `lens` mm lens: (cam, aim)."""
        b = self.board
        cx, cz = (b["x0"] + b["x1"]) / 2, (b["z0"] + b["z1"]) / 2
        d = (b["x1"] - b["x0"]) / 2 * lens / (sensor / 2)
        face = WALL_Y - 0.03
        return [cx, face - d, cz], [cx, face, cz]

    def board_px(self, point) -> tuple[float, float]:
        """Where a point of the board's face is, in board pixels (the inverse of board_point)."""
        b = self.board
        return ((point[0] - b["x0"]) / (b["x1"] - b["x0"]) * 1920, (b["z1"] - point[2]) / (b["z1"] - b["z0"]) * 1080)

    # ------------------------------------------------------------ what the compiler and the cameras ask of a set
    def anchors(self) -> dict:
        return {"screen": tuple(self.screen_center())}

    def desk_edge(self, s):
        raise ValueError("a lecture hall has no desk to pull a chair in to")

    def solids(self) -> list:
        return ([(self.table_top, 0.0)] if self.table_top else []) + ([(self.leaf, 0.0)] if self.door else [])

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
