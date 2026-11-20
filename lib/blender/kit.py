"""Low-level building blocks for scenes built from code (runs inside Blender).

Everything here is plain bpy/bmesh: materials, primitives, empties, text, and keyframe helpers. The higher
modules (character, sets, cameras) are written only in terms of these, so a Blender API change is fixed here once.
Tested on Blender 5.3.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

R = math.radians
D = bpy.data


FPS = 24  # set by setup_render; acting is written in frames at 24 fps and n() converts


def n(frames24: float) -> int:
    """A length written in frames at 24 fps, as frames at the scene's rate (at least 1), so a blink or a
    typing rhythm lasts the same time at 24, 25, 30 or 60 fps."""
    return max(1, int(round(frames24 * FPS / 24)))


def ticks(f0: int, f1: int, step24: float) -> list[int]:
    """Frames from f0 up to (not including) f1, one every `step24` frames at 24 fps: the same moments in time at
    any frame rate, which stepping by a rounded n() is not (7 frames at 24 fps is 8.75 at 30)."""
    out, i = [], 0
    while True:
        f = f0 + int(round(i * step24 * FPS / 24))
        if f >= f1 or (out and f <= out[-1]):
            return out
        out.append(f)
        i += 1


def scene():
    return bpy.context.scene


def link(ob):
    scene().collection.objects.link(ob)
    return ob


def hex_rgb(h: str) -> tuple[float, float, float]:
    """'#F26B1F' -> linear RGB (Blender colours are linear; hex codes are sRGB)."""
    h = h.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return tuple(out)


def color(c) -> tuple[float, float, float]:
    return hex_rgb(c) if isinstance(c, str) else tuple(c)


def mat(name, base, rough=0.5, metal=0.0, emit=0.0, emit_color=None):
    m = D.materials.new(name)
    m.use_nodes = True  # deprecated in 5.x (always on) but still required on older builds
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color(base), 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*color(emit_color or base), 1)
        b.inputs["Emission Strength"].default_value = emit
    return m


def glow(name, c, strength=2.0):
    """Unlit-looking emissive material (screen UI, signage)."""
    return mat(name, (0, 0, 0), 0.5, emit=strength, emit_color=c)


def emission_socket(m):
    return m.node_tree.nodes["Principled BSDF"].inputs["Emission Color"]


def mesh_obj(name, bm, m=None, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), smooth=True):
    for f in bm.faces:
        f.smooth = smooth
    me = D.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if m:
        me.materials.append(m)
    ob = link(D.objects.new(name, me))
    ob.parent, ob.location, ob.rotation_euler, ob.scale = parent, loc, rot, scale
    return ob


def sphere(name, r, m, parent=None, loc=(0, 0, 0), scale=(1, 1, 1), seg=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    return mesh_obj(name, bm, m, parent, loc, scale=scale)


def box(name, size, m, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), bevel=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges) + list(bm.verts), offset=bevel, segments=3, affect="EDGES", profile=0.5)
    return mesh_obj(name, bm, m, parent, loc, rot, smooth=bool(bevel))


def cyl(name, r, depth, m, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), seg=24, smooth=False):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=depth)
    return mesh_obj(name, bm, m, parent, loc, rot, smooth=smooth)


def arc_tube(name, radius, r, u0, u1, m, parent, loc, seg=24, ring=10, wrap=0.0):
    """A tube bent along an arc in the XZ plane (mouth shapes).

    `wrap` is the radius of a sphere centred on the parent that the arc should lie on (a head): each point is
    pushed back to its surface, so the ends and the bottom of a smile sit on the face instead of floating in
    front of it and shadowing it.
    """
    bm = bmesh.new()
    rows = []
    for i in range(seg + 1):
        u = u0 + (u1 - u0) * i / seg
        c = Vector((radius * math.cos(u), 0, radius * math.sin(u)))
        if wrap:
            x, z = loc[0] + c.x, loc[2] + c.z
            c.y = -math.sqrt(max(0.0, wrap * wrap - x * x - z * z)) - loc[1]
        n = Vector((math.cos(u), 0, math.sin(u)))
        rows.append([bm.verts.new(c + r * (math.cos(v) * n + math.sin(v) * Vector((0, 1, 0))))
                     for v in (2 * math.pi * j / ring for j in range(ring))])
    for i in range(seg):
        for j in range(ring):
            bm.faces.new((rows[i][j], rows[i][(j + 1) % ring], rows[i + 1][(j + 1) % ring], rows[i + 1][j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_obj(name, bm, m, parent, loc)


def quad(name, x0, x1, z0, z1, m, parent, loc=(0, 0, 0)):
    """A flat panel in the XZ plane facing +Y (screen UI)."""
    bm = bmesh.new()
    f = bm.faces.new([bm.verts.new(p) for p in ((x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1))])
    bm.normal_update()
    if f.normal.y < 0:
        f.normal_flip()
    # UVs as the viewer in front of the panel sees it: looking along -Y, their left is +X
    uv = bm.loops.layers.uv.new("UVMap")
    lo, hi = min(x0, x1), max(x0, x1)
    for loop in f.loops:
        co = loop.vert.co
        loop[uv].uv = ((hi - co.x) / (hi - lo), (co.z - z0) / (z1 - z0))
    return mesh_obj(name, bm, m, parent, loc, smooth=False)


def media(name, path, strength=1.0, frame_start=1, offset=0):
    """A screen showing an image or a video file (a Remotion render), unlit so its colours survive.

    A video advances one of its frames per scene frame, so render it at the spec's fps. `frame_start` is the
    scene frame its first frame shows on; `offset` skips into it. Past its end it holds the last frame.
    """
    m = D.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0, 0, 0, 1)
    b.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in b.inputs:  # no highlights from the room's lights over the picture
        b.inputs["Specular IOR Level"].default_value = 0.0
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = D.images.load(str(path), check_existing=True)
    tex.image, tex.extension = img, "EXTEND"
    if img.source == "MOVIE":
        iu = tex.image_user
        iu.frame_duration = max(1, img.frame_duration)
        iu.frame_start, iu.frame_offset = frame_start, offset
        iu.use_auto_refresh, iu.use_cyclic = True, False
    nt.links.new(tex.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = strength
    return m


def empty(name, parent=None, loc=(0, 0, 0), rot=(0, 0, 0)):
    e = link(D.objects.new(name, None))
    e.parent, e.location, e.rotation_euler = parent, loc, rot
    e.empty_display_size = 0.08
    return e


def text(name, body, size, m, parent, loc, rot, align="CENTER", extrude=0.0):
    cu = D.curves.new(name, "FONT")
    cu.body, cu.size, cu.align_x, cu.align_y, cu.extrude = body, size, align, "CENTER", extrude
    cu.materials.append(m)
    ob = link(D.objects.new(name, cu))
    ob.parent, ob.location, ob.rotation_euler = parent, loc, rot
    return ob


def area_light(name, loc, energy, size, c, target):
    light = D.lights.new(name, "AREA")
    light.energy, light.size, light.color = energy, size, color(c)
    o = link(D.objects.new(name, light))
    o.location = loc
    o.constraints.new("TRACK_TO").target = target
    return o


class Lamps:
    """Lights and lamp panels that come on over time from a beat (`lights`): a set puts each of its lights (their
    energy) and each glowing material (its emission strength) in a bank, front to back, and the beat keys every
    bank from off to full, each on its own frame. A light is on or off, so every key holds until the next one (a
    step, not a fade), and a fluorescent tube stutters before it holds: a short pattern of flashes, its own for
    each bank (seeded), about half a second. `changes` lists the first and last frame of each change, so the
    render's flicker check can leave out those frames and no others."""

    # a tube starting: (when, in frames at 24 fps from the bank's frame, and its share of full)
    STUTTER = ((0, 0.55), (2, 0.0), (5, 0.8), (6, 0.08), (9, 0.65), (10, 0.0))

    def __init__(self):
        self.banks: list[list] = []          # per bank: [(owner of the curves, thing, path, full, dark)]
        self.changes: list[list[int]] = []   # [first, last] frame of each change of light

    def add(self, bank: int, thing, dark: float = 0.0):
        """A light object (its energy) or a glowing material (its emission strength) into bank `bank`. `dark` is
        its share of full while the bank is off (a picture on a board, faint by the windows' light, not black)."""
        while len(self.banks) <= bank:
            self.banks.append([])
        if isinstance(thing, bpy.types.Object):
            self.banks[bank].append((thing.data, thing.data, "energy", thing.data.energy, dark))
        else:
            sock = thing.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
            self.banks[bank].append((thing.node_tree, sock, "default_value", sock.default_value, dark))

    def level(self, bank: int, f: int, share: float):
        """Bank `bank` at `share` of full from frame f, held until its next key."""
        for owner, thing, path, full, dark in self.banks[bank]:
            setattr(thing, path, full * (dark + (1 - dark) * share))
            thing.keyframe_insert(path, frame=f)
            for fc in fcurves(owner):
                for kp in fc.keyframe_points:
                    kp.interpolation = "CONSTANT"

    def off(self, f: int = 0):
        for b in range(len(self.banks)):
            self.level(b, f, 0.0)

    def on(self, bank: int, f: int, stutter: bool = True, seed: int = 0) -> int:
        """Bank `bank` comes on at frame f (with `stutter`, a few flashes first, a pattern of its own for each
        `seed`: banks given one seed flash together); returns the frame it holds full from."""
        import random
        rng = random.Random(seed)
        last = f
        if stutter:
            t = 0.0
            for dt, share in self.STUTTER:
                t = dt * (1 + rng.uniform(-0.2, 0.2))
                self.level(bank, f + (n(t) if t else 0), min(1.0, share + rng.uniform(-0.1, 0.1)) if share else 0.0)
            last = f + n(t + 2 + rng.uniform(0, 2))
        self.level(bank, last, 1.0)
        self.changes.append([f, last])
        return last


# ---------------------------------------------------------------- keyframes
def fcurves(ob):
    """An object's animation curves, wherever this Blender keeps them (layered actions from 4.4)."""
    ad = ob.animation_data
    if ad is None or ad.action is None:
        return []
    act = ad.action
    out = []
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    if not out and getattr(act, "fcurves", None) is not None:
        out = list(act.fcurves)
    return out


def key(t, attr, f, val=None):
    if val is not None:
        setattr(t, attr, val)
    t.keyframe_insert(attr, frame=f)


def tween(t, attr, f0, f1, val):
    """Hold the current value at f0 and arrive at val by f1."""
    t.keyframe_insert(attr, frame=f0)
    setattr(t, attr, val)
    t.keyframe_insert(attr, frame=f1)


def show(ob, f, on, full=(1, 1, 1), dur=2):
    tween(ob, "scale", f - n(dur), f, full if on else (0, 0, 0))


def unit_y_limb(name, r, m, parent, loc):
    """Unit-length tube along +Y (for Stretch To rigs)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=r, radius2=r * 0.88, depth=1.0)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(R(-90), 3, "X"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0.5, 0))
    return mesh_obj(name, bm, m, parent, loc)


def cycles_gpu():
    """Turn on the first GPU backend this machine has (Metal, OptiX, CUDA, HIP, oneAPI). Returns its name or None."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return None
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:  # this build doesn't have the backend
            continue
        prefs.get_devices()
        devices = [d for d in prefs.devices if d.type == backend]
        if devices:
            for d in prefs.devices:
                d.use = d.type == backend
            return backend
    return None


def setup_render(fps, res, engine="eevee", exposure=-0.9, look="AgX - Punchy", samples=32, world=(0.05, 0.055, 0.07),
                 view="AgX", transparent=False, raytracing=True):
    global FPS
    FPS = fps
    sc = scene()
    sc.render.fps = fps
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    if engine == "cycles":
        sc.render.engine = "CYCLES"
        backend = cycles_gpu()
        if backend:
            sc.cycles.device = "GPU"
        print("cycles device:", backend or "CPU (no GPU backend found: correct, only slower)")
        sc.cycles.samples, sc.cycles.use_denoising = max(samples, 64), True
    else:
        names = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in names else "BLENDER_EEVEE"
        for a, v in (("taa_render_samples", samples), ("use_raytracing", bool(raytracing))):
            if hasattr(sc.eevee, a):
                setattr(sc.eevee, a, v)
    try:
        sc.view_settings.view_transform = view
    except TypeError:
        sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = look if sc.view_settings.view_transform == "AgX" else "None"
    except TypeError:
        pass
    sc.view_settings.exposure = exposure
    w = D.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (*world, 1)


def bloom(cfg) -> str | None:
    """Glow round what is bright, added after the render (the compositor's glare node), so a part can look lit
    without its emission going past the point where a colour clips. cfg: true, or {"strength", "threshold",
    "size"} (strength 0-1, threshold in scene light where 1 is white, size 0-1). Returns a note if it could not
    be set up (the render goes on without it). The compositor's Python API changed in Blender 5: both are here."""
    cfg = cfg if isinstance(cfg, dict) else {}
    strength, threshold, size = float(cfg.get("strength", 0.5)), float(cfg.get("threshold", 0.9)), float(cfg.get("size", 0.5))
    sc = scene()
    if sc.render.film_transparent:
        return "bloom is left out of a transparent clip (the glow has no alpha to sit on): add it in the video"
    try:
        if hasattr(sc, "compositing_node_group"):  # Blender 5: a node group with its own output socket
            tree = D.node_groups.new("bloom", "CompositorNodeTree")
            tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            sc.compositing_node_group = tree
            out = tree.nodes.new("NodeGroupOutput")
        else:  # 4.2 to 4.5: the scene's own tree and a Composite node
            sc.use_nodes = True
            tree = sc.node_tree
            tree.nodes.clear()
            out = tree.nodes.new("CompositorNodeComposite")
        layers, glare = tree.nodes.new("CompositorNodeRLayers"), tree.nodes.new("CompositorNodeGlare")
        if "Type" in glare.inputs:  # 5: the node's settings are sockets
            glare.inputs["Type"].default_value = "Bloom"
            glare.inputs["Threshold"].default_value = threshold
            glare.inputs["Strength"].default_value = strength
            glare.inputs["Size"].default_value = size
            if "Quality" in glare.inputs:
                glare.inputs["Quality"].default_value = "High"
        else:
            glare.glare_type, glare.quality, glare.threshold = "BLOOM", "HIGH", threshold
            glare.mix = strength - 1  # -1 is the picture alone, 0 the picture and its glare in equal parts
            glare.size = max(1, min(9, round(1 + size * 8)))
        tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], out.inputs[0])
        sc.render.use_compositing = True
        return None
    except Exception as e:  # noqa: BLE001 (an API that moved: say so, render without the glow)
        return f"bloom could not be set up on this Blender ({type(e).__name__}: {e}): rendered without it"


def set_output(path, video: bool, alpha: bool = False):
    """A video file, a still, or (a path ending in a name stem, rendered as an animation) a PNG sequence."""
    r = scene().render
    ims = r.image_settings
    if hasattr(ims, "media_type"):  # Blender 5+: pick the media type before the format
        ims.media_type = "VIDEO" if video else "IMAGE"
    if video:
        ims.file_format = "FFMPEG"
        r.ffmpeg.format, r.ffmpeg.codec, r.ffmpeg.constant_rate_factor = "MPEG4", "H264", "HIGH"
    else:
        ims.file_format = "PNG"
        ims.color_mode = "RGBA" if alpha else "RGB"
        ims.compression = 15  # PNG effort: the frames are re-encoded on the host, so favour speed
    r.filepath = str(path)
