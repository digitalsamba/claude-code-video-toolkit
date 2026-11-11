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


# ---------------------------------------------------------------- keyframes
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
                 view="AgX", transparent=False):
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
        for a, v in (("taa_render_samples", samples), ("use_raytracing", True)):
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
