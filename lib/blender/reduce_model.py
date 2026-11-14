"""A smaller copy of a generated model to keep in a repository (runs inside Blender, from blender_render.py).

    blender -b -P reduce_model.py -- --model in.glb --out out.glb --faces 60000 --texture 1024 [--keep-all]

The mesh is welded along the seams of its texture (the importer splits them, and a split mesh reduced opens into
cracks), thinned inside the islands of the texture and never across their seams (hands.reduce: across a seam a
face is left painted from the wrong place), and written with its base colour texture alone at `texture` pixels.
`--faces 0` keeps every face and only shrinks the texture. Prints what share of the faces lost their paint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402

import hands  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--faces", type=int, default=60000)
    ap.add_argument("--texture", type=int, default=1024)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.model)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"{args.model} has no mesh")
    for o in bpy.context.scene.objects:
        o.select_set(o in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.object.mode_set(mode="OBJECT")
    made = len(ob.data.polygons)
    lost = held = 0
    if args.faces and made > args.faces:
        was = hands.surface(ob)
        held = hands.reduce(ob, args.faces)
        lost = hands.paint_lost(ob, was)
    for mat in ob.data.materials:
        tree = mat.node_tree
        for node in list(tree.nodes):
            if node.type != "TEX_IMAGE":
                continue
            links = node.outputs[0].links
            if links and links[0].to_socket.name == "Base Color":
                if max(node.image.size) > args.texture:
                    node.image.scale(args.texture, args.texture)
            else:
                tree.nodes.remove(node)
        for link in list(tree.links):
            if link.to_node.type == "NORMAL_MAP":
                tree.links.remove(link)
    for o in bpy.context.scene.objects:
        o.select_set(o is ob)
    bpy.ops.export_scene.gltf(filepath=args.out, use_selection=True, export_format="GLB",
                              export_image_format="JPEG", export_jpeg_quality=85)
    print(f"REDUCED {made} to {len(ob.data.polygons)} faces, {held} vertices on the texture's seams left alone, "
          f"{100 * lost:.2f}% of the faces painted from the wrong place, texture {args.texture} px -> {args.out}")


main()
