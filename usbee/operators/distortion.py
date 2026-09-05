"""In-place distortion preview: shows each piece at its original 3D
position/shape (post material-thickness offset), colored by how much each
face had to stretch or compress to flatten - a quick visual sanity check of
where a pattern will pull or bunch, without leaving the curved model view.
"""

import bpy
from bpy.types import Operator

from ..backends import bff
from ..geometry import islands
from . import flatten as flatten_ops

DISTORTION_COLLECTION_PREFIX = "USBee Distortion — "
DISTORTION_ATTR = "usbee_distortion"
DISTORTION_MATERIAL = "USBee Distortion Preview"


def _distortion_collection(context, mesh_obj):
    name = f"{DISTORTION_COLLECTION_PREFIX}{mesh_obj.name}"
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        context.scene.collection.children.link(coll)
    return coll


def _get_or_create_distortion_material():
    mat = bpy.data.materials.get(DISTORTION_MATERIAL)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(DISTORTION_MATERIAL)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = DISTORTION_ATTR
    ramp = nodes.new("ShaderNodeValToRGB")
    # 1.0 (no distortion) = green, stretched/compressed = red at the ends.
    ramp.color_ramp.elements[0].position = 0.7
    ramp.color_ramp.elements[0].color = (0.9, 0.1, 0.1, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.1, 0.8, 0.1, 1.0)
    mid = ramp.color_ramp.elements.new(0.85)
    mid.color = (0.9, 0.9, 0.1, 1.0)

    # Map the raw ratio (roughly 0.5-1.5) into the ramp's 0-1 factor range,
    # centered on 1.0 = no distortion.
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = 0.7
    map_range.inputs["From Max"].default_value = 1.3
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0

    links.new(attr_node.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    if bsdf is not None:
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _build_distortion_piece(context, mesh_obj, piece, verts_3d, faces, distortion_per_face):
    coll = _distortion_collection(context, mesh_obj)
    obj_name = f"{mesh_obj.name}.{piece.name}.distortion"

    existing = bpy.data.objects.get(obj_name)
    if existing is not None and existing.type == "MESH":
        mesh = existing.data
        mesh.clear_geometry()
        obj = existing
    else:
        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        coll.objects.link(obj)

    mesh.from_pydata(verts_3d, [], faces)
    mesh.update()

    attr = mesh.attributes.get(DISTORTION_ATTR)
    if attr is None:
        attr = mesh.attributes.new(name=DISTORTION_ATTR, type="FLOAT", domain="FACE")
    for i, value in enumerate(distortion_per_face):
        attr.data[i].value = value

    mat = _get_or_create_distortion_material()
    if not mesh.materials:
        mesh.materials.append(mat)
    else:
        mesh.materials[0] = mat

    obj.matrix_world = mesh_obj.matrix_world.copy()
    obj["usbee_source_object"] = mesh_obj.name
    obj["usbee_piece_uuid"] = piece.uuid
    return obj


def _compute_distortion_pieces(context, mesh_obj, piece_ids=None):
    """Returns a list of errors (strings). Builds/updates one distortion
    preview object per requested piece (default: all pieces)."""
    bm = flatten_ops._evaluated_bmesh(context, mesh_obj)

    seam_curves = flatten_ops._find_seam_curves_for(mesh_obj)
    seam_edges = set()
    for curve_obj in seam_curves:
        seam_edges |= islands.resolve_curve_seam_edges(bm, curve_obj, mesh_obj)

    face_island, island_count = islands.isolate_islands(bm, seam_edges)
    islands.sync_piece_settings(mesh_obj, bm, face_island, island_count)

    binary_path = bff.find_binary(flatten_ops._addon_dir())
    errors = []

    for piece in mesh_obj.usbee_pieces:
        if piece_ids is not None and piece.piece_id not in piece_ids:
            continue

        island_face_indices = [i for i, isl in face_island.items() if isl == piece.piece_id]
        try:
            islands.validate_island_topology(bm, island_face_indices)
        except islands.TopologyError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        verts_list, faces_local = islands.split_island_for_flatten(bm, island_face_indices, seam_edges)

        try:
            out_verts, out_faces = bff.flatten_island(binary_path, verts_list, faces_local)
        except bff.BFFError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        distortion_per_face = []
        for f_idx, face in enumerate(faces_local):
            area_3d = flatten_ops._face_area_3d(verts_list, face)
            area_2d = flatten_ops._polygon_area_2d([out_verts[i] for i in out_faces[f_idx]])
            distortion_per_face.append(area_2d / area_3d if area_3d > 1e-12 else 1.0)

        _build_distortion_piece(context, mesh_obj, piece, verts_list, faces_local, distortion_per_face)

    bm.free()
    return errors


class USBEE_OT_show_distortion_preview(Operator):
    """Show each piece at its original 3D position, colored by how much it
    stretches or compresses when flattened (green = little distortion,
    red = heavily stretched/compressed)"""

    bl_idname = "usbee.show_distortion_preview"
    bl_label = "Show Distortion Preview"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object
        errors = _compute_distortion_pieces(context, mesh_obj)
        if errors:
            self.report({"WARNING"}, "; ".join(errors))
        self.report({"INFO"}, f"Distortion preview shown for '{mesh_obj.name}'")
        return {"FINISHED"}


class USBEE_OT_hide_distortion_preview(Operator):
    """Remove the distortion preview objects for the active object"""

    bl_idname = "usbee.hide_distortion_preview"
    bl_label = "Hide Distortion Preview"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object
        name = f"{DISTORTION_COLLECTION_PREFIX}{mesh_obj.name}"
        coll = bpy.data.collections.get(name)
        if coll is not None:
            for obj in list(coll.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(coll)
        return {"FINISHED"}


_classes = (
    USBEE_OT_show_distortion_preview,
    USBEE_OT_hide_distortion_preview,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
