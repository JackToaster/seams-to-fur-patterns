import math

import bmesh
import bpy
from bpy.types import Operator

from ..backends import bff
from ..geometry import boundary, islands, layout, polygon_offset
from .seam_curve import SEAM_CURVE_COLLECTION, SOURCE_OBJECT_PROP

PATTERN_COLLECTION_PREFIX = "USBee Pattern — "


def _addon_dir():
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_seam_curves_for(mesh_obj):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        return []
    return [o for o in coll.objects if o.get(SOURCE_OBJECT_PROP) == mesh_obj.name]


def _face_area_3d(verts, face):
    # Fan-triangulate from vertex 0 for area, fine for the small convex-ish
    # faces produced by ordinary modeling; good enough for the global scale
    # correction below (not used for the flattening itself).
    if len(face) < 3:
        return 0.0
    from mathutils import Vector

    area = 0.0
    v0 = Vector(verts[face[0]])
    for i in range(1, len(face) - 1):
        v1 = Vector(verts[face[i]])
        v2 = Vector(verts[face[i + 1]])
        area += (v1 - v0).cross(v2 - v0).length / 2.0
    return area


def _polygon_area_2d(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _pattern_collection(context, mesh_obj):
    name = f"{PATTERN_COLLECTION_PREFIX}{mesh_obj.name}"
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        context.scene.collection.children.link(coll)
    return coll


def _build_or_update_flat_object(context, mesh_obj, piece, verts2d, faces, grid_offset_xy):
    """Creates the flat object on first bake (applying the automatic
    shelf-layout offset so it doesn't overlap other pieces). On re-bake of
    an existing object, the shape is updated in place without re-applying
    the grid offset - this is what makes a manual reposition survive a
    re-bake, without needing depsgraph-based move detection."""
    pattern_coll = _pattern_collection(context, mesh_obj)

    flat_mesh_name = f"{mesh_obj.name}.{piece.name}"
    existing_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None

    if existing_obj is not None and existing_obj.type == "MESH":
        placed = [(x, y, 0.0) for (x, y, _z) in verts2d]
        flat_mesh = existing_obj.data
        flat_mesh.clear_geometry()
        flat_obj = existing_obj
    else:
        ox, oy = grid_offset_xy
        placed = [(x + ox, y + oy, 0.0) for (x, y, _z) in verts2d]
        flat_mesh = bpy.data.meshes.new(flat_mesh_name)
        flat_obj = bpy.data.objects.new(flat_mesh_name, flat_mesh)
        pattern_coll.objects.link(flat_obj)

    flat_mesh.from_pydata(placed, [], faces)
    flat_mesh.update()

    flat_obj["usbee_source_object"] = mesh_obj.name
    flat_obj["usbee_piece_uuid"] = piece.uuid

    piece.flattened_object = flat_obj.name
    return flat_obj, [(p[0], p[1]) for p in placed]


def _update_cut_line(context, mesh_obj, piece, flat_obj, boundary_loop_2d):
    if piece.offset_mm == 0.0:
        if piece.cut_line_object:
            old = bpy.data.objects.get(piece.cut_line_object)
            if old is not None:
                bpy.data.objects.remove(old, do_unlink=True)
            piece.cut_line_object = ""
        return

    offset_m = piece.offset_mm / 1000.0
    try:
        offset_points = polygon_offset.offset_polygon(boundary_loop_2d, offset_m)
    except polygon_offset.OffsetError as exc:
        raise RuntimeError(f"Piece '{piece.name}': {exc}") from exc

    curve_name = f"{flat_obj.name}.cut_line"
    existing = bpy.data.objects.get(piece.cut_line_object) if piece.cut_line_object else None
    if existing is not None and existing.type == "CURVE":
        curve_data = existing.data
        curve_data.splines.clear()
        curve_obj = existing
    else:
        curve_data = bpy.data.curves.new(curve_name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_obj = bpy.data.objects.new(curve_name, curve_data)
        _pattern_collection(context, mesh_obj).objects.link(curve_obj)

    spline = curve_data.splines.new(type="POLY")
    spline.points.add(len(offset_points) - 1)
    for i, (x, y) in enumerate(offset_points):
        spline.points[i].co = (x, y, 0.0, 1.0)
    spline.use_cyclic_u = True

    curve_obj.parent = flat_obj
    piece.cut_line_object = curve_obj.name


def _flatten_pieces(context, mesh_obj, piece_ids):
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    seam_curves = _find_seam_curves_for(mesh_obj)
    seam_edges = set()
    for curve_obj in seam_curves:
        seam_edges |= islands.resolve_curve_seam_edges(bm, curve_obj, mesh_obj)

    face_island, island_count = islands.isolate_islands(bm, seam_edges)
    islands.sync_piece_settings(mesh_obj, face_island, island_count)
    islands.write_piece_id_attribute(mesh_obj.data, face_island)

    binary_path = bff.find_binary(_addon_dir())

    errors = []
    boundaries_for_layout = []
    flat_results = {}  # piece_uuid -> (verts2d, faces, boundary_loop_2d)

    for piece in mesh_obj.usbee_pieces:
        if piece_ids is not None and piece.piece_id not in piece_ids:
            continue
        if not piece.flatten_dirty and piece_ids is None:
            continue

        island_face_indices = [i for i, isl in face_island.items() if isl == piece.piece_id]
        try:
            islands.validate_island_topology(bm, island_face_indices)
        except islands.TopologyError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        # Duplicates vertices along any seam edges internal to this island
        # (e.g. a dart cut) so the cut can actually open up when flattened,
        # rather than BFF seeing a fully-connected mesh with no cut at all.
        verts_list, faces_local = islands.split_island_for_flatten(
            bm, island_face_indices, seam_edges
        )

        try:
            out_verts, out_faces = bff.flatten_island(binary_path, verts_list, faces_local)
        except bff.BFFError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        area_3d = sum(_face_area_3d(verts_list, f) for f in faces_local)
        area_2d = sum(_polygon_area_2d([out_verts[i] for i in f]) for f in out_faces)
        scale = math.sqrt(area_3d / area_2d) if area_2d > 1e-12 else 1.0
        out_verts = [(x * scale, y * scale, 0.0) for (x, y, _z) in out_verts]

        try:
            loop_indices = boundary.ordered_boundary_loop(out_faces)
        except boundary.BoundaryError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue
        boundary_loop_2d = [(out_verts[i][0], out_verts[i][1]) for i in loop_indices]

        flat_results[piece.uuid] = (out_verts, out_faces, loop_indices)
        boundaries_for_layout.append((piece.uuid, boundary_loop_2d))

    placements = layout.shelf_layout(boundaries_for_layout)

    for piece in mesh_obj.usbee_pieces:
        if piece.uuid not in flat_results:
            continue
        out_verts, out_faces, loop_indices = flat_results[piece.uuid]
        grid_offset_xy = placements.get(piece.uuid, (0.0, 0.0))

        flat_obj, placed_verts_2d = _build_or_update_flat_object(
            context, mesh_obj, piece, out_verts, out_faces, grid_offset_xy
        )
        placed_boundary = [placed_verts_2d[i] for i in loop_indices]

        try:
            _update_cut_line(context, mesh_obj, piece, flat_obj, placed_boundary)
        except RuntimeError as exc:
            errors.append(str(exc))

        piece.flatten_dirty = False

    bm.free()
    return errors


class USBEE_OT_flatten_piece(Operator):
    """Re-flatten the active pattern piece"""

    bl_idname = "usbee.flatten_piece"
    bl_label = "Flatten Piece"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and len(obj.usbee_pieces) > 0

    def execute(self, context):
        mesh_obj = context.active_object
        if mesh_obj.usbee_active_piece_index >= len(mesh_obj.usbee_pieces):
            self.report({"ERROR"}, "No active piece selected")
            return {"CANCELLED"}
        piece = mesh_obj.usbee_pieces[mesh_obj.usbee_active_piece_index]
        # _flatten_pieces re-syncs (clears + rebuilds) usbee_pieces, which
        # invalidates this reference - capture the values we need as plain
        # strings/ints now, don't touch `piece` again after the call.
        piece_id, piece_name = piece.piece_id, piece.name

        errors = _flatten_pieces(context, mesh_obj, {piece_id})
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Flattened '{piece_name}'")
        return {"FINISHED"}


class USBEE_OT_flatten_all(Operator):
    """Re-flatten every piece that needs it (or all pieces, if none are dirty)"""

    bl_idname = "usbee.flatten_all"
    bl_label = "Flatten All"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object
        wm = context.window_manager
        wm.progress_begin(0, 1)
        try:
            errors = _flatten_pieces(context, mesh_obj, None)
        finally:
            wm.progress_end()

        n_pieces = len(mesh_obj.usbee_pieces)
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            if len(errors) >= n_pieces:
                return {"CANCELLED"}
        self.report({"INFO"}, f"Flattened {n_pieces - len(errors)}/{n_pieces} pieces")
        return {"FINISHED"}


class USBEE_OT_reset_placement(Operator):
    """Reset the active piece's flattened object back to the automatic grid layout"""

    bl_idname = "usbee.reset_placement"
    bl_label = "Reset Placement"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and len(obj.usbee_pieces) > 0

    def execute(self, context):
        mesh_obj = context.active_object
        piece = mesh_obj.usbee_pieces[mesh_obj.usbee_active_piece_index]

        # Deleting the existing flattened object (rather than just moving
        # it) is what makes _build_or_update_flat_object treat this as a
        # first bake and re-apply the automatic grid layout offset.
        old = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        piece.flattened_object = ""
        piece.flatten_dirty = True

        errors = _flatten_pieces(context, mesh_obj, {piece.piece_id})
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            return {"CANCELLED"}
        return {"FINISHED"}


_classes = (
    USBEE_OT_flatten_piece,
    USBEE_OT_flatten_all,
    USBEE_OT_reset_placement,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
