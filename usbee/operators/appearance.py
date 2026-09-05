"""Phase 2: per-piece coloring, grain-direction picking, and a fur/hair
preview on the source (curved) mesh.

Coloring and grain-direction both need to know which base-mesh faces belong
to which piece. Island membership itself is only computed on the evaluated
(post-modifier) mesh during flatten (see operators/flatten.py), so here we
recompute it the same way and project it onto the base mesh's faces via
nearest-centroid matching (geometry/face_matching.py) - an approximation,
but robust regardless of what a generative modifier does to face
indices/counts.
"""

import bmesh
import bpy
from bpy.props import FloatProperty, IntProperty
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..geometry import face_matching, islands
from .flatten import _evaluated_bmesh, _find_seam_curves_for

GRAIN_ARROW_COLLECTION = "USBee Grain Directions"
GRAIN_ARROW_COLOR = (0.1, 0.6, 1.0, 1.0)

FUR_MODIFIER_NAME = "USBee Fur Preview"
FUR_MATERIAL_NAME = "USBee Fur"


# --- Per-piece coloring -----------------------------------------------------


def _get_or_create_piece_material(piece):
    mat_name = f"USBee Piece {piece.uuid[:8]}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    color = tuple(piece.color)
    mat.diffuse_color = color
    if mat.use_nodes:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color
    return mat


class USBEE_OT_sync_piece_colors(Operator):
    """Rebuild per-piece material slots and per-face material assignment on
    the source mesh from each piece's color, and refresh flattened objects'
    viewport colors to match. Run after a re-bake if colors look stale."""

    bl_idname = "usbee.sync_piece_colors"
    bl_label = "Sync Piece Colors"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and len(obj.usbee_pieces) > 0

    def execute(self, context):
        mesh_obj = context.active_object

        bm = _evaluated_bmesh(context, mesh_obj)
        seam_edges = set()
        for curve_obj in _find_seam_curves_for(mesh_obj):
            seam_edges |= islands.resolve_curve_seam_edges(bm, curve_obj, mesh_obj)
        face_island, _island_count = islands.isolate_islands(bm, seam_edges)

        ref_centers = [
            (tuple(f.calc_center_median()), face_island[f.index])
            for f in bm.faces
            if f.index in face_island
        ]
        bm.free()

        base_centers = [tuple(poly.center) for poly in mesh_obj.data.polygons]
        base_islands = face_matching.nearest_face_island(base_centers, ref_centers)

        while mesh_obj.data.materials:
            mesh_obj.data.materials.pop()

        slot_index_by_piece_id = {}
        for piece in mesh_obj.usbee_pieces:
            mat = _get_or_create_piece_material(piece)
            mesh_obj.data.materials.append(mat)
            slot_index_by_piece_id[piece.piece_id] = len(mesh_obj.data.materials) - 1

            flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
            if flat_obj is not None:
                flat_obj.color = tuple(piece.color)

        for poly, island_id in zip(mesh_obj.data.polygons, base_islands):
            if island_id in slot_index_by_piece_id:
                poly.material_index = slot_index_by_piece_id[island_id]

        self.report({"INFO"}, f"Synced colors for {len(mesh_obj.usbee_pieces)} piece(s)")
        return {"FINISHED"}


# --- Grain direction ---------------------------------------------------------


def _get_or_create_grain_arrow_material():
    mat = bpy.data.materials.get("USBee Grain Direction")
    if mat is None:
        mat = bpy.data.materials.new("USBee Grain Direction")
        mat.diffuse_color = GRAIN_ARROW_COLOR
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Emission Color"].default_value = GRAIN_ARROW_COLOR
                bsdf.inputs["Emission Strength"].default_value = 1.5
    return mat


def _get_or_create_arrow_collection(context):
    coll = bpy.data.collections.get(GRAIN_ARROW_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(GRAIN_ARROW_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _arrow_object_name(mesh_obj, piece):
    return f"{mesh_obj.name}.{piece.name}.grain_arrow"


def _place_arrow(context, mesh_obj, name, start_local, direction_local, length):
    """(Re)builds a 2-point beveled curve from start_local along
    direction_local, in mesh_obj's local space (via parenting with an
    identity-at-bind-time parent inverse, matching the local-space
    convention already used for sample_point/grain_direction)."""
    end_local = start_local + direction_local * length

    curve_obj = bpy.data.objects.get(name)
    if curve_obj is None or curve_obj.type != "CURVE":
        curve_data = bpy.data.curves.new(name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.splines.new(type="POLY")
        curve_obj = bpy.data.objects.new(name, curve_data)
        _get_or_create_arrow_collection(context).objects.link(curve_obj)
        curve_obj.parent = mesh_obj
        curve_obj.matrix_parent_inverse = mesh_obj.matrix_world.inverted()

    spline = curve_obj.data.splines[0]
    if len(spline.points) < 2:
        spline.points.add(2 - len(spline.points))
    spline.points[0].co = (start_local.x, start_local.y, start_local.z, 1.0)
    spline.points[1].co = (end_local.x, end_local.y, end_local.z, 1.0)

    curve_obj.data.bevel_depth = max(length * 0.04, 1e-5)
    curve_obj.show_in_front = True
    curve_obj.color = GRAIN_ARROW_COLOR
    mat = _get_or_create_grain_arrow_material()
    if mat.name not in (m.name for m in curve_obj.data.materials if m):
        curve_obj.data.materials.append(mat)
    return curve_obj


def refresh_grain_arrow(context, mesh_obj, piece):
    """(Re)builds the persistent grain-direction arrow for one piece,
    anchored at its sample_point. Safe to call repeatedly."""
    direction = Vector(piece.grain_direction)
    if direction.length < 1e-8:
        return None
    direction = direction.normalized()
    max_dim = max(mesh_obj.dimensions) if max(mesh_obj.dimensions) > 0 else 1.0
    length = max_dim * 0.12
    start = Vector(piece.sample_point)
    return _place_arrow(context, mesh_obj, _arrow_object_name(mesh_obj, piece), start, direction, length)


class USBEE_OT_set_grain_direction(Operator):
    """Click on the active piece then drag to set its fabric/fur grain
    direction. Release to confirm; Esc cancels without changing anything."""

    bl_idname = "usbee.set_grain_direction"
    bl_label = "Set Grain Direction"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and obj is not None
            and obj.type == "MESH"
            and len(obj.usbee_pieces) > 0
        )

    def _build_bvh(self, context):
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = self.mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()

    def _raycast(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        mat_inv = self.mesh_obj.matrix_world.inverted()
        local_origin = mat_inv @ ray_origin
        local_dir = (mat_inv.to_3x3() @ ray_dir).normalized()

        location, _normal, _face_index, _dist = self.bvh.ray_cast(local_origin, local_dir)
        return location

    def invoke(self, context, event):
        self.mesh_obj = context.active_object
        idx = self.mesh_obj.usbee_active_piece_index
        if not (0 <= idx < len(self.mesh_obj.usbee_pieces)):
            self.report({"ERROR"}, "No active piece selected")
            return {"CANCELLED"}
        self.piece = self.mesh_obj.usbee_pieces[idx]
        self.start = None
        self._build_bvh(context)

        context.area.header_text_set(
            "Click and drag across the piece to set grain direction | Esc: cancel"
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._raycast(context, event)
            if hit is not None:
                self.start = hit.copy()
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE" and self.start is not None:
            hit = self._raycast(context, event)
            if hit is not None and (hit - self.start).length > 1e-8:
                direction = (hit - self.start).normalized()
                length = (hit - self.start).length
                _place_arrow(
                    context,
                    self.mesh_obj,
                    _arrow_object_name(self.mesh_obj, self.piece),
                    self.start,
                    direction,
                    length,
                )
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self.start is not None:
                hit = self._raycast(context, event)
                if hit is not None and (hit - self.start).length > 1e-8:
                    direction = (hit - self.start).normalized()
                    self.piece.grain_direction = (direction.x, direction.y, direction.z)
                    refresh_grain_arrow(context, self.mesh_obj, self.piece)
                    self.report({"INFO"}, f"Grain direction set for '{self.piece.name}'")
            context.area.header_text_set(None)
            return {"FINISHED"}

        if event.type == "ESC":
            context.area.header_text_set(None)
            if self.start is not None:
                # Drop the live preview arrow; restore the committed one (if any).
                refresh_grain_arrow(context, self.mesh_obj, self.piece)
            return {"CANCELLED"}

        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"} or event.alt:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


# --- Fur / hair preview ------------------------------------------------------


def _get_or_create_fur_material():
    mat = bpy.data.materials.get(FUR_MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(FUR_MATERIAL_NAME)
        color = (0.55, 0.4, 0.25, 1.0)
        mat.diffuse_color = color
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Base Color"].default_value = color
                if "Roughness" in bsdf.inputs:
                    bsdf.inputs["Roughness"].default_value = 0.8
    return mat


class USBEE_OT_apply_fur_preview(Operator):
    """Add (or refresh) a hair particle system on the source mesh to
    preview a faux-fur look. Idempotent - re-running updates the existing
    system by name instead of stacking duplicates.

    Limitation: strands are combed with a gentle uniform tangent lean, not
    driven per-piece by grain_direction - true per-face direction control
    needs either interactive particle-edit combing or a geometry-nodes hair
    setup, both out of scope for this pass.
    """

    bl_idname = "usbee.apply_fur_preview"
    bl_label = "Apply Fur Preview"
    bl_options = {"REGISTER", "UNDO"}

    hair_length: FloatProperty(name="Hair Length", default=0.02, min=0.0001, soft_max=0.2)
    density: IntProperty(name="Density", default=2000, min=10, soft_max=20000)

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object

        existing_mod = mesh_obj.modifiers.get(FUR_MODIFIER_NAME)
        if existing_mod is not None and existing_mod.type == "PARTICLE_SYSTEM":
            psys = existing_mod.particle_system
        else:
            mesh_obj.modifiers.new(name=FUR_MODIFIER_NAME, type="PARTICLE_SYSTEM")
            psys = mesh_obj.particle_systems[-1]
            psys.name = FUR_MODIFIER_NAME

        settings = psys.settings
        settings.type = "HAIR"
        settings.count = self.density
        settings.hair_length = self.hair_length
        settings.use_advanced_hair = True
        settings.tangent_factor = 0.3
        settings.tangent_phase = 0.0
        settings.child_type = "INTERPOLATED"
        settings.rendered_child_count = 10
        settings.render_step = 3

        mat = _get_or_create_fur_material()
        if mat.name not in (m.name for m in mesh_obj.data.materials if m):
            mesh_obj.data.materials.append(mat)
        try:
            settings.material_slot = mat.name
        except TypeError:
            settings.material = mesh_obj.data.materials.find(mat.name) + 1

        self.report(
            {"INFO"},
            "Fur preview applied - switch to Material Preview/Rendered shading to see strands",
        )
        return {"FINISHED"}


_classes = (
    USBEE_OT_sync_piece_colors,
    USBEE_OT_set_grain_direction,
    USBEE_OT_apply_fur_preview,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
