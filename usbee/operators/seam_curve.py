import json

import bmesh
import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils.bvhtree import BVHTree

from ..geometry import islands

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"

CLOSE_LOOP_PIXEL_RADIUS = 15


def _get_or_create_collection(context):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(SEAM_CURVE_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


class USBEE_OT_draw_seam_curve(Operator):
    """Click on the mesh to place seam points directly on its surface.
    Right-click or Enter finishes an open cut (e.g. a dart); clicking back
    near the first point closes it into a loop. Esc cancels."""

    bl_idname = "usbee.draw_seam_curve"
    bl_label = "Draw Seam Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and context.active_object is not None
            and context.active_object.type == "MESH"
        )

    def _build_evaluated_bvh(self, context):
        # Raycast against the fully evaluated (post-modifier) surface - e.g.
        # a Subdivision Surface result - not the low-poly base cage, so
        # clicks land on the surface that will actually get flattened.
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

        location, _normal, face_index, _dist = self.bvh.ray_cast(local_origin, local_dir)
        if location is None:
            return None
        return location, face_index

    def _add_point(self, world_co, face_index, local_co):
        spline = self.curve_obj.data.splines[0]
        if len(self.hits) == 0:
            pass  # spline already has its one default point
        else:
            spline.points.add(1)
        spline.points[-1].co = (world_co.x, world_co.y, world_co.z, 1.0)
        self.hits.append({"face_index": face_index, "local_co": [local_co.x, local_co.y, local_co.z]})

    def _finish(self, context, cyclic):
        self.curve_obj.data.splines[0].use_cyclic_u = cyclic
        self.curve_obj[islands.BIND_DATA_PROP] = json.dumps(self.hits)
        for entry in self.mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        context.area.header_text_set(None)
        self.report({"INFO"}, f"Seam curve with {len(self.hits)} point(s) ({'closed' if cyclic else 'open'})")

    def _cancel(self, context):
        context.area.header_text_set(None)
        bpy.data.objects.remove(self.curve_obj, do_unlink=True)

    def invoke(self, context, event):
        self.mesh_obj = context.active_object
        self.hits = []
        self._build_evaluated_bvh(context)

        curve_data = bpy.data.curves.new(name="SeamCurve", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.splines.new(type="POLY")
        self.curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        self.curve_obj[SOURCE_OBJECT_PROP] = self.mesh_obj.name
        _get_or_create_collection(context).objects.link(self.curve_obj)

        context.area.header_text_set(
            "Click: add point | Click near start: close loop | "
            "Enter/Right-click: finish open cut | Esc: cancel"
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if len(self.hits) >= 3:
                first_2d = view3d_utils.location_3d_to_region_2d(
                    context.region, context.region_data, self.curve_obj.data.splines[0].points[0].co.to_3d()
                )
                if first_2d is not None:
                    dx = event.mouse_region_x - first_2d.x
                    dy = event.mouse_region_y - first_2d.y
                    if (dx * dx + dy * dy) ** 0.5 <= CLOSE_LOOP_PIXEL_RADIUS:
                        self._finish(context, cyclic=True)
                        return {"FINISHED"}

            hit = self._raycast(context, event)
            if hit is not None:
                local_co, face_index = hit
                world_co = self.mesh_obj.matrix_world @ local_co
                self._add_point(world_co, face_index, local_co)
            return {"RUNNING_MODAL"}

        if event.type in {"RIGHTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if len(self.hits) < 2:
                self.report({"WARNING"}, "Need at least 2 points - cancelled")
                self._cancel(context)
                return {"CANCELLED"}
            self._finish(context, cyclic=False)
            return {"FINISHED"}

        if event.type == "ESC":
            self._cancel(context)
            return {"CANCELLED"}

        # Let camera navigation (middle-mouse orbit/pan, scroll zoom) through.
        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"} or event.alt:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


class USBEE_OT_add_seam_curve(Operator):
    """Add a blank seam curve near the active mesh for manual shaping with
    Blender's normal curve tools (Tab into edit mode, extrude/move points) -
    an alternative to 'Draw Seam Curve' for precise manual control. Run
    'Bind Seam Curve to Mesh' after shaping it, and whenever you've moved
    points and want to re-snap them to the surface."""

    bl_idname = "usbee.add_seam_curve"
    bl_label = "Add Blank Seam Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object

        curve_data = bpy.data.curves.new(name="SeamCurve", type="CURVE")
        curve_data.dimensions = "3D"
        spline = curve_data.splines.new(type="POLY")
        spline.points.add(1)  # POLY splines start with 1 point; add 1 more -> 2 total

        # Seed two points near the mesh's origin so there's something
        # immediately visible/selectable to edit.
        center = mesh_obj.matrix_world.translation
        spline.points[0].co = (center.x - 0.05, center.y, center.z, 1.0)
        spline.points[1].co = (center.x + 0.05, center.y, center.z, 1.0)

        curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name

        coll = _get_or_create_collection(context)
        coll.objects.link(curve_obj)

        context.view_layer.objects.active = curve_obj
        for obj in context.selected_objects:
            obj.select_set(False)
        curve_obj.select_set(True)

        self.report({"INFO"}, f"Added seam curve for '{mesh_obj.name}' - edit it, then Bind")
        return {"FINISHED"}


class USBEE_OT_bind_seam_curve(Operator):
    """Snap the active seam curve's points onto its source mesh's surface
    and store the binding. Run this after creating a curve and after any
    edit to its points."""

    bl_idname = "usbee.bind_seam_curve"
    bl_label = "Bind Seam Curve to Mesh"
    bl_options = {"REGISTER", "UNDO"}

    source_object: StringProperty(
        name="Source Mesh",
        description="Overrides the curve's remembered source mesh if set",
        default="",
    )

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "CURVE"

    def execute(self, context):
        curve_obj = context.active_object
        mesh_name = self.source_object or curve_obj.get(SOURCE_OBJECT_PROP, "")
        mesh_obj = bpy.data.objects.get(mesh_name)
        if mesh_obj is None or mesh_obj.type != "MESH":
            self.report({"ERROR"}, f"No valid source mesh found for '{curve_obj.name}'")
            return {"CANCELLED"}

        curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name
        islands.bind_curve_to_mesh(curve_obj, mesh_obj)

        for entry in mesh_obj.usbee_pieces:
            entry.flatten_dirty = True

        self.report({"INFO"}, f"Bound '{curve_obj.name}' to '{mesh_obj.name}'")
        return {"FINISHED"}


_classes = (
    USBEE_OT_draw_seam_curve,
    USBEE_OT_add_seam_curve,
    USBEE_OT_bind_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
