import bmesh
import bpy
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils.bvhtree import BVHTree

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"

CLOSE_LOOP_PIXEL_RADIUS = 15
# Ignore a click landing within this distance (in mesh-local units) of the
# previously placed point - a real click that close is almost always an
# accidental double-registration, not an intentional second point.
MIN_POINT_SPACING = 1e-5

SEAM_CURVE_COLOR = (1.0, 0.15, 0.05, 1.0)
SEAM_CURVE_BEVEL_DEPTH = 0.0025


def _get_or_create_collection(context):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(SEAM_CURVE_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


def _get_or_create_seam_curve_material():
    mat = bpy.data.materials.get("USBee Seam Curve")
    if mat is None:
        mat = bpy.data.materials.new("USBee Seam Curve")
        mat.diffuse_color = SEAM_CURVE_COLOR
        if mat.use_nodes:
            emission = mat.node_tree.nodes.get("Principled BSDF")
            if emission is not None:
                emission.inputs["Emission Color"].default_value = SEAM_CURVE_COLOR
                emission.inputs["Emission Strength"].default_value = 1.5
    return mat


def _make_seam_curve_visible(curve_obj):
    """Give the curve real on-screen thickness and draw it through occluding
    geometry, so it's never invisible just because it sits flush against
    (or slightly inside) a convex/concave part of the surface."""
    curve_obj.data.bevel_depth = SEAM_CURVE_BEVEL_DEPTH
    curve_obj.data.fill_mode = "FULL"
    curve_obj.show_in_front = True
    curve_obj.color = SEAM_CURVE_COLOR
    mat = _get_or_create_seam_curve_material()
    if mat.name not in (m.name for m in curve_obj.data.materials if m):
        curve_obj.data.materials.append(mat)


def _add_shrinkwrap(curve_obj, mesh_obj):
    """Keeps the curve glued to mesh_obj's surface regardless of exactly
    where its points were placed, and as the mesh deforms."""
    mod = curve_obj.modifiers.new(name="Shrinkwrap", type="SHRINKWRAP")
    mod.target = mesh_obj
    mod.wrap_method = "NEAREST_SURFACEPOINT"
    return mod


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

    def _add_point(self, world_co):
        spline = self.curve_obj.data.splines[0]
        if self.num_points == 0:
            pass  # spline already has its one default point
        else:
            spline.points.add(1)
        spline.points[-1].co = (world_co.x, world_co.y, world_co.z, 1.0)
        self.num_points += 1
        self.last_world_co = world_co.copy()
        self._update_header(context_area=self._area)

    def _update_header(self, context_area):
        context_area.header_text_set(
            f"{self.num_points} point(s) placed | Click: add point | "
            f"Click near start: close loop | Enter/Right-click: finish open cut | Esc: cancel"
        )

    def _finish(self, context, cyclic):
        self.curve_obj.data.splines[0].use_cyclic_u = cyclic
        for entry in self.mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        context.area.header_text_set(None)
        self.report(
            {"INFO"}, f"Seam curve with {self.num_points} point(s) ({'closed' if cyclic else 'open'})"
        )

    def _cancel(self, context):
        context.area.header_text_set(None)
        bpy.data.objects.remove(self.curve_obj, do_unlink=True)

    def invoke(self, context, event):
        self.mesh_obj = context.active_object
        self.num_points = 0
        self.last_world_co = None
        self._area = context.area
        self._build_evaluated_bvh(context)

        curve_data = bpy.data.curves.new(name="SeamCurve", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.splines.new(type="POLY")
        self.curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        self.curve_obj[SOURCE_OBJECT_PROP] = self.mesh_obj.name
        _get_or_create_collection(context).objects.link(self.curve_obj)
        _make_seam_curve_visible(self.curve_obj)
        _add_shrinkwrap(self.curve_obj, self.mesh_obj)

        self._update_header(context.area)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if self.num_points >= 3:
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
                local_co, _face_index = hit
                world_co = self.mesh_obj.matrix_world @ local_co
                if self.last_world_co is None or (world_co - self.last_world_co).length > MIN_POINT_SPACING:
                    self._add_point(world_co)
            return {"RUNNING_MODAL"}

        if event.type in {"RIGHTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if self.num_points < 2:
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
    an alternative to 'Draw Seam Curve' for precise manual control. A
    Shrinkwrap modifier keeps it glued to the surface as you move points."""

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
        # immediately visible/selectable to edit; the Shrinkwrap modifier
        # will pull them onto the surface once bound below.
        center = mesh_obj.matrix_world.translation
        spline.points[0].co = (center.x - 0.05, center.y, center.z, 1.0)
        spline.points[1].co = (center.x + 0.05, center.y, center.z, 1.0)

        curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name

        coll = _get_or_create_collection(context)
        coll.objects.link(curve_obj)
        _make_seam_curve_visible(curve_obj)
        _add_shrinkwrap(curve_obj, mesh_obj)

        context.view_layer.objects.active = curve_obj
        for obj in context.selected_objects:
            obj.select_set(False)
        curve_obj.select_set(True)

        self.report({"INFO"}, f"Added seam curve for '{mesh_obj.name}' - Tab into edit mode to shape it")
        return {"FINISHED"}


_classes = (
    USBEE_OT_draw_seam_curve,
    USBEE_OT_add_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
