import bpy
from bpy.props import StringProperty
from bpy.types import Operator

from ..geometry import islands

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"


def _get_or_create_collection(context):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(SEAM_CURVE_COLLECTION)
        context.scene.collection.children.link(coll)
    return coll


class USBEE_OT_add_seam_curve(Operator):
    """Add a new seam curve near the active mesh, ready to edit with
    Blender's normal curve tools (Tab into edit mode, extrude/move points).
    Run 'Bind Seam Curve to Mesh' after shaping it, and whenever you've
    moved points and want to re-snap them to the surface."""

    bl_idname = "usbee.add_seam_curve"
    bl_label = "Add Seam Curve"
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
    USBEE_OT_add_seam_curve,
    USBEE_OT_bind_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
