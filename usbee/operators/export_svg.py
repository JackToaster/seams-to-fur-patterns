import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

from ..export import svg as svg_export
from ..geometry import boundary


def _blender_units_to_mm(context):
    unit_settings = context.scene.unit_settings
    if unit_settings.system not in {"METRIC", "NONE"}:
        # Imperial scenes would need a different constant; rather than
        # silently exporting a wrong-scale pattern (the exact failure mode
        # that ruins a physical cut), Phase 1 only supports metric/none.
        raise RuntimeError(
            f"Scene unit system is '{unit_settings.system}', not METRIC - "
            f"switch it in Scene Properties > Units before exporting, so "
            f"the SVG comes out at the correct real-world size"
        )
    return unit_settings.scale_length * 1000.0


def _piece_export_data(mesh_obj, mm_scale):
    pieces = []
    for piece in mesh_obj.usbee_pieces:
        flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
        if flat_obj is None or flat_obj.type != "MESH":
            continue

        faces = [list(poly.vertices) for poly in flat_obj.data.polygons]
        try:
            loop_indices = boundary.ordered_boundary_loop(faces)
        except boundary.BoundaryError:
            continue

        mat = flat_obj.matrix_world
        boundary_mm = []
        for i in loop_indices:
            co = mat @ flat_obj.data.vertices[i].co
            boundary_mm.append((co.x * mm_scale, co.y * mm_scale))

        cut_line_mm = None
        cut_obj = bpy.data.objects.get(piece.cut_line_object) if piece.cut_line_object else None
        if cut_obj is not None and cut_obj.type == "CURVE" and cut_obj.data.splines:
            spline = cut_obj.data.splines[0]
            cmat = cut_obj.matrix_world
            cut_line_mm = []
            for pt in spline.points:
                co = cmat @ pt.co.to_3d()
                cut_line_mm.append((co.x * mm_scale, co.y * mm_scale))

        xs = [p[0] for p in boundary_mm]
        ys = [p[1] for p in boundary_mm]
        label_pos_mm = (sum(xs) / len(xs), sum(ys) / len(ys))

        pieces.append(
            {
                "name": piece.name,
                "boundary_mm": boundary_mm,
                "cut_line_mm": cut_line_mm,
                "label_pos_mm": label_pos_mm,
            }
        )
    return pieces


class USBEE_OT_export_svg(Operator, ExportHelper):
    """Export all flattened pattern pieces for the active object to an SVG file"""

    bl_idname = "usbee.export_svg"
    bl_label = "Export Sewing Pattern (SVG)"
    bl_options = {"REGISTER"}

    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and len(obj.usbee_pieces) > 0

    def execute(self, context):
        mesh_obj = context.active_object

        try:
            mm_scale = _blender_units_to_mm(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        pieces = _piece_export_data(mesh_obj, mm_scale)
        if not pieces:
            self.report({"ERROR"}, "No flattened pieces to export - run Flatten All first")
            return {"CANCELLED"}

        svg_doc = svg_export.build_svg(pieces)
        with open(self.filepath, "w") as f:
            f.write(svg_doc)

        self.report({"INFO"}, f"Exported {len(pieces)} piece(s) to {self.filepath}")
        return {"FINISHED"}


_classes = (USBEE_OT_export_svg,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
