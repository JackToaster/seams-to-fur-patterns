import bpy
from bpy.props import FloatProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

from ..export import dxf as dxf_export
from . import export_common, target


class SEAMS_TO_FUR_OT_export_dxf(Operator, ExportHelper):
    """Export all flattened pattern pieces for the active object to a DXF
    file, with layers real cutting/CNC/garment software expects
    (SEWLINE/CUTLINE/GRAINLINE/NOTCHES/LABELS)"""

    bl_idname = "seams_to_fur.export_dxf"
    bl_label = "Export Sewing Pattern (DXF)"
    bl_options = {"REGISTER"}

    filename_ext = ".dxf"
    filter_glob: StringProperty(default="*.dxf", options={"HIDDEN"})

    seam_allowance_mm: FloatProperty(
        name="Seam Allowance (mm)",
        description=(
            "Default seam allowance for pieces that don't have their own set "
            "(offset_mm == 0). Pieces with their own nonzero seam allowance are "
            "left exactly as configured - this never overrides them. 0 leaves "
            "every not-yet-configured piece with no cut line, as before"
        ),
        default=0.0,
        min=0.0,
        soft_max=25.0,
    )

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)

        try:
            mm_scale = export_common.blender_units_to_mm(context)
        except RuntimeError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        override = self.seam_allowance_mm if self.seam_allowance_mm > 0.0 else None
        pieces = export_common.gather_piece_export_data(
            context, mesh_obj, mm_scale, seam_allowance_override_mm=override
        )
        if not pieces:
            self.report({"ERROR"}, "No flattened pieces to export - run Flatten All first")
            return {"CANCELLED"}

        dxf_doc = dxf_export.build_dxf(pieces)
        with open(self.filepath, "w") as f:
            f.write(dxf_doc)

        self.report({"INFO"}, f"Exported {len(pieces)} piece(s) to {self.filepath}")
        return {"FINISHED"}


_classes = (SEAMS_TO_FUR_OT_export_dxf,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
