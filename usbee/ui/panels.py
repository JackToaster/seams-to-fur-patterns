import bpy
from bpy.types import Panel, UIList

from ..backends import bff


class USBEE_UL_pieces(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="MESH_DATA")
        if item.error_message:
            row.label(text="", icon="ERROR")
        elif item.flatten_dirty:
            row.label(text="", icon="FILE_REFRESH")
        row.prop(item, "offset_mm", text="")


class USBEE_PT_main(Panel):
    bl_label = "USBee Sewing Patterns"
    bl_idname = "USBEE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "USBee"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        import os

        try:
            bff.find_binary(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            backend_ok = True
        except bff.BFFError:
            backend_ok = False

        box = layout.box()
        row = box.row()
        if backend_ok:
            row.label(text="Flattening backend ready", icon="CHECKMARK")
        else:
            row.label(text="Flattening backend missing", icon="ERROR")
            box.label(text="See Preferences > Add-ons > USBee", icon="INFO")

        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh object", icon="INFO")
            return

        layout.operator("usbee.draw_seam_curve", icon="GREASEPENCIL", text="Draw Seam Curve")
        layout.operator("usbee.add_seam_curve", icon="CURVE_DATA", text="Add Blank Seam Curve")

        col = layout.column()
        col.label(text="Pieces:")
        col.template_list(
            "USBEE_UL_pieces",
            "",
            obj,
            "usbee_pieces",
            obj,
            "usbee_active_piece_index",
            rows=4,
        )

        if 0 <= obj.usbee_active_piece_index < len(obj.usbee_pieces):
            active_piece = obj.usbee_pieces[obj.usbee_active_piece_index]
            if active_piece.error_message:
                err_box = layout.box()
                err_box.alert = True
                col = err_box.column(align=True)
                col.label(text=f"'{active_piece.name}' failed to flatten:", icon="ERROR")
                for line in active_piece.error_message.split("\n"):
                    if line.strip():
                        col.label(text=line.strip())

        row = layout.row(align=True)
        row.operator("usbee.flatten_piece", icon="FILE_REFRESH")
        row.operator("usbee.reset_placement", icon="LOOP_BACK")

        layout.operator("usbee.flatten_all", icon="MOD_TRIANGULATE")
        layout.separator()
        layout.operator("usbee.export_svg", icon="EXPORT")


_classes = (
    USBEE_UL_pieces,
    USBEE_PT_main,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
