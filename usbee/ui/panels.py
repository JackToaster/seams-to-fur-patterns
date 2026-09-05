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
        row.prop(item, "color", text="")


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

        settings_box = layout.box()
        settings_box.prop(obj, "usbee_thickness_mm")
        settings_box.prop(obj, "usbee_placement_mode")
        distort_row = settings_box.row(align=True)
        distort_row.operator("usbee.show_distortion_preview", icon="SHADING_RENDERED")
        distort_row.operator("usbee.hide_distortion_preview", icon="X", text="")

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


class USBEE_PT_appearance(Panel):
    bl_label = "Appearance (Color / Fur)"
    bl_idname = "USBEE_PT_appearance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "USBee"
    bl_parent_id = "USBEE_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        layout.operator("usbee.sync_piece_colors", icon="MATERIAL")
        layout.operator("usbee.set_grain_direction", icon="EMPTY_SINGLE_ARROW")

        if 0 <= obj.usbee_active_piece_index < len(obj.usbee_pieces):
            active_piece = obj.usbee_pieces[obj.usbee_active_piece_index]
            layout.prop(active_piece, "grain_direction", text="Grain Dir.")

        layout.separator()
        fur_box = layout.box()
        fur_box.label(text="Fur / Hair Preview", icon="OUTLINER_OB_HAIR")
        fur_box.operator("usbee.apply_fur_preview", icon="PARTICLES")


_classes = (
    USBEE_UL_pieces,
    USBEE_PT_main,
    USBEE_PT_appearance,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
