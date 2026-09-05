import bpy


def draw_object_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("usbee.draw_seam_curve", text="Draw Seam Curve")
    layout.operator("usbee.add_seam_curve", text="Add Blank Seam Curve")
    layout.operator("usbee.flatten_all", text="Flatten All Pieces")


def register():
    bpy.types.VIEW3D_MT_object.append(draw_object_menu)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(draw_object_menu)
