import bpy


def draw_object_menu(self, context):
    layout = self.layout
    layout.separator()
    layout.operator("seams_to_fur.draw_seam_curve", text="Draw Seam Curve")
    layout.operator("seams_to_fur.mirror_seam_curve", text="Mirror Seam Curve")
    layout.operator("seams_to_fur.flatten_all", text="Flatten All Pieces")


def register():
    bpy.types.VIEW3D_MT_object.append(draw_object_menu)


def unregister():
    bpy.types.VIEW3D_MT_object.remove(draw_object_menu)
