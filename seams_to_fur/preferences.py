import os

import bpy
from bpy.types import AddonPreferences

from .backends import bff


class SeamsToFurPreferences(AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        addon_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            path = bff.find_binary(addon_dir)
            layout.label(text=f"BFF backend: {path}", icon="CHECKMARK")
        except bff.BFFError as exc:
            layout.label(text="BFF backend not found", icon="ERROR")
            layout.label(text=str(exc))
            layout.label(
                text=(
                    "Phase 1: build bff-command-line yourself (see "
                    "GeometryCollective/boundary-first-flattening) and "
                    "place it under backends/bin/<platform>/ in this addon."
                )
            )


_classes = (SeamsToFurPreferences,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
