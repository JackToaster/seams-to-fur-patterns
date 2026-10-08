import bpy
import bpy.utils.previews
from bpy.app.handlers import persistent
from bpy.types import Panel, UIList

from ..backends import bff
from ..fur_colors import FUR_COLOR_SWATCHES
from ..operators import distortion, preview, target
from ..operators.seam_curve import SOURCE_OBJECT_PROP


# Session-only cache (not scene data) of the last active object we synced
# the Pieces list's selection from - see _sync_from_object.
_last_synced_object_name = None

# Dummy owner object for the msgbus subscription below - Blender's
# recommended pattern (see bpy.msgbus docs) for a stable, addon-lifetime-
# scoped subscription handle.
_msgbus_owner = object()

# One flat-color icon per FUR_COLOR_SWATCHES entry, generated once (not
# loaded from files - there are no swatch image files, just RGB tuples) so
# each color button in the Fur Colors panel actually shows its own color
# instead of a generic icon. Built lazily on first draw, not at register()
# time, since a fresh preview needs Blender's image/icon system already
# running.
_fur_color_previews = None


def _get_fur_color_previews():
    global _fur_color_previews
    if _fur_color_previews is None:
        _fur_color_previews = bpy.utils.previews.new()
        for name, color in FUR_COLOR_SWATCHES:
            preview = _fur_color_previews.new(name)
            preview.image_size = (8, 8)
            r, g, b = color
            pixels = [r, g, b, 1.0] * (8 * 8)
            preview.image_pixels_float = pixels
    return _fur_color_previews


def _sync_from_object(context, raw_obj):
    """Selecting a piece's object in the viewport (Sliced/Distortion
    Preview, or a flattened pattern piece) selects the same piece in the
    Pieces list, so e.g. the grain direction / fur length fields update to
    match what's actually selected. Only re-syncs when the active object
    itself just changed, tracked via a session-only cache - otherwise this
    would fight a manual click in the list itself on every subsequent call
    while the same viewport object stays active."""
    global _last_synced_object_name
    current_name = raw_obj.name if raw_obj is not None else None
    if current_name == _last_synced_object_name:
        return
    _last_synced_object_name = current_name
    if raw_obj is None:
        return
    mesh_obj = target.resolve_mesh_obj(context)
    if mesh_obj is None:
        return
    index = target.resolve_piece_index(context, mesh_obj)
    if index is not None and mesh_obj.seams_to_fur_active_piece_index != index:
        mesh_obj.seams_to_fur_active_piece_index = index


def _sync_active_piece_from_selection(context, mesh_obj):
    """Panel-draw-time entry point - thin wrapper for _sync_from_object,
    kept as a cheap fallback alongside the msgbus subscription below (the
    real, reliable trigger: a panel only redraws on its own schedule, which
    isn't guaranteed to fire promptly - or in a background/headless
    context, at all - on a bare selection change).

    Reads target.view_layer_active(context), not context.active_object -
    the two differ once the active object is hidden (confirmed live: a
    preview's refresh/toggle hides the base mesh, and context.active_object
    then goes None even though the view layer's active pointer still holds
    it - see that function's docstring)."""
    _sync_from_object(context, target.view_layer_active(context))


def _on_active_object_change():
    """msgbus notify callback: fires specifically when the active object
    (bpy.types.LayerObjects.active) changes, which is the actual reliable
    signal for "the user clicked something else in the viewport" - unlike
    panel draw() timing, or bpy.app.handlers.depsgraph_update_post (which
    doesn't necessarily fire for a pure selection/active-object change with
    no other scene data touched)."""
    context = bpy.context
    view_layer = context.view_layer
    raw_obj = view_layer.objects.active if view_layer is not None else None
    _sync_from_object(context, raw_obj)
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


@persistent
def _subscribe_msgbus(*_args):
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_msgbus_owner,
        args=(),
        notify=_on_active_object_change,
    )


def _draw_preview_row(layout, obj, kind, icon, label):
    """One row shared by every cached preview (Cut/Sliced/Distortion, and
    Fur in the appearance panel): a refresh button (styled with .alert
    when the underlying source has actually changed since the last
    refresh, i.e. clicking it would trigger a real recompute rather than
    a free no-op that just re-shows what's already cached), then a
    toggle-visibility button and a clear ("X") button, both greyed out
    until a refresh has actually cached something for this kind to
    toggle/clear."""
    cached = preview.is_preview_cached(obj, preview.PREVIEW_KINDS[kind][0])
    visible = preview.is_preview_visible(obj, preview.PREVIEW_KINDS[kind][0])
    stale = preview.is_preview_stale(obj, preview.PREVIEW_KINDS[kind][0])

    row = layout.row(align=True)

    refresh_sub = row.row(align=True)
    refresh_sub.alert = stale
    op = refresh_sub.operator("seams_to_fur.refresh_preview", icon=icon, text=label)
    op.kind = kind

    toggle_sub = row.row(align=True)
    toggle_sub.enabled = cached
    op = toggle_sub.operator(
        "seams_to_fur.toggle_preview", icon="HIDE_OFF" if visible else "HIDE_ON", text=""
    )
    op.kind = kind

    clear_sub = row.row(align=True)
    clear_sub.enabled = cached
    op = clear_sub.operator("seams_to_fur.clear_preview", icon="X", text="")
    op.kind = kind


class SEAMS_TO_FUR_UL_pieces(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon="MESH_DATA")
        if item.error_message:
            row.label(text="", icon="ERROR")
        elif item.flatten_dirty:
            row.label(text="", icon="FILE_REFRESH")
        row.prop(item, "offset_mm", text="")
        row.prop(item, "color", text="")


class SEAMS_TO_FUR_PT_main(Panel):
    bl_label = "Seams to Fur Patterns"
    bl_idname = "SEAMS_TO_FUR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Seams to Fur"

    def draw(self, context):
        layout = self.layout
        # target.view_layer_active(context), not context.active_object -
        # the two differ once the active object is hidden (confirmed live:
        # showing a same-shape preview hides the base mesh, and
        # context.active_object then goes None even though the view
        # layer's active pointer still holds it - see that function's
        # docstring). Using the raw pointer is what keeps this panel from
        # dropping to "Select a mesh object" the instant a preview hides
        # its own source mesh.
        raw_obj = target.view_layer_active(context)

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
            box.label(text="See Preferences > Add-ons > Seams to Fur Patterns", icon="INFO")

        if raw_obj is None:
            layout.label(text="Select a mesh object", icon="INFO")
            return

        source_name = raw_obj.get(SOURCE_OBJECT_PROP)
        if source_name is not None:
            # The active object is itself a seam curve skeleton, not the
            # source mesh - Tab into Edit Mode to move/extrude/merge/delete
            # its points directly, and add modifiers (Mirror, Array, ...)
            # via the normal Modifier Properties panel.
            layout.label(text=f"Seam curve for '{source_name}'", icon="MESH_DATA")
            layout.operator("seams_to_fur.mirror_seam_curve", icon="MOD_MIRROR", text="Mirror Seam Curve")
            layout.operator("seams_to_fur.refresh_seam_display", icon="FILE_REFRESH", text="Refresh Display")
            return

        # Resolves any Seams to Fur Patterns-generated object (cut/sliced/distortion
        # preview, flattened pattern piece) the user clicked on back to its
        # source mesh, so these controls stay put instead of disappearing
        # the moment the user clicks something other than the base mesh.
        obj = target.resolve_mesh_obj(context)
        if obj is None or obj.type != "MESH":
            layout.label(text="Select a mesh object", icon="INFO")
            return

        if obj is not raw_obj:
            layout.label(text=f"Editing '{obj.name}'", icon="INFO")

        _sync_active_piece_from_selection(context, obj)

        layout.operator("seams_to_fur.draw_seam_curve", icon="GREASEPENCIL", text="Draw Seam Curve")

        settings_box = layout.box()
        settings_box.prop(obj, "seams_to_fur_thickness_mm")
        settings_box.prop(obj, "seams_to_fur_placement_mode")
        _draw_preview_row(settings_box, obj, "DISTORTION", "SHADING_RENDERED", "Distortion")

        # The ramp shown here is the actual node driving the preview
        # material (see distortion.get_or_create_distortion_material) -
        # not a separately hand-drawn approximation - so it can't drift
        # out of sync with what's actually rendered. Uses the read-only
        # getter - draw() must never create/mutate the material itself
        # (see that function's docstring for why) - so the legend simply
        # stays hidden until a Distortion refresh has actually built it.
        _distortion_mat, _distortion_ramp = distortion.get_distortion_material_if_exists()
        if _distortion_ramp is not None:
            legend_col = settings_box.column(align=True)
            legend_col.template_color_ramp(_distortion_ramp, "color_ramp", expand=False)
            legend_row = legend_col.row(align=True)
            limit_pct = distortion.DISTORTION_STRETCH_LIMIT * 100.0
            legend_row.label(text=f"-{limit_pct:.0f}% bunched")
            legend_row.label(text="0%")
            legend_row.label(text=f"+{limit_pct:.0f}% stretched")

        _draw_preview_row(settings_box, obj, "CUT", "MOD_EDGESPLIT", "Cut")
        _draw_preview_row(settings_box, obj, "SLICED", "MOD_EXPLODE", "Sliced")

        col = layout.column()
        col.label(text="Pieces:")
        col.template_list(
            "SEAMS_TO_FUR_UL_pieces",
            "",
            obj,
            "seams_to_fur_pieces",
            obj,
            "seams_to_fur_active_piece_index",
            rows=4,
        )

        if 0 <= obj.seams_to_fur_active_piece_index < len(obj.seams_to_fur_pieces):
            active_piece = obj.seams_to_fur_pieces[obj.seams_to_fur_active_piece_index]
            if active_piece.error_message:
                err_box = layout.box()
                err_box.alert = True
                col = err_box.column(align=True)
                col.label(text=f"'{active_piece.name}' failed to flatten:", icon="ERROR")
                for line in active_piece.error_message.split("\n"):
                    if line.strip():
                        col.label(text=line.strip())

        row = layout.row(align=True)
        row.operator("seams_to_fur.flatten_piece", icon="FILE_REFRESH")
        row.operator("seams_to_fur.reset_placement", icon="LOOP_BACK")

        layout.operator("seams_to_fur.flatten_all", icon="MOD_TRIANGULATE")
        layout.separator()
        export_row = layout.row(align=True)
        export_row.operator("seams_to_fur.export_svg", icon="EXPORT")
        export_row.operator("seams_to_fur.export_dxf", icon="EXPORT")


class SEAMS_TO_FUR_PT_appearance(Panel):
    bl_label = "Appearance (Color / Fur)"
    bl_idname = "SEAMS_TO_FUR_PT_appearance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Seams to Fur"
    bl_parent_id = "SEAMS_TO_FUR_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        layout = self.layout
        obj = target.resolve_mesh_obj(context)
        if obj is None:
            return

        color_row = layout.row(align=True)
        color_row.operator("seams_to_fur.sync_piece_colors", icon="MATERIAL")
        color_row.prop(obj, "seams_to_fur_auto_color", icon="COLOR", text="Auto", toggle=True)

        fur_colors_header = layout.row(align=True)
        fur_colors_header.prop(
            obj,
            "seams_to_fur_show_fur_colors",
            icon="TRIA_DOWN" if obj.seams_to_fur_show_fur_colors else "TRIA_RIGHT",
            text="Fur Colors",
            emboss=False,
        )
        if obj.seams_to_fur_show_fur_colors:
            fur_colors_box = layout.box()
            fur_colors_box.label(text="Select piece(s) below/in viewport, then click a swatch", icon="INFO")
            pcoll = _get_fur_color_previews()
            grid = fur_colors_box.grid_flow(row_major=True, columns=3, even_columns=True, align=True)
            for name, color in FUR_COLOR_SWATCHES:
                op = grid.operator("seams_to_fur.apply_swatch_color", text=name, icon_value=pcoll[name].icon_id)
                op.color = color

        from ..operators import appearance as appearance_ops

        grain_row = layout.row(align=True)
        grain_row.operator("seams_to_fur.set_grain_direction", icon="EMPTY_SINGLE_ARROW")
        grain_row.prop(obj, "seams_to_fur_grain_mirror_edit", icon="MOD_MIRROR", text="", toggle=True)

        arrows_visible_sub = grain_row.row(align=True)
        arrows_visible_sub.enabled = appearance_ops.grain_arrows_exist()
        arrows_visible_sub.operator(
            "seams_to_fur.toggle_grain_arrows",
            icon="HIDE_OFF" if appearance_ops.grain_arrows_visible() else "HIDE_ON",
            text="",
        )
        layout.label(text="Click a piece in the Sliced Preview, then drag", icon="INFO")

        reset_row = layout.row(align=True)
        reset_row.operator("seams_to_fur.reset_grain_direction", icon="X", text="Reset")
        reset_row.operator("seams_to_fur.reset_all_grain_directions", icon="X", text="Reset All")

        if 0 <= obj.seams_to_fur_active_piece_index < len(obj.seams_to_fur_pieces):
            active_piece = obj.seams_to_fur_pieces[obj.seams_to_fur_active_piece_index]
            layout.prop(active_piece, "grain_direction", text="Grain Dir.")
            layout.prop(active_piece, "fur_length", text="Fur Length")
            length_row = layout.row(align=True)
            for label, inches in (('2"', 2.0), ('1.5"', 1.5), ('1"', 1.0), ('0.5"', 0.5), ('0.25"', 0.25)):
                length_op = length_row.operator("seams_to_fur.set_fur_length", text=label)
                length_op.length_m = inches * 0.0254

        layout.separator()
        fur_box = layout.box()
        fur_box.label(text="Fur / Hair Preview", icon="OUTLINER_OB_CURVES")

        fur_cached = appearance_ops.is_fur_cached(obj)
        fur_visible = appearance_ops.is_fur_visible(obj)
        fur_stale = appearance_ops.is_fur_stale(obj)

        fur_row = fur_box.row(align=True)
        refresh_sub = fur_row.row(align=True)
        refresh_sub.alert = fur_stale
        refresh_sub.operator("seams_to_fur.refresh_fur_preview", icon="PARTICLES")

        toggle_sub = fur_row.row(align=True)
        toggle_sub.enabled = fur_cached
        toggle_sub.operator(
            "seams_to_fur.toggle_fur_preview", icon="HIDE_OFF" if fur_visible else "HIDE_ON", text=""
        )

        clear_sub = fur_row.row(align=True)
        clear_sub.enabled = fur_cached
        clear_sub.operator("seams_to_fur.clear_fur_preview", icon="X", text="")


_classes = (
    SEAMS_TO_FUR_UL_pieces,
    SEAMS_TO_FUR_PT_main,
    SEAMS_TO_FUR_PT_appearance,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    _subscribe_msgbus()
    if _subscribe_msgbus not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_subscribe_msgbus)


def unregister():
    global _fur_color_previews
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _subscribe_msgbus in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_subscribe_msgbus)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
    if _fur_color_previews is not None:
        bpy.utils.previews.remove(_fur_color_previews)
        _fur_color_previews = None
