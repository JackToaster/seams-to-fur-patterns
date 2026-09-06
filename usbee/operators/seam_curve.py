"""Seam curve creation and display refresh.

A seam curve is a plain Mesh object - vertices and edges, no faces - that
the addon never adds modifiers to and never locks out of native Edit Mode.
Deliberately so: it was specifically *Curve* objects where a Mirror
modifier behaved unreliably (functionally rasterizing toward mesh output
instead of preserving curve-domain data), and a curve whose display came
from a modifier/Geometry-Nodes stack couldn't be edited via native curve
Edit Mode without the two fighting each other. Mirror/Array on a plain
Mesh is one of Blender's most standard, reliable modifier use cases, and
native mesh Edit Mode (move/extrude/merge/delete vertices) is exactly the
tool a user should be able to rely on. So: draw the initial points with
usbee.draw_seam_curve (it can raycast onto the target mesh's surface,
which native Edit Mode can't), then edit further with Tab + Edit Mode and
whatever modifiers you like - resolution and the display tube both read
the skeleton's *evaluated* geometry (see geometry.curve_display and
geometry.islands), so anything you do to it is automatically respected.
"""

import bmesh
import bpy
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector

from ..geometry import curve_display

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"

CLOSE_LOOP_PIXEL_RADIUS = 15
SNAP_PIXEL_RADIUS = 20
# Ignore a click landing within this distance (in world units) of the
# previously placed point - a real click that close is almost always an
# accidental double-registration, not an intentional second point.
MIN_POINT_SPACING = 1e-5

SEAM_CURVE_COLOR = (1.0, 0.15, 0.05, 1.0)
PREVIEW_POINT_RADIUS = curve_display.TUBE_RADIUS * 4

# Suffix distinguishing a skeleton's visible tube companion object, e.g.
# "SeamCurve" -> "SeamCurve.tube". Kept in its own (hidden-from-selection-
# by-convention) collection so it doesn't clutter the outliner next to the
# skeletons users actually edit.
TUBE_SUFFIX = ".tube"
TUBE_COLLECTION = "USBee Seam Curve Tubes"


def _get_or_create_collection(context, name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
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


def _mirror_plane(mesh_obj):
    """Returns (plane_point_world, plane_normal_world, has_real_mirror,
    mirror_modifier) for the mesh's Mirror modifier symmetry plane, or the
    world Y-Z plane (X=0) with has_real_mirror=False if it has none."""
    for mod in mesh_obj.modifiers:
        if mod.type == "MIRROR":
            axis_index = next((i for i in range(3) if mod.use_axis[i]), 0)
            pivot = mod.mirror_object or mesh_obj
            normal_local = Vector((1.0, 0.0, 0.0)) if axis_index == 0 else (
                Vector((0.0, 1.0, 0.0)) if axis_index == 1 else Vector((0.0, 0.0, 1.0))
            )
            normal_world = (pivot.matrix_world.to_3x3() @ normal_local).normalized()
            return pivot.matrix_world.translation.copy(), normal_world, True, mod
    return Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), False, None


def _nearest_point_on_segment(point, a, b):
    ab = b - a
    len_sq = ab.length_squared
    if len_sq < 1e-12:
        return a.copy()
    t = max(0.0, min(1.0, (point - a).dot(ab) / len_sq))
    return a + ab * t


def _intersect_segment_plane(a, b, plane_point, plane_normal):
    """Returns the point where segment a-b crosses the plane, or None if it
    doesn't cross within the segment's bounds (or runs parallel to it)."""
    d1 = (a - plane_point).dot(plane_normal)
    d2 = (b - plane_point).dot(plane_normal)
    if abs(d1 - d2) < 1e-9:
        return None
    t = d1 / (d1 - d2)
    if t < 0.0 or t > 1.0:
        return None
    return a + (b - a) * t


def _find_seam_curves_for(mesh_obj):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        return []
    return [o for o in coll.objects if o.get(SOURCE_OBJECT_PROP) == mesh_obj.name]


def _other_seam_curve_segments(context, mesh_obj, exclude_curve_obj):
    """World-space (a, b) segments from every *other* seam curve's
    evaluated geometry on this mesh, so a new curve can snap onto them
    (e.g. to make two darts meet exactly, or continue an existing cut)."""
    depsgraph = context.evaluated_depsgraph_get()
    segments = []
    for obj in _find_seam_curves_for(mesh_obj):
        if obj is exclude_curve_obj:
            continue
        verts, edges = curve_display.evaluated_skeleton_geometry(obj, depsgraph)
        segments.extend((verts[a], verts[b]) for a, b in edges)
    return segments


def _bias_for(mesh_obj):
    segment_length = curve_display.segment_length_for(mesh_obj)
    _plane_point, plane_normal, has_real_mirror, _mod = _mirror_plane(mesh_obj)
    return curve_display.mirror_bias(mesh_obj, segment_length, plane_normal, has_real_mirror)


def _tube_object_for(curve_obj):
    return bpy.data.objects.get(curve_obj.name + TUBE_SUFFIX)


def refresh_display(context, curve_obj):
    """(Re)builds the visible tube companion object for a seam curve from
    its *current evaluated* geometry - call this after any edit (drawing,
    native Edit Mode changes, modifier changes)."""
    mesh_obj = bpy.data.objects.get(curve_obj.get(SOURCE_OBJECT_PROP))
    if mesh_obj is None:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    verts, edges = curve_display.evaluated_skeleton_geometry(curve_obj, depsgraph)
    bvh = curve_display.build_surface_bvh(mesh_obj, depsgraph)
    bias = _bias_for(mesh_obj)

    tube_obj = _tube_object_for(curve_obj)
    if tube_obj is None:
        tube_data = bpy.data.meshes.new(curve_obj.name + TUBE_SUFFIX)
        tube_obj = bpy.data.objects.new(curve_obj.name + TUBE_SUFFIX, tube_data)
        tube_obj.show_in_front = True
        tube_obj.color = SEAM_CURVE_COLOR
        tube_data.materials.append(_get_or_create_seam_curve_material())
        _get_or_create_collection(context, TUBE_COLLECTION).objects.link(tube_obj)

    curve_display.rebuild_display(tube_obj.data, verts, edges, mesh_obj, bvh, bias_world=bias)
    return tube_obj


def create_seam_curve_object(context, mesh_obj, points, closed, name="SeamCurve"):
    """Creates a new seam curve skeleton object (plain Mesh, verts+edges,
    no faces) from a world-space point list, plus its initial display
    tube."""
    mesh_data = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bm_verts = [bm.verts.new(p) for p in points]
    for i in range(len(bm_verts) - 1):
        bm.edges.new((bm_verts[i], bm_verts[i + 1]))
    if closed and len(bm_verts) > 2:
        bm.edges.new((bm_verts[-1], bm_verts[0]))
    bm.to_mesh(mesh_data)
    bm.free()

    curve_obj = bpy.data.objects.new(name, mesh_data)
    curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name
    _get_or_create_collection(context, SEAM_CURVE_COLLECTION).objects.link(curve_obj)

    refresh_display(context, curve_obj)

    for entry in mesh_obj.usbee_pieces:
        entry.flatten_dirty = True

    return curve_obj


class USBEE_OT_draw_seam_curve(Operator):
    """Click on the mesh to place seam points directly on its surface.
    Right-click or Enter finishes an open cut (e.g. a dart); clicking back
    near the first point closes it into a loop. Esc cancels. Clicks near a
    mesh boundary edge, another seam curve, or the model's mirror/symmetry
    plane snap to it. Once created, edit further with Tab + Edit Mode and
    modifiers - see this module's docstring."""

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

    def _build_evaluated_geometry(self, context):
        depsgraph = context.evaluated_depsgraph_get()
        self.bvh = curve_display.build_surface_bvh(self.mesh_obj, depsgraph)

        eval_obj = self.mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        mat = self.mesh_obj.matrix_world
        boundary_segments = [
            (mat @ e.verts[0].co, mat @ e.verts[1].co) for e in bm.edges if len(e.link_faces) == 1
        ]
        bm.free()
        eval_obj.to_mesh_clear()

        other_seam_segments = _other_seam_curve_segments(context, self.mesh_obj, exclude_curve_obj=None)
        self.snap_segments = boundary_segments + other_seam_segments

        self.mirror_plane_point, self.mirror_plane_normal, self.has_real_mirror, _mod = _mirror_plane(
            self.mesh_obj
        )
        self.bias = _bias_for(self.mesh_obj)

    def _raycast(self, context, event):
        region = context.region
        rv3d = context.region_data
        coord = (event.mouse_region_x, event.mouse_region_y)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

        mat_inv = self.mesh_obj.matrix_world.inverted()
        local_origin = mat_inv @ ray_origin
        local_dir = (mat_inv.to_3x3() @ ray_dir).normalized()

        location, _normal, _face_index, _dist = self.bvh.ray_cast(local_origin, local_dir)
        if location is None:
            return None
        return self.mesh_obj.matrix_world @ location

    def _nearest_snap_segment(self, world_co):
        best_seg = None
        best_dist_sq = None
        for a, b in self.snap_segments:
            candidate = _nearest_point_on_segment(world_co, a, b)
            d = (candidate - world_co).length_squared
            if best_dist_sq is None or d < best_dist_sq:
                best_dist_sq = d
                best_seg = (a, b, candidate)
        return best_seg

    def _screen_dist(self, context, event, world_co):
        pos_2d = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, world_co)
        if pos_2d is None:
            return None
        dx = event.mouse_region_x - pos_2d.x
        dy = event.mouse_region_y - pos_2d.y
        return (dx * dx + dy * dy) ** 0.5

    def _within_snap_radius(self, context, event, world_co):
        dist = self._screen_dist(context, event, world_co)
        return dist is not None and dist <= SNAP_PIXEL_RADIUS

    def _compute_placement(self, context, event):
        """Returns a world-space Vector for where a point would land if
        clicked right now - or None if the ray misses the mesh entirely.

        Snap priority, all checked in screen-space against the cursor:
        1. Where the nearest snap-worthy segment (a mesh boundary edge, or
           another already-drawn seam curve) crosses the mirror/symmetry
           plane - for the common case of a dart or seam that should meet
           the model's centerline exactly at its silhouette.
        2. That segment's plain nearest point (no plane involved).
        3. The mirror/symmetry plane alone.
        4. The raw raycast hit, unsnapped.
        """
        world_hit = self._raycast(context, event)
        if world_hit is None:
            return None

        seg = self._nearest_snap_segment(world_hit)
        if seg is not None:
            a, b, seg_point = seg
            combined = _intersect_segment_plane(a, b, self.mirror_plane_point, self.mirror_plane_normal)
            if combined is not None and self._within_snap_radius(context, event, combined):
                return combined
            if self._within_snap_radius(context, event, seg_point):
                return seg_point

        normal = self.mirror_plane_normal
        plane_point = self.mirror_plane_point
        projected = world_hit - normal * (world_hit - plane_point).dot(normal)
        if self._within_snap_radius(context, event, projected):
            return projected

        return world_hit

    def _ensure_preview_object(self, context):
        if self.preview_obj is not None:
            return
        mesh = bpy.data.meshes.new("USBeeSeamPointPreview")
        bmesh_tmp = bmesh.new()
        bmesh.ops.create_uvsphere(bmesh_tmp, u_segments=8, v_segments=6, radius=PREVIEW_POINT_RADIUS)
        bmesh_tmp.to_mesh(mesh)
        bmesh_tmp.free()
        self.preview_obj = bpy.data.objects.new("USBeeSeamPointPreview", mesh)
        self.preview_obj.show_in_front = True
        self.preview_obj.color = (0.1, 0.6, 1.0, 1.0)
        mat = bpy.data.materials.get("USBee Seam Preview Point")
        if mat is None:
            mat = bpy.data.materials.new("USBee Seam Preview Point")
            mat.diffuse_color = (0.1, 0.6, 1.0, 1.0)
        mesh.materials.append(mat)
        context.scene.collection.objects.link(self.preview_obj)

    def _update_preview(self, context, event):
        placement = self._compute_placement(context, event)
        if placement is None:
            if self.preview_obj is not None:
                self.preview_obj.hide_viewport = True
            return
        self._ensure_preview_object(context)
        self.preview_obj.hide_viewport = False
        self.preview_obj.location = placement
        context.area.tag_redraw()

    def _remove_preview_object(self):
        if self.preview_obj is not None:
            bpy.data.objects.remove(self.preview_obj, do_unlink=True)
            self.preview_obj = None

    def _rebuild_live(self):
        curve_display.rebuild_display(
            self.tube_obj.data,
            self.points,
            list(zip(range(len(self.points) - 1), range(1, len(self.points)))),
            self.mesh_obj,
            self.bvh,
            bias_world=self.bias,
        )

    def _add_point(self, world_co, context):
        self.points.append(world_co.copy())
        self.last_world_co = world_co.copy()
        if len(self.points) >= 2:
            self._rebuild_live()
        self._update_header(context.area)

    def _update_header(self, context_area):
        context_area.header_text_set(
            f"{len(self.points)} point(s) placed | Click: add point | "
            f"Click near start: close loop | Enter/Right-click: finish open cut | Esc: cancel"
        )

    def _finish_skeleton(self, closed):
        bm = bmesh.new()
        bm_verts = [bm.verts.new(p) for p in self.points]
        for i in range(len(bm_verts) - 1):
            bm.edges.new((bm_verts[i], bm_verts[i + 1]))
        if closed and len(bm_verts) > 2:
            bm.edges.new((bm_verts[-1], bm_verts[0]))
        bm.to_mesh(self.curve_obj.data)
        bm.free()
        self.curve_obj.data.update()

    def _finish(self, context, cyclic):
        self._remove_preview_object()
        self._finish_skeleton(cyclic)
        refresh_display(context, self.curve_obj)
        for entry in self.mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        context.area.header_text_set(None)
        self.report(
            {"INFO"}, f"Seam curve with {len(self.points)} point(s) ({'closed' if cyclic else 'open'})"
        )

    def _cancel(self, context):
        self._remove_preview_object()
        context.area.header_text_set(None)
        tube_obj = _tube_object_for(self.curve_obj)
        if tube_obj is not None:
            bpy.data.objects.remove(tube_obj, do_unlink=True)
        bpy.data.objects.remove(self.curve_obj, do_unlink=True)

    def invoke(self, context, event):
        self.mesh_obj = context.active_object
        self.points = []
        self.last_world_co = None
        self.preview_obj = None
        self._build_evaluated_geometry(context)

        mesh_data = bpy.data.meshes.new("SeamCurve")
        self.curve_obj = bpy.data.objects.new("SeamCurve", mesh_data)
        self.curve_obj[SOURCE_OBJECT_PROP] = self.mesh_obj.name
        _get_or_create_collection(context, SEAM_CURVE_COLLECTION).objects.link(self.curve_obj)

        tube_data = bpy.data.meshes.new(self.curve_obj.name + TUBE_SUFFIX)
        self.tube_obj = bpy.data.objects.new(self.curve_obj.name + TUBE_SUFFIX, tube_data)
        self.tube_obj.show_in_front = True
        self.tube_obj.color = SEAM_CURVE_COLOR
        tube_data.materials.append(_get_or_create_seam_curve_material())
        _get_or_create_collection(context, TUBE_COLLECTION).objects.link(self.tube_obj)

        self._update_header(context.area)
        self._update_preview(context, event)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            self._update_preview(context, event)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if len(self.points) >= 3:
                first_2d = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, self.points[0])
                if first_2d is not None:
                    dx = event.mouse_region_x - first_2d.x
                    dy = event.mouse_region_y - first_2d.y
                    if (dx * dx + dy * dy) ** 0.5 <= CLOSE_LOOP_PIXEL_RADIUS:
                        self._finish(context, cyclic=True)
                        return {"FINISHED"}

            placement = self._compute_placement(context, event)
            if placement is not None:
                if self.last_world_co is None or (placement - self.last_world_co).length > MIN_POINT_SPACING:
                    self._add_point(placement, context)
            return {"RUNNING_MODAL"}

        if event.type in {"RIGHTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if len(self.points) < 2:
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


class USBEE_OT_refresh_seam_display(Operator):
    """Rebuilds the visible tube for the active seam curve from its current
    geometry - use after editing it in Edit Mode or changing its modifiers,
    if the display hasn't already caught up automatically."""

    bl_idname = "usbee.refresh_seam_display"
    bl_label = "Refresh Seam Display"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.get(SOURCE_OBJECT_PROP) is not None

    def execute(self, context):
        curve_obj = context.active_object
        tube_obj = refresh_display(context, curve_obj)
        if tube_obj is None:
            self.report({"ERROR"}, "Source mesh for this seam curve no longer exists")
            return {"CANCELLED"}
        mesh_obj = bpy.data.objects.get(curve_obj[SOURCE_OBJECT_PROP])
        for entry in mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        return {"FINISHED"}


class USBEE_OT_mirror_seam_curve(Operator):
    """Adds a Mirror modifier to the active seam curve, configured to match
    the source mesh's own Mirror-modifier symmetry plane (or the world Y-Z
    plane if it has none) - so drawing a dart/seam once also gets you the
    matching one on the other side. This is a plain, ordinary Blender
    modifier on a Mesh object - reliable, unlike Mirror on a Curve - so you
    can also add/adjust it yourself via the regular Modifier Properties
    panel instead of this button."""

    bl_idname = "usbee.mirror_seam_curve"
    bl_label = "Mirror Seam Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.get(SOURCE_OBJECT_PROP) is not None

    def execute(self, context):
        curve_obj = context.active_object
        mesh_obj = bpy.data.objects.get(curve_obj[SOURCE_OBJECT_PROP])
        if mesh_obj is None:
            self.report({"ERROR"}, "Source mesh for this seam curve no longer exists")
            return {"CANCELLED"}

        if any(m.type == "MIRROR" for m in curve_obj.modifiers):
            self.report({"INFO"}, f"'{curve_obj.name}' already has a Mirror modifier")
            return {"FINISHED"}

        _plane_point, _normal, has_real_mirror, source_mirror_mod = _mirror_plane(mesh_obj)
        mod = curve_obj.modifiers.new(name="Mirror", type="MIRROR")
        if has_real_mirror:
            mod.use_axis = tuple(source_mirror_mod.use_axis)
            mod.mirror_object = source_mirror_mod.mirror_object
        else:
            mod.use_axis = (True, False, False)
            self.report(
                {"WARNING"}, f"'{mesh_obj.name}' has no Mirror modifier - mirroring across world X=0 instead"
            )

        refresh_display(context, curve_obj)
        for entry in mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        return {"FINISHED"}


def _refresh_display_handler(_scene, depsgraph):
    """Keeps each seam curve's visible tube in sync automatically as the
    user edits the skeleton in Edit Mode or changes its modifiers, instead
    of requiring an explicit refresh every time. Never allowed to raise -
    a handler that crashes Blender's update cycle is far worse than a
    display that's briefly stale (use usbee.refresh_seam_display to force
    a rebuild if that ever happens)."""
    for update in depsgraph.updates:
        obj = update.id
        if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
            continue
        if obj.get(SOURCE_OBJECT_PROP) is None:
            continue
        if not (update.is_updated_geometry or update.is_updated_transform):
            continue
        try:
            refresh_display(bpy.context, obj)
        except Exception:
            pass


_classes = (
    USBEE_OT_draw_seam_curve,
    USBEE_OT_refresh_seam_display,
    USBEE_OT_mirror_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    if _refresh_display_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_refresh_display_handler)


def unregister():
    if _refresh_display_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_refresh_display_handler)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
