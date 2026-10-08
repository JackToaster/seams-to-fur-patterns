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
seams_to_fur.draw_seam_curve (it can raycast onto the target mesh's surface,
which native Edit Mode can't), then edit further with Tab + Edit Mode and
whatever modifiers you like - resolution and the display tube both read
the skeleton's *evaluated* geometry (see geometry.curve_display and
geometry.islands), so anything you do to it is automatically respected.
"""

import bmesh
import bpy
from bpy.types import Operator
from bpy_extras import view3d_utils

from ..geometry import curve_display

SEAM_CURVE_COLLECTION = "STF Seam Curves"
SOURCE_OBJECT_PROP = "seams_to_fur_seam_source_object"

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
TUBE_COLLECTION = "STF Seam Curve Tubes"


def _get_or_create_collection(context, name):
    from . import collections

    return collections.get_or_create_child_collection(context, name)


def _get_or_create_seam_curve_material():
    mat = bpy.data.materials.get("STF Seam Curve")
    if mat is None:
        mat = bpy.data.materials.new("STF Seam Curve")
        mat.diffuse_color = SEAM_CURVE_COLOR
        if mat.use_nodes:
            emission = mat.node_tree.nodes.get("Principled BSDF")
            if emission is not None:
                emission.inputs["Emission Color"].default_value = SEAM_CURVE_COLOR
                emission.inputs["Emission Strength"].default_value = 1.5
    return mat


def _source_mirror_modifier(mesh_obj):
    """The source mesh's own Mirror modifier, if any - used by
    seams_to_fur.mirror_seam_curve to copy its axis/pivot configuration."""
    return next((m for m in mesh_obj.modifiers if m.type == "MIRROR"), None)


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




def _tube_object_for(curve_obj):
    return bpy.data.objects.get(curve_obj.name + TUBE_SUFFIX)


# curve_obj.name -> the (verts, edges) signature the tube was last rebuilt
# from - session-only, not scene data (see refresh_display).
_last_skeleton_signature = {}


def refresh_display(context, curve_obj, force=False):
    """(Re)builds the visible tube companion object for a seam curve from
    its *current evaluated* geometry - call this after any edit (drawing,
    native Edit Mode changes, modifier changes).

    Skips the rebuild entirely if the curve's evaluated (verts, edges)
    haven't actually changed since the last rebuild (force=True bypasses
    this - see seams_to_fur.refresh_seam_display, the manual escape hatch
    for "the display hasn't caught up"). This isn't just an optimization:
    _refresh_display_handler calls this from depsgraph_update_post, and
    curve_display.rebuild_display always touches tube_obj.data (clear_
    geometry + from_pydata + mesh.update), which unconditionally marks
    that mesh ID as updated - feeding right back into the next depsgraph
    evaluation pass regardless of whether anything about the *curve*
    itself genuinely changed. Confirmed live: a single vertex selection
    in the seam curve's Edit Mode triggered 331 separate depsgraph
    evaluation passes and ~25 seconds of wall-clock time before this
    guard existed - Blender iterating a self-feeding update cascade, not
    doing 331 separate genuine edits. With the guard, the second and
    every subsequent pass in that cascade sees the identical skeleton
    signature, skips the rebuild, and the cascade collapses to the one
    real rebuild that was actually needed."""
    mesh_obj = bpy.data.objects.get(curve_obj.get(SOURCE_OBJECT_PROP))
    if mesh_obj is None:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    verts, edges = curve_display.evaluated_skeleton_geometry(curve_obj, depsgraph)

    # Gated on the tube actually existing too, not just the signature
    # matching - a stale cache entry (the tube was deleted - by a user, a
    # test's scene reset, or a different .blend file that happens to reuse
    # this curve's name - since the whole session-only signature is just a
    # name-keyed dict) must never make this skip creating a tube that
    # isn't there. With that check, a stale/wrong entry can only ever cost
    # one redundant rebuild, never leave the tube missing - same
    # correctness guarantee island_cache.py's signature cache documents
    # for itself.
    signature = (tuple(tuple(round(c, 6) for c in v) for v in verts), tuple(edges))
    tube_obj = _tube_object_for(curve_obj)
    if not force and tube_obj is not None and _last_skeleton_signature.get(curve_obj.name) == signature:
        return tube_obj

    bvh = curve_display.build_surface_bvh(mesh_obj, depsgraph)
    bias = curve_display.bias_for(mesh_obj)

    if tube_obj is None:
        tube_data = bpy.data.meshes.new(curve_obj.name + TUBE_SUFFIX)
        tube_obj = bpy.data.objects.new(curve_obj.name + TUBE_SUFFIX, tube_data)
        tube_obj.show_in_front = True
        tube_obj.color = SEAM_CURVE_COLOR
        tube_data.materials.append(_get_or_create_seam_curve_material())
        _get_or_create_collection(context, TUBE_COLLECTION).objects.link(tube_obj)

    curve_display.rebuild_display(tube_obj.data, verts, edges, mesh_obj, bvh, bias_world=bias)
    _last_skeleton_signature[curve_obj.name] = signature
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

    for entry in mesh_obj.seams_to_fur_pieces:
        entry.flatten_dirty = True

    return curve_obj


class SEAMS_TO_FUR_OT_draw_seam_curve(Operator):
    """Click on the mesh to place seam points directly on its surface.
    Right-click or Enter finishes an open cut (e.g. a dart); clicking back
    near the first point closes it into a loop. Esc cancels. Clicks near a
    mesh boundary edge, another seam curve, or the model's mirror/symmetry
    plane snap to it. Once created, edit further with Tab + Edit Mode and
    modifiers - see this module's docstring."""

    bl_idname = "seams_to_fur.draw_seam_curve"
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

        (
            self.mirror_plane_point,
            self.mirror_plane_normal,
            self.has_real_mirror,
            _mirror_merge_threshold,
        ) = curve_display.mirror_plane_for(self.mesh_obj)
        self.bias = curve_display.bias_for(self.mesh_obj)

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
        mesh = bpy.data.meshes.new("STFSeamPointPreview")
        bmesh_tmp = bmesh.new()
        bmesh.ops.create_uvsphere(bmesh_tmp, u_segments=8, v_segments=6, radius=PREVIEW_POINT_RADIUS)
        bmesh_tmp.to_mesh(mesh)
        bmesh_tmp.free()
        self.preview_obj = bpy.data.objects.new("STFSeamPointPreview", mesh)
        self.preview_obj.show_in_front = True
        self.preview_obj.color = (0.1, 0.6, 1.0, 1.0)
        mat = bpy.data.materials.get("STF Seam Preview Point")
        if mat is None:
            mat = bpy.data.materials.new("STF Seam Preview Point")
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
        for entry in self.mesh_obj.seams_to_fur_pieces:
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


class SEAMS_TO_FUR_OT_refresh_seam_display(Operator):
    """Rebuilds the visible tube for the active seam curve from its current
    geometry - use after editing it in Edit Mode or changing its modifiers,
    if the display hasn't already caught up automatically."""

    bl_idname = "seams_to_fur.refresh_seam_display"
    bl_label = "Refresh Seam Display"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.get(SOURCE_OBJECT_PROP) is not None

    def execute(self, context):
        curve_obj = context.active_object
        tube_obj = refresh_display(context, curve_obj, force=True)
        if tube_obj is None:
            self.report({"ERROR"}, "Source mesh for this seam curve no longer exists")
            return {"CANCELLED"}
        mesh_obj = bpy.data.objects.get(curve_obj[SOURCE_OBJECT_PROP])
        for entry in mesh_obj.seams_to_fur_pieces:
            entry.flatten_dirty = True
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_mirror_seam_curve(Operator):
    """Adds a Mirror modifier to the active seam curve, configured to match
    the source mesh's own Mirror-modifier symmetry plane (or the world Y-Z
    plane if it has none) - so drawing a dart/seam once also gets you the
    matching one on the other side. This is a plain, ordinary Blender
    modifier on a Mesh object - reliable, unlike Mirror on a Curve - so you
    can also add/adjust it yourself via the regular Modifier Properties
    panel instead of this button."""

    bl_idname = "seams_to_fur.mirror_seam_curve"
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

        source_mirror_mod = _source_mirror_modifier(mesh_obj)
        mod = curve_obj.modifiers.new(name="Mirror", type="MIRROR")
        if source_mirror_mod is not None:
            mod.use_axis = tuple(source_mirror_mod.use_axis)
            mod.mirror_object = source_mirror_mod.mirror_object
        else:
            mod.use_axis = (True, False, False)
            self.report(
                {"WARNING"}, f"'{mesh_obj.name}' has no Mirror modifier - mirroring across world X=0 instead"
            )

        refresh_display(context, curve_obj)
        for entry in mesh_obj.seams_to_fur_pieces:
            entry.flatten_dirty = True
        return {"FINISHED"}


# Names of seam curve objects the depsgraph handler below has seen a
# relevant update for since the last deferred refresh ran - see
# _refresh_display_handler's docstring for why this hand-off exists.
_pending_refresh_names = set()
_refresh_timer_scheduled = False


def _run_deferred_refreshes():
    """bpy.app.timers callback - the actual evaluated-geometry-reading,
    tube-rebuilding work _refresh_display_handler used to do inline, now
    running on its own later tick instead of nested inside dependency
    graph evaluation (see that function's docstring for why). Runs once
    and does not reschedule itself (returning None, not a number, is what
    tells bpy.app.timers that)."""
    global _refresh_timer_scheduled
    _refresh_timer_scheduled = False
    names = list(_pending_refresh_names)
    _pending_refresh_names.clear()
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        try:
            refresh_display(bpy.context, obj)
        except Exception:
            pass
    return None


def _refresh_display_handler(_scene, depsgraph):
    """Keeps each seam curve's visible tube in sync automatically as the
    user edits the skeleton in Edit Mode or changes its modifiers, instead
    of requiring an explicit refresh every time.

    Deliberately does NOT call refresh_display (or read any evaluated
    depsgraph data at all) directly here - only records which curve(s)
    need refreshing and schedules a bpy.app.timers callback to actually do
    it. Blender's own depsgraph_update_post documentation warns that
    reading evaluated data (curve_obj.evaluated_get(depsgraph).to_mesh(),
    which evaluated_skeleton_geometry does) from inside this handler can
    itself trigger further dependency graph evaluation - confirmed live,
    the hard way: even after gating the actual tube *rebuild* on a
    signature cache (see refresh_display), a single vertex selection in
    a seam curve's Edit Mode still drove 662 recursive depsgraph_update_
    post passes and ~12 seconds of wall-clock time, from evaluated-
    geometry reads alone, with the tube rebuild itself never running past
    the first pass. Moving that evaluation to a timer tick - genuinely
    outside dependency graph evaluation, not just skipping redundant work
    within it - is what actually breaks the cascade (confirmed live: down
    to 1 depsgraph pass and native Edit Mode responsiveness).

    Never allowed to raise - a handler that crashes Blender's update
    cycle is far worse than a display that's briefly stale (use
    seams_to_fur.refresh_seam_display to force a rebuild if that ever
    happens)."""
    global _refresh_timer_scheduled
    for update in depsgraph.updates:
        obj = update.id
        if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
            continue
        if obj.get(SOURCE_OBJECT_PROP) is None:
            continue
        if not (update.is_updated_geometry or update.is_updated_transform):
            continue
        _pending_refresh_names.add(obj.name)

    if _pending_refresh_names and not _refresh_timer_scheduled:
        _refresh_timer_scheduled = True
        bpy.app.timers.register(_run_deferred_refreshes, first_interval=0.0)


_classes = (
    SEAMS_TO_FUR_OT_draw_seam_curve,
    SEAMS_TO_FUR_OT_refresh_seam_display,
    SEAMS_TO_FUR_OT_mirror_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    if _refresh_display_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_refresh_display_handler)


def unregister():
    if _refresh_display_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_refresh_display_handler)
    if bpy.app.timers.is_registered(_run_deferred_refreshes):
        bpy.app.timers.unregister(_run_deferred_refreshes)
    _pending_refresh_names.clear()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
