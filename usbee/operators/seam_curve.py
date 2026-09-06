import bmesh
import bpy
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector

from ..geometry import curve_display, seam_points

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"

CLOSE_LOOP_PIXEL_RADIUS = 15
SNAP_PIXEL_RADIUS = 20
SELECT_PIXEL_RADIUS = 15
# Ignore a click landing within this distance (in world units) of the
# previously placed point - a real click that close is almost always an
# accidental double-registration, not an intentional second point.
MIN_POINT_SPACING = 1e-5

SEAM_CURVE_COLOR = (1.0, 0.15, 0.05, 1.0)
PREVIEW_POINT_RADIUS = curve_display.TUBE_RADIUS * 4


def _get_or_create_collection(context):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(SEAM_CURVE_COLLECTION)
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


def _make_seam_curve_visible(curve_obj):
    """Draw the curve through occluding geometry, so it's never invisible
    just because it sits flush against (or slightly inside) a convex/
    concave part of the surface."""
    curve_obj.show_in_front = True
    curve_obj.color = SEAM_CURVE_COLOR
    if curve_obj.data.materials:
        curve_obj.data.materials[0] = _get_or_create_seam_curve_material()
    else:
        curve_obj.data.materials.append(_get_or_create_seam_curve_material())


def _mirror_plane(mesh_obj):
    """Returns (plane_point_world, plane_normal_world, has_real_mirror) for
    the mesh's Mirror modifier symmetry plane, or the world Y-Z plane (X=0)
    with has_real_mirror=False if it has none."""
    for mod in mesh_obj.modifiers:
        if mod.type == "MIRROR":
            axis_index = next((i for i in range(3) if mod.use_axis[i]), 0)
            pivot = mod.mirror_object or mesh_obj
            normal_local = Vector((1.0, 0.0, 0.0)) if axis_index == 0 else (
                Vector((0.0, 1.0, 0.0)) if axis_index == 1 else Vector((0.0, 0.0, 1.0))
            )
            normal_world = (pivot.matrix_world.to_3x3() @ normal_local).normalized()
            return pivot.matrix_world.translation.copy(), normal_world, True
    return Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), False


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


def _other_seam_curve_segments(mesh_obj, exclude_curve_obj):
    """World-space (a, b) segments from every *other* seam curve on this
    mesh, so a new/edited curve can snap onto them (e.g. to make two darts
    meet exactly, or continue an existing cut)."""
    segments = []
    for obj in _find_seam_curves_for(mesh_obj):
        if obj is exclude_curve_obj:
            continue
        points, closed = seam_points.load_points(obj)
        segments.extend(zip(points, points[1:]))
        if closed and len(points) > 2:
            segments.append((points[-1], points[0]))
    return segments


def _bias_for(mesh_obj):
    segment_length = curve_display.segment_length_for(mesh_obj)
    _plane_point, plane_normal, has_real_mirror = _mirror_plane(mesh_obj)
    return curve_display.mirror_bias(mesh_obj, segment_length, plane_normal, has_real_mirror)


def create_seam_curve_object(context, mesh_obj, points, closed, name="SeamCurve"):
    """Creates a new seam curve object (a plain Mesh - see
    geometry.curve_display's module docstring for why) from a world-space
    point list, stores the points, and builds its initial display."""
    mesh_data = bpy.data.meshes.new(name)
    curve_obj = bpy.data.objects.new(name, mesh_data)
    curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name
    seam_points.store_points(curve_obj, points, closed)

    _get_or_create_collection(context).objects.link(curve_obj)
    _make_seam_curve_visible(curve_obj)

    depsgraph = context.evaluated_depsgraph_get()
    bvh = curve_display.build_surface_bvh(mesh_obj, depsgraph)
    curve_display.rebuild_display(mesh_data, points, closed, mesh_obj, bvh, bias_world=_bias_for(mesh_obj))

    for entry in mesh_obj.usbee_pieces:
        entry.flatten_dirty = True

    return curve_obj


class USBEE_OT_draw_seam_curve(Operator):
    """Click on the mesh to place seam points directly on its surface.
    Right-click or Enter finishes an open cut (e.g. a dart); clicking back
    near the first point closes it into a loop. Esc cancels. Clicks near a
    mesh boundary edge, another seam curve, or the model's mirror/symmetry
    plane snap to it."""

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

        other_seam_segments = _other_seam_curve_segments(self.mesh_obj, exclude_curve_obj=None)
        self.snap_segments = boundary_segments + other_seam_segments

        self.mirror_plane_point, self.mirror_plane_normal, self.has_real_mirror = _mirror_plane(self.mesh_obj)
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
            self.curve_obj.data, self.points, False, self.mesh_obj, self.bvh, bias_world=self.bias
        )

    def _add_point(self, world_co, context):
        self.points.append(world_co.copy())
        self.last_world_co = world_co.copy()
        self._rebuild_live()
        self._update_header(context.area)

    def _update_header(self, context_area):
        context_area.header_text_set(
            f"{len(self.points)} point(s) placed | Click: add point | "
            f"Click near start: close loop | Enter/Right-click: finish open cut | Esc: cancel"
        )

    def _finish(self, context, cyclic):
        self._remove_preview_object()
        seam_points.store_points(self.curve_obj, self.points, cyclic)
        curve_display.rebuild_display(
            self.curve_obj.data, self.points, cyclic, self.mesh_obj, self.bvh, bias_world=self.bias
        )
        for entry in self.mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        context.area.header_text_set(None)
        self.report(
            {"INFO"}, f"Seam curve with {len(self.points)} point(s) ({'closed' if cyclic else 'open'})"
        )

    def _cancel(self, context):
        self._remove_preview_object()
        context.area.header_text_set(None)
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
        _get_or_create_collection(context).objects.link(self.curve_obj)
        _make_seam_curve_visible(self.curve_obj)

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


class USBEE_OT_edit_seam_curve(Operator):
    """Click an existing point to drag it; click a spot along the curve to
    insert a new point there and drag it into place; hover a point and
    press X or Delete to remove it. Enter/right-click finishes, Esc cancels
    and reverts. The same boundary-edge/other-seam/mirror-plane snapping as
    drawing applies while dragging."""

    bl_idname = "usbee.edit_seam_curve"
    bl_label = "Edit Seam Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            context.area is not None
            and context.area.type == "VIEW_3D"
            and obj is not None
            and obj.get(SOURCE_OBJECT_PROP) is not None
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

        other_seam_segments = _other_seam_curve_segments(self.mesh_obj, exclude_curve_obj=self.curve_obj)
        self.snap_segments = boundary_segments + other_seam_segments

        self.mirror_plane_point, self.mirror_plane_normal, self.has_real_mirror = _mirror_plane(self.mesh_obj)
        self.bias = _bias_for(self.mesh_obj)

    # -- shared with USBEE_OT_draw_seam_curve; kept local to avoid coupling
    # two operator classes together over private helper methods.
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

    def _screen_dist(self, context, event, world_co):
        pos_2d = view3d_utils.location_3d_to_region_2d(context.region, context.region_data, world_co)
        if pos_2d is None:
            return None
        dx = event.mouse_region_x - pos_2d.x
        dy = event.mouse_region_y - pos_2d.y
        return (dx * dx + dy * dy) ** 0.5

    def _within_radius(self, context, event, world_co, radius):
        dist = self._screen_dist(context, event, world_co)
        return dist is not None and dist <= radius

    def _compute_placement(self, context, event):
        world_hit = self._raycast(context, event)
        if world_hit is None:
            return None

        best_seg = None
        best_dist_sq = None
        for a, b in self.snap_segments:
            candidate = _nearest_point_on_segment(world_hit, a, b)
            d = (candidate - world_hit).length_squared
            if best_dist_sq is None or d < best_dist_sq:
                best_dist_sq = d
                best_seg = (a, b, candidate)

        if best_seg is not None:
            a, b, seg_point = best_seg
            combined = _intersect_segment_plane(a, b, self.mirror_plane_point, self.mirror_plane_normal)
            if combined is not None and self._within_radius(context, event, combined, SNAP_PIXEL_RADIUS):
                return combined
            if self._within_radius(context, event, seg_point, SNAP_PIXEL_RADIUS):
                return seg_point

        normal = self.mirror_plane_normal
        plane_point = self.mirror_plane_point
        projected = world_hit - normal * (world_hit - plane_point).dot(normal)
        if self._within_radius(context, event, projected, SNAP_PIXEL_RADIUS):
            return projected

        return world_hit

    def _nearest_point_index(self, context, event):
        best_index = None
        best_dist = None
        for i, p in enumerate(self.points):
            dist = self._screen_dist(context, event, p)
            if dist is not None and dist <= SELECT_PIXEL_RADIUS and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_index = i
        return best_index

    def _rebuild_live(self):
        curve_display.rebuild_display(
            self.curve_obj.data, self.points, self.closed, self.mesh_obj, self.bvh, bias_world=self.bias
        )

    def invoke(self, context, event):
        self.curve_obj = context.active_object
        self.mesh_obj = bpy.data.objects.get(self.curve_obj[SOURCE_OBJECT_PROP])
        if self.mesh_obj is None:
            self.report({"ERROR"}, "Source mesh for this seam curve no longer exists")
            return {"CANCELLED"}

        self.points, self.closed = seam_points.load_points(self.curve_obj)
        self.original_points = [p.copy() for p in self.points]
        self.original_closed = self.closed
        self._build_evaluated_geometry(context)

        self.dragging_index = None
        self.hover_index = None

        context.area.header_text_set(
            "Click a point: drag it | Click the curve: insert a point | "
            "X/Delete: remove hovered point | Enter/Right-click: finish | Esc: cancel"
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _finish_editing(self, context):
        context.area.header_text_set(None)

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            if self.dragging_index is not None:
                placement = self._compute_placement(context, event)
                if placement is not None:
                    self.points[self.dragging_index] = placement
                    self._rebuild_live()
            else:
                self.hover_index = self._nearest_point_index(context, event)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit_index = self._nearest_point_index(context, event)
            if hit_index is not None:
                self.dragging_index = hit_index
                return {"RUNNING_MODAL"}

            # No existing point under the cursor - try inserting a new one
            # along this curve's own segments.
            own_segments = list(zip(self.points, self.points[1:]))
            if self.closed and len(self.points) > 2:
                own_segments.append((self.points[-1], self.points[0]))
            best_index = None
            best_point = None
            best_dist = None
            world_hit = self._raycast(context, event)
            for i, (a, b) in enumerate(own_segments):
                if world_hit is None:
                    continue
                # Nearest point on this segment to the raw raycast hit, not
                # the snapped placement, so insertion position is
                # independent of snap state.
                candidate = _nearest_point_on_segment(world_hit, a, b)
                dist = self._screen_dist(context, event, candidate)
                if dist is not None and dist <= SNAP_PIXEL_RADIUS and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_index = i
                    best_point = candidate
            if best_index is not None:
                insert_at = best_index + 1
                self.points.insert(insert_at, best_point)
                self.dragging_index = insert_at
                self._rebuild_live()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self.dragging_index = None
            return {"RUNNING_MODAL"}

        if event.type in {"X", "DEL"} and event.value == "PRESS":
            if self.hover_index is not None and len(self.points) > 2:
                del self.points[self.hover_index]
                self.hover_index = None
                self._rebuild_live()
            return {"RUNNING_MODAL"}

        if event.type in {"RIGHTMOUSE", "RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            if len(self.points) < 2:
                self.report({"WARNING"}, "Need at least 2 points - reverted")
                self.points = self.original_points
                self.closed = self.original_closed
                self._rebuild_live()
                self._finish_editing(context)
                return {"CANCELLED"}
            seam_points.store_points(self.curve_obj, self.points, self.closed)
            self._rebuild_live()
            for entry in self.mesh_obj.usbee_pieces:
                entry.flatten_dirty = True
            self._finish_editing(context)
            self.report({"INFO"}, f"Seam curve updated: {len(self.points)} point(s)")
            return {"FINISHED"}

        if event.type == "ESC":
            self.points = self.original_points
            self.closed = self.original_closed
            self._rebuild_live()
            self._finish_editing(context)
            return {"CANCELLED"}

        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"} or event.alt:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


class USBEE_OT_mirror_seam_curve(Operator):
    """Duplicates the active seam curve, reflecting its points across the
    source mesh's Mirror-modifier symmetry plane (or the world Y-Z plane if
    it has none) - for drawing a dart/seam once and getting the matching
    one on the other side, without relying on a Mirror modifier on the
    curve itself (which doesn't reliably preserve curve-domain data)."""

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

        points, closed = seam_points.load_points(curve_obj)
        plane_point, plane_normal, has_real_mirror = _mirror_plane(mesh_obj)
        if not has_real_mirror:
            self.report(
                {"WARNING"}, f"'{mesh_obj.name}' has no Mirror modifier - mirroring across world X=0 instead"
            )

        mirrored_points = curve_display.mirror_points(points, plane_point, plane_normal)
        new_curve = create_seam_curve_object(context, mesh_obj, mirrored_points, closed, name=f"{curve_obj.name}.mirror")

        context.view_layer.objects.active = new_curve
        for obj in context.selected_objects:
            obj.select_set(False)
        new_curve.select_set(True)

        self.report({"INFO"}, f"Created mirrored seam curve '{new_curve.name}'")
        return {"FINISHED"}


_classes = (
    USBEE_OT_draw_seam_curve,
    USBEE_OT_edit_seam_curve,
    USBEE_OT_mirror_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
