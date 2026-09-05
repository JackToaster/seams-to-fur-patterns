import bmesh
import bpy
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from ..geometry import curve_display

SEAM_CURVE_COLLECTION = "USBee Seam Curves"
SOURCE_OBJECT_PROP = "usbee_seam_source_object"

CLOSE_LOOP_PIXEL_RADIUS = 15
SNAP_PIXEL_RADIUS = 20
# Ignore a click landing within this distance (in mesh-local units) of the
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
    concave part of the surface. Actual on-screen thickness comes from the
    "USBee Surface Follow" geometry-nodes modifier (see curve_display),
    not curve bevel - a curve's own bevel is baked into its base mesh
    *before* modifiers run, which would make a Shrinkwrap-style surface
    projection squash the tube instead of gluing a thin centerline."""
    curve_obj.show_in_front = True
    curve_obj.color = SEAM_CURVE_COLOR
    mat = _get_or_create_seam_curve_material()
    if mat.name not in (m.name for m in curve_obj.data.materials if m):
        curve_obj.data.materials.append(mat)


def _segment_length_for(mesh_obj):
    return max(mesh_obj.dimensions) * 0.02 or 0.02


def _add_surface_follow(curve_obj, mesh_obj):
    """Wires up the display tube, including a Bias nudge toward the mirror
    plane's normal when the mesh actually has a Mirror modifier - see
    geometry.curve_display's module docstring for why: a point sitting
    exactly on a mirrored seam is equidistant from both mirrored halves, so
    "nearest surface" flips unpredictably between them from one resampled
    point to the next, producing a visibly squiggly tube. The nudge only
    makes sense for a *real* mirror plane; it's skipped for the world-Y-Z
    fallback used for click-snapping on non-mirrored meshes, since there's
    no actual symmetric-surface ambiguity to break there."""
    segment_length = _segment_length_for(mesh_obj)
    _plane_point, plane_normal, has_real_mirror = _mirror_plane(mesh_obj)
    bias_local = (0.0, 0.0, 0.0)
    if has_real_mirror:
        bias_world = plane_normal * (segment_length * 0.25)
        bias_local = curve_obj.matrix_world.to_3x3().inverted() @ bias_world
    return curve_display.add_or_update_modifier(curve_obj, mesh_obj, segment_length, bias_local)


def _mirror_plane(mesh_obj):
    """Returns (plane_point_world, plane_normal_world, has_real_mirror) for
    the mesh's Mirror modifier symmetry plane, or the world Y-Z plane (X=0)
    with has_real_mirror=False if it has none - see
    geometry.curve_display module docstring for why this matters for
    dart/seam placement on symmetric models."""
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


def _other_seam_curve_segments(context, mesh_obj, exclude_curve_obj):
    """World-space (a, b) segments from every *other* already-drawn seam
    curve on this mesh, so a new curve can snap onto them (e.g. to make two
    darts meet exactly, or to continue an existing cut)."""
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        return []
    segments = []
    for obj in coll.objects:
        if obj is exclude_curve_obj or obj.get(SOURCE_OBJECT_PROP) != mesh_obj.name:
            continue
        for spline in obj.data.splines:
            points = [obj.matrix_world @ p.co.to_3d() for p in spline.points]
            segments.extend(zip(points, points[1:]))
            if spline.use_cyclic_u and len(points) > 2:
                segments.append((points[-1], points[0]))
    return segments


class USBEE_OT_draw_seam_curve(Operator):
    """Click on the mesh to place seam points directly on its surface.
    Right-click or Enter finishes an open cut (e.g. a dart); clicking back
    near the first point closes it into a loop. Esc cancels. Clicks near a
    mesh boundary edge or the model's mirror/symmetry plane snap to it."""

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
        # Raycast/snap against the fully evaluated (post-modifier) surface -
        # e.g. a Subdivision Surface result - not the low-poly base cage, so
        # clicks land on the surface that will actually get flattened.
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = self.mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        self.bvh = BVHTree.FromBMesh(bm)
        mat = self.mesh_obj.matrix_world
        boundary_segments = [
            (mat @ e.verts[0].co, mat @ e.verts[1].co) for e in bm.edges if len(e.link_faces) == 1
        ]
        bm.free()
        eval_obj.to_mesh_clear()

        # Gathered once here (before self.curve_obj exists) so this curve
        # can never snap to itself - only to *other*, already-finished seam
        # curves on the same mesh (self-closing the loop is handled
        # separately by the close-loop-click check).
        other_seam_segments = _other_seam_curve_segments(context, self.mesh_obj, exclude_curve_obj=None)
        self.snap_segments = boundary_segments + other_seam_segments

        self.mirror_plane_point, self.mirror_plane_normal, self.has_real_mirror = _mirror_plane(self.mesh_obj)

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

    def _add_point(self, world_co, context):
        spline = self.curve_obj.data.splines[0]
        if self.num_points == 0:
            pass  # spline already has its one default point
        else:
            spline.points.add(1)
        spline.points[-1].co = (world_co.x, world_co.y, world_co.z, 1.0)
        self.num_points += 1
        self.last_world_co = world_co.copy()
        self._update_header(context.area)

    def _update_header(self, context_area):
        context_area.header_text_set(
            f"{self.num_points} point(s) placed | Click: add point | "
            f"Click near start: close loop | Enter/Right-click: finish open cut | Esc: cancel"
        )

    def _finish(self, context, cyclic):
        self._remove_preview_object()
        self.curve_obj.data.splines[0].use_cyclic_u = cyclic
        for entry in self.mesh_obj.usbee_pieces:
            entry.flatten_dirty = True
        context.area.header_text_set(None)
        self.report(
            {"INFO"}, f"Seam curve with {self.num_points} point(s) ({'closed' if cyclic else 'open'})"
        )

    def _cancel(self, context):
        self._remove_preview_object()
        context.area.header_text_set(None)
        bpy.data.objects.remove(self.curve_obj, do_unlink=True)

    def invoke(self, context, event):
        self.mesh_obj = context.active_object
        self.num_points = 0
        self.last_world_co = None
        self.preview_obj = None
        self._build_evaluated_geometry(context)

        curve_data = bpy.data.curves.new(name="SeamCurve", type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.splines.new(type="POLY")
        self.curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        self.curve_obj[SOURCE_OBJECT_PROP] = self.mesh_obj.name
        _get_or_create_collection(context).objects.link(self.curve_obj)
        _make_seam_curve_visible(self.curve_obj)
        _add_surface_follow(self.curve_obj, self.mesh_obj)

        self._update_header(context.area)
        self._update_preview(context, event)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "MOUSEMOVE":
            self._update_preview(context, event)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            if self.num_points >= 3:
                first_2d = view3d_utils.location_3d_to_region_2d(
                    context.region, context.region_data, self.curve_obj.data.splines[0].points[0].co.to_3d()
                )
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
            if self.num_points < 2:
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


class USBEE_OT_add_seam_curve(Operator):
    """Add a blank seam curve near the active mesh for manual shaping with
    Blender's normal curve tools (Tab into edit mode, extrude/move points) -
    an alternative to 'Draw Seam Curve' for precise manual control. A
    surface-follow modifier keeps it glued to the surface as you move
    points."""

    bl_idname = "usbee.add_seam_curve"
    bl_label = "Add Blank Seam Curve"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        mesh_obj = context.active_object

        curve_data = bpy.data.curves.new(name="SeamCurve", type="CURVE")
        curve_data.dimensions = "3D"
        spline = curve_data.splines.new(type="POLY")
        spline.points.add(1)  # POLY splines start with 1 point; add 1 more -> 2 total

        # Seed two points near the mesh's origin so there's something
        # immediately visible/selectable to edit; the surface-follow
        # modifier will pull them onto the surface once bound below.
        center = mesh_obj.matrix_world.translation
        spline.points[0].co = (center.x - 0.05, center.y, center.z, 1.0)
        spline.points[1].co = (center.x + 0.05, center.y, center.z, 1.0)

        curve_obj = bpy.data.objects.new("SeamCurve", curve_data)
        curve_obj[SOURCE_OBJECT_PROP] = mesh_obj.name

        coll = _get_or_create_collection(context)
        coll.objects.link(curve_obj)
        _make_seam_curve_visible(curve_obj)
        _add_surface_follow(curve_obj, mesh_obj)

        context.view_layer.objects.active = curve_obj
        for obj in context.selected_objects:
            obj.select_set(False)
        curve_obj.select_set(True)

        self.report({"INFO"}, f"Added seam curve for '{mesh_obj.name}' - Tab into edit mode to shape it")
        return {"FINISHED"}


_classes = (
    USBEE_OT_draw_seam_curve,
    USBEE_OT_add_seam_curve,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
