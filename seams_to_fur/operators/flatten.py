import concurrent.futures
import math

import bmesh
import bpy
from bpy.types import Operator
from mathutils import Vector

from ..backends import bff
from ..geometry import (
    boundary,
    curve_display,
    island_cache,
    islands,
    layout,
    polygon_offset,
)
from .seam_curve import SEAM_CURVE_COLLECTION, SOURCE_OBJECT_PROP
from . import target

PATTERN_COLLECTION_PREFIX = "STF Pattern — "


def _addon_dir():
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_seam_curves_for(mesh_obj):
    coll = bpy.data.collections.get(SEAM_CURVE_COLLECTION)
    if coll is None:
        return []
    return [o for o in coll.objects if o.get(SOURCE_OBJECT_PROP) == mesh_obj.name]


def _face_area_3d(verts, face):
    # Fan-triangulate from vertex 0 for area, fine for the small convex-ish
    # faces produced by ordinary modeling; good enough for the global scale
    # correction below (not used for the flattening itself).
    if len(face) < 3:
        return 0.0
    from mathutils import Vector

    area = 0.0
    v0 = Vector(verts[face[0]])
    for i in range(1, len(face) - 1):
        v1 = Vector(verts[face[i]])
        v2 = Vector(verts[face[i + 1]])
        area += (v1 - v0).cross(v2 - v0).length / 2.0
    return area


def _polygon_area_2d(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _pattern_collection(context, mesh_obj):
    from . import collections

    name = f"{PATTERN_COLLECTION_PREFIX}{mesh_obj.name}"
    return collections.get_or_create_child_collection(context, name)


def _build_or_update_flat_object(context, mesh_obj, piece, verts2d, faces, grid_offset_xy, orig_verts_3d=None):
    """Creates the flat object on first bake, placed per
    mesh_obj.seams_to_fur_placement_mode ('GRID': automatic shelf layout so pieces
    don't overlap, for exporting/printing; 'ORIGIN': the piece's own center
    placed at the world-space position its curved piece's center occupied
    in 3D, for an "exploded view" comparison against the model). On re-bake
    of an existing object, the shape is updated in place without
    re-applying placement - this is what makes a manual reposition survive
    a re-bake, without needing depsgraph-based move detection.

    orig_verts_3d, if given (same length/order as verts2d - the caller is
    responsible for reconciling this against BFF's own vertex numbering,
    which does NOT always match its input 1:1 - see _flatten_pieces), is
    stamped onto the flat mesh as a per-vertex "seams_to_fur_orig_co" attribute: the point's
    original pre-flatten position (mesh_obj's local space, pre-cut/curved
    surface), so a debugging/investigation workflow can always recover
    exactly where a given flattened vertex came from on the real mesh -
    even after the user has moved/distorted the flattened piece - without
    needing to re-run the (nondeterministic-in-ordering, drift-prone
    between separate calls) cut+flatten pipeline again to reconstruct it."""
    pattern_coll = _pattern_collection(context, mesh_obj)

    flat_mesh_name = f"{mesh_obj.name}.{piece.name}"
    existing_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None

    placement_mode = getattr(mesh_obj, "seams_to_fur_placement_mode", "GRID")

    if existing_obj is not None and existing_obj.type == "MESH":
        placed = [(x, y, 0.0) for (x, y, _z) in verts2d]
        flat_mesh = existing_obj.data
        flat_mesh.clear_geometry()
        flat_obj = existing_obj
    else:
        flat_mesh = bpy.data.meshes.new(flat_mesh_name)
        flat_obj = bpy.data.objects.new(flat_mesh_name, flat_mesh)
        pattern_coll.objects.link(flat_obj)

        if placement_mode == "ORIGIN":
            cx = sum(x for x, _y, _z in verts2d) / len(verts2d)
            cy = sum(y for _x, y, _z in verts2d) / len(verts2d)
            placed = [(x - cx, y - cy, 0.0) for (x, y, _z) in verts2d]
            flat_obj.location = mesh_obj.matrix_world @ Vector(piece.sample_point)
        else:
            ox, oy = grid_offset_xy
            placed = [(x + ox, y + oy, 0.0) for (x, y, _z) in verts2d]

    flat_mesh.from_pydata(placed, [], faces)
    flat_mesh.update()

    if orig_verts_3d is not None and len(orig_verts_3d) == len(placed):
        attr = flat_mesh.attributes.get("seams_to_fur_orig_co")
        if attr is None or attr.domain != "POINT" or attr.data_type != "FLOAT_VECTOR":
            if attr is not None:
                flat_mesh.attributes.remove(attr)
            attr = flat_mesh.attributes.new("seams_to_fur_orig_co", "FLOAT_VECTOR", "POINT")
        attr.data.foreach_set("vector", [c for co in orig_verts_3d for c in co])

    flat_obj["seams_to_fur_source_object"] = mesh_obj.name
    flat_obj["seams_to_fur_piece_uuid"] = piece.uuid

    piece.flattened_object = flat_obj.name
    return flat_obj, [(p[0], p[1]) for p in placed]


def _update_cut_line(context, mesh_obj, piece, flat_obj, boundary_loop_2d):
    if piece.offset_mm == 0.0:
        if piece.cut_line_object:
            old = bpy.data.objects.get(piece.cut_line_object)
            if old is not None:
                bpy.data.objects.remove(old, do_unlink=True)
            piece.cut_line_object = ""
        return

    offset_m = piece.offset_mm / 1000.0
    try:
        offset_points = polygon_offset.offset_polygon(boundary_loop_2d, offset_m)
    except polygon_offset.OffsetError as exc:
        raise RuntimeError(f"Piece '{piece.name}': {exc}") from exc

    curve_name = f"{flat_obj.name}.cut_line"
    existing = bpy.data.objects.get(piece.cut_line_object) if piece.cut_line_object else None
    if existing is not None and existing.type == "CURVE":
        curve_data = existing.data
        curve_data.splines.clear()
        curve_obj = existing
    else:
        curve_data = bpy.data.curves.new(curve_name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_obj = bpy.data.objects.new(curve_name, curve_data)
        _pattern_collection(context, mesh_obj).objects.link(curve_obj)

    spline = curve_data.splines.new(type="POLY")
    spline.points.add(len(offset_points) - 1)
    for i, (x, y) in enumerate(offset_points):
        spline.points[i].co = (x, y, 0.0, 1.0)
    spline.use_cyclic_u = True

    curve_obj.parent = flat_obj
    piece.cut_line_object = curve_obj.name


def _evaluated_bmesh(context, mesh_obj):
    """Builds a BMesh from the object's fully evaluated (post-modifier)
    mesh, so e.g. a Subdivision Surface modifier's smoothed result gets
    flattened rather than the low-poly base cage. If seams_to_fur_thickness_mm is
    set, the surface is shelled outward along its normals by that amount
    first, so the flattened pattern accounts for material thickness (e.g.
    foam) instead of just the bare mesh surface."""
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    eval_obj.to_mesh_clear()

    # Triangulate up front: geometry.mesh_cut's face-walking cut assumes
    # each face is planar (it projects points into a 2D basis derived from
    # the face's own normal+one vertex) to test path/edge crossings - a
    # generative modifier stack (Subdivision Surface, geometry nodes) can
    # leave genuinely non-planar n-gons, where that 2D projection doesn't
    # faithfully represent the face's real shape and can miss a crossing
    # that's geometrically really there (confirmed on a real mesh: several
    # percent of seam segments silently failed to find any exit on such
    # faces). Triangles are always exactly planar, removing the ambiguity
    # at the source rather than working around it in the cutter.
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()

    thickness_mm = getattr(mesh_obj, "seams_to_fur_thickness_mm", 0.0)
    if thickness_mm:
        scale_length = context.scene.unit_settings.scale_length or 1.0
        thickness_units = (thickness_mm / 1000.0) / scale_length
        bm.normal_update()
        for v in bm.verts:
            v.co += v.normal * thickness_units

    return bm


def compute_islands(context, mesh_obj, seam_curves=None):
    """Run the full expensive pipeline - build the evaluated+triangulated
    BMesh, resolve every seam curve into a real mesh cut, flood-fill into
    islands (folding away sliver islands), and reconcile
    mesh_obj.seams_to_fur_pieces with the islands found - returning ``(bm,
    face_island, seam_edges, island_count)``.

    Shared verbatim by flatten, distortion, and (on an island_cache miss)
    the color sync/preview operators, so all of them derive island
    membership identically - the numbering in face_island is what
    sync_piece_settings turns into piece_id, so it must not diverge between
    callers.

    Returns:
      - bm: the cut BMesh - the caller OWNS it and must bm.free() it.
      - face_island: dict face_index -> island_id (== piece_id).
      - seam_edges: set of bm edge indices marking the resolved cut.
      - island_count: number of islands (pieces).
    """
    if seam_curves is None:
        seam_curves = _find_seam_curves_for(mesh_obj)

    bm = _evaluated_bmesh(context, mesh_obj)

    # resolve_curve_seam_edges returns actual BMEdge objects (not indices -
    # those go stale the moment anything else touches bm, and this function
    # calls it once per curve, mutating bm each time). BMEdge references
    # themselves stay valid handles across those later mutations even
    # though their .index does; ensure_lookup_table() + is_valid right
    # before use (below) is all that's needed to read a trustworthy .index
    # once every curve is done.
    seam_edge_objs = set()
    for curve_obj in seam_curves:
        seam_edge_objs |= islands.resolve_curve_seam_edges(bm, curve_obj, mesh_obj)

    bm.edges.ensure_lookup_table()
    seam_edges = {e.index for e in seam_edge_objs if e.is_valid}

    face_island, island_count = islands.isolate_islands(bm, seam_edges)
    if seam_curves:
        # Bodge for the thin sliver islands ribbon-based bisection can
        # occasionally leave on strongly curved surfaces (see
        # geometry.mesh_cut) - fold anything implausibly small back into
        # its real neighbor rather than treating it as its own piece.
        segment_length = curve_display.segment_length_for(mesh_obj)
        min_area = (segment_length * 0.25) ** 2
        seam_edges = islands.merge_small_islands(bm, face_island, seam_edges, min_area)
        island_count = len(set(face_island.values()))
    islands.sync_piece_settings(mesh_obj, bm, face_island, island_count)
    islands.populate_seam_partners(mesh_obj, bm, face_island, seam_edges)

    return bm, face_island, seam_edges, island_count


def _flatten_prepare(context, mesh_obj, piece_ids):
    """Stage 1 (must run on the main thread - builds/reads a live bmesh):
    the cut/flood-fill, plus per-piece topology validation and island
    splitting down to plain vertex/face lists. Returns a state dict to
    hand to _flatten_run_one (stage 2, safe to run on a background
    thread per piece - see that function) and then _flatten_finish
    (stage 3, main thread again)."""
    seam_curves = _find_seam_curves_for(mesh_obj)
    bm, face_island, seam_edges, island_count = compute_islands(context, mesh_obj, seam_curves)

    # Feed the appearance-side cache: after a bake, coloring by piece needs
    # exactly this mapping and shouldn't re-cut to get it.
    island_cache.store_from_islands(mesh_obj, seam_curves, bm, face_island)

    binary_path = bff.find_binary(_addon_dir())

    errors = []
    pending = []  # (piece_uuid, piece_name, verts_list, faces_local)

    for piece in mesh_obj.seams_to_fur_pieces:
        if piece_ids is not None and piece.piece_id not in piece_ids:
            continue
        if not piece.flatten_dirty and piece_ids is None:
            continue

        piece.error_message = ""

        island_face_indices = [i for i, isl in face_island.items() if isl == piece.piece_id]
        try:
            islands.validate_island_topology(bm, island_face_indices)
        except islands.TopologyError as exc:
            msg = str(exc)
            piece.error_message = msg
            errors.append(f"Piece '{piece.name}': {msg}")
            continue

        # Duplicates vertices along any seam edges internal to this island
        # (e.g. a dart cut) so the cut can actually open up when flattened,
        # rather than BFF seeing a fully-connected mesh with no cut at all.
        verts_list, faces_local = islands.split_island_for_flatten(
            bm, island_face_indices, seam_edges
        )
        pending.append((piece.uuid, piece.name, verts_list, faces_local))

    return {"bm": bm, "binary_path": binary_path, "pending": pending, "errors": errors}


def _flatten_run_one(binary_path, verts_list, faces_local):
    """Stage 2, one piece: the BFF subprocess call plus all the pure-Python
    post-processing around it (scale correction, boundary loop ordering,
    orig-vertex reconciliation). Touches no bpy/bmesh data at all - just
    plain lists/tuples in, plain tuples out - which is what makes this the
    one piece of the pipeline safe to run on a background thread (see
    SEAMS_TO_FUR_OT_flatten_all, which runs one of these per piece on a
    ThreadPoolExecutor instead of one at a time on the main thread).

    Returns (out_verts, out_faces, loop_indices, orig_verts_3d,
    boundary_loop_2d). Raises bff.BFFError or boundary.BoundaryError on
    failure - the caller is responsible for turning that into a
    piece.error_message (each of those touches bpy data, so must happen
    back on the main thread)."""
    out_verts, out_faces = bff.flatten_island(binary_path, verts_list, faces_local)

    area_3d = sum(_face_area_3d(verts_list, f) for f in faces_local)
    area_2d = sum(_polygon_area_2d([out_verts[i] for i in f]) for f in out_faces)
    scale = math.sqrt(area_3d / area_2d) if area_2d > 1e-12 else 1.0
    out_verts = [(x * scale, y * scale, 0.0) for (x, y, _z) in out_verts]

    loop_indices = boundary.ordered_boundary_loop(out_faces)
    boundary_loop_2d = [(out_verts[i][0], out_verts[i][1]) for i in loop_indices]

    # BFF's out_verts/out_faces are NOT guaranteed to be verts_list/
    # faces_local 1:1 by index - BFF's own internal mesh only keeps
    # face-referenced vertices, so any vertex in verts_list that no
    # face actually uses (confirmed: split_island_for_flatten's wedge
    # grouping can produce one) is silently dropped, shifting every
    # later index. Rebuild the correspondence by FACE STRUCTURE
    # instead (out_faces and faces_local have the same face count and
    # per-face corner order, just possibly-renumbered vertex indices),
    # so each debugging attribute value still lands on the right
    # vertex regardless of any such drop/renumbering.
    orig_by_out_index = {}
    for face_out, face_orig in zip(out_faces, faces_local):
        for out_i, orig_i in zip(face_out, face_orig):
            orig_by_out_index[out_i] = verts_list[orig_i]
    orig_verts_3d = [orig_by_out_index.get(i, (0.0, 0.0, 0.0)) for i in range(len(out_verts))]

    return out_verts, out_faces, loop_indices, orig_verts_3d, boundary_loop_2d


def _flatten_finish(context, mesh_obj, state, piece_results):
    """Stage 3 (main thread - builds real bpy objects): given
    piece_results (dict piece_uuid -> (stage-2 result tuple, error string),
    exactly one entry per item in state["pending"]), lays out and
    builds/updates every flat object + cut line, then frees state["bm"].
    Returns the accumulated error list."""
    errors = list(state["errors"])
    flat_results = {}  # piece_uuid -> (verts2d, faces, loop_indices, orig_verts_3d)
    boundaries_for_layout = []
    piece_by_uuid = {p.uuid: p for p in mesh_obj.seams_to_fur_pieces}

    for piece_uuid, piece_name, _verts_list, _faces_local in state["pending"]:
        piece = piece_by_uuid.get(piece_uuid)
        result, err = piece_results.get(piece_uuid, (None, "missing result"))
        if piece is None:
            continue
        if err is not None:
            piece.error_message = err
            errors.append(f"Piece '{piece_name}': {err}")
            continue
        out_verts, out_faces, loop_indices, orig_verts_3d, boundary_loop_2d = result
        flat_results[piece_uuid] = (out_verts, out_faces, loop_indices, orig_verts_3d)
        boundaries_for_layout.append((piece_uuid, boundary_loop_2d))

    placements = layout.shelf_layout(boundaries_for_layout)

    for piece in mesh_obj.seams_to_fur_pieces:
        if piece.uuid not in flat_results:
            continue
        out_verts, out_faces, loop_indices, verts_list = flat_results[piece.uuid]
        grid_offset_xy = placements.get(piece.uuid, (0.0, 0.0))

        flat_obj, placed_verts_2d = _build_or_update_flat_object(
            context, mesh_obj, piece, out_verts, out_faces, grid_offset_xy, orig_verts_3d=verts_list
        )
        placed_boundary = [placed_verts_2d[i] for i in loop_indices]

        try:
            _update_cut_line(context, mesh_obj, piece, flat_obj, placed_boundary)
        except RuntimeError as exc:
            piece.error_message = str(exc)
            errors.append(str(exc))

        piece.flatten_dirty = False

    state["bm"].free()
    return errors


def _flatten_pieces(context, mesh_obj, piece_ids):
    """Synchronous, single-call convenience wrapper - runs stages 1-3 back
    to back on the calling thread, identical in behavior/output to before
    this module grew a background-thread-capable path (see
    SEAMS_TO_FUR_OT_flatten_all). Still what flatten_piece/reset_placement
    (single-piece, already fast enough not to need it) use, and what
    flatten_all itself falls back to in a headless/background context (no
    window - so no event loop available to drive the async modal path)."""
    state = _flatten_prepare(context, mesh_obj, piece_ids)
    piece_results = {}
    for piece_uuid, _piece_name, verts_list, faces_local in state["pending"]:
        try:
            result = _flatten_run_one(state["binary_path"], verts_list, faces_local)
            piece_results[piece_uuid] = (result, None)
        except (bff.BFFError, boundary.BoundaryError) as exc:
            piece_results[piece_uuid] = (None, str(exc))
    return _flatten_finish(context, mesh_obj, state, piece_results)


class SEAMS_TO_FUR_OT_flatten_piece(Operator):
    """Re-flatten the active pattern piece"""

    bl_idname = "seams_to_fur.flatten_piece"
    bl_label = "Flatten Piece"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj.seams_to_fur_active_piece_index >= len(mesh_obj.seams_to_fur_pieces):
            self.report({"ERROR"}, "No active piece selected")
            return {"CANCELLED"}
        piece = mesh_obj.seams_to_fur_pieces[mesh_obj.seams_to_fur_active_piece_index]
        # _flatten_pieces re-syncs (clears + rebuilds) seams_to_fur_pieces, which
        # invalidates this reference - capture the values we need as plain
        # strings/ints now, don't touch `piece` again after the call.
        piece_id, piece_name = piece.piece_id, piece.name

        errors = _flatten_pieces(context, mesh_obj, {piece_id})
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Flattened '{piece_name}'")
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_flatten_all(Operator):
    """Re-flatten every piece that needs it (or all pieces, if none are dirty).

    Interactively (a real window is available), stage 2 of the pipeline -
    the per-piece BFF flattening, the one part that's genuinely independent
    per piece and touches no bpy/bmesh data at all (see _flatten_run_one) -
    runs on a background thread pool instead of one piece at a time on the
    main thread, and a modal timer drives waiting for it instead of
    blocking execute(), so Blender's UI (viewport redraws, other windows,
    the Esc key) stays responsive for however long that takes. Stage 1
    (compute_islands - the mesh cut/flood-fill) still runs synchronously up
    front regardless: it touches live bmesh/bpy data, which Blender's
    Python API isn't safe to touch from a background thread without a much
    larger refactor - so on a very heavy mesh there's still a real pause
    before the background part even starts.

    In a headless/background context (no window - e.g. the test suite),
    there's no event loop available to drive a modal timer at all, so this
    falls back to running every stage synchronously and returning
    immediately - identical in behavior/output to how this operator worked
    before it grew the background-thread path."""

    bl_idname = "seams_to_fur.flatten_all"
    bl_label = "Flatten All"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _executor = None
    _futures = None  # list[(piece_uuid, Future)]
    _state = None
    _mesh_obj = None

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)

        # bpy.app.background, not context.window is None - confirmed live
        # that -b/--background mode still has a (non-functional, no real
        # display) Window object in the default startup file, so
        # context.window is never actually None there; but there's no
        # windowing event loop running to ever call modal() in that mode
        # either, so a modal_handler_add()'d operator would just return
        # RUNNING_MODAL and hang forever - bpy.app.background is the
        # actual documented way to detect "no event loop available".
        if bpy.app.background or context.window is None:
            wm = context.window_manager
            wm.progress_begin(0, 1)
            try:
                errors = _flatten_pieces(context, mesh_obj, None)
            finally:
                wm.progress_end()
            return self._finish_report(mesh_obj, errors)

        self._mesh_obj = mesh_obj
        self._state = _flatten_prepare(context, mesh_obj, None)
        pending = self._state["pending"]

        if not pending:
            errors = _flatten_finish(context, mesh_obj, self._state, {})
            return self._finish_report(mesh_obj, errors)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(pending)))
        self._futures = [
            (
                piece_uuid,
                self._executor.submit(_flatten_run_one, self._state["binary_path"], verts_list, faces_local),
            )
            for piece_uuid, _name, verts_list, faces_local in pending
        ]

        wm = context.window_manager
        wm.progress_begin(0, len(self._futures))
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self._cleanup(context)
            self.report({"WARNING"}, "Flatten All cancelled")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        done_count = sum(1 for _uuid, future in self._futures if future.done())
        context.window_manager.progress_update(done_count)
        if done_count < len(self._futures):
            return {"PASS_THROUGH"}

        piece_results = {}
        for piece_uuid, future in self._futures:
            try:
                piece_results[piece_uuid] = (future.result(), None)
            except (bff.BFFError, boundary.BoundaryError) as exc:
                piece_results[piece_uuid] = (None, str(exc))

        errors = _flatten_finish(context, self._mesh_obj, self._state, piece_results)
        mesh_obj = self._mesh_obj
        self._cleanup(context)
        return self._finish_report(mesh_obj, errors)

    def cancel(self, context):
        self._cleanup(context)

    def _cleanup(self, context):
        context.window_manager.progress_end()
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._futures = None
        self._state = None

    def _finish_report(self, mesh_obj, errors):
        n_pieces = len(mesh_obj.seams_to_fur_pieces)
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            if len(errors) >= n_pieces:
                return {"CANCELLED"}
        self.report({"INFO"}, f"Flattened {n_pieces - len(errors)}/{n_pieces} pieces")
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_reset_placement(Operator):
    """Reset the active piece's flattened object back to the automatic grid layout"""

    bl_idname = "seams_to_fur.reset_placement"
    bl_label = "Reset Placement"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        piece = mesh_obj.seams_to_fur_pieces[mesh_obj.seams_to_fur_active_piece_index]

        # Deleting the existing flattened object (rather than just moving
        # it) is what makes _build_or_update_flat_object treat this as a
        # first bake and re-apply the automatic grid layout offset.
        old = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        piece.flattened_object = ""
        piece.flatten_dirty = True

        errors = _flatten_pieces(context, mesh_obj, {piece.piece_id})
        if errors:
            self.report({"ERROR"}, "; ".join(errors))
            return {"CANCELLED"}
        return {"FINISHED"}


_classes = (
    SEAMS_TO_FUR_OT_flatten_piece,
    SEAMS_TO_FUR_OT_flatten_all,
    SEAMS_TO_FUR_OT_reset_placement,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
