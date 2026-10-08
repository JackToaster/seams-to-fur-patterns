"""Phase 2: per-piece coloring, grain-direction picking, and a fur/hair
preview on the source (curved) mesh.

Per-piece coloring is applied *exactly* to the actual cut mesh (the
evaluated, post-modifier, actually-cut geometry - see operators/preview.py),
where each face's island membership is known precisely; that is the only
coloring target, correct even with a Mirror modifier. The base source mesh
is deliberately never colored: its topology doesn't follow the seam edges
(a face straddling a seam can't be assigned to just one piece there), and
under a Mirror modifier it only has one mirrored half's faces at all, so
any coloring on it would bleed across pieces or misrepresent mirrored ones.

Goes through geometry/island_cache.py so that re-syncing after only a color
change (not a seam/mesh edit) never re-cuts: the actual cut mesh's existing
face->material assignments are still correct as long as the cached
signature is unchanged, so only each piece's material color needs updating
in place (which, since materials are shared datablocks, instantly refreshes
every object already using them - cut preview, sliced previews, flattened
objects - with no rebuild).
"""

import bmesh
import colorsys
import math

import bpy
from bpy.props import FloatProperty, FloatVectorProperty, IntProperty
from bpy.types import Operator
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import tessellate_polygon

from ..geometry import curve_display, island_cache
from . import flatten as flatten_ops
from . import preview
from . import target
from .preview import get_or_create_piece_material as _get_or_create_piece_material

GRAIN_ARROW_COLLECTION = "STF Grain Directions"
GRAIN_ARROW_COLOR = (0.1, 0.6, 1.0, 1.0)

# Fixed regardless of arrow length ("length" here means the drag distance,
# which still scales head_length/shaft_length so a longer drag reads as a
# longer arrow) - a long arrow with a thickness scaled up to match used to
# look like a giant blunt cone, not a direction indicator. Head diameter is
# the user-specified 1cm; shaft diameter is 40% of that, a fairly standard
# arrow proportion, chosen since it wasn't specified.
GRAIN_ARROW_HEAD_DIAMETER = 0.01
GRAIN_ARROW_SHAFT_DIAMETER = 0.004
GRAIN_ARROW_SEGMENTS = 12

FUR_MODIFIER_NAME = "STF Fur Preview"
FUR_MATERIAL_NAME = "STF Fur"

# Fraction of a combed piece's total hair length lifted toward the surface
# normal (asin(0.38) ~ 22 degrees off the tangent plane) so grain-combed fur
# stands up slightly instead of lying perfectly flat against the piece -
# split this way (rather than just adding a flat amount) to keep the total
# rest length exactly piece.fur_length, since settings.hair_length is not
# independently controllable (see _sync_fur_particle_system).
FUR_NORMAL_LIFT = 0.38

# Golden-ratio conjugate: stepping hue by this amount keeps consecutive hues
# maximally spread around the wheel regardless of how many pieces there are
# (unlike hue = i / count, which degenerates for small/odd counts and
# reshuffles every existing piece's color whenever count changes as pieces
# are added/removed later - golden-ratio stepping is stable, each index's
# hue never changes).
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


# --- Per-piece coloring -----------------------------------------------------
#
# _get_or_create_piece_material now lives in operators/preview.py (imported
# above under its original name) so the cut/sliced previews and this coloring
# share one material per piece, keyed by the piece UUID - update the color in
# one place and it propagates to every object that uses it.


def sync_piece_colors(context, mesh_obj):
    """Color each piece by its color. The only coloring target is the actual
    cut mesh (operators/preview.py's cut-preview object) and anything that
    shares its materials (flattened objects, sliced-preview objects) - never
    the base source mesh, whose topology doesn't follow the seam edges (a
    face straddling a seam can't be assigned to just one piece there), and
    which - under a Mirror modifier - can't represent both mirrored pieces
    at all since only one half's faces actually exist on it.

    If mesh_obj.seams_to_fur_auto_color is on, every piece's color is first
    overwritten with a fresh golden-ratio-hue-stepped value (so manual
    per-piece edits never "stick" while auto mode is on - that's the whole
    point of the toggle) before syncing.

    Cheap on a repeat call with nothing seam/mesh-relevant changed: updates
    each piece's material color in place (instant, no re-cut - materials are
    shared datablocks, so this alone refreshes every object already using
    them). Only rebuilds the cut/sliced preview face assignments when the
    cache says the underlying cut actually changed.
    """
    if getattr(mesh_obj, "seams_to_fur_auto_color", False):
        for i, piece in enumerate(mesh_obj.seams_to_fur_pieces):
            hue = (i * _GOLDEN_RATIO_CONJUGATE) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
            piece.color = (r, g, b, 1.0)

    seam_curves = flatten_ops._find_seam_curves_for(mesh_obj)

    # Always cheap: refresh (or create) each piece's material in place. On
    # its own this already updates every object that references the
    # material - cut preview, sliced previews, flattened objects - with no
    # re-cut, as long as their face->material_index assignments (from
    # whenever they were last actually built) are still valid for the
    # current island structure.
    for piece in mesh_obj.seams_to_fur_pieces:
        mat = _get_or_create_piece_material(piece)
        flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
        if flat_obj is not None:
            flat_obj.color = tuple(piece.color)
            flat_mesh = flat_obj.data
            if len(flat_mesh.materials) == 0:
                flat_mesh.materials.append(mat)
            elif flat_mesh.materials[0] is not mat:
                flat_mesh.materials[0] = mat

    bm = None
    # The cut mesh is the *authoritative* coloring target (not just an
    # optional toggleable preview) - it must exist after any sync, not only
    # when the user has separately refreshed/shown the Cut preview. So rebuild it
    # whenever either the cache says the underlying cut is stale, or it
    # simply doesn't exist yet (the very first sync on this mesh, or the
    # user deleted it manually) - there's no way to create it without a
    # real bm, and the cache alone (just centroid summaries) can't provide
    # one, so a first-ever build always costs one real cut regardless.
    cut_obj_exists = bpy.data.objects.get(f"{mesh_obj.name}.cut") is not None
    if not island_cache.is_cut_cached(mesh_obj, seam_curves) or not cut_obj_exists:
        bm, face_island, seam_edges, _island_count = flatten_ops.compute_islands(
            context, mesh_obj, seam_curves
        )
        island_cache.store_from_islands(mesh_obj, seam_curves, bm, face_island)

        preview.build_cut_preview_object(context, mesh_obj, bm, seam_edges, face_island)
        sliced_coll_name = f"{preview.SLICED_COLLECTION_PREFIX}{mesh_obj.name}"
        if bpy.data.collections.get(sliced_coll_name) is not None:
            preview.build_sliced_preview_objects(
                context, mesh_obj, bm, face_island, seam_edges, mesh_obj.seams_to_fur_pieces
            )
            # build_sliced_preview_objects rebuilds each Sliced Preview
            # object's mesh DATA from scratch (mesh.clear_geometry() then
            # from_pydata) - it knows nothing about fur, so this wipes the
            # fur UV layer and collapses the material slots back down to
            # just the piece material. The FUR_MODIFIER_NAME particle
            # modifier itself lives on the *object*, not the mesh data, so
            # it survives the rebuild untouched - now silently pointing at
            # a UV layer and material slot that no longer exist. Confirmed
            # live: this is what made a plain color-apply appear to "break
            # grain direction on every piece" (the fur direction, not the
            # stored piece.grain_direction data, which this rebuild's own
            # UUID-carryover leaves genuinely intact) whenever this rebuild
            # path fired while fur was already shown. Re-sync any piece
            # whose fur modifier is still attached so it comes back
            # correct instead of silently broken.
            for piece in mesh_obj.seams_to_fur_pieces:
                refresh_piece_fur_full(context, mesh_obj, piece)

    if bm is not None:
        bm.free()


class SEAMS_TO_FUR_OT_sync_piece_colors(Operator):
    """Color each piece by its color. Applied to the actual cut mesh (the
    evaluated, mirrored, ACTUALLY-cut geometry - see operators/preview.py),
    where per-face island membership is known exactly, so it is correct even
    with a Mirror modifier; the base source mesh is never colored, since its
    topology doesn't follow the seam edges. Flattened objects and any
    sliced-preview objects are refreshed to match. Run after a re-bake if
    colors look stale."""

    bl_idname = "seams_to_fur.sync_piece_colors"
    bl_label = "Sync Piece Colors"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        sync_piece_colors(context, mesh_obj)
        self.report({"INFO"}, f"Synced colors for {len(mesh_obj.seams_to_fur_pieces)} piece(s)")
        return {"FINISHED"}


def _selected_pieces(context, mesh_obj):
    """Pieces a swatch click should color: any of the mesh's pieces whose
    Sliced/Distortion Preview or flattened pattern-piece object is
    currently selected in the viewport (any of those object types carry
    target.PIECE_UUID_PROP - see distortion.py/flatten.py/preview.py), or -
    if none of those are selected - just the active piece from the Pieces
    list. Covers both ways of "selecting pieces to color" the UI offers."""
    selected_uuids = {
        obj.get(target.PIECE_UUID_PROP)
        for obj in context.selected_objects
        if obj.get(target.PIECE_UUID_PROP)
    }
    if selected_uuids:
        return [p for p in mesh_obj.seams_to_fur_pieces if p.uuid in selected_uuids]
    if 0 <= mesh_obj.seams_to_fur_active_piece_index < len(mesh_obj.seams_to_fur_pieces):
        return [mesh_obj.seams_to_fur_pieces[mesh_obj.seams_to_fur_active_piece_index]]
    return []


class SEAMS_TO_FUR_OT_apply_swatch_color(Operator):
    """Apply a swatch color to the selected piece(s) - select piece(s)
    first (in the Pieces list, or by selecting their object(s) in the
    viewport - Sliced/Distortion Preview or a flattened pattern piece all
    work), then click a swatch.

    Deliberately a plain execute() operator, not modal/click-drag: an
    earlier version was modal, invoked by clicking a swatch button then
    dragging over pieces in the viewport - but a button click's own invoke
    event belongs to the panel's UI region, not the 3D viewport's WINDOW
    region, so the very first raycast used nonsense region/region_data and
    produced garbage hits; only the drag portion (once the mouse actually
    reached the viewport) worked at all, unreliably. Select-then-apply
    sidesteps region/raycast timing entirely."""

    bl_idname = "seams_to_fur.apply_swatch_color"
    bl_label = "Apply Swatch Color"
    bl_options = {"REGISTER", "UNDO"}

    color: FloatVectorProperty(name="Color", subtype="COLOR", size=3, min=0.0, max=1.0)

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        pieces = _selected_pieces(context, mesh_obj)
        if not pieces:
            self.report(
                {"ERROR"},
                "No piece selected - select a piece in the Pieces list, or select its "
                "Sliced Preview/flattened object in the viewport, then click a swatch",
            )
            return {"CANCELLED"}

        if getattr(mesh_obj, "seams_to_fur_auto_color", False):
            mesh_obj.seams_to_fur_auto_color = False
        for piece in pieces:
            piece.color = tuple(self.color) + (1.0,)
        sync_piece_colors(context, mesh_obj)
        self.report({"INFO"}, f"Applied color to {len(pieces)} piece(s)")
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_set_fur_length(Operator):
    """Quick-set the selected piece(s)' fur length to a preset value - same
    selection semantics as seams_to_fur.apply_swatch_color (see _selected_pieces):
    pieces selected in the viewport, or the active piece from the list if
    none are."""

    bl_idname = "seams_to_fur.set_fur_length"
    bl_label = "Set Fur Length"
    bl_options = {"REGISTER", "UNDO"}

    length_m: FloatProperty(name="Length (m)", default=0.025, min=0.0001)

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        pieces = _selected_pieces(context, mesh_obj)
        if not pieces:
            self.report(
                {"ERROR"},
                "No piece selected - select a piece in the Pieces list, or select its "
                "Sliced Preview/flattened object in the viewport, then click a length",
            )
            return {"CANCELLED"}
        for piece in pieces:
            piece.fur_length = self.length_m
        self.report({"INFO"}, f"Set fur length for {len(pieces)} piece(s)")
        return {"FINISHED"}


# --- Grain direction ---------------------------------------------------------


def _get_or_create_grain_arrow_material():
    mat = bpy.data.materials.get("STF Grain Direction")
    if mat is None:
        mat = bpy.data.materials.new("STF Grain Direction")
        mat.diffuse_color = GRAIN_ARROW_COLOR
        if mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is not None:
                bsdf.inputs["Emission Color"].default_value = GRAIN_ARROW_COLOR
                bsdf.inputs["Emission Strength"].default_value = 1.5
    return mat


def _get_or_create_arrow_collection(context):
    from . import collections

    return collections.get_or_create_child_collection(context, GRAIN_ARROW_COLLECTION)


def _arrow_object_name(mesh_obj, piece):
    return f"{mesh_obj.name}.{piece.name}.grain_arrow"


def _flat_arrow_object_name(flat_obj):
    # NOT f"{flat_obj.name}.grain_arrow" - flat_obj is itself already named
    # f"{mesh_obj.name}.{piece.name}" (see operators/flatten.py), which
    # would make that collide exactly with _arrow_object_name's own
    # f"{mesh_obj.name}.{piece.name}.grain_arrow" for the 3D arrow.
    return f"{flat_obj.name}.flat_grain_arrow"


def _remove_object(name):
    obj = bpy.data.objects.get(name)
    if obj is not None:
        bpy.data.objects.remove(obj, do_unlink=True)


def _project_tangent(direction, normal):
    """direction with any component along normal removed, renormalized -
    keeps the arrow lying flat against the surface at the anchor point
    instead of poking through it/floating off it, which is especially
    visible on curved pieces where a raw drag-to-drag vector rarely stays
    in the surface's local tangent plane. Returns None if direction turns
    out to be (near) parallel to normal, i.e. no meaningful tangent."""
    normal = normal.normalized()
    tangent = direction - normal * direction.dot(normal)
    return tangent.normalized() if tangent.length > 1e-6 else None


def _place_arrow(context, parent_obj, name, start_local, direction_local, length):
    """(Re)builds a real arrow-shaped MESH (cylinder shaft + cone head,
    built directly via bmesh - not a beveled curve, see below) from
    start_local along direction_local, in parent_obj's local space -
    parented with an IDENTITY parent-inverse (not
    parent_obj.matrix_world.inverted()) so the mesh's own vertex
    coordinates are genuinely interpreted as parent-local and transformed
    by parent_obj's world matrix, the same as any other child object;
    using matrix_world.inverted() instead would cancel that transform out
    and make the points render as literal world coordinates, which only
    happens to look right for a parent that itself sits at the world
    origin (true for the source mesh in most scenes, but false for a
    flattened pattern piece placed off in its own grid layout - confirmed
    as the cause of a flat-piece arrow rendering at the origin instead of
    on the piece). Used both for the 3D arrow on the source mesh and the
    2D arrow on a flattened pattern piece - same shape either way, just a
    different parent/local space.

    An earlier version used a beveled Curve (a uniform-radius shaft
    section, then two coincident points meant to jump the radius wide
    before tapering to a point) - confirmed by inspection that Blender
    doesn't reliably render a crisp flare at a zero-length curve segment,
    so it looked like a plain smooth double-taper instead of a real
    arrowhead. A real mesh with two actual (non-coincident-radius) vertex
    rings at the shaft/head boundary has no such ambiguity - it's just
    geometry, not an interpolation edge case.

    Diameters (GRAIN_ARROW_SHAFT_DIAMETER/HEAD_DIAMETER) are fixed
    regardless of `length` - only the shaft/head lengths scale with it -
    so a long dragged arrow reads as a long, normal-thickness arrow, not
    an oversized cone."""
    head_length = min(length * 0.3, length * 0.9)
    shaft_length = length - head_length
    shaft_radius = GRAIN_ARROW_SHAFT_DIAMETER / 2.0
    head_radius = GRAIN_ARROW_HEAD_DIAMETER / 2.0

    direction_local = direction_local.normalized()
    up = Vector((0.0, 0.0, 1.0)) if abs(direction_local.z) < 0.99 else Vector((1.0, 0.0, 0.0))
    side1 = direction_local.cross(up).normalized()
    side2 = direction_local.cross(side1).normalized()

    bm = bmesh.new()

    def ring(z, radius):
        verts = []
        for i in range(GRAIN_ARROW_SEGMENTS):
            theta = 2.0 * math.pi * i / GRAIN_ARROW_SEGMENTS
            offset = side1 * (radius * math.cos(theta)) + side2 * (radius * math.sin(theta))
            verts.append(bm.verts.new(start_local + direction_local * z + offset))
        return verts

    base = ring(0.0, shaft_radius)
    shaft_top = ring(shaft_length, shaft_radius)
    head_base = ring(shaft_length, head_radius)  # same z as shaft_top - a real step, not a taper
    base_center = bm.verts.new(start_local)
    apex = bm.verts.new(start_local + direction_local * length)  # a plain single-point cone tip

    n = GRAIN_ARROW_SEGMENTS
    smooth_faces = []
    for i in range(n):
        j = (i + 1) % n
        smooth_faces.append(bm.faces.new((base[i], base[j], shaft_top[j], shaft_top[i])))  # shaft side (round)
        bm.faces.new((base_center, base[j], base[i]))  # base cap (flat)
        bm.faces.new((shaft_top[i], shaft_top[j], head_base[j], head_base[i]))  # shoulder flare (flat)
        smooth_faces.append(bm.faces.new((head_base[i], head_base[j], apex)))  # head cone side (round)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    # Smooth shading only on the actually-curved faces (shaft cylinder,
    # head cone) - the base cap and the shoulder flare are genuinely flat
    # discs/annuli and should keep their crisp edges, not get blended into
    # the curved surface's shading.
    for face in smooth_faces:
        face.smooth = True

    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type != "MESH":
        # Migrates any arrow object left over from the old Curve-based
        # version, cheaply and idempotently.
        bpy.data.objects.remove(obj, do_unlink=True)
        obj = None
    if obj is None:
        mesh_data = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh_data)
        _get_or_create_arrow_collection(context).objects.link(obj)
    else:
        obj.data.clear_geometry()

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    obj.parent = parent_obj
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.show_in_front = True
    obj.color = GRAIN_ARROW_COLOR
    mat = _get_or_create_grain_arrow_material()
    if mat.name not in (m.name for m in obj.data.materials if m):
        obj.data.materials.append(mat)
    return obj


def _build_flat_proxy(flat_obj):
    """A BVH over the flattened piece's own faces, but positioned at each
    vertex's pre-flatten 3D location (the "seams_to_fur_orig_co" attribute - see
    operators/flatten.py) instead of its actual flat 2D one. Querying this
    proxy with a 3D point on the curved piece therefore returns a face
    index that's already in exact 1:1 correspondence with a triangle on the
    flat mesh (same vertex indices), which is what makes mapping a 3D
    anchor/direction onto its flat 2D counterpart possible without a
    separate UV-unwrap pass. Returns None if the piece has no such data
    (not flattened, or an unreferenced-vertex edge case left it empty).

    Every entry in the returned `polygons` is a real triangle - BFF's
    *output* isn't guaranteed to stay triangulated even though its input
    always is (confirmed empirically: quads and the occasional pentagon
    do show up), and _map_point_to_flat/_map_direction_to_flat only know
    how to solve a 3-vertex barycentric/edge basis, so a proxy face that
    wasn't a triangle used to make every point whose nearest face happened
    to land there silently fail to map at all - collapsing to a stray
    (0, 0) UV instead of its real position ("really messed up" fur UVs).

    Triangulated via mathutils.geometry.tessellate_polygon (proper ear-
    clipping on the n-gon's own best-fit plane), not a naive fan from
    vertex 0 - confirmed empirically that fanning can produce a genuinely
    degenerate (near-zero-area, "denom" ~1e-18) sliver triangle for a
    poorly-shaped n-gon even though the n-gon itself isn't degenerate,
    which fails the exact same way (silently unmappable, (0, 0) UV) as
    the original all-n-gon bug it was meant to fix."""
    flat_mesh = flat_obj.data
    attr = flat_mesh.attributes.get("seams_to_fur_orig_co")
    if attr is None or not flat_mesh.polygons:
        return None
    orig_co = [Vector(item.vector) for item in attr.data]
    polygons = []
    for poly in flat_mesh.polygons:
        verts = tuple(poly.vertices)
        if len(verts) < 3:
            continue
        if len(verts) == 3:
            polygons.append(verts)
            continue
        for tri in tessellate_polygon([[orig_co[v] for v in verts]]):
            polygons.append(tuple(verts[i] for i in tri))
    bvh = BVHTree.FromPolygons(orig_co, polygons, all_triangles=True)
    return flat_obj, orig_co, polygons, bvh


def _nearest_flat_triangle(orig_co, polygons, bvh, point_local):
    """Shared first half of the 3D->2D mapping used by both
    _map_point_to_flat and _map_direction_to_flat: finds the flat-proxy
    triangle nearest point_local and returns everything needed to solve
    barycentric/basis coordinates against it, or None if there's no nearby
    face at all.

    Does NOT filter out a (near-)zero-area "denom" here - callers see it
    and decide for themselves whether they can do something useful with a
    degenerate triangle anyway (see _map_point_to_flat's nearest-vertex
    fallback) rather than this shared helper unilaterally giving up for
    all of them."""
    hit_co, _n, face_index, _dist = bvh.find_nearest(point_local)
    if hit_co is None or face_index is None or face_index >= len(polygons):
        return None
    verts_idx = polygons[face_index]
    if len(verts_idx) != 3:
        return None
    i0, i1, i2 = verts_idx
    p0, p1, p2 = orig_co[i0], orig_co[i1], orig_co[i2]
    e1, e2 = p1 - p0, p2 - p0
    d00, d01, d11 = e1.dot(e1), e1.dot(e2), e2.dot(e2)
    denom = d00 * d11 - d01 * d01
    return hit_co, verts_idx, p0, e1, e2, d00, d01, d11, denom


def _map_point_to_flat(flat_obj, orig_co, polygons, bvh, point_local):
    """Maps a 3D point on the curved piece to the corresponding 2D point on
    its flattened counterpart, via the nearest triangle's own barycentric
    weights (see _map_direction_to_flat for the full rationale). Returns
    None if there's no usable nearby triangle at all.

    If the nearest triangle turns out to be (near-)degenerate - confirmed
    empirically that this happens even after proper ear-clipping
    triangulation, for a genuinely thin/sliver n-gon in BFF's flat output
    that stays near-degenerate no matter which way it gets cut into
    triangles - proper barycentric interpolation is ill-defined (or would
    divide by ~0), but the nearest of its 3 corners is still a perfectly
    good, real point on the piece, so use that directly instead of
    silently collapsing to (0, 0)."""
    tri = _nearest_flat_triangle(orig_co, polygons, bvh, point_local)
    if tri is None:
        return None
    hit_co, (i0, i1, i2), p0, e1, e2, d00, d01, d11, denom = tri
    flat_verts = flat_obj.data.vertices
    q0, q1, q2 = flat_verts[i0].co.copy(), flat_verts[i1].co.copy(), flat_verts[i2].co.copy()

    if abs(denom) < 1e-16:
        candidates = ((p0, q0), (p0 + e1, q1), (p0 + e2, q2))
        _best_p, best_q = min(candidates, key=lambda pq: (hit_co - pq[0]).length_squared)
        return best_q

    rel = hit_co - p0
    d20, d21 = rel.dot(e1), rel.dot(e2)
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    return q0 * w0 + q1 * w1 + q2 * w2


def _map_direction_to_flat(flat_obj, orig_co, polygons, bvh, anchor_local, direction_local):
    """Maps a 3D anchor point + tangent direction on the curved piece to
    the corresponding 2D point + direction on its flattened counterpart.

    Finds the nearest original-shape triangle to anchor_local, then
    transfers both the point (via its barycentric weights) and the
    direction (by solving for its coordinates in the triangle's own edge
    basis, then applying those same coordinates to the flat triangle's edge
    basis) - i.e. the triangle's own 3D->2D affine map, which is exactly
    what BFF's locally-close-to-conformal flattening does to everything
    inside that triangle anyway, so this stays visually consistent with how
    the piece itself got flattened. Only meaningful for triangles (the cut
    mesh is always triangulated - see operators/flatten.py's _cut_mesh),
    so returns None for anything else, or if the geometry is too
    degenerate to solve."""
    tri = _nearest_flat_triangle(orig_co, polygons, bvh, anchor_local)
    if tri is None:
        return None
    hit_co, (i0, i1, i2), p0, e1, e2, d00, d01, d11, denom = tri
    if abs(denom) < 1e-16:
        # Unlike _map_point_to_flat, there's no sensible fallback here - a
        # degenerate (zero-area) triangle has no well-defined 2D affine
        # map to carry a *direction* through, only individual corner
        # points.
        return None
    flat_verts = flat_obj.data.vertices
    q0, q1, q2 = flat_verts[i0].co.copy(), flat_verts[i1].co.copy(), flat_verts[i2].co.copy()

    # Barycentric weights of the (on-triangle) hit point w.r.t. (p0, p1, p2).
    rel = hit_co - p0
    d20, d21 = rel.dot(e1), rel.dot(e2)
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2
    anchor_2d = q0 * w0 + q1 * w1 + q2 * w2

    # Coordinates of direction_local in the (e1, e2) basis, then applied to
    # the flat triangle's own (e1_2d, e2_2d) basis - a direct transfer of
    # the tangent vector through this triangle's own 3D->2D affine map.
    rhs0, rhs1 = direction_local.dot(e1), direction_local.dot(e2)
    a = (rhs0 * d11 - rhs1 * d01) / denom
    b = (rhs1 * d00 - rhs0 * d01) / denom
    e1_2d, e2_2d = q1 - q0, q2 - q0
    direction_2d = e1_2d * a + e2_2d * b
    if direction_2d.length < 1e-9:
        return None
    return anchor_2d, direction_2d.normalized()


def refresh_grain_arrow(context, mesh_obj, piece):
    """(Re)builds the persistent grain-direction arrows for one piece: one
    on the curved source mesh, anchored exactly where the user clicked
    (piece.grain_anchor), and - if the piece has been flattened - a second
    one on the flattened pattern piece, mapped through the piece's own
    3D->2D triangle correspondence (see _map_direction_to_flat). Safe to
    call repeatedly. Removes both arrows if the piece has no grain
    direction set (has_grain_direction False)."""
    arrow_name = _arrow_object_name(mesh_obj, piece)
    flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None

    if not piece.has_grain_direction:
        _remove_object(arrow_name)
        if flat_obj is not None:
            _remove_object(_flat_arrow_object_name(flat_obj))
        return None

    direction = Vector(piece.grain_direction)
    if direction.length < 1e-8:
        return None
    direction = direction.normalized()
    max_dim = max(mesh_obj.dimensions) if max(mesh_obj.dimensions) > 0 else 1.0
    length = max_dim * 0.12
    start = Vector(piece.grain_anchor)
    arrow = _place_arrow(context, mesh_obj, arrow_name, start, direction, length)

    if flat_obj is not None and flat_obj.type == "MESH":
        proxy = _build_flat_proxy(flat_obj)
        if proxy is not None:
            mapped = _map_direction_to_flat(*proxy, start, direction)
            if mapped is not None:
                anchor_2d, direction_2d = mapped
                flat_dim = max(flat_obj.dimensions)
                flat_length = flat_dim * 0.15 if flat_dim > 0 else length
                _place_arrow(
                    context, flat_obj, _flat_arrow_object_name(flat_obj), anchor_2d, direction_2d, flat_length
                )
    return arrow


def _clear_grain_direction(context, mesh_obj, piece):
    piece.has_grain_direction = False
    piece.grain_direction = (1.0, 0.0, 0.0)
    piece.grain_anchor = (0.0, 0.0, 0.0)
    refresh_grain_arrow(context, mesh_obj, piece)
    refresh_piece_fur_full(context, mesh_obj, piece)


def _ensure_sliced_preview(context, mesh_obj):
    """Builds (or rebuilds, if stale - see preview.ensure_preview_fresh)
    the Sliced Preview, then clears the base mesh (and any other
    same-shape preview) out of the way so it's actually clickable/usable
    - shared by the grain-direction operator and fur preview, both of
    which need one real, genuinely up-to-date per-piece object to do a
    nearest-surface piece lookup against."""
    preview.ensure_preview_fresh(context, mesh_obj, "SLICED")
    preview.hide_same_shape_previews(mesh_obj, except_prefix=preview.SLICED_COLLECTION_PREFIX)
    preview.set_base_mesh_hidden(mesh_obj, True)


def _piece_sliced_bvhs(mesh_obj):
    """One BVH per piece, built from its Sliced Preview object - same
    local space as mesh_obj itself (see preview.build_sliced_preview_objects),
    so it can be queried directly with mesh_obj-local points/faces to find
    which piece a given point or face belongs to."""
    piece_bvhs = []
    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects.get(f"{mesh_obj.name}.{piece.name}.sliced")
        if sliced_obj is None or sliced_obj.type != "MESH":
            continue
        bm = bmesh.new()
        bm.from_mesh(sliced_obj.data)
        piece_bvhs.append((piece, BVHTree.FromBMesh(bm)))
        bm.free()
    return piece_bvhs


class SEAMS_TO_FUR_OT_set_grain_direction(Operator):
    """Click a piece in the Sliced Preview then drag across it to set that
    piece's fabric/fur grain direction. Release to confirm; Esc cancels
    without changing anything. Which piece is edited is determined entirely
    by where you click - no piece needs to be pre-selected in the piece
    list first. Builds the Sliced Preview automatically if it isn't already
    shown.

    With Mirror Edit (the mirror-icon toggle next to this button) on, and
    the mesh has a Mirror modifier, the same drag also sets the grain
    direction on whichever piece sits at the click point's reflection
    across the mirror plane - found freshly by that geometric lookup each
    time, not by tracking which pieces are "actually" mirror pairs, so it
    keeps working correctly even after pieces are added, removed, or
    renamed."""

    bl_idname = "seams_to_fur.set_grain_direction"
    bl_label = "Set Grain Direction"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return context.area is not None and context.area.type == "VIEW_3D" and obj is not None and len(obj.seams_to_fur_pieces) > 0

    def _ensure_sliced_preview(self, context):
        _ensure_sliced_preview(context, self.mesh_obj)

    def _build_bvhs(self, context):
        """Builds one BVH over the whole evaluated base mesh, used to track
        the drag continuously, plus one BVH per sliced-preview piece object
        (same local space as the base mesh, since sliced pieces are built
        at the source mesh's own world transform with unmoved vertex
        positions - see preview.build_sliced_preview_objects), used to
        identify *which* piece a given local-space point belongs to and to
        get an accurate surface normal there for tangent-constraining the
        arrow."""
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = self.mesh_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(eval_mesh)
        self.bvh = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()

        self.piece_bvhs = _piece_sliced_bvhs(self.mesh_obj)
        self.flat_proxies = {}

    def _get_flat_proxy(self, piece):
        """_build_flat_proxy result for piece, built once per operator
        invocation and cached - rebuilding a BVH on every mouse-move tick
        would make the live flat-piece preview noticeably laggy."""
        if piece.uuid not in self.flat_proxies:
            flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
            self.flat_proxies[piece.uuid] = _build_flat_proxy(flat_obj) if flat_obj is not None else None
        return self.flat_proxies[piece.uuid]

    def _place_flat_preview(self, context, piece, anchor_local, direction_local):
        proxy = self._get_flat_proxy(piece)
        if proxy is None:
            return
        flat_obj = proxy[0]
        mapped = _map_direction_to_flat(*proxy, anchor_local, direction_local)
        if mapped is None:
            return
        anchor_2d, direction_2d = mapped
        flat_dim = max(flat_obj.dimensions)
        flat_length = flat_dim * 0.15 if flat_dim > 0 else 0.05
        _place_arrow(context, flat_obj, _flat_arrow_object_name(flat_obj), anchor_2d, direction_2d, flat_length)

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
        return location

    def _piece_at(self, point_local, exclude=None):
        """Nearest piece (by its sliced-preview surface) to a local-space
        point, plus the nearest surface point and its normal there - used
        both to identify the piece under the click (and get an on-surface
        anchor/normal for it) and, with a mirrored point, to find whichever
        piece sits at the mirror-plane reflection of the click. Returns
        (None, None, None) if there are no pieces to check."""
        best_piece, best_co, best_normal, best_dist = None, None, None, None
        for piece, bvh in self.piece_bvhs:
            if piece is exclude:
                continue
            hit_co, hit_normal, _i, dist = bvh.find_nearest(point_local)
            if hit_co is not None and (best_dist is None or dist < best_dist):
                best_piece, best_co, best_normal, best_dist = piece, hit_co, hit_normal, dist
        return best_piece, best_co, best_normal

    def _mirror_local_point(self, point_local):
        mat_world = self.mesh_obj.matrix_world
        world = mat_world @ point_local
        offset = (world - self.mirror_point).dot(self.mirror_normal)
        mirrored_world = world - self.mirror_normal * (2.0 * offset)
        return mat_world.inverted() @ mirrored_world

    def _mirror_local_direction(self, direction_local):
        mat3 = self.mesh_obj.matrix_world.to_3x3()
        world_dir = mat3 @ direction_local
        mirrored_world_dir = world_dir - self.mirror_normal * (2.0 * world_dir.dot(self.mirror_normal))
        return (mat3.inverted() @ mirrored_world_dir).normalized()

    def invoke(self, context, event):
        self.mesh_obj = target.resolve_mesh_obj(context)
        self._ensure_sliced_preview(context)
        self._build_bvhs(context)
        if not self.piece_bvhs:
            self.report({"ERROR"}, "No sliced preview pieces to click on")
            return {"CANCELLED"}

        (
            self.mirror_point,
            self.mirror_normal,
            self.has_real_mirror,
            _mirror_merge_threshold,
        ) = curve_display.mirror_plane_for(self.mesh_obj)
        self.mirror_edit = self.mesh_obj.seams_to_fur_grain_mirror_edit and self.has_real_mirror

        self.piece = None
        self.mirror_piece = None
        self.start = None
        self.start_normal = None
        self.mirror_anchor = None
        self.mirror_anchor_normal = None

        context.area.header_text_set(
            "Click a piece then drag to set grain direction | Esc: cancel"
        )
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _apply(self, context, direction, length, commit):
        """Places (and, if commit, saves) the arrow(s) for a drag currently
        ending direction/length away from self.start. direction is already
        surface-tangent at self.start; the mirror piece's direction is
        derived fresh from it each call, not stored/reused, so it stays
        correct even if the primary direction changes mid-drag."""
        _place_arrow(context, self.mesh_obj, _arrow_object_name(self.mesh_obj, self.piece), self.start, direction, length)
        self._place_flat_preview(context, self.piece, self.start, direction)
        if commit:
            self.piece.grain_anchor = tuple(self.start)
            self.piece.grain_direction = tuple(direction)
            self.piece.has_grain_direction = True
            refresh_piece_fur_full(context, self.mesh_obj, self.piece)

        if self.mirror_piece is not None:
            mirrored_raw = self._mirror_local_direction(direction)
            mirrored_dir = _project_tangent(mirrored_raw, self.mirror_anchor_normal) or mirrored_raw
            _place_arrow(
                context,
                self.mesh_obj,
                _arrow_object_name(self.mesh_obj, self.mirror_piece),
                self.mirror_anchor,
                mirrored_dir,
                length,
            )
            self._place_flat_preview(context, self.mirror_piece, self.mirror_anchor, mirrored_dir)
            if commit:
                self.mirror_piece.grain_anchor = tuple(self.mirror_anchor)
                self.mirror_piece.grain_direction = tuple(mirrored_dir)
                self.mirror_piece.has_grain_direction = True
                refresh_piece_fur_full(context, self.mesh_obj, self.mirror_piece)

    def modal(self, context, event):
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = self._raycast(context, event)
            self.piece = self.mirror_piece = None
            self.start = self.start_normal = None
            self.mirror_anchor = self.mirror_anchor_normal = None
            if hit is not None:
                piece, co, normal = self._piece_at(hit)
                if piece is not None:
                    self.piece, self.start, self.start_normal = piece, co, normal
                    if self.mirror_edit:
                        mirrored_point = self._mirror_local_point(self.start)
                        mpiece, mco, mnormal = self._piece_at(mirrored_point, exclude=self.piece)
                        if mpiece is not None:
                            self.mirror_piece, self.mirror_anchor, self.mirror_anchor_normal = mpiece, mco, mnormal
            return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE" and self.start is not None:
            hit = self._raycast(context, event)
            if hit is not None and (hit - self.start).length > 1e-8:
                raw_direction = (hit - self.start).normalized()
                length = (hit - self.start).length
                direction = _project_tangent(raw_direction, self.start_normal)
                if direction is not None:
                    self._apply(context, direction, length, commit=False)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            if self.start is not None:
                hit = self._raycast(context, event)
                if hit is not None and (hit - self.start).length > 1e-8:
                    raw_direction = (hit - self.start).normalized()
                    length = (hit - self.start).length
                    direction = _project_tangent(raw_direction, self.start_normal)
                    if direction is not None:
                        self._apply(context, direction, length, commit=True)
                        names = [self.piece.name]
                        if self.mirror_piece is not None:
                            names.append(self.mirror_piece.name)
                        self.report({"INFO"}, f"Grain direction set for {', '.join(repr(n) for n in names)}")
            context.area.header_text_set(None)
            return {"FINISHED"}

        if event.type == "ESC":
            context.area.header_text_set(None)
            if self.start is not None:
                # Drop the live preview arrow(s); restore the committed one (if any).
                if self.piece is not None:
                    refresh_grain_arrow(context, self.mesh_obj, self.piece)
                if self.mirror_piece is not None:
                    refresh_grain_arrow(context, self.mesh_obj, self.mirror_piece)
            return {"CANCELLED"}

        if event.type in {"MIDDLEMOUSE", "WHEELUPMOUSE", "WHEELDOWNMOUSE"} or event.alt:
            return {"PASS_THROUGH"}

        return {"RUNNING_MODAL"}


class SEAMS_TO_FUR_OT_reset_grain_direction(Operator):
    """Clear the active piece's grain direction and remove its arrow(s)"""

    bl_idname = "seams_to_fur.reset_grain_direction"
    bl_label = "Reset Grain Direction"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and 0 <= obj.seams_to_fur_active_piece_index < len(obj.seams_to_fur_pieces)

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        piece = mesh_obj.seams_to_fur_pieces[mesh_obj.seams_to_fur_active_piece_index]
        _clear_grain_direction(context, mesh_obj, piece)
        self.report({"INFO"}, f"Grain direction reset for '{piece.name}'")
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_reset_all_grain_directions(Operator):
    """Clear every piece's grain direction and remove all arrows"""

    bl_idname = "seams_to_fur.reset_all_grain_directions"
    bl_label = "Reset All Grain Directions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        for piece in mesh_obj.seams_to_fur_pieces:
            _clear_grain_direction(context, mesh_obj, piece)
        self.report({"INFO"}, f"Grain direction reset for {len(mesh_obj.seams_to_fur_pieces)} piece(s)")
        return {"FINISHED"}


def grain_arrows_exist():
    """One shared collection across every source mesh in the scene (unlike
    the per-mesh Cut/Sliced/Distortion previews) - arrows are cheap and
    already built on-demand as each grain direction is set, so there's no
    refresh/cache concept here, just this existence check (for the panel
    to grey out the toggle when there's nothing to show/hide) and the
    visibility flip below."""
    coll = bpy.data.collections.get(GRAIN_ARROW_COLLECTION)
    return coll is not None and len(coll.objects) > 0


def grain_arrows_visible():
    coll = bpy.data.collections.get(GRAIN_ARROW_COLLECTION)
    return coll is not None and not coll.hide_viewport


class SEAMS_TO_FUR_OT_toggle_grain_arrows(Operator):
    """Show/hide every grain-direction arrow in the scene at once"""

    bl_idname = "seams_to_fur.toggle_grain_arrows"
    bl_label = "Toggle Grain Direction Arrows"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def description(cls, context, properties):
        if not grain_arrows_exist():
            return "No grain direction arrows to show/hide yet"
        return "Hide all grain direction arrows" if grain_arrows_visible() else "Show all grain direction arrows"

    def execute(self, context):
        coll = bpy.data.collections.get(GRAIN_ARROW_COLLECTION)
        if coll is None or len(coll.objects) == 0:
            self.report({"WARNING"}, "No grain direction arrows to show/hide yet")
            return {"CANCELLED"}
        coll.hide_viewport = not coll.hide_viewport
        return {"FINISHED"}


# --- Fur / hair preview ------------------------------------------------------
#
# Lives on each piece's own Sliced Preview object (built automatically if
# not already shown - see operators/preview.py), one hair particle system
# per object, not one shared system with vertex-group masking on the
# source mesh. This is a deliberate departure from an earlier version that
# tried to make one particle system per piece work on the source mesh
# itself, restricted to each piece's faces via a best-effort nearest-
# surface guess (a real approximation: a face straddling a seam had to be
# assigned to whichever piece's centroid it happened to land nearest, and
# a piece that only exists via a Mirror modifier has no real geometry on
# the source mesh to assign to at all, so got no fur whatsoever) - and it
# depended on (and had to carefully avoid corrupting) the source mesh's
# own pre-existing UV map, which may not exist, or may not be usable for
# this at all. The Sliced Preview object already has clean, unambiguous,
# piece-exact topology (real geometry even for Mirror-only pieces - the
# evaluated+cut bmesh it's built from already includes the mirrored
# geometry), and starts with no UV layers of its own to worry about
# disturbing - a strictly better fit for this feature, and one that
# previews fur on the same cut/separated pieces the garment is actually
# constructed from.
#
# Direction: found empirically (see project history) that Blender's hair
# "tangent" (settings.tangent_factor/tangent_phase) is derived from UV
# layer index 0 specifically (not whatever's marked "active"/"active
# render"), and that `use_advanced_hair = True` was the actual root cause
# of an earlier "long straight hairs ignoring grain direction" bug: that
# flag switches the hair's rest shape to a cloth-simulation result, which
# stays fully extended and unbent until the physics is actually
# baked/stepped, silently making normal_factor/tangent_factor/
# tangent_phase inert. With use_advanced_hair off, those factors directly
# control the *static* hair shape with no baking needed.
#
# The UV this preview writes (FUR_UV_LAYER) is the piece's OWN real
# flattened 2D coordinates (via _map_point_to_flat, the same 3D<->2D
# correspondence the grain-direction arrows use), not an independently
# computed per-face tangent axis - an earlier version built the UV from
# cross(normal, grain_direction) freshly at each face, which is correct
# per-face in isolation but has no reason to stay CONSISTENT face to face
# on a curved surface, so the comb direction visibly twisted across a
# piece instead of reading as one uniform direction matching the flat
# pattern's own grain arrow. Reusing the actual flattening fixes that: the
# UV gradient is exactly BFF's own (locally-conformal) parameterization,
# so the fur's local growth direction varies smoothly and predictably
# exactly the way the flattening itself does - and a single UV rotation
# (computed once per piece in _fur_alignment_rotation, not per face) is
# then enough to align the whole piece with its flat-pattern grain arrow.

FUR_UV_LAYER = "seams_to_fur_fur_grain"


def _get_or_create_fur_material(piece):
    """A real Hair BSDF (not Principled BSDF) - the correct node for actual
    strand geometry: it models the anisotropic highlight that runs *along*
    the strand tangent, which a surface-oriented Principled BSDF doesn't
    reproduce correctly on thin hair. Uses the Reflection component (the
    manual's own suggested setup mixes a Reflection- and a Transmission-
    component Hair BSDF together - skipped here for simplicity, since a
    single Reflection-only node with the piece's color is already a solid,
    much-closer-to-real-fur preview than Principled BSDF was)."""
    mat_name = f"{FUR_MATERIAL_NAME} {piece.uuid[:8]}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Hair BSDF")
    if bsdf is None:
        # Materials are looked up and reused purely by name (mat_name above),
        # so a pre-existing material by this exact name with some other node
        # setup - confirmed live: a leftover Principled-BSDF-only material
        # from before this function switched to Hair BSDF - gets silently
        # reused as-is. use_nodes is already True by this point, so toggling
        # it again is a no-op (confirmed empirically: it only auto-populates
        # a default node tree the very first time a material goes from no
        # node_tree at all to use_nodes True, not on a later toggle) - the
        # fix has to explicitly (re)build the node this function expects, or
        # its color updates keep silently no-op'ing forever, stuck at
        # whatever default color the stale node type happened to start with
        # (this exact failure mode already bit one real piece once, when the
        # node type this function expected changed the other way around).
        mat.node_tree.nodes.clear()
        output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfHair")
        bsdf.name = "Hair BSDF"
        mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bsdf.component = "Reflection"
    color = tuple(piece.color)
    mat.diffuse_color = color
    bsdf.inputs["Color"].default_value = color
    return mat


def _build_sliced_piece_uv(sliced_obj, proxy):
    """Writes FUR_UV_LAYER on sliced_obj - a brand new UV layer (its mesh
    starts with none of its own, built fresh via mesh.from_pydata - see
    preview.build_sliced_preview_objects - so it's automatically UV layer
    index 0, no swapping/renaming needed the way sharing the source mesh
    would have required). Each vertex's UV is its own real position within
    the piece's actual flattening (see module docstring for why).

    proxy (see _build_flat_proxy) may be None - piece has no grain
    direction set, or isn't flattened yet - in which case every vertex
    just gets (0, 0); fine, since tangent_factor stays 0 for such pieces
    and this UV goes unused."""
    mesh = sliced_obj.data
    uv_layer = mesh.uv_layers.get(FUR_UV_LAYER) or mesh.uv_layers.new(name=FUR_UV_LAYER)
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert_co = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv_2d = _map_point_to_flat(*proxy, vert_co) if proxy is not None else None
            uv_layer.data[loop_index].uv = (uv_2d.x, uv_2d.y) if uv_2d is not None else (0.0, 0.0)
    return uv_layer


def _measured_fur_direction(context, sliced_obj, psys, proxy):
    """The actual rendered comb direction of this system, mapped back into
    the piece's own flat space (via the same 3D<->2D correspondence the
    grain arrows use) and averaged - as a circular mean of unit vectors,
    to handle angle wraparound correctly - over several particles, not
    just one. A single particle's own local triangle can carry noticeably
    more of BFF's local (non-perfectly-conformal) distortion than the
    piece as a whole, which was enough to visibly throw off the alignment
    correction in _fur_alignment_rotation when only sampling particle 0.
    Returns the (approximate) flat-space direction as a Vector, or None if
    there's nothing to measure (e.g. this piece emits no particles)."""
    depsgraph = context.evaluated_depsgraph_get()
    depsgraph.update()
    eval_obj = sliced_obj.evaluated_get(depsgraph)
    for eval_psys in eval_obj.particle_systems:
        if eval_psys.name != psys.name:
            continue
        n = min(len(eval_psys.particles), 20)
        if n == 0:
            return None
        accum = Vector((0.0, 0.0, 0.0))
        n_valid = 0
        for i in range(n):
            keys = eval_psys.particles[i].hair_keys
            if len(keys) < 2:
                continue
            root, tip = keys[0].co, keys[-1].co
            direction = tip - root
            if direction.length < 1e-9:
                continue
            mapped = _map_direction_to_flat(*proxy, root, direction.normalized())
            if mapped is None:
                continue
            accum += mapped[1]
            n_valid += 1
        if n_valid == 0 or accum.length < 1e-9:
            return None
        return accum.normalized()
    return None


def _rotate_fur_uv(sliced_obj, angle_deg):
    """Rotates sliced_obj's whole FUR_UV_LAYER by angle_deg (a plain 2D
    rotation of the (u, v) values themselves) - see _fur_alignment_rotation
    for why this, not settings.tangent_phase, is what actually corrects
    the fur's comb direction to match the flat-pattern grain arrow. No
    per-face piece filtering needed (unlike an earlier version) - the
    whole object is this one piece."""
    uv_layer = sliced_obj.data.uv_layers.get(FUR_UV_LAYER)
    if uv_layer is None or abs(angle_deg) < 1e-6:
        return
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    for item in uv_layer.data:
        u, v = item.uv
        item.uv = (u * cos_t - v * sin_t, u * sin_t + v * cos_t)


def _fur_alignment_rotation(context, sliced_obj, psys, piece, proxy):
    """The UV rotation (degrees, for _rotate_fur_uv) that aligns the fur's
    actual comb direction with the piece's flat-pattern grain arrow
    direction.

    settings.tangent_phase turned out NOT to be a reliable way to do this
    rotation (found empirically: even after measuring its effect and
    solving for the exact phase that should land on the target angle, the
    real resulting direction on an actual BFF-flattened piece didn't
    match - tangent_phase's relationship to the final angle isn't the
    simple, uniform +180-degrees-per-unit rotation it appeared to be on a
    plain default-UV test case). Rotating the UV *data* itself is a
    transform we fully control and verify directly, so it isn't subject
    to whatever unpredictable internal computation tangent_phase goes
    through.

    Even so, a plain "measure once, rotate by target-minus-measured" isn't
    quite enough: confirmed empirically (on a real Mirror-modifier mesh)
    that rotating a piece's UV by +X degrees moves the resulting hair
    angle by +X on some pieces but by -X on others - the sign flips for a
    piece whose sliced-object faces have reversed winding, which the
    mirrored half of a Mirror modifier genuinely has (kept that way so its
    normals still point outward after the mirror transform). Assuming it
    was always +X sent those pieces' fur off in a wildly wrong, unrelated
    direction (up to ~240 degrees off) rather than leaving a small
    residual error. So: probe with a small known rotation first to
    measure which sign actually applies *for this piece*, undo the probe,
    then solve for the real correction using that sign."""
    baseline_dir_2d = _measured_fur_direction(context, sliced_obj, psys, proxy)
    if baseline_dir_2d is None:
        return 0.0
    baseline_angle = math.degrees(math.atan2(baseline_dir_2d.y, baseline_dir_2d.x))

    target = _map_direction_to_flat(*proxy, Vector(piece.grain_anchor), Vector(piece.grain_direction))
    if target is None:
        return 0.0
    _target_anchor_2d, target_dir_2d = target
    target_angle = math.degrees(math.atan2(target_dir_2d.y, target_dir_2d.x))

    probe_deg = 15.0
    _rotate_fur_uv(sliced_obj, probe_deg)
    probe_dir_2d = _measured_fur_direction(context, sliced_obj, psys, proxy)
    _rotate_fur_uv(sliced_obj, -probe_deg)  # undo - net no-op on the UV data
    if probe_dir_2d is None:
        sign = 1.0
    else:
        probe_angle = math.degrees(math.atan2(probe_dir_2d.y, probe_dir_2d.x))
        observed_delta = ((probe_angle - baseline_angle + 180.0) % 360.0) - 180.0
        sign = 1.0 if observed_delta >= 0.0 else -1.0

    angle_diff = ((target_angle - baseline_angle + 180.0) % 360.0) - 180.0
    return sign * angle_diff


def _sync_fur_particle_system(context, sliced_obj, piece, density, face_fraction):
    existing_mod = sliced_obj.modifiers.get(FUR_MODIFIER_NAME)
    if existing_mod is not None and existing_mod.type == "PARTICLE_SYSTEM":
        mod = existing_mod
        psys = existing_mod.particle_system
    else:
        mod = sliced_obj.modifiers.new(name=FUR_MODIFIER_NAME, type="PARTICLE_SYSTEM")
        psys = sliced_obj.particle_systems[-1]
        psys.name = FUR_MODIFIER_NAME

    # A previous seams_to_fur.toggle_fur_preview call (which flips every
    # piece's modifier uniformly) can leave a *pre-existing* modifier
    # hidden - refreshing is a deliberate "show me the fur" action, so it
    # should always end with the preview actually visible, the same way a
    # brand new modifier already defaults to visible. Without this, adding
    # a piece and refreshing only ever made the new piece's modifier (which
    # starts visible) show up, leaving every already-toggled-off piece
    # looking untouched/bare even though its particle system was in fact
    # freshly synced - confirmed live: exactly this, on a file where fur
    # preview had been toggled off then two new pieces were split out.
    mod.show_viewport = True
    mod.show_render = True

    settings = psys.settings
    settings.type = "HAIR"
    settings.emit_from = "FACE"
    settings.use_modifier_stack = False
    settings.count = max(1, round(density * face_fraction))
    settings.hair_step = 4

    # NOT use_advanced_hair - see module docstring: that switches the rest
    # shape to an unbaked cloth-sim result, which stays straight until
    # actually simulated, silently defeating normal_factor/tangent_factor
    # below regardless of their values.
    settings.use_advanced_hair = False
    settings.physics_type = "NEWTON"

    flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
    proxy = _build_flat_proxy(flat_obj) if flat_obj is not None else None
    _build_sliced_piece_uv(sliced_obj, proxy if piece.has_grain_direction else None)

    # settings.hair_length is NOT an independent control once growth is
    # velocity-driven (physics_type != 'NO') - confirmed empirically that
    # Blender always recomputes the actual rest-shape length as
    # hair_step * sqrt(normal_factor**2 + tangent_factor**2) and silently
    # overwrites whatever hair_length was set to as a side effect of
    # assigning normal_factor/tangent_factor. So instead of fighting that
    # (which is what produced the "long straight hairs" bug to begin with -
    # the previous version set hair_length thinking it controlled length,
    # while tangent_factor/normal_factor silently overrode it), embrace it:
    # solve directly for the factor magnitude that gives piece.fur_length.
    factor = piece.fur_length / settings.hair_step
    settings.tangent_phase = 0.0
    if piece.has_grain_direction and proxy is not None:
        settings.normal_factor = 0.0
        settings.tangent_factor = factor

        # Comb direction alignment is done by rotating this piece's UV
        # data (_rotate_fur_uv), not settings.tangent_phase - see
        # _fur_alignment_rotation for why. Measures the actual resulting
        # direction with the freshly-(re)written, still-unrotated UV, then
        # rotates it by the measured correction - a real depsgraph
        # evaluation each time seams_to_fur.refresh_fur_preview runs, but that's a
        # one-click preview operator, not a per-frame cost.
        rotation_deg = _fur_alignment_rotation(context, sliced_obj, psys, piece, proxy)
        _rotate_fur_uv(sliced_obj, rotation_deg)

        # Blend a bit of normal_factor back in *after* alignment, so combed
        # fur lifts slightly off the surface instead of lying perfectly
        # flat. Safe to do after the fact: alignment only measured the
        # in-plane projected angle (_map_direction_to_flat discards any
        # out-of-plane component via its dot-product basis solve), and the
        # UV rotation that produced rotation_deg is a direction-only
        # transform, unaffected by how the two factors are later split. The
        # split itself is chosen so sqrt(normal_factor**2 + tangent_factor**2)
        # still equals factor exactly, preserving piece.fur_length.
        settings.normal_factor = factor * FUR_NORMAL_LIFT
        settings.tangent_factor = factor * math.sqrt(1.0 - FUR_NORMAL_LIFT**2)
    else:
        settings.normal_factor = factor
        settings.tangent_factor = 0.0

    # Reasonable general-purpose fur look (children for density without
    # simulating thousands of real strands, light clumping into tufts,
    # gentle roughness for texture so strands don't read as perfectly
    # straight/uniform) - values scaled off piece.fur_length rather than
    # fixed absolutes so they stay proportionate whether a piece's fur is
    # short fuzz or a longer shag.
    #
    # Deliberately NOT using settings.kink: kink_axis is only ever a fixed
    # X/Y/Z axis (confirmed empirically - there's no "relative to the
    # strand's own growth direction" option), but a piece's combed tangent
    # direction sweeps across many different absolute orientations on a
    # curved surface. Bending every strand relative to one fixed global
    # axis reads as a visibly inconsistent "center part" wherever the comb
    # direction's angle to that axis crosses zero (confirmed live: this
    # exact artifact on the top-of-head pieces), even though the underlying
    # tangent_factor comb direction is still correct - unlike clump
    # (pulls toward each child's real nearby-parent tip, a genuine 3D
    # position) or roughness (a per-particle random offset), which don't
    # depend on any fixed axis and stayed consistent.
    settings.child_type = "INTERPOLATED"
    settings.child_percent = 60  # viewport display child count, not just rendered_child_count
    settings.rendered_child_count = 60
    settings.render_step = 3
    settings.child_radius = piece.fur_length * 0.5
    settings.child_roundness = 0.9
    # clump_shape/roughness_*/radius_scale/root_radius/tip_radius below
    # match a manual, by-hand tuning of Piece 19's own particle system in
    # the live Blender UI (copied via direct property readback, not
    # re-guessed) - a noticeably tighter, thinner, more naturally-clumped
    # look than this function's own earlier guessed values. roughness_1/
    # roughness_2 are kept fur_length-proportional (0.13x/0.04x, matching
    # the ratio actually measured on Piece 19, fur_length 0.025) since
    # those genuinely scale with strand length; roughness_*_size/
    # clump_shape/radius_scale/root_radius/tip_radius are noise-frequency/
    # thickness controls, not lengths, so copied as flat constants exactly
    # as tuned.
    #
    # clump_factor scales with fur_length too (longer shag fur reads as
    # more visibly clumped into tufts than short fuzz) - 0.2 at the
    # property's own default fur_length (0.025), capped at 0.5 so very
    # long fur doesn't clump into a few big matted locks.
    settings.clump_factor = min(piece.fur_length * 8.0, 0.5)
    settings.clump_shape = 0.421
    settings.roughness_1 = piece.fur_length * 0.13
    settings.roughness_1_size = 0.042
    settings.roughness_endpoint = 0.0
    settings.roughness_2 = piece.fur_length * 0.04
    settings.roughness_2_size = 0.018
    settings.kink = "NO"
    # Also from Piece 19's manual tuning: root_radius/tip_radius equal (no
    # taper - a uniform-thickness strand, not tapering to a point) and a
    # much smaller radius_scale than this function used to set on its own.
    settings.radius_scale = 0.001
    settings.root_radius = 0.1
    settings.tip_radius = 0.1

    mat = _get_or_create_fur_material(piece)
    if mat.name not in (m.name for m in sliced_obj.data.materials if m):
        sliced_obj.data.materials.append(mat)
    try:
        settings.material_slot = mat.name
    except TypeError:
        settings.material = sliced_obj.data.materials.find(mat.name) + 1


# --- Live fur-preview refresh hooks ------------------------------------------
#
# Called from property update callbacks (properties.py) and from the grain-
# direction operator below, so an already-shown fur preview never goes
# visually stale after a color/length/direction edit without the user
# having to manually re-click Apply Fur Preview. Split into three tiers by
# cost, since color and fur_length are both edited via things that fire on
# every tick of a live drag (a color picker, a float slider) - only a grain-
# direction *commit* (which only ever writes the property once, on mouse
# release - see SEAMS_TO_FUR_OT_set_grain_direction._apply) can afford the full,
# depsgraph-evaluating re-sync.


def _piece_fur_modifier(mesh_obj, piece):
    """The live FUR_MODIFIER_NAME modifier for this piece, or None if the
    fur preview isn't currently shown for it - the shared "is there
    anything to even refresh" check for all three hooks below."""
    sliced_obj = bpy.data.objects.get(f"{mesh_obj.name}.{piece.name}.sliced")
    if sliced_obj is None or sliced_obj.type != "MESH":
        return None, None
    mod = sliced_obj.modifiers.get(FUR_MODIFIER_NAME)
    if mod is None or mod.particle_system is None:
        return None, None
    return sliced_obj, mod


def refresh_piece_fur_color(mesh_obj, piece):
    """Cheapest hook: just refreshes the fur material's color in place - no
    BVH/depsgraph work, safe to call on every tick of a live color-picker
    drag (piece.color's update callback does exactly that)."""
    sliced_obj, mod = _piece_fur_modifier(mesh_obj, piece)
    if mod is None:
        return
    _get_or_create_fur_material(piece)


def refresh_piece_fur_length(mesh_obj, piece):
    """Rescales normal_factor/tangent_factor to the piece's current
    fur_length only - deliberately skips rebuilding the UV or re-running
    _fur_alignment_rotation, since the comb *angle* doesn't depend on
    length, only magnitude does. Safe to call on every tick of a live
    fur_length slider drag (piece.fur_length's update callback does
    exactly that); a full _sync_fur_particle_system call here would be
    noticeably laggy mid-drag since alignment involves a real depsgraph
    evaluation."""
    sliced_obj, mod = _piece_fur_modifier(mesh_obj, piece)
    if mod is None:
        return
    settings = mod.particle_system.settings
    factor = piece.fur_length / settings.hair_step
    if piece.has_grain_direction:
        settings.normal_factor = factor * FUR_NORMAL_LIFT
        settings.tangent_factor = factor * math.sqrt(1.0 - FUR_NORMAL_LIFT**2)
    else:
        settings.normal_factor = factor
        settings.tangent_factor = 0.0


def refresh_piece_fur_full(context, mesh_obj, piece):
    """Full re-sync (UV rebuild + alignment calibration included) - for a
    grain-direction commit or reset, where the comb *angle* itself may
    have changed and the cheap magnitude-only rescale above isn't enough.
    Reuses the piece's own already-set particle count as the density
    (scaled 1:1, face_fraction=1.0) so this refresh alone doesn't shift the
    piece's fur density."""
    sliced_obj, mod = _piece_fur_modifier(mesh_obj, piece)
    if mod is None:
        return
    existing_count = mod.particle_system.settings.count or 1
    _sync_fur_particle_system(context, sliced_obj, piece, density=existing_count, face_fraction=1.0)


class SEAMS_TO_FUR_OT_refresh_fur_preview(Operator):
    """Add (or refresh) a hair particle system on each piece's Sliced
    Preview object (built automatically if not already shown) to preview
    a faux-fur look, each one following that piece's own grain direction
    (if set - see seams_to_fur.set_grain_direction), fur length, and color.
    Idempotent - re-running updates existing systems in place instead of
    stacking duplicates.

    Previews fur on the same cut/separated pieces the garment is actually
    constructed from, using a UV built directly from each piece's own
    flattened shape - not the source mesh's own UV map, which this never
    touches (and doesn't need to exist, or be usable for this, at all)."""

    bl_idname = "seams_to_fur.refresh_fur_preview"
    bl_label = "Refresh Fur Preview"
    bl_options = {"REGISTER", "UNDO"}

    density: IntProperty(name="Density", default=4000, min=10, soft_max=40000)

    @classmethod
    def poll(cls, context):
        obj = target.resolve_mesh_obj(context)
        return obj is not None and len(obj.seams_to_fur_pieces) > 0

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        if is_fur_stale(mesh_obj):
            return "Build the Sliced Preview then apply fur - can take a while on a complex mesh"
        return "Refresh fur particle settings on each piece - sliced preview already exists, so this is cheap"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        _ensure_sliced_preview(context, mesh_obj)

        sliced_objs = []
        for piece in mesh_obj.seams_to_fur_pieces:
            sliced_obj = bpy.data.objects.get(f"{mesh_obj.name}.{piece.name}.sliced")
            if sliced_obj is not None and sliced_obj.type == "MESH":
                sliced_objs.append((piece, sliced_obj))

        if not sliced_objs:
            self.report({"ERROR"}, "No sliced preview pieces to apply fur to")
            return {"CANCELLED"}

        total_faces = sum(len(sliced_obj.data.polygons) for _piece, sliced_obj in sliced_objs) or 1

        for piece, sliced_obj in sliced_objs:
            face_fraction = len(sliced_obj.data.polygons) / total_faces
            _sync_fur_particle_system(context, sliced_obj, piece, self.density, face_fraction)

        self.report(
            {"INFO"},
            "Fur preview applied - switch to Material Preview/Rendered shading to see strands",
        )
        return {"FINISHED"}


def is_fur_cached(mesh_obj):
    """True once at least one piece has a live fur particle system - what
    gates whether toggle/clear_fur_preview have anything to do."""
    return any(_piece_fur_modifier(mesh_obj, piece)[1] is not None for piece in mesh_obj.seams_to_fur_pieces)


def is_fur_visible(mesh_obj):
    return any(
        mod.show_viewport
        for piece in mesh_obj.seams_to_fur_pieces
        for mod in (_piece_fur_modifier(mesh_obj, piece)[1],)
        if mod is not None
    )


def is_fur_stale(mesh_obj):
    """True if refreshing fur right now would first need a real Sliced
    Preview rebuild (see _ensure_sliced_preview, which now shares
    preview.py's own signature-based staleness check - not just "does an
    object with the expected name exist per piece", which silently missed
    a seam edit reshaping a piece while its same-named Sliced Preview
    object stuck around unchanged), rather than just a cheap particle
    re-sync on an already-correct Sliced Preview."""
    prefix = preview.SLICED_COLLECTION_PREFIX
    return not preview.is_preview_cached(mesh_obj, prefix) or preview.is_preview_stale(mesh_obj, prefix)


class SEAMS_TO_FUR_OT_toggle_fur_preview(Operator):
    """Show/hide the already-refreshed fur preview - just flips each
    piece's particle-system modifier show_viewport/show_render, never
    recomputes/rebuilds anything. Only available once a refresh has
    actually cached something (see is_fur_cached)."""

    bl_idname = "seams_to_fur.toggle_fur_preview"
    bl_label = "Toggle Fur Preview"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        if not is_fur_cached(mesh_obj):
            return "No fur preview built yet - use Refresh first"
        return "Hide the fur preview" if is_fur_visible(mesh_obj) else "Show the fur preview"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        if not is_fur_cached(mesh_obj):
            self.report({"WARNING"}, "Fur preview hasn't been built yet - use Refresh first")
            return {"CANCELLED"}

        now_visible = not is_fur_visible(mesh_obj)
        for piece in mesh_obj.seams_to_fur_pieces:
            _sliced_obj, mod = _piece_fur_modifier(mesh_obj, piece)
            if mod is not None:
                mod.show_viewport = now_visible
                mod.show_render = now_visible
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_clear_fur_preview(Operator):
    """Remove the fur particle system from every piece's Sliced Preview
    object (removing a PARTICLE_SYSTEM modifier cascades to remove its
    underlying particle system too - confirmed empirically). Leaves the
    Sliced Preview objects themselves, and the rest of their data, alone -
    this only undoes seams_to_fur.refresh_fur_preview, not
    seams_to_fur.clear_preview(kind='SLICED')."""

    bl_idname = "seams_to_fur.clear_fur_preview"
    bl_label = "Clear Fur Preview"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        if not is_fur_cached(mesh_obj):
            return "No fur preview to clear"
        return "Remove the fur particle systems and UV from every piece"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        removed = 0
        for piece in mesh_obj.seams_to_fur_pieces:
            sliced_obj = bpy.data.objects.get(f"{mesh_obj.name}.{piece.name}.sliced")
            if sliced_obj is None or sliced_obj.type != "MESH":
                continue
            mod = sliced_obj.modifiers.get(FUR_MODIFIER_NAME)
            if mod is not None:
                sliced_obj.modifiers.remove(mod)
                removed += 1
            uv_layer = sliced_obj.data.uv_layers.get(FUR_UV_LAYER)
            if uv_layer is not None:
                sliced_obj.data.uv_layers.remove(uv_layer)
        self.report({"INFO"}, f"Fur preview hidden ({removed} piece(s))")
        return {"FINISHED"}


_classes = (
    SEAMS_TO_FUR_OT_sync_piece_colors,
    SEAMS_TO_FUR_OT_apply_swatch_color,
    SEAMS_TO_FUR_OT_set_fur_length,
    SEAMS_TO_FUR_OT_set_grain_direction,
    SEAMS_TO_FUR_OT_reset_grain_direction,
    SEAMS_TO_FUR_OT_reset_all_grain_directions,
    SEAMS_TO_FUR_OT_toggle_grain_arrows,
    SEAMS_TO_FUR_OT_refresh_fur_preview,
    SEAMS_TO_FUR_OT_toggle_fur_preview,
    SEAMS_TO_FUR_OT_clear_fur_preview,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
