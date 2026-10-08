"""In-scene debug previews of the ACTUAL topological cut, built from the
exact same evaluated+cut pipeline flatten uses (flatten.compute_islands) so
what the user sees here is the real geometry that gets flattened, not a
separate approximation. Two views, both placed at the source mesh's own 3D
position/orientation (like distortion.py's previews):

  - Cut preview (one object): the whole evaluated+triangulated+cut mesh with
    all islands still joined, but the resolved seam edges marked
    use_seam=True so the real cut shows as red seam edges in the viewport.
    Optionally each face is also colored by exact per-piece island
    membership (used by appearance.sync_piece_colors - no approximate
    nearest-centroid matching, so it is correct even with a Mirror
    modifier, where the two mirrored halves are genuinely distinct faces
    here unlike on the base mesh).

  - Sliced preview (one object per piece): each island split into its own
    separate still-3D (NOT flattened) object, colored by its piece color, so
    the user can see how the mesh got sliced into pieces before flattening.

Each is built by a shared refresh/toggle/clear operator trio
(SEAMS_TO_FUR_OT_refresh_preview/_toggle_preview/_clear_preview, parameterized
by a `kind` enum) instead of a separate show_*/hide_* pair per kind:

  - Refresh is the only path that ever re-runs compute_islands (the
    expensive cut). It's skipped in favor of just re-showing the existing
    result when the underlying source geometry/seams/modifiers haven't
    changed since the last refresh (see is_preview_stale) - a genuinely
    free no-op in that case, not just a relabeled rebuild.
  - Toggle only ever flips Collection.hide_viewport (confirmed live: this
    alone hides/shows every object inside a collection, no per-object
    hide_set needed) - never recomputes, and is only enabled once a
    refresh has actually cached something (see is_preview_cached).
  - Clear deletes the cached objects/collection outright (what hide_*
    used to do unconditionally on every hide).
"""

import bpy
from bpy.types import Operator

from ..geometry import islands
from . import flatten as flatten_ops
from . import target

CUT_PREVIEW_COLLECTION_PREFIX = "STF Cut — "
SLICED_COLLECTION_PREFIX = "STF Sliced — "
# distortion.py's own preview shares this same-shape-overlap concern but
# lives in a separate module (it predates this one) - defined here, the one
# place hide_same_shape_previews/any_same_shape_preview_visible need a
# single source of truth for all three, and imported back into distortion.py
# so there's still only one definition.
DISTORTION_COLLECTION_PREFIX = "STF Distortion — "

# All three occupy literally the same 3D space as the base mesh (unlike the
# flattened pattern pieces, which live elsewhere in the scene and never
# overlap it) - so at most one should ever be visible at once, and the base
# mesh itself needs to get out of the way too, or nothing under it is
# clickable/readable.
SAME_SHAPE_COLLECTION_PREFIXES = (
    CUT_PREVIEW_COLLECTION_PREFIX,
    DISTORTION_COLLECTION_PREFIX,
    SLICED_COLLECTION_PREFIX,
)

# kind -> (collection prefix, display label). The single source of truth for
# every operator/helper below that needs to go from "which preview" to
# "which collection" - and the one place to add a new kind (see
# appearance.py for the same pattern applied to fur preview).
PREVIEW_KINDS = {
    "CUT": (CUT_PREVIEW_COLLECTION_PREFIX, "Cut"),
    "SLICED": (SLICED_COLLECTION_PREFIX, "Sliced"),
    "DISTORTION": (DISTORTION_COLLECTION_PREFIX, "Distortion"),
}

_PREVIEW_ENUM_ITEMS = [(kind, label, "") for kind, (_prefix, label) in PREVIEW_KINDS.items()]

# Stored as a custom ID-property on the preview's own collection - the
# cut-input signature (island_cache.compute_signature) that was current at
# the moment this preview was last actually (re)built, so a later refresh
# can tell a real source change apart from "nothing to do".
PREVIEW_SIGNATURE_PROP = "seams_to_fur_preview_signature"


def _preview_collection(context, prefix, mesh_obj):
    from . import collections

    name = f"{prefix}{mesh_obj.name}"
    return collections.get_or_create_child_collection(context, name)


def _remove_preview_collection(prefix, mesh_obj):
    name = f"{prefix}{mesh_obj.name}"
    coll = bpy.data.collections.get(name)
    if coll is not None:
        for obj in list(coll.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll)


def get_preview_collection(prefix, mesh_obj):
    """The existing collection for this preview kind, or None if a refresh
    has never (yet) built one - never creates one; that's refresh's job."""
    return bpy.data.collections.get(f"{prefix}{mesh_obj.name}")


def is_preview_cached(mesh_obj, prefix):
    """True once a refresh has actually built something for this preview -
    what gates whether toggle/clear should even be clickable, since there's
    nothing for either of them to do otherwise."""
    coll = get_preview_collection(prefix, mesh_obj)
    return coll is not None and len(coll.objects) > 0


def is_preview_visible(mesh_obj, prefix):
    coll = get_preview_collection(prefix, mesh_obj)
    return coll is not None and len(coll.objects) > 0 and not coll.hide_viewport


def is_preview_stale(mesh_obj, prefix):
    """True if refreshing this preview right now would be a real recompute
    - either nothing is cached yet, or the source mesh/modifiers/seam
    curves/thickness have changed since the signature was last stamped onto
    this preview's collection (see PREVIEW_SIGNATURE_PROP) - as opposed to
    a free no-op that just re-shows an already-correct cached result."""
    coll = get_preview_collection(prefix, mesh_obj)
    if coll is None:
        return True
    stored_sig = coll.get(PREVIEW_SIGNATURE_PROP)
    if not stored_sig:
        return True
    from ..geometry import island_cache

    seam_curves = flatten_ops._find_seam_curves_for(mesh_obj)
    return stored_sig != island_cache.compute_signature(mesh_obj, seam_curves)


def set_collection_visible(mesh_obj, prefix, visible):
    """The cheap path: flips Collection.hide_viewport alone (confirmed live
    that this hides/shows every object inside at once - no per-object
    hide_set needed) - never recomputes anything, unlike a refresh."""
    coll = get_preview_collection(prefix, mesh_obj)
    if coll is not None:
        coll.hide_viewport = not visible


def hide_same_shape_previews(mesh_obj, except_prefix=None):
    """Hides (not deletes - see clear_preview for that) every same-shape
    preview (cut/distortion/sliced) other than except_prefix, so showing
    one never leaves another sitting visible in the exact same 3D space
    behind/inside it. Whatever was hidden stays cached, so switching back
    to it is instant."""
    for prefix in SAME_SHAPE_COLLECTION_PREFIXES:
        if prefix == except_prefix:
            continue
        set_collection_visible(mesh_obj, prefix, False)


def any_same_shape_preview_visible(mesh_obj):
    return any(is_preview_visible(mesh_obj, prefix) for prefix in SAME_SHAPE_COLLECTION_PREFIXES)


def set_base_mesh_hidden(mesh_obj, hidden):
    """Toggles the base mesh's viewport "eye" visibility - the same one
    Alt+H restores - so a same-shape preview isn't stuck behind/inside the
    base mesh (and unclickable because of it). Restored automatically once
    no same-shape preview remains (see the hide_* operators below).

    Also toggles each of the mesh's seam-curve tube displays (the always-
    show_in_front, beveled indicator tracing each drawn seam - see
    seam_curve.py) along with it. It stays visible (and, being
    show_in_front, draws right on top of everything) regardless of which
    same-shape preview is active - an unrelated overlay that happens to
    trace the same lines as a piece boundary, so it's worth keeping out of
    the way while a same-shape preview has the surface to itself. (Not
    the cause of the pale-ring distortion-preview artifact that turned out
    to be - see distortion._smooth_distortion_to_vertices - just a
    separate, real bit of visual clutter worth clearing regardless.)"""
    try:
        mesh_obj.hide_set(hidden)
    except RuntimeError:
        # Not in the current view layer (e.g. a headless test scene where
        # the object was created but never became part of any view layer's
        # visibility state) - fall back to the always-available flag.
        mesh_obj.hide_viewport = hidden

    from . import seam_curve

    for curve_obj in seam_curve._find_seam_curves_for(mesh_obj):
        tube_obj = seam_curve._tube_object_for(curve_obj)
        if tube_obj is None:
            continue
        try:
            tube_obj.hide_set(hidden)
        except RuntimeError:
            tube_obj.hide_viewport = hidden


def get_or_create_piece_material(piece):
    """One material per piece, keyed by the piece's stable UUID, colored from
    piece.color. Shared by every object that represents this piece (base
    mesh, cut preview, sliced preview, flattened object) so updating the
    color once here propagates everywhere the material is used."""
    mat_name = f"STF Piece {piece.uuid[:8]}"
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    color = tuple(piece.color)
    mat.diffuse_color = color
    if mat.use_nodes:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color
    return mat


def _rebuild_piece_material_slots(mesh_data, pieces):
    """Clear mesh_data's material slots and (re)add one per piece from its
    color. Returns slot_index_by_piece_id."""
    while mesh_data.materials:
        mesh_data.materials.pop()
    slot_index_by_piece_id = {}
    for piece in pieces:
        mesh_data.materials.append(get_or_create_piece_material(piece))
        slot_index_by_piece_id[piece.piece_id] = len(mesh_data.materials) - 1
    return slot_index_by_piece_id


def build_cut_preview_object(context, mesh_obj, bm, seam_edges, face_island=None):
    """(Re)build the single cut-preview object for mesh_obj from the cut
    BMesh: the whole mesh as one object at the source's world transform, with
    every resolved seam edge marked use_seam=True. If face_island is given,
    also (re)assigns a per-piece material to each face by *exact* island
    membership (face indices are preserved 1:1 from bm into the mesh here, so
    no approximate matching is needed). Returns the object.

    Does not free bm; the caller owns it.
    """
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    coll = _preview_collection(context, CUT_PREVIEW_COLLECTION_PREFIX, mesh_obj)
    obj_name = f"{mesh_obj.name}.cut"
    existing = bpy.data.objects.get(obj_name)
    if existing is not None and existing.type == "MESH":
        mesh = existing.data
        mesh.clear_geometry()
        while mesh.materials:
            mesh.materials.pop()
        obj = existing
    else:
        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        coll.objects.link(obj)

    verts = [tuple(v.co) for v in bm.verts]
    faces = [[v.index for v in f.verts] for f in bm.faces]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # from_pydata preserves vertex order, so a bm edge's endpoint indices
    # name the same mesh vertices - match by unordered index pair.
    seam_pairs = {frozenset((bm.edges[i].verts[0].index, bm.edges[i].verts[1].index)) for i in seam_edges}
    for edge in mesh.edges:
        if frozenset(edge.vertices) in seam_pairs:
            edge.use_seam = True

    if face_island is not None:
        slot_index_by_piece_id = _rebuild_piece_material_slots(mesh, mesh_obj.seams_to_fur_pieces)
        # mesh.polygons[i] corresponds to bm.faces[i] (order preserved).
        for poly in mesh.polygons:
            island_id = face_island.get(poly.index)
            if island_id in slot_index_by_piece_id:
                poly.material_index = slot_index_by_piece_id[island_id]

    mesh.update()
    obj.matrix_world = mesh_obj.matrix_world.copy()
    obj["seams_to_fur_source_object"] = mesh_obj.name
    return obj


def build_sliced_preview_objects(context, mesh_obj, bm, face_island, seam_edges, pieces):
    """(Re)build one still-3D object per piece: each island's faces split out
    (islands.split_island_for_flatten - the same split flatten/distortion
    use, minus the flatten/BFF step) into its own mesh at the source's world
    transform, colored by the piece's material. Returns the list of objects.

    Does not free bm; the caller owns it.
    """
    bm.faces.ensure_lookup_table()
    coll = _preview_collection(context, SLICED_COLLECTION_PREFIX, mesh_obj)

    built = []
    for piece in pieces:
        island_face_indices = [i for i, isl in face_island.items() if isl == piece.piece_id]
        if not island_face_indices:
            continue
        verts_list, faces_local = islands.split_island_for_flatten(bm, island_face_indices, seam_edges)

        obj_name = f"{mesh_obj.name}.{piece.name}.sliced"
        existing = bpy.data.objects.get(obj_name)
        if existing is not None and existing.type == "MESH":
            mesh = existing.data
            mesh.clear_geometry()
            while mesh.materials:
                mesh.materials.pop()
            obj = existing
        else:
            mesh = bpy.data.meshes.new(obj_name)
            obj = bpy.data.objects.new(obj_name, mesh)
            coll.objects.link(obj)

        mesh.from_pydata(verts_list, [], faces_local)
        mesh.update()
        mesh.materials.append(get_or_create_piece_material(piece))

        obj.matrix_world = mesh_obj.matrix_world.copy()
        obj["seams_to_fur_source_object"] = mesh_obj.name
        obj["seams_to_fur_piece_uuid"] = piece.uuid
        built.append(obj)
    return built


def _rebuild_cut(context, mesh_obj):
    bm, face_island, seam_edges, _count = flatten_ops.compute_islands(context, mesh_obj)
    build_cut_preview_object(context, mesh_obj, bm, seam_edges, face_island)
    bm.free()
    return []


def _rebuild_sliced(context, mesh_obj):
    bm, face_island, seam_edges, _count = flatten_ops.compute_islands(context, mesh_obj)
    build_sliced_preview_objects(context, mesh_obj, bm, face_island, seam_edges, mesh_obj.seams_to_fur_pieces)
    bm.free()
    return []


def _rebuild_distortion(context, mesh_obj):
    from . import distortion

    return distortion._compute_distortion_pieces(context, mesh_obj)


# kind -> (context, mesh_obj) -> list[str] errors. The actual expensive
# rebuild for each preview kind - only ever called from refresh_preview, and
# only when is_preview_stale says the last cached result is out of date.
_REBUILD_FUNCS = {
    "CUT": _rebuild_cut,
    "SLICED": _rebuild_sliced,
    "DISTORTION": _rebuild_distortion,
}


def _show_preview(context, mesh_obj, prefix):
    set_collection_visible(mesh_obj, prefix, True)
    hide_same_shape_previews(mesh_obj, except_prefix=prefix)
    set_base_mesh_hidden(mesh_obj, True)


def ensure_preview_fresh(context, mesh_obj, kind):
    """Rebuilds this preview's cached objects if stale or missing at all -
    the pure recompute half of SEAMS_TO_FUR_OT_refresh_preview.execute(),
    without the show/hide/report side effects, so other code that
    depends on one of these previews being genuinely up to date (not just
    "an object with the expected name happens to exist") can reuse the
    exact same staleness check instead of a separate, weaker one of its
    own drifting out of sync.

    This split exists because appearance.py's _ensure_sliced_preview (the
    fur/grain-direction machinery's own "make sure a Sliced Preview
    object exists per piece" helper) used to only check object existence
    by name, never whether the cut itself had changed since that object
    was built - confirmed live, the hard way: after editing a seam curve,
    fur applied via that existence-only check kept rendering on the
    stale, pre-edit shape for every piece the edit hadn't happened to
    leave untouched, while a piece whose shape genuinely didn't change
    still looked fine - "fur broken on all but two pieces" from the
    outside, not an obvious cache-staleness bug. Both now share this one
    signature-based check (see is_preview_stale) instead of two
    divergent implementations.

    Returns the list of errors _REBUILD_FUNCS[kind] produced (empty if
    nothing needed rebuilding)."""
    prefix, _label = PREVIEW_KINDS[kind]
    if is_preview_cached(mesh_obj, prefix) and not is_preview_stale(mesh_obj, prefix):
        return []

    from ..geometry import island_cache

    seam_curves = flatten_ops._find_seam_curves_for(mesh_obj)
    errors = _REBUILD_FUNCS[kind](context, mesh_obj)
    coll = get_preview_collection(prefix, mesh_obj)
    if coll is not None:
        coll[PREVIEW_SIGNATURE_PROP] = island_cache.compute_signature(mesh_obj, seam_curves)
    return errors


class SEAMS_TO_FUR_OT_refresh_preview(Operator):
    """(Re)build this preview from the current mesh/seams and show it. A
    real recompute unless the source geometry/seams haven't changed since
    the last refresh, in which case this is a free no-op that just re-shows
    the still-valid cached result"""

    bl_idname = "seams_to_fur.refresh_preview"
    bl_label = "Refresh Preview"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(items=_PREVIEW_ENUM_ITEMS)

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        prefix, label = PREVIEW_KINDS[properties.kind]
        if is_preview_cached(mesh_obj, prefix) and not is_preview_stale(mesh_obj, prefix):
            return f"{label} preview is already up to date - just re-shows the cached result, no recompute"
        return f"Recompute the {label} preview from the current mesh/seams - re-cuts the mesh, can take a while"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        prefix, label = PREVIEW_KINDS[self.kind]

        was_fresh = is_preview_cached(mesh_obj, prefix) and not is_preview_stale(mesh_obj, prefix)
        errors = ensure_preview_fresh(context, mesh_obj, self.kind)

        if errors:
            self.report({"WARNING"}, "; ".join(errors))
        _show_preview(context, mesh_obj, prefix)
        if was_fresh:
            self.report({"INFO"}, f"{label} preview already up to date")
        else:
            self.report({"INFO"}, f"{label} preview refreshed for '{mesh_obj.name}'")
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_toggle_preview(Operator):
    """Show/hide an already-refreshed preview - never recomputes anything,
    only available once a refresh has actually cached something"""

    bl_idname = "seams_to_fur.toggle_preview"
    bl_label = "Toggle Preview"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(items=_PREVIEW_ENUM_ITEMS)

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        prefix, label = PREVIEW_KINDS[properties.kind]
        if not is_preview_cached(mesh_obj, prefix):
            return f"No {label} preview built yet - use Refresh first"
        return f"Hide the {label} preview" if is_preview_visible(mesh_obj, prefix) else f"Show the {label} preview"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        prefix, label = PREVIEW_KINDS[self.kind]

        if not is_preview_cached(mesh_obj, prefix):
            self.report({"WARNING"}, f"{label} preview hasn't been built yet - use Refresh first")
            return {"CANCELLED"}

        if is_preview_visible(mesh_obj, prefix):
            set_collection_visible(mesh_obj, prefix, False)
            if not any_same_shape_preview_visible(mesh_obj):
                set_base_mesh_hidden(mesh_obj, False)
        else:
            _show_preview(context, mesh_obj, prefix)
        return {"FINISHED"}


class SEAMS_TO_FUR_OT_clear_preview(Operator):
    """Delete this preview's cached objects outright"""

    bl_idname = "seams_to_fur.clear_preview"
    bl_label = "Clear Preview"
    bl_options = {"REGISTER", "UNDO"}

    kind: bpy.props.EnumProperty(items=_PREVIEW_ENUM_ITEMS)

    @classmethod
    def poll(cls, context):
        return target.resolve_mesh_obj(context) is not None

    @classmethod
    def description(cls, context, properties):
        mesh_obj = target.resolve_mesh_obj(context)
        if mesh_obj is None:
            return None
        prefix, label = PREVIEW_KINDS[properties.kind]
        if not is_preview_cached(mesh_obj, prefix):
            return f"No {label} preview to clear"
        return f"Delete the cached {label} preview objects"

    def execute(self, context):
        mesh_obj = target.resolve_mesh_obj(context)
        prefix, _label = PREVIEW_KINDS[self.kind]
        _remove_preview_collection(prefix, mesh_obj)
        if not any_same_shape_preview_visible(mesh_obj):
            set_base_mesh_hidden(mesh_obj, False)
        return {"FINISHED"}


_classes = (
    SEAMS_TO_FUR_OT_refresh_preview,
    SEAMS_TO_FUR_OT_toggle_preview,
    SEAMS_TO_FUR_OT_clear_preview,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
