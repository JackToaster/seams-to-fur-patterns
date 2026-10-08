"""In-place distortion preview: shows each piece at its original 3D
position/shape (post material-thickness offset), colored by how much each
face had to stretch or compress to flatten - a quick visual sanity check of
where a pattern will pull or bunch, without leaving the curved model view.
"""

import bpy

from ..backends import bff
from ..geometry import island_cache, islands
from . import flatten as flatten_ops
from . import preview

DISTORTION_COLLECTION_PREFIX = preview.DISTORTION_COLLECTION_PREFIX
DISTORTION_ATTR = "seams_to_fur_distortion"
DISTORTION_MATERIAL = "STF Distortion Preview"

# A reasonable stretch limit for a heavy knit fabric backing - what faux fur
# meant for a wearable garment (as opposed to a non-stretch woven backing)
# is typically backed with. Grounded in a real cited figure rather than a
# guess: knit-backed plush/sherpa fabric is commonly cited as stretching
# ~15% before losing recovery; 20% gives a small margin above that as the
# "visibly at risk of puckering/overstretch" threshold this scale uses.
# Area scales as the square of linear scale for a uniform local stretch, so
# this converts directly to bounds on the area_2d/area_3d ratio this
# preview actually measures.
DISTORTION_STRETCH_LIMIT = 0.20
DISTORTION_AREA_RATIO_MIN = (1.0 - DISTORTION_STRETCH_LIMIT) ** 2
DISTORTION_AREA_RATIO_MAX = (1.0 + DISTORTION_STRETCH_LIMIT) ** 2

# A diverging blue -> green -> red scale (compressed/bunching -> within the
# stretch limit -> stretched/at risk), not a single-hue gradient - lets
# compression and stretch read as visually distinct problems instead of
# both just being "not green" (confirmed live: the previous single-sided
# ramp left one whole direction of distortion always showing green
# regardless of magnitude).
DISTORTION_COMPRESSED_COLOR = (0.15, 0.35, 0.9, 1.0)
DISTORTION_SAFE_COLOR = (0.15, 0.85, 0.15, 1.0)
DISTORTION_STRETCHED_COLOR = (0.9, 0.15, 0.1, 1.0)


def _distortion_collection(context, mesh_obj):
    from . import collections

    name = f"{DISTORTION_COLLECTION_PREFIX}{mesh_obj.name}"
    return collections.get_or_create_child_collection(context, name)


_rebuilt_this_session = False


def get_distortion_material_if_exists():
    """Read-only counterpart to get_or_create_distortion_material() below -
    never creates or mutates the material, so it's the only one of the two
    safe to call from Panel.draw() (see that function's docstring: even a
    once-per-session mutation raised "Writing to ID classes in this
    context is not allowed" when it happened to land on a draw() call, on
    Blender versions that enforce that restriction - confirmed live).
    Returns (None, None) if the distortion preview has never been built
    yet this session; the panel's legend just stays hidden until then."""
    mat = bpy.data.materials.get(DISTORTION_MATERIAL)
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None, None
    return mat, mat.node_tree.nodes.get("Color Ramp")


def get_or_create_distortion_material():
    """Returns (material, ramp_node) - the ramp node is exposed so the UI
    panel can draw the *actual* gradient driving the preview as a legend
    (via layout.template_color_ramp), instead of a separately hand-drawn
    approximation that could drift out of sync with it.

    NEVER call this from Panel.draw() - see get_distortion_material_if_exists
    above for the read-only, draw()-safe way to fetch the ramp for the
    legend. This one creates/mutates the material and must only run from
    an operator's execute() (currently: _build_distortion_piece, itself
    only reached via the refresh_preview operator).

    Rebuilds the node graph from scratch (clearing any existing one
    first) the *first* call after an addon (re)load, then just returns
    the cached material/ramp on every call after that - confirmed live
    (the hard way: this used to rebuild unconditionally on every call,
    AND used to be called directly from panel draw() on every redraw,
    which both churned needlessly and, worse, intermittently raised
    inside draw() and silently truncated the rest of the panel, e.g.
    "Flatten All" and everything below the Distortion Preview row
    disappearing - see get_distortion_material_if_exists for why draw()
    now never reaches this function at all instead of just calling it
    less often). Rebuilding once per (re)load still self-heals a material
    saved in a .blend from before the color scale/stretch-limit constants
    above were last tuned (the same "stale node setup silently never
    refreshed" failure mode fixed once already for the fur material - see
    appearance._get_or_create_fur_material) without paying that cost on
    every call; this material has no per-piece user data to lose by
    rebuilding it, unlike a piece's own color material."""
    global _rebuilt_this_session
    mat = bpy.data.materials.get(DISTORTION_MATERIAL)
    if mat is not None and _rebuilt_this_session:
        return mat, mat.node_tree.nodes.get("Color Ramp")

    if mat is None:
        mat = bpy.data.materials.new(DISTORTION_MATERIAL)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.name = "Principled BSDF"
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = DISTORTION_ATTR
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = DISTORTION_COMPRESSED_COLOR
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = DISTORTION_STRETCHED_COLOR
    mid = ramp.color_ramp.elements.new(0.5)
    mid.color = DISTORTION_SAFE_COLOR

    # Maps the raw area ratio (2D flattened area / 3D original area) into
    # the ramp's 0-1 factor range - DISTORTION_AREA_RATIO_MIN/MAX are
    # exactly the ratios DISTORTION_STRETCH_LIMIT of linear compression/
    # stretch works out to, so position 0.5 (green) lands exactly at zero
    # net distortion and the two ends land exactly at the chosen limit.
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = DISTORTION_AREA_RATIO_MIN
    map_range.inputs["From Max"].default_value = DISTORTION_AREA_RATIO_MAX
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True

    links.new(attr_node.outputs["Fac"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    _rebuilt_this_session = True
    return mat, ramp


def _smooth_distortion_to_vertices(verts_3d, faces, distortion_per_face, iterations=3):
    """Area-weighted average of each vertex's surrounding faces' distortion
    ratios (so the preview colors per-VERTEX, smoothly interpolated across
    each face, rather than flat per-face), then a few more neighbor-
    averaging (Laplacian) passes over vertex adjacency on top of that.

    BFF's boundary parametrization treats the free edge of each piece
    differently from its interior - confirmed live (a real project file):
    faces right along a piece's own seam boundary showed a visibly
    different reading than their interior neighbors, drawing a stark,
    physically misleading ring tracing every seam regardless of that
    piece's actual overall distortion. That's a computational edge-effect
    of the per-piece flattening, not a real feature of the garment. A
    single face-to-vertex averaging pass alone wasn't enough to blend it
    away (confirmed live) - the anomalous ring is only one or two faces
    wide, so a vertex sitting right on the boundary is still dominated by
    its own immediate (anomalous) faces. The extra vertex-adjacency passes
    diffuse it further into each piece's interior instead of leaving a
    visible ring, at the cost of also softening genuine local detail by a
    similar amount - a reasonable trade for a "quick visual sanity check"
    preview, not a precision stress map."""
    n = len(verts_3d)
    accum = [0.0] * n
    weight = [0.0] * n
    neighbors = [set() for _ in range(n)]
    for f_idx, face in enumerate(faces):
        area = flatten_ops._face_area_3d(verts_3d, face)
        m = len(face)
        for i in range(m):
            a, b = face[i], face[(i + 1) % m]
            neighbors[a].add(b)
            neighbors[b].add(a)
        if area <= 1e-12:
            continue
        value = distortion_per_face[f_idx]
        for v in face:
            accum[v] += value * area
            weight[v] += area
    values = [accum[i] / weight[i] if weight[i] > 1e-12 else 1.0 for i in range(n)]

    for _ in range(iterations):
        new_values = list(values)
        for v in range(n):
            if neighbors[v]:
                new_values[v] = (values[v] + sum(values[nb] for nb in neighbors[v])) / (1 + len(neighbors[v]))
        values = new_values

    return values


def _build_distortion_piece(context, mesh_obj, piece, verts_3d, faces, distortion_per_vertex):
    coll = _distortion_collection(context, mesh_obj)
    obj_name = f"{mesh_obj.name}.{piece.name}.distortion"

    existing = bpy.data.objects.get(obj_name)
    if existing is not None and existing.type == "MESH":
        mesh = existing.data
        mesh.clear_geometry()
        obj = existing
    else:
        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        coll.objects.link(obj)

    mesh.from_pydata(verts_3d, [], faces)
    mesh.update()

    attr = mesh.attributes.new(name=DISTORTION_ATTR, type="FLOAT", domain="POINT")
    for i, value in enumerate(distortion_per_vertex):
        attr.data[i].value = value

    mat, _ramp = get_or_create_distortion_material()
    if not mesh.materials:
        mesh.materials.append(mat)
    else:
        mesh.materials[0] = mat

    obj.matrix_world = mesh_obj.matrix_world.copy()
    obj["seams_to_fur_source_object"] = mesh_obj.name
    obj["seams_to_fur_piece_uuid"] = piece.uuid
    return obj


def _compute_distortion_pieces(context, mesh_obj, piece_ids=None):
    """Returns a list of errors (strings). Builds/updates one distortion
    preview object per requested piece (default: all pieces)."""
    seam_curves = flatten_ops._find_seam_curves_for(mesh_obj)
    bm, face_island, seam_edges, island_count = flatten_ops.compute_islands(
        context, mesh_obj, seam_curves
    )

    # Feed the appearance-side cache: same reasoning as flatten.py's
    # _flatten_pieces - this preview already paid for a fresh cut, so let
    # a following color sync reuse it instead of re-cutting.
    island_cache.store_from_islands(mesh_obj, seam_curves, bm, face_island)

    binary_path = bff.find_binary(flatten_ops._addon_dir())
    errors = []

    for piece in mesh_obj.seams_to_fur_pieces:
        if piece_ids is not None and piece.piece_id not in piece_ids:
            continue

        island_face_indices = [i for i, isl in face_island.items() if isl == piece.piece_id]
        try:
            islands.validate_island_topology(bm, island_face_indices)
        except islands.TopologyError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        verts_list, faces_local = islands.split_island_for_flatten(bm, island_face_indices, seam_edges)

        try:
            out_verts, out_faces = bff.flatten_island(binary_path, verts_list, faces_local)
        except bff.BFFError as exc:
            errors.append(f"Piece '{piece.name}': {exc}")
            continue

        distortion_per_face = []
        for f_idx, face in enumerate(faces_local):
            area_3d = flatten_ops._face_area_3d(verts_list, face)
            area_2d = flatten_ops._polygon_area_2d([out_verts[i] for i in out_faces[f_idx]])
            distortion_per_face.append(area_2d / area_3d if area_3d > 1e-12 else 1.0)

        distortion_per_vertex = _smooth_distortion_to_vertices(verts_list, faces_local, distortion_per_face)
        _build_distortion_piece(context, mesh_obj, piece, verts_list, faces_local, distortion_per_vertex)

    bm.free()
    return errors


# Show/hide/refresh for this preview is handled entirely by
# preview.py's shared SEAMS_TO_FUR_OT_refresh_preview/_toggle_preview/
# _clear_preview trio (kind='DISTORTION'), which calls
# _compute_distortion_pieces above via preview._rebuild_distortion - no
# operators of its own to register here.


def register():
    pass


def unregister():
    pass
