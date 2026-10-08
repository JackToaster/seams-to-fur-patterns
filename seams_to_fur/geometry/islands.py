"""Seam curve resolution to mesh edges, and island isolation.

A seam curve is a skeleton Mesh object (vertices + edges, no faces - see
operators.seam_curve and geometry.curve_display for why). Resolution reads
its *evaluated* geometry (respecting any Mirror/Array/etc. modifiers the
user added directly to it), resamples and surface-projects it exactly the
way geometry.curve_display builds the visible tube preview, and then
actually *bisects* the target mesh's faces along that path
(geometry.mesh_cut) - inserting new geometry precisely where the cut
crosses each face, rather than snapping to whichever existing vertex
happens to be nearby. The latter produced jagged, unusable pattern edges
on any mesh not modeled with edge loops matching every seam; real
bisection also means the preview tube and the actual cut are the same
geometry by construction; they can't visually disagree. Working edge-by-
edge (not tracing an ordered chain) means this handles any topology a user
might build in Edit Mode - open paths, loops, branches, multiple pieces -
without needing to know which.

Resolution always re-evaluates the curve fresh (no separate "bind" step or
cache): it's cheap (seam curves have few points), and correctness only
ever depends on whatever the user last edited them (or their modifiers) to.

The scene mesh itself is never mutated by anything in this module; callers
pass an evaluated BMesh copy, which *is* mutated (bisected) here.
"""

import uuid as uuid_mod
from collections import Counter

import bpy

from . import curve_display, mesh_cut


class TopologyError(Exception):
    """Raised when an island isn't valid disk topology for flattening."""


def resolve_curve_seam_edges(bm, curve_obj, mesh_obj):
    """Resolve a seam curve skeleton to a set of real BMEdge objects (not
    indices, which go stale the moment anything else touches bm; a caller
    resolving more than one seam curve against the same bm should
    accumulate these *objects* across calls, and only read their .index -
    after an ensure_lookup_table() - once every curve is done. BMEdge
    object identity itself stays a valid handle across later mutations
    elsewhere in bm even though .index doesn't), by evaluating the curve's
    current geometry fresh every call (see module docstring) - respecting
    any Mirror/Array/etc. modifiers on it - and actually cutting bm's faces
    along the same resampled, surface-projected path that drives the
    visible tube preview, by walking face-to-face
    (geometry.mesh_cut.cut_path_into_mesh - see its module docstring for
    why, versus two earlier, abandoned designs). Mutates bm.

    Returns every edge cut_path_into_mesh reports, including a
    zero-length one where two of its own insertions land on the same real
    point (e.g. bmesh.ops.connect_vert_pair, used internally to bridge two
    vertices that aren't on a shared/adjacent face, can insert a new
    vertex essentially on top of an existing one from earlier in the same
    walk) - such an edge is still exactly what the walk itself concluded
    should be a barrier there, and dropping it (the previous, coordinate-
    key-based version of this function did, since two coincident points
    round to an unusable key) left isolate_islands' flood-fill an
    ungated, single-point gap to leak two pattern pieces together through
    (confirmed on a real hood mesh)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts, edges = curve_display.evaluated_skeleton_geometry(curve_obj, depsgraph)
    if len(edges) < 1:
        raise TopologyError(f"Seam curve '{curve_obj.name}' needs at least one edge")

    segment_length = curve_display.segment_length_for(mesh_obj)

    oversized = mesh_cut.check_oversized_faces(bm, segment_length)
    if oversized:
        print(
            f"Seams to Fur Patterns: '{mesh_obj.name}' has {oversized} face(s) much larger than "
            f"seam curve '{curve_obj.name}''s resample spacing ({segment_length:.4g}) - "
            f"the cut may follow the seam's curvature less precisely than the "
            f"preview suggests there. Consider subdividing the mesh more."
        )

    # Build the projection surface from bm itself (not a fresh evaluation of
    # mesh_obj) so it matches whatever bm actually is right now - e.g.
    # flatten's material-thickness shell offset, applied to bm before this
    # runs, moves the surface outward from mesh_obj's own evaluated mesh.
    bvh = curve_display.build_surface_bvh_from_bmesh(bm)
    to_mesh_local = mesh_obj.matrix_world.inverted()

    # A point projected onto the surface exactly at a Mirror modifier's weld
    # is equidistant from both mirrored halves, so the nearest-surface query
    # above resolves it to whichever side wins essentially at random from
    # one resampled point to the next - zigzagging the walk between the two
    # halves' interiors instead of tracing the shared vertex chain that
    # actually connects them, leaving most of the real weld edges untouched
    # and the two "separated" pieces still fully connected through them
    # (confirmed on a real mesh: 40% of the weld's own edges never got
    # tagged as seam this way). Snapping any point already within the
    # modifier's own merge_threshold of the plane exactly onto it makes
    # get_or_insert's vertex-reuse reliably land on the weld's real,
    # already-shared vertices instead - see snap_to_mirror_plane for why
    # merge_threshold specifically is the right radius.
    mirror_point, mirror_normal, _has_mirror, mirror_snap_dist = curve_display.mirror_plane_for(mesh_obj)

    # A seam curve endpoint meant to reach the mesh's own edge (to fully
    # separate two pieces there) is rarely drawn pixel-perfect on it, so its
    # nearest-surface projection naturally lands just short of the true
    # boundary too - close enough that it's obviously *meant* to be the
    # boundary, but just far enough that get_or_insert's vertex-reuse search
    # doesn't reach it (confirmed on a real mesh: a curve endpoint 0.0008
    # units short of the boundary left the piece it should have separated
    # still fully connected there). Snapping only a curve's own *true*
    # endpoints (a degree-1 vertex in its skeleton - see the vertex_degree
    # use below) that land within a small fraction of the resample spacing
    # of a real open-boundary edge onto that edge fixes this without ever
    # redirecting an ordinary interior point that merely passes near some
    # unrelated part of the mesh's boundary.
    boundary_snap_dist = segment_length * 0.1
    boundary_edges_world = [
        (mesh_obj.matrix_world @ e.verts[0].co, mesh_obj.matrix_world @ e.verts[1].co)
        for e in bm.edges
        if len(e.link_faces) == 1
    ]
    vertex_degree = Counter()
    for a_idx, b_idx in edges:
        vertex_degree[a_idx] += 1
        vertex_degree[b_idx] += 1

    # One shared vertex_cache across every edge of this curve, so a point
    # shared between two edges - a branch/junction point in the original
    # skeleton - reuses the exact same bmesh vertex both times instead of
    # each edge's walk creating its own coincident-but-separate one. Edges
    # are otherwise independent (no need to trace connected runs/chains -
    # cut_path_into_mesh handles arbitrary topology this way).
    #
    # vertex_cache alone isn't quite enough for that, though: it only
    # reuses an existing vertex on an *exact* (rounded) position match, and
    # resampling "the same" shared skeleton point via two different edges -
    # different walk directions, different accumulated floating-point
    # error - can land on two coordinates that are obviously the same
    # point but differ in, say, the 7th decimal digit. At an ordinary
    # 2-edge join that's harmless (both resolve to the same rounded key
    # almost always), but at a true branch point (3+ edges meeting at one
    # skeleton vertex) confirmed on a real mesh to occasionally produce 2-3
    # *different* nearby vertices instead of one shared one, each
    # connected to only some of the branch's edges - leaving the rest of
    # the branch's own boundary untagged and the pieces it should separate
    # still connected. skeleton_point_cache pins each skeleton vertex's
    # projected (and mirror/boundary-snapped) position the first time any
    # edge computes it, so every other edge sharing that same skeleton
    # vertex reuses the identical coordinates - no floating-point drift
    # left for vertex_cache's exact-match reuse to miss.
    skeleton_point_cache = {}
    vertex_cache = {}
    seam_edges = set()
    for a_idx, b_idx in edges:
        edge_verts, _edge_edges = curve_display.resample_edges(verts, [(a_idx, b_idx)], segment_length)
        projected_world = curve_display.project_onto_surface(edge_verts, bvh, mesh_obj)
        projected_world = curve_display.snap_to_mirror_plane(
            projected_world, mirror_point, mirror_normal, mirror_snap_dist
        )
        if vertex_degree[a_idx] == 1:
            projected_world[0] = curve_display.snap_point_to_mesh_boundary(
                projected_world[0], boundary_edges_world, boundary_snap_dist
            )
        if vertex_degree[b_idx] == 1:
            projected_world[-1] = curve_display.snap_point_to_mesh_boundary(
                projected_world[-1], boundary_edges_world, boundary_snap_dist
            )
        if a_idx in skeleton_point_cache:
            projected_world[0] = skeleton_point_cache[a_idx]
        else:
            skeleton_point_cache[a_idx] = projected_world[0]
        if b_idx in skeleton_point_cache:
            projected_world[-1] = skeleton_point_cache[b_idx]
        else:
            skeleton_point_cache[b_idx] = projected_world[-1]
        points_local = [to_mesh_local @ p for p in projected_world]
        cut_edges = mesh_cut.cut_path_into_mesh(
            bm, points_local, vertex_cache=vertex_cache
        )
        seam_edges.update(cut_edges)

    return seam_edges


def isolate_islands(bm, seam_edge_indices):
    """Flood-fill faces into islands, treating seam_edge_indices as barriers.

    Returns a dict: face_index -> island_id (0-based, contiguous).
    """
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    seam_set = set(seam_edge_indices)

    face_island = {}
    island_id = 0
    for start_face in bm.faces:
        if start_face.index in face_island:
            continue
        stack = [start_face]
        face_island[start_face.index] = island_id
        while stack:
            f = stack.pop()
            for e in f.edges:
                if e.index in seam_set:
                    continue
                for nf in e.link_faces:
                    if nf.index not in face_island:
                        face_island[nf.index] = island_id
                        stack.append(nf)
        island_id += 1

    return face_island, island_id


def merge_small_islands(bm, face_island, seam_edge_indices, min_area):
    """Merges any island whose total face area is below min_area into
    whichever neighboring island it shares the most (seam) boundary with.

    A safety-net bodge for degenerate thin sliver islands (numerical
    artifacts of cutting, not intended pattern pieces) rather than
    something expected to trigger often. Small stray islands are folded
    back into their real neighbor and the seam edges between them are
    dropped, since they're
    no longer a genuine island boundary.

    Mutates face_island in place (island ids remain contiguous 0..N-1
    after merging/renumbering). Returns the updated seam edge index set.
    """
    bm.faces.ensure_lookup_table()
    seam_set = set(seam_edge_indices)

    island_faces = {}
    for f_idx, isl in face_island.items():
        island_faces.setdefault(isl, []).append(f_idx)
    island_area = {isl: sum(bm.faces[f].calc_area() for f in faces) for isl, faces in island_faces.items()}

    changed = True
    while changed:
        changed = False
        for isl, area in list(island_area.items()):
            if area >= min_area or len(island_area) <= 1:
                continue

            neighbor_votes = {}
            for f_idx in island_faces[isl]:
                for e in bm.faces[f_idx].edges:
                    if e.index not in seam_set:
                        continue
                    for nf in e.link_faces:
                        nf_isl = face_island.get(nf.index)
                        if nf_isl is not None and nf_isl != isl:
                            neighbor_votes[nf_isl] = neighbor_votes.get(nf_isl, 0) + 1
            if not neighbor_votes:
                continue

            target = max(neighbor_votes, key=neighbor_votes.get)
            for f_idx in island_faces[isl]:
                face_island[f_idx] = target
            island_faces[target].extend(island_faces[isl])
            island_area[target] += area
            del island_faces[isl]
            del island_area[isl]

            for f_idx in island_faces[target]:
                for e in bm.faces[f_idx].edges:
                    if e.index in seam_set and len({face_island.get(lf.index) for lf in e.link_faces}) <= 1:
                        seam_set.discard(e.index)
            changed = True

    remap = {old: new for new, old in enumerate(sorted(island_area.keys()))}
    for f_idx in list(face_island.keys()):
        face_island[f_idx] = remap[face_island[f_idx]]

    return seam_set


def split_island_for_flatten(bm, island_face_indices, seam_edge_indices):
    """Build the vertex/face lists to send to the flattening backend for one
    island, duplicating vertices along any seam edges that are *internal* to
    the island (e.g. a dart/slit cut that doesn't fully separate it into two
    islands) so the two sides of the cut can actually open up into a gap
    when flattened, instead of BFF seeing a fully-connected mesh and folding
    the "cut" shut.

    Boundary-of-island seam edges (where the seam separates this island from
    a *different* island) don't need special handling here - they're already
    single-sided since only this island's faces are included - but passing
    the full resolved seam edge set is harmless for those too: the
    wedge-grouping below degenerates to a single wedge whenever a vertex's
    other seam-adjacent faces simply aren't part of this island.
    """
    seam_set = set(seam_edge_indices)
    island_face_set = set(island_face_indices)

    vert_faces = {}
    for f_idx in island_face_indices:
        face = bm.faces[f_idx]
        for v in face.verts:
            vert_faces.setdefault(v.index, []).append(face)

    # (vert_index, face_index) -> wedge id, grouping a vertex's incident
    # island faces via face-adjacency that doesn't cross a seam edge at
    # that vertex - the same barrier-flood-fill idea as isolate_islands,
    # just scoped to one vertex's face fan instead of the whole mesh.
    wedge_of = {}
    for v_idx, faces in vert_faces.items():
        visited = {}
        next_wedge = 0
        for f in faces:
            if f.index in visited:
                continue
            stack = [f]
            visited[f.index] = next_wedge
            while stack:
                cur = stack.pop()
                for e in cur.edges:
                    if v_idx not in (e.verts[0].index, e.verts[1].index):
                        continue
                    if e.index in seam_set:
                        continue
                    for nf in e.link_faces:
                        if nf.index in island_face_set and nf.index not in visited:
                            visited[nf.index] = next_wedge
                            stack.append(nf)
            next_wedge += 1
        for f_idx2, wid in visited.items():
            wedge_of[(v_idx, f_idx2)] = wid

    local_index = {}
    verts_list = []
    faces_local = []
    for f_idx in island_face_indices:
        face = bm.faces[f_idx]
        local_face = []
        for v in face.verts:
            key = (v.index, wedge_of[(v.index, f_idx)])
            if key not in local_index:
                local_index[key] = len(verts_list)
                verts_list.append(tuple(v.co))
            local_face.append(local_index[key])
        faces_local.append(local_face)

    return verts_list, faces_local


def validate_island_topology(bm, face_indices):
    """Check that the given faces form a single disk (genus-0, one boundary
    loop, manifold). Raises TopologyError with a human-readable reason if
    not, rather than letting bad topology reach the flattening backend.
    """
    bm.faces.ensure_lookup_table()
    faces = [bm.faces[i] for i in face_indices]
    face_set = set(face_indices)

    verts = set()
    edges = {}
    for f in faces:
        for v in f.verts:
            verts.add(v.index)
        for e in f.edges:
            edges[e.index] = edges.get(e.index, 0) + 1

    v_count = len(verts)
    e_count = len(edges)
    f_count = len(faces)

    # Non-manifold within the island: an edge shared by >2 island faces.
    for e_idx, count in edges.items():
        if count > 2:
            raise TopologyError(
                f"Island has a non-manifold edge (shared by {count} faces); "
                f"clean up the mesh before flattening"
            )

    # Boundary edges are those touched by exactly one island face.
    boundary_edge_count = sum(1 for c in edges.values() if c == 1)
    if boundary_edge_count == 0:
        raise TopologyError(
            "Island has no boundary (it wraps around closed, e.g. an "
            "un-seamed sphere) - add a seam curve to cut it open"
        )

    # Disk check: Euler characteristic V - E + F should be 1 for a single
    # disk with one boundary loop.
    euler = v_count - e_count + f_count
    if euler != 1:
        raise TopologyError(
            f"Island is not a single disk (Euler characteristic {euler}, "
            f"expected 1) - it likely has a hole or multiple boundary loops; "
            f"add another seam curve to split it further"
        )

    return True


def _island_centroid(bm, face_indices):
    from mathutils import Vector

    total = Vector((0.0, 0.0, 0.0))
    for f_idx in face_indices:
        total += bm.faces[f_idx].calc_center_median()
    return total / len(face_indices)


def _island_hash(bm, face_indices):
    """A fingerprint of one island's cut geometry - its faces and their
    vertex positions (rounded, so float noise between two identical
    recomputes can't make them differ). Changes whenever the piece's
    actual shape would: a seam moving, the source mesh being edited, or a
    modifier upstream changing its result."""
    import hashlib

    bm.faces.ensure_lookup_table()
    digest = hashlib.sha1()
    for face_idx in sorted(face_indices):
        for vert in bm.faces[face_idx].verts:
            digest.update(("%.6f,%.6f,%.6f;" % tuple(vert.co)).encode())
        digest.update(b"|")
    return digest.hexdigest()


def sync_piece_settings(obj, bm, face_island, island_count):
    """Reconcile obj.seams_to_fur_pieces with the freshly computed islands, keeping
    existing UUIDs (and therefore offset/color/etc settings) for islands
    that are spatially close to a previous piece's sample point.

    Matching is by nearest 3D centroid rather than a persisted per-face
    attribute: islands are computed from the *evaluated* (post-modifier)
    mesh, whose face indices/count aren't stable across bakes when a
    generative modifier like Subdivision Surface is in the stack, so a
    face-index-keyed attribute on the base mesh can't be used to track
    identity here.
    """
    island_faces = {}
    for face_idx, island_id in face_island.items():
        island_faces.setdefault(island_id, []).append(face_idx)

    island_centroids = {
        island_id: _island_centroid(bm, faces) for island_id, faces in island_faces.items()
    }
    island_hashes = {island_id: _island_hash(bm, faces) for island_id, faces in island_faces.items()}
    from mathutils import Vector

    prev_points = [(p.uuid, Vector(p.sample_point)) for p in obj.seams_to_fur_pieces if p.uuid]

    # Greedy nearest-centroid matching: repeatedly pick the closest
    # (island, previous piece) pair, consuming both sides, until no pairs
    # remain. Good enough for the common case of one incremental seam edit
    # at a time; a full optimal assignment isn't worth the complexity here.
    candidates = []
    for island_id, centroid in island_centroids.items():
        for uuid_str, sample_point in prev_points:
            dist = (centroid - sample_point).length
            candidates.append((dist, island_id, uuid_str))
    candidates.sort(key=lambda c: c[0])

    island_to_uuid = {}
    used_uuids = set()
    for _dist, island_id, uuid_str in candidates:
        if island_id in island_to_uuid or uuid_str in used_uuids:
            continue
        island_to_uuid[island_id] = uuid_str
        used_uuids.add(uuid_str)

    new_pieces_data = []
    for island_id in range(island_count):
        piece_uuid = island_to_uuid.get(island_id)
        if piece_uuid is None:
            piece_uuid = str(uuid_mod.uuid4())
        new_pieces_data.append((island_id, piece_uuid))

    # Snapshot into plain dicts before clearing - obj.seams_to_fur_pieces.clear()
    # invalidates any live PropertyGroup references still held afterward,
    # so holding onto the RNA items themselves (rather than copied values)
    # here would silently read back garbage/default values below.
    old_by_uuid = {
        p.uuid: {
            "name": p.name,
            "offset_mm": p.offset_mm,
            "color": tuple(p.color),
            "grain_direction": tuple(p.grain_direction),
            "grain_anchor": tuple(p.grain_anchor),
            "has_grain_direction": p.has_grain_direction,
            "fur_length": p.fur_length,
            "flattened_object": p.flattened_object,
            "cut_line_object": p.cut_line_object,
            "error_message": p.error_message,
            "flatten_dirty": p.flatten_dirty,
            "baked_island_hash": p.baked_island_hash,
        }
        for p in obj.seams_to_fur_pieces
    }

    # Names for genuinely new pieces (no matched old UUID) must be unique
    # among *all* current pieces, not just derived from this bake's own
    # island_id: island_id numbering isn't a stable identity across re-bakes
    # (islands appear/disappear/reorder as the mesh or seam curves change),
    # so two different re-bakes each naming a different *new* piece
    # "Piece {island_id + 1}" could otherwise independently pick the same
    # number once still-matched, differently-named pieces are interspersed -
    # producing visible duplicate names in the pieces panel. Seed the used
    # set with the names the matched (retained) pieces are about to keep, and
    # hand each new piece the next unused "Piece N".
    used_names = {
        old_by_uuid[piece_uuid]["name"]
        for _island_id, piece_uuid in new_pieces_data
        if piece_uuid in old_by_uuid
    }

    def _next_unused_name():
        n = 1
        while f"Piece {n}" in used_names:
            n += 1
        name = f"Piece {n}"
        used_names.add(name)
        return name

    obj.seams_to_fur_pieces.clear()
    for island_id, piece_uuid in new_pieces_data:
        entry = obj.seams_to_fur_pieces.add()
        entry.piece_id = island_id
        entry.uuid = piece_uuid
        entry.sample_point = island_centroids[island_id]
        old = old_by_uuid.get(piece_uuid)
        if old is not None:
            entry.name = old["name"]
            entry.offset_mm = old["offset_mm"]
            entry.color = old["color"]
            entry.grain_direction = old["grain_direction"]
            entry.grain_anchor = old["grain_anchor"]
            entry.has_grain_direction = old["has_grain_direction"]
            entry.fur_length = old["fur_length"]
            entry.flattened_object = old["flattened_object"]
            entry.cut_line_object = old["cut_line_object"]
            entry.error_message = old["error_message"]
            entry.baked_island_hash = old["baked_island_hash"]
        else:
            entry.name = _next_unused_name()
        entry.island_hash = island_hashes[island_id]
        # Recomputing islands happens for plenty of reasons that don't
        # change a piece (a preview or color sync re-runs this whole
        # pipeline) - only a piece whose actual cut geometry now differs
        # from what it was last baked from needs re-flattening. A piece
        # already marked dirty (e.g. by a seam edit) stays dirty until it
        # is actually re-baked.
        entry.flatten_dirty = (
            old is None or old["flatten_dirty"] or entry.baked_island_hash != entry.island_hash
        )


def _chain_edges_into_runs(edges):
    """Groups a flat set of BMEdge (all known to lie on the same piece-to-
    piece boundary) into ordered polyline runs by walking shared vertices.
    Handles simple open chains and (starting from an arbitrary vertex)
    closed loops; a real branching junction (three-plus pieces meeting at
    one point) would produce more than one run through the branch point
    rather than a single correct traversal - an accepted MVP limitation,
    same spirit as the notch-placement rules that consume this."""
    adjacency = {}
    for edge in edges:
        v0, v1 = edge.verts
        adjacency.setdefault(v0, []).append(v1)
        adjacency.setdefault(v1, []).append(v0)

    def edge_key(a, b):
        return frozenset((a.index, b.index))

    visited_edges = set()
    endpoints = [v for v, neighbors in adjacency.items() if len(neighbors) == 1]
    other_verts = [v for v in adjacency if v not in endpoints]

    runs = []
    for start in endpoints + other_verts:
        for first_neighbor in list(adjacency.get(start, [])):
            key = edge_key(start, first_neighbor)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            run = [start, first_neighbor]
            cur = first_neighbor
            while True:
                nexts = [n for n in adjacency.get(cur, []) if edge_key(cur, n) not in visited_edges]
                if not nexts or cur is start:
                    break
                nxt = nexts[0]
                visited_edges.add(edge_key(cur, nxt))
                run.append(nxt)
                cur = nxt
            runs.append(run)
    return runs


def find_seam_partners(bm, face_island, seam_edge_indices):
    """Groups resolved seam edges that separate two *different* islands
    (the sewn boundary between two pattern pieces, as opposed to an
    internal dart/slit seam or the mesh's own true boundary, which
    split_island_for_flatten already handles separately) into contiguous
    runs per (island_a, island_b) pair - the piece-to-piece adjacency data
    notch placement needs, which nothing has computed until now
    (SeamsToFurPieceSettings.seam_partners has sat reserved-but-empty since
    Phase 1).

    Returns {(island_a, island_b): [run, run, ...]} with island_a <
    island_b, each run an ordered list of bm.verts.co (mathutils.Vector,
    in the evaluated mesh's own local space) tracing that shared edge
    chain from one end to the other. Positions, not vertex indices -
    see SeamsToFurSeamPartner's docstring for why a raw index doesn't survive
    the flatten pipeline's own vertex duplication."""
    bm.edges.ensure_lookup_table()

    edges_by_pair = {}
    for e_idx in seam_edge_indices:
        edge = bm.edges[e_idx]
        faces = edge.link_faces
        if len(faces) != 2:
            continue
        isl0 = face_island.get(faces[0].index)
        isl1 = face_island.get(faces[1].index)
        if isl0 is None or isl1 is None or isl0 == isl1:
            continue
        pair = (isl0, isl1) if isl0 < isl1 else (isl1, isl0)
        edges_by_pair.setdefault(pair, []).append(edge)

    result = {}
    for pair, edges in edges_by_pair.items():
        runs = _chain_edges_into_runs(edges)
        result[pair] = [[v.co.copy() for v in run] for run in runs]
    return result


def populate_seam_partners(obj, bm, face_island, seam_edge_indices):
    """Computes find_seam_partners and writes the result into each piece's
    own obj.seams_to_fur_pieces[i].seam_partners - call after sync_piece_settings
    so piece_id -> uuid is current. Always a full clear+rewrite (like
    sample_point, this is recomputed fresh every rebake, never carried
    over from old data)."""
    partners_by_island = find_seam_partners(bm, face_island, seam_edge_indices)
    piece_by_id = {p.piece_id: p for p in obj.seams_to_fur_pieces}

    for piece in obj.seams_to_fur_pieces:
        piece.seam_partners.clear()

    for (island_a, island_b), runs in partners_by_island.items():
        piece_a = piece_by_id.get(island_a)
        piece_b = piece_by_id.get(island_b)
        if piece_a is None or piece_b is None:
            continue
        for run in runs:
            for piece, partner in ((piece_a, piece_b), (piece_b, piece_a)):
                entry = piece.seam_partners.add()
                entry.partner_piece_uuid = partner.uuid
                for co in run:
                    point = entry.points.add()
                    point.co = co
