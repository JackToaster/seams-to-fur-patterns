"""Seam curve resolution to mesh edges, and island isolation.

Phase 1 scope: seam curves are resolved to mesh seam edges by evaluating the
curve's live (post-modifier) geometry - so a Shrinkwrap keeping it glued to
the surface, or Array/Mirror duplicating it, are respected automatically -
then snapping each resulting point to its nearest mesh vertex and walking a
shortest edge path between consecutive snapped points. This is a deliberate
scope reduction from true continuous curve-to-mesh bisection (see plan) - it
keeps seam curves editable without requiring users to pick existing edges,
while avoiding the much harder arbitrary-line mesh-cutting problem.

Resolution always re-evaluates the curve fresh (no separate "bind" step or
cache): correctness with live modifiers requires it, and it's cheap enough
(seam curves have few points) to not need caching in Phase 1.

The scene mesh is never mutated by anything in this module; callers pass an
evaluated BMesh copy.
"""

import heapq
import uuid as uuid_mod

import bpy


class TopologyError(Exception):
    """Raised when an island isn't valid disk topology for flattening."""


def _extract_curve_chains(curve_obj, depsgraph):
    """Evaluate curve_obj (with modifiers applied) and return a list of
    (is_closed, [Vector, ...]) chains in the curve object's local space -
    one chain per resulting spline, so Array/Mirror modifiers that produce
    multiple loops/paths from one source spline are all included.

    Bevel/fill (used to make the curve visibly thick in the viewport) turns
    to_mesh() into a tube/ribbon surface instead of a plain polyline, which
    would corrupt path extraction below - temporarily zero them for this
    evaluation only, then restore, so display and resolution don't conflict.
    """
    orig_bevel_depth = curve_obj.data.bevel_depth
    curve_obj.data.bevel_depth = 0.0
    depsgraph.update()
    try:
        eval_obj = curve_obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        verts = [v.co.copy() for v in eval_mesh.vertices]
        adjacency = {}
        for e in eval_mesh.edges:
            a, b = e.vertices[0], e.vertices[1]
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        eval_obj.to_mesh_clear()
    finally:
        curve_obj.data.bevel_depth = orig_bevel_depth
        depsgraph.update()

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    visited_edges = set()
    chains = []

    # Open chains: walk from each degree-1 vertex to its other end.
    endpoints = [v for v, nbrs in adjacency.items() if len(nbrs) == 1]
    for start in endpoints:
        if edge_key(start, adjacency[start][0]) in visited_edges:
            continue
        chain_indices = [start]
        current = start
        while True:
            next_v = next(
                (n for n in adjacency[current] if edge_key(current, n) not in visited_edges),
                None,
            )
            if next_v is None:
                break
            visited_edges.add(edge_key(current, next_v))
            chain_indices.append(next_v)
            current = next_v
        chains.append((False, [verts[i] for i in chain_indices]))

    # Closed loops: any vertices with remaining unvisited edges are cycles
    # (everything of degree 1 was already consumed by the open-chain pass).
    for start in adjacency:
        remaining = [n for n in adjacency[start] if edge_key(start, n) not in visited_edges]
        if not remaining:
            continue
        chain_indices = [start]
        current = start
        while True:
            next_v = next(
                (n for n in adjacency[current] if edge_key(current, n) not in visited_edges),
                None,
            )
            if next_v is None:
                break
            visited_edges.add(edge_key(current, next_v))
            if next_v == start:
                break
            chain_indices.append(next_v)
            current = next_v
        chains.append((True, [verts[i] for i in chain_indices]))

    return chains


def _nearest_vert_on_mesh(bm, local_co):
    best_vert = None
    best_dist_sq = None
    # BVH-accelerated nearest-vertex lookup would be preferable at scale;
    # Phase 1 keeps this straightforward since seam curves have few points.
    for v in bm.verts:
        d = (v.co - local_co).length_squared
        if best_dist_sq is None or d < best_dist_sq:
            best_dist_sq = d
            best_vert = v
    return best_vert


def _shortest_edge_path(bm, start_vert, end_vert):
    """Dijkstra over bmesh edges, weighted by edge length."""
    if start_vert is end_vert:
        return []

    dist = {start_vert.index: 0.0}
    prev_edge = {}
    visited = set()
    heap = [(0.0, start_vert.index)]
    verts_by_index = {v.index: v for v in bm.verts}

    while heap:
        d, vi = heapq.heappop(heap)
        if vi in visited:
            continue
        visited.add(vi)
        v = verts_by_index[vi]
        if v is end_vert:
            break
        for e in v.link_edges:
            other = e.other_vert(v)
            nd = d + e.calc_length()
            if other.index not in dist or nd < dist[other.index]:
                dist[other.index] = nd
                prev_edge[other.index] = (vi, e)
                heapq.heappush(heap, (nd, other.index))

    if end_vert.index not in dist:
        return None  # disconnected

    path_edges = []
    cur = end_vert.index
    while cur != start_vert.index:
        prev_idx, edge = prev_edge[cur]
        path_edges.append(edge)
        cur = prev_idx
    return path_edges


def resolve_curve_seam_edges(bm, curve_obj, mesh_obj):
    """Resolve a seam curve to a set of bmesh edge indices on bm.

    Evaluates curve_obj fresh (through its modifier stack) every call, so a
    Shrinkwrap keeping it glued to the surface or an Array/Mirror producing
    extra copies are always respected - see module docstring.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chains = _extract_curve_chains(curve_obj, depsgraph)
    if not chains:
        raise TopologyError(f"Seam curve '{curve_obj.name}' has no points")

    to_mesh_local = mesh_obj.matrix_world.inverted() @ curve_obj.matrix_world

    seam_edges = set()
    for is_closed, points in chains:
        if len(points) < 2:
            continue
        snapped_verts = [_nearest_vert_on_mesh(bm, to_mesh_local @ p) for p in points]

        pairs = list(zip(snapped_verts, snapped_verts[1:]))
        if is_closed and len(snapped_verts) > 2:
            pairs.append((snapped_verts[-1], snapped_verts[0]))

        for a, b in pairs:
            path = _shortest_edge_path(bm, a, b)
            if path is None:
                raise TopologyError(
                    f"Seam curve '{curve_obj.name}' has points that resolve to "
                    f"disconnected regions of the mesh"
                )
            for e in path:
                seam_edges.add(e.index)

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


def sync_piece_settings(obj, bm, face_island, island_count):
    """Reconcile obj.usbee_pieces with the freshly computed islands, keeping
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
    from mathutils import Vector

    prev_points = [(p.uuid, Vector(p.sample_point)) for p in obj.usbee_pieces if p.uuid]

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

    # Snapshot into plain dicts before clearing - obj.usbee_pieces.clear()
    # invalidates any live PropertyGroup references still held afterward,
    # so holding onto the RNA items themselves (rather than copied values)
    # here would silently read back garbage/default values below.
    old_by_uuid = {
        p.uuid: {
            "name": p.name,
            "offset_mm": p.offset_mm,
            "color": tuple(p.color),
            "grain_direction": tuple(p.grain_direction),
            "flattened_object": p.flattened_object,
            "cut_line_object": p.cut_line_object,
            "error_message": p.error_message,
        }
        for p in obj.usbee_pieces
    }
    obj.usbee_pieces.clear()
    for island_id, piece_uuid in new_pieces_data:
        entry = obj.usbee_pieces.add()
        entry.piece_id = island_id
        entry.uuid = piece_uuid
        entry.sample_point = island_centroids[island_id]
        old = old_by_uuid.get(piece_uuid)
        if old is not None:
            entry.name = old["name"]
            entry.offset_mm = old["offset_mm"]
            entry.color = old["color"]
            entry.grain_direction = old["grain_direction"]
            entry.flattened_object = old["flattened_object"]
            entry.cut_line_object = old["cut_line_object"]
            entry.error_message = old["error_message"]
        else:
            entry.name = f"Piece {island_id + 1}"
        entry.flatten_dirty = True
