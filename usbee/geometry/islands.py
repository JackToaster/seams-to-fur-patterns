"""Seam curve binding, resolution to mesh edges, and island isolation.

Phase 1 scope: seam curves are bound to the surface via raycast at bind time
(barycentric-ish nearest-face binding), then resolved at bake time by
snapping each bound point to its nearest mesh vertex and walking a shortest
edge path between consecutive snapped points. This is a deliberate scope
reduction from true continuous curve-to-mesh bisection (see plan) - it keeps
seam curves editable without requiring users to pick existing edges, while
avoiding the much harder arbitrary-line mesh-cutting problem.

The scene mesh is never mutated by anything in this module; callers pass an
evaluated BMesh copy.
"""

import heapq
import json
import uuid as uuid_mod

import bmesh
import bpy
from mathutils.bvhtree import BVHTree

BIND_DATA_PROP = "usbee_bind_data"
PIECE_ID_ATTR = "usbee_piece_id"
SEAM_RESOLVED_ATTR = "usbee_seam_resolved"


class TopologyError(Exception):
    """Raised when an island isn't valid disk topology for flattening."""


def bind_curve_to_mesh(curve_obj, mesh_obj):
    """Raycast each spline point of curve_obj onto mesh_obj and store the
    binding (nearest face index + the hit point's local-space position, used
    to re-derive barycentric weights at resolve time) as a JSON blob on the
    curve object.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)

    bindings = []
    for spline in curve_obj.data.splines:
        points = spline.bezier_points if spline.type == "BEZIER" else spline.points
        for pt in points:
            # Project the curve point (world space via curve's own transform)
            # into the mesh object's local space, then find the nearest
            # surface point on the mesh.
            co_world = curve_obj.matrix_world @ pt.co.to_3d()
            co_mesh_local = mesh_obj.matrix_world.inverted() @ co_world
            hit_co, hit_normal, hit_face_idx, hit_dist = bvh.find_nearest(co_mesh_local)
            if hit_face_idx is None:
                bindings.append(None)
                continue
            bindings.append(
                {
                    "face_index": hit_face_idx,
                    "local_co": [hit_co.x, hit_co.y, hit_co.z],
                }
            )

    bm.free()
    eval_obj.to_mesh_clear()

    curve_obj[BIND_DATA_PROP] = json.dumps(bindings)


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
    """Resolve a bound seam curve to a set of bmesh edge indices on bm."""
    raw = curve_obj.get(BIND_DATA_PROP)
    if not raw:
        raise TopologyError(f"Seam curve '{curve_obj.name}' is not bound to a mesh yet")
    bindings = json.loads(raw)

    snapped_verts = []
    for b in bindings:
        if b is None:
            continue
        from mathutils import Vector

        local_co = Vector(b["local_co"])
        snapped_verts.append(_nearest_vert_on_mesh(bm, local_co))

    seam_edges = set()
    is_closed = False
    if curve_obj.data.splines and curve_obj.data.splines[0].use_cyclic_u:
        is_closed = True

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


def sync_piece_settings(obj, face_island, island_count):
    """Reconcile obj.usbee_pieces with the freshly computed islands, keeping
    existing UUIDs (and therefore offset/color/etc settings) for islands that
    substantially overlap a previous piece_id, via best-face-count-overlap
    matching rather than resetting on every re-bake.
    """
    mesh = obj.data
    prev_attr = mesh.attributes.get(PIECE_ID_ATTR)
    prev_face_to_piece = {}
    if prev_attr is not None:
        for poly in mesh.polygons:
            prev_face_to_piece[poly.index] = prev_attr.data[poly.index].value

    # island_id -> {prev_piece_id: overlap_count}
    overlap = {}
    for face_idx, island_id in face_island.items():
        prev_piece_id = prev_face_to_piece.get(face_idx)
        if prev_piece_id is None:
            continue
        bucket = overlap.setdefault(island_id, {})
        bucket[prev_piece_id] = bucket.get(prev_piece_id, 0) + 1

    prev_piece_id_to_uuid = {p.piece_id: p.uuid for p in obj.usbee_pieces}

    new_pieces_data = []
    used_uuids = set()
    for island_id in range(island_count):
        best_prev = None
        if island_id in overlap:
            best_prev = max(overlap[island_id].items(), key=lambda kv: kv[1])[0]
        piece_uuid = prev_piece_id_to_uuid.get(best_prev)
        if piece_uuid is None or piece_uuid in used_uuids:
            piece_uuid = str(uuid_mod.uuid4())
        used_uuids.add(piece_uuid)
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
        }
        for p in obj.usbee_pieces
    }
    obj.usbee_pieces.clear()
    for island_id, piece_uuid in new_pieces_data:
        entry = obj.usbee_pieces.add()
        entry.piece_id = island_id
        entry.uuid = piece_uuid
        old = old_by_uuid.get(piece_uuid)
        if old is not None:
            entry.name = old["name"]
            entry.offset_mm = old["offset_mm"]
            entry.color = old["color"]
            entry.grain_direction = old["grain_direction"]
            entry.flattened_object = old["flattened_object"]
            entry.cut_line_object = old["cut_line_object"]
        else:
            entry.name = f"Piece {island_id + 1}"
        entry.flatten_dirty = True


def write_piece_id_attribute(mesh, face_island):
    attr = mesh.attributes.get(PIECE_ID_ATTR)
    if attr is None:
        attr = mesh.attributes.new(name=PIECE_ID_ATTR, type="INT", domain="FACE")
    for poly in mesh.polygons:
        attr.data[poly.index].value = face_island.get(poly.index, -1)
