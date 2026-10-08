"""Pure-Python seam curve display: reads a seam curve skeleton object's
*evaluated* geometry (respecting whatever modifiers - Mirror, Array,
whatever - the user has added to it directly), resamples each edge,
projects the resampled points onto the target mesh's surface, and builds a
simple tube mesh around the result into a separate display object.

The skeleton itself is a plain Mesh (vertices + edges, no faces) that the
addon never adds modifiers to and never locks out of native Edit Mode -
see operators.seam_curve's module docstring for why: Mirror/Array on a
Mesh is one of Blender's most standard, reliable modifier use cases (it
was specifically Curve objects where Mirror behaved like it rasterized
toward mesh output), and native mesh Edit Mode (move/extrude/merge/delete
vertices) is exactly the tool a user should be able to rely on for
touching up points after the fact.

The tube itself still can't be a modifier on the skeleton - stacking our
own modifier there would reintroduce the exact ordering fragility this
design avoids by not adding any modifier of our own at all - so it's a
separate object we regenerate in plain Python/bmesh whenever the skeleton
(or its modifiers) change.
"""

import math

import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree

TUBE_RADIUS = 0.0025 / 3.0
RING_SEGMENTS = 8


def build_surface_bvh(mesh_obj, depsgraph):
    """Evaluated (post-modifier) surface, e.g. a Subdivision Surface
    result - not the low-poly base cage - so hugging/snapping matches the
    surface that will actually get flattened."""
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()
    eval_obj.to_mesh_clear()
    return bvh


def build_surface_bvh_from_bmesh(bm):
    """Same as build_surface_bvh, but built directly from an existing bmesh
    (already in local space of whatever object it came from) instead of
    re-evaluating the object - needed when bisection targets a bm that
    isn't identical to the object's own evaluated mesh, e.g. flatten's
    material-thickness shell offset is applied to bm *before* seam
    resolution runs. Projecting the seam path against the unshelled
    surface while bisecting the shelled one would mean the two disagree by
    the shell distance, which for a large thickness can be enough that a
    segment's small search margin finds no faces at all."""
    return BVHTree.FromBMesh(bm)


def segment_length_for(mesh_obj):
    return max(mesh_obj.dimensions) * 0.02 or 0.02


def evaluated_skeleton_geometry(curve_obj, depsgraph):
    """Returns (verts: list[Vector] world space, edges: list[(int, int)])
    from the skeleton object's *evaluated* geometry - i.e. after any
    Mirror/Array/etc. modifiers the user added directly to it. Only edges
    are read (ignoring any faces some modifier might incidentally add), so
    this is robust to whatever's in the modifier stack.
    """
    eval_obj = curve_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    mat = curve_obj.matrix_world
    verts = [mat @ v.co for v in eval_mesh.vertices]
    edges = [tuple(e.vertices) for e in eval_mesh.edges]
    eval_obj.to_mesh_clear()
    return verts, edges


def resample_edges(verts, edges, segment_length):
    """Subdivides each edge independently into ~segment_length-long pieces.
    Edge-wise (not chain-traced) so this works for any topology a user
    might build in Edit Mode - open paths, loops, branches, multiple
    disconnected pieces - without needing to trace connected chains.
    Returns a new (verts, edges) pair.
    """
    new_verts = []
    new_edges = []
    for a_idx, b_idx in edges:
        a, b = verts[a_idx], verts[b_idx]
        length = (b - a).length
        n_segments = max(1, round(length / segment_length))
        start = len(new_verts)
        for i in range(n_segments + 1):
            t = i / n_segments
            new_verts.append(a + (b - a) * t)
        for i in range(n_segments):
            new_edges.append((start + i, start + i + 1))
    return new_verts, new_edges


def resample_path(verts, path, segment_length):
    """Resamples one continuous chain of skeleton vertex indices (see
    geometry.mesh_cut.decompose_into_runs - a "run": a maximal
    non-branching walk between branch points/endpoints, or a full closed
    loop) into a single, arc-length-spaced point sequence.

    Unlike resample_edges, which treats every edge independently and so
    creates a *separate* point at each internal joint for each of the two
    edges meeting there, this produces exactly one point per joint - the
    whole point of working in "runs" rather than raw edges is to build a
    single, genuinely connected ribbon later, with no near-duplicate
    vertices at shared joints for the exact boolean solver to get
    confused by.

    If path is closed (path[0] == path[-1], a loop with no branch points
    at all), the returned list does *not* repeat the shared start/end
    point - callers should wrap the last point back to the first
    themselves (see mesh_cut.bisect_with_ribbon).
    """
    points = [verts[path[0]]]
    for i in range(len(path) - 1):
        a, b = verts[path[i]], verts[path[i + 1]]
        length = (b - a).length
        n_segments = max(1, round(length / segment_length))
        for k in range(1, n_segments + 1):
            t = k / n_segments
            points.append(a + (b - a) * t)

    closed = path[0] == path[-1]
    if closed:
        points.pop()  # last point duplicates points[0]
    return points, closed


def project_onto_surface(points, bvh, mesh_obj, bias_world=None):
    """Snaps each point (world space) to the nearest point on the target
    mesh's surface. bias_world, when given, nudges the *query* position
    (not the result) by a small world-space vector before the nearest-
    point lookup - see mirror_bias() below for why."""
    projected, _normals = project_onto_surface_with_normals(points, bvh, mesh_obj, bias_world=bias_world)
    return projected


def project_onto_surface_with_normals(points, bvh, mesh_obj, bias_world=None):
    """Same as project_onto_surface, but also returns the surface normal at
    each projected point (world space) - needed by mesh_cut's ribbon-based
    bisection, which extrudes along these normals to build a cutting
    surface guaranteed to cross the target mesh at exactly this point."""
    mat = mesh_obj.matrix_world
    mat_inv = mat.inverted()
    normal_mat = mat_inv.to_3x3().transposed()
    bias_local = mat_inv.to_3x3() @ bias_world if bias_world is not None else Vector((0.0, 0.0, 0.0))

    projected = []
    normals = []
    for p in points:
        query_local = mat_inv @ p + bias_local
        hit_co, hit_normal, hit_idx, _hit_dist = bvh.find_nearest(query_local)
        if hit_co is not None:
            projected.append(mat @ hit_co)
            normals.append((normal_mat @ hit_normal).normalized())
        else:
            projected.append(p)
            normals.append(Vector((0.0, 0.0, 1.0)))
    return projected, normals


def mirror_plane_for(mesh_obj):
    """Returns (plane_point_world, plane_normal_world, has_real_mirror,
    merge_threshold) for the mesh's Mirror modifier symmetry plane, or the
    world Y-Z plane (X=0) with has_real_mirror=False and
    merge_threshold=0.0 if it has none. merge_threshold is the modifier's
    own weld distance (0.0 if the modifier exists but merge is off - see
    snap_to_mirror_plane for why a caller would want this)."""
    for mod in mesh_obj.modifiers:
        if mod.type == "MIRROR":
            axis_index = next((i for i in range(3) if mod.use_axis[i]), 0)
            pivot = mod.mirror_object or mesh_obj
            normal_local = Vector((1.0, 0.0, 0.0)) if axis_index == 0 else (
                Vector((0.0, 1.0, 0.0)) if axis_index == 1 else Vector((0.0, 0.0, 1.0))
            )
            normal_world = (pivot.matrix_world.to_3x3() @ normal_local).normalized()
            merge_threshold = mod.merge_threshold if mod.use_mirror_merge else 0.0
            return pivot.matrix_world.translation.copy(), normal_world, True, merge_threshold
    return Vector((0.0, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), False, 0.0


def snap_point_to_mesh_boundary(p, boundary_edges_world, snap_dist):
    """If p is within snap_dist of the mesh's own open-boundary edge loop
    (edges with only one linked face - a real edge of the surface, e.g. a
    garment's hem or neckline), returns the nearest point on that boundary
    instead; otherwise returns p unchanged.

    A seam curve that's meant to reach the mesh's edge to fully separate
    two pieces there rarely lands pixel-perfect on it - the artist's curve
    endpoint sits just inside the true edge, so the nearest-surface
    projection naturally lands just short of it too (confirmed on a real
    mesh: consistently a few times mesh_cut.MERGE_DIST, comfortably closer
    than one resampled step, but just far enough that get_or_insert's own
    vertex-reuse search doesn't reach the boundary vertex it should).
    Meant to be applied only to a seam curve's own true endpoints (a
    degree-1 vertex in its skeleton), not every resampled point - a
    interior point merely passing near some unrelated part of the mesh
    boundary should never be redirected there."""
    if snap_dist <= 0.0 or not boundary_edges_world:
        return p
    best = None
    for a, b in boundary_edges_world:
        d = b - a
        denom = d.length_squared
        if denom < 1e-12:
            continue
        t = max(0.0, min(1.0, (p - a).dot(d) / denom))
        closest = a + d * t
        dist = (closest - p).length
        if dist <= snap_dist and (best is None or dist < best[0]):
            best = (dist, closest)
    return best[1] if best is not None else p


def snap_to_mirror_plane(points_world, plane_point, plane_normal, snap_dist):
    """Projects any point within snap_dist of the mirror plane exactly onto
    it. A seam curve running along (or crossing) a Mirror-modifier weld
    line needs its resampled/projected points to land exactly on the
    weld's own shared vertex chain, not just near it: the "nearest
    surface" query used to project a point onto the mesh is inherently
    ambiguous exactly at the mirror plane (a point there is equidistant
    from both mirrored halves), and left uncorrected resolves to whichever
    side wins essentially at random from one resampled point to the next -
    zigzagging the walk between the two mirrored halves' interiors instead
    of tracing the shared vertices that actually connect them, leaving
    most of the real weld edges untouched and still fully connecting the
    two "separated" pieces (confirmed on a real mesh: 40% of the weld's
    own edges were never tagged as a seam boundary this way).

    snap_dist should be the Mirror modifier's own merge_threshold: any
    point that starts out farther than that from the plane cannot possibly
    belong to a face actually adjacent to the weld - the modifier would
    have merged it into the seam otherwise - so it's safe to leave alone;
    any point within it must genuinely belong there, since the modifier's
    own weld already collapsed everything closer than that onto the
    plane."""
    if snap_dist <= 0.0:
        return points_world
    snapped = []
    for p in points_world:
        offset = (p - plane_point).dot(plane_normal)
        snapped.append(p - plane_normal * offset if abs(offset) <= snap_dist else p)
    return snapped


def mirror_bias(mesh_obj, segment_length, mirror_normal_world, has_real_mirror):
    """A point sitting exactly on a mirrored mesh's seam is equidistant
    from both mirrored halves, so "nearest surface" has two equally valid
    answers and can flip between them from one resampled point to the next
    based on nothing but floating-point noise, which produces a visibly
    squiggly/zigzagging tube (and, now, a zigzagging *cut*, since
    resolution bisects along this same projected path) along what should
    be a straight seam line.

    Tested empirically whether real mesh bisection alone (instead of
    nearest-vertex snapping) would make this moot - it doesn't: the tie
    lives entirely in the "nearest point on surface" query that projects
    each resampled point before bisection even starts, so it reproduces
    identically regardless of what resolution does with the result
    afterward. Re-enabled: nudges the query position toward
    mirror_normal_world by a small fraction of segment_length, just enough
    to consistently break the tie one way without visibly deflecting the
    seam off the mirror line."""
    if not has_real_mirror:
        return None
    return mirror_normal_world.normalized() * (segment_length * 0.02)


def bias_for(mesh_obj):
    """Convenience: the mirror-seam anti-zigzag bias for mesh_obj, or None
    if it has no real Mirror modifier."""
    segment_length = segment_length_for(mesh_obj)
    _plane_point, plane_normal, has_real_mirror, _merge_threshold = mirror_plane_for(mesh_obj)
    return mirror_bias(mesh_obj, segment_length, plane_normal, has_real_mirror)


def build_tube_mesh(mesh_data, verts, edges, radius=TUBE_RADIUS, ring_segments=RING_SEGMENTS):
    """Writes a tube mesh hugging (verts, edges) (world space) into
    mesh_data - one ring per vertex, one quad strip per edge. Works for
    arbitrary topology (open paths, loops, branches) since it only needs
    per-vertex tangents and per-edge connectivity, not an ordered chain."""
    bm = bmesh.new()
    n = len(verts)
    if n < 2 or not edges:
        bm.to_mesh(mesh_data)
        bm.free()
        mesh_data.update()
        return

    neighbors = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    def tangent_at(i):
        nbrs = neighbors.get(i)
        if not nbrs:
            return Vector((0.0, 0.0, 1.0))
        if len(nbrs) == 1:
            d = verts[i] - verts[nbrs[0]]
        else:
            d = verts[nbrs[0]] - verts[nbrs[-1]]
        return d.normalized() if d.length > 1e-9 else Vector((0.0, 0.0, 1.0))

    rings = {}
    for i in range(n):
        if i not in neighbors:
            continue
        p = verts[i]
        t = tangent_at(i)
        # Not parallel-transported between rings, so a long, twisty curve
        # can show a little visual twist in the tube - acceptable for a
        # thin display tube; not worth the extra complexity for Phase 1.
        ref = Vector((0.0, 0.0, 1.0)) if abs(t.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        side1 = t.cross(ref).normalized()
        side2 = t.cross(side1).normalized()
        rings[i] = [
            bm.verts.new(
                p
                + side1 * math.cos(2 * math.pi * k / ring_segments) * radius
                + side2 * math.sin(2 * math.pi * k / ring_segments) * radius
            )
            for k in range(ring_segments)
        ]

    bm.verts.ensure_lookup_table()

    for a, b in edges:
        ring_a, ring_b = rings[a], rings[b]
        for k in range(ring_segments):
            k2 = (k + 1) % ring_segments
            bm.faces.new((ring_a[k], ring_a[k2], ring_b[k2], ring_b[k]))

    bm.normal_update()
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()


def rebuild_display(mesh_data, skeleton_verts, skeleton_edges, mesh_obj, bvh, bias_world=None):
    """Full pipeline: resample skeleton edges -> project onto surface -> tube mesh."""
    segment_length = segment_length_for(mesh_obj)
    resampled_verts, resampled_edges = resample_edges(skeleton_verts, skeleton_edges, segment_length)
    projected = project_onto_surface(resampled_verts, bvh, mesh_obj, bias_world=bias_world)
    build_tube_mesh(mesh_data, projected, resampled_edges)
