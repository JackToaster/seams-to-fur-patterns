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


def project_onto_surface(points, bvh, mesh_obj, bias_world=None):
    """Snaps each point (world space) to the nearest point on the target
    mesh's surface. bias_world, when given, nudges the *query* position
    (not the result) by a small world-space vector before the nearest-
    point lookup - see mirror_bias() below for why."""
    mat = mesh_obj.matrix_world
    mat_inv = mat.inverted()
    bias_local = mat_inv.to_3x3() @ bias_world if bias_world is not None else Vector((0.0, 0.0, 0.0))

    result = []
    for p in points:
        query_local = mat_inv @ p + bias_local
        hit_co, _hit_normal, hit_idx, _hit_dist = bvh.find_nearest(query_local)
        result.append(mat @ hit_co if hit_co is not None else p)
    return result


def mirror_bias(mesh_obj, segment_length, mirror_normal_world, has_real_mirror):
    """A point sitting exactly on a mirrored mesh's seam is equidistant
    from both mirrored halves, so "nearest surface" has two equally valid
    answers and flips between them from one resampled point to the next
    based on nothing but floating-point noise, producing a visibly
    squiggly/zigzagging tube along what should be a straight seam line.
    Nudging the query position by a small, consistent offset toward the
    mirror normal breaks the tie deterministically; the nudge is tiny
    relative to surface features so the result is still effectively on the
    seam. Only meaningful for a *real* Mirror modifier - there's no actual
    ambiguity to break for the world-Y-Z-plane fallback used elsewhere for
    click-snapping on non-mirrored meshes."""
    if not has_real_mirror:
        return None
    return mirror_normal_world * (segment_length * 0.25)


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
