"""Pure-Python seam curve display: resamples the raw authored point list,
projects each resampled point onto the target mesh's surface, and builds a
simple tube mesh around the result - directly into the seam curve object's
own Mesh datablock.

Deliberately no modifiers, no Curve object type, no Geometry Nodes: this
addon fully owns seam-curve geometry end to end, because both of the
"let Blender do it" alternatives turned out to be unreliable in practice -
a classic Mirror modifier doesn't reliably preserve curve-domain data when
applied to a Curve object (it behaves like it rasterizes toward mesh
output), and a curve whose display comes from a modifier/GN stack can't be
edited via native curve Edit Mode without the two fighting each other. With
plain Python + bmesh, "hugging the surface" and "mirroring the seam" are
both just vector math we control completely, and the result is always a
plain mesh with no stack to reason about.
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


def resample_polyline(points, closed, segment_length):
    """Arc-length resample of a polyline (list of Vector, world space) to
    roughly even spacing of segment_length. Denser sampling makes the
    surface-projection step below hug curvature instead of a straight
    chord cutting through the interior between sparse authored points."""
    pts = list(points)
    if len(pts) < 2:
        return pts
    if closed:
        pts = pts + [pts[0]]

    total_length = sum((pts[i + 1] - pts[i]).length for i in range(len(pts) - 1))
    if total_length < 1e-9:
        return list(points)

    n_segments = max(1, round(total_length / segment_length))
    step = total_length / n_segments

    result = [pts[0]]
    accumulated = 0.0
    next_target = step
    max_points = n_segments if closed else n_segments + 1
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_vec = b - a
        seg_len = seg_vec.length
        seg_start = accumulated
        seg_end = accumulated + seg_len
        while next_target <= seg_end + 1e-9 and len(result) < max_points:
            t = (next_target - seg_start) / seg_len if seg_len > 1e-12 else 0.0
            result.append(a + seg_vec * t)
            next_target += step
        accumulated = seg_end

    if not closed and (result[-1] - pts[-1]).length > 1e-6:
        result.append(pts[-1])
    return result


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


def build_tube_mesh(mesh_data, points, closed, radius=TUBE_RADIUS, ring_segments=RING_SEGMENTS):
    """Writes a tube mesh hugging `points` (world space) into mesh_data."""
    bm = bmesh.new()
    n = len(points)
    if n < 2:
        bm.to_mesh(mesh_data)
        bm.free()
        mesh_data.update()
        return

    def tangent_at(i):
        if closed:
            prev_p, next_p = points[(i - 1) % n], points[(i + 1) % n]
        else:
            prev_p, next_p = points[max(i - 1, 0)], points[min(i + 1, n - 1)]
        d = next_p - prev_p
        return d.normalized() if d.length > 1e-9 else Vector((0.0, 0.0, 1.0))

    rings = []
    for i in range(n):
        p = points[i]
        t = tangent_at(i)
        # Not parallel-transported between rings, so a long, twisty curve
        # can show a little visual twist in the tube - acceptable for a
        # thin display tube; not worth the extra complexity for Phase 1.
        ref = Vector((0.0, 0.0, 1.0)) if abs(t.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        side1 = t.cross(ref).normalized()
        side2 = t.cross(side1).normalized()
        ring = [
            bm.verts.new(
                p
                + side1 * math.cos(2 * math.pi * k / ring_segments) * radius
                + side2 * math.sin(2 * math.pi * k / ring_segments) * radius
            )
            for k in range(ring_segments)
        ]
        rings.append(ring)

    bm.verts.ensure_lookup_table()

    ring_pairs = n if closed else n - 1
    for i in range(ring_pairs):
        ring_a, ring_b = rings[i], rings[(i + 1) % n]
        for k in range(ring_segments):
            k2 = (k + 1) % ring_segments
            bm.faces.new((ring_a[k], ring_a[k2], ring_b[k2], ring_b[k]))

    if not closed:
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])

    bm.normal_update()
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()


def rebuild_display(mesh_data, points_world, closed, mesh_obj, bvh, bias_world=None):
    """Full pipeline: resample -> project onto surface -> tube mesh."""
    segment_length = segment_length_for(mesh_obj)
    resampled = resample_polyline(points_world, closed, segment_length)
    projected = project_onto_surface(resampled, bvh, mesh_obj, bias_world=bias_world)
    build_tube_mesh(mesh_data, projected, closed)


def mirror_points(points, plane_point, plane_normal):
    """Reflects each point across the given plane."""
    result = []
    for p in points:
        d = (p - plane_point).dot(plane_normal)
        result.append(p - plane_normal * (2.0 * d))
    return result
