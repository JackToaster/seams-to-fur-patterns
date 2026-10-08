"""2D polygon offsetting (grow/shrink) for pattern-piece boundaries.

Deliberately dependency-free (no Clipper2/pyclipr) so it stays fast and
trivially bundleable across platforms: this module has no bpy import either,
so it's directly testable with plain pytest. It implements the standard
"offset each edge along its normal, then re-intersect adjacent edges at each
vertex" (miter join) algorithm.

Known limitation vs. a full library like Clipper2: on aggressively concave
boundaries, a large shrink offset can produce self-intersecting output
(edges crossing) rather than being automatically cleaned up. offset_polygon
detects the case where the result's winding/area goes the wrong way (a
strong signal of a collapsed or inverted polygon) and raises OffsetError
instead of silently returning garbage. A Clipper2-backed implementation is
the natural upgrade path if this proves too limiting in practice - see the
project plan.
"""

import math


class OffsetError(Exception):
    pass


def _signed_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def polygon_area(points):
    """Unsigned area of a simple (x, y) polygon, any winding order -
    the public entry point for callers that just want a magnitude (e.g.
    filtering out slivers too small to be a real piece), not the signed
    value _signed_area uses internally to detect winding/inversion."""
    return abs(_signed_area(points))


def _normalize(dx, dy):
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return 0.0, 0.0
    return dx / length, dy / length


def _line_intersect(p1, d1, p2, d2):
    """Intersect two lines given as point+direction. Returns None if
    (near-)parallel."""
    x1, y1 = p1
    dx1, dy1 = d1
    x2, y2 = p2
    dx2, dy2 = d2

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-9:
        return None

    t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom
    return (x1 + dx1 * t, y1 + dy1 * t)


MAX_SELF_INTERSECTION_REPAIRS = 20

# Interior angle (degrees), measured directly on the *source* boundary at
# each vertex, below which a corner is beveled (two points, one hugging
# each edge's own offset) instead of mitered to a single sharp point. A
# real seam allowance can't be sewn to a fine point at a sharp corner - a
# patternmaker mitres (clips) it by hand, the same real-world operation a
# "bevel" join is standing in for here - and even a mathematically exact
# full miter at a sharp angle spikes out for a distance (1/sin(angle/2)
# times the offset) that's wildly impractical well before the angle gets
# anywhere near degenerate (confirmed live: real ~30-50 degree corners on
# an actual pattern piece spiking tens of mm past the piece). 60 degrees
# is chosen directly off what a real corner needs, not derived from any
# distance/ratio math - and applies uniformly to both a convex spike and a
# reflex notch (e.g. a dart tip) of the same sharpness, since the angle
# here is measured between the two rays from the vertex to its neighbors,
# which can't tell convex from reflex (nor does it need to: both produce
# an unusably long miter at the same sharpness and both should bevel).
MITER_MIN_ANGLE_DEG = 60.0


def _interior_angle_deg(prev_pt, vertex_pt, next_pt):
    v1x, v1y = prev_pt[0] - vertex_pt[0], prev_pt[1] - vertex_pt[1]
    v2x, v2y = next_pt[0] - vertex_pt[0], next_pt[1] - vertex_pt[1]
    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 < 1e-12 or len2 < 1e-12:
        return 180.0
    cos_a = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
    return math.degrees(math.acos(cos_a))


def _segment_intersect_interior(p1, p2, p3, p4):
    """(point) where segment p1-p2 crosses segment p3-p4 strictly in both
    segments' interiors (not just touching at/near an endpoint - shared
    endpoints between adjacent edges are normal, not self-intersections),
    or None if they don't cross that way."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return None
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    s = ((x3 - x1) * d1y - (y3 - y1) * d1x) / denom
    eps = 1e-9
    if eps < t < 1 - eps and eps < s < 1 - eps:
        return (x1 + d1x * t, y1 + d1y * t)
    return None


def _find_self_intersection(points):
    """First pair of non-adjacent edges (i, i+1) and (j, j+1), i < j, that
    cross - as (i, j, crossing_point) - or None if the polygon (closed,
    implicit last->first edge) is simple."""
    n = len(points)
    for i in range(n):
        a1, a2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue  # adjacent edges share a vertex - not a crossing
            b1, b2 = points[j], points[(j + 1) % n]
            point = _segment_intersect_interior(a1, a2, b1, b2)
            if point is not None:
                return i, j, point
    return None


def _remove_self_intersection_loop(points, i, j, point):
    """points has a self-intersection between edge (i, i+1) and edge
    (j, j+1) at `point` (i < j). Splits the polygon into the two closed
    loops that crossing implies and keeps the larger one (by area) - the
    smaller is the spurious "bowtie" loop a narrow-channel over-offset
    (e.g. a dart's own tip) produces, not part of the real shape."""
    loop_a = points[i + 1 : j + 1] + [point]
    loop_b = [point] + points[j + 1 :] + points[: i + 1]
    if len(loop_a) < 3 or len(loop_b) < 3:
        raise OffsetError("Self-intersection repair produced a degenerate loop")
    return loop_a if abs(_signed_area(loop_a)) >= abs(_signed_area(loop_b)) else loop_b


def offset_polygon(points, distance):
    """points: list of (x, y), assumed simple (non-self-intersecting)
    polygon in any winding order. distance: positive grows the polygon
    (seam allowance), negative shrinks it (fit/kerf adjustment).

    Returns a new list of (x, y), not necessarily the same length as
    points. A corner is mitered to a clean point only when its interior
    angle is at least MITER_MIN_ANGLE_DEG (60 degrees); anything sharper
    is beveled (two points hugging each edge's own offset) instead,
    matching how a seam allowance is actually mitred (clipped) by hand at
    a sharp corner rather than left to spike out. A genuine self-
    intersection this can still produce (typically a
    narrow channel - most commonly a dart's own tip - offset wider than the
    channel itself, which no per-corner join rule alone can prevent) is
    detected directly on the resulting polygon and repaired by discarding
    the small spurious loop the crossing implies (see
    _remove_self_intersection_loop) - which, like beveling, can leave the
    output a different length than the input. Note: eliminating the
    *spurious* near-degenerate "corners" mesh-noise can leave along a
    boundary is geometry.simplify's job, applied to the boundary before it
    ever reaches this function - a real design corner reaching this
    function is trusted to be real.
    """
    n = len(points)
    if n < 3:
        raise OffsetError("A polygon needs at least 3 points to offset")
    if distance == 0.0:
        return list(points)

    orig_area = _signed_area(points)
    if orig_area == 0.0:
        raise OffsetError("Polygon has zero area, cannot offset")
    winding = 1.0 if orig_area > 0 else -1.0

    # Outward normal for a CCW polygon edge (p1->p2) is (dy, -dx) normalized;
    # flip for CW so "outward" is consistent regardless of input winding.
    offset_edge_lines = []  # (point_on_offset_line, edge_direction, normal)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        ex, ey = _normalize(dx, dy)
        nx, ny = ey * winding, -ex * winding
        offset_edge_lines.append(((x1 + nx * distance, y1 + ny * distance), (ex, ey), (nx, ny)))

    new_points = []
    for i in range(n):
        vx, vy = points[i]
        prev_line = offset_edge_lines[(i - 1) % n]
        this_line = offset_edge_lines[i]

        angle = _interior_angle_deg(points[(i - 1) % n], (vx, vy), points[(i + 1) % n])
        if angle < MITER_MIN_ANGLE_DEG:
            # Bevel: hug each edge's own offset instead of mitering to a
            # single point - see MITER_MIN_ANGLE_DEG's docstring.
            pnx, pny = prev_line[2]
            tnx, tny = this_line[2]
            new_points.append((vx + pnx * distance, vy + pny * distance))
            new_points.append((vx + tnx * distance, vy + tny * distance))
            continue

        hit = _line_intersect(prev_line[0], prev_line[1], this_line[0], this_line[1])
        if hit is None:
            # Parallel edges meeting at this vertex (straight-through point);
            # just use the offset edge's start point.
            new_points.append(this_line[0])
        else:
            new_points.append(hit)

    for _ in range(MAX_SELF_INTERSECTION_REPAIRS):
        found = _find_self_intersection(new_points)
        if found is None:
            break
        i, j, point = found
        new_points = _remove_self_intersection_loop(new_points, i, j, point)
    else:
        raise OffsetError(
            f"Offset of {distance} produced a self-intersecting result that "
            f"couldn't be resolved after {MAX_SELF_INTERSECTION_REPAIRS} repair "
            f"attempts - the boundary may be too complex/narrow for this offset "
            f"distance"
        )

    new_area = _signed_area(new_points)
    if (new_area > 0) != (orig_area > 0):
        raise OffsetError(
            f"Offset of {distance} inverted the polygon (likely a shrink "
            f"larger than the smallest local feature size) - try a smaller "
            f"magnitude"
        )
    if distance < 0 and abs(new_area) > abs(orig_area):
        raise OffsetError(
            f"Offset of {distance} grew the polygon instead of shrinking it "
            f"in at least one region - the boundary is too concave for this "
            f"offset distance"
        )

    return new_points
