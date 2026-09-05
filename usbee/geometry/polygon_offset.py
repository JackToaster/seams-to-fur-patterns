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


def offset_polygon(points, distance):
    """points: list of (x, y), assumed simple (non-self-intersecting)
    polygon in any winding order. distance: positive grows the polygon
    (seam allowance), negative shrinks it (fit/kerf adjustment).

    Returns a new list of (x, y) of the same length as points.
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
    offset_edge_lines = []  # (point_on_offset_line, edge_direction)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        ex, ey = _normalize(dx, dy)
        nx, ny = ey * winding, -ex * winding
        offset_edge_lines.append(((x1 + nx * distance, y1 + ny * distance), (ex, ey)))

    new_points = []
    for i in range(n):
        prev_line = offset_edge_lines[(i - 1) % n]
        this_line = offset_edge_lines[i]
        hit = _line_intersect(prev_line[0], prev_line[1], this_line[0], this_line[1])
        if hit is None:
            # Parallel edges meeting at this vertex (straight-through point);
            # just use the offset edge's start point.
            hit = this_line[0]
        new_points.append(hit)

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
