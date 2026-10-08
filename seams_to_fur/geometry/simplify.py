"""Polyline/polygon simplification (Ramer-Douglas-Peucker) for cleaning up
fabrication-output export geometry before it's used for anything else -
drawn as the sew line, offset for a cut line, or searched for grain-arrow
placement.

A flattened pattern piece's boundary is walked directly off the flattened
mesh's own topology (operators/export_common.py), which can carry many
near-collinear vertices from whatever subdivision/triangulation density
the source mesh happened to have there - individually tiny (sub-mm)
deviations that are invisible on the boundary itself, but get amplified
into visible zigzag/doubling-back artifacts once offset outward for a
seam allowance (confirmed on a real export: prevalent on one piece
specifically, at a boundary run shared with several neighboring pieces,
each contributing their own slightly different vertex density along
what's meant to be the same, single logical edge). Simplifying the
boundary first - dropping any vertex that's within a small tolerance of
the straight line its neighbors would already draw - removes the noise
without visibly changing the piece's real shape.

bpy-free, matching this project's other geometry modules.
"""

import math

# A tolerance small enough to be invisible on the boundary itself (well
# under a millimetre) but large enough to absorb real mesh-topology noise
# - not a design-intent parameter a user should need to tune.
DEFAULT_EPSILON_MM = 0.75


def _perpendicular_distance(point, a, b):
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-12:
        return math.hypot(px - ax, py - ay)
    # |cross product| / |base| = perpendicular distance from point to the
    # infinite line through a-b.
    cross = abs(dx * (ay - py) - dy * (ax - px))
    return cross / seg_len


def _rdp(points, epsilon):
    if len(points) < 3:
        return list(points)

    start, end = points[0], points[-1]
    max_dist = -1.0
    split_index = -1
    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            split_index = i

    if max_dist > epsilon:
        left = _rdp(points[: split_index + 1], epsilon)
        right = _rdp(points[split_index:], epsilon)
        return left[:-1] + right
    return [start, end]


def simplify_polyline(points, epsilon=DEFAULT_EPSILON_MM):
    """points: an OPEN chain (not implicitly closed) of (x, y). Returns a
    simplified chain with the same endpoints, dropping any interior point
    whose deviation from the straight line it would otherwise be on is
    under epsilon."""
    return _rdp(list(points), epsilon)


def simplify_polygon(points, epsilon=DEFAULT_EPSILON_MM):
    """points: a CLOSED polygon (implicitly closed - no repeated first/last
    point) of (x, y). Returns a simplified polygon of at least 3 points,
    same convention (no repeated closing point).

    A closed loop has no natural start/end for RDP (which needs two fixed
    endpoints to measure deviation from), so this treats points[0] as both
    - running RDP on [points[0], points[1], ..., points[-1], points[0]]
    and dropping the duplicated closing point from the result. This finds
    a real split point on the first pass anyway (whichever point is
    farthest from points[0], since the "base line" from points[0] back to
    itself is degenerate and _perpendicular_distance falls back to plain
    point distance in that case), so it simplifies correctly rather than
    just leaving points[0] as an awkward permanent corner.
    """
    if len(points) < 4:
        # Already at (or below) the minimum a polygon needs - nothing to
        # simplify away without going invalid.
        return list(points)
    closed_chain = list(points) + [points[0]]
    simplified = _rdp(closed_chain, epsilon)
    result = simplified[:-1]
    return result if len(result) >= 3 else list(points)
