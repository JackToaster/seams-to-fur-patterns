"""Alignment-notch geometry for laser-cut patterns.

A notch is cut as a slit running from a point on the sew line straight out
to the seam-allowance cut line - so the laser cuts it in the same pass as
the piece outline, leaving a slit in the seam allowance that lines up with
its partner piece's matching slit, without ever cutting past the sew line
into the piece itself.

bpy-free, so it's directly testable with plain pytest (matching this
project's other geometry modules).
"""

import math

# Fallback slit length (mm) when a piece has no seam allowance at all -
# the slit then runs outward from the (sole) cut edge into the waste
# around the piece, which is harmless to cut and still leaves a visible
# alignment mark on the scrap.
NOTCH_FALLBACK_LENGTH_MM = 3.0

# How far (mm) to step off the boundary when testing which side of it is
# "outside" - small enough not to cross a neighboring edge near a corner.
_SIDE_PROBE_MM = 0.25


def point_in_polygon(pt, points):
    """Standard even-odd ray-casting test."""
    x, y = pt
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


def _ray_polygon_hit(origin, direction, points):
    """Distance along direction (unit) from origin to the nearest crossing
    of the closed polygon `points`, or None if the ray never crosses it."""
    ox, oy = origin
    dx, dy = direction
    best = None
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue
        # origin + t*direction == p1 + s*edge
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
        s = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        if t > 1e-6 and -1e-9 <= s <= 1.0 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def notch_slit(point, perp, boundary, cut_line=None, fallback_length=NOTCH_FALLBACK_LENGTH_MM):
    """(start, end) of a notch slit: start is `point` on the sew line
    (boundary), end is where a ray from it - along whichever of +/-perp
    points *out* of the piece - meets cut_line (the seam-allowance line).
    perp's own sign isn't trusted, since it comes from a seam run's point
    order, which isn't guaranteed consistent between pieces.

    With no cut_line (no seam allowance), or if the ray somehow misses it,
    the slit just runs fallback_length outward from the boundary instead.
    """
    px, py = point
    plen = math.hypot(*perp)
    if plen < 1e-12:
        return point, point
    ux, uy = perp[0] / plen, perp[1] / plen
    if point_in_polygon((px + ux * _SIDE_PROBE_MM, py + uy * _SIDE_PROBE_MM), boundary):
        ux, uy = -ux, -uy

    length = None
    if cut_line and len(cut_line) >= 3:
        length = _ray_polygon_hit((px, py), (ux, uy), cut_line)
    if length is None:
        length = fallback_length
    return (px, py), (px + ux * length, py + uy * length)
