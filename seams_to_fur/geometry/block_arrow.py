"""Fits a single-headed 2D block-arrow (a filled shaft + triangular head,
like Blender's own "Single Arrow" empty display, flattened to 2D) inside an
arbitrary simple polygon, aligned to a given direction and sized to the
largest actually-open run of space available along that direction - not
just centered on the polygon's own bounding box, which can clip through a
notch/dart or run off the edge of a non-rectangular piece.

Dependency-free (no numpy/shapely) and bpy-free, matching this project's
other geometry modules (see polygon_offset.py's docstring for why) - so
it's directly testable with plain pytest.

Algorithm: sweep a series of scanlines perpendicular to the target
direction; at each scanline, find the polygon's fill interval(s) there
(classic even-odd scanline rasterization) and, for a candidate arrow
*width*, the sub-range of positions along that scanline wide enough to
host it. Then sweep scanlines in order, intersecting each new scanline's
valid-position range with the running range carried from the previous
scanline - this is the same idea as "largest rectangle in a histogram",
just expressed as a running interval intersection instead of a stack, and
it finds the single longest run without ever needing an O(rows x cols)
grid of point-in-polygon tests. Tried at a few candidate widths, widest
first, so the arrow only gets narrower than ideal when the piece genuinely
doesn't have room for it.
"""

import math

# Fallback/default sizing, all in millimetres - a caller can override any
# of these via fit_block_arrow's keyword arguments.
IDEAL_WIDTH_MM = 8.0
MIN_WIDTH_MM = 2.5
MIN_LENGTH_MM = 12.0
SHAFT_WIDTH_RATIO = 0.42  # shaft width as a fraction of the head width
HEAD_LENGTH_RATIO = 0.3  # head length as a fraction of total arrow length
HEAD_LENGTH_MAX_MM = 14.0
N_SCANLINES = 220
N_WIDTH_STEPS = 5


def _to_local_frame(points, origin, direction):
    """World (x, y) points into a local frame where +Y is `direction` and
    +X is perpendicular to it (direction rotated -90deg, so the frame
    stays right-handed) - origin subtracted first."""
    dx, dy = direction
    out = []
    for px, py in points:
        lx = px - origin[0]
        ly = py - origin[1]
        # World vector (lx, ly) expressed in the (perp, direction) basis:
        # local_x = lx*perp_x + ly*perp_y, local_y = lx*dx + ly*dy
        perp_x, perp_y = dy, -dx
        out.append((lx * perp_x + ly * perp_y, lx * dx + ly * dy))
    return out


def _from_local_frame(points, origin, direction):
    dx, dy = direction
    perp_x, perp_y = dy, -dx
    out = []
    for lx, ly in points:
        wx = origin[0] + lx * perp_x + ly * dx
        wy = origin[1] + lx * perp_y + ly * dy
        out.append((wx, wy))
    return out


def _scanline_intervals(polygon, y):
    """x-intervals (sorted, non-overlapping pairs) where the horizontal
    line at height y crosses the polygon's interior - standard even-odd
    scanline rasterization. polygon: list of (x, y), implicitly closed."""
    xs = []
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 == y2:
            continue
        if (y1 <= y < y2) or (y2 <= y < y1):
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    return [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]


def _widest_range_for_width(intervals, width):
    """Among this scanline's fill intervals, the one that can host `width`
    with the most room to spare, reduced to the (lo, hi) range a
    candidate's *center* position could occupy while still fitting -  or
    None if nothing here is wide enough."""
    best = None
    for a, b in intervals:
        if b - a < width:
            continue
        lo, hi = a + width / 2.0, b - width / 2.0
        if best is None or (hi - lo) > (best[1] - best[0]):
            best = (lo, hi)
    return best


def _best_run_at_width(local_polygon, y_min, y_max, width):
    """Sweeps N_SCANLINES scanlines top to bottom, tracking the longest
    contiguous run of rows where some single center-column stays valid
    for a width-wide arrow the whole way through. Returns
    (length, center_x, center_y) for the best run found, or None."""
    if y_max <= y_min:
        return None
    step = (y_max - y_min) / N_SCANLINES

    best = None
    current_range = None
    run_start_row = None

    def flush(end_row):
        nonlocal best
        if run_start_row is None or current_range is None:
            return
        length = (end_row - run_start_row) * step
        if length < MIN_LENGTH_MM:
            return
        center_x = (current_range[0] + current_range[1]) / 2.0
        center_y = y_min + (run_start_row + end_row) / 2.0 * step
        if best is None or length > best[0]:
            best = (length, center_x, center_y)

    for row in range(N_SCANLINES + 1):
        y = y_min + row * step
        intervals = _scanline_intervals(local_polygon, y)
        row_range = _widest_range_for_width(intervals, width)

        if row_range is None:
            flush(row)
            current_range = None
            run_start_row = None
            continue

        if current_range is None:
            current_range = row_range
            run_start_row = row
        else:
            new_lo = max(current_range[0], row_range[0])
            new_hi = min(current_range[1], row_range[1])
            if new_lo <= new_hi:
                current_range = (new_lo, new_hi)
            else:
                flush(row)
                current_range = row_range
                run_start_row = row

    flush(N_SCANLINES)
    return best


def _block_arrow_polygon(width, length):
    """A single-headed block-arrow silhouette (7 points, closed), tip at
    +Y, tail at -Y, centered on the origin - local (across, along) frame,
    across=X, along=Y (arrow direction)."""
    shaft_w = width * SHAFT_WIDTH_RATIO
    head_len = min(length * HEAD_LENGTH_RATIO, HEAD_LENGTH_MAX_MM)
    head_len = min(head_len, length * 0.9)  # never let the head eat the whole shaft
    half_len = length / 2.0
    shoulder_y = half_len - head_len

    return [
        (-shaft_w / 2.0, -half_len),
        (shaft_w / 2.0, -half_len),
        (shaft_w / 2.0, shoulder_y),
        (width / 2.0, shoulder_y),
        (0.0, half_len),
        (-width / 2.0, shoulder_y),
        (-shaft_w / 2.0, shoulder_y),
    ]


def fit_block_arrow(
    boundary,
    direction,
    max_length,
    ideal_width=IDEAL_WIDTH_MM,
    min_width=MIN_WIDTH_MM,
):
    """boundary: list of (x, y) - a simple polygon (implicitly closed).
    direction: unit (dx, dy) - the arrow points this way.
    max_length: the longest arrow this will ever return, even if more
    open space is available (keeps it from feeling oversized on a huge
    plain piece - the block-arrow equivalent of the old line-based
    grainline's GRAINLINE_SPAN_FRACTION).

    Returns a list of (x, y) points (a closed polygon, in the same space
    as `boundary`) for the fitted arrow, or None if no placement fits
    even at min_width/MIN_LENGTH_MM.
    """
    dlen = math.hypot(*direction)
    if dlen < 1e-9:
        return None
    direction = (direction[0] / dlen, direction[1] / dlen)

    origin = boundary[0]
    local_boundary = _to_local_frame(boundary, origin, direction)
    ys = [p[1] for p in local_boundary]
    y_min, y_max = min(ys), max(ys)

    if ideal_width <= min_width:
        widths = [ideal_width]
    else:
        widths = [
            ideal_width - (ideal_width - min_width) * i / (N_WIDTH_STEPS - 1)
            for i in range(N_WIDTH_STEPS)
        ]

    best = None  # (length, width, center_x, center_y)
    for width in widths:
        run = _best_run_at_width(local_boundary, y_min, y_max, width)
        if run is not None:
            length, center_x, center_y = run
            best = (length, width, center_x, center_y)
            break  # widest viable width wins - see module docstring

    if best is None:
        return None

    length, width, center_x, center_y = best
    length = min(length, max_length)

    local_arrow = _block_arrow_polygon(width, length)
    local_arrow = [(x + center_x, y + center_y) for x, y in local_arrow]
    return _from_local_frame(local_arrow, origin, direction)
