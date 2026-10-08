"""Non-overlapping placement of freshly flattened pattern pieces.

This is NOT a nesting/material-usage optimizer (that's Phase 2+) - it's a
simple deterministic shelf layout so newly baked pieces are visible and
distinct rather than stacked on top of each other, matching Blender-unit
scale (assumes the scene is in meters; pieces are already sized in real
units by the flatten step).
"""

MARGIN = 0.02  # 2cm gap between pieces, in Blender scene units (meters)


def bbox_2d(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def shelf_layout(piece_boundaries, shelf_width=2.0):
    """piece_boundaries: ordered list of (piece_uuid, [(x, y), ...]) in each
    piece's own local (already-flattened, origin-relative) coordinates.

    Returns dict piece_uuid -> (offset_x, offset_y) to translate that piece
    to a non-overlapping position.
    """
    placements = {}
    cursor_x = 0.0
    cursor_y = 0.0
    shelf_height = 0.0

    for piece_uuid, boundary in piece_boundaries:
        min_x, min_y, max_x, max_y = bbox_2d(boundary)
        width = max_x - min_x
        height = max_y - min_y

        if cursor_x > 0.0 and cursor_x + width > shelf_width:
            cursor_x = 0.0
            cursor_y += shelf_height + MARGIN
            shelf_height = 0.0

        offset_x = cursor_x - min_x
        offset_y = cursor_y - min_y
        placements[piece_uuid] = (offset_x, offset_y)

        cursor_x += width + MARGIN
        shelf_height = max(shelf_height, height)

    return placements
