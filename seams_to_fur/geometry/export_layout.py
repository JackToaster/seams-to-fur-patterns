"""Regenerates a print/cut-friendly grid layout for fabrication-output
export, instead of reusing wherever flatten_all's own viewport-oriented
placement (geometry.layout.shelf_layout, Blender-unit meters - a
different concern: scene-space placement for on-screen comparison/
editing against the curved model) or a user's manual drag happened to
leave each piece.

bpy-free (works on the already-gathered plain mm-space piece dicts
operators/export_common.py's gather_piece_export_data produces, after
they've been extracted from Blender objects) so it's directly testable
with plain pytest, matching this project's other geometry modules.
"""

import math

# A reasonable default: about the width of a common wide-format
# plotter/fabric roll (~36in). Not user-configurable today - nothing in
# this project's scope has asked for that yet.
SHELF_WIDTH_MM = 900.0
MARGIN_MM = 15.0
# Extra vertical gap between one fur-type group's rows and the next
# group's first row - clearly bigger than MARGIN_MM (the gap between
# individual pieces) so a group boundary reads as a real break at a
# glance, not just another row wrap.
GROUP_MARGIN_MM = 40.0


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _piece_bbox(piece):
    """The bbox of everything that actually gets cut - boundary_mm (sew
    line) unioned with cut_line_mm (seam allowance) when present, not just
    the sew line alone. A piece with a seam allowance extends past its own
    sew line on every side, so spacing shelf placement off the sew line's
    bbox alone can leave two adjacent pieces' *cut* lines overlapping even
    though margin_mm looks respected between their sew lines (confirmed
    live: two pieces with a 5mm allowance and pointed, tapered tips ended
    up with cut lines crossing)."""
    points = list(piece["boundary_mm"])
    if piece.get("cut_line_mm"):
        points += piece["cut_line_mm"]
    return _bbox(points)


def _translate_piece_geometry(piece, dx, dy):
    """Shifts every mm-space geometry field in a gathered piece dict by
    (dx, dy) - point-only translation (a flattened piece is never rotated
    by this layout, only repositioned, so this is complete for every
    field that carries real coordinates). Mutates piece in place."""
    piece["boundary_mm"] = [(x + dx, y + dy) for x, y in piece["boundary_mm"]]

    if piece.get("cut_line_mm"):
        piece["cut_line_mm"] = [(x + dx, y + dy) for x, y in piece["cut_line_mm"]]

    if piece.get("grainline_mm"):
        (sx, sy), (ex, ey) = piece["grainline_mm"]
        piece["grainline_mm"] = ((sx + dx, sy + dy), (ex + dx, ey + dy))

    if piece.get("grainline_arrow_mm"):
        piece["grainline_arrow_mm"] = [(x + dx, y + dy) for x, y in piece["grainline_arrow_mm"]]

    for notch in piece.get("notches_mm", []):
        px, py = notch["point_mm"]
        notch["point_mm"] = (px + dx, py + dy)

    for label in piece.get("seam_labels", []):
        if label.get("mode") == "straight":
            px, py = label["pos_mm"]
            label["pos_mm"] = (px + dx, py + dy)
        else:
            label["path_mm"] = [(x + dx, y + dy) for x, y in label["path_mm"]]

    lx, ly = piece["label_pos_mm"]
    piece["label_pos_mm"] = (lx + dx, ly + dy)


def rotate_piece_geometry(piece, angle_rad, pivot):
    """Rotates every mm-space geometry field in a gathered piece dict by
    angle_rad (standard math convention: positive = counterclockwise)
    about pivot=(px, py) - the same field coverage as
    _translate_piece_geometry, except direction-only fields (a notch's
    perp, a straight seam label's dir_mm) rotate in place without the
    pivot, since a direction has no position to rotate about. Mutates
    piece in place.

    Used to physically rotate a piece so its own grain direction points
    straight down before export_layout ever repositions it - real fabric
    has a single straight-of-grain axis, so every piece's grain arrow
    needs to line up the same way once pieces are laid out on a cut roll,
    not just have a consistent up/down polarity within each piece's own
    (otherwise arbitrary) flattened orientation."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    px, py = pivot

    def rot_point(pt):
        x, y = pt[0] - px, pt[1] - py
        return (px + x * cos_a - y * sin_a, py + x * sin_a + y * cos_a)

    def rot_dir(d):
        x, y = d
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    piece["boundary_mm"] = [rot_point(p) for p in piece["boundary_mm"]]

    if piece.get("cut_line_mm"):
        piece["cut_line_mm"] = [rot_point(p) for p in piece["cut_line_mm"]]

    if piece.get("grainline_mm"):
        s, e = piece["grainline_mm"]
        piece["grainline_mm"] = (rot_point(s), rot_point(e))

    if piece.get("grainline_arrow_mm"):
        piece["grainline_arrow_mm"] = [rot_point(p) for p in piece["grainline_arrow_mm"]]

    for notch in piece.get("notches_mm", []):
        notch["point_mm"] = rot_point(notch["point_mm"])
        notch["perp"] = rot_dir(notch["perp"])

    for label in piece.get("seam_labels", []):
        if label.get("mode") == "straight":
            label["pos_mm"] = rot_point(label["pos_mm"])
            label["dir_mm"] = rot_dir(label["dir_mm"])
        else:
            label["path_mm"] = [rot_point(p) for p in label["path_mm"]]

    piece["label_pos_mm"] = rot_point(piece["label_pos_mm"])


def regenerate_export_layout(
    pieces, shelf_width_mm=SHELF_WIDTH_MM, margin_mm=MARGIN_MM, group_margin_mm=GROUP_MARGIN_MM
):
    """Groups pieces by fur_type (color + fur length - see
    operators/export_common.py; not just swatch_name, since two pieces
    the same color but a different fur length are still a different
    "type" of fabric to cut from) so all of one type end up contiguous,
    each type always starting on its own fresh row - never sharing a row
    with the previous type, even if there'd be room - with group_margin_mm
    of empty space above it, clearly more than the margin_mm between
    individual pieces, so a group reads as a deliberate break at a glance.
    This is deliberately *not* real nesting/pack optimization (the project
    plan defers that) - actual piece placement for cutting from whatever
    scrap fabric is on hand still happens by hand; this only makes it easy
    to spot and pull out "everything of one type" as a contiguous block in
    the exported file.

    Within each group, pieces still lay out via the same shelf-style grid
    in mm-space as before, spaced by margin_mm between each piece's
    outermost cut edge (its seam allowance line, when it has one - see
    _piece_bbox) so cut lines never end up closer than margin_mm apart,
    not just sew lines. Mutates every piece dict's geometry fields in
    place (see _translate_piece_geometry) and returns the list re-ordered
    to match the new grouped layout - the caller should use the returned
    list (and can discard the input order)."""
    ordered = sorted(pieces, key=lambda p: (p.get("fur_type") or "", p["name"]))

    cursor_x = 0.0
    cursor_y = 0.0
    shelf_height = 0.0
    current_group = None
    first_group = True

    for piece in ordered:
        group = piece.get("fur_type")
        if group != current_group:
            if not first_group:
                cursor_y += shelf_height + group_margin_mm
            cursor_x = 0.0
            shelf_height = 0.0
            current_group = group
            first_group = False

        min_x, min_y, max_x, max_y = _piece_bbox(piece)
        width = max_x - min_x
        height = max_y - min_y

        if cursor_x > 0.0 and cursor_x + width > shelf_width_mm:
            cursor_x = 0.0
            cursor_y += shelf_height + margin_mm
            shelf_height = 0.0

        dx = cursor_x - min_x
        dy = cursor_y - min_y
        _translate_piece_geometry(piece, dx, dy)

        cursor_x += width + margin_mm
        shelf_height = max(shelf_height, height)

    return ordered
