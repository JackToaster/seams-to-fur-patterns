"""Shared per-piece data gathering for fabrication-output exporters
(SVG, DXF, ...) - one place that knows how to turn a mesh's baked pieces
into plain, format-agnostic mm-space data, so no exporter duplicates this
logic or drifts out of sync with another.
"""

import math

import bpy
from mathutils import Vector

from ..fur_colors import nearest_swatch_name
from ..geometry import block_arrow, boundary, export_layout, polygon_offset, simplify, text_fit
from . import appearance

# Pieces smaller than this (sew-line area) are skipped entirely - a
# defensive catch for a genuinely-degenerate sliver artifact, not a
# general "small piece" cutoff: a real small-but-legitimate piece (~980mm^2
# was the smallest confirmed real piece on a live file) must never be
# caught by this. The original tiny-sliver report on that same file turned
# out, on inspection, to be stale orphaned objects (leftover from an
# earlier rebuild) that were never actually part of the export in the
# first place - not something this filter can or should be catching -
# so this is kept deliberately low, well under any real piece's size, as
# a safety net for the rare case where one genuinely does slip through.
MIN_PIECE_AREA_MM2 = 100.0

# Notch placement thresholds (mm), in real-world scale regardless of the
# scene's Blender-unit scale (mm_scale already converts everything here) -
# see the project plan: skip a shared seam run too short to need/fit a
# mark, otherwise inset from each end rather than marking the corner
# itself, and add a midpoint mark for a long run.
NOTCH_MIN_RUN_LENGTH_MM = 30.0
NOTCH_INSET_MM = 10.0
NOTCH_MIDPOINT_MIN_LENGTH_MM = 150.0

# Grainline arrow spans this fraction of the piece's actual extent along
# the grain direction, centered - a real commercial-pattern grainline is a
# long double-headed arrow close to (but not touching) the cut line at
# either end, not a short single-headed indicator like the on-screen
# preview arrow.
GRAINLINE_SPAN_FRACTION = 0.85
# ...but never longer than this outright, regardless of how much open
# space a large piece has - a real grainline arrow is a direction
# indicator, not a ruler.
ARROW_MAX_LENGTH_MM = 50.8  # 2 inches
# The arrow's own search space is the boundary shrunk inward by this much
# first - keeps the arrow from ever touching/crowding the cut or sew line,
# even when the piece has plenty of room.
ARROW_EDGE_INSET_MM = 5.0

# Text sizing (mm) - shrinks to fit the piece's own bounding width instead
# of overflowing a small piece at a fixed size, down to a floor below which
# it's better to just accept a slight overflow than render illegibly small.
LABEL_MAX_FONT_MM = 4.0
LABEL_MIN_FONT_MM = 1.5
SEAM_LABEL_FONT_MM = 2.5
# Fraction of the piece's own bounding width text is allowed to occupy
# before shrinking - never the full width, so a label doesn't visually
# touch the piece's own cut/sew lines on either side.
LABEL_WIDTH_FRACTION = 0.85

# How far a seam-partner label's path is inset from the actual seam edge,
# toward the piece's own centroid - far enough to clearly read as "just
# inside the piece", not sitting on top of the sew/cut line itself.
SEAM_LABEL_INSET_MM = 5.0
# A curved textPath looks distorted/illegible if the edge it follows bends
# sharper than this anywhere along the run - falls back to a single
# straight, inset label at the run's midpoint instead of following the
# curve through a sharp kink.
SEAM_LABEL_MAX_TURN_DEG = 30.0


def blender_units_to_mm(context):
    unit_settings = context.scene.unit_settings
    if unit_settings.system not in {"METRIC", "NONE"}:
        # Imperial scenes would need a different constant; rather than
        # silently exporting a wrong-scale pattern (the exact failure mode
        # that ruins a physical cut), only metric/none is supported.
        raise RuntimeError(
            f"Scene unit system is '{unit_settings.system}', not METRIC - "
            f"switch it in Scene Properties > Units before exporting, so "
            f"the export comes out at the correct real-world size"
        )
    return unit_settings.scale_length * 1000.0


def _local_to_mm(mat, mm_scale, point_local):
    """A local-space point (any object's own vertex/attribute coordinates)
    to a plain (x, y) mm-space tuple - world transform applied, Z dropped
    (flattened pieces live in their own local XY plane at real-world scale,
    only translated - never rotated out of that plane - by the shelf
    layout; see geometry/layout.py)."""
    co = mat @ point_local
    return (co.x * mm_scale, co.y * mm_scale)


def _local_dir_to_mm_2d(mat, direction_local):
    """A local-space *direction* (not a point - no translation) to a plain
    2D (x, y) tuple in the same world-XY space _local_to_mm points live in."""
    d = mat.to_3x3() @ direction_local
    return Vector((d.x, d.y))


def _piece_display_name(piece):
    """piece.name without a redundant leading "Piece " - every piece is
    already labeled "Piece ##" by construction (see
    geometry.islands.sync_piece_settings), so the export title only needs
    the number. Falls back to the full name unchanged for a piece a user
    has renamed to something that doesn't start with "Piece " - still a
    sensible label, just not shortened."""
    prefix = "Piece "
    return piece.name[len(prefix) :] if piece.name.startswith(prefix) else piece.name


def _fur_inches_label(piece, mm_scale):
    inches = piece.fur_length * mm_scale / 25.4
    return text_fit.format_inches(inches)


def _point_along_polyline(points, target_length):
    """points: ordered list of 2D mm-space (x, y) tuples. Walks the
    polyline's cumulative length and returns (point, tangent_2d) at
    target_length along it (tangent from whichever segment contains that
    point, not normalized to unit length by direction alone - callers
    normalize). Clamps to the polyline's actual ends."""
    if target_length <= 0.0:
        p0, p1 = points[0], points[1]
        return points[0], (p1[0] - p0[0], p1[1] - p0[1])
    acc = 0.0
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        seg_len = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if acc + seg_len >= target_length or i == len(points) - 1:
            t = 0.0 if seg_len < 1e-9 else (target_length - acc) / seg_len
            t = max(0.0, min(1.0, t))
            point = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            return point, (x1 - x0, y1 - y0)
        acc += seg_len
    return points[-1], (points[-1][0] - points[-2][0], points[-1][1] - points[-2][1])


def _run_to_mm_polyline(run_points_local, flat_obj, proxy, mat, mm_scale):
    """A seam_partners run (3D points in the source mesh's own local space)
    mapped through this piece's own flat proxy into mm-space 2D points on
    its flattened output - the same 3D<->2D correspondence the grain-
    direction arrows already use (appearance._map_point_to_flat)."""
    points_mm = []
    for co in run_points_local:
        mapped = appearance._map_point_to_flat(*proxy, co)
        if mapped is None:
            continue
        points_mm.append(_local_to_mm(mat, mm_scale, mapped))
    return points_mm


def _closest_point_on_boundary(pt, boundary_mm):
    """(point, tangent) - the closest point on boundary_mm's own edges to
    pt, and that edge's tangent direction there. A seam_partners run can
    sample the seam more coarsely than the piece's actual boundary loop
    does (confirmed live: a mesh with faces much larger than the seam
    curve's own resample spacing, flagged by a real warning at flatten
    time) - _point_along_polyline's straight-line interpolation between
    two such sparse run points can then cut across a corner the boundary
    itself has extra vertices for, landing several mm off the edge instead
    of on it. Snapping onto the real boundary afterward is robust
    regardless of how coarse the run's own sampling is."""
    x, y = pt
    best_d2 = None
    best_point = boundary_mm[0]
    best_tangent = (1.0, 0.0)
    n = len(boundary_mm)
    for i in range(n):
        x1, y1 = boundary_mm[i]
        x2, y2 = boundary_mm[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg_len2))
        px, py = x1 + t * dx, y1 + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_point = (px, py)
            best_tangent = (dx, dy)
    return best_point, best_tangent


def _notches_for_piece(piece, flat_obj, proxy, mat, mm_scale, boundary_mm):
    notches = []
    for partner in piece.seam_partners:
        run_points_local = [Vector(p.co) for p in partner.points]
        if len(run_points_local) < 2:
            continue
        points_mm = _run_to_mm_polyline(run_points_local, flat_obj, proxy, mat, mm_scale)
        if len(points_mm) < 2:
            continue

        total_length = sum(
            ((points_mm[i][0] - points_mm[i - 1][0]) ** 2 + (points_mm[i][1] - points_mm[i - 1][1]) ** 2) ** 0.5
            for i in range(1, len(points_mm))
        )
        if total_length < NOTCH_MIN_RUN_LENGTH_MM:
            continue

        targets = [NOTCH_INSET_MM, total_length - NOTCH_INSET_MM]
        if total_length >= NOTCH_MIDPOINT_MIN_LENGTH_MM:
            targets.append(total_length / 2.0)

        for target in targets:
            point, _tangent = _point_along_polyline(points_mm, target)
            point, tangent = _closest_point_on_boundary(point, boundary_mm)
            tlen = (tangent[0] ** 2 + tangent[1] ** 2) ** 0.5
            if tlen < 1e-9:
                continue
            perp = (-tangent[1] / tlen, tangent[0] / tlen)
            notches.append({"point_mm": point, "perp": perp})
    return notches


def _seam_labels_for_piece(piece, flat_obj, proxy, mat, mm_scale, piece_by_uuid, boundary_mm):
    """One label per seam_partners run long enough to bother marking (same
    threshold as notches - a run too short for a notch is too short to fit
    readable text along either): a polyline inset SEAM_LABEL_INSET_MM
    toward the piece's own centroid from the actual seam edge (so the text
    sits just inside the piece, not on top of the sew/cut line itself),
    labeled with the *other* piece's name - "this edge sews to that piece"
    - for a caller to render as text-on-path, following the edge's curve
    rather than a single straight run of text across it."""
    labels = []
    if len(boundary_mm) < 3:
        return labels
    cx = sum(p[0] for p in boundary_mm) / len(boundary_mm)
    cy = sum(p[1] for p in boundary_mm) / len(boundary_mm)

    for partner in piece.seam_partners:
        other = piece_by_uuid.get(partner.partner_piece_uuid)
        if other is None:
            continue
        run_points_local = [Vector(p.co) for p in partner.points]
        if len(run_points_local) < 2:
            continue
        points_mm = _run_to_mm_polyline(run_points_local, flat_obj, proxy, mat, mm_scale)
        n = len(points_mm)
        if n < 2:
            continue

        total_length = sum(
            ((points_mm[i][0] - points_mm[i - 1][0]) ** 2 + (points_mm[i][1] - points_mm[i - 1][1]) ** 2) ** 0.5
            for i in range(1, n)
        )
        if total_length < NOTCH_MIN_RUN_LENGTH_MM:
            continue

        inset_points = []
        for i, (x, y) in enumerate(points_mm):
            prev_pt = points_mm[i - 1] if i > 0 else points_mm[i]
            next_pt = points_mm[i + 1] if i < n - 1 else points_mm[i]
            tx, ty = next_pt[0] - prev_pt[0], next_pt[1] - prev_pt[1]
            tlen = (tx * tx + ty * ty) ** 0.5
            if tlen < 1e-9:
                inset_points.append((x, y))
                continue
            # A 2D perpendicular can point either way depending on the
            # run's own point order (not something to rely on being
            # consistent between two pieces sharing the same seam, or
            # even along one piece's own run) - explicitly picking the
            # side that points toward this piece's centroid is what
            # actually guarantees "inward", not just "a" perpendicular.
            perp = (-ty / tlen, tx / tlen)
            to_center = (cx - x, cy - y)
            if perp[0] * to_center[0] + perp[1] * to_center[1] < 0.0:
                perp = (-perp[0], -perp[1])
            inset_points.append((x + perp[0] * SEAM_LABEL_INSET_MM, y + perp[1] * SEAM_LABEL_INSET_MM))

        text = _piece_display_name(other)

        # A curved textPath looks distorted/illegible if the edge it
        # follows bends sharply anywhere along the run - fall back to a
        # single straight, inset label at the run's midpoint instead of
        # forcing the text through a visible kink.
        has_sharp_turn = False
        for i in range(1, n - 1):
            ax, ay = points_mm[i - 1]
            bx, by = points_mm[i]
            dx1, dy1 = bx - ax, by - ay
            cx2, cy2 = points_mm[i + 1]
            dx2, dy2 = cx2 - bx, cy2 - by
            len1 = math.hypot(dx1, dy1)
            len2 = math.hypot(dx2, dy2)
            if len1 < 1e-9 or len2 < 1e-9:
                continue
            cos_turn = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (len1 * len2)))
            if math.degrees(math.acos(cos_turn)) > SEAM_LABEL_MAX_TURN_DEG:
                has_sharp_turn = True
                break

        if has_sharp_turn:
            mid_point, tangent = _point_along_polyline(inset_points, total_length / 2.0)
            tlen = math.hypot(*tangent)
            if tlen < 1e-9:
                continue
            labels.append(
                {
                    "text": text,
                    "mode": "straight",
                    "pos_mm": mid_point,
                    "dir_mm": (tangent[0] / tlen, tangent[1] / tlen),
                }
            )
        else:
            labels.append({"text": text, "mode": "curved", "path_mm": inset_points})
    return labels


def _grainline_for_piece(piece, flat_obj, proxy, mat, mm_scale, boundary_mm):
    if not piece.has_grain_direction or proxy is None or len(boundary_mm) < 3:
        return None
    mapped = appearance._map_direction_to_flat(*proxy, Vector(piece.grain_anchor), Vector(piece.grain_direction))
    if mapped is None:
        return None
    _anchor_2d_local, direction_2d_local = mapped
    direction_mm = _local_dir_to_mm_2d(mat, direction_2d_local)
    if direction_mm.length < 1e-9:
        return None
    direction_mm = direction_mm.normalized()

    # No sign/polarity normalization here, deliberately - unlike woven-
    # fabric grain (just an axis), fur has a real nap direction, and
    # piece.grain_direction already carries the correct one (it's the same
    # vector the on-screen fur combing aligns to). An earlier version
    # flipped this whenever direction_mm.y > 0 "for consistent display",
    # meant only to pick a stable -Y-leaning polarity before the rotation
    # below - but that flip depended on the piece's own incidental BFF-
    # flattening orientation, not on anything about the true nap, so it
    # silently reversed the real fur direction on whichever pieces
    # happened to flatten with a +Y-leaning direction (confirmed live:
    # roughly half the pieces on a real file, rendering their exported
    # grain arrow exactly backwards from the piece's actual combed fur).
    # gather_piece_export_data's later "rotate the whole piece so this
    # vector points straight down" step is a pure rotation - it doesn't
    # need the vector pre-flipped to end up pointing down, and applying
    # it to the *true* direction_mm here is what keeps the exported arrow
    # correct.

    # Centered on the piece's own boundary centroid, not the (possibly
    # off-to-one-side) point the user originally clicked to set the grain
    # direction - a real grainline mark is a piece-centered artifact, not
    # tied to interactive click-feedback the on-screen preview arrow needs.
    cx = sum(p[0] for p in boundary_mm) / len(boundary_mm)
    cy = sum(p[1] for p in boundary_mm) / len(boundary_mm)
    center = Vector((cx, cy))

    ts = [(Vector(p) - center).dot(direction_mm) for p in boundary_mm]
    t_min, t_max = min(ts), max(ts)
    half_span = (t_max - t_min) * GRAINLINE_SPAN_FRACTION / 2.0
    if half_span < 1e-6:
        return None

    start = center - direction_mm * half_span
    end = center + direction_mm * half_span
    return ((start.x, start.y), (end.x, end.y))


def _grainline_arrow_for_piece(boundary_mm, grainline_mm):
    """A single-headed block-arrow polygon (see geometry.block_arrow) fitted
    to the largest open run of space inside the piece along the grain
    direction, capped both to the same span _grainline_for_piece's
    centerline already computed (reusing that span - rather than
    recomputing it - both avoids a duplicate direction-mapping call and
    keeps the two in visual agreement about how long a "full-length"
    grainline mark is for this piece) and to ARROW_MAX_LENGTH_MM outright.

    Searches within the boundary shrunk inward by ARROW_EDGE_INSET_MM so
    the arrow never ends up touching the cut/sew line on a piece with
    plenty of room, but falls back to the unshrunk boundary - either
    because the shrink itself failed outright (polygon_offset.OffsetError:
    a piece too thin/complex to survive it), or because it succeeded but
    left too little room for even the smallest arrow (confirmed live: a
    long, tapered piece can lose most of its length to a uniform inset -
    offsetting a shallow taper inward removes a disproportionate amount of
    its pointed ends - even though the *unshrunk* boundary has plenty of
    room). Either way, a slightly-too-close-to-the-edge arrow beats no
    arrow at all. Doesn't try to steer clear of seam-partner labels or the
    title either, for the same reason - on a densely-seamed piece there's
    often nowhere that wouldn't come close to *something*, and a piece
    missing its arrow/title entirely because of that turned out far more
    disruptive than an occasional visual overlap (confirmed: dropping was
    originally added to avoid overlap, but ended up hiding labels/arrows
    on a large fraction of a real multi-piece file, small and large pieces
    alike)."""
    if grainline_mm is None or len(boundary_mm) < 3:
        return None
    (sx, sy), (ex, ey) = grainline_mm
    direction = (ex - sx, ey - sy)
    span = math.hypot(*direction)
    if span < 1e-6:
        return None
    max_length = min(span, ARROW_MAX_LENGTH_MM)

    try:
        search_boundary = polygon_offset.offset_polygon(boundary_mm, -ARROW_EDGE_INSET_MM)
    except polygon_offset.OffsetError:
        search_boundary = boundary_mm

    arrow = block_arrow.fit_block_arrow(search_boundary, direction, max_length)
    if arrow is None and search_boundary is not boundary_mm:
        arrow = block_arrow.fit_block_arrow(boundary_mm, direction, max_length)
    return arrow


def gather_piece_export_data(context, mesh_obj, mm_scale, seam_allowance_override_mm=None):
    """One dict per flattened piece, in plain mm-space data any exporter
    can consume without touching bpy/Blender-specific types again:

    name (the raw "Piece N" piece name), boundary_mm (sew line - the
    zero-offset flattened boundary, simplified - see geometry.simplify -
    to drop mesh-topology zigzag noise before it's used for anything
    else), cut_line_mm (the offset boundary, or None if offset_mm == 0),
    label_pos_mm (the piece's own centroid - always centered), label_text
    (the piece *number* + fur length + swatch name, e.g. '12 (1/4"
    Saffron)' - see _piece_display_name for why "Piece " is dropped),
    label_number/label_suffix (label_text split into the bare number and
    everything after it, e.g. "12" / ' (1/4" Saffron)' - so a renderer can
    bold just the number), label_font_size_mm (shrunk to fit the piece's
    own width - see text_fit.fit_font_size - rather than a fixed size that
    can overflow a small piece), offset_mm, swatch_name (nearest
    fur_colors.py match), fur_type (swatch_name + fur length together,
    e.g. '1/4" Saffron' - the same text as label_suffix, minus the
    parens - what actually distinguishes one "type of fabric to cut from"
    from another; two pieces sharing a swatch_name but a different fur
    length are still a different fur_type. Used to group pieces into
    their own rows in the export layout - see
    geometry.export_layout.regenerate_export_layout), grainline_mm (a (start, end) mm point pair
    spanning the piece along the grain direction, always pointing exactly
    straight down the sheet - or None if the piece has no grain direction
    set. This is achieved by physically rotating the *entire* piece
    (boundary, cut line, notches, labels - every field here) about its own
    centroid, not just picking which end of the direction vector counts as
    "down": real fabric has one straight-of-grain axis, so every piece's
    grain arrow needs to align the same way once laid out on the actual
    cut roll, not just agree on polarity within each piece's own
    (otherwise arbitrary, BFF-flattening-derived) orientation. A piece
    with no grain direction is left in its natural, unrotated orientation
    - kept as simple centerline metadata), grainline_arrow_mm (a single-headed block-arrow polygon,
    capped at ARROW_MAX_LENGTH_MM and inset ARROW_EDGE_INSET_MM from the
    edges, fitted to the largest open space along that same direction/span
    - see geometry.block_arrow - or None if nothing large enough fits
    anywhere at all), notches_mm (list of {"point_mm", "perp"} dicts, perp
    a unit 2D tangent-perpendicular direction for drawing a short tick
    mark there), seam_labels (list of {"text", "mode", ...} dicts - one
    per seam_partners run long enough to bother marking, "text" the
    *other* piece's display number - "this edge sews to that piece" -
    "mode" either "curved" (path_mm: an inset polyline just inside that
    edge, for text-on-path rendering) or "straight" (pos_mm/dir_mm: a
    single inset anchor point + direction, used instead when the edge
    bends too sharply - SEAM_LABEL_MAX_TURN_DEG - for curved text to stay
    legible).

    The title and grainline arrow are always drawn at their natural
    (centered / largest-open-space) position and never hidden just
    because they'd come close to a seam-partner label - an earlier
    drop-on-overlap design traded that occasional overlap for missing
    piece numbers/arrows on a large fraction of a real multi-piece file,
    which was worse.

    A piece whose sew-line area is below MIN_PIECE_AREA_MM2 is skipped
    entirely - a defensive catch for a genuinely-degenerate sliver, kept
    deliberately low so it never touches a real, legitimately small piece
    (see MIN_PIECE_AREA_MM2's own comment for why).

    seam_allowance_override_mm: a fallback, not a blanket override - only
    applies to a piece whose own offset_mm is still 0 (i.e. no seam
    allowance has ever been set on it individually), computing cut_line_mm
    fresh at this one uniform distance via geometry.polygon_offset instead
    of leaving it None. Any piece that already has its own nonzero
    offset_mm keeps using that and its already-baked cut_line_object,
    completely unaffected by this value - the point is to give every
    not-yet-configured piece a sane default seam allowance for this
    export, not to clobber allowances someone already deliberately tuned
    per piece. None (the default) leaves every offset_mm==0 piece with no
    cut line at all, exactly as before this parameter existed. A piece
    whose boundary can't be offset that far (geometry.polygon_offset.
    OffsetError - too concave a boundary for the distance) is skipped the
    same way a piece missing its baked cut_line_object already was:
    cut_line_mm just comes back None, not a hard failure for the whole
    export.
    """
    piece_by_uuid = {p.uuid: p for p in mesh_obj.seams_to_fur_pieces}

    pieces = []
    for piece in mesh_obj.seams_to_fur_pieces:
        flat_obj = bpy.data.objects.get(piece.flattened_object) if piece.flattened_object else None
        if flat_obj is None or flat_obj.type != "MESH":
            continue

        faces = [list(poly.vertices) for poly in flat_obj.data.polygons]
        try:
            loop_indices = boundary.ordered_boundary_loop(faces)
        except boundary.BoundaryError:
            continue

        mat = flat_obj.matrix_world
        boundary_mm = [_local_to_mm(mat, mm_scale, flat_obj.data.vertices[i].co) for i in loop_indices]
        # Dropping near-collinear mesh-topology noise before it's used for
        # anything else (the sew line itself, offsetting, arrow search) -
        # see geometry.simplify's own docstring for why this specifically
        # matters for the seam-allowance line.
        boundary_mm = simplify.simplify_polygon(boundary_mm)

        if polygon_offset.polygon_area(boundary_mm) < MIN_PIECE_AREA_MM2:
            continue

        cut_line_mm = None
        use_override = (
            piece.offset_mm == 0.0
            and seam_allowance_override_mm is not None
            and seam_allowance_override_mm > 0.0
        )
        if use_override:
            offset_mm = seam_allowance_override_mm
            try:
                cut_line_mm = polygon_offset.offset_polygon(boundary_mm, offset_mm)
            except polygon_offset.OffsetError:
                cut_line_mm = None
        else:
            offset_mm = piece.offset_mm
            cut_obj = bpy.data.objects.get(piece.cut_line_object) if piece.cut_line_object else None
            if cut_obj is not None and cut_obj.type == "CURVE" and cut_obj.data.splines:
                spline = cut_obj.data.splines[0]
                cmat = cut_obj.matrix_world
                cut_line_mm = [_local_to_mm(cmat, mm_scale, pt.co.to_3d()) for pt in spline.points]

        xs = [p[0] for p in boundary_mm]
        ys = [p[1] for p in boundary_mm]
        label_pos_mm = (sum(xs) / len(xs), sum(ys) / len(ys))
        piece_width_mm = max(xs) - min(xs)
        available_width_mm = piece_width_mm * LABEL_WIDTH_FRACTION

        proxy = appearance._build_flat_proxy(flat_obj)

        grainline_mm = (
            _grainline_for_piece(piece, flat_obj, proxy, mat, mm_scale, boundary_mm)
            if proxy is not None
            else None
        )

        swatch_name = nearest_swatch_name(piece.color)
        display_name = _piece_display_name(piece)
        fur_inches = _fur_inches_label(piece, mm_scale)
        fur_type = f"{fur_inches} {swatch_name}" if swatch_name else fur_inches
        label_suffix = f" ({fur_type})"
        label_text = f"{display_name}{label_suffix}"
        label_font_size_mm = text_fit.fit_font_size(label_text, available_width_mm, LABEL_MAX_FONT_MM, LABEL_MIN_FONT_MM)

        seam_labels = (
            _seam_labels_for_piece(piece, flat_obj, proxy, mat, mm_scale, piece_by_uuid, boundary_mm)
            if proxy is not None
            else []
        )

        piece_dict = {
            "name": piece.name,
            "boundary_mm": boundary_mm,
            "cut_line_mm": cut_line_mm,
            "label_pos_mm": label_pos_mm,
            "label_text": label_text,
            "label_number": display_name,
            "label_suffix": label_suffix,
            "label_font_size_mm": label_font_size_mm,
            "offset_mm": offset_mm,
            "swatch_name": swatch_name,
            "fur_type": fur_type,
            "grainline_mm": grainline_mm,
            "grainline_arrow_mm": _grainline_arrow_for_piece(boundary_mm, grainline_mm),
            "notches_mm": (
                _notches_for_piece(piece, flat_obj, proxy, mat, mm_scale, boundary_mm) if proxy is not None else []
            ),
            "seam_labels": seam_labels,
        }

        # Physically rotate the whole piece (not just the grainline's own
        # sign) so its grain direction points exactly straight down - real
        # fabric has one straight-of-grain axis, so every piece needs to
        # line up against it the same way once laid out on the cut roll,
        # not just agree on up-vs-down within its own otherwise-arbitrary
        # flattened orientation. Pieces with no grain direction set are
        # left in their natural orientation - there's nothing to align to.
        if grainline_mm is not None:
            (sx, sy), (ex, ey) = grainline_mm
            current_angle = math.atan2(ey - sy, ex - sx)
            target_angle = -math.pi / 2.0  # straight down
            export_layout.rotate_piece_geometry(piece_dict, target_angle - current_angle, label_pos_mm)

        pieces.append(piece_dict)

    # Fabric-swatch-grouped grid layout regenerated fresh for export - not
    # whatever positions flatten_all's own viewport-oriented layout (or a
    # user's manual drag) happened to leave each piece at, which have
    # nothing to do with what's convenient for actually cutting fabric.
    return export_layout.regenerate_export_layout(pieces)
