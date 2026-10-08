import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import export_layout


def _square_piece(name, swatch_name, x0, y0, size=100.0, fur_type=None):
    return {
        "name": name,
        "swatch_name": swatch_name,
        "fur_type": fur_type if fur_type is not None else swatch_name,
        "boundary_mm": [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)],
        "cut_line_mm": [(x0 - 10, y0 - 10), (x0 + size + 10, y0 - 10), (x0 + size + 10, y0 + size + 10), (x0 - 10, y0 + size + 10)],
        "label_pos_mm": (x0 + size / 2.0, y0 + size / 2.0),
        "grainline_mm": ((x0 + 10, y0 + 50), (x0 + 90, y0 + 50)),
        "grainline_arrow_mm": [(x0 + 40, y0 + 10), (x0 + 60, y0 + 10), (x0 + 50, y0 + 90)],
        "notches_mm": [{"point_mm": (x0, y0 + 50), "perp": (1.0, 0.0)}],
        "seam_labels": [{"path_mm": [(x0 + 5, y0 + 5), (x0 + 5, y0 + 95)], "text": "Piece Z"}],
    }


def test_groups_pieces_by_fur_type_not_input_order():
    # Deliberately interleaved input order - Red, Blue, Red - to prove the
    # output groups by fur_type (color+length), not by whatever order
    # they were gathered in.
    pieces = [
        _square_piece("Piece A", "Red", 500, 500, fur_type='1/4" Red'),
        _square_piece("Piece B", "Blue", 900, 900, fur_type='1" Blue'),
        _square_piece("Piece C", "Red", 100, 100, fur_type='1/4" Red'),
    ]
    ordered = export_layout.regenerate_export_layout(list(pieces))
    types = [p["fur_type"] for p in ordered]
    assert types == sorted(types), f"expected pieces grouped by fur_type, got order {types}"
    # The two same-type pieces should be adjacent in the output.
    red_indices = [i for i, t in enumerate(types) if t == '1/4" Red']
    assert red_indices == [red_indices[0], red_indices[0] + 1]


def test_same_swatch_different_fur_length_is_a_different_group():
    # Same color, different length - fur_type (not swatch_name alone) is
    # what actually distinguishes one type of fabric to cut from another.
    pieces = [
        _square_piece("Piece A", "Red", 0, 0, fur_type='1/4" Red'),
        _square_piece("Piece B", "Red", 0, 0, fur_type='1" Red'),
    ]
    ordered = export_layout.regenerate_export_layout(list(pieces), margin_mm=10.0)
    # Different groups must start fresh rows, never share one.
    ys = [p["boundary_mm"][0][1] for p in ordered]
    assert ys[0] != ys[1], "pieces with different fur_type must not share a row"


def test_group_boundary_leaves_group_margin_not_just_piece_margin():
    pieces = [
        _square_piece("Piece A", "Red", 0, 0, fur_type='1/4" Red'),
        _square_piece("Piece B", "Blue", 0, 0, fur_type='1" Blue'),
    ]
    ordered = export_layout.regenerate_export_layout(
        list(pieces), margin_mm=10.0, group_margin_mm=40.0
    )
    first, second = ordered  # whichever fur_type actually sorts first
    assert first["fur_type"] != second["fur_type"]
    first_max_y = max(pt[1] for pt in first["cut_line_mm"])
    second_min_y = min(pt[1] for pt in second["cut_line_mm"])
    gap = second_min_y - first_max_y
    assert gap >= 40.0 - 1e-9, f"expected at least group_margin_mm (40.0) between groups, got {gap:.1f}"


def test_pieces_dont_overlap_after_layout():
    pieces = [_square_piece(f"Piece {i}", "Red", i * 1000, i * 1000, size=200.0) for i in range(4)]
    ordered = export_layout.regenerate_export_layout(list(pieces), shelf_width_mm=450.0, margin_mm=10.0)

    def bbox(p):
        xs = [pt[0] for pt in p["boundary_mm"]]
        ys = [pt[1] for pt in p["boundary_mm"]]
        return min(xs), min(ys), max(xs), max(ys)

    boxes = [bbox(p) for p in ordered]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            overlap = not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)
            assert not overlap, f"pieces {i} and {j} overlap after layout: {boxes[i]} vs {boxes[j]}"


def test_all_geometry_fields_translate_together_consistently():
    piece = _square_piece("Piece A", "Red", 500, 500)
    orig_boundary = list(piece["boundary_mm"])
    orig_grain_start = piece["grainline_mm"][0]
    orig_notch = piece["notches_mm"][0]["point_mm"]
    orig_seam_label_pt = piece["seam_labels"][0]["path_mm"][0]

    ordered = export_layout.regenerate_export_layout([piece])
    moved = ordered[0]

    dx = moved["boundary_mm"][0][0] - orig_boundary[0][0]
    dy = moved["boundary_mm"][0][1] - orig_boundary[0][1]
    assert (dx, dy) != (0.0, 0.0), "expected the single piece to actually be repositioned by the layout"

    assert abs(moved["grainline_mm"][0][0] - (orig_grain_start[0] + dx)) < 1e-9
    assert abs(moved["grainline_mm"][0][1] - (orig_grain_start[1] + dy)) < 1e-9
    assert abs(moved["notches_mm"][0]["point_mm"][0] - (orig_notch[0] + dx)) < 1e-9
    assert abs(moved["seam_labels"][0]["path_mm"][0][0] - (orig_seam_label_pt[0] + dx)) < 1e-9
    assert abs(moved["label_pos_mm"][0] - (550.0 + dx)) < 1e-9  # original center x was 500+100/2


def test_layout_spaces_cut_lines_not_just_sew_lines():
    # Two tapered pieces (pointed tips, like a real leaf-shaped piece) with
    # a seam allowance that reaches past their own sew line - if layout
    # only spaced sew-line bboxes apart, the pointed cut-line tips could
    # still end up crossing (confirmed live on a real file: exactly this).
    def tapered_piece(name, x0):
        boundary = [(x0, 0), (x0 + 40, 50), (x0, 100), (x0 - 40, 50)]
        cut_line = [(x0, -8), (x0 + 48, 50), (x0, 108), (x0 - 48, 50)]
        return {
            "name": name,
            "swatch_name": "Red",
            "boundary_mm": boundary,
            "cut_line_mm": cut_line,
            "label_pos_mm": (x0, 50),
            "grainline_mm": None,
            "grainline_arrow_mm": None,
            "notches_mm": [],
            "seam_labels": [],
        }

    pieces = [tapered_piece("Piece A", 0), tapered_piece("Piece B", 1000)]
    ordered = export_layout.regenerate_export_layout(list(pieces), margin_mm=15.0)

    a_cut_max_x = max(pt[0] for pt in ordered[0]["cut_line_mm"])
    b_cut_min_x = min(pt[0] for pt in ordered[1]["cut_line_mm"])
    assert b_cut_min_x - a_cut_max_x >= 15.0 - 1e-9, (
        f"cut lines are only {b_cut_min_x - a_cut_max_x:.1f}mm apart, expected >= margin_mm (15.0)"
    )


def test_rotate_piece_geometry_rotates_every_field_about_the_pivot():
    piece = _square_piece("Piece A", "Red", 0, 0)
    piece["seam_labels"].append(
        {"mode": "straight", "pos_mm": (60.0, 50.0), "dir_mm": (1.0, 0.0), "text": "Piece Y"}
    )
    pivot = (50.0, 50.0)

    export_layout.rotate_piece_geometry(piece, math.pi / 2.0, pivot)

    # +90 degrees (CCW, standard math convention) about (50,50): a point
    # straight right of the pivot ends up straight above it.
    x, y = piece["grainline_mm"][1]  # was (90, 50) - 40mm right of pivot
    assert abs(x - 50.0) < 1e-6
    assert abs(y - 90.0) < 1e-6

    # Direction-only fields rotate in place, no pivot/translation.
    perp = piece["notches_mm"][0]["perp"]
    assert abs(perp[0] - 0.0) < 1e-6 and abs(perp[1] - 1.0) < 1e-6

    straight_label = next(l for l in piece["seam_labels"] if l.get("mode") == "straight")
    assert abs(straight_label["dir_mm"][0] - 0.0) < 1e-6
    assert abs(straight_label["dir_mm"][1] - 1.0) < 1e-6
    # pos_mm rotates about the pivot like any other point: (60,50) is 10mm
    # right of (50,50), so it ends up 10mm above it.
    assert abs(straight_label["pos_mm"][0] - 50.0) < 1e-6
    assert abs(straight_label["pos_mm"][1] - 60.0) < 1e-6

    # label_pos_mm (the piece centroid, (50,50)) sits exactly on the pivot,
    # so it must be unmoved by any rotation about that same point.
    assert abs(piece["label_pos_mm"][0] - 50.0) < 1e-6
    assert abs(piece["label_pos_mm"][1] - 50.0) < 1e-6


def test_rotate_piece_geometry_aligns_an_arbitrary_grainline_straight_down():
    # Mirrors how operators/export_common.py actually uses this: solve for
    # the angle that brings an arbitrary-direction grainline to point
    # exactly straight down, then rotate the whole piece by it.
    piece = _square_piece("Piece A", "Red", 0, 0)
    piece["grainline_mm"] = ((10.0, 10.0), (90.0, 90.0))  # 45 degrees, not vertical
    pivot = piece["label_pos_mm"]

    (sx, sy), (ex, ey) = piece["grainline_mm"]
    current_angle = math.atan2(ey - sy, ex - sx)
    target_angle = -math.pi / 2.0
    export_layout.rotate_piece_geometry(piece, target_angle - current_angle, pivot)

    (sx, sy), (ex, ey) = piece["grainline_mm"]
    dx, dy = ex - sx, ey - sy
    assert abs(dx) < 1e-6, "grainline must point exactly straight down after alignment"
    assert dy < 0.0, "and specifically downward (negative Y), not upward"


def test_shelf_wraps_to_a_new_row_when_it_would_exceed_shelf_width():
    pieces = [_square_piece(f"Piece {i}", "Red", 0, 0, size=300.0) for i in range(3)]
    ordered = export_layout.regenerate_export_layout(list(pieces), shelf_width_mm=700.0, margin_mm=10.0)
    ys = [p["boundary_mm"][0][1] for p in ordered]
    # Two pieces of width 300 fit in a 700-wide shelf (300+10+300=610), the
    # third doesn't (610+10+300=920>700) and must wrap to a new row.
    assert ys[0] == ys[1], "first two pieces should share a row"
    assert ys[2] > ys[1], "third piece should have wrapped to a new row"
