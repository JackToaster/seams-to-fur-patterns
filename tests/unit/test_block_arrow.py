import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import block_arrow as ba


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def test_large_square_gets_ideal_width_and_capped_length():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    arrow = ba.fit_block_arrow(square, (0, 1), max_length=60.0)
    assert arrow is not None
    min_x, min_y, max_x, max_y = _bbox(arrow)
    width = max_x - min_x
    length = max_y - min_y
    assert abs(width - ba.IDEAL_WIDTH_MM) < 1e-6
    assert abs(length - 60.0) < 1e-6
    # Roughly centered in the square.
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    assert abs(cx - 50.0) < 2.0
    assert abs(cy - 50.0) < 2.0


def test_direction_need_not_be_axis_aligned():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    direction = (1.0, 1.0)  # 45 degrees, not normalized on input
    arrow = ba.fit_block_arrow(square, direction, max_length=40.0)
    assert arrow is not None
    assert len(arrow) == 7


def test_shrinks_width_to_fit_a_narrow_strip():
    strip = [(0, 0), (4, 0), (4, 100), (0, 100)]
    arrow = ba.fit_block_arrow(strip, (0, 1), max_length=60.0, ideal_width=ba.IDEAL_WIDTH_MM)
    assert arrow is not None
    min_x, _min_y, max_x, _max_y = _bbox(arrow)
    width = max_x - min_x
    assert width < ba.IDEAL_WIDTH_MM, "a 4mm-wide strip can't fit an 8mm-wide arrow unshrunk"
    assert width <= 4.0 + 1e-6


def test_returns_none_when_nothing_fits():
    sliver = [(0, 0), (0.5, 0), (0.5, 100), (0, 100)]
    arrow = ba.fit_block_arrow(sliver, (0, 1), max_length=60.0)
    assert arrow is None


def test_avoids_a_notch_finds_open_space_beside_it():
    # A 100x100 square with a deep notch bitten out of the middle third,
    # from the top down to y=40 - the only way an arrow taller than 60mm
    # fits at full/ideal width is entirely to one side of the notch, not
    # straddling the middle where it's bitten out.
    notched = [
        (0, 0), (100, 0), (100, 100), (70, 100), (70, 40), (30, 40), (30, 100), (0, 100),
    ]
    arrow = ba.fit_block_arrow(notched, (0, 1), max_length=90.0, ideal_width=8.0)
    assert arrow is not None
    min_x, min_y, max_x, max_y = _bbox(arrow)
    assert max_y - min_y > 60.0, "expected a long arrow using the full-height side column, not squeezed into the notch's 40mm"
    center_x = (min_x + max_x) / 2.0
    assert center_x < 30.0 or center_x > 70.0, (
        f"arrow (center_x={center_x:.1f}) should sit in one of the full-height side strips, not straddle the notch"
    )


def test_arrow_polygon_is_a_closed_7_point_block_arrow_shape():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    arrow = ba.fit_block_arrow(square, (0, 1), max_length=50.0)
    assert len(arrow) == 7
    # All x coordinates distinct enough to not be degenerate.
    xs = sorted(set(round(p[0], 3) for p in arrow))
    assert len(xs) >= 3
