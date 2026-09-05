import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "usbee"))

from geometry import polygon_offset as po


def area(points):
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


def test_square_grow():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    grown = po.offset_polygon(square, 2)
    assert grown == [(-2.0, -2.0), (12.0, -2.0), (12.0, 12.0), (-2.0, 12.0)]
    assert area(grown) == 14 * 14


def test_square_shrink():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    shrunk = po.offset_polygon(square, -2)
    assert shrunk == [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)]
    assert area(shrunk) == 6 * 6


def test_zero_offset_is_identity():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert po.offset_polygon(square, 0.0) == square


def test_concave_l_shape_grow_and_shrink():
    lshape = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    grown = po.offset_polygon(lshape, 1)
    shrunk = po.offset_polygon(lshape, -1)
    assert area(grown) > area(lshape) > area(shrunk)


def test_thin_sliver_large_shrink_raises():
    # A thin 10x1 rectangle can't survive a shrink of 1 (half-width) without
    # inverting.
    sliver = [(0, 0), (10, 0), (10, 1), (0, 1)]
    with pytest.raises(po.OffsetError):
        po.offset_polygon(sliver, -1)


def test_degenerate_polygon_raises():
    with pytest.raises(po.OffsetError):
        po.offset_polygon([(0, 0), (1, 0)], 1)


def test_zero_area_polygon_raises():
    with pytest.raises(po.OffsetError):
        po.offset_polygon([(0, 0), (1, 0), (2, 0)], 1)


def test_offset_is_roughly_isotropic_on_circle_like_polygon():
    n = 64
    radius = 10.0
    circle = [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
    grown = po.offset_polygon(circle, 1.0)
    radii = [math.hypot(x, y) for x, y in grown]
    assert all(abs(r - (radius + 1.0)) < 0.05 for r in radii)
