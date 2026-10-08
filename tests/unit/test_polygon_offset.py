import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import polygon_offset as po


def area(points):
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def test_polygon_area_is_unsigned_and_winding_independent():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    reversed_square = list(reversed(square))
    assert po.polygon_area(square) == 100.0
    assert po.polygon_area(reversed_square) == 100.0


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


def test_corner_sharper_than_60_degrees_is_beveled_not_spiked():
    # An ~11.4-degree spike on top of an otherwise unremarkable rectangle
    # (its two shoulders are ~95.7 degrees - comfortably above the 60
    # degree threshold, so only the tip itself should bevel) - a real seam
    # allowance can't be sewn to a fine point at a corner this sharp, so
    # it must be mitred (clipped/beveled) rather than left to spike out to
    # the mathematically-exact but practically unusable full miter
    # distance (confirmed live: real ~30-50 degree pattern-piece corners
    # spiking tens of mm past the piece).
    tip = (50.0, 70.0)
    spike = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 20.0),
        (55.0, 20.0), tip, (45.0, 20.0),
        (0.0, 20.0),
    ]
    assert po._interior_angle_deg((55.0, 20.0), tip, (45.0, 20.0)) < po.MITER_MIN_ANGLE_DEG
    assert po._interior_angle_deg((100.0, 20.0), (55.0, 20.0), tip) > po.MITER_MIN_ANGLE_DEG

    distance = 5.0
    grown = po.offset_polygon(spike, distance)

    # Beveling the tip replaces it with two points, one per edge -
    # everything else in this simple, non-self-intersecting shape is
    # untouched, so the count grows by exactly one.
    assert len(grown) == len(spike) + 1

    # Neither bevel point should land anywhere near the full-miter
    # distance a spike this sharp would otherwise reach.
    full_miter_dist = distance / math.sin(math.radians(11.4) / 2.0)
    dists = sorted(math.hypot(pt[0] - tip[0], pt[1] - tip[1]) for pt in grown)
    for d in dists[:2]:  # the two points closest to the tip - the bevel
        assert d < full_miter_dist * 0.5, f"bevel point is {d:.1f} from the tip, much farther than it should reach"
    assert area(grown) > area(spike)


def test_corner_at_or_above_60_degrees_still_gets_a_full_miter():
    # A 90-degree corner (comfortably at/above MITER_MIN_ANGLE_DEG) must
    # still come to a single, exact mitered point - beveling isn't a
    # blanket policy, only sharper-than-60-degree corners get it.
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    distance = 2.0
    grown = po.offset_polygon(square, distance)
    assert len(grown) == len(square)
    assert grown == [(-2.0, -2.0), (12.0, -2.0), (12.0, 12.0), (-2.0, 12.0)]


def test_dart_self_intersection_is_repaired_when_growing():
    # A thin V-shaped dart notch cut into the top edge (mirrors a real
    # flattened piece with a dart - see geometry.simplify's own dart
    # fixture). Growing (seam allowance) by more than the notch's own
    # half-width would - with a plain per-corner miter join and nothing
    # else - cross the offset lines from either side of the notch before
    # they reach the offset tip, producing a self-intersecting "bowtie"
    # (confirmed on a real flattened piece with a dart: exactly this).
    # The repair must produce a genuinely simple (non-self-intersecting)
    # polygon instead of silently returning the crossed result.
    dart_shape = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 100.0),
        (55.0, 100.0), (50.0, 40.0), (45.0, 100.0),
        (0.0, 100.0),
    ]
    grown = po.offset_polygon(dart_shape, 10.0)
    assert po._find_self_intersection(grown) is None, "offset result must be a simple polygon"
    assert area(grown) > area(dart_shape)


def _point_in_polygon(pt, points):
    # Standard ray-casting even-odd test.
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


def test_needle_spike_is_beveled_not_left_as_an_unbounded_miter():
    # A near-zero-width ~8mm-tall needle poking out of an otherwise normal
    # edge (mirrors a real flattened piece: a genuine boundary defect, not
    # mesh-noise geometry.simplify would remove, since its amplitude is
    # well above simplify's epsilon). A plain per-vertex miter join has no
    # cap, so growing this by 5mm sent the tip's offset point ~150mm away
    # in a basically arbitrary direction (confirmed on the real piece) -
    # this must instead bevel that one vertex and stay local.
    spike = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 20.0),
        (51.0, 20.0), (50.5, 28.2), (50.0, 20.0),
        (0.0, 20.0),
    ]
    distance = 5.0
    grown = po.offset_polygon(spike, distance)
    assert po._find_self_intersection(grown) is None
    min_x, min_y, max_x, max_y = _bbox(grown)
    # A bevel point is exactly `distance` from its vertex (vertex + unit
    # normal * distance) - nowhere near the ~30x blowout an unbounded
    # miter produced on the real file.
    assert max_y < 28.2 + distance + 1e-6
    assert max_y > 20.0 + distance - 1e-6, "the spike should still poke out past the offset edge, just not unboundedly"


def test_narrow_reflex_dart_tapers_to_a_point_without_crossing_the_sew_line():
    # A dart whose channel is much narrower than 2x the offset distance,
    # with an asymmetric vertex count on each leg (like a real flattened
    # piece's dart, which rarely samples both sides identically) - the
    # naive miter at the reflex tip vertex is numerically unstable (the two
    # adjacent edges nearly double back on themselves) and, uncapped, used
    # to land the tip's offset point in a essentially arbitrary spot far
    # from the dart entirely. That wild point then corrupted self-
    # intersection repair into discarding the wrong side of the dart,
    # leaving the "seam allowance" cut line crossing back inside the sew
    # line on one wall of the dart (confirmed on a real dart-bearing
    # piece). With the tip properly beveled, repair should fall back to
    # its ordinary job: taper the offset to a single clean point wherever
    # the two walls' offsets first cross, and every resulting point should
    # stay outside (or on) the original boundary - never inside it.
    dart_shape = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 100.0),
        (55.0, 100.0), (50.0, 85.0), (49.0, 95.0), (45.0, 100.0),
        (0.0, 100.0),
    ]
    distance = 5.0
    grown = po.offset_polygon(dart_shape, distance)
    assert po._find_self_intersection(grown) is None
    assert area(grown) > area(dart_shape)
    for pt in grown:
        assert not _point_in_polygon(pt, dart_shape), (
            f"offset point {pt} landed inside the original boundary - the "
            f"seam allowance line must never cross back inside the sew line"
        )


def test_offset_is_roughly_isotropic_on_circle_like_polygon():
    n = 64
    radius = 10.0
    circle = [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
    grown = po.offset_polygon(circle, 1.0)
    radii = [math.hypot(x, y) for x, y in grown]
    assert all(abs(r - (radius + 1.0)) < 0.05 for r in radii)
