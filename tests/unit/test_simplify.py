import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import simplify


def test_removes_near_collinear_zigzag_points():
    # A mostly-straight line with tiny (0.1mm) zigzag noise added to
    # interior points - well under a 0.75mm tolerance, should all drop,
    # leaving just the two real endpoints.
    points = [(0.0, 0.0), (10.0, 0.05), (20.0, -0.05), (30.0, 0.03), (40.0, 0.0)]
    simplified = simplify.simplify_polyline(points, epsilon=0.75)
    assert simplified == [(0.0, 0.0), (40.0, 0.0)]


def test_keeps_a_real_corner():
    # A genuine right-angle turn, deviation (10mm) far over tolerance -
    # the corner point must survive simplification.
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    simplified = simplify.simplify_polyline(points, epsilon=0.75)
    assert simplified == points


def test_endpoints_always_preserved():
    points = [(0.0, 0.0), (5.0, 0.01), (10.0, 0.0), (15.0, -0.01), (20.0, 0.0)]
    simplified = simplify.simplify_polyline(points, epsilon=0.5)
    assert simplified[0] == points[0]
    assert simplified[-1] == points[-1]


def test_short_polyline_passes_through_unchanged():
    assert simplify.simplify_polyline([(0.0, 0.0)]) == [(0.0, 0.0)]
    assert simplify.simplify_polyline([(0.0, 0.0), (1.0, 1.0)]) == [(0.0, 0.0), (1.0, 1.0)]


def test_simplify_polygon_removes_noise_but_keeps_square_corners():
    # A square with a couple of zigzag-noise points inserted along one
    # edge - the 4 real corners must survive, the noise must not.
    square_with_noise = [
        (0.0, 0.0),
        (5.0, 0.03),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    ]
    simplified = simplify.simplify_polygon(square_with_noise, epsilon=0.75)
    assert simplified == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_simplify_polygon_never_drops_below_three_points():
    triangle = [(0.0, 0.0), (10.0, 0.01), (5.0, 10.0)]
    simplified = simplify.simplify_polygon(triangle, epsilon=1000.0)
    assert len(simplified) >= 3


def test_simplify_polygon_preserves_a_sharp_dart_tip():
    # A thin dart notch (two nearly-parallel edges converging to a tip) -
    # a real, intentional design feature, not noise - must survive even
    # at a real zigzag-cleanup tolerance.
    dart_shape = [
        (0.0, 0.0), (100.0, 0.0), (100.0, 100.0),
        (55.0, 100.0), (50.0, 40.0), (45.0, 100.0),
        (0.0, 100.0),
    ]
    simplified = simplify.simplify_polygon(dart_shape, epsilon=0.75)
    assert (50.0, 40.0) in simplified, "the dart tip must not be simplified away"
