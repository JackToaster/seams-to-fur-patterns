import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from export import notch

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
GROWN = [(-5.0, -5.0), (105.0, -5.0), (105.0, 105.0), (-5.0, 105.0)]


def test_slit_runs_from_sew_line_out_to_cut_line():
    start, end = notch.notch_slit((0.0, 50.0), (1.0, 0.0), SQUARE, GROWN)
    assert start == (0.0, 50.0)
    assert math.isclose(end[0], -5.0, abs_tol=1e-9)
    assert math.isclose(end[1], 50.0, abs_tol=1e-9)


def test_perp_sign_is_not_trusted():
    # perp pointing *into* the piece must still produce an outward slit.
    for perp in ((1.0, 0.0), (-1.0, 0.0)):
        _, end = notch.notch_slit((0.0, 50.0), perp, SQUARE, GROWN)
        assert end[0] < 0.0, f"slit for perp={perp} went into the piece: {end}"


def test_slit_length_matches_allowance_on_slanted_edge():
    tri = [(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)]
    grown = [(-8.0, -5.0), (108.0, -5.0), (50.0, 90.0)]
    _, end = notch.notch_slit((50.0, 0.0), (0.0, 1.0), tri, grown)
    assert math.isclose(end[1], -5.0, abs_tol=1e-9)


def test_no_cut_line_falls_back_to_short_outward_slit():
    start, end = notch.notch_slit((100.0, 50.0), (-1.0, 0.0), SQUARE, None)
    assert math.isclose(math.dist(start, end), notch.NOTCH_FALLBACK_LENGTH_MM)
    assert end[0] > 100.0
