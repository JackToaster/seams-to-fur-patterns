import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "usbee"))

from export import svg as svg_export


def _viewbox(svg_text):
    m = re.search(r'viewBox="([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"', svg_text)
    assert m, "no viewBox found in SVG"
    return tuple(float(g) for g in m.groups())


def test_known_dimension_round_trips_through_viewbox():
    # A 100mm x 50mm rectangle, offset from the origin so this also
    # exercises the min/max bbox math, not just a piece already at (0,0).
    piece = {
        "name": "Test",
        "boundary_mm": [(20, 20), (120, 20), (120, 70), (20, 70)],
        "cut_line_mm": None,
        "label_pos_mm": (70, 45),
    }
    svg_text = svg_export.build_svg([piece], margin_mm=5.0)
    _, _, width, height = _viewbox(svg_text)
    assert abs(width - (100 + 10)) < 1e-6  # +2*margin
    assert abs(height - (50 + 10)) < 1e-6


def test_cut_line_included_when_present():
    piece = {
        "name": "Test",
        "boundary_mm": [(0, 0), (10, 0), (10, 10), (0, 10)],
        "cut_line_mm": [(-2, -2), (12, -2), (12, 12), (-2, 12)],
        "label_pos_mm": (5, 5),
    }
    svg_text = svg_export.build_svg([piece])
    assert svg_text.count("<polygon") == 2


def test_no_cut_line_omits_second_polygon():
    piece = {
        "name": "Test",
        "boundary_mm": [(0, 0), (10, 0), (10, 10), (0, 10)],
        "cut_line_mm": None,
        "label_pos_mm": (5, 5),
    }
    svg_text = svg_export.build_svg([piece])
    assert svg_text.count("<polygon") == 1


def test_empty_pieces_list_produces_valid_empty_svg():
    svg_text = svg_export.build_svg([])
    assert "<svg" in svg_text and "</svg>" in svg_text
