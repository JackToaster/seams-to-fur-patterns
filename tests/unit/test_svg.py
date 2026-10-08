import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

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


def _full_piece(name, x0=0.0):
    return {
        "name": name,
        "boundary_mm": [(x0, 0), (x0 + 100, 0), (x0 + 100, 100), (x0, 100)],
        "cut_line_mm": [(x0 - 5, -5), (x0 + 105, -5), (x0 + 105, 105), (x0 - 5, 105)],
        "label_pos_mm": (x0 + 50, 50),
        "label_number": "6",
        "label_suffix": ' (1/4" Black)',
        "label_font_size_mm": 4.0,
        "fur_type": '1/4" Black',
        "grainline_arrow_mm": [(x0 + 46, 20), (x0 + 54, 20), (x0 + 50, 80)],
        "notches_mm": [{"point_mm": (x0, 50), "perp": (1.0, 0.0)}],
        "seam_labels": [
            {"mode": "curved", "path_mm": [(x0 + 5, 5), (x0 + 5, 95)], "text": "9"},
            {"mode": "straight", "pos_mm": (x0 + 50, 95), "dir_mm": (-1.0, 0.0), "text": "16"},
        ],
    }


def _piece_groups(svg_text):
    return re.findall(r'<g id="piece-\d+" data-name="[^"]*">(.*?)</g>', svg_text)


def test_each_piece_is_one_group_holding_all_its_layers():
    svg_text = svg_export.build_svg([_full_piece("Piece 1"), _full_piece("Piece 2", x0=200)])
    groups = _piece_groups(svg_text)
    assert len(groups) == 2
    for body in groups:
        for color in svg_export.LAYER_COLORS["cut"], svg_export.LAYER_COLORS["sew"], svg_export.LAYER_COLORS["mark"]:
            assert f'stroke="{color}"' in body, f"piece group missing its {color} elements"
    # No inkscape layers wrapping everything per-category.
    assert "inkscape:groupmode" not in svg_text


def test_all_strokes_use_exact_lightburn_palette_colors():
    svg_text = svg_export.build_svg([_full_piece("Piece 1")])
    used = set(re.findall(r'stroke="(#[0-9A-Fa-f]{6})"', svg_text))
    assert used <= set(svg_export.LAYER_COLORS.values()), used


def test_no_svg_text_elements_text_is_vector_paths():
    svg_text = svg_export.build_svg([_full_piece("Piece 1")])
    assert "<text" not in svg_text and "<textPath" not in svg_text
    assert "<path" in svg_text


def test_notch_is_a_cut_slit_from_sew_line_to_cut_line():
    svg_text = svg_export.build_svg([_full_piece("Piece 1")], margin_mm=0.0)
    lines = re.findall(r'<line x1="([\d.-]+)" y1="([\d.-]+)" x2="([\d.-]+)" y2="([\d.-]+)" stroke="(#\w+)"', svg_text)
    assert len(lines) == 1
    x1, y1, x2, y2, color = lines[0]
    assert color == svg_export.LAYER_COLORS["cut"]
    assert abs(abs(float(x2) - float(x1)) - 5.0) < 1e-6  # spans exactly the 5mm allowance
    assert float(y1) == float(y2)


def test_outline_is_cut_color_when_there_is_no_seam_allowance():
    piece = _full_piece("Piece 1")
    piece["cut_line_mm"] = None
    svg_text = svg_export.build_svg([piece])
    first_polygon = re.search(r'<polygon [^>]*stroke="(#\w+)"', svg_text).group(1)
    assert first_polygon == svg_export.LAYER_COLORS["cut"]
