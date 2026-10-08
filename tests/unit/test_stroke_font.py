import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from export import stroke_font as sf


def _rot180(strokes):
    return sorted(
        tuple(sorted((round(4 - x, 3), round(6 - y, 3)) for x, y in stroke)) for stroke in strokes
    )


def _norm(strokes):
    return sorted(tuple(sorted((round(x, 3), round(y, 3)) for x, y in stroke)) for stroke in strokes)


def test_six_and_nine_are_not_rotations_of_each_other():
    assert _rot180(sf.GLYPHS["6"]) != _norm(sf.GLYPHS["9"])


def test_every_swatch_name_and_fur_length_character_has_a_real_glyph():
    import re

    src = (Path(__file__).resolve().parents[2] / "seams_to_fur" / "fur_colors.py").read_text()
    names = re.findall(r'\(\s*"([^"]+)",\s*\(', src)
    assert len(names) > 10, "swatch name extraction regex no longer matches fur_colors.py"
    chars = set("".join(names)) | set('0123456789 ()/"')
    missing = {c for c in chars if sf._normalize_char(c) is None}
    assert not missing, f"no glyph for: {sorted(missing)}"


def test_straight_layout_width_and_anchor():
    strokes = sf.layout_straight("12", 100.0, 50.0, 10.0, anchor="middle")
    xs = [x for s in strokes for x, _ in s]
    width = sf.text_width("12", 10.0)
    assert math.isclose(min(xs) + max(xs), 200.0, abs_tol=width * 0.3)
    assert max(xs) - min(xs) <= width + 1e-9


def test_straight_layout_is_y_down_with_text_above_baseline():
    strokes = sf.layout_straight("1", 0.0, 50.0, 10.0)
    ys = [y for s in strokes for _, y in s]
    assert max(ys) <= 50.0 + 1e-9 and min(ys) < 50.0


def test_text_on_right_to_left_path_is_flipped_upright():
    # Same path, both directions - the glyphs must come out identical
    # (upright), not mirrored/upside-down on the reversed one.
    path = [(0.0, 0.0), (100.0, 0.0)]
    a = sf.layout_on_path("69", path, 5.0)
    b = sf.layout_on_path("69", list(reversed(path)), 5.0)
    assert _norm(a) == _norm(b)


def test_underline_adds_strokes_below_text():
    plain = sf.layout_on_path("6", [(0.0, 0.0), (50.0, 0.0)], 5.0)
    lined = sf.layout_on_path("6", [(0.0, 0.0), (50.0, 0.0)], 5.0, underline=True)
    assert len(lined) == len(plain) + 1
    underline_y = lined[-1][0][1]
    assert underline_y > max(y for s in plain for _, y in s)  # y-down: below


def test_unknown_char_draws_a_box_and_accents_are_stripped():
    assert sf._glyph("☃") == sf._MISSING
    assert sf._glyph("é") == sf.GLYPHS["E"]
    assert sf._glyph("a") == sf.GLYPHS["A"]
