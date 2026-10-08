import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import text_fit


def test_fit_font_size_uses_max_when_text_fits():
    size = text_fit.fit_font_size("A", available_width_mm=100.0, max_size_mm=4.0, min_size_mm=1.5)
    assert size == 4.0


def test_fit_font_size_shrinks_for_long_text_on_a_narrow_piece():
    # Chosen so the fitted size lands strictly between the floor and
    # ceiling - not clamped at min_size_mm (see the floor test below for
    # that case) - so this actually exercises the width-driven formula.
    text = "Piece 3"
    size = text_fit.fit_font_size(text, available_width_mm=15.0, max_size_mm=4.0, min_size_mm=1.5)
    assert 1.5 < size < 4.0
    est_width = len(text) * size * text_fit.CHAR_WIDTH_FACTOR
    assert abs(est_width - 15.0) < 1e-6


def test_fit_font_size_never_goes_below_floor():
    size = text_fit.fit_font_size("Something Quite Long Indeed", available_width_mm=1.0, max_size_mm=4.0, min_size_mm=1.5)
    assert size == 1.5


def test_fit_font_size_empty_text_or_zero_width_returns_max():
    assert text_fit.fit_font_size("", 50.0, 4.0, 1.5) == 4.0
    assert text_fit.fit_font_size("hi", 0.0, 4.0, 1.5) == 4.0


def test_format_inches_whole_number():
    assert text_fit.format_inches(1.0) == '1"'
    assert text_fit.format_inches(2.0) == '2"'
    assert text_fit.format_inches(0.0) == '0"'


def test_format_inches_simple_fractions():
    assert text_fit.format_inches(0.5) == '1/2"'
    assert text_fit.format_inches(0.25) == '1/4"'
    assert text_fit.format_inches(0.75) == '3/4"'


def test_format_inches_whole_plus_fraction():
    assert text_fit.format_inches(1.5) == '1 1/2"'
    assert text_fit.format_inches(1.25) == '1 1/4"'


def test_format_inches_rounds_to_nearest_sixteenth():
    # 1.02" isn't a clean sewing fraction - should round to the nearest 1/16".
    assert text_fit.format_inches(1.02) == '1"'
    assert text_fit.format_inches(0.98) == '1"'


def test_format_inches_carries_into_next_whole_number():
    # 16/16 must roll over to the next whole inch, not print "1 16/16"".
    assert text_fit.format_inches(1.9999) == '2"'
