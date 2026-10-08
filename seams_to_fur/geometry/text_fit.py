"""Plain-text helpers for fabrication-output export annotations - font-size
fitting and real-world unit formatting. bpy-free (no mesh/scene data
involved at all, just strings and numbers) so it's directly testable with
plain pytest, matching this project's other geometry modules (see
polygon_offset.py's docstring on why bpy-free modules exist as a category
here).
"""

import math

# Rough average glyph width as a fraction of font-size for a generic
# sans-serif font, at the precision an SVG/DXF viewer's default font
# renders text - not exact per-font metrics (nothing here has a font
# rasterizer available), but close enough to reliably prevent the specific
# failure mode this exists for: a label wider than the piece it's printed on.
CHAR_WIDTH_FACTOR = 0.62


def fit_font_size(text, available_width_mm, max_size_mm, min_size_mm):
    """The largest font size in [min_size_mm, max_size_mm] that keeps
    text's estimated rendered width within available_width_mm. Never
    returns below min_size_mm even if that still overflows (a legible-
    but-slightly-wide label beats an illegibly tiny one on a very small
    piece)."""
    if not text or available_width_mm <= 0:
        return max_size_mm
    est_width_at_max = len(text) * max_size_mm * CHAR_WIDTH_FACTOR
    if est_width_at_max <= available_width_mm:
        return max_size_mm
    fitted = available_width_mm / (len(text) * CHAR_WIDTH_FACTOR)
    return max(min_size_mm, min(max_size_mm, fitted))


def format_inches(value_inches):
    """A real-world length in inches as a garment-industry-style fraction
    string (e.g. 1", 1 1/2", 3/4") rounded to the nearest 1/16" - not a
    decimal (e.g. "1.5in"), which isn't how a sewer/pattern-cutter reads
    or marks a ruler."""
    whole = int(value_inches)
    remainder = value_inches - whole
    sixteenths = round(remainder * 16)
    if sixteenths >= 16:
        whole += 1
        sixteenths = 0
    if sixteenths == 0:
        return f'{whole}"' if whole > 0 else '0"'

    divisor = math.gcd(sixteenths, 16)
    num, den = sixteenths // divisor, 16 // divisor
    if whole > 0:
        return f'{whole} {num}/{den}"'
    return f'{num}/{den}"'
