"""A small single-stroke vector font for pattern annotations.

Laser software (LightBurn in particular) doesn't import SVG <text> reliably
- it substitutes whatever font it has, and doesn't support <textPath> at
all - so every piece of text in the export is drawn as plain stroked
polylines instead. Single-stroke (not outlined) glyphs are also what a
laser actually wants for marking: each stroke engraves as one line, not a
filled outline traced twice.

Glyphs live on a 4-wide x 6-tall grid (baseline at y=0, cap height 6, y
up), with a 5-unit advance. Lowercase is drawn as uppercase, accented
letters lose their accents (via unicodedata), and anything not in the table
draws as a small box rather than silently vanishing.

Upright-ness: text laid out along a seam edge is always flipped to read
left-to-right (never upside down), and seam-partner numbers are underlined
- 6/9, 66/99, 16/91 and friends are otherwise indistinguishable once a
piece has been rotated by hand on the cutting bed.

bpy-free, directly testable with plain pytest.
"""

import math
import unicodedata

GRID_CAP = 6.0
GRID_ADVANCE = 5.0
# Cap height as a fraction of the nominal font size, matching how an SVG
# font-size (em) relates to the visible height of capitals in a typical
# font - keeps sizes chosen against text_fit.CHAR_WIDTH_FACTOR sensible.
CAP_HEIGHT_PER_FONT_SIZE = 0.7

_O = [(1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0)]
_P = [(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)]

GLYPHS = {
    " ": [],
    # Zero is slashed so it can't be read as the letter O.
    "0": [_O, [(0.6, 1), (3.4, 5)]],
    "1": [[(1, 5), (2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "2": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)]],
    "3": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1.5, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(4, 6), (0, 6), (0, 3.5), (3, 3.5), (4, 2.5), (4, 1), (3, 0), (1, 0), (0, 1)]],
    # 6 has a curved hook on top, 9 a straight stem - not each other's
    # 180-degree rotation, which most fonts' 6/9 are.
    "6": [[(3.5, 6), (1.5, 6), (0, 4.5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3), (1, 3), (0, 2)]],
    "7": [[(0, 6), (4, 6), (1.5, 0)]],
    "8": [[(1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3)]],
    "9": [[(4, 4), (3, 3), (1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 0)]],
    "A": [[(0, 0), (0, 4), (2, 6), (4, 4), (4, 0)], [(0, 2.5), (4, 2.5)]],
    "B": [[(0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)], [(3, 3), (4, 2), (4, 1), (3, 0), (0, 0)]],
    "C": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)]],
    "D": [[(0, 0), (0, 6), (2.5, 6), (4, 4.5), (4, 1.5), (2.5, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3), (3, 3)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3), (3, 3)]],
    "G": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3), (4, 3)]],
    "I": [[(1, 6), (3, 6)], [(2, 6), (2, 0)], [(1, 0), (3, 0)]],
    "J": [[(4, 6), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "K": [[(0, 0), (0, 6)], [(4, 6), (0, 2)], [(1.3, 3.3), (4, 0)]],
    "L": [[(0, 6), (0, 0), (4, 0)]],
    "M": [[(0, 0), (0, 6), (2, 3), (4, 6), (4, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [_O],
    "P": [_P],
    "Q": [_O, [(2.5, 1.5), (4, 0)]],
    "R": [_P, [(2, 3), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1, 0), (2, 3), (3, 0), (4, 6)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    '"': [[(1.2, 6), (1.2, 4.5)], [(2.8, 6), (2.8, 4.5)]],
    "'": [[(2, 6), (2, 4.5)]],
    "/": [[(0, 0), (4, 6)]],
    "(": [[(2.5, 6.5), (1.5, 5), (1.5, 1), (2.5, -0.5)]],
    ")": [[(1.5, 6.5), (2.5, 5), (2.5, 1), (1.5, -0.5)]],
    "-": [[(1, 3), (3, 3)]],
    "+": [[(2, 1), (2, 5)], [(0, 3), (4, 3)]],
    ".": [[(1.8, 0), (2.2, 0), (2.2, 0.4), (1.8, 0.4), (1.8, 0)]],
    ",": [[(2.2, 0.4), (1.6, -1)]],
    ":": [[(2, 1), (2, 1.4)], [(2, 4), (2, 4.4)]],
    "&": [[(4, 0), (1, 4), (1, 5), (2, 6), (3, 5), (3, 4), (0, 2), (0, 1), (1, 0), (2, 0), (4, 2)]],
    "#": [[(1, 0), (1.5, 6)], [(2.5, 0), (3, 6)], [(0, 2), (4, 2)], [(0, 4), (4, 4)]],
}
_MISSING = [[(0.5, 0), (3.5, 0), (3.5, 6), (0.5, 6), (0.5, 0)]]
_UNDERLINE_Y = -1.2


def _normalize_char(ch):
    if ch in GLYPHS:
        return ch
    up = ch.upper()
    if up in GLYPHS:
        return up
    stripped = unicodedata.normalize("NFKD", up).encode("ascii", "ignore").decode("ascii")
    if stripped and stripped[0] in GLYPHS:
        return stripped[0]
    return None


def _glyph(ch):
    key = _normalize_char(ch)
    return GLYPHS[key] if key is not None else _MISSING


def cap_height(font_size):
    return font_size * CAP_HEIGHT_PER_FONT_SIZE


def text_width(text, font_size):
    """Width (same units as font_size) the text occupies when laid out."""
    if not text:
        return 0.0
    scale = cap_height(font_size) / GRID_CAP
    # The last glyph doesn't need its trailing inter-character gap.
    return (len(text) * GRID_ADVANCE - (GRID_ADVANCE - 4.0)) * scale


def _place(local_strokes, origin, angle_rad, scale):
    """Glyph-grid strokes (y up) -> SVG-space polylines (y down), scaled,
    rotated by angle_rad (SVG convention: clockwise-positive on screen,
    since y points down) about origin."""
    ox, oy = origin
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    out = []
    for stroke in local_strokes:
        pts = []
        for gx, gy in stroke:
            x, y = gx * scale, -gy * scale
            pts.append((ox + x * c - y * s, oy + x * s + y * c))
        out.append(pts)
    return out


def layout_straight(
    text, x, y, font_size, angle_deg=0.0, anchor="start", underline=False, bold_prefix=0, vcenter=False
):
    """Polylines (SVG coordinates, y down) for text on a straight baseline
    through (x, y), rotated angle_deg (SVG convention). anchor is "start"
    or "middle" (centers the text on x along the baseline); vcenter puts
    (x, y) at mid-cap-height instead of on the baseline. The first
    bold_prefix characters get a second, slightly offset pass so they read
    as bold."""
    scale = cap_height(font_size) / GRID_CAP
    width = text_width(text, font_size)
    start = -width / 2.0 if anchor == "middle" else 0.0
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    if vcenter:
        half = cap_height(font_size) / 2.0
        # The text's own "down" direction, rotated with it.
        x, y = x - s * half, y + c * half
    strokes = []
    for i, ch in enumerate(text):
        gx0 = start + i * GRID_ADVANCE * scale
        origin = (x + gx0 * c, y + gx0 * s)
        glyph = _glyph(ch)
        strokes += _place(glyph, origin, angle, scale)
        if i < bold_prefix:
            nudge = 0.35 * scale
            strokes += _place(glyph, (origin[0] + nudge * c, origin[1] + nudge * s), angle, scale)
    if underline and text:
        strokes += _place([[(0, _UNDERLINE_Y), (width / scale, _UNDERLINE_Y)]], (x + start * c, y + start * s), angle, scale)
    return strokes


def _polyline_length(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))


def _point_and_angle_at(pts, target):
    acc = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-12:
            continue
        if acc + seg >= target or i == len(pts) - 1:
            t = max(0.0, min(1.0, (target - acc) / seg))
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), math.atan2(y1 - y0, x1 - x0)
        acc += seg
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    return pts[-1], math.atan2(y1 - y0, x1 - x0)


def layout_on_path(text, path_pts, font_size, underline=False):
    """Polylines (SVG coordinates) for text following path_pts (SVG
    coordinates), centered along it and vertically centered on it, each
    glyph rotated to the local tangent. The path is reversed first if it
    runs right-to-left overall, so the text always reads upright rather
    than upside down. Underlining (if requested) follows the curve too,
    one short segment per character."""
    if len(path_pts) < 2 or not text:
        return []
    if path_pts[-1][0] < path_pts[0][0]:
        path_pts = list(reversed(path_pts))
    scale = cap_height(font_size) / GRID_CAP
    advance = GRID_ADVANCE * scale
    width = text_width(text, font_size)
    start = (_polyline_length(path_pts) - width) / 2.0
    strokes = []
    for i, ch in enumerate(text):
        center_s = start + i * advance + 2.0 * scale
        (cx, cy), angle = _point_and_angle_at(path_pts, max(0.0, center_s))
        glyph = [list(stroke) for stroke in _glyph(ch)]
        if underline:
            # Extend each char's underline across its whole advance (but
            # not past the last character) so the pieces join up.
            right = GRID_ADVANCE if i < len(text) - 1 else 4.0
            glyph.append([(0, _UNDERLINE_Y), (right, _UNDERLINE_Y)])
        # Glyph center (x=2, mid-cap) sits on the path point.
        shifted = [[(gx - 2.0, gy - GRID_CAP / 2.0) for gx, gy in stroke] for stroke in glyph]
        strokes += _place(shifted, (cx, cy), angle, scale)
    return strokes


def strokes_to_path_d(strokes):
    parts = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        parts.append("M " + " L ".join(f"{x:.3f},{y:.3f}" for x, y in stroke))
    return " ".join(parts)
