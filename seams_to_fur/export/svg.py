"""SVG export of flattened pattern pieces at real-world millimetre scale.

bpy-light by design: takes plain piece dicts (already extracted from Blender
objects by the calling operator, via operators/export_common.py's
gather_piece_export_data) so the geometry/scaling logic here is testable
without a running Blender session, aside from the unit-scale lookup which
callers resolve via bpy.context.scene.unit_settings first.

Structured for laser software - LightBurn specifically - rather than for a
general-purpose SVG editor:

- LightBurn ignores SVG/Inkscape layers entirely and assigns a cut layer
  per *exact stroke color*, matching colors against its own fixed palette
  first. So each kind of line uses one of LightBurn's palette colors
  (LAYER_COLORS below), and lands on a predictable layer with predictable
  settings on every import.
- LightBurn does import <g> elements as groups. So each piece's elements -
  sew line, cut line, notches, arrow, and text, across all those layers -
  share a single <g>, and the whole piece drags as one unit while laying
  pieces out on the actual material by hand.
- LightBurn doesn't import <text>/<textPath> faithfully (it substitutes
  fonts, and has no text-on-path at all), so all text is emitted as
  single-stroke vector paths - see export/stroke_font.
"""

import math

from . import notch as notch_geom
from . import stroke_font

# Matches operators/export_common.py's SEAM_LABEL_FONT_MM - a purely
# presentation-layer constant (this module never imports bpy-side code),
# so kept independently rather than a shared import.
SEAM_LABEL_FONT_MM = 2.5

# Exact LightBurn palette colors (layers 00, 01, 02, 16) - an exact
# match is what makes each kind of line land on the same LightBurn layer
# every time, instead of being assigned whatever palette slot is free.
LAYER_COLORS = {
    "cut": "#000000",  # 00 - seam-allowance cut line + notch slits
    "sew": "#0000FF",  # 01 - sew line (mark/score only)
    "mark": "#FF0000",  # 02 - grain arrow + all text
    "guide": "#808080",  # 16 - fur-type group boxes (layout aid; disable output)
}

# A dashed box drawn around each fur_type row group (see
# geometry.export_layout.regenerate_export_layout, which is what actually
# lays pieces out into these groups/rows in the first place - this module
# just draws a frame around whatever it finds already grouped together),
# labeled with the fur type so a group can be spotted at a glance. Padding
# on the sides/bottom is just breathing room; the top gets extra room to
# fit the label line above the pieces without touching them.
GROUP_BOX_PADDING_MM = 8.0
GROUP_LABEL_FONT_MM = 6.0
GROUP_BOX_TOP_PADDING_MM = GROUP_BOX_PADDING_MM + GROUP_LABEL_FONT_MM * 1.6


def _polyline_points_attr(points):
    return " ".join(f"{x:.4f},{y:.4f}" for x, y in points)


def _xml_attr_escape(text):
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _text_path(strokes, color):
    d = stroke_font.strokes_to_path_d(strokes)
    if not d:
        return ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="0.2" />'


def build_svg(pieces, margin_mm=10.0):
    """pieces: list of dicts - see operators/export_common.py's
    gather_piece_export_data docstring for the full field list (name,
    boundary_mm, cut_line_mm, label_pos_mm/label_text/label_number/
    label_suffix/label_font_size_mm, offset_mm, fur_type,
    grainline_mm/grainline_arrow_mm, notches_mm, seam_labels - all but
    name/boundary_mm are optional here, each guarded with .get()).

    Returns an SVG document string sized to fit all pieces plus margin_mm on
    each side, with y flipped (SVG y-down) relative to the mm coordinates.

    Each piece is one <g id="piece-N" data-name="..."> containing, by
    LightBurn layer (stroke color - see LAYER_COLORS): its sew line (or,
    with no seam allowance, its outline drawn as the cut line instead,
    since then it *is* what gets cut), its seam-allowance cut line and
    notch slits (cut from the sew line out to the cut line - see
    export/notch), and its grain arrow, title, and seam-partner labels
    (mark). Seam-partner labels are always laid out reading upright and
    underlined, so 6/9-style numbers stay unambiguous however the piece
    is later rotated on the cutting bed. Fur-type group boxes come first,
    in their own <g>, on the guide color.
    """
    all_points = []
    for p in pieces:
        all_points.extend(p["boundary_mm"])
        if p.get("cut_line_mm"):
            all_points.extend(p["cut_line_mm"])
        if p.get("grainline_mm"):
            all_points.extend(p["grainline_mm"])
        for notch in p.get("notches_mm", []):
            all_points.append(notch["point_mm"])

    # Group boxes computed in raw mm-space (before to_svg exists - it
    # needs the final, box-inclusive bounds below) - one padded bbox per
    # distinct fur_type, from the union of its pieces' boundary_mm and
    # cut_line_mm, keyed by first-seen order.
    group_mm_points = {}
    group_order = []
    for p in pieces:
        key = p.get("fur_type") or ""
        if not key:
            continue
        if key not in group_mm_points:
            group_mm_points[key] = []
            group_order.append(key)
        group_mm_points[key].extend(p["boundary_mm"])
        if p.get("cut_line_mm"):
            group_mm_points[key].extend(p["cut_line_mm"])

    group_boxes_mm = {}
    for key in group_order:
        pts = group_mm_points[key]
        gxs = [pt[0] for pt in pts]
        gys = [pt[1] for pt in pts]
        box = (
            min(gxs) - GROUP_BOX_PADDING_MM,
            min(gys) - GROUP_BOX_PADDING_MM,
            max(gxs) + GROUP_BOX_PADDING_MM,
            max(gys) + GROUP_BOX_TOP_PADDING_MM,
        )
        group_boxes_mm[key] = box
        all_points.append((box[0], box[1]))
        all_points.append((box[2], box[3]))

    if not all_points:
        min_x = min_y = 0.0
        max_x = max_y = 0.0
    else:
        xs = [pt[0] for pt in all_points]
        ys = [pt[1] for pt in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

    width_mm = (max_x - min_x) + 2 * margin_mm
    height_mm = (max_y - min_y) + 2 * margin_mm

    def to_svg(pt):
        x, y = pt
        # Flip Y (SVG is y-down) and shift so the drawing starts at the margin.
        return (x - min_x + margin_mm, (max_y - y) + margin_mm)

    def to_svg_dir(d):
        # A direction flips its Y component the same way a point does, but
        # has no translation to undo.
        dx, dy = d
        return (dx, -dy)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_mm:.4f}mm" height="{height_mm:.4f}mm" '
        f'viewBox="0 0 {width_mm:.4f} {height_mm:.4f}">'
    )

    if group_order:
        lines.append('<g id="fur-type-groups">')
        for group_idx, key in enumerate(group_order):
            box_min_x, box_min_y, box_max_x, box_max_y = group_boxes_mm[key]
            corners = [to_svg((box_min_x, box_min_y)), to_svg((box_max_x, box_max_y))]
            sx1, sx2 = sorted(c[0] for c in corners)
            sy1, sy2 = sorted(c[1] for c in corners)
            label = stroke_font.layout_straight(
                key, sx1 + GROUP_BOX_PADDING_MM, sy1 + GROUP_LABEL_FONT_MM, GROUP_LABEL_FONT_MM
            )
            lines.append(
                f'<g id="group-{group_idx}" data-fur-type="{_xml_attr_escape(key)}">'
                f'<rect x="{sx1:.4f}" y="{sy1:.4f}" width="{sx2 - sx1:.4f}" height="{sy2 - sy1:.4f}" '
                f'fill="none" stroke="{LAYER_COLORS["guide"]}" stroke-width="0.4" stroke-dasharray="3,2" />'
                f'{_text_path(label, LAYER_COLORS["guide"])}</g>'
            )
        lines.append("</g>")

    for piece_idx, p in enumerate(pieces):
        parts = []
        cut_line_mm = p.get("cut_line_mm")
        boundary_mm = p["boundary_mm"]

        outline_color = LAYER_COLORS["sew"] if cut_line_mm else LAYER_COLORS["cut"]
        parts.append(
            f'<polygon points="{_polyline_points_attr([to_svg(pt) for pt in boundary_mm])}" '
            f'fill="none" stroke="{outline_color}" stroke-width="0.2" />'
        )
        if cut_line_mm:
            parts.append(
                f'<polygon points="{_polyline_points_attr([to_svg(pt) for pt in cut_line_mm])}" '
                f'fill="none" stroke="{LAYER_COLORS["cut"]}" stroke-width="0.3" />'
            )

        for notch in p.get("notches_mm", []):
            start, end = notch_geom.notch_slit(notch["point_mm"], notch["perp"], boundary_mm, cut_line_mm)
            (x1, y1), (x2, y2) = to_svg(start), to_svg(end)
            parts.append(
                f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" '
                f'stroke="{LAYER_COLORS["cut"]}" stroke-width="0.3" />'
            )

        grainline_arrow_mm = p.get("grainline_arrow_mm")
        if grainline_arrow_mm:
            parts.append(
                f'<polygon points="{_polyline_points_attr([to_svg(pt) for pt in grainline_arrow_mm])}" '
                f'fill="none" stroke="{LAYER_COLORS["mark"]}" stroke-width="0.4" stroke-linejoin="round" />'
            )

        lx, ly = to_svg(p["label_pos_mm"])
        label_font_size = p.get("label_font_size_mm", 4.0)
        label_number = p.get("label_number")
        if label_number is not None:
            title = f"{label_number}{p.get('label_suffix', '')}"
            bold = len(label_number)
        else:
            title = p.get("label_text") or p["name"]
            bold = 0
        parts.append(
            _text_path(
                stroke_font.layout_straight(title, lx, ly, label_font_size, anchor="middle", bold_prefix=bold),
                LAYER_COLORS["mark"],
            )
        )

        for seam_label in p.get("seam_labels", []):
            if seam_label.get("mode") == "straight":
                pos = to_svg(seam_label["pos_mm"])
                dx, dy = to_svg_dir(seam_label["dir_mm"])
                angle = math.degrees(math.atan2(dy, dx))
                # Keep the text reading left-to-right/upright rather than
                # upside down - a line at angle and angle-180 look
                # identical, only "which way is forward" (and therefore
                # which way text drawn along it reads) differs.
                if angle > 90.0:
                    angle -= 180.0
                elif angle <= -90.0:
                    angle += 180.0
                strokes = stroke_font.layout_straight(
                    seam_label["text"], pos[0], pos[1], SEAM_LABEL_FONT_MM,
                    angle_deg=angle, anchor="middle", underline=True, vcenter=True,
                )
            else:
                path_pts = [to_svg(pt) for pt in seam_label.get("path_mm", [])]
                strokes = stroke_font.layout_on_path(seam_label["text"], path_pts, SEAM_LABEL_FONT_MM, underline=True)
            parts.append(_text_path(strokes, LAYER_COLORS["mark"]))

        lines.append(
            f'<g id="piece-{piece_idx}" data-name="{_xml_attr_escape(p["name"])}">' + "".join(parts) + "</g>"
        )

    lines.append("</svg>")
    return "\n".join(lines)
