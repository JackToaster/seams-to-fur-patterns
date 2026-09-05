"""SVG export of flattened pattern pieces at real-world millimetre scale.

bpy-light by design: takes plain piece dicts (already extracted from Blender
objects by the calling operator) so the geometry/scaling logic here is
testable without a running Blender session, aside from the unit-scale
lookup which callers resolve via bpy.context.scene.unit_settings first.
"""


def _polyline_points_attr(points_mm):
    return " ".join(f"{x:.4f},{y:.4f}" for x, y in points_mm)


def build_svg(pieces, margin_mm=10.0):
    """pieces: list of dicts with keys:
        name: str
        boundary_mm: list of (x, y) in millimetres (reference flattened boundary)
        cut_line_mm: list of (x, y) in millimetres, or None if no offset set
        label_pos_mm: (x, y) for the piece name text

    Returns an SVG document string sized to fit all pieces plus margin_mm on
    each side, with y flipped (SVG y-down) relative to the mm coordinates.
    """
    all_points = []
    for p in pieces:
        all_points.extend(p["boundary_mm"])
        if p["cut_line_mm"]:
            all_points.extend(p["cut_line_mm"])

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

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_mm:.4f}mm" height="{height_mm:.4f}mm" '
        f'viewBox="0 0 {width_mm:.4f} {height_mm:.4f}">'
    )

    for p in pieces:
        ref_pts = [to_svg(pt) for pt in p["boundary_mm"]]
        lines.append(
            f'<polygon points="{_polyline_points_attr(ref_pts)}" '
            f'fill="none" stroke="#999999" stroke-width="0.2" />'
        )

        if p["cut_line_mm"]:
            cut_pts = [to_svg(pt) for pt in p["cut_line_mm"]]
            lines.append(
                f'<polygon points="{_polyline_points_attr(cut_pts)}" '
                f'fill="none" stroke="#000000" stroke-width="0.3" />'
            )

        lx, ly = to_svg(p["label_pos_mm"])
        lines.append(
            f'<text x="{lx:.4f}" y="{ly:.4f}" font-size="4" '
            f'text-anchor="middle" fill="#000000">{p["name"]}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)
