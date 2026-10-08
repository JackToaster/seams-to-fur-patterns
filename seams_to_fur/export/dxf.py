"""DXF export of flattened pattern pieces at real-world millimetre scale.

A minimal, hand-written ASCII DXF R12 writer (HEADER/TABLES/ENTITIES
sections, POLYLINE/VERTEX/SEQEND + LINE + TEXT entities) - not a wrapper
around an external DXF library (matches this project's existing preference
for small, dependency-free writers over pulling in a library; see
geometry/polygon_offset.py's docstring on why Clipper2/pyclipr was skipped
for the same reason). R12 is the most broadly-compatible DXF revision for
cutting/CNC/garment software to read, and doesn't need LWPOLYLINE (an R14+
entity) for anything this export needs.

bpy-light by design, same as export/svg.py: takes plain piece dicts (via
operators/export_common.py's gather_piece_export_data) so this is directly
testable without a running Blender session.

Layers follow the AAMA/ASTM garment-industry DXF convention, organized by
*purpose* (not per piece, matching real garment CAD practice):
  SEWLINE   - the reference (zero-offset) flattened boundary
  CUTLINE   - the offset boundary (seam-allowance cut line)
  GRAINLINE - the filled single-headed block-arrow grain-direction marks
  NOTCHES   - piece-to-piece alignment slits, cut from the sew line out
              to the seam-allowance cut line (see export/notch)
  LABELS    - piece number / fur length / fabric swatch text
"""

from . import notch as notch_geom

# (layer name, AutoCAD Color Index) - entities reference these by name and
# use color 256 (BYLAYER) so the layer's own color controls display.
_LAYERS = (
    ("SEWLINE", 7),
    ("CUTLINE", 1),
    ("GRAINLINE", 5),
    ("NOTCHES", 3),
    ("LABELS", 7),
)


def _group(code, value):
    return f"{code}\n{value}\n"


def _header():
    return (
        "0\nSECTION\n2\nHEADER\n"
        + _group(9, "$ACADVER")
        + _group(1, "AC1009")
        + _group(9, "$INSUNITS")
        + _group(70, 4)  # 4 = millimeters
        + "0\nENDSEC\n"
    )


def _tables():
    out = ["0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n" + _group(70, len(_LAYERS))]
    for name, color in _LAYERS:
        out.append(
            "0\nLAYER\n"
            + _group(2, name)
            + _group(70, 0)
            + _group(62, color)
            + _group(6, "CONTINUOUS")
        )
    out.append("0\nENDTAB\n0\nENDSEC\n")
    return "".join(out)


def _polyline_entity(layer, points_mm, closed=False):
    if len(points_mm) < 2:
        return ""
    out = [
        "0\nPOLYLINE\n"
        + _group(8, layer)
        + _group(62, 256)
        + _group(66, 1)
        + _group(70, 1 if closed else 0)
    ]
    for x, y in points_mm:
        out.append("0\nVERTEX\n" + _group(8, layer) + _group(10, f"{x:.4f}") + _group(20, f"{y:.4f}"))
    out.append("0\nSEQEND\n")
    return "".join(out)


def _line_entity(layer, x1, y1, x2, y2):
    return (
        "0\nLINE\n"
        + _group(8, layer)
        + _group(62, 256)
        + _group(10, f"{x1:.4f}")
        + _group(20, f"{y1:.4f}")
        + _group(11, f"{x2:.4f}")
        + _group(21, f"{y2:.4f}")
    )


def _text_entity(layer, x, y, height, text):
    safe_text = text.replace("\n", " ")
    return (
        "0\nTEXT\n"
        + _group(8, layer)
        + _group(62, 256)
        + _group(10, f"{x:.4f}")
        + _group(20, f"{y:.4f}")
        + _group(40, f"{height:.4f}")
        + _group(1, safe_text)
    )


def build_dxf(pieces):
    """pieces: same piece-dict shape build_svg (export/svg.py) takes - see
    its docstring. Returns a DXF document string."""
    entities = []
    for p in pieces:
        entities.append(_polyline_entity("SEWLINE", p["boundary_mm"], closed=True))

        cut_line_mm = p.get("cut_line_mm")
        if cut_line_mm:
            entities.append(_polyline_entity("CUTLINE", cut_line_mm, closed=True))

        grainline_arrow_mm = p.get("grainline_arrow_mm")
        if grainline_arrow_mm:
            entities.append(_polyline_entity("GRAINLINE", grainline_arrow_mm, closed=True))

        for notch in p.get("notches_mm", []):
            (x1, y1), (x2, y2) = notch_geom.notch_slit(notch["point_mm"], notch["perp"], p["boundary_mm"], cut_line_mm)
            entities.append(_line_entity("NOTCHES", x1, y1, x2, y2))

        lx, ly = p["label_pos_mm"]
        label = p.get("label_text") or p["name"]
        entities.append(_text_entity("LABELS", lx, ly, p.get("label_font_size_mm", 4.0), label))

    return (
        _header()
        + _tables()
        + "0\nSECTION\n2\nENTITIES\n"
        + "".join(entities)
        + "0\nENDSEC\n0\nEOF\n"
    )
