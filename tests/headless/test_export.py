"""Headless regression tests for the fabrication-output export pipeline
(export_common.gather_piece_export_data, export/svg.py, export/dxf.py), run
via:

    blender --background --factory-startup --python tests/headless/test_export.py

Separate file from test_appearance.py/test_flatten_pipeline.py so this
larger new feature area doesn't bloat either of those - same small
assert-based runner pattern (no pytest inside Blender's bundled
interpreter).
"""

import sys
import traceback

import bpy

ADDON_MODULE = "bl_ext.user_default.seams_to_fur"

_results = []


def test(name):
    def decorator(fn):
        _results.append((name, fn))
        return fn

    return decorator


def _clean_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def _flat_plane_with_piece():
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    return mesh_obj


def _cube_with_seam_loop():
    """A unit cube with a closed seam curve looping just inside its top
    face - two pieces sharing one long (~3.2m) closed-loop seam, well over
    both the 3cm skip and 15cm midpoint-notch thresholds."""
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    shrunk = [centroid + (p - centroid) * 0.8 for p in world_pts]

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    seam_curve_ops.create_seam_curve_object(bpy.context, mesh_obj, shrunk, closed=True)
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    assert len(mesh_obj.seams_to_fur_pieces) == 2
    return mesh_obj


@test("gather_piece_export_data: grainline_mm is None without a grain direction, a real span with one")
def test_grainline_presence_and_span():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    assert len(pieces) == 1
    assert pieces[0]["grainline_mm"] is None, "no grain direction set yet - expected no grainline"

    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.grain_direction = (1.0, 0.0, 0.0)
    piece.grain_anchor = tuple(piece.sample_point)
    piece.has_grain_direction = True

    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    grainline = pieces[0]["grainline_mm"]
    assert grainline is not None, "expected a grainline once has_grain_direction is set"
    start, end = grainline
    span = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    assert span > 100.0, f"expected a real, long (mm-scale) span for a 2x2m plane, got {span:.2f}mm"


@test("gather_piece_export_data: swatch_name matches the nearest fur_colors.py entry")
def test_swatch_name_nearest_match():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    from bl_ext.user_default.seams_to_fur.operators import export_common
    from bl_ext.user_default.seams_to_fur.fur_colors import FUR_COLOR_SWATCHES

    exact_name, exact_rgb = FUR_COLOR_SWATCHES[10]
    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.color = (exact_rgb[0], exact_rgb[1], exact_rgb[2], 1.0)

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    assert pieces[0]["swatch_name"] == exact_name


@test("gather_piece_export_data: notches skip a run < 3cm, inset + midpoint on a long shared seam")
def test_notches_from_seam_partners():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    assert len(pieces) == 2

    for p in pieces:
        notches = p["notches_mm"]
        # One shared closed-loop run per piece, well over both thresholds:
        # 2 inset notches + 1 midpoint notch.
        assert len(notches) == 3, f"{p['name']}: expected 3 notches (2 inset + 1 midpoint), got {len(notches)}"
        for n in notches:
            perp = n["perp"]
            perp_len = (perp[0] ** 2 + perp[1] ** 2) ** 0.5
            assert abs(perp_len - 1.0) < 1e-3, f"expected a unit perpendicular direction, got length {perp_len}"


@test("gather_piece_export_data: seam_labels names the other piece sewn to that edge")
def test_seam_labels_reference_partner_piece_name():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    assert len(pieces) == 2
    # seam_labels reference the *other* piece's display name (the "Piece "
    # prefix is stripped for the label text, e.g. "Piece 2" -> "2").
    display_names = {
        p["name"][len("Piece "):] if p["name"].startswith("Piece ") else p["name"]
        for p in pieces
    }

    for p in pieces:
        own_display = p["name"][len("Piece "):] if p["name"].startswith("Piece ") else p["name"]
        other_names = display_names - {own_display}
        assert len(p["seam_labels"]) >= 1, f"{p['name']}: expected a seam label for the long shared seam"
        for label in p["seam_labels"]:
            assert label["text"] in other_names, f"expected the *other* piece's name, got {label['text']!r}"
            if label.get("mode") == "straight":
                assert "pos_mm" in label and "dir_mm" in label
            else:
                assert len(label["path_mm"]) >= 2


@test("gather_piece_export_data: seam_allowance_override_mm fills in a cut line only for a piece with no allowance set")
def test_seam_allowance_override_fills_in_default():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    piece = mesh_obj.seams_to_fur_pieces[0]
    assert piece.offset_mm == 0.0
    assert not piece.cut_line_object

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces_no_override = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)
    assert pieces_no_override[0]["cut_line_mm"] is None, "no allowance set and no override - expected no cut line"

    pieces_override = export_common.gather_piece_export_data(
        bpy.context, mesh_obj, mm_scale, seam_allowance_override_mm=8.0
    )
    assert pieces_override[0]["offset_mm"] == 8.0
    assert pieces_override[0]["cut_line_mm"] is not None
    assert len(pieces_override[0]["cut_line_mm"]) == len(pieces_override[0]["boundary_mm"])


@test("gather_piece_export_data: seam_allowance_override_mm never touches a piece with its own nonzero allowance")
def test_seam_allowance_override_respects_existing_offset():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.offset_mm = 5.0
    assert bpy.ops.seams_to_fur.flatten_piece() == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]
    assert piece.cut_line_object, "expected a baked cut line for the piece's own 5mm allowance"

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(
        bpy.context, mesh_obj, mm_scale, seam_allowance_override_mm=8.0
    )
    assert len(pieces) == 1
    assert pieces[0]["offset_mm"] == 5.0, "a piece with its own nonzero allowance must not be overridden"
    assert pieces[0]["cut_line_mm"] is not None


@test("gather_piece_export_data: export layout is regenerated, not reused from flatten_all's own viewport placement")
def test_export_layout_regenerated_not_reused():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    from bl_ext.user_default.seams_to_fur.operators import export_common

    # Simulate flatten_all's own shelf layout (or a user's manual drag)
    # having put a piece somewhere far away in the viewport/scene.
    piece = mesh_obj.seams_to_fur_pieces[0]
    flat_obj = bpy.data.objects[piece.flattened_object]
    flat_obj.location.x += 500.0

    mm_scale = export_common.blender_units_to_mm(bpy.context)
    pieces = export_common.gather_piece_export_data(bpy.context, mesh_obj, mm_scale)

    far_away_mm = 500.0 * mm_scale * 0.5
    for p in pieces:
        xs = [pt[0] for pt in p["boundary_mm"]]
        assert max(xs) < far_away_mm, (
            f"{p['name']}: export layout still reflects the manual viewport drag, not a fresh regeneration"
        )


@test("build_svg: renders the block-arrow grainline, notches, fitted labels, and seam labels when present")
def test_svg_renders_new_elements():
    from bl_ext.user_default.seams_to_fur.export import svg as svg_export

    pieces = [
        {
            "name": "Piece 1",
            "boundary_mm": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "cut_line_mm": [(-10, -10), (110, -10), (110, 110), (-10, 110)],
            "label_pos_mm": (50, 50),
            "label_text": '1 (1/4" Electric Blue)',
            "label_number": "1",
            "label_suffix": ' (1/4" Electric Blue)',
            "label_font_size_mm": 4.0,
            "offset_mm": 10.0,
            "swatch_name": "Electric Blue",
            "fur_type": '1/4" Electric Blue',
            "grainline_mm": ((20, 50), (80, 50)),
            "grainline_arrow_mm": [
                (46, 20), (54, 20), (54, 70), (60, 70), (50, 80), (40, 70), (46, 70),
            ],
            "notches_mm": [{"point_mm": (0, 50), "perp": (1.0, 0.0)}],
            "seam_labels": [{"mode": "curved", "path_mm": [(5, 5), (5, 95)], "text": "2"}],
        }
    ]
    svg_doc = svg_export.build_svg(pieces)
    assert "<svg" in svg_doc and "</svg>" in svg_doc
    # The block-arrow grainline is a stroke-only <polygon> (not a <line> +
    # separate arrowhead <polygon>s) - only the notch slit is a <line>.
    assert svg_doc.count("<line") == 1, "expected only the notch slit line"
    # LightBurn-oriented structure: one <g> per piece holding every one of
    # its elements (so it drags as a unit), layers by exact palette stroke
    # color, and no <text>/<textPath> (LightBurn can't import them) - all
    # text is single-stroke vector paths.
    assert '<g id="piece-0" data-name="Piece 1">' in svg_doc
    assert '<g id="group-0" data-fur-type="1/4&quot; Electric Blue">' in svg_doc
    assert "<rect" in svg_doc, "expected a fur-type group box"
    assert "<text" not in svg_doc and "inkscape:" not in svg_doc
    piece_start = svg_doc.index('<g id="piece-0"')
    piece_body = svg_doc[piece_start:svg_doc.index("</g>", piece_start)]
    for color in svg_export.LAYER_COLORS["cut"], svg_export.LAYER_COLORS["sew"], svg_export.LAYER_COLORS["mark"]:
        assert f'stroke="{color}"' in piece_body, f"piece group is missing its {color} layer elements"
    assert svg_doc.count("<g ") == svg_doc.count("</g>") == 3  # group-box wrapper + 1 box + 1 piece


@test("build_dxf: well-formed (balanced sections, ENTITIES/ENDSEC/EOF present)")
def test_dxf_well_formed():
    from bl_ext.user_default.seams_to_fur.export import dxf as dxf_export

    pieces = [
        {
            "name": "Piece 1",
            "boundary_mm": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "cut_line_mm": [(-10, -10), (110, -10), (110, 110), (-10, 110)],
            "label_pos_mm": (50, 50),
            "offset_mm": 10.0,
            "swatch_name": "Electric Blue",
            "grainline_mm": ((20, 50), (80, 50)),
            "notches_mm": [{"point_mm": (0, 50), "perp": (1.0, 0.0)}],
        }
    ]
    doc = dxf_export.build_dxf(pieces)
    assert doc.count("0\nSECTION\n") == doc.count("0\nENDSEC\n"), "unbalanced SECTION/ENDSEC"
    assert "2\nENTITIES\n" in doc
    assert doc.rstrip().endswith("0\nEOF")


@test("build_dxf: all five garment-convention layers are defined")
def test_dxf_layer_names():
    from bl_ext.user_default.seams_to_fur.export import dxf as dxf_export

    pieces = [
        {
            "name": "Piece 1",
            "boundary_mm": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "cut_line_mm": None,
            "label_pos_mm": (50, 50),
            "offset_mm": 0.0,
            "swatch_name": None,
            "grainline_mm": None,
            "notches_mm": [],
        }
    ]
    doc = dxf_export.build_dxf(pieces)
    for layer in ("SEWLINE", "CUTLINE", "GRAINLINE", "NOTCHES", "LABELS"):
        assert f"2\n{layer}\n" in doc, f"expected a LAYER table entry for {layer}"


@test("build_dxf: the sew-line polyline's vertex count matches the piece's boundary point count")
def test_dxf_polyline_vertex_count_roundtrip():
    from bl_ext.user_default.seams_to_fur.export import dxf as dxf_export

    boundary = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 50)]
    pieces = [
        {
            "name": "Piece 1",
            "boundary_mm": boundary,
            "cut_line_mm": None,
            "label_pos_mm": (50, 50),
            "offset_mm": 0.0,
            "swatch_name": None,
            "grainline_mm": None,
            "notches_mm": [],
        }
    ]
    doc = dxf_export.build_dxf(pieces)
    # Everything from the SEWLINE's own POLYLINE up to its SEQEND.
    start = doc.index("0\nPOLYLINE\n")
    end = doc.index("0\nSEQEND\n", start)
    segment = doc[start:end]
    assert segment.count("0\nVERTEX\n") == len(boundary)


def run_all():
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)

    failures = []
    for name, fn in _results:
        try:
            fn()
            print(f"PASS: {name}")
        except Exception:
            failures.append(name)
            print(f"FAIL: {name}")
            traceback.print_exc()

    total = len(_results)
    passed = total - len(failures)
    print(f"\n{passed}/{total} tests passed")
    if failures:
        print("Failed: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    run_all()
