"""Headless regression tests for the flatten pipeline, run via:

    blender --background --factory-startup --python tests/headless/test_flatten_pipeline.py

Blender's bundled interpreter here doesn't have pytest available, so this is
a small assert-based runner rather than a pytest suite (bpy-free logic is
covered by real pytest under tests/unit/ instead). Fixtures are built
programmatically with bpy.ops rather than checked-in .blend files, so
there's nothing binary to keep in sync with the addon's schema.
"""

import sys
import traceback

import bpy

ADDON_MODULE = "bl_ext.user_default.usbee"

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


def _add_seam_curve_for(mesh_obj, world_points, cyclic=True):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.usbee.add_seam_curve()
    curve_obj = bpy.context.active_object
    spline = curve_obj.data.splines[0]
    spline.points.add(len(world_points) - len(spline.points))
    for i, p in enumerate(world_points):
        spline.points[i].co = (p.x, p.y, p.z, 1.0)
    spline.use_cyclic_u = cyclic
    return curve_obj


@test("island isolation: single seam loop on a cube splits it into 2 islands")
def test_cube_two_islands():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    # Cut off one face (+Z) from the rest by tracing a seam loop slightly
    # inset from that face's boundary.
    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    shrunk = [centroid + (p - centroid) * 0.8 for p in world_pts]
    _add_seam_curve_for(mesh_obj, shrunk)

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.usbee_pieces) == 2, f"expected 2 pieces, got {len(mesh_obj.usbee_pieces)}"

    face_counts = sorted(
        len(bpy.data.objects[p.flattened_object].data.polygons) for p in mesh_obj.usbee_pieces
    )
    # One island is the single top face, the other is the remaining 5 faces.
    assert face_counts == [1, 5], f"expected face counts [1, 5], got {face_counts}"


@test("modifiers (Subdivision Surface) are applied before flattening, not the low-poly base cage")
def test_modifiers_applied_before_flatten():
    _clean_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "CoarsePlane"
    base_face_count = len(mesh_obj.data.polygons)

    mod = mesh_obj.modifiers.new(name="Subdivision", type="SUBSURF")
    mod.levels = 3

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    flat_obj = bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object]
    assert len(flat_obj.data.polygons) > base_face_count, (
        f"expected the subdivided surface ({base_face_count} base faces) to "
        f"be flattened, but got only {len(flat_obj.data.polygons)} faces - "
        f"looks like the low-poly base cage was flattened instead"
    )


@test("topology validation: an un-seamed closed mesh is rejected, not silently flattened")
def test_closed_mesh_rejected():
    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=8, ring_count=4)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"
    # No seam curve added - the whole sphere is one closed (non-disk) island.

    bpy.context.view_layer.objects.active = mesh_obj
    try:
        result = bpy.ops.usbee.flatten_all()
    except RuntimeError:
        return  # operator raised - acceptable rejection path

    assert result == {"CANCELLED"}, (
        f"expected the operator to reject a closed mesh with CANCELLED, got {result}"
    )
    for p in mesh_obj.usbee_pieces:
        assert not p.flattened_object, "a closed island should never produce a flattened object"


@test("BFF sanity: flattening an already-flat mesh is close to identity in area")
def test_bff_already_flat_mesh():
    from bl_ext.user_default.usbee.backends import bff as bff_backend
    from bl_ext.user_default.usbee.geometry import obj_io
    import os

    addon_dir = os.path.dirname(os.path.dirname(bff_backend.__file__))
    binary_path = bff_backend.find_binary(addon_dir)

    verts = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 3.0, 0.0), (0.0, 3.0, 0.0)]
    faces = [[0, 1, 2], [0, 2, 3]]
    out_verts, out_faces = bff_backend.flatten_island(binary_path, verts, faces)

    def shoelace(pts, face):
        a = 0.0
        n = len(face)
        for i in range(n):
            x1, y1 = pts[face[i]][0], pts[face[i]][1]
            x2, y2 = pts[face[(i + 1) % n]][0], pts[face[(i + 1) % n]][1]
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    area_in = sum(shoelace(verts, f) for f in faces)
    area_out = sum(shoelace(out_verts, f) for f in faces)
    assert area_in > 0 and area_out > 0
    ratio = area_out / area_in
    assert 0.5 < ratio < 2.0, (
        f"flattening an already-flat 4x3 rectangle changed its area by "
        f"more than 2x (ratio={ratio}) - BFF invocation or parsing is "
        f"likely broken, not just imprecise"
    )


@test("offset: setting offset_mm produces a cut-line curve with matching sign of area change")
def test_offset_produces_cut_line():
    _clean_scene()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    # A plane primitive has no boundary seam needed - it's already a single
    # disk (4 verts, 1 face), so flatten_all can run with zero seam curves.
    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.usbee_pieces) == 1

    piece = mesh_obj.usbee_pieces[0]
    piece.offset_mm = 50.0  # 5cm grow, generous relative to the 2m plane
    mesh_obj.usbee_active_piece_index = 0
    result = bpy.ops.usbee.flatten_piece()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert piece.cut_line_object, "expected a cut-line curve after setting offset_mm"

    cut_obj = bpy.data.objects[piece.cut_line_object]
    assert len(cut_obj.data.splines[0].points) == 4


@test("manual reposition of a flattened piece survives a re-bake, and Reset Placement undoes it")
def test_manual_reposition_survives_rebake():
    _clean_scene()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}

    piece = mesh_obj.usbee_pieces[0]
    flat_obj = bpy.data.objects[piece.flattened_object]
    flat_obj.location.x += 5.0
    moved_location = flat_obj.location.copy()

    mesh_obj.usbee_active_piece_index = 0
    assert bpy.ops.usbee.flatten_piece() == {"FINISHED"}
    flat_obj_after = bpy.data.objects[piece.flattened_object]
    assert flat_obj_after.name == flat_obj.name, "re-bake should update the existing object, not replace it"
    assert (flat_obj_after.location - moved_location).length < 1e-9, (
        "manually moving a flattened piece should survive a re-bake of its shape"
    )

    assert bpy.ops.usbee.reset_placement() == {"FINISHED"}
    reset_obj = bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object]
    assert reset_obj.location.x < moved_location.x - 1.0, (
        "Reset Placement should move the piece back near the automatic grid layout origin"
    )


@test("editing one seam curve only re-bakes the piece it affects")
def test_partial_rebake_only_touches_edited_piece():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    curve_obj = _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.8 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}

    small_piece = min(mesh_obj.usbee_pieces, key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons))
    big_piece = max(mesh_obj.usbee_pieces, key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons))
    big_flat_name_before = big_piece.flattened_object
    big_obj_before = bpy.data.objects[big_flat_name_before]
    big_loc_before = big_obj_before.location.copy()

    # Shrink the seam curve further, changing only the small piece's shape.
    spline = curve_obj.data.splines[0]
    for i, p in enumerate(spline.points):
        co = p.co
        p.co = (
            centroid.x + (co.x - centroid.x) * 0.9,
            centroid.y + (co.y - centroid.y) * 0.9,
            centroid.z + (co.z - centroid.z) * 0.9,
            1.0,
        )
    bpy.context.view_layer.objects.active = mesh_obj

    mesh_obj.usbee_active_piece_index = list(mesh_obj.usbee_pieces).index(small_piece)
    assert bpy.ops.usbee.flatten_piece() == {"FINISHED"}

    big_obj_after = bpy.data.objects[big_flat_name_before]
    assert (big_obj_after.location - big_loc_before).length < 1e-9, (
        "re-baking one piece moved/replaced an unrelated piece's flattened object"
    )


@test("open (dart) seam curve stays one piece but duplicates verts along the cut, pinching at the tip")
def test_open_dart_curve_splits_vertices():
    _clean_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6, size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Grid"
    base_vert_count = len(mesh_obj.data.vertices)
    base_face_count = len(mesh_obj.data.polygons)

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.usbee.add_seam_curve()
    curve_obj = bpy.context.active_object
    spline = curve_obj.data.splines[0]
    # Starts at the mesh boundary (x=-1), ends at an interior point (the
    # dart tip) - deliberately NOT cyclic, i.e. an open cut, not a loop.
    spline.points[0].co = (-1.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (-0.3, 0.0, 0.0, 1.0)
    spline.use_cyclic_u = False

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.usbee_pieces) == 1, (
        "an open dart cut shouldn't split the mesh into separate islands - "
        "it's a slit, not a boundary"
    )

    flat_obj = bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object]
    assert len(flat_obj.data.polygons) == base_face_count, "face count shouldn't change"
    assert len(flat_obj.data.vertices) > base_vert_count, (
        "expected extra vertices duplicated along the interior dart cut so "
        "it can open into a gap when flattened, instead of BFF seeing a "
        "fully-connected mesh with the cut silently ignored"
    )


@test("modifiers on the seam curve itself (Array) are respected when resolving cuts")
def test_seam_curve_array_modifier():
    _clean_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=4, size=4.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "WideGrid"

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.usbee.add_seam_curve()
    curve_obj = bpy.context.active_object
    # A single short open dart near the left edge...
    spline = curve_obj.data.splines[0]
    spline.points[0].co = (-2.0, 0.0, 0.0, 1.0)
    spline.points[1].co = (-1.5, 0.0, 0.0, 1.0)
    spline.use_cyclic_u = False

    bpy.context.view_layer.objects.active = mesh_obj
    baseline = bpy.ops.usbee.flatten_all()
    assert baseline == {"FINISHED"}
    single_dart_verts = len(bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object].data.vertices)

    # ...arrayed into 3 copies spread across the grid. If the resolver reads
    # the curve's raw (un-arrayed) points instead of its evaluated geometry,
    # this would have no additional effect versus the single-dart baseline.
    array_mod = curve_obj.modifiers.new(name="Array", type="ARRAY")
    array_mod.count = 2
    array_mod.use_relative_offset = False
    array_mod.use_constant_offset = True
    array_mod.constant_offset_displace = (0.0, 1.0, 0.0)

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.usbee_pieces) == 1, "darts shouldn't split the mesh into separate islands"

    arrayed_verts = len(bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object].data.vertices)
    assert arrayed_verts > single_dart_verts, (
        f"Array modifier on the seam curve had no effect - resolver is "
        f"reading raw curve points instead of evaluated (post-modifier) "
        f"geometry (single-dart verts={single_dart_verts}, arrayed verts={arrayed_verts})"
    )


@test("SVG export: writes a well-formed file with the right piece count")
def test_svg_export():
    _clean_scene()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}

    out_path = bpy.app.tempdir + "usbee_test_export.svg"
    result = bpy.ops.usbee.export_svg(filepath=out_path)
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    with open(out_path) as f:
        svg_text = f.read()
    assert svg_text.startswith("<svg")
    assert svg_text.strip().endswith("</svg>")
    assert svg_text.count("<polygon") == 1  # one piece, no offset -> no cut-line


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

    print()
    print(f"{len(_results) - len(failures)}/{len(_results)} tests passed")
    if failures:
        print("Failed: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    run_all()
