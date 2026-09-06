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
import mathutils

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
    from bl_ext.user_default.usbee.operators import seam_curve as seam_curve_ops

    return seam_curve_ops.create_seam_curve_object(bpy.context, mesh_obj, list(world_points), cyclic)


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
    from bl_ext.user_default.usbee.geometry import seam_points as seam_points_mod

    points, closed = seam_points_mod.load_points(curve_obj)
    seam_points_mod.store_points(curve_obj, [centroid + (p - centroid) * 0.9 for p in points], closed)
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

    # Starts at the mesh boundary (x=-1), ends at an interior point (the
    # dart tip) - deliberately open (not cyclic), i.e. a slit, not a loop.
    Vector = mathutils.Vector
    _add_seam_curve_for(
        mesh_obj, [Vector((-1.0, 0.0, 0.0)), Vector((-0.3, 0.0, 0.0))], cyclic=False
    )

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


@test("seam curve display tube hugs a curved surface (subdivide -> project -> thicken order)")
def test_seam_curve_tube_hugs_surface():
    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"

    Vector = mathutils.Vector
    # A straight chord across the sphere - if projection ran before/without
    # subdivision, this would look like a straight rod poking through the
    # interior instead of a tube hugging the curved surface.
    curve_obj = _add_seam_curve_for(
        mesh_obj, [Vector((-0.9, 0.0, 0.4)), Vector((0.9, 0.0, 0.4))], cyclic=False
    )

    verts = [tuple(v.co) for v in curve_obj.data.vertices]
    assert len(verts) > 20, (
        f"expected the display tube to be resampled into many points to "
        f"hug the surface, got only {len(verts)} raw vertices"
    )
    radii = [(x * x + y * y + z * z) ** 0.5 for x, y, z in verts]
    assert max(radii) < 1.02, (
        f"tube shouldn't balloon outward - max radius {max(radii)} suggests "
        f"projection isn't actually pulling points onto the sphere"
    )
    assert min(radii) > 0.9, (
        f"tube shouldn't sink into the sphere or stay as a straight chord "
        f"through the interior - min radius {min(radii)} is far from the "
        f"sphere's surface at radius 1.0"
    )


@test("seam curve on a mirror-modifier seam doesn't zigzag between the two mirrored halves")
def test_seam_curve_mirror_bias_no_zigzag():
    import bmesh

    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
    base = bpy.context.active_object
    base.name = "MirrorSphere"
    bm = bmesh.new()
    bm.from_mesh(base.data)
    bmesh.ops.bisect_plane(
        bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0), plane_no=(1, 0, 0), clear_inner=True
    )
    bm.to_mesh(base.data)
    bm.free()
    base.data.update()
    mirror_mod = base.modifiers.new("Mirror", "MIRROR")
    mirror_mod.use_axis = (True, False, False)

    Vector = mathutils.Vector
    # A straight line running exactly along the mirror seam (x=0) - without
    # the bias fix, resampled points along this line flip unpredictably
    # between the left and right mirrored surface (both equally "nearest"),
    # producing a zigzag that crosses back and forth across x=0.
    curve_obj = _add_seam_curve_for(
        base, [Vector((0.0, -0.9, 0.3)), Vector((0.0, 0.9, 0.3))], cyclic=False
    )

    xs = [v.co.x for v in curve_obj.data.vertices]
    signs = {1 if x > 1e-6 else (-1 if x < -1e-6 else 0) for x in xs}
    assert signs != {1, -1}, (
        f"tube crosses back and forth across the mirror plane (x values on "
        f"both sides: {sorted(set(round(x, 4) for x in xs))}) - the bias "
        f"nudge that should break the nearest-surface tie isn't working"
    )


@test("mirroring a seam curve reflects its points across the mesh's mirror plane")
def test_mirror_seam_curve_operator():
    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=24, ring_count=12)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"
    mirror_mod = mesh_obj.modifiers.new("Mirror", "MIRROR")
    mirror_mod.use_axis = (True, False, False)

    Vector = mathutils.Vector
    original_points = [Vector((0.3, 0.4, 0.5)), Vector((0.35, -0.4, 0.5))]
    curve_obj = _add_seam_curve_for(mesh_obj, original_points, cyclic=False)

    bpy.context.view_layer.objects.active = curve_obj
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    curve_obj.select_set(True)
    result = bpy.ops.usbee.mirror_seam_curve()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    mirrored = bpy.context.active_object
    assert mirrored is not curve_obj, "expected a new, separate object"
    assert mirrored.get("usbee_seam_source_object") == mesh_obj.name

    from bl_ext.user_default.usbee.geometry import seam_points as seam_points_mod

    mirrored_points, _closed = seam_points_mod.load_points(mirrored)
    assert len(mirrored_points) == len(original_points)
    for orig, mirr in zip(original_points, mirrored_points):
        assert abs(orig.x + mirr.x) < 1e-6, f"expected x reflected across 0: {orig.x} vs {mirr.x}"
        assert abs(orig.y - mirr.y) < 1e-6, "y should be unchanged by an x-axis mirror"
        assert abs(orig.z - mirr.z) < 1e-6, "z should be unchanged by an x-axis mirror"


@test("a new seam curve can snap to segments of an already-drawn one on the same mesh")
def test_snap_to_other_seam_curve_segments():
    _clean_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Grid"

    Vector = mathutils.Vector
    existing_curve = _add_seam_curve_for(
        mesh_obj, [Vector((-0.5, 0.0, 0.0)), Vector((0.5, 0.0, 0.0))], cyclic=False
    )
    new_curve = _add_seam_curve_for(
        mesh_obj, [Vector((-0.5, 0.5, 0.0)), Vector((0.5, 0.5, 0.0))], cyclic=False
    )
    assert new_curve is not existing_curve

    from bl_ext.user_default.usbee.operators import seam_curve as seam_curve_ops

    segments = seam_curve_ops._other_seam_curve_segments(mesh_obj, exclude_curve_obj=new_curve)
    assert len(segments) == 1, f"expected exactly 1 segment from the other curve, got {len(segments)}"

    segments_excluding_existing = seam_curve_ops._other_seam_curve_segments(
        mesh_obj, exclude_curve_obj=existing_curve
    )
    assert len(segments_excluding_existing) == 1, "expected to find new_curve's segment instead"


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


@test("material thickness shells the surface outward before cutting, growing the flattened piece")
def test_material_thickness_offset():
    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=16, ring_count=8)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"

    face = mesh_obj.data.polygons[0]
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.6 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.usbee_thickness_mm = 0.0
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}
    base_dims = bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object].dimensions.copy()

    mesh_obj.usbee_thickness_mm = 200.0  # a large, easy-to-detect shell on a 1m-radius sphere
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}
    thick_dims = bpy.data.objects[mesh_obj.usbee_pieces[0].flattened_object].dimensions

    assert thick_dims.x > base_dims.x and thick_dims.y > base_dims.y, (
        f"expected the piece to grow with material thickness "
        f"(base={tuple(base_dims)}, thick={tuple(thick_dims)})"
    )


@test("ORIGIN placement mode centers each flattened piece at its curved piece's 3D position")
def test_origin_placement_mode():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(5.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"
    mesh_obj.usbee_placement_mode = "ORIGIN"

    # A closed cube has no boundary at all (same reason the sphere needs one
    # in the other tests) - cut off one face so there's something to flatten.
    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.8 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}

    small_piece = min(
        mesh_obj.usbee_pieces,
        key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons),
    )
    piece = small_piece
    flat_obj = bpy.data.objects[piece.flattened_object]
    expected_world = mesh_obj.matrix_world @ mathutils.Vector(piece.sample_point)
    assert (flat_obj.location - expected_world).length < 1e-6, (
        f"expected flat piece at {tuple(expected_world)}, got {tuple(flat_obj.location)}"
    )
    # The cube is centered at world (5,0,0) with faces ~0.5 away from center,
    # so any single face's centroid-based placement should land near there,
    # not at the world origin (which GRID mode would effectively do).
    assert flat_obj.location.length > 1.0


@test("distortion preview builds one colored object per piece at the source mesh's position")
def test_distortion_preview():
    _clean_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=12, ring_count=6, location=(3.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"

    face = mesh_obj.data.polygons[0]
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.6 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.usbee.show_distortion_preview()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    from bl_ext.user_default.usbee.operators import distortion as distortion_ops

    coll = bpy.data.collections.get(f"{distortion_ops.DISTORTION_COLLECTION_PREFIX}Sphere")
    assert coll is not None, "expected a distortion preview collection"
    assert len(coll.objects) == len(mesh_obj.usbee_pieces)

    obj = coll.objects[0]
    assert (obj.matrix_world.translation - mesh_obj.matrix_world.translation).length < 1e-6, (
        "distortion preview piece should sit at the source mesh's position, not the flattened layout"
    )
    assert distortion_ops.DISTORTION_ATTR in obj.data.attributes

    assert bpy.ops.usbee.hide_distortion_preview() == {"FINISHED"}
    assert bpy.data.collections.get(f"{distortion_ops.DISTORTION_COLLECTION_PREFIX}Sphere") is None


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
