"""Headless regression tests for the flatten pipeline, run via:

    blender --background --factory-startup --python tests/headless/test_flatten_pipeline.py

Blender's bundled interpreter here doesn't have pytest available, so this is
a small assert-based runner rather than a pytest suite (bpy-free logic is
covered by real pytest under tests/unit/ instead). Fixtures are built
programmatically with bpy.ops rather than checked-in .blend files, so
there's nothing binary to keep in sync with the addon's schema.
"""

import math
import sys
import traceback

import bpy
import mathutils

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


def _add_seam_curve_for(mesh_obj, world_points, cyclic=True):
    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

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
    result = bpy.ops.seams_to_fur.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.seams_to_fur_pieces) == 2, f"expected 2 pieces, got {len(mesh_obj.seams_to_fur_pieces)}"

    face_counts = sorted(
        len(bpy.data.objects[p.flattened_object].data.polygons) for p in mesh_obj.seams_to_fur_pieces
    )
    # Resolution now densifies the mesh before bisecting (so bisect_plane's
    # unbounded infinite-plane cut stays local to the seam), so exact face
    # counts are no longer meaningful - only the area ratio is: the shrunk
    # top-face loop encloses ~0.64 of that face's area (1/6 of the cube's
    # total surface), the rest is the remaining ~5.36/6.
    total = sum(face_counts)
    small, big = face_counts
    ratio = small / total
    expected_ratio = (0.64 / 6.0)
    assert abs(ratio - expected_ratio) < 0.05, (
        f"expected the smaller island to be about {expected_ratio:.2%} of the "
        f"total face count, got {ratio:.2%} ({small}/{total})"
    )


@test("find_seam_partners/populate_seam_partners: two pieces sharing a closed-loop seam each get one matching run")
def test_seam_partners_closed_loop():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    shrunk = [centroid + (p - centroid) * 0.8 for p in world_pts]
    _add_seam_curve_for(mesh_obj, shrunk)

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    assert len(mesh_obj.seams_to_fur_pieces) == 2

    piece_a, piece_b = mesh_obj.seams_to_fur_pieces[0], mesh_obj.seams_to_fur_pieces[1]
    assert len(piece_a.seam_partners) == 1, f"expected exactly 1 shared run, got {len(piece_a.seam_partners)}"
    assert len(piece_b.seam_partners) == 1

    partner_of_a = piece_a.seam_partners[0]
    partner_of_b = piece_b.seam_partners[0]
    assert partner_of_a.partner_piece_uuid == piece_b.uuid
    assert partner_of_b.partner_piece_uuid == piece_a.uuid

    # A closed loop should walk back to (near) its own start, and have a
    # real number of points (not degenerate/empty), on both sides.
    assert len(partner_of_a.points) >= 4, f"expected a real polyline, got {len(partner_of_a.points)} points"
    assert len(partner_of_b.points) == len(partner_of_a.points), (
        "both pieces should see the same run (same point count) for the seam they share"
    )
    pts = [mathutils.Vector(p.co) for p in partner_of_a.points]
    assert (pts[0] - pts[-1]).length < 1e-4, "a closed-loop seam run should end back where it started"

    # Sanity on the actual geometry: the shrunk loop is an 0.8-unit square
    # inset from a 1x1 top face, so its perimeter should be close to 3.2.
    total_length = sum((pts[i] - pts[i - 1]).length for i in range(1, len(pts)))
    assert abs(total_length - 3.2) < 0.1, f"expected a run length near 3.2 (0.8-unit square perimeter), got {total_length:.3f}"


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
    result = bpy.ops.seams_to_fur.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    flat_obj = bpy.data.objects[mesh_obj.seams_to_fur_pieces[0].flattened_object]
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
        result = bpy.ops.seams_to_fur.flatten_all()
    except RuntimeError:
        return  # operator raised - acceptable rejection path

    assert result == {"CANCELLED"}, (
        f"expected the operator to reject a closed mesh with CANCELLED, got {result}"
    )
    for p in mesh_obj.seams_to_fur_pieces:
        assert not p.flattened_object, "a closed island should never produce a flattened object"


@test("BFF sanity: flattening an already-flat mesh is close to identity in area")
def test_bff_already_flat_mesh():
    from bl_ext.user_default.seams_to_fur.backends import bff as bff_backend
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
    result = bpy.ops.seams_to_fur.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.seams_to_fur_pieces) == 1

    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.offset_mm = 50.0  # 5cm grow, generous relative to the 2m plane
    mesh_obj.seams_to_fur_active_piece_index = 0
    result = bpy.ops.seams_to_fur.flatten_piece()
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
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    piece = mesh_obj.seams_to_fur_pieces[0]
    flat_obj = bpy.data.objects[piece.flattened_object]
    flat_obj.location.x += 5.0
    moved_location = flat_obj.location.copy()

    mesh_obj.seams_to_fur_active_piece_index = 0
    assert bpy.ops.seams_to_fur.flatten_piece() == {"FINISHED"}
    flat_obj_after = bpy.data.objects[piece.flattened_object]
    assert flat_obj_after.name == flat_obj.name, "re-bake should update the existing object, not replace it"
    assert (flat_obj_after.location - moved_location).length < 1e-9, (
        "manually moving a flattened piece should survive a re-bake of its shape"
    )

    assert bpy.ops.seams_to_fur.reset_placement() == {"FINISHED"}
    reset_obj = bpy.data.objects[mesh_obj.seams_to_fur_pieces[0].flattened_object]
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
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    small_piece = min(mesh_obj.seams_to_fur_pieces, key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons))
    big_piece = max(mesh_obj.seams_to_fur_pieces, key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons))
    big_flat_name_before = big_piece.flattened_object
    big_obj_before = bpy.data.objects[big_flat_name_before]
    big_loc_before = big_obj_before.location.copy()

    # Shrink the seam curve further, changing only the small piece's shape -
    # directly edit the skeleton mesh's vertex data, the way native Edit
    # Mode would. curve_obj has an identity transform (see
    # create_seam_curve_object), so its local vertex coords are world-space.
    for v in curve_obj.data.vertices:
        v.co = centroid + (v.co - centroid) * 0.9
    curve_obj.data.update()
    bpy.context.view_layer.objects.active = mesh_obj

    mesh_obj.seams_to_fur_active_piece_index = list(mesh_obj.seams_to_fur_pieces).index(small_piece)
    assert bpy.ops.seams_to_fur.flatten_piece() == {"FINISHED"}

    big_obj_after = bpy.data.objects[big_flat_name_before]
    assert (big_obj_after.location - big_loc_before).length < 1e-9, (
        "re-baking one piece moved/replaced an unrelated piece's flattened object"
    )


@test("recomputing islands without changing a piece (e.g. a fur preview refresh) leaves it clean; editing the mesh dirties only the affected piece")
def test_dirty_flag_tracks_actual_geometry_changes():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.8 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    assert not any(p.flatten_dirty for p in mesh_obj.seams_to_fur_pieces), "expected every piece clean right after Flatten All"

    # Refreshing the fur preview re-runs the whole island pipeline, but
    # nothing about either piece changed - it used to mark every piece as
    # needing a re-bake anyway (confirmed live, on the bundled example).
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    dirty = [p.name for p in mesh_obj.seams_to_fur_pieces if p.flatten_dirty]
    assert not dirty, f"a preview refresh marked unchanged pieces dirty: {dirty}"

    # Reshaping the source mesh away from the seam must still dirty the
    # piece it actually changed (the bottom/sides), and only that one.
    small = min(mesh_obj.seams_to_fur_pieces, key=lambda p: len(bpy.data.objects[p.flattened_object].data.polygons))
    small_name = small.name
    for v in mesh_obj.data.vertices:
        if v.co.z < 0:
            v.co.z -= 0.2
    mesh_obj.data.update()
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    states = {p.name: p.flatten_dirty for p in mesh_obj.seams_to_fur_pieces}
    assert not states[small_name], f"the untouched top piece was marked dirty: {states}"
    assert any(dirty for name, dirty in states.items() if name != small_name), (
        f"editing the mesh didn't mark the reshaped piece dirty: {states}"
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
    result = bpy.ops.seams_to_fur.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.seams_to_fur_pieces) == 1, (
        "an open dart cut shouldn't split the mesh into separate islands - "
        "it's a slit, not a boundary"
    )

    flat_obj = bpy.data.objects[mesh_obj.seams_to_fur_pieces[0].flattened_object]
    # Resolution now densifies the mesh before bisecting (see
    # test_cube_two_islands), so face count legitimately grows rather than
    # staying exactly equal - just check it didn't shrink or explode.
    assert base_face_count <= len(flat_obj.data.polygons) <= base_face_count * 100, (
        f"expected face count to grow moderately from densification, got "
        f"{base_face_count} -> {len(flat_obj.data.polygons)}"
    )
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

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    tube_obj = seam_curve_ops._tube_object_for(curve_obj)
    assert tube_obj is not None, "expected a display tube companion object"
    verts = [tuple(v.co) for v in tube_obj.data.vertices]
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

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    tube_obj = seam_curve_ops._tube_object_for(curve_obj)
    assert tube_obj is not None, "expected a display tube companion object"
    xs = [v.co.x for v in tube_obj.data.vertices]
    signs = {1 if x > 1e-6 else (-1 if x < -1e-6 else 0) for x in xs}
    assert signs != {1, -1}, (
        f"tube crosses back and forth across the mirror plane (x values on "
        f"both sides: {sorted(set(round(x, 4) for x in xs))}) - the bias "
        f"nudge that should break the nearest-surface tie isn't working"
    )


@test("a seam along a mirror weld and one crossing it together split the mesh into 4 pieces")
def test_mirror_weld_crossing_seams_split_into_quadrants():
    """Regression test for a real-file bug: on a Mirror (merge) + Subdivision
    Surface mesh, one seam running exactly along the mirror weld (x=0) and
    another crossing it transversally should together produce 4 separate
    pieces (front-left, front-right, back-left, back-right) - the same
    shape of cut as the user's actual hood mesh, which has "a seam right
    along the mirror line and a seam that crosses the mirror line." Before
    the fix in geometry.mesh_cut/geometry.islands, this silently produced
    only 1 piece: bmesh.geometry.intersect_face_point (used by _on_face to
    test whether a point lies within a face) can false-positive for points
    far outside a small face's own footprint whenever that face's plane
    happens to have little/no extent in the direction the far point varies
    in - confirmed on this exact fixture, where a small boundary triangle
    falsely "contained" target points more than a mesh-width away, causing
    the seam walk to skip almost the entire path and never actually cut it.
    A second, compounding cause: the Mirror-seam anti-zigzag bias
    (curve_display.mirror_bias) displaces projected points away from the
    weld by more than mesh_cut's old fixed vertex-merge tolerance, so a
    forced endpoint meant to land exactly on an existing weld-boundary
    vertex created a redundant near-duplicate instead, leaving one boundary
    edge unsevered."""
    import bmesh

    _clean_scene()
    nx, ny = 6, 12
    bm = bmesh.new()
    verts = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = i / nx  # 0..1, mirrored below to -1..1
            y = -1.0 + 2.0 * j / ny
            z = 0.6 * math.cos(0.5 * math.pi * x)  # dome, ridge at x=0
            verts[(i, j)] = bm.verts.new((x, y, z))
    for i in range(nx):
        for j in range(ny):
            bm.faces.new((verts[(i, j)], verts[(i + 1, j)], verts[(i + 1, j + 1)], verts[(i, j + 1)]))
    mesh = bpy.data.meshes.new("Hood")
    bm.to_mesh(mesh)
    bm.free()
    mesh_obj = bpy.data.objects.new("Hood", mesh)
    bpy.context.scene.collection.objects.link(mesh_obj)

    mirror_mod = mesh_obj.modifiers.new("Mirror", "MIRROR")
    mirror_mod.use_axis = (True, False, False)
    mirror_mod.use_mirror_merge = True
    mirror_mod.merge_threshold = 0.001
    subsurf_mod = mesh_obj.modifiers.new("Subsurf", "SUBSURF")
    subsurf_mod.levels = 2
    subsurf_mod.render_levels = 2

    Vector = mathutils.Vector
    # (a) along the mirror weld itself (x=0), from one Y boundary to the
    # other, crossing the transversal seam at (0, 0.3).
    _add_seam_curve_for(
        mesh_obj,
        [Vector((0.0, -1.2, 0.6)), Vector((0.0, 0.3, 0.6)), Vector((0.0, 1.2, 0.6))],
        cyclic=False,
    )
    # (b) crossing transversally from the +X half to the -X half, meeting
    # the along-weld seam at that same junction point.
    _add_seam_curve_for(
        mesh_obj,
        [Vector((1.2, 0.3, -0.3)), Vector((0.0, 0.3, 0.6)), Vector((-1.2, 0.3, -0.3))],
        cyclic=False,
    )

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.seams_to_fur.flatten_all()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"
    assert len(mesh_obj.seams_to_fur_pieces) == 4, (
        f"expected the mesh to split into 4 pieces (front/back x left/right "
        f"of the mirror weld), got {len(mesh_obj.seams_to_fur_pieces)} - the along-"
        f"weld and crossing seams aren't fully severing the mesh"
    )


@test("seams_to_fur.mirror_seam_curve adds a Mirror modifier matching the source mesh's plane")
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
    result = bpy.ops.seams_to_fur.mirror_seam_curve()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    mods = [m for m in curve_obj.modifiers if m.type == "MIRROR"]
    assert len(mods) == 1, f"expected exactly one Mirror modifier, got {len(mods)}"
    assert tuple(mods[0].use_axis) == (True, False, False), "should match the source mesh's mirror axis"

    from bl_ext.user_default.seams_to_fur.geometry import curve_display

    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts, edges = curve_display.evaluated_skeleton_geometry(curve_obj, depsgraph)
    assert len(verts) == 2 * len(original_points), (
        f"expected the Mirror modifier to double the point count, got {len(verts)} from {len(original_points)}"
    )
    xs = sorted(v.x for v in verts)
    assert xs[0] < 0 < xs[-1], "expected points on both sides of the mirror plane after evaluation"

    # Re-running should be a no-op, not stack a second Mirror modifier.
    assert bpy.ops.seams_to_fur.mirror_seam_curve() == {"FINISHED"}
    assert len([m for m in curve_obj.modifiers if m.type == "MIRROR"]) == 1


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

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    segments = seam_curve_ops._other_seam_curve_segments(bpy.context, mesh_obj, exclude_curve_obj=new_curve)
    assert len(segments) == 1, f"expected exactly 1 segment from the other curve, got {len(segments)}"

    segments_excluding_existing = seam_curve_ops._other_seam_curve_segments(
        bpy.context, mesh_obj, exclude_curve_obj=existing_curve
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
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    out_path = bpy.app.tempdir + "seams_to_fur_test_export.svg"
    result = bpy.ops.seams_to_fur.export_svg(filepath=out_path)
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    with open(out_path) as f:
        svg_text = f.read()
    assert svg_text.startswith("<svg")
    assert svg_text.strip().endswith("</svg>")
    assert svg_text.count("<polygon") == 1  # one piece, no offset -> no cut-line


@test("material thickness shells the surface outward before cutting")
def test_material_thickness_offset():
    _clean_scene()
    # Tests the shell-offset mechanism itself (geometry.flatten's
    # _evaluated_bmesh) directly, rather than round-tripping through the
    # full flatten -> BFF -> piece-uuid-tracking -> area-rescale pipeline:
    # that higher-level path has its own, separate flakiness unrelated to
    # material thickness (an explicit area-correction rescale in
    # _flatten_pieces forces each piece's *output* area to always exactly
    # equal its *3D input* area by construction - so a real assertion here
    # already reduces to this same lower-level check anyway, without the
    # extra noise from piece re-identification between two independent
    # bakes).
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=16, ring_count=8)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Sphere"

    from bl_ext.user_default.seams_to_fur.operators import flatten as flatten_ops

    mesh_obj.seams_to_fur_thickness_mm = 0.0
    bm = flatten_ops._evaluated_bmesh(bpy.context, mesh_obj)
    base_radius = sum(v.co.length for v in bm.verts) / len(bm.verts)
    bm.free()

    mesh_obj.seams_to_fur_thickness_mm = 50.0  # 5% of the sphere's own radius
    bm = flatten_ops._evaluated_bmesh(bpy.context, mesh_obj)
    thick_radius = sum(v.co.length for v in bm.verts) / len(bm.verts)
    bm.free()

    assert thick_radius > base_radius * 1.03, (
        f"expected material thickness to shell the surface outward "
        f"(base_radius={base_radius}, thick_radius={thick_radius})"
    )


@test("ORIGIN placement mode centers each flattened piece at its curved piece's 3D position")
def test_origin_placement_mode():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(5.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"
    mesh_obj.seams_to_fur_placement_mode = "ORIGIN"

    # A closed cube has no boundary at all (same reason the sphere needs one
    # in the other tests) - cut off one face so there's something to flatten.
    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.8 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    small_piece = min(
        mesh_obj.seams_to_fur_pieces,
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
    # A cube, not a sphere - see test_material_thickness_offset. This test
    # only cares about object count/placement, not curvature.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.6 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    result = bpy.ops.seams_to_fur.refresh_preview(kind="DISTORTION")
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    from bl_ext.user_default.seams_to_fur.operators import distortion as distortion_ops

    coll = bpy.data.collections.get(f"{distortion_ops.DISTORTION_COLLECTION_PREFIX}Cube")
    assert coll is not None, "expected a distortion preview collection"
    assert len(coll.objects) == len(mesh_obj.seams_to_fur_pieces)

    obj = coll.objects[0]
    assert (obj.matrix_world.translation - mesh_obj.matrix_world.translation).length < 1e-6, (
        "distortion preview piece should sit at the source mesh's position, not the flattened layout"
    )
    assert distortion_ops.DISTORTION_ATTR in obj.data.attributes

    assert bpy.ops.seams_to_fur.clear_preview(kind="DISTORTION") == {"FINISHED"}
    assert bpy.data.collections.get(f"{distortion_ops.DISTORTION_COLLECTION_PREFIX}Cube") is None


@test("distortion preview hides the always-in-front seam-curve tube overlay while shown, restores it when hidden")
def test_distortion_preview_hides_seam_tube():
    # The seam-curve "tube" display (an always show_in_front beveled mesh
    # tracing every drawn seam - see seam_curve.refresh_display) stayed
    # visible regardless of which same-shape preview was active, drawing a
    # bright, always-on-top line tracing every piece boundary - an
    # unrelated overlay worth keeping out of the way of any same-shape
    # preview, distortion or otherwise (this is not what caused the
    # distortion preview's own pale-ring artifact - that was a flat-per-
    # face-shading issue, fixed by smoothing the color to per-vertex - see
    # distortion._smooth_distortion_to_vertices).
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    curve_obj = _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.6 for p in world_pts])

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    tube_obj = seam_curve_ops._tube_object_for(curve_obj)
    assert tube_obj is not None, "expected create_seam_curve_object to have built a tube display"
    assert tube_obj.visible_get(), "tube should start out visible"

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.refresh_preview(kind="DISTORTION") == {"FINISHED"}
    assert not tube_obj.visible_get(), "seam tube should be hidden while the distortion preview is shown"

    assert bpy.ops.seams_to_fur.clear_preview(kind="DISTORTION") == {"FINISHED"}
    assert tube_obj.visible_get(), "seam tube should be restored once the distortion preview is hidden"


@test("re-bake creating a new piece never produces duplicate default names (Bug 2)")
def test_new_pieces_get_unique_names():
    import bmesh

    _clean_scene()
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Host"

    from bl_ext.user_default.seams_to_fur.geometry import islands

    def make_bm(centers_x):
        """One unit quad per island, centered near each x, so each island's
        3D centroid lands at a controllable, well-separated position."""
        bm = bmesh.new()
        faces = []
        for x in centers_x:
            vs = [bm.verts.new((x + dx, dy, 0.0)) for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]]
            faces.append(bm.faces.new(vs))
        bm.faces.index_update()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        face_island = {f.index: isl for isl, f in enumerate(faces)}
        return bm, face_island

    # Bake 1: two islands -> "Piece 1" @x~0.5, "Piece 2" @x~10.5.
    bm, face_island = make_bm([0.0, 10.0])
    islands.sync_piece_settings(mesh_obj, bm, face_island, 2)
    bm.free()
    assert [p.name for p in mesh_obj.seams_to_fur_pieces] == ["Piece 1", "Piece 2"]

    # Bake 2: three islands positioned so greedy centroid matching keeps the
    # old "Piece 2" (island 0, near x~10.5) and the old "Piece 1" (island 2,
    # near x~0.5), leaving island 1 (far away, x~500.5) as a genuinely NEW
    # piece. Old logic named a new piece "Piece {island_id + 1}" = "Piece 2",
    # colliding with the retained "Piece 2".
    bm, face_island = make_bm([10.0, 500.0, 0.0])
    islands.sync_piece_settings(mesh_obj, bm, face_island, 3)
    bm.free()

    names = [p.name for p in mesh_obj.seams_to_fur_pieces]
    assert len(mesh_obj.seams_to_fur_pieces) == 3, f"expected 3 pieces, got {len(names)}"
    assert len(names) == len(set(names)), f"duplicate piece names after re-bake: {names}"
    # The retained piece really did keep "Piece 2" (so this exercised the
    # collision path, not some trivially-unique case).
    assert "Piece 2" in names and "Piece 1" in names, names


@test("every generated collection (seam curves/tubes, pattern, previews, grain arrows) nests under one shared root")
def test_collections_nest_under_shared_root():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0))
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.6 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}
    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}
    assert bpy.ops.seams_to_fur.refresh_preview(kind="DISTORTION") == {"FINISHED"}

    piece = mesh_obj.seams_to_fur_pieces[0]
    mesh_obj.seams_to_fur_active_piece_index = 0
    from bl_ext.user_default.seams_to_fur.operators import appearance, collections as coll_ops, preview

    piece.grain_direction = (0.0, 1.0, 0.0)
    piece.grain_anchor = (0.1, 0.1, 0.0)
    piece.has_grain_direction = True
    appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)

    root = bpy.data.collections.get(coll_ops.ROOT_COLLECTION_NAME)
    assert root is not None, "expected the shared root collection to exist"
    assert coll_ops.ROOT_COLLECTION_NAME in bpy.context.scene.collection.children

    root_child_names = {c.name for c in root.children}
    expected = [
        "STF Seam Curves",
        "STF Seam Curve Tubes",
        "STF Pattern — Cube",
        f"{preview.CUT_PREVIEW_COLLECTION_PREFIX}Cube",
        f"{preview.SLICED_COLLECTION_PREFIX}Cube",
        f"{preview.DISTORTION_COLLECTION_PREFIX}Cube",
        "STF Grain Directions",
    ]
    for name in expected:
        assert name in root_child_names, f"expected '{name}' nested under the shared root, got {root_child_names}"

    # Nothing STF-related should be sitting as its own loose top-level
    # collection outside the shared root - that's the whole point of it.
    top_level_names = {c.name for c in bpy.context.scene.collection.children}
    assert top_level_names == {coll_ops.ROOT_COLLECTION_NAME}, (
        f"only the shared root should be top-level, got {top_level_names}"
    )


@test("flatten_all's background-thread stages (prepare/run_one/finish) produce the same result as the synchronous path")
def test_flatten_background_stages_via_thread_pool():
    # SEAMS_TO_FUR_OT_flatten_all's interactive (real-window) path drives
    # exactly this sequence - _flatten_prepare on the main thread, one
    # _flatten_run_one per piece submitted to a ThreadPoolExecutor, then
    # _flatten_finish back on the main thread - via a modal timer instead
    # of a blocking call. There's no window/event loop in this headless
    # test to drive that modal loop (bpy.app.background is True, so the
    # operator itself always takes the synchronous _flatten_pieces
    # fallback here - see test_cube_two_islands etc.), so this test
    # exercises the same three staged functions directly with a real
    # ThreadPoolExecutor, the one piece of the interactive path that's
    # otherwise never covered by the headless suite.
    import concurrent.futures

    from bl_ext.user_default.seams_to_fur.operators import flatten as flatten_ops

    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    _add_seam_curve_for(mesh_obj, [centroid + (p - centroid) * 0.8 for p in world_pts])

    bpy.context.view_layer.objects.active = mesh_obj
    context = bpy.context

    state = flatten_ops._flatten_prepare(context, mesh_obj, None)
    pending = state["pending"]
    assert len(pending) == 2, f"expected 2 pieces prepared, got {len(pending)}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as executor:
        futures = [
            (piece_uuid, executor.submit(flatten_ops._flatten_run_one, state["binary_path"], verts_list, faces_local))
            for piece_uuid, _name, verts_list, faces_local in pending
        ]
        piece_results = {}
        for piece_uuid, future in futures:
            piece_results[piece_uuid] = (future.result(), None)

    errors = flatten_ops._flatten_finish(context, mesh_obj, state, piece_results)
    assert not errors, f"expected no errors, got {errors}"

    assert len(mesh_obj.seams_to_fur_pieces) == 2
    for piece in mesh_obj.seams_to_fur_pieces:
        assert not piece.flatten_dirty, f"{piece.name} should be clean after the background-staged flatten"
        flat_obj = bpy.data.objects.get(piece.flattened_object)
        assert flat_obj is not None and len(flat_obj.data.polygons) > 0, (
            f"{piece.name}'s flattened object should have real geometry"
        )


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
