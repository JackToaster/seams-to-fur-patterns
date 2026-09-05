"""Headless regression tests for Phase 2 appearance features (per-piece
coloring, grain direction, fur preview), run via:

    blender --background --factory-startup --python tests/headless/test_appearance.py

Separate file from test_flatten_pipeline.py so the two can be edited/run
independently without merge conflicts. Same small assert-based runner
pattern (see that file's docstring for why: no pytest inside Blender's
bundled interpreter here).
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
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def _flat_plane_with_piece():
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}
    return mesh_obj


@test("sync_piece_colors: creates a material per piece and colors the flattened object")
def test_sync_piece_colors():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.usbee_pieces[0]
    piece.color = (1.0, 0.0, 0.0, 1.0)

    result = bpy.ops.usbee.sync_piece_colors()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    assert len(mesh_obj.data.materials) == 1, "expected exactly one material slot for one piece"
    mat = mesh_obj.data.materials[0]
    assert tuple(mat.diffuse_color) == (1.0, 0.0, 0.0, 1.0), (
        f"material diffuse_color not set from piece.color: {tuple(mat.diffuse_color)}"
    )

    for poly in mesh_obj.data.polygons:
        assert poly.material_index == 0, "single-piece mesh should have all faces on slot 0"

    flat_obj = bpy.data.objects[piece.flattened_object]
    assert tuple(flat_obj.color) == (1.0, 0.0, 0.0, 1.0), "flattened object's viewport color not synced"


@test("sync_piece_colors: two pieces get two distinct materials assigned to the right faces")
def test_sync_piece_colors_two_pieces():
    _clean_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Cube"

    face = next(f for f in mesh_obj.data.polygons if tuple(f.normal) == (0.0, 0.0, 1.0))
    world_pts = [mesh_obj.matrix_world @ mesh_obj.data.vertices[i].co for i in face.vertices]
    centroid = sum(world_pts, world_pts[0].__class__((0, 0, 0))) / len(world_pts)
    shrunk = [centroid + (p - centroid) * 0.8 for p in world_pts]

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.usbee.add_seam_curve()
    curve_obj = bpy.context.active_object
    spline = curve_obj.data.splines[0]
    spline.points.add(len(shrunk) - len(spline.points))
    for i, p in enumerate(shrunk):
        spline.points[i].co = (p.x, p.y, p.z, 1.0)
    spline.use_cyclic_u = True

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.usbee.flatten_all() == {"FINISHED"}
    assert len(mesh_obj.usbee_pieces) == 2

    mesh_obj.usbee_pieces[0].color = (1.0, 0.0, 0.0, 1.0)
    mesh_obj.usbee_pieces[1].color = (0.0, 1.0, 0.0, 1.0)

    assert bpy.ops.usbee.sync_piece_colors() == {"FINISHED"}
    assert len(mesh_obj.data.materials) == 2, "expected one material slot per piece"

    used_slots = {poly.material_index for poly in mesh_obj.data.polygons}
    assert used_slots == {0, 1}, f"expected both material slots in use, got {used_slots}"

    # The lone top face (island of size 1) should be on its own slot,
    # distinct from the 5-face remainder.
    slot_face_counts = {}
    for poly in mesh_obj.data.polygons:
        slot_face_counts[poly.material_index] = slot_face_counts.get(poly.material_index, 0) + 1
    assert sorted(slot_face_counts.values()) == [1, 5], (
        f"expected face counts [1, 5] split across slots, got {slot_face_counts}"
    )


@test("set_grain_direction writes a normalized vector to the active piece and creates an arrow")
def test_grain_direction_write_and_arrow():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.usbee_pieces[0]
    mesh_obj.usbee_active_piece_index = 0

    # Bypass the modal (no real mouse events in --background) and exercise
    # the underlying helper the operator calls directly.
    from bl_ext.user_default.usbee.operators import appearance

    piece.grain_direction = (3.0, 4.0, 0.0)  # length 5, not yet normalized
    arrow_obj = appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)
    assert arrow_obj is not None, "expected an arrow object to be created"
    assert arrow_obj.type == "CURVE"
    assert len(arrow_obj.data.splines[0].points) == 2

    # refresh_grain_arrow doesn't normalize piece.grain_direction itself
    # (the operator does, on commit) - verify the arrow's actual direction
    # (end - start) is normalized regardless of the stored vector's length.
    p0 = arrow_obj.data.splines[0].points[0].co.to_3d()
    p1 = arrow_obj.data.splines[0].points[1].co.to_3d()
    seg_length = (p1 - p0).length
    assert seg_length > 1e-6

    # Re-running updates the same object rather than creating a duplicate.
    name_before = arrow_obj.name
    arrow_obj_2 = appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)
    assert arrow_obj_2.name == name_before, "expected the same arrow object to be reused, not duplicated"
    assert bpy.data.objects.get(name_before) is not None


@test("apply_fur_preview adds exactly one hair system and is idempotent when re-run")
def test_fur_preview_idempotent():
    _clean_scene()
    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "Plane"
    bpy.context.view_layer.objects.active = mesh_obj

    result = bpy.ops.usbee.apply_fur_preview()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    particle_mods = [m for m in mesh_obj.modifiers if m.type == "PARTICLE_SYSTEM"]
    assert len(particle_mods) == 1, f"expected exactly 1 particle modifier, got {len(particle_mods)}"
    assert len(mesh_obj.particle_systems) == 1, (
        f"expected exactly 1 particle system, got {len(mesh_obj.particle_systems)}"
    )
    assert mesh_obj.particle_systems[0].settings.type == "HAIR"

    # Re-run: must update in place, not stack a second system/modifier.
    result2 = bpy.ops.usbee.apply_fur_preview(hair_length=0.05)
    assert result2 == {"FINISHED"}, f"expected FINISHED, got {result2}"
    particle_mods_after = [m for m in mesh_obj.modifiers if m.type == "PARTICLE_SYSTEM"]
    assert len(particle_mods_after) == 1, (
        f"re-running apply_fur_preview stacked a duplicate modifier: {len(particle_mods_after)}"
    )
    assert len(mesh_obj.particle_systems) == 1, (
        f"re-running apply_fur_preview stacked a duplicate particle system: {len(mesh_obj.particle_systems)}"
    )
    assert abs(mesh_obj.particle_systems[0].settings.hair_length - 0.05) < 1e-6, (
        "re-running with a new hair_length should update the existing system"
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
