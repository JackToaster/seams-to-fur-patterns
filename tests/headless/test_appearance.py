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


@test("sync_piece_colors: creates a material per piece and colors the cut mesh and flattened object")
def test_sync_piece_colors():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.color = (1.0, 0.0, 0.0, 1.0)

    result = bpy.ops.seams_to_fur.sync_piece_colors()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    cut_obj = bpy.data.objects[f"{mesh_obj.name}.cut"]
    assert len(cut_obj.data.materials) == 1, "expected exactly one material slot for one piece"
    mat = cut_obj.data.materials[0]
    assert tuple(mat.diffuse_color) == (1.0, 0.0, 0.0, 1.0), (
        f"material diffuse_color not set from piece.color: {tuple(mat.diffuse_color)}"
    )

    for poly in cut_obj.data.polygons:
        assert poly.material_index == 0, "single-piece mesh should have all faces on slot 0"

    assert len(mesh_obj.data.materials) == 0, "base source mesh must never be colored"

    flat_obj = bpy.data.objects[piece.flattened_object]
    assert tuple(flat_obj.color) == (1.0, 0.0, 0.0, 1.0), "flattened object's viewport color not synced"
    assert len(flat_obj.data.materials) == 1 and flat_obj.data.materials[0] is mat, (
        "flattened object's mesh should carry the piece material"
    )


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

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    seam_curve_ops.create_seam_curve_object(bpy.context, mesh_obj, shrunk, closed=True)

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    assert len(mesh_obj.seams_to_fur_pieces) == 2

    mesh_obj.seams_to_fur_pieces[0].color = (1.0, 0.0, 0.0, 1.0)
    mesh_obj.seams_to_fur_pieces[1].color = (0.0, 1.0, 0.0, 1.0)

    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
    assert len(mesh_obj.data.materials) == 0, "base source mesh must never be colored"

    cut_obj = bpy.data.objects[f"{mesh_obj.name}.cut"]
    assert len(cut_obj.data.materials) == 2, "expected one material slot per piece"

    used_slots = {poly.material_index for poly in cut_obj.data.polygons}
    assert used_slots == {0, 1}, f"expected both material slots in use, got {used_slots}"

    # The small inset top piece should be its own slot, clearly smaller than
    # the remainder-of-the-cube slot (exact counts aren't asserted: the cut
    # mesh's topology near the seam - extra vertices/faces from resolving
    # the seam curve onto the surface - legitimately differs from the base
    # mesh's raw 6 quads).
    slot_face_counts = {}
    for poly in cut_obj.data.polygons:
        slot_face_counts[poly.material_index] = slot_face_counts.get(poly.material_index, 0) + 1
    counts = sorted(slot_face_counts.values())
    assert len(counts) == 2 and counts[0] < counts[1], (
        f"expected an asymmetric split (small top piece vs. rest), got {slot_face_counts}"
    )


@test("set_grain_direction writes a normalized vector to the active piece and creates an arrow")
def test_grain_direction_write_and_arrow():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    mesh_obj.seams_to_fur_active_piece_index = 0

    # Bypass the modal (no real mouse events in --background) and exercise
    # the underlying helper the operator calls directly.
    from bl_ext.user_default.seams_to_fur.operators import appearance

    from mathutils import Vector

    piece.grain_direction = (3.0, 4.0, 0.0)  # length 5, not yet normalized
    piece.grain_anchor = (0.1, 0.1, 0.0)
    piece.has_grain_direction = True
    arrow_obj = appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)
    assert arrow_obj is not None, "expected an arrow object to be created"
    assert arrow_obj.type == "MESH"
    # shaft rings + head-base ring + apex + base center - real arrowhead
    # geometry, not just a couple of curve control points.
    assert len(arrow_obj.data.vertices) > 4

    # refresh_grain_arrow doesn't normalize piece.grain_direction itself
    # (the operator does, on commit) - verify the arrow's actual length
    # (anchor to its farthest vertex, i.e. the tip) is a real, nonzero
    # length regardless of the stored vector's length.
    anchor = Vector(piece.grain_anchor)
    verts = [Vector(v.co) for v in arrow_obj.data.vertices]
    tip = max(verts, key=lambda v: (v - anchor).length)
    seg_length = (tip - anchor).length
    assert seg_length > 1e-6

    # Re-running updates the same object rather than creating a duplicate.
    name_before = arrow_obj.name
    arrow_obj_2 = appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)
    assert arrow_obj_2.name == name_before, "expected the same arrow object to be reused, not duplicated"
    assert bpy.data.objects.get(name_before) is not None

    # has_grain_direction False removes the arrow entirely.
    piece.has_grain_direction = False
    assert appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece) is None
    assert bpy.data.objects.get(name_before) is None, "arrow should be removed once has_grain_direction is False"


def pytest_approx(value, tol=1e-5):
    class _Approx:
        def __eq__(self, other):
            return abs(other - value) <= tol

    return _Approx()


@test("set_grain_direction: clicking a piece identifies it, and Mirror Edit finds+mirrors the piece across the plane")
def test_grain_direction_click_pick_and_mirror():
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()
    mesh_obj.seams_to_fur_grain_mirror_edit = True

    from bl_ext.user_default.seams_to_fur.operators import appearance
    from mathutils import Vector

    # bpy Operator subclasses can't be instantiated directly (they only
    # exist as bpy_struct instances Blender creates itself when actually
    # run) - use a plain stand-in and call the unbound methods on it.
    class _FakeOp:
        pass

    OpCls = appearance.SEAMS_TO_FUR_OT_set_grain_direction
    op = _FakeOp()
    op.mesh_obj = mesh_obj
    OpCls._ensure_sliced_preview(op, bpy.context)
    OpCls._build_bvhs(op, bpy.context)
    assert len(op.piece_bvhs) == 2, "expected a sliced-preview BVH for each of the two pieces"

    (
        op.mirror_point,
        op.mirror_normal,
        op.has_real_mirror,
        _thresh,
    ) = appearance.curve_display.mirror_plane_for(mesh_obj)
    assert op.has_real_mirror
    op.mirror_edit = True

    # Click a point clearly on the right half (x > 0) and confirm the
    # correct piece is identified, and the mirror-plane reflection (x < 0)
    # correctly resolves to the OTHER piece.
    right_point = Vector((0.75, 0.25, 0.0))
    clicked_piece, clicked_co, clicked_normal = OpCls._piece_at(op, right_point)
    assert clicked_piece is not None
    assert clicked_co is not None and clicked_normal is not None

    mirrored_point = OpCls._mirror_local_point(op, right_point)
    assert mirrored_point.x < 0.0, "expected the mirror plane (x=0) to reflect to the left half"
    mirror_piece, mirror_co, mirror_normal = OpCls._piece_at(op, mirrored_point, exclude=clicked_piece)
    assert mirror_piece is not None
    assert mirror_co is not None and mirror_normal is not None
    assert mirror_piece is not clicked_piece, "mirror piece must be the OTHER piece, not the one clicked"

    # A direction reflected across the x=0 plane should have its x
    # component flipped and y/z preserved.
    direction = Vector((0.6, 0.8, 0.0))
    mirrored_direction = OpCls._mirror_local_direction(op, direction)
    assert mirrored_direction.x == pytest_approx(-direction.x)
    assert mirrored_direction.y == pytest_approx(direction.y)

    clicked_piece.grain_direction = (direction.x, direction.y, direction.z)
    mirror_piece.grain_direction = (mirrored_direction.x, mirrored_direction.y, mirrored_direction.z)
    assert clicked_piece.grain_direction[0] == pytest_approx(direction.x)
    assert mirror_piece.grain_direction[0] == pytest_approx(-direction.x)


@test("set_grain_direction: tangent-constrains the direction and maps an arrow onto the flattened piece too")
def test_grain_direction_tangent_and_flat_mapping():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]

    import types

    from bl_ext.user_default.seams_to_fur.operators import appearance
    from mathutils import Vector

    class _FakeOp:
        pass

    OpCls = appearance.SEAMS_TO_FUR_OT_set_grain_direction
    op = _FakeOp()
    op.mesh_obj = mesh_obj
    OpCls._ensure_sliced_preview(op, bpy.context)
    OpCls._build_bvhs(op, bpy.context)
    (op.mirror_point, op.mirror_normal, op.has_real_mirror, _thresh) = appearance.curve_display.mirror_plane_for(
        mesh_obj
    )
    op.mirror_edit = False
    # _apply calls self._place_flat_preview/_get_flat_proxy (and, with a
    # mirror piece, self._mirror_local_direction) - bind these onto the
    # plain stand-in so that unbound-method call resolves.
    for name in ("_place_flat_preview", "_get_flat_proxy", "_mirror_local_direction"):
        setattr(op, name, types.MethodType(getattr(OpCls, name), op))

    click_point = Vector((0.2, 0.3, 0.0))  # a point on the plane's own surface
    piece_hit, co, normal = OpCls._piece_at(op, click_point)
    assert piece_hit is not None
    op.piece, op.mirror_piece = piece_hit, None
    op.start, op.start_normal = co, normal

    # A drag direction with a large out-of-plane component should still end
    # up perfectly flat (tangent) once projected - the plane's normal is
    # +/-Z, so any projected direction should end up with zero Z.
    raw_dir = Vector((1.0, 1.0, 5.0)).normalized()
    tangent_dir = appearance._project_tangent(raw_dir, op.start_normal)
    assert tangent_dir is not None
    assert abs(tangent_dir.z) < 1e-5, "expected the tangent-projected direction to have no out-of-plane component"

    OpCls._apply(op, bpy.context, tangent_dir, 0.5, commit=True)

    assert piece_hit.has_grain_direction
    assert (Vector(piece_hit.grain_anchor) - co).length < 1e-6, "arrow should anchor at the clicked point, not a median"
    assert (Vector(piece_hit.grain_direction) - tangent_dir).length < 1e-5

    arrow_3d = bpy.data.objects.get(appearance._arrow_object_name(mesh_obj, piece_hit))
    assert arrow_3d is not None, "expected the 3D arrow to be built"
    closest_dist = min((Vector(v.co) - co).length for v in arrow_3d.data.vertices)
    assert closest_dist < 1e-6, "expected a vertex (the base center) to sit exactly at the anchor point"

    flat_obj = bpy.data.objects.get(piece_hit.flattened_object)
    assert flat_obj is not None
    flat_arrow = bpy.data.objects.get(appearance._flat_arrow_object_name(flat_obj))
    assert flat_arrow is not None, "expected an arrow on the flattened piece too"


@test("reset_grain_direction / reset_all_grain_directions clear direction and remove arrows")
def test_reset_grain_direction():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    mesh_obj.seams_to_fur_active_piece_index = 0

    from bl_ext.user_default.seams_to_fur.operators import appearance

    def _set_and_refresh():
        piece.grain_direction = (0.0, 1.0, 0.0)
        piece.grain_anchor = (0.1, 0.1, 0.0)
        piece.has_grain_direction = True
        appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)

    arrow_name = appearance._arrow_object_name(mesh_obj, piece)

    _set_and_refresh()
    assert bpy.data.objects.get(arrow_name) is not None
    assert bpy.ops.seams_to_fur.reset_grain_direction() == {"FINISHED"}
    assert not piece.has_grain_direction
    assert bpy.data.objects.get(arrow_name) is None

    _set_and_refresh()
    assert bpy.data.objects.get(arrow_name) is not None
    assert bpy.ops.seams_to_fur.reset_all_grain_directions() == {"FINISHED"}
    assert not piece.has_grain_direction
    assert bpy.data.objects.get(arrow_name) is None


@test("toggle_grain_arrows shows/hides the shared arrow collection; disabled/no-op when no arrows exist")
def test_toggle_grain_arrows():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    mesh_obj.seams_to_fur_active_piece_index = 0

    from bl_ext.user_default.seams_to_fur.operators import appearance

    assert not appearance.grain_arrows_exist()
    assert bpy.ops.seams_to_fur.toggle_grain_arrows() == {"CANCELLED"}, (
        "no arrows built yet, so toggling should be a safe no-op, not an error"
    )

    piece.grain_direction = (0.0, 1.0, 0.0)
    piece.grain_anchor = (0.1, 0.1, 0.0)
    piece.has_grain_direction = True
    appearance.refresh_grain_arrow(bpy.context, mesh_obj, piece)

    assert appearance.grain_arrows_exist()
    assert appearance.grain_arrows_visible(), "arrow collection should start out visible"

    assert bpy.ops.seams_to_fur.toggle_grain_arrows() == {"FINISHED"}
    assert not appearance.grain_arrows_visible()
    assert appearance.grain_arrows_exist(), "toggling off must not delete the arrows"

    assert bpy.ops.seams_to_fur.toggle_grain_arrows() == {"FINISHED"}
    assert appearance.grain_arrows_visible()


@test("target.resolve_mesh_obj: clicking a preview/flattened object resolves back to the source mesh")
def test_resolve_mesh_obj_from_generated_objects():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import preview, target

    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}
    sliced_objs = list(bpy.data.collections[f"{preview.SLICED_COLLECTION_PREFIX}{mesh_obj.name}"].objects)
    assert sliced_objs

    bpy.context.view_layer.objects.active = sliced_objs[0]
    assert target.resolve_mesh_obj(bpy.context) is mesh_obj, (
        "clicking a sliced-preview object should resolve back to the source mesh"
    )

    piece = mesh_obj.seams_to_fur_pieces[0]
    flat_obj = bpy.data.objects.get(piece.flattened_object)
    assert flat_obj is not None
    bpy.context.view_layer.objects.active = flat_obj
    assert target.resolve_mesh_obj(bpy.context) is mesh_obj, (
        "clicking a flattened pattern piece should resolve back to the source mesh"
    )

    # An operator that used to require context.active_object == mesh_obj
    # directly now works with a sliced-preview object active instead.
    bpy.context.view_layer.objects.active = sliced_objs[0]
    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}


@test("target.resolve_piece_index: selecting a piece's object resolves to that piece's list index")
def test_resolve_piece_index():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    assert len(mesh_obj.seams_to_fur_pieces) == 2

    from bl_ext.user_default.seams_to_fur.operators import preview, target

    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}
    sliced_coll = bpy.data.collections[f"{preview.SLICED_COLLECTION_PREFIX}{mesh_obj.name}"]

    for expected_index, piece in enumerate(mesh_obj.seams_to_fur_pieces):
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        assert sliced_obj in list(sliced_coll.objects)
        bpy.context.view_layer.objects.active = sliced_obj
        assert target.resolve_piece_index(bpy.context, mesh_obj) == expected_index, (
            f"clicking {piece.name}'s sliced object should resolve to piece list index {expected_index}"
        )

    # A flattened pattern piece object should resolve the same way.
    piece0 = mesh_obj.seams_to_fur_pieces[0]
    flat_obj = bpy.data.objects.get(piece0.flattened_object)
    assert flat_obj is not None
    bpy.context.view_layer.objects.active = flat_obj
    assert target.resolve_piece_index(bpy.context, mesh_obj) == 0

    # An object with no piece tag at all (the base mesh itself) resolves
    # to None, not some stale/garbage index.
    bpy.context.view_layer.objects.active = mesh_obj
    assert target.resolve_piece_index(bpy.context, mesh_obj) is None


@test("Pieces list selection follows the viewport: clicking a sliced piece updates seams_to_fur_active_piece_index")
def test_panel_syncs_active_piece_from_selection():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}

    from bl_ext.user_default.seams_to_fur.ui import panels

    panels._last_synced_object_name = None  # this module-level cache persists across tests

    piece1 = mesh_obj.seams_to_fur_pieces[1]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece1.name}.sliced"]
    bpy.context.view_layer.objects.active = sliced_obj
    mesh_obj.seams_to_fur_active_piece_index = 0  # start on a different piece than the one we're about to select

    panels._sync_active_piece_from_selection(bpy.context, mesh_obj)
    assert mesh_obj.seams_to_fur_active_piece_index == 1, (
        "selecting piece 1's object in the viewport should select it in the Pieces list"
    )

    # A manual list click (index changed without the active object
    # changing) must NOT get immediately overridden back on the next sync.
    mesh_obj.seams_to_fur_active_piece_index = 0
    panels._sync_active_piece_from_selection(bpy.context, mesh_obj)
    assert mesh_obj.seams_to_fur_active_piece_index == 0, (
        "a manual list selection shouldn't be fought while the same viewport object stays active"
    )


@test("refresh_fur_preview adds one hair system per piece, on each piece's own Sliced Preview object, and is idempotent")
def test_fur_preview_idempotent():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()

    result = bpy.ops.seams_to_fur.refresh_fur_preview()
    assert result == {"FINISHED"}, f"expected FINISHED, got {result}"

    # refresh_fur_preview's own _ensure_sliced_preview -> compute_islands call
    # re-syncs (clears + rebuilds) seams_to_fur_pieces, invalidating any reference
    # captured before the call - re-fetch afterward (same caveat flatten.py
    # itself documents for _flatten_pieces).
    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]

    assert len(mesh_obj.particle_systems) == 0, "fur must live on the piece's Sliced Preview object, not the source mesh"
    particle_mods = [m for m in sliced_obj.modifiers if m.type == "PARTICLE_SYSTEM"]
    assert len(particle_mods) == 1, f"expected exactly 1 particle modifier, got {len(particle_mods)}"
    assert len(sliced_obj.particle_systems) == 1
    assert sliced_obj.particle_systems[0].settings.type == "HAIR"
    assert abs(sliced_obj.particle_systems[0].settings.hair_length - piece.fur_length) < 1e-6, (
        "hair_length should come from the piece's own fur_length, not a global operator property"
    )

    # Re-run after changing the piece's fur_length: must update in place,
    # not stack a second system/modifier.
    piece.fur_length = 0.05
    result2 = bpy.ops.seams_to_fur.refresh_fur_preview()
    assert result2 == {"FINISHED"}, f"expected FINISHED, got {result2}"
    particle_mods_after = [m for m in sliced_obj.modifiers if m.type == "PARTICLE_SYSTEM"]
    assert len(particle_mods_after) == 1, (
        f"re-running refresh_fur_preview stacked a duplicate modifier: {len(particle_mods_after)}"
    )
    assert len(sliced_obj.particle_systems) == 1, (
        f"re-running refresh_fur_preview stacked a duplicate particle system: {len(sliced_obj.particle_systems)}"
    )
    assert abs(sliced_obj.particle_systems[0].settings.hair_length - 0.05) < 1e-6, (
        "re-running after changing piece.fur_length should update the existing system"
    )
    assert not sliced_obj.particle_systems[0].settings.use_advanced_hair, (
        "use_advanced_hair must stay off - it makes the rest shape an unbaked, unbent cloth-sim result "
        "(the actual cause of the 'long straight hairs ignoring grain direction' bug)"
    )


@test("refresh_fur_preview: fur material tracks the piece's own color")
def test_fur_preview_respects_piece_color():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.color = (0.1, 0.9, 0.2, 1.0)

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]  # re-fetch - compute_islands re-syncs seams_to_fur_pieces
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]

    psys = sliced_obj.particle_systems[0]
    mat_name = psys.settings.material_slot
    mat = sliced_obj.data.materials.get(mat_name)
    assert mat is not None, "fur material slot should reference a real material on the sliced object"
    assert tuple(mat.diffuse_color)[:3] == (
        piece.color[0],
        piece.color[1],
        piece.color[2],
    ), "fur material color should match the piece's own color"


@test("refresh_fur_preview: recolors even a pre-existing fur material with an incompatible (non-Hair-BSDF) node setup")
def test_fur_preview_recolors_stale_non_hair_material():
    # Materials are looked up and reused purely by name (see
    # appearance._get_or_create_fur_material) - confirmed live on a real
    # project file that a material by that exact name can already exist
    # with a totally different node setup (a stray Principled BSDF node,
    # left over from before this function switched to a real Hair BSDF)
    # gets silently reused as-is. Silently skipping the color update
    # whenever the expected node isn't found left that piece's fur stuck at
    # the node's default grey forever, no matter how many times
    # refresh_fur_preview was re-run - reproduce that exact setup here so a
    # regression can't silently reappear.
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    piece.color = (0.1, 0.9, 0.2, 1.0)

    from bl_ext.user_default.seams_to_fur.operators import appearance

    stale_mat_name = f"{appearance.FUR_MATERIAL_NAME} {piece.uuid[:8]}"
    stale_mat = bpy.data.materials.new(stale_mat_name)
    stale_mat.use_nodes = True
    stale_mat.node_tree.nodes.clear()
    stale_mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]  # re-fetch - compute_islands re-syncs seams_to_fur_pieces
    mat = bpy.data.materials.get(stale_mat_name)
    assert mat is stale_mat, "should still reuse the same material datablock by name, not create a second one"
    bsdf = mat.node_tree.nodes.get("Hair BSDF")
    assert bsdf is not None, "a stale non-Hair-BSDF node setup must be rebuilt, not left as-is"
    assert tuple(bsdf.inputs["Color"].default_value)[:3] == (
        piece.color[0],
        piece.color[1],
        piece.color[2],
    ), "must actually recolor the rebuilt material to the piece's current color, not leave it at a node default"


@test("refresh_fur_preview: grain direction drives tangent_factor/normal_factor, one system per piece's own sliced object")
def test_fur_preview_grain_direction_per_piece():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    for piece in mesh_obj.seams_to_fur_pieces:
        piece.grain_direction = (0.0, 1.0, 0.0)
        piece.grain_anchor = (0.0, 0.0, 0.5)
        piece.has_grain_direction = True

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    assert len(mesh_obj.particle_systems) == 0, "fur must live on each piece's Sliced Preview object, not the source mesh"

    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        assert len(sliced_obj.particle_systems) == 1, f"expected exactly one fur system on {sliced_obj.name}"
        assert sliced_obj.particle_systems[0].settings.tangent_factor > 0.0, (
            "a piece with a grain direction set should comb via tangent_factor"
        )


@test("refresh_fur_preview: a Mirror-modifier-only piece gets its own fur too (no base-mesh geometry needed)")
def test_fur_preview_mirrored_piece_gets_fur():
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    assert len(mesh_obj.seams_to_fur_pieces) == 2
    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        assert len(sliced_obj.particle_systems) == 1, (
            f"{piece.name} (one half of a mirrored mesh) should get its own fur system too, "
            "not be skipped for having no real base-mesh geometry"
        )


@test("refresh_fur_preview: comb direction is correctly aligned on BOTH halves of a mirrored piece")
def test_fur_preview_mirrored_piece_alignment():
    # A mirrored piece's sliced-preview faces have reversed winding (kept
    # that way so normals stay outward-facing after the mirror transform)
    # - confirmed empirically that this flips the SIGN of how rotating a
    # piece's UV maps to a change in the actual rendered hair angle, so a
    # correction computed assuming the sign is always the same sent a
    # mirrored piece's fur in a wildly wrong direction (up to ~240 degrees
    # off) instead of just leaving a small residual error. This is the
    # single fixture most likely to silently regress that fix.
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()

    for piece in mesh_obj.seams_to_fur_pieces:
        piece.grain_direction = (0.6, 0.8, 0.0)
        piece.grain_anchor = tuple(piece.sample_point)
        piece.has_grain_direction = True

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}

    from bl_ext.user_default.seams_to_fur.operators import appearance
    from mathutils import Vector
    import math

    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        flat_obj = bpy.data.objects.get(piece.flattened_object)
        assert flat_obj is not None
        proxy = appearance._build_flat_proxy(flat_obj)
        assert proxy is not None

        target = appearance._map_direction_to_flat(
            *proxy, Vector(piece.grain_anchor), Vector(piece.grain_direction)
        )
        assert target is not None
        target_angle = math.degrees(math.atan2(target[1].y, target[1].x))

        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        eval_psys = sliced_obj.evaluated_get(depsgraph).particle_systems[0]
        p0 = eval_psys.particles[0]
        root, tip = p0.hair_keys[0].co, p0.hair_keys[-1].co
        mapped = appearance._map_direction_to_flat(*proxy, root, (tip - root).normalized())
        assert mapped is not None
        actual_angle = math.degrees(math.atan2(mapped[1].y, mapped[1].x))

        diff = abs(((actual_angle - target_angle + 180) % 360) - 180)
        assert diff < 20.0, (
            f"{piece.name}: fur direction {actual_angle:.1f} deg is {diff:.1f} deg off the "
            f"target {target_angle:.1f} deg - the mirror-winding sign fix may have regressed"
        )


@test("refresh_fur_preview: combed fur lifts slightly off the surface (normal_factor > 0) while keeping fur_length")
def test_fur_preview_combed_lift():
    import math

    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    for piece in mesh_obj.seams_to_fur_pieces:
        piece.grain_direction = (0.0, 1.0, 0.0)
        piece.grain_anchor = (0.0, 0.0, 0.5)
        piece.has_grain_direction = True

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}

    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        settings = sliced_obj.particle_systems[0].settings
        assert settings.normal_factor > 0.0, (
            "combed fur should lift slightly off the surface, not lie perfectly flat "
            f"(tangent-only) - normal_factor was {settings.normal_factor}"
        )
        assert settings.tangent_factor > 0.0, "combed fur should still be mostly combed along the grain"
        magnitude = math.sqrt(settings.normal_factor**2 + settings.tangent_factor**2)
        actual_length = settings.hair_step * magnitude
        assert abs(actual_length - piece.fur_length) < 1e-5, (
            f"lifting combed fur off the surface must still preserve fur_length - "
            f"got {actual_length}, expected {piece.fur_length}"
        )


@test("clear_fur_preview removes the particle system and fur UV from every piece, and is safe to re-run")
def test_clear_fur_preview():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    assert len(sliced_obj.particle_systems) == 1

    from bl_ext.user_default.seams_to_fur.operators import appearance

    assert bpy.ops.seams_to_fur.clear_fur_preview() == {"FINISHED"}
    assert len(sliced_obj.particle_systems) == 0, "clear_fur_preview should remove the particle system"
    assert sliced_obj.modifiers.get(appearance.FUR_MODIFIER_NAME) is None
    assert sliced_obj.data.uv_layers.get(appearance.FUR_UV_LAYER) is None, (
        "clear_fur_preview should also remove the fur UV layer it wrote"
    )
    # Sliced Preview object itself must survive - this only undoes the fur,
    # not seams_to_fur.clear_preview(kind='SLICED').
    assert bpy.data.objects.get(sliced_obj.name) is not None

    # Re-running with no fur applied should be a safe no-op, not an error.
    assert bpy.ops.seams_to_fur.clear_fur_preview() == {"FINISHED"}


@test("toggle_fur_preview is a cheap show_viewport flip; is_fur_stale reflects whether the Sliced Preview exists yet")
def test_fur_preview_cache_toggle():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()

    from bl_ext.user_default.seams_to_fur.operators import appearance

    assert not appearance.is_fur_cached(mesh_obj)
    assert appearance.is_fur_stale(mesh_obj), "no Sliced Preview built yet, so a refresh must build it first"

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    assert appearance.is_fur_cached(mesh_obj)
    assert appearance.is_fur_visible(mesh_obj)
    assert not appearance.is_fur_stale(mesh_obj), "the Sliced Preview now exists for every piece"

    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    mod = sliced_obj.modifiers.get(appearance.FUR_MODIFIER_NAME)
    assert mod is not None

    assert bpy.ops.seams_to_fur.toggle_fur_preview() == {"FINISHED"}
    assert not appearance.is_fur_visible(mesh_obj)
    assert not mod.show_viewport
    assert appearance.is_fur_cached(mesh_obj), "toggling off must not remove the particle system"

    assert bpy.ops.seams_to_fur.toggle_fur_preview() == {"FINISHED"}
    assert appearance.is_fur_visible(mesh_obj)
    assert mod.show_viewport

    assert bpy.ops.seams_to_fur.clear_fur_preview() == {"FINISHED"}
    assert not appearance.is_fur_cached(mesh_obj)

    # Deleting the Sliced Preview object makes a refresh stale again (it
    # would need to rebuild it), and toggle on an uncached preview is a
    # safe no-op, not an error.
    assert bpy.ops.seams_to_fur.toggle_fur_preview() == {"CANCELLED"}


@test("refreshing after a toggle-off restores visibility on every already-existing piece, not just newly-added ones")
def test_fur_preview_refresh_restores_visibility_after_toggle_off():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import appearance

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    assert appearance.is_fur_visible(mesh_obj)

    assert bpy.ops.seams_to_fur.toggle_fur_preview() == {"FINISHED"}
    assert not appearance.is_fur_visible(mesh_obj)

    # Refreshing again is the user asking to see the fur preview - it must
    # not leave any already-existing piece's modifier hidden from an
    # earlier toggle-off. A brand new modifier defaults to show_viewport =
    # True, so this bug only showed up on pieces that already had one:
    # confirmed live on a real file where fur had been toggled off, two
    # new pieces were split out, and a refresh afterward left only those
    # two (freshly-created) pieces showing fur while every
    # previously-toggled-off piece stayed invisible despite having a
    # freshly re-synced particle system underneath.
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    assert appearance.is_fur_visible(mesh_obj)
    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        mod = sliced_obj.modifiers.get(appearance.FUR_MODIFIER_NAME)
        assert mod is not None and mod.show_viewport, f"{piece.name}'s fur modifier is still hidden after refresh"


def _cube_with_seam_loop():
    """A unit cube with a closed seam curve looping just inside its top face,
    which cuts it into two islands (the lone top face + the 5-face
    remainder) - the same fixture shape as test_sync_piece_colors_two_pieces.
    """
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


@test("fur preview stays in sync after a seam curve edit reshapes a piece, instead of fur rendering on stale Sliced Preview geometry")
def test_fur_preview_rebuilds_after_seam_edit():
    import bmesh

    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    piece = mesh_obj.seams_to_fur_pieces[0]

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    verts_before = [tuple(v.co) for v in sliced_obj.data.vertices]

    # A real seam adjustment: shrink the loop further, so the actual cut
    # (which faces belong to which piece) genuinely changes shape - not
    # just a no-op re-save of the same geometry.
    curve_obj = seam_curve_ops._find_seam_curves_for(mesh_obj)[0]
    bm = bmesh.new()
    bm.from_mesh(curve_obj.data)
    bm.verts.ensure_lookup_table()
    center = sum((v.co for v in bm.verts), bm.verts[0].co.__class__((0, 0, 0))) / len(bm.verts)
    for v in bm.verts:
        v.co = center + (v.co - center) * 0.5
    bm.to_mesh(curve_obj.data)
    curve_obj.data.update()
    bm.free()

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    sliced_obj_after = bpy.data.objects.get(f"{mesh_obj.name}.{piece.name}.sliced")
    assert sliced_obj_after is not None
    verts_after = [tuple(v.co) for v in sliced_obj_after.data.vertices]

    assert verts_after != verts_before, (
        "Sliced Preview geometry (and therefore the fur built on it) must be rebuilt after a seam "
        "edit changes the cut - it must not be left stale just because an object with this piece's "
        "name already existed"
    )


def _mirrored_plane_two_pieces():
    """A plane occupying x in [0, 1] with a Mirror modifier (merge OFF) that
    duplicates it to x in [-1, 0], giving two mirror-image halves that are
    two separate islands/pieces on the evaluated mesh - but only ONE half of
    faces on the base (pre-mirror) mesh. This is the exact configuration that
    used to make base-mesh nearest-centroid coloring collapse both pieces
    onto the same faces (the user-reported mirrored-coloring bug).
    """
    import bmesh

    bpy.ops.mesh.primitive_plane_add(size=2.0)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "MirrorPlane"
    me = mesh_obj.data

    bm = bmesh.new()
    bm.from_mesh(me)
    for v in bm.verts:
        v.co.x = (v.co.x * 0.5) + 0.5  # remap x from [-1, 1] to [0, 1]
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=2, use_grid_fill=True)
    bm.to_mesh(me)
    bm.free()

    mod = mesh_obj.modifiers.new("Mirror", "MIRROR")
    mod.use_axis[0] = True
    mod.use_mirror_merge = False

    bpy.context.view_layer.objects.active = mesh_obj
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}
    assert len(mesh_obj.seams_to_fur_pieces) == 2
    return mesh_obj


@test("refresh/clear_preview(kind=CUT) builds one cut mesh with marked seam edges and removes it")
def test_cut_preview_show_hide():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import preview

    assert not mesh_obj.hide_get(), "base mesh should start visible"
    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}

    coll_name = f"{preview.CUT_PREVIEW_COLLECTION_PREFIX}{mesh_obj.name}"
    coll = bpy.data.collections.get(coll_name)
    assert coll is not None, "cut preview collection was not created"
    assert len(coll.objects) == 1, f"expected exactly one cut object, got {len(coll.objects)}"

    cut_obj = bpy.data.objects.get(f"{mesh_obj.name}.cut")
    assert cut_obj is not None and cut_obj.type == "MESH"
    # The resolved seam must be present as actual marked mesh edges.
    n_seam = sum(1 for e in cut_obj.data.edges if e.use_seam)
    assert n_seam > 0, "cut preview mesh has no edges marked use_seam"
    # Placed at the source mesh's own 3D transform.
    assert cut_obj.matrix_world == mesh_obj.matrix_world
    # The base mesh occupies the exact same 3D space as the cut preview -
    # it must get out of the way so the preview is actually clickable.
    assert mesh_obj.hide_get(), "base mesh should be hidden while the cut preview is shown"

    # Re-running updates in place rather than stacking duplicates.
    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}
    assert len(bpy.data.collections[coll_name].objects) == 1

    assert bpy.ops.seams_to_fur.clear_preview(kind="CUT") == {"FINISHED"}
    assert bpy.data.collections.get(coll_name) is None, "cut preview collection not removed"
    assert bpy.data.objects.get(f"{mesh_obj.name}.cut") is None, "cut preview object not removed"
    assert not mesh_obj.hide_get(), "base mesh should be restored once the cut preview is hidden"


@test("refresh/clear_preview(kind=SLICED) builds one still-3D object per piece and removes them")
def test_sliced_preview_show_hide():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import preview

    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}

    coll_name = f"{preview.SLICED_COLLECTION_PREFIX}{mesh_obj.name}"
    coll = bpy.data.collections.get(coll_name)
    assert coll is not None, "sliced preview collection was not created"
    assert len(coll.objects) == 2, f"expected one sliced object per piece (2), got {len(coll.objects)}"

    for obj in coll.objects:
        assert obj.type == "MESH"
        assert len(obj.data.materials) == 1, "each sliced object should carry its own piece material"
        assert obj.matrix_world == mesh_obj.matrix_world, "sliced object not at source's 3D transform"

    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}
    assert len(bpy.data.collections[coll_name].objects) == 2, "re-running stacked duplicate sliced objects"

    assert bpy.ops.seams_to_fur.clear_preview(kind="SLICED") == {"FINISHED"}
    assert bpy.data.collections.get(coll_name) is None, "sliced preview collection not removed"
    assert not mesh_obj.hide_get(), "base mesh should be restored once the sliced preview is hidden"


@test("cut/distortion/sliced previews are mutually exclusive and hide/restore the base mesh")
def test_same_shape_previews_mutually_exclusive():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import preview

    def visible_prefixes():
        return {
            prefix
            for prefix in preview.SAME_SHAPE_COLLECTION_PREFIXES
            if preview.is_preview_visible(mesh_obj, prefix)
        }

    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}
    assert visible_prefixes() == {preview.CUT_PREVIEW_COLLECTION_PREFIX}
    assert mesh_obj.hide_get()

    # Showing the sliced preview next must hide (not delete - it stays
    # cached) the cut preview (same 3D space) rather than leaving both
    # visible at once, and the base mesh must stay hidden throughout since
    # a same-shape preview is still up.
    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}
    assert visible_prefixes() == {preview.SLICED_COLLECTION_PREFIX}
    assert mesh_obj.hide_get()
    assert bpy.data.collections.get(f"{preview.CUT_PREVIEW_COLLECTION_PREFIX}{mesh_obj.name}") is not None, (
        "cut preview should still be cached (hidden, not deleted) after switching to sliced"
    )

    assert bpy.ops.seams_to_fur.refresh_preview(kind="DISTORTION") == {"FINISHED"}
    assert visible_prefixes() == {preview.DISTORTION_COLLECTION_PREFIX}
    assert mesh_obj.hide_get()

    assert bpy.ops.seams_to_fur.clear_preview(kind="DISTORTION") == {"FINISHED"}
    assert visible_prefixes() == set()
    assert not mesh_obj.hide_get(), "base mesh should be restored once no same-shape preview remains"


@test("toggle_preview is a cheap visibility flip; refresh skips recompute when nothing changed; clear deletes")
def test_preview_cache_toggle_refresh_clear():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.operators import preview

    prefix = preview.CUT_PREVIEW_COLLECTION_PREFIX

    assert not preview.is_preview_cached(mesh_obj, prefix)
    assert preview.is_preview_stale(mesh_obj, prefix), "nothing cached yet, so a refresh must be a real build"

    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}
    assert preview.is_preview_cached(mesh_obj, prefix)
    assert preview.is_preview_visible(mesh_obj, prefix)
    assert not preview.is_preview_stale(mesh_obj, prefix), "nothing changed since the refresh that just ran"

    cut_obj = bpy.data.objects[f"{mesh_obj.name}.cut"]
    original_mesh = cut_obj.data

    # A second refresh with nothing changed must be a true no-op - same
    # mesh datablock, not rebuilt from scratch.
    assert bpy.ops.seams_to_fur.refresh_preview(kind="CUT") == {"FINISHED"}
    assert bpy.data.objects[f"{mesh_obj.name}.cut"].data is original_mesh

    # Toggle off/on never touches the cached object at all.
    assert bpy.ops.seams_to_fur.toggle_preview(kind="CUT") == {"FINISHED"}
    assert not preview.is_preview_visible(mesh_obj, prefix)
    assert preview.is_preview_cached(mesh_obj, prefix), "toggling off must not delete the cached preview"
    assert not mesh_obj.hide_get(), "base mesh should be restored once no same-shape preview is visible"

    assert bpy.ops.seams_to_fur.toggle_preview(kind="CUT") == {"FINISHED"}
    assert preview.is_preview_visible(mesh_obj, prefix)
    assert mesh_obj.hide_get()

    # Changing the source geometry invalidates the cache. Edited directly
    # (not via bpy.ops.object.mode_set(mode="EDIT")) since the base mesh is
    # currently hidden - refresh_preview hid it to get it out of the way of
    # the same-shape cut preview - and entering Edit Mode on a hidden
    # object is a silent no-op.
    for v in mesh_obj.data.vertices:
        v.co = v.co * 1.3
    mesh_obj.data.update()
    assert preview.is_preview_stale(mesh_obj, prefix), "resizing the base mesh should invalidate the cached cut"

    assert bpy.ops.seams_to_fur.clear_preview(kind="CUT") == {"FINISHED"}
    assert not preview.is_preview_cached(mesh_obj, prefix)
    assert bpy.data.objects.get(f"{mesh_obj.name}.cut") is None

    # toggle/clear on an uncached preview are safe no-ops, not errors.
    assert bpy.ops.seams_to_fur.toggle_preview(kind="CUT") == {"CANCELLED"}
    assert bpy.ops.seams_to_fur.clear_preview(kind="CUT") == {"FINISHED"}


@test("sync_piece_colors: two MIRRORED pieces with different colors get different, non-bled materials on the cut mesh")
def test_sync_piece_colors_mirrored():
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()

    # Two different colors on the two mirror-image pieces.
    mesh_obj.seams_to_fur_pieces[0].color = (1.0, 0.0, 0.0, 1.0)
    mesh_obj.seams_to_fur_pieces[1].color = (0.0, 1.0, 0.0, 1.0)

    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}

    from bl_ext.user_default.seams_to_fur.operators import preview

    cut_obj = bpy.data.objects.get(f"{mesh_obj.name}.cut")
    assert cut_obj is not None, "sync_piece_colors did not build the exact cut mesh"
    cut_mesh = cut_obj.data

    assert len(cut_mesh.materials) == 2, "expected one material slot per piece on the cut mesh"
    slot_colors = [tuple(m.diffuse_color) for m in cut_mesh.materials]
    assert slot_colors[0] != slot_colors[1], f"the two piece materials are not distinct: {slot_colors}"

    # Each mirror half (x>0 vs x<0) must be entirely on ONE slot, and the two
    # halves must be on DIFFERENT slots - i.e. no bleed, and the two mirrored
    # pieces really do carry their two different colors.
    slots_by_side = {"right": set(), "left": set()}
    for poly in cut_mesh.polygons:
        cx = sum(cut_mesh.vertices[i].co.x for i in poly.vertices) / len(poly.vertices)
        slots_by_side["right" if cx > 0 else "left"].add(poly.material_index)

    assert len(slots_by_side["right"]) == 1, f"right half bled across slots: {slots_by_side['right']}"
    assert len(slots_by_side["left"]) == 1, f"left half bled across slots: {slots_by_side['left']}"
    assert slots_by_side["right"] != slots_by_side["left"], (
        "both mirror halves collapsed onto the SAME material - the mirrored-coloring bug is NOT fixed"
    )

    right_color = slot_colors[next(iter(slots_by_side["right"]))]
    left_color = slot_colors[next(iter(slots_by_side["left"]))]
    assert right_color != left_color, (
        f"mirror halves ended up the same color ({right_color}) despite different piece.color values"
    )

    # The base (pre-mirror) mesh only has one half of the faces and can't
    # represent both mirrored pieces at all, so it must never be colored.
    assert len(mesh_obj.data.materials) == 0, "base source mesh must never be colored"


def _piece_by_uuid(mesh_obj, piece_uuid):
    return next(p for p in mesh_obj.seams_to_fur_pieces if p.uuid == piece_uuid)


@test("seams_to_fur_auto_color toggle assigns distinct colors and overrides manual edits while on")
def test_auto_color_toggle():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()

    # Start every piece on the same default color, so any post-condition of
    # "distinct colors" can only be explained by the auto-color toggle.
    for piece in mesh_obj.seams_to_fur_pieces:
        piece.color = (0.8, 0.8, 0.8, 1.0)

    # Turning the toggle on immediately auto-assigns and syncs (the
    # update callback calls sync_piece_colors itself - no separate button).
    mesh_obj.seams_to_fur_auto_color = True

    # Track one piece by UUID, not index - a sync can internally re-run
    # compute_islands (island_cache miss), which re-syncs (clears +
    # rebuilds) seams_to_fur_pieces and doesn't guarantee index order is
    # preserved (only piece identity, via UUID, is) - see islands.py's
    # nearest-centroid re-matching.
    tracked_uuid = mesh_obj.seams_to_fur_pieces[0].uuid
    piece_colors = {p.uuid: tuple(p.color) for p in mesh_obj.seams_to_fur_pieces}
    assert len(set(piece_colors.values())) == len(piece_colors), (
        f"auto-color toggle did not give every piece a distinct color: {piece_colors}"
    )
    for color in piece_colors.values():
        r, g, b, a = color
        assert max(r, g, b) - min(r, g, b) > 0.05, f"color looks unsaturated/gray: {color}"
        assert a == 1.0

    cut_obj = bpy.data.objects.get(f"{mesh_obj.name}.cut")
    assert cut_obj is not None, "turning on auto-color did not trigger the sync/cut-mesh rebuild"
    cut_mats = [tuple(m.diffuse_color) for m in cut_obj.data.materials]
    assert set(cut_mats) == set(piece_colors.values()), (
        f"cut mesh materials {cut_mats} don't reflect the auto-assigned piece colors {piece_colors}"
    )

    # While the toggle stays on, a manual edit gets overridden the next
    # time colors are (re)synced - that's the whole point of "override".
    _piece_by_uuid(mesh_obj, tracked_uuid).color = (0.0, 0.0, 0.0, 1.0)
    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
    assert tuple(_piece_by_uuid(mesh_obj, tracked_uuid).color) == piece_colors[tracked_uuid], (
        "sync_piece_colors should re-apply the auto-assigned color while seams_to_fur_auto_color is on"
    )

    # Turning it back off leaves colors exactly where they are (no forced
    # revert to some prior manual value) and manual edits stick again.
    mesh_obj.seams_to_fur_auto_color = False
    _piece_by_uuid(mesh_obj, tracked_uuid).color = (0.3, 0.6, 0.9, 1.0)
    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
    # A float64 literal here doesn't exactly equal what's read back - the
    # property is stored as float32, so 0.3 round-trips as
    # 0.30000001192092896 - approximate, not exact, equality is needed.
    final_color = tuple(_piece_by_uuid(mesh_obj, tracked_uuid).color)
    assert all(final_color[i] == pytest_approx(v) for i, v in enumerate((0.3, 0.6, 0.9, 1.0))), (
        f"manual color edits should stick once auto-color is off, got {final_color}"
    )


@test("sync_piece_colors reuses the cached cut after a bake instead of re-cutting (Bug 1)")
def test_sync_piece_colors_reuses_cache():
    _clean_scene()

    # A cube with an inset top-face seam loop: two islands, and - crucially -
    # a real seam curve, so the cut pipeline (resolve_curve_seam_edges) truly
    # runs and is worth caching. A bare plane has no seam curve to resolve.
    mesh_obj = _cube_with_seam_loop()

    from bl_ext.user_default.seams_to_fur.geometry import island_cache, islands

    # Count real cut runs by wrapping the (shared) module-level function that
    # every caller - including flatten.compute_islands, which the color sync
    # now routes through on a cache miss - resolves off this module.
    calls = {"n": 0}
    original = islands.resolve_curve_seam_edges

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    islands.resolve_curve_seam_edges = counting
    try:
        # flatten_all already populated the cache, AND (via sync_piece_colors
        # being called at least once elsewhere in this run) the cut-mesh
        # object may or may not exist yet for *this* mesh_obj - build it
        # once up front so the cache-hit path below has nothing left to do.
        assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
        calls["n"] = 0

        # With nothing seam-relevant changed and the cut-mesh object already
        # present, coloring by piece again must NOT re-run the expensive cut.
        assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
        assert calls["n"] == 0, (
            f"sync_piece_colors re-ran the full cut {calls['n']}x despite an "
            f"unchanged mesh, seam curve, and already-built cut mesh - the "
            f"cache isn't being reused"
        )
        # A second identical sync (e.g. after only a color tweak) is free too.
        assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
        assert calls["n"] == 0

        # Sanity that the counter and the cache actually gate the cut: with a
        # cold cache, the very same sync must fall back to a real cut.
        island_cache.clear()
        calls["n"] = 0
        assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}
        assert calls["n"] >= 1, "with a cold cache, sync_piece_colors must run the cut once"
    finally:
        islands.resolve_curve_seam_edges = original


@test("sync_piece_colors: a cache-miss rebuild of the Sliced Preview objects doesn't leave already-applied fur broken")
def test_sync_piece_colors_cache_miss_preserves_fur():
    # sync_piece_colors's cache-miss path rebuilds each Sliced Preview
    # object's mesh data from scratch (build_sliced_preview_objects calls
    # mesh.clear_geometry() then from_pydata) - confirmed live on a real
    # project file that this wipes the fur UV layer and collapses the
    # material slots back to just the piece material, while the fur
    # particle-system modifier itself (an object-level, not mesh-data,
    # property) survives untouched - now silently pointing at a UV layer
    # and material slot that no longer exist. This looked like "assigning
    # a color breaks the fur/grain direction on every piece" even though
    # the underlying piece.grain_direction data was untouched.
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    for piece in mesh_obj.seams_to_fur_pieces:
        piece.grain_direction = (0.0, 1.0, 0.0)
        piece.grain_anchor = (0.0, 0.0, 0.5)
        piece.has_grain_direction = True
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}

    from bl_ext.user_default.seams_to_fur.operators import appearance
    from bl_ext.user_default.seams_to_fur.geometry import island_cache

    grain_before = {p.uuid: tuple(p.grain_direction) for p in mesh_obj.seams_to_fur_pieces}

    island_cache.clear()  # force the next sync_piece_colors onto the rebuild path
    assert bpy.ops.seams_to_fur.sync_piece_colors() == {"FINISHED"}

    grain_after = {p.uuid: tuple(p.grain_direction) for p in mesh_obj.seams_to_fur_pieces}
    assert grain_before == grain_after, "the rebuild's own UUID carryover must still preserve grain_direction"

    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        uv = sliced_obj.data.uv_layers.get(appearance.FUR_UV_LAYER)
        assert uv is not None, f"{piece.name}: fur UV layer must survive (or be rebuilt after) a Sliced Preview rebuild"
        assert len(sliced_obj.particle_systems) == 1
        settings = sliced_obj.particle_systems[0].settings
        assert settings.tangent_factor > 0.0, f"{piece.name}: should still be combed via tangent_factor, not reset to normal-only"
        mat_name = settings.material_slot
        assert sliced_obj.data.materials.get(mat_name) is not None, (
            f"{piece.name}: fur particle system's material_slot must reference a real material slot on the rebuilt mesh"
        )


@test("set_fur_length: quick-set writes to the active piece's fur_length when nothing is selected in the viewport")
def test_set_fur_length():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    mesh_obj.seams_to_fur_active_piece_index = 0
    bpy.ops.object.select_all(action="DESELECT")

    assert bpy.ops.seams_to_fur.set_fur_length(length_m=0.0508) == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]
    assert abs(piece.fur_length - 0.0508) < 1e-6


@test("set_fur_length: quick-set applies to every piece whose object is selected in the viewport, not just the active one")
def test_set_fur_length_viewport_selection():
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()
    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}

    bpy.ops.object.select_all(action="DESELECT")
    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        sliced_obj.select_set(True)
    mesh_obj.seams_to_fur_active_piece_index = 0  # only the active piece should NOT matter here

    assert bpy.ops.seams_to_fur.set_fur_length(length_m=0.0762) == {"FINISHED"}
    for piece in mesh_obj.seams_to_fur_pieces:
        assert abs(piece.fur_length - 0.0762) < 1e-6, (
            f"{piece.name} should have gotten the quick-set length too, not just the active piece"
        )


@test("apply_swatch_color: applies to the active piece when nothing is selected in the viewport")
def test_apply_swatch_color_active_piece():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    mesh_obj.seams_to_fur_active_piece_index = 0
    bpy.ops.object.select_all(action="DESELECT")

    assert bpy.ops.seams_to_fur.apply_swatch_color(color=(0.2, 0.5, 0.9)) == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]
    assert abs(piece.color[0] - 0.2) < 1e-5
    assert abs(piece.color[1] - 0.5) < 1e-5
    assert abs(piece.color[2] - 0.9) < 1e-5


@test("apply_swatch_color: applies to every piece whose object is selected in the viewport, turning off auto-color")
def test_apply_swatch_color_viewport_selection():
    _clean_scene()
    mesh_obj = _mirrored_plane_two_pieces()
    mesh_obj.seams_to_fur_auto_color = True
    assert bpy.ops.seams_to_fur.refresh_preview(kind="SLICED") == {"FINISHED"}

    bpy.ops.object.select_all(action="DESELECT")
    for piece in mesh_obj.seams_to_fur_pieces:
        sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
        sliced_obj.select_set(True)

    assert bpy.ops.seams_to_fur.apply_swatch_color(color=(0.15, 0.85, 0.35)) == {"FINISHED"}
    assert mesh_obj.seams_to_fur_auto_color is False, (
        "applying a swatch must turn auto-color off, or it would immediately overwrite the just-applied color"
    )
    for piece in mesh_obj.seams_to_fur_pieces:
        assert abs(piece.color[0] - 0.15) < 1e-5
        assert abs(piece.color[1] - 0.85) < 1e-5
        assert abs(piece.color[2] - 0.35) < 1e-5


@test("apply_swatch_color: reports an error and applies nothing when no piece can be resolved")
def test_apply_swatch_color_no_selection():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    mesh_obj.seams_to_fur_active_piece_index = -1
    bpy.ops.object.select_all(action="DESELECT")

    try:
        result = bpy.ops.seams_to_fur.apply_swatch_color(color=(0.2, 0.5, 0.9))
    except RuntimeError:
        return  # self.report({'ERROR'}, ...) + CANCELLED raises - acceptable rejection path
    assert result == {"CANCELLED"}, f"expected CANCELLED with no piece resolvable, got {result}"


@test("live fur refresh: editing piece.color while fur is shown updates the fur material immediately")
def test_live_refresh_color():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    from bl_ext.user_default.seams_to_fur.operators import appearance

    mat = bpy.data.materials.get(f"{appearance.FUR_MATERIAL_NAME} {piece.uuid[:8]}")
    bsdf = mat.node_tree.nodes.get("Hair BSDF")

    # A plain property assignment (what the color-picker widget does) must
    # trigger the update callback - no explicit refresh_fur_preview call here.
    piece.color = (0.2, 0.4, 0.9, 1.0)
    assert abs(bsdf.inputs["Color"].default_value[0] - 0.2) < 1e-5
    assert abs(bsdf.inputs["Color"].default_value[1] - 0.4) < 1e-5
    assert abs(bsdf.inputs["Color"].default_value[2] - 0.9) < 1e-5

    # No particle-system settings should be touched by a color-only change.
    settings = sliced_obj.particle_systems[0].settings
    assert settings.count > 0


@test("live fur refresh: editing piece.fur_length while fur is shown rescales normal/tangent factor immediately, without recalibrating alignment")
def test_live_refresh_fur_length():
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    for piece in mesh_obj.seams_to_fur_pieces:
        piece.grain_direction = (0.0, 1.0, 0.0)
        piece.grain_anchor = (0.0, 0.0, 0.5)
        piece.has_grain_direction = True
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}

    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    settings = sliced_obj.particle_systems[0].settings
    uv_before = [tuple(item.uv) for item in sliced_obj.data.uv_layers[0].data]

    piece.fur_length = 0.1
    settings = sliced_obj.particle_systems[0].settings  # re-fetch defensively
    import math

    magnitude = math.sqrt(settings.normal_factor**2 + settings.tangent_factor**2)
    actual_length = settings.hair_step * magnitude
    assert abs(actual_length - 0.1) < 1e-5, (
        f"expected rescaled length 0.1, got {actual_length}"
    )

    # The UV data (which encodes the comb angle) must be untouched by a
    # length-only change - a full re-sync would rebuild it, but that's
    # deliberately skipped here for live-drag performance.
    uv_after = [tuple(item.uv) for item in sliced_obj.data.uv_layers[0].data]
    assert uv_before == uv_after, "fur_length-only refresh must not rebuild the UV layer"


@test("live fur refresh: committing a grain direction while fur is shown fully re-syncs that piece's fur")
def test_live_refresh_grain_direction_commit():
    # Exercises exactly what SEAMS_TO_FUR_OT_set_grain_direction._apply's commit
    # branch does (property writes + refresh_piece_fur_full), without
    # routing through the full modal operator (which also needs a bound
    # _place_flat_preview/instance context irrelevant to this refresh
    # behavior - see set_grain_direction's own tests for that part).
    _clean_scene()
    mesh_obj = _cube_with_seam_loop()
    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}

    piece = mesh_obj.seams_to_fur_pieces[0]
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    settings_before = sliced_obj.particle_systems[0].settings
    assert settings_before.tangent_factor == 0.0, "no grain direction yet - should still be normal-only"

    from bl_ext.user_default.seams_to_fur.operators import appearance

    piece.grain_anchor = (0.0, 0.0, 0.5)
    piece.grain_direction = (0.0, 1.0, 0.0)
    piece.has_grain_direction = True
    appearance.refresh_piece_fur_full(bpy.context, mesh_obj, piece)

    piece = next(p for p in mesh_obj.seams_to_fur_pieces if p.uuid == piece.uuid)  # re-fetch, refresh may re-sync pieces
    sliced_obj = bpy.data.objects[f"{mesh_obj.name}.{piece.name}.sliced"]
    settings_after = sliced_obj.particle_systems[0].settings
    assert settings_after.tangent_factor > 0.0, "grain-direction commit must live-refresh the fur to comb, not stay normal-only"


@test("live fur refresh hooks are safe no-ops when the fur preview isn't currently shown")
def test_live_refresh_noop_when_fur_not_shown():
    _clean_scene()
    mesh_obj = _flat_plane_with_piece()
    piece = mesh_obj.seams_to_fur_pieces[0]
    # No refresh_fur_preview call - no fur modifier exists on the sliced object (if any).
    piece.color = (0.3, 0.6, 0.1, 1.0)
    piece.fur_length = 0.05
    from bl_ext.user_default.seams_to_fur.operators import appearance

    appearance.refresh_piece_fur_full(bpy.context, mesh_obj, piece)  # must not raise


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
