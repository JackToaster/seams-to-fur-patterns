"""Builds examples/bee_plush.blend (and its exported pattern) from scratch.

A small, self-contained demo of the whole pipeline: a stylized plush bee
body (an elongated sphere) cut into stripes by seams, flattened, given
per-piece fur colors, fur lengths and grain directions, previewed with the
live fur particle preview, then exported as a LightBurn-ready SVG with a
5mm seam allowance.

Regenerate after changing the add-on (it must be installed and enabled -
see the README):

    blender --background --factory-startup --python examples/build_example.py
"""

import math
import sys
from pathlib import Path

import bpy

ADDON_MODULE = "bl_ext.user_default.seams_to_fur"
HERE = Path(__file__).resolve().parent
BLEND_PATH = HERE / "bee_plush.blend"
SVG_PATH = HERE / "bee_plush_pattern.svg"

# Body: an ellipsoid with a 10cm radius, 36cm long along Z.
RADIUS = 0.10
HALF_LENGTH = 0.18
# Stripe seams, as Z heights of closed loops around the body.
STRIPE_Z = (-0.10, -0.035, 0.035, 0.10)
LOOP_POINTS = 48
GORE_STEPS = 8
# Lengthwise seams run down the back (-Y) and front (+Y), nudged off-axis
# so they don't line up exactly with the base mesh's own edges.
BACK_ANGLE = -math.pi / 2 + 0.07

INCH = 0.0254
BLACK = (0.04, 0.04, 0.04, 1.0)
YELLOW = (0.85, 0.66, 0.20, 1.0)


def _surface_point(z, angle):
    r = RADIUS * math.sqrt(max(0.0, 1.0 - (z / HALF_LENGTH) ** 2))
    return (r * math.cos(angle), r * math.sin(angle), z)


def _seam_skeleton():
    """(points, edges) for one branching seam skeleton: a closed loop at
    each stripe height, plus a front and a back seam running between the
    outermost loops. Each middle band is a tube, which can't be flattened
    until it's split - two lengthwise seams turn each band into a front
    and back half, the way a real plush pattern splits a tube into gores.

    All of it is one skeleton, with the lengthwise seams sharing vertices
    with the loops they cross, so every junction cuts cleanly."""
    points, edges = [], []
    loop_starts = []
    for z in STRIPE_Z:
        start = len(points)
        loop_starts.append(start)
        for i in range(LOOP_POINTS):
            points.append(_surface_point(z, BACK_ANGLE + 2 * math.pi * i / LOOP_POINTS))
            edges.append((start + i, start + (i + 1) % LOOP_POINTS))

    for offset in (0, LOOP_POINTS // 2):  # back, then front
        angle = BACK_ANGLE + 2 * math.pi * offset / LOOP_POINTS
        for band, (z_a, z_b) in enumerate(zip(STRIPE_Z, STRIPE_Z[1:])):
            prev = loop_starts[band] + offset
            for k in range(1, GORE_STEPS):
                points.append(_surface_point(z_a + (z_b - z_a) * k / GORE_STEPS, angle))
                edges.append((prev, len(points) - 1))
                prev = len(points) - 1
            edges.append((prev, loop_starts[band + 1] + offset))
    return points, edges


def main():
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    from bl_ext.user_default.seams_to_fur.operators import seam_curve as seam_curve_ops

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=RADIUS)
    body = bpy.context.active_object
    body.name = "Bee Body"
    body.scale.z = HALF_LENGTH / RADIUS
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()

    points, edges = _seam_skeleton()
    seam_curve_ops.create_seam_curve_object(bpy.context, body, points, closed=False, name="Bee Seams", edges=edges)

    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    assert bpy.ops.seams_to_fur.flatten_all() == {"FINISHED"}

    # Stripes alternate black/yellow from the bottom up; the end caps get
    # shorter fur than the bands, so the export shows several fur types.
    pieces = list(body.seams_to_fur_pieces)
    expected = 2 + 2 * (len(STRIPE_Z) - 1)
    assert len(pieces) == expected, f"expected {expected} pieces, got {len(pieces)}"
    for piece in pieces:
        z = piece.sample_point[2]
        stripe = sum(1 for s in STRIPE_Z if z > s)  # 0 = bottom cap
        is_cap = stripe in (0, len(STRIPE_Z))
        piece.color = BLACK if stripe % 2 == 0 else YELLOW
        piece.fur_length = (0.25 if is_cap else 1.0) * INCH
        # Fur lies down the body, toward the tail end (-Z).
        piece.grain_anchor = tuple(piece.sample_point)
        piece.grain_direction = (0.0, 0.0, -1.0)
        piece.has_grain_direction = True

    assert bpy.ops.seams_to_fur.refresh_fur_preview() == {"FINISHED"}
    dirty = [p.name for p in body.seams_to_fur_pieces if p.flatten_dirty]
    assert not dirty, f"pieces left needing a re-bake after the fur preview: {dirty}"
    # Open on the fur, not a gray solid-shaded mesh.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"

    assert bpy.ops.seams_to_fur.export_svg(filepath=str(SVG_PATH), seam_allowance_mm=5.0) == {"FINISHED"}
    # Drop the factory-startup scene's leftovers (the deleted default
    # cube's mesh, and the material it still references).
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), compress=True)
    print(f"Wrote {BLEND_PATH} and {SVG_PATH} ({len(pieces)} pieces)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
