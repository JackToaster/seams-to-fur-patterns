---
name: seams-to-fur
description: Use when working with the Seams to Fur Patterns Blender extension (seams_to_fur, repo ~/Projects/seams-to-fur-patterns) - drawing/scripting seam curves, flattening a mesh into sewing-pattern pieces, setting per-piece fur color/length/grain, the fur/sliced/distortion previews, exporting SVG/DXF patterns (incl. for LightBurn laser cutting), regenerating the example project, or developing/testing the extension itself. Covers the scripting API, the export format, known cutting limitations, and the dev/test loop. Pair with the `blender` skill for live-session MCP mechanics.
---

# Seams to Fur Patterns

Blender 5.x extension that turns a 3D model into flat, laser-cuttable
sewing patterns, aimed at faux-fur plush/costume work. Repo:
`~/Projects/seams-to-fur-patterns` (GitHub `JackToaster/seams-to-fur-patterns`,
branch `main`). Package dir: `seams_to_fur/`; installed module name:
`bl_ext.user_default.seams_to_fur`. UI: 3D Viewport sidebar (N) >
**Seams to Fur** tab ("Seams to Fur Patterns" + "Appearance (Color / Fur)"
panels). Read `ARCHITECTURE.md` in the repo for the full pipeline design.

For anything involving the *live* Blender session (MCP calls, reloading the
add-on, screenshots), also follow the `blender` skill - its gotchas
(lazy evaluated data, stale UI redraw, module caching on reload) all apply.

## Pipeline in one line

Seam curves (plain verts+edges mesh objects in "STF Seam Curves") -> cut the
evaluated source mesh -> flood-fill into islands (one per pattern **piece**)
-> flatten each with BFF (external `bff-command-line` subprocess, blocking,
~45s for a ~27-piece model) -> one flat mesh object per piece at real-world mm scale
-> color/grain/fur/export all attach to pieces by a stable `uuid`.

## Data model (on the source mesh object)

- `obj.seams_to_fur_pieces` - CollectionProperty of pieces. Per piece:
  `name` ("Piece N"), `uuid`, `piece_id` (island index - NOT stable across
  re-bakes; use `uuid`), `sample_point` (3D centroid), `offset_mm` (per-piece
  seam allowance; 0 = none), `color` (RGBA), `fur_length` (**meters**, e.g.
  1/4" = 0.00635), `grain_direction` + `grain_anchor` (3D vectors on the
  curved surface) + `has_grain_direction`, `flattened_object` /
  `cut_line_object` (object names), `flatten_dirty`, `error_message`,
  `seam_partners` (which piece each edge sews to - drives notches/labels).
- Object-level: `seams_to_fur_active_piece_index`, `seams_to_fur_thickness_mm`
  (shells outward before cutting, for foam), `seams_to_fur_auto_color`,
  `seams_to_fur_placement_mode` (`GRID` | `ORIGIN`),
  `seams_to_fur_grain_mirror_edit`.
- Operators resolve their target via `operators/target.resolve_mesh_obj`:
  the active object, or the source mesh a selected preview/flat piece/seam
  curve points back to. **Make the source mesh active** before calling ops
  from a script, or poll fails ("context is incorrect").
- Piece identity survives re-bakes by nearest-centroid matching, so settings
  stick to "the same" piece after a seam edit. `flatten_dirty` is only set
  when a piece's cut geometry actually changes (fingerprinted at bake time).

## Operators (`bpy.ops.seams_to_fur.*`)

| Op | Notes |
|---|---|
| `draw_seam_curve` | Interactive modal (click points on the surface; snaps to mesh boundary / other seams / mirror plane). Not scriptable - use `create_seam_curve_object` below. |
| `mirror_seam_curve`, `refresh_seam_display` | Mirror-modifier seams; force tube display rebuild. |
| `flatten_all`, `flatten_piece` | Re-bake dirty pieces / the active piece. Raises RuntimeError with per-piece reasons on topology failure. |
| `reset_placement` | Undo manual moves of a flat piece. |
| `refresh_preview` / `toggle_preview` / `clear_preview` | `kind=` `CUT` \| `SLICED` \| `DISTORTION`. |
| `apply_swatch_color`, `sync_piece_colors`, `set_fur_length(length_m=)` | Act on the active piece, or every piece whose object is selected. |
| `set_grain_direction` (modal drag), `reset_grain_direction`, `reset_all_grain_directions`, `toggle_grain_arrows` | |
| `refresh_fur_preview(density=4000)` / `toggle_fur_preview` / `clear_fur_preview` | Builds the Sliced preview + a hair particle system per piece. Refresh always ends visible. |
| `export_svg(filepath=, seam_allowance_mm=)`, `export_dxf(...)` | `seam_allowance_mm` only fills in pieces whose own `offset_mm` is 0 - never overrides a set one. Needs METRIC (or NONE) scene units. |

## Scripting a pattern (headless or via MCP)

Canonical, working end-to-end reference: `examples/build_example.py` in the
repo (builds `examples/bee_plush.blend` + its SVG). Core moves:

```python
from bl_ext.user_default.seams_to_fur.operators import seam_curve as sc

# Simple path (closed=True for a loop):
sc.create_seam_curve_object(ctx, body, world_points, closed=True, name="Seam")
# Branching network - crossing/meeting seams MUST share vertices in ONE skeleton:
sc.create_seam_curve_object(ctx, body, points, closed=False, name="Seams", edges=[(i, j), ...])

ctx.view_layer.objects.active = body
bpy.ops.seams_to_fur.flatten_all()
for p in body.seams_to_fur_pieces:
    p.color = (0.85, 0.66, 0.20, 1.0)
    p.fur_length = 0.0254                 # meters
    p.grain_anchor = tuple(p.sample_point)
    p.grain_direction = (0, 0, -1)        # fur lies toward -Z (real nap polarity!)
    p.has_grain_direction = True
bpy.ops.seams_to_fur.export_svg(filepath="out.svg", seam_allowance_mm=5.0)
```

Points must lie on (or very near) the mesh surface - they're resampled and
projected, so exact point positions/shared points between *separate* seam
objects do NOT survive.

## Cutting limitations (learned the hard way)

Every piece must flatten as a **topological disc**. Errors like "Island is
not a single disk (Euler characteristic 0)", "has no boundary", or
"non-manifold edge (shared by N faces)" mean the seam layout, not BFF:

1. **Tubes need two lengthwise seams, not one.** A single seam slitting a
   band open (both ends on other seams) leaves it a ring - only free-tipped
   darts get split vertices. Split tubes into gores (front + back halves).
2. **Crossing seams must be one skeleton** sharing the crossing vertex
   (`edges=`). Crossings between separate seam objects can leave a band
   uncut.
3. **Don't lay a seam exactly along existing mesh edges** (e.g. a UV
   sphere's equator ring or a meridian) - produces non-manifold pieces.
   Nudge it off-axis, or use an icosphere / irregular mesh.
4. A closed mesh with no seams = "no boundary". Tiny sliver islands are
   folded into neighbors automatically; pieces < 100 mm^2 are dropped from
   export (`MIN_PIECE_AREA_MM2`).

## Export format (SVG built for LightBurn)

LightBurn ignores SVG/Inkscape layers and assigns cut layers by **exact
stroke color**, imports `<g>` as draggable groups, and can't import
`<text>`/`<textPath>`. So `export/svg.py` emits:

- One `<g id="piece-N">` per piece holding *all* its elements (drags as a
  unit); fur-type group boxes in a separate `<g id="fur-type-groups">`.
- Palette colors: `#000000` (00) cut line + notch slits, `#0000FF` (01) sew
  line, `#FF0000` (02) grain arrow + all text, `#808080` (16) group boxes
  (user disables output). A piece with no allowance draws its outline black.
- Text as single-stroke vector paths (`export/stroke_font.py`, uppercase,
  distinct 6/9, slashed 0). Seam-partner numbers always read upright and
  are underlined.
- Notches = slits from the sew line out to the cut line (`export/notch.py`),
  cut in the same pass.
- Layout: pieces rotated so grain points straight down (true nap kept),
  grouped into rows by fur type (color + length), corners sharper than 60
  degrees beveled ("mitred") on the seam allowance. Real nesting is manual.

DXF (`export/dxf.py`) uses AAMA-style layers SEWLINE/CUTLINE/GRAINLINE/
NOTCHES/LABELS with real TEXT entities.

Verify SVG output visually with **headless Chromium**
(`chromium --headless --disable-gpu --no-sandbox --screenshot=out.png
--window-size=W,H file:///abs/path.svg`) - not `rsvg-convert`. For close-ups,
rewrite the root `viewBox` and use *pixel* width/height (Chromium renders mm
at ~3.78 px/mm, so mm-sized crops come out tiny).

## Developing / testing the extension

```bash
cd ~/Projects/seams-to-fur-patterns
python3 -m pytest tests/unit -q                       # bpy-free geometry/export
rm -f /tmp/seams_to_fur-0.1.0.zip
blender --command extension build --source-dir seams_to_fur --output-dir /tmp
blender --command extension install-file --repo user_default --enable /tmp/seams_to_fur-0.1.0.zip
for t in test_flatten_pipeline test_appearance test_export; do
  blender --background --factory-startup --python tests/headless/$t.py 2>&1 | grep -E "^FAIL|tests passed"
done
```

- Headless suites test the **installed** copy - always rebuild + reinstall
  first, or you're testing stale code. "Error: ..." lines in their output
  are expected messages from tests exercising error paths; only `FAIL`
  lines matter.
- bpy-free modules (`geometry/*`, `export/*`) get real pytest coverage; keep
  new pure logic bpy-free. Inside `export/`, import siblings relatively
  (`from . import notch`) - unit tests import `export` as a top-level
  package, so `..geometry` imports break them.
- Live session: after reinstalling, purge `bl_ext.user_default.seams_to_fur*`
  from `sys.modules` and re-enable (see the `blender` skill) or the old code
  keeps running.
- Regenerate the example after behavior changes:
  `blender --background --factory-startup --python examples/build_example.py`
  (asserts no piece is left dirty; commit the .blend + SVG it writes).
- `.gitignore` excludes the machine-specific `seams_to_fur/backends/bin/*/bff-command-line`
  binary - never commit it.
- Determinism matters: the cutter must produce identical geometry for
  identical input (dirty tracking fingerprints it). Never iterate a Python
  `set` of BMesh elements where order affects the result - use
  `dict.fromkeys(...)` / sorted lists.
