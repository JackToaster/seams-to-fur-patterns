# Architecture

This is a Blender 5.x extension (`seams_to_fur/`), a plain Python add-on
package with a `blender_manifest.toml`. Nothing here needs a build step
beyond zipping the package (`blender --command extension build`).

## Pipeline overview

```
Seam curve(s) (edited on the source mesh)
        │
        ▼
geometry/islands.py    — resolve each curve into real mesh cuts, flood-fill
                          the cut mesh into islands (pattern pieces),
                          validate topology, find piece-to-piece adjacency
        │
        ▼
backends/bff.py        — one BFF subprocess call per island: write a temp
                          OBJ, run bff-command-line, parse the flattened
                          OBJ back
        │
        ▼
geometry/layout.py      — non-overlapping shelf placement for freshly-baked
                          pieces (not a real nesting optimizer)
        │
        ▼
One flat mesh Object per piece, real-world mm scale, in a per-source-mesh
Collection — this is the thing everything else (color, grain, fur, export)
attaches itself to via the piece's stable uuid.
```

Flattening is a plain, blocking operator (`seams_to_fur.flatten_all` /
`.flatten_piece`) — BFF runs as an external subprocess with real latency,
so there's no live/interactive flattening. `seams_to_fur.flatten_all` on the
example mesh used throughout development (`hood.001`, ~27 pieces) takes
roughly 45 seconds; that cost is why `geometry/island_cache.py` exists (see
below) — nothing outside an actual re-bake should ever pay it again.

## Data model

`SeamsToFurPieceSettings` (`properties.py`), one entry per piece in
`Object.seams_to_fur_pieces`, is the single source of truth every other
system reads from:

| Field | Purpose |
|---|---|
| `uuid` | Stable identity across re-bakes (see below) - everything else keys off this, not `piece_id` or list index. |
| `piece_id` | The island id from the *current* bake - matches a face attribute on the cut mesh; not stable across bakes. |
| `name`, `offset_mm` | User-facing name and seam-allowance offset. |
| `flattened_object`, `cut_line_object` | Back-references (object names) to the generated flat piece and its offset cut-line curve. |
| `color` | Per-piece color, RGBA. |
| `grain_direction`, `grain_anchor`, `has_grain_direction` | Local-space (on the *curved* source mesh) tangent direction + the surface point it was set from. |
| `fur_length` | Per-piece fur/hair length. |
| `sample_point` | A stable 3D reference point for this island, used only for identity-matching across re-bakes. |
| `seam_partners` | Which *other* pieces this one shares a sewn boundary with, and the actual 3D points along each shared run - see "Notches" below. |

**Piece identity across re-bakes.** Island face indices are *not* stable
(a generative modifier like Subdivision Surface changes them), so identity
is tracked by nearest-3D-centroid matching against each piece's previous
`sample_point` (`geometry/islands.py::sync_piece_settings`), carrying the
UUID (and therefore color/grain/fur_length/offset) forward. This is a
greedy nearest-match, not a globally optimal assignment - good enough for
incremental seam edits, not guaranteed correct after a drastic reshape.

## Module map

- **`geometry/`** - pure(ish) mesh algorithms, kept as bpy-light as
  practical so the non-Blender-specific parts (`boundary.py`,
  `polygon_offset.py`) are plain-`pytest`-testable (`tests/unit/`).
  - `mesh_cut.py` / `islands.py` - seam curve → real mesh bisection → BFS
    flood-fill into islands → topology validation → (new) piece-to-piece
    adjacency (`find_seam_partners`).
  - `curve_display.py` - the always-visible tube mesh that traces a seam
    curve on the surface (a separate display object, not the curve's own
    render - matters if you're chasing a rendering artifact near a seam,
    see `operators/preview.py::set_base_mesh_hidden`, which hides it
    alongside the base mesh whenever a same-shape preview is shown).
  - `layout.py` - shelf placement for new pieces.
  - `polygon_offset.py` - hand-rolled 2D polygon grow/shrink (miter-join
    offset-each-edge-then-reintersect) - deliberately *not* a Clipper2/
    pyclipr binding, to stay a small dependency-free, cross-platform-
    trivial pure-Python module. Known limitation: can produce a
    self-intersecting result on an aggressively concave boundary with a
    large shrink offset (raises `OffsetError` rather than returning
    garbage).
  - `island_cache.py` - session-only cache of the last real cut's
    per-face island membership, keyed by a signature hash of the mesh +
    modifiers + seam curves + material thickness. Lets a color-only sync
    (`sync_piece_colors`) skip a full re-cut entirely unless something
    that actually affects the cut changed. Does not survive a file/addon
    reload - the first post-reload call is just a cache miss, not a bug.
  - `obj_io.py` - minimal OBJ read/write for the BFF subprocess round trip.

- **`backends/bff.py`** - subprocess wrapper around the vendored/fetched
  `bff-command-line` binary.

- **`operators/`** - the actual Blender operators, plus a few modules that
  are really "shared logic with a thin operator wrapper":
  - `seam_curve.py` - seam curve creation/editing/display refresh.
  - `flatten.py` - `compute_islands` (the shared entry point flatten,
    distortion, and any color-sync-on-cache-miss all route through, so
    island numbering never diverges between callers) and the flatten
    operators themselves.
  - `target.py` - resolves "the source mesh the active object belongs to"
    - clicking a flattened piece, a Sliced Preview object, or the base
      mesh itself should all control the same N-panel, so every operator
      polls/executes against `target.resolve_mesh_obj(context)`, not
      `context.active_object` directly.
  - `preview.py` - Cut/Sliced Preview objects (debug/working views of the
    actual cut, not the flattened output), and the same-shape-preview
    mutual-exclusion bookkeeping (`hide_same_shape_previews`,
    `set_base_mesh_hidden`) that Cut/Sliced/Distortion preview all share.
  - `distortion.py` - the in-place distortion preview: colors each piece
    (at its original curved position) by a per-*vertex* (smoothed, not
    flat-per-face - a raw BFF per-face area ratio has a real, confirmed-
    live boundary artifact right at each piece's own seam edge) area-ratio
    heatmap, mapped onto a real ±20%-linear-stretch physical scale via a
    diverging blue/green/red `ColorRamp` the N-panel legend reads directly
    off the material (so the legend can't drift out of sync with what's
    actually rendered).
  - `appearance.py` - the largest module: per-piece color sync, the
    click-drag grain-direction tool (with mirror-piece editing), the
    grain-direction arrow mesh (a real cylinder+cone, not a beveled
    curve - see its docstring for why), and the fur particle-system
    preview. Fur lives on each piece's own **Sliced Preview** object, not
    the base source mesh - the base mesh's own UV map may not exist or be
    usable, a Mirror-modifier piece has no real geometry on the base mesh
    at all, and hair "tangent" direction turns out to read from UV layer
    index 0 specifically, which a fresh Sliced Preview object always
    starts at (no index-swapping needed). The comb direction itself comes
    from mapping the 3D grain direction into the piece's own flattened 2D
    space (`_build_flat_proxy`/`_map_direction_to_flat`, a BVH built over
    the flat mesh but positioned at each vertex's *pre-flatten* 3D
    location, so a 3D point/direction query returns the corresponding
    flat-space point/direction) - the same mapping the exported grainline
    arrow and notches reuse.
  - `export_common.py` - `gather_piece_export_data`, the one place that
    turns baked pieces + the data above into plain mm-space dicts
    (boundary, cut line, grainline span, notch points, nearest fabric
    swatch name) - both exporters consume this, so SVG and DXF can never
    disagree about what a piece's data actually is.
  - `export_svg.py` / `export_dxf.py` - thin operators around
    `export/svg.py` / `export/dxf.py`.

- **`export/`** - bpy-free document builders (plain piece dicts in,
  document text out - directly `pytest`-testable):
  - `svg.py` - reference/cut-line polygons, grainline (a real double-
    headed arrow), notch tick marks, swatch-name label.
  - `dxf.py` - a minimal hand-written ASCII DXF R12 writer (no `ezdxf`
    dependency, same reasoning as `polygon_offset.py`) with
    `SEWLINE`/`CUTLINE`/`GRAINLINE`/`NOTCHES`/`LABELS` layers, following
    the AAMA/ASTM garment-industry convention of organizing layers by
    *purpose*, not per piece.

- **`fur_colors.py`** - the 66-entry faux-fur color palette (sourced from
  a real fabric vendor's physical swatch card - see the module docstring)
  plus `nearest_swatch_name`, used both by the color-picker panel and the
  export's fabric annotation.

- **`ui/panels.py`** - the N-panel. Piece selection syncs both ways
  (clicking a piece's object in the viewport updates the Pieces list, via
  an `bpy.msgbus` subscription on the active object - `Panel.draw()` alone
  is *not* a reliable trigger for reacting to a selection change, a real
  dead end hit during development).

## Notches (piece-to-piece alignment marks)

`geometry/islands.py::find_seam_partners` walks the resolved cut edges a
rebake already computes and groups the ones whose two adjacent faces
belong to *different* islands into contiguous runs per piece-pair - i.e.
exactly the boundaries that get sewn together, as opposed to a piece's own
free edge or an internal dart seam. Stored as literal 3D points (not
boundary-loop indices) on each piece's `seam_partners`, because
`split_island_for_flatten` duplicates vertices along internal wedge
boundaries, so a raw index doesn't survive into either piece's own
flattened output - matching by position against each piece's own
`seams_to_fur_orig_co` proxy (the same 3D↔2D mapping grain direction uses)
is what actually works.

At export time (`export_common.py`), a shared run shorter than 3cm is
skipped entirely; otherwise a notch is placed 1cm inset from each end
(not on the corner itself) plus a midpoint notch if the run is longer
than ~15cm.

## Known rough edges / deliberately deferred

- **Nesting** - `geometry/layout.py` avoids overlap, nothing more; a real
  bin-packing optimizer for material-usage efficiency is out of scope for
  now.
- **Per-edge seam allowance** - `SeamsToFurEdgeAllowance` exists as a
  reserved, currently-unused PropertyGroup; only the simpler uniform
  per-piece `offset_mm` is wired up.
- **BFF binary** - vendored per-platform build only for the platform
  actively developed on; the fetch-on-first-use / cross-platform-CI-build
  strategy for real distribution is future work (see `backends/bff.py`).
