"""Per-piece pattern data model.

Phase 1 only reads/writes `piece_id`, `uuid`, `name`, `offset_mm`, and
`flatten_dirty`. The remaining fields are reserved now so Phase 2 (cloth-sim
pre-alignment, fabric/fur grain direction, per-piece coloring, per-edge seam
allowance, DXF/STL export layers) can consume them without a data migration.
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


class SeamsToFurEdgeAllowance(PropertyGroup):
    """Reserved for Phase 2 per-edge seam allowance. Unused in Phase 1."""

    boundary_index: IntProperty(
        name="Boundary Segment Index",
        description="Index of the boundary edge segment this allowance applies to",
    )
    allowance_mm: FloatProperty(
        name="Allowance (mm)",
        default=0.0,
    )


class SeamsToFurSeamPartnerPoint(PropertyGroup):
    """One point along a shared-seam run - see SeamsToFurSeamPartner."""

    co: FloatVectorProperty(size=3)


class SeamsToFurSeamPartner(PropertyGroup):
    """One contiguous run of the resolved cut that separates this piece from
    a specific other piece - i.e. a boundary the two pieces actually get
    sewn together along. Computed by geometry.islands.find_seam_partners
    from the same evaluated-mesh cut every rebake already produces, and
    consumed by the fabrication-output exporters for notch placement.

    `points` are ordered 3D positions in the *source mesh's own local
    space* (not this piece's flattened space) - a raw boundary-loop index
    doesn't correspond 1:1 to a position in either piece's own flattened
    output, since split_island_for_flatten duplicates vertices along
    internal wedge boundaries. Storing real positions and matching them
    against each flattened piece's own "seams_to_fur_orig_co" attribute (the same
    3D<->2D correspondence the grain-direction arrows already use) is the
    robust way to find the corresponding point post-flatten, on either
    side, regardless of any duplication.
    """

    partner_piece_uuid: StringProperty(name="Partner Piece UUID")
    points: CollectionProperty(type=SeamsToFurSeamPartnerPoint)


class SeamsToFurPieceSettings(PropertyGroup):
    """One entry per pattern piece (mesh island) on the source object."""

    piece_id: IntProperty(
        name="Piece ID",
        description="Matches the seams_to_fur_piece_id face attribute value on the source mesh",
    )
    uuid: StringProperty(
        name="UUID",
        description="Stable identity across re-bakes, independent of piece_id reassignment",
    )
    name: StringProperty(
        name="Name",
        default="Piece",
    )
    offset_mm: FloatProperty(
        name="Offset (mm)",
        description="Grow (positive) or shrink (negative) the flattened boundary",
        default=0.0,
        soft_min=-50.0,
        soft_max=50.0,
        unit="NONE",
    )
    flatten_dirty: BoolProperty(
        name="Needs Re-bake",
        default=True,
    )
    # Fingerprints of this piece's cut island geometry (see
    # geometry.islands._island_hash): island_hash is the latest one
    # computed, baked_island_hash the one its flattened object was last
    # baked from. A differing pair is what marks the piece dirty when
    # islands are recomputed - not merely the recompute itself.
    island_hash: StringProperty(options={"HIDDEN"})
    baked_island_hash: StringProperty(options={"HIDDEN"})
    flattened_object: StringProperty(
        name="Flattened Object",
        description="Name of the flat mesh Object produced by the last bake, if any",
    )
    cut_line_object: StringProperty(
        name="Cut Line Object",
        description="Name of the generated offset-boundary Curve object, if offset_mm != 0",
    )
    error_message: StringProperty(
        name="Error",
        description="Set when the last flatten attempt for this piece failed; cleared on success",
    )
    sample_point: FloatVectorProperty(
        name="Sample Point",
        description=(
            "Local-space centroid of one face in this piece from the last "
            "bake, used to re-identify the same piece across re-bakes since "
            "modifiers (e.g. Subdivision) mean face indices aren't stable "
            "between evaluations"
        ),
        size=3,
    )

    # --- Phase 2 ---
    def _update_color(self, context):
        # Cheap live feedback only (flattened object's viewport color, and -
        # if a fur preview is currently shown for this piece - its fur
        # material's color) so dragging the color picker stays responsive;
        # the full per-piece material rebuild on the source mesh is
        # explicit (seams_to_fur.sync_piece_colors) since it's too heavy to run on
        # every color-drag tick.
        flat_obj = bpy.data.objects.get(self.flattened_object) if self.flattened_object else None
        if flat_obj is not None:
            flat_obj.color = self.color
        from .operators import appearance

        appearance.refresh_piece_fur_color(self.id_data, self)

    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8, 1.0),
        update=_update_color,
    )
    grain_direction: FloatVectorProperty(
        name="Grain Direction",
        description="Local-space fabric/fur grain direction, tangent to the surface at grain_anchor",
        size=3,
        default=(1.0, 0.0, 0.0),
    )
    grain_anchor: FloatVectorProperty(
        name="Grain Anchor",
        description=(
            "Local-space point on the piece's surface where the grain "
            "direction was clicked/dragged from - the arrow starts here, "
            "not at the piece's median point, so it stays readable on "
            "curved pieces"
        ),
        size=3,
    )
    has_grain_direction: BoolProperty(
        name="Has Grain Direction",
        description="Whether grain_direction/grain_anchor have actually been set (vs. still at their defaults)",
        default=False,
    )
    def _update_fur_length(self, context):
        # Cheap live feedback: rescales an already-shown fur preview's
        # length in place (no UV/alignment rebuild - see
        # appearance.refresh_piece_fur_length for why that's both safe and
        # necessary here), so this stays responsive on every tick of a
        # live fur_length slider drag.
        from .operators import appearance

        appearance.refresh_piece_fur_length(self.id_data, self)

    fur_length: FloatProperty(
        name="Fur Length",
        description="Length of this piece's fur strands, in scene units (default 0.025 = 25mm assuming 1 unit = 1 m)",
        default=0.025,
        min=0.0001,
        soft_max=0.3,
        update=_update_fur_length,
    )
    seam_allowance: CollectionProperty(type=SeamsToFurEdgeAllowance)
    seam_partners: CollectionProperty(type=SeamsToFurSeamPartner)


_classes = (
    SeamsToFurEdgeAllowance,
    SeamsToFurSeamPartnerPoint,
    SeamsToFurSeamPartner,
    SeamsToFurPieceSettings,
)


def _update_auto_color(self, context):
    # Deferred import - operators/appearance.py imports this module (via
    # bpy registration order), so importing it back at module load time
    # here would be circular; safe once this callback actually fires.
    if self.seams_to_fur_auto_color:
        from .operators import appearance

        appearance.sync_piece_colors(context, self)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.seams_to_fur_pieces = CollectionProperty(type=SeamsToFurPieceSettings)
    bpy.types.Object.seams_to_fur_active_piece_index = IntProperty(default=0)
    bpy.types.Object.seams_to_fur_thickness_mm = FloatProperty(
        name="Material Thickness (mm)",
        description=(
            "Shells the surface outward along its normals by half this "
            "amount before cutting, so the flattened pattern accounts for "
            "material thickness (e.g. foam) rather than just the base "
            "mesh surface"
        ),
        default=0.0,
        min=0.0,
        soft_max=20.0,
    )
    bpy.types.Object.seams_to_fur_grain_mirror_edit = BoolProperty(
        name="Mirror Edit",
        description=(
            "When setting grain direction, also mirror the same direction "
            "onto whichever piece sits at the click point's reflection "
            "across the Mirror modifier plane"
        ),
        default=False,
    )
    bpy.types.Object.seams_to_fur_auto_color = BoolProperty(
        name="Auto Color",
        description=(
            "Keep every piece's color auto-assigned (golden-ratio hue "
            "stepping) instead of picking colors by hand - overrides "
            "whatever color each piece currently has and re-syncs "
            "immediately when turned on"
        ),
        default=False,
        update=_update_auto_color,
    )
    bpy.types.Object.seams_to_fur_show_fur_colors = BoolProperty(
        name="Show Fur Colors",
        description="Show/hide the faux-fur color swatch panel",
        default=False,
    )
    bpy.types.Object.seams_to_fur_placement_mode = EnumProperty(
        name="Placement",
        description="Where newly-flattened pieces are placed",
        items=[
            ("GRID", "Grid Layout", "Non-overlapping grid, for exporting/printing"),
            (
                "ORIGIN",
                "At Original Position",
                "Each piece's center placed where its curved piece's center was in 3D",
            ),
        ],
        default="GRID",
    )


def unregister():
    del bpy.types.Object.seams_to_fur_placement_mode
    del bpy.types.Object.seams_to_fur_show_fur_colors
    del bpy.types.Object.seams_to_fur_grain_mirror_edit
    del bpy.types.Object.seams_to_fur_thickness_mm
    del bpy.types.Object.seams_to_fur_active_piece_index
    del bpy.types.Object.seams_to_fur_pieces

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
