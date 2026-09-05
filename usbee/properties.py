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
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup


class UsbeeEdgeAllowance(PropertyGroup):
    """Reserved for Phase 2 per-edge seam allowance. Unused in Phase 1."""

    boundary_index: IntProperty(
        name="Boundary Segment Index",
        description="Index of the boundary edge segment this allowance applies to",
    )
    allowance_mm: FloatProperty(
        name="Allowance (mm)",
        default=0.0,
    )


class UsbeeSeamPartner(PropertyGroup):
    """Reserved for Phase 2 cloth-sim auto-alignment.

    Captured as a byproduct of island isolation in Phase 1 (which boundary
    segment on this piece is shared with which boundary segment on another
    piece), even though nothing consumes it until Phase 2.
    """

    partner_piece_uuid: StringProperty(name="Partner Piece UUID")
    own_boundary_start: IntProperty()
    own_boundary_end: IntProperty()
    partner_boundary_start: IntProperty()
    partner_boundary_end: IntProperty()


class UsbeePieceSettings(PropertyGroup):
    """One entry per pattern piece (mesh island) on the source object."""

    piece_id: IntProperty(
        name="Piece ID",
        description="Matches the usbee_piece_id face attribute value on the source mesh",
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
    flattened_object: StringProperty(
        name="Flattened Object",
        description="Name of the flat mesh Object produced by the last bake, if any",
    )
    cut_line_object: StringProperty(
        name="Cut Line Object",
        description="Name of the generated offset-boundary Curve object, if offset_mm != 0",
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

    # --- Reserved for Phase 2, unused in Phase 1 ---
    color: FloatVectorProperty(
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8, 1.0),
    )
    grain_direction: FloatVectorProperty(
        name="Grain Direction",
        description="Local-space fabric/fur grain direction, same convention as material_direction",
        size=3,
        default=(1.0, 0.0, 0.0),
    )
    seam_allowance: CollectionProperty(type=UsbeeEdgeAllowance)
    seam_partners: CollectionProperty(type=UsbeeSeamPartner)


_classes = (
    UsbeeEdgeAllowance,
    UsbeeSeamPartner,
    UsbeePieceSettings,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.usbee_pieces = CollectionProperty(type=UsbeePieceSettings)
    bpy.types.Object.usbee_active_piece_index = IntProperty(default=0)


def unregister():
    del bpy.types.Object.usbee_active_piece_index
    del bpy.types.Object.usbee_pieces

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
