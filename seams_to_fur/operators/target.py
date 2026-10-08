"""Resolves "the Seams to Fur Patterns source mesh the user is currently working on" from
whatever object happens to be active, instead of requiring it to literally
be the source mesh object itself.

Every object Seams to Fur Patterns generates from a source mesh - seam curve skeleton,
cut/sliced/distortion preview, flattened pattern piece - is tagged with a
back-reference to its source. Clicking around the viewport to inspect a
sliced-preview piece or a flattened pattern piece is the natural thing to
do, but each of those is a *different* bpy object from the source mesh, so
resolving mesh_obj from context.active_object directly would silently lose
track of what the user is actually editing the moment they click one -
poll functions fail quietly, panels fall back to "select a mesh object",
and the user has to go re-select the source mesh to get their controls
back. Used by every panel/operator that isn't itself specifically about a
seam curve object (see operators/seam_curve.py, which legitimately wants
the raw active object - a curve skeleton, not its source).
"""

import bpy

from .seam_curve import SOURCE_OBJECT_PROP

PREVIEW_SOURCE_PROP = "seams_to_fur_source_object"
PIECE_UUID_PROP = "seams_to_fur_piece_uuid"


def view_layer_active(context):
    """context.view_layer.objects.active rather than context.active_object -
    confirmed live these two genuinely differ once the active object is
    hidden (e.g. seams_to_fur.refresh_preview/toggle_preview hides the base
    mesh while a same-shape preview is shown): the view layer's raw active
    pointer still holds it, but context.active_object - the convenience
    property real Panel.draw()/Operator.poll() calls actually receive -
    silently goes None for a hidden object. Using the view layer's pointer
    is what keeps the whole panel (and every operator's poll()) working on
    the source mesh while its preview is up, instead of the panel dropping
    to "Select a mesh object" the moment a preview hides it."""
    view_layer = context.view_layer
    return view_layer.objects.active if view_layer is not None else context.active_object


def resolve_mesh_obj(context):
    """Returns the Seams to Fur Patterns source mesh object to act on, or None."""
    obj = view_layer_active(context)
    if obj is None:
        return None
    source_name = obj.get(PREVIEW_SOURCE_PROP) or obj.get(SOURCE_OBJECT_PROP)
    if source_name:
        resolved = bpy.data.objects.get(source_name)
        if resolved is not None and resolved.type == "MESH":
            return resolved
    return obj if obj.type == "MESH" else None


def resolve_piece_index(context, mesh_obj):
    """If the active object is a per-piece Seams to Fur Patterns object (sliced/distortion
    preview, flattened pattern piece - anything tagged with
    PIECE_UUID_PROP, see operators/preview.py, distortion.py, flatten.py),
    returns that piece's index in mesh_obj.seams_to_fur_pieces, so selecting a
    piece in the viewport can drive the Pieces list's active selection.
    Returns None if the active object isn't tagged, or its UUID doesn't
    match any current piece (e.g. a stale object from before a re-bake)."""
    obj = view_layer_active(context)
    if obj is None:
        return None
    piece_uuid = obj.get(PIECE_UUID_PROP)
    if not piece_uuid:
        return None
    for i, piece in enumerate(mesh_obj.seams_to_fur_pieces):
        if piece.uuid == piece_uuid:
            return i
    return None
