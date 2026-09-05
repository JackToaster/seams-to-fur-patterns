"""Geometry Nodes setup that makes a seam curve visibly hug the target
surface as a round tube, in the order that actually matters: resample
(subdivide) the raw path first, then project each of those denser points
onto the nearest point of the target mesh, and only then generate a tube
around the result.

This has to be geometry nodes, not classic modifiers: for a Curve object,
the curve's own bevel/thickness is baked into its base mesh *before* the
modifier stack runs, so a classic Shrinkwrap modifier ends up projecting
the already-thick tube's outer surface (squashing it) instead of a thin
centerline. Geometry nodes gives explicit control over the order.

Modifier input sockets (identifiers are stable once created, but re-fetch
by interface name via socket_id_for() rather than hardcoding "Socket_N"):
    Target        (Object)  - the mesh to project onto
    Segment Length (Float)  - resample spacing; smaller = hugs curvature better
    Tube Radius    (Float)  - visual thickness only, doesn't affect resolution
    Bias           (Vector) - curve-local-space nudge applied before sampling
                              the nearest surface point (see below)

Bias exists to fix a real symmetric-surface ambiguity: a point sitting
exactly on a mirrored mesh's seam is equidistant from both mirrored
halves, so "nearest surface" has two equally valid answers and flips
between them from one resampled point to the next based on nothing but
floating-point noise - producing a visibly squiggly/zigzagging tube along
what should be a straight seam line. Nudging the *query* position (not the
result) by a small, consistent offset before the nearest-surface lookup
breaks the tie deterministically; since the nudge is tiny relative to
surface features, the returned point is still effectively on the seam.
"""

import bpy

NODE_GROUP_NAME = "USBee Seam Curve Surface Follow"
MODIFIER_NAME = "USBee Surface Follow"

TUBE_RADIUS = 0.0025 / 3.0


def socket_id_for(node_group, name, in_out="INPUT"):
    for item in node_group.interface.items_tree:
        if item.name == name and item.in_out == in_out:
            return item.identifier
    raise KeyError(f"No {in_out} socket named {name!r} in {node_group.name!r}")


def get_or_create_node_group():
    ng = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(NODE_GROUP_NAME, "GeometryNodeTree")
    iface = ng.interface
    iface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    iface.new_socket(name="Target", in_out="INPUT", socket_type="NodeSocketObject")
    sock_len = iface.new_socket(name="Segment Length", in_out="INPUT", socket_type="NodeSocketFloat")
    sock_len.default_value = 0.02
    sock_len.min_value = 0.0001
    sock_radius = iface.new_socket(name="Tube Radius", in_out="INPUT", socket_type="NodeSocketFloat")
    sock_radius.default_value = TUBE_RADIUS
    sock_radius.min_value = 0.0
    iface.new_socket(name="Bias", in_out="INPUT", socket_type="NodeSocketVector")
    iface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = ng.nodes
    links = ng.links

    n_in = nodes.new("NodeGroupInput")
    n_out = nodes.new("NodeGroupOutput")

    n_resample = nodes.new("GeometryNodeResampleCurve")
    n_resample.inputs["Mode"].default_value = "Length"

    n_objinfo = nodes.new("GeometryNodeObjectInfo")
    n_objinfo.transform_space = "RELATIVE"

    n_sample = nodes.new("GeometryNodeSampleNearestSurface")
    n_sample.data_type = "FLOAT_VECTOR"

    n_pos = nodes.new("GeometryNodeInputPosition")
    n_setpos = nodes.new("GeometryNodeSetPosition")

    n_circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
    n_circle.mode = "RADIUS"

    n_c2m = nodes.new("GeometryNodeCurveToMesh")

    # Sample Position defaults to each point's own (unbiased) position if
    # left unconnected - explicitly add Bias so callers can break the
    # mirror-seam tie described above. Bias is (0,0,0) when not needed, so
    # this is a no-op for non-symmetric curves.
    n_biased_pos = nodes.new("ShaderNodeVectorMath")
    n_biased_pos.operation = "ADD"

    links.new(n_in.outputs["Geometry"], n_resample.inputs["Curve"])
    links.new(n_in.outputs["Segment Length"], n_resample.inputs["Length"])
    links.new(n_in.outputs["Target"], n_objinfo.inputs["Object"])
    links.new(n_objinfo.outputs["Geometry"], n_sample.inputs["Mesh"])
    links.new(n_pos.outputs["Position"], n_sample.inputs["Value"])
    links.new(n_pos.outputs["Position"], n_biased_pos.inputs[0])
    links.new(n_in.outputs["Bias"], n_biased_pos.inputs[1])
    links.new(n_biased_pos.outputs["Vector"], n_sample.inputs["Sample Position"])
    links.new(n_resample.outputs["Curve"], n_setpos.inputs["Geometry"])
    links.new(n_sample.outputs["Value"], n_setpos.inputs["Position"])
    links.new(n_in.outputs["Tube Radius"], n_circle.inputs["Radius"])
    links.new(n_setpos.outputs["Geometry"], n_c2m.inputs["Curve"])
    links.new(n_circle.outputs["Curve"], n_c2m.inputs["Profile Curve"])
    links.new(n_c2m.outputs["Mesh"], n_out.inputs["Geometry"])

    return ng


def add_or_update_modifier(curve_obj, mesh_obj, segment_length, bias_local=(0.0, 0.0, 0.0)):
    ng = get_or_create_node_group()
    mod = curve_obj.modifiers.get(MODIFIER_NAME)
    if mod is None or mod.type != "NODES":
        mod = curve_obj.modifiers.new(name=MODIFIER_NAME, type="NODES")
    mod.node_group = ng

    mod[socket_id_for(ng, "Target")] = mesh_obj
    mod[socket_id_for(ng, "Segment Length")] = segment_length
    mod[socket_id_for(ng, "Tube Radius")] = TUBE_RADIUS
    mod[socket_id_for(ng, "Bias")] = tuple(bias_local)

    # Modifiers a user adds afterward (Array/Mirror to repeat the cut
    # pattern) need to run *before* this one, so each duplicate gets
    # independently resampled and projected onto its own patch of surface -
    # otherwise Array would just rigid-copy the first copy's already-glued
    # shape. Keep this modifier pinned last on every call, since we have no
    # hook into modifiers the user adds through Blender's own UI.
    last_index = len(curve_obj.modifiers) - 1
    current_index = list(curve_obj.modifiers).index(mod)
    if current_index != last_index:
        curve_obj.modifiers.move(current_index, last_index)

    return mod


def set_modifier_enabled(curve_obj, enabled):
    """Used by seam resolution to temporarily hide this modifier's tube
    output so to_mesh() sees the plain (post Array/Mirror) path instead -
    see geometry.islands._extract_curve_chains."""
    mod = curve_obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        return None
    was_enabled = mod.show_viewport
    mod.show_viewport = enabled
    return was_enabled
