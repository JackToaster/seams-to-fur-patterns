"""Shared storage format for a seam curve's authored points: a JSON blob of
world-space [x, y, z] triples plus a closed/open flag, stored as custom
properties directly on the seam curve object - a plain Mesh object, not a
Curve, with no modifier stack involved at all. See geometry.curve_display's
module docstring for why this addon owns seam-curve geometry end to end
instead of using Blender's native Curve type/modifiers/Edit Mode.
"""

import json

from mathutils import Vector

POINTS_PROP = "usbee_points"
CLOSED_PROP = "usbee_closed"


def load_points(curve_obj):
    """Returns (points: list[Vector] in world space, closed: bool)."""
    raw = curve_obj.get(POINTS_PROP)
    points = [Vector(p) for p in json.loads(raw)] if raw else []
    closed = bool(curve_obj.get(CLOSED_PROP, False))
    return points, closed


def store_points(curve_obj, points, closed):
    curve_obj[POINTS_PROP] = json.dumps([[p.x, p.y, p.z] for p in points])
    curve_obj[CLOSED_PROP] = bool(closed)
