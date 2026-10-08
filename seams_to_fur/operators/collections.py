"""One shared top-level collection every generated collection (seam curves,
seam curve tubes, grain-direction arrows, cut/sliced/distortion previews,
flattened pattern pieces) lives under, instead of each being linked
directly to the scene's own root collection - keeps the Outliner from
filling up with a dozen loose top-level collections per source mesh.

Standalone on purpose (no imports from elsewhere in this package) so every
other operator module can import it with zero circular-import risk.
"""

import bpy

ROOT_COLLECTION_NAME = "Seams to Fur Patterns"


def get_or_create_root_collection(context):
    coll = bpy.data.collections.get(ROOT_COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(ROOT_COLLECTION_NAME)
        context.scene.collection.children.link(coll)
    elif coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)
    return coll


def get_or_create_child_collection(context, name):
    """A named collection linked under the shared root (not directly under
    the scene) - the usual way every other module here should create one
    of its own generated collections."""
    root = get_or_create_root_collection(context)
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        root.children.link(coll)
    elif coll.name not in root.children:
        # Migrates a collection created before this shared root existed
        # (or before a name change) - unlink from wherever it currently is
        # first, a collection can't be linked under two parents at once.
        for parent in bpy.data.collections:
            if coll.name in parent.children:
                parent.children.unlink(coll)
        if coll.name in context.scene.collection.children:
            context.scene.collection.children.unlink(coll)
        root.children.link(coll)
    return coll
