"""Session-scoped cache of the expensive cut + flood-fill result, so
appearance/metadata operators feel instant.

"Sync piece colors" (and anything else that only needs to know *which cut-
mesh face belongs to which piece*) used to freeze Blender for many seconds
on a dense mesh, even though nothing about the cut had changed, because it
re-ran the full seam-curve -> mesh cut -> flood-fill pipeline from scratch
(geometry.mesh_cut / geometry.islands) - the pipeline the *flatten*
operators exist to run - on every call.

This module removes that cost. It caches, per source object, keyed by a
cheap signature that changes whenever anything the cut depends on changes
(the source mesh's geometry + modifier stack, each seam curve's geometry +
modifier stack, and the material-thickness offset), the ``ref_centers`` -
the ``(center_xyz, island_id)`` list on the cut mesh - populated for free by
the flatten/distortion pass that just computed it, so a following color sync
never re-cuts.

A signature miss simply recomputes and re-stores - correctness never
depends on the cache being warm, so a stale entry can only ever cost time,
never correctness. The cache lives in module state for the session only; it
does not survive a file reload (the first post-reload call is just a miss).
"""

import array
import hashlib

# source_object_name -> {"sig": str, "ref_centers": list|None}
_cache = {}


def _hash_mesh(h, obj):
    """Fold an object's mesh geometry (vertex coords + edge topology) into h."""
    me = getattr(obj, "data", None)
    verts = getattr(me, "vertices", None)
    if verts is None:
        h.update(b"\x00nomesh")
        return
    n = len(verts)
    h.update(n.to_bytes(8, "little"))
    if n:
        buf = array.array("f", bytes(12 * n))
        verts.foreach_get("co", buf)
        h.update(buf.tobytes())
    edges = getattr(me, "edges", None)
    if edges is not None:
        ne = len(edges)
        h.update(ne.to_bytes(8, "little"))
        if ne:
            ebuf = array.array("i", bytes(8 * ne))
            edges.foreach_get("vertices", ebuf)
            h.update(ebuf.tobytes())


def _hash_modifiers(h, obj):
    """Fold an object's modifier stack (types + all editable settings) into h,
    so e.g. a Subdivision Surface level change or a Mirror axis toggle
    invalidates the cache even though it leaves the base mesh untouched."""
    for m in obj.modifiers:
        h.update(m.type.encode())
        h.update(m.name.encode())
        h.update(b"\x01" if m.show_viewport else b"\x00")
        for prop in m.bl_rna.properties:
            pid = prop.identifier
            if pid in ("rna_type", "name", "type"):
                continue
            if getattr(prop, "is_readonly", False):
                continue
            try:
                val = getattr(m, pid)
            except Exception:
                continue
            if prop.type == "POINTER":
                val = getattr(val, "name", None) if val is not None else None
            try:
                h.update(repr(val).encode())
            except Exception:
                pass


def compute_signature(mesh_obj, seam_curves):
    h = hashlib.blake2b(digest_size=16)
    _hash_mesh(h, mesh_obj)
    _hash_modifiers(h, mesh_obj)
    h.update(repr(round(float(getattr(mesh_obj, "seams_to_fur_thickness_mm", 0.0)), 6)).encode())
    for c in sorted(seam_curves, key=lambda o: o.name):
        h.update(c.name.encode())
        _hash_mesh(h, c)
        _hash_modifiers(h, c)
    return h.hexdigest()


def _ref_centers_from(bm, face_island):
    bm.faces.ensure_lookup_table()
    return [
        (tuple(bm.faces[f_idx].calc_center_median()), island_id)
        for f_idx, island_id in face_island.items()
    ]


def store_from_islands(mesh_obj, seam_curves, bm, face_island):
    """Cache the face->island mapping a flatten/distortion pass just computed,
    so a following color sync can reuse it instead of re-cutting."""
    sig = compute_signature(mesh_obj, seam_curves)
    _cache[mesh_obj.name] = {
        "sig": sig,
        "ref_centers": _ref_centers_from(bm, face_island),
    }


def is_cut_cached(mesh_obj, seam_curves):
    """True if a previously-stored cut (ref_centers) is still valid for the
    current signature - i.e. a caller that needs real per-face island
    membership can skip re-cutting entirely and reuse whatever it built off
    that earlier cut (e.g. a cut-preview object's existing per-face
    material assignments are still correct)."""
    sig = compute_signature(mesh_obj, seam_curves)
    entry = _cache.get(mesh_obj.name)
    return entry is not None and entry["sig"] == sig and entry.get("ref_centers") is not None


def invalidate(mesh_obj):
    _cache.pop(getattr(mesh_obj, "name", mesh_obj), None)


def clear():
    _cache.clear()
