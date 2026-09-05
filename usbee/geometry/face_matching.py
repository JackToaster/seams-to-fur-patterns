"""Nearest-point face-to-island matching, bpy-free so it's plain-pytest
testable.

Used to project island membership (computed on the evaluated, post-modifier
mesh during flatten) back onto the *base* mesh's faces for appearance
preview (per-piece color/material) on the un-evaluated source mesh, since a
generative modifier (e.g. Subdivision) means evaluated and base face
indices/counts don't correspond 1:1.
"""


def _dist_sq(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3))


def nearest_face_island(base_centers, ref_centers_with_island):
    """base_centers: list of (x, y, z) - one per base-mesh face.
    ref_centers_with_island: list of ((x, y, z), island_id) - one per
    evaluated-mesh face, already assigned to an island.

    Returns a list the same length as base_centers, giving the island_id of
    the nearest ref center to each base center (brute force - fine for the
    face counts involved here, no spatial index needed).
    """
    if not ref_centers_with_island:
        return [None] * len(base_centers)

    result = []
    for center in base_centers:
        best_island = None
        best_dist = None
        for ref_center, island_id in ref_centers_with_island:
            d = _dist_sq(center, ref_center)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_island = island_id
        result.append(best_island)
    return result
