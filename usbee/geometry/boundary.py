"""Extract an ordered boundary loop from a flat (2D, single-island) mesh's
face list. bpy-free - operates on plain vertex/face index lists so it's
usable both from the flatten pipeline and from standalone tests.
"""


class BoundaryError(Exception):
    pass


def ordered_boundary_loop(faces):
    """faces: list of lists of vertex indices (any polygon size).

    Returns an ordered list of vertex indices tracing the single boundary
    loop (edges used by exactly one face), assuming the input is a valid
    disk (guaranteed by islands.validate_island_topology upstream).
    """
    edge_count = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            key = (a, b)
            edge_count[key] = edge_count.get(key, 0) + 1

    # A boundary half-edge (a, b) is one whose reverse (b, a) doesn't appear
    # (or appears from a different face) - i.e. only one directed usage.
    boundary_next = {}
    for (a, b), _count in edge_count.items():
        if (b, a) not in edge_count:
            boundary_next[a] = b

    if not boundary_next:
        raise BoundaryError("No boundary edges found - is this a closed (non-disk) mesh?")

    start = next(iter(boundary_next))
    loop = [start]
    current = boundary_next[start]
    while current != start:
        loop.append(current)
        if current not in boundary_next:
            raise BoundaryError("Boundary loop is not closed/simple")
        current = boundary_next[current]
        if len(loop) > len(boundary_next) + 1:
            raise BoundaryError("Boundary loop did not close - mesh may have multiple loops")

    return loop
