import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "seams_to_fur"))

from geometry import boundary


def test_fan_of_triangles():
    # 4 triangles fanned around a center vertex (4), forming a square disk.
    faces = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    loop = boundary.ordered_boundary_loop(faces)
    assert set(loop) == {0, 1, 2, 3}
    assert 4 not in loop
    assert len(loop) == 4


def test_single_quad():
    faces = [[0, 1, 2, 3]]
    loop = boundary.ordered_boundary_loop(faces)
    assert set(loop) == {0, 1, 2, 3}


def test_closed_mesh_has_no_boundary():
    # A tetrahedron (closed, no boundary) - every edge is shared by two
    # faces, so there should be no boundary loop.
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    with pytest.raises(boundary.BoundaryError):
        boundary.ordered_boundary_loop(faces)
