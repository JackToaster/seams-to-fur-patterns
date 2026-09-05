import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "usbee"))

from geometry import face_matching


def test_nearest_face_island_picks_closest():
    base_centers = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    ref_centers = [((0.1, 0.0, 0.0), 1), ((9.9, 0.0, 0.0), 2)]
    result = face_matching.nearest_face_island(base_centers, ref_centers)
    assert result == [1, 2]


def test_nearest_face_island_ties_break_to_first_seen():
    base_centers = [(1.0, 0.0, 0.0)]
    ref_centers = [((0.0, 0.0, 0.0), 1), ((2.0, 0.0, 0.0), 2)]
    result = face_matching.nearest_face_island(base_centers, ref_centers)
    assert result == [1]


def test_empty_refs_returns_none_for_all():
    result = face_matching.nearest_face_island([(0, 0, 0), (1, 1, 1)], [])
    assert result == [None, None]


def test_empty_base_returns_empty():
    result = face_matching.nearest_face_island([], [((0, 0, 0), 1)])
    assert result == []
