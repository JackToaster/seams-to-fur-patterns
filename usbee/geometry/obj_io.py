"""Minimal Wavefront OBJ read/write for round-tripping single islands
through the BFF backend. Deliberately doesn't use bpy's full OBJ exporter -
we only need positions and triangle/polygon indices, and the full exporter
carries overhead (materials, UVs, scene selection side effects) we don't
want in a tight per-island subprocess loop.
"""


def write_obj(path, verts, faces):
    """verts: list of (x, y, z). faces: list of lists of 0-based vertex
    indices (triangles or ngons)."""
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x!r} {y!r} {z!r}\n")
        for face in faces:
            idx_str = " ".join(str(i + 1) for i in face)
            f.write(f"f {idx_str}\n")


def read_obj(path):
    """Returns (verts, faces) with 0-based face indices. Reads only v/f
    lines - BFF's flattened output OBJ has no materials/normals to worry
    about."""
    verts = []
    faces = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                x, y, z = (float(p) for p in parts[1:4])
                verts.append((x, y, z))
            elif tag == "f":
                face = []
                for p in parts[1:]:
                    # OBJ face refs can be "v", "v/vt", "v/vt/vn", "v//vn"
                    vi = int(p.split("/")[0])
                    face.append(vi - 1 if vi > 0 else vi)
                faces.append(face)
    return verts, faces
