"""Subprocess wrapper around the vendored bff-command-line binary
(Boundary First Flattening) - the accurate flattening backend that replaces
FreeCAD's flatten operation.

NOTE on the vendored Linux binary: it's built locally against this machine's
system SuiteSparse libraries (dynamically linked), which is fine for local
development but is *not* yet a redistributable build (a real release needs
static linking or bundled .so's per platform, and Windows/macOS builds via
CI). See the plan doc's "BFF binary strategy" section.
"""

import os
import platform
import stat
import subprocess
import tempfile
from pathlib import Path

_PLATFORM_DIR = {
    ("Linux", "x86_64"): "linux-x64",
    ("Windows", "AMD64"): "windows-x64",
    ("Darwin", "x86_64"): "macos-x64",
    ("Darwin", "arm64"): "macos-arm64",
}

_BINARY_NAME = "bff-command-line.exe" if platform.system() == "Windows" else "bff-command-line"


class BFFError(Exception):
    """Raised when the BFF backend fails or its binary can't be found."""


def find_binary(addon_dir):
    key = (platform.system(), platform.machine())
    plat_dir = _PLATFORM_DIR.get(key)
    if plat_dir is None:
        raise BFFError(f"No vendored BFF binary for platform {key}")

    candidate = Path(addon_dir) / "backends" / "bin" / plat_dir / _BINARY_NAME
    if candidate.exists():
        if platform.system() != "Windows" and not os.access(candidate, os.X_OK):
            # Extension zips don't reliably preserve the executable bit.
            candidate.chmod(candidate.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return candidate

    raise BFFError(
        f"BFF binary not found at {candidate}. It isn't vendored for this "
        f"platform yet - see USBee preferences to fetch/build one, or build "
        f"bff-command-line yourself and point preferences at it."
    )


def flatten_island(binary_path, verts, faces, timeout=30):
    """Flatten a single island's verts/faces (as returned by
    geometry.obj_io) via BFF. Returns (out_verts, out_faces) with the same
    face connectivity and flattened (z=0) positions.

    Raises BFFError on subprocess failure or unparseable output, with the
    binary's stderr included for diagnostics.
    """
    from ..geometry import obj_io

    with tempfile.TemporaryDirectory(prefix="usbee_bff_") as tmp:
        in_path = Path(tmp) / "island.obj"
        out_path = Path(tmp) / "island_flat.obj"
        obj_io.write_obj(in_path, verts, faces)

        try:
            result = subprocess.run(
                [
                    str(binary_path),
                    str(in_path),
                    str(out_path),
                    "--writeOnlyUVs",
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise BFFError(
                f"BFF timed out after {timeout}s flattening an island of "
                f"{len(faces)} faces - it may have non-disk topology that "
                f"slipped past validation, or be too large"
            ) from exc

        if result.returncode != 0 or not out_path.exists():
            raise BFFError(
                f"BFF failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        out_verts, out_faces = obj_io.read_obj(out_path)

    if not out_verts:
        raise BFFError("BFF produced an empty flattened mesh")

    return out_verts, out_faces
