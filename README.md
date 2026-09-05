# USBee Sewing Patterns

Blender extension for generating flat sewing/cutting patterns from 3D models
(foam crafting, laser cutting, fabric sewing - especially faux fur for
plush/costume pieces). Phase 1 covers non-destructive seam marking,
distortion-bounded flattening via [Boundary First
Flattening](https://github.com/GeometryCollective/boundary-first-flattening)
(BFF), per-piece offset/shrink, and SVG export. See
`.claude/plans` (or ask) for the full Phase 1/2+ design doc.

## Development setup

The addon lives in `usbee/` as a Blender 5.x extension package.

Build BFF's CLI backend once (Linux shown; needs `cmake` and SuiteSparse):

```sh
git clone --depth 1 https://github.com/GeometryCollective/boundary-first-flattening.git /tmp/bff-build
cd /tmp/bff-build
git submodule update --init deps/rectangle-bin-pack
mkdir build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
      -DBFF_BUILD_GUI=OFF -DBFF_BUILD_CLI=ON \
      -DCMAKE_CXX_FLAGS="-include cstdint" -DCMAKE_EXE_LINKER_FLAGS="-lcblas" ..
make -j"$(nproc)"
cp bff-command-line /path/to/seams-to-fur-patterns/usbee/backends/bin/linux-x64/
chmod +x /path/to/seams-to-fur-patterns/usbee/backends/bin/linux-x64/bff-command-line
```

This build dynamically links against the system SuiteSparse install, so it's
a dev-only build, not yet a redistributable one (see `backends/bff.py`'s
module docstring). Windows/macOS builds and a portable Linux build are
future work.

Build and install the extension into Blender's local "User Default" repo:

```sh
blender --command extension build --source-dir usbee --output-dir /tmp
blender --command extension install-file --repo user_default --enable /tmp/usbee-0.1.0.zip
```

## Tests

```sh
python3 -m pytest tests/unit/                 # bpy-free geometry/export logic
blender --background --factory-startup --python tests/headless/test_flatten_pipeline.py
```
