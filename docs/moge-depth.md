# MoGe metric depth PNG

`SaveMoGeDepthPNG` writes one MoGe depth map using the same response contract as
RuntimeHost's `/v1/images/depth` endpoint:

- output is a same-size, single-channel 16-bit PNG;
- zero marks pixels outside MoGe's mask, non-finite pixels, and non-positive depth;
- values 1 through 65535 linearly encode metric-scale depth between the emitted
  minimum and maximum.

The node embeds `moge_depth_min`, `moge_depth_max`, and `moge_depth_encoding` as
PNG text metadata. It also writes a sibling JSON file and returns both bounds in
ComfyUI history under `outputs.<node-id>.moge_depth[0]`. When every valid pixel
has the same depth, all valid pixels are encoded as 1.

Load the checkpoint through the normal `loaders/geometry_estimation` model
category. The supplied editable workflow expects
`moge_3_vitl_normal.pt`, which may be the pinned upstream `model.pt` saved under
that name. No request downloads a checkpoint.

For one input image, execute:

`LoadImage -> LoadMoGeModel -> MoGeInference -> SaveMoGeDepthPNG`

The workflow in `user/default/workflows/Image Depth Estimation (MoGe-3 Runtime).json`
defaults to resolution level 6 and three refinement steps for the RTX 4070. A
batch larger than one is rejected because the RuntimeRouter depth API is a
single-image endpoint.

## Numeric geometry interchange

`SaveMoGeGeometry` exports one at-most-4-megapixel `MOGE_GEOMETRY` packet to
`output/moge_geometry/<run UUID>/`. Connect `MoGeRender`'s selected normal map to
`normal_preview`, and optionally connect `MoGePointMapToMesh` to `mesh` for GLB.
The output history contains a `moge_geometry` manifest descriptor. A final atomic
`manifest.json` references SHA-256/size-verified NPZ, 16-bit depth PNG, normal PNG,
mask PNG and optional GLB. No loader, downloader or queue owner is added.

The version-1 NPZ is pickle-free and contains float32 H×W×3 points/normals/image,
float32 H×W depth, uint8 H×W mask, and perspective-only normalized 3×3 intrinsics.
`schema_version`, `projection` and `normal_source` are scalar arrays. Invalid
points/depth/normals are zero; mask=0 means invalid and zero normals mean undefined.
`LoadMoGeGeometry` reads only bounded archives from ComfyUI's input directory,
checks shape/dtype/finite data and restores invalid geometry to infinity for the
existing mesher. It never loads a model. Saved maps retain texture and camera
data for subsequent normal rendering or mesh conversion.

Perspective points use OpenCV X right, Y down, Z forward, estimated metric scale;
depth is camera Z. Camera JSON includes normalized and pixel intrinsics,
horizontal/vertical FOV degrees via `MoGeGeometryToFOV`, and focal length in pixels.
Pixel intrinsics scale row 0 by width and row 1 by height, retaining native
pixel-center UV conventions. This is not lens calibration or verified measurement.

Panorama inference returns relative radial distance without intrinsics; per-view
metric scale is discarded by the native merger. Its points are spherical Z-up:
theta=(1-u)*2pi, phi=v*pi, direction=(sin(phi)cos(theta), sin(phi)sin(theta),
cos(phi)). Normals remain in that native basis, not tangent space. Native normal
PNG rendering flips Y/Z for OpenGL or Z for DirectX before mapping to RGB.
Native GLB meshing uses (x,-y,-z) and reversed winding for perspective, or (y,z,x)
for panorama. These are visible-surface meshes, not complete/watertight scenes;
the equirectangular seam is not welded.

Run `tests-unit/comfy_extras_test/moge_geometry_io_test.py` and
`tests-unit/comfy_extras_test/nodes_moge_test.py` with the prepared interpreter.
Fixtures verify numeric round trips, FOV, PNG encoding, native GLB and rejected
untrusted input. Source/CPU checks do not prove live checkpoint quality.
