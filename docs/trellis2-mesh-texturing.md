# Texturing existing meshes with TRELLIS.2

This fork adds `VaeEncodeMeshTrellis2` (Trellis2 VAE Encode Mesh). It converts a
native `MESH` to TRELLIS.2's shape latent using Microsoft's published shape
encoder. It also returns a centered, uniformly scaled Y-up mesh and exact
surface subdivisions for the existing texture decoder. It does not regenerate
geometry: triangle connectivity is retained during encoding. UV unwrapping and
optional decimation are separate downstream operations.

## Requirements and setup

Use the normal ComfyUI Python environment and native Trellis2 models. The encoder
uses ComfyUI sparse convolutions, model patching, dtype selection and GPU memory
management. Repeated blocks use the allocation compiler after sparse neighbor
maps are built; topology-changing downsampling and variable-size sparse-convolution
gather workspaces stay outside that repeated scope. Dense normalization and MLP
operations retain allocation recording.
Its CPU mesh-to-dual-grid component requires a C++20 compiler, matching
PyTorch development headers (included with PyTorch), setuptools, ninja and Eigen
3 headers. No full O-Voxel, CuMesh, FlexGEMM or separate inference worker is needed.

Run the repository worktree setup command with the optional encoder build:

```sh
COMFYUI_PYTHON=/path/to/comfy/python EIGEN_INCLUDE_DIR=/path/to/eigen \
  python3 scripts/setup_worktree.py --trellis-mesh
```

`EIGEN_INCLUDE_DIR` must contain `Eigen/Dense`; `/usr/include/eigen3` is the default.
The extension is built into this checkout and must be rebuilt after changing
Python/PyTorch ABI or native source. Reruns reuse matching build outputs. Setup
never downloads models, changes packages or starts services. Eigen 3.3.7 from its
official GitHub mirror was used for the Linux build; other platforms are unverified.
The build may emit Eigen deprecation warnings with recent compilers.

Provision `microsoft/TRELLIS.2-4B`, revision
`af44b45f2e35a493886929c6d786e563ec68364d`, file
`ckpts/shape_enc_next_dc_f16c32_fp16.safetensors` in a configured `vae` directory.
The local filename is `trellis_2_shape_encoder_fp16.safetensors` (708,797,208 bytes;
SHA256 `f37c5ff5b983b68e9946060000f09bc131f3e84318a2c8b7430a81e4b4636c41`).
Use Download Manager 2 for provisioning. VAELoader detects the original checkpoint;
no conversion or downloaded Python code is executed when loading it.

Texture generation additionally needs the native Trellis2 diffusion bundle,
DINOv3-L image encoder and texture VAE. Discover exact filenames through
`/object_info`; never substitute the TripoSplat DINOv3-H encoder. Image preprocessing
uses the foreground mask, a black background and `ImageCropToMask` pad_factor 1.0.

## Workflow and contracts

1. Generate Gaussians with TripoSplat, extract with `SplatToMesh`, and optionally
   remesh/clean the result.
2. Load the shape **encoder** with VAELoader. Connect mesh and encoder to
   `VaeEncodeMeshTrellis2`. `resolution` is `"512"` or `"1024"`; start at 512.
3. Connect `shape_latent` to `Trellis2TextureStage`, alongside image conditioning
   from `Trellis2Conditioning` using DINOv3-L.
4. Sample the texture model (Euler, 12 steps, CFG 1, flow shift 3 are upstream
   texturing defaults). The latent output is texture data, not a new shape.
5. Decode with `VaeDecodeTextureTrellis`, supplying the encoder's
   `shape_subdivides` output and the texture VAE.
6. Unwrap the encoder's `normalized_mesh`, optionally decimating first. Bake the
   decoded voxel colors with `BakeTextureFromVoxel`; use the normalized dense
   mesh as `reference_mesh`. Apply base color, metallic and roughness, then SaveGLB.

Example API fragment (IDs refer to nodes in the same prompt):

```json
{
  "141": {
    "class_type": "VaeEncodeMeshTrellis2",
    "inputs": {"mesh": ["133", 0], "vae": ["140", 0], "resolution": "512"}
  }
}
```

Inputs: native Y-up mesh (including variable-size batches), encoder VAE, resolution.
Outputs, in order: LATENT, normalized MESH, SHAPE_SUBDIVIDES. Latents carry sparse
coordinates, per-item counts, coordinate resolution and explicit `z_up` frame.
Subdivisions are the original mesh's occupancy hierarchy, not predictions from a
shape decoder. Texture voxels are converted back to Y-up by the existing decoder.
The encoder node has no file writes or network side effects. Saving is explicit.

The normalizing transform changes world location/scale. Use the returned mesh for
baking, not the original unnormalized input. Existing materials/UVs are discarded
by this geometry-only encoding output. Texture appearance is conditioned by the
reference image; it does not recover invisible true materials. Splat extraction
quality still limits the geometry. Higher resolution costs more RAM/VRAM and time;
CPU voxelization can take substantial time for dense meshes.

## Verification and recovery

Run `python -m pytest tests-unit/comfy_test/test_trellis2_mesh_encoder.py -q`
with the prepared interpreter. Geometry tests require the optional extension;
skipped tests are not native-conversion acceptance. Validate API prompts and run
a real model job before claiming inference readiness. Missing encoder weights,
wrong VAE selection and a missing native extension produce actionable errors.
For out-of-memory failures, use 512 resolution and reduce source mesh complexity;
do not replace the encoder with occupancy-only shape generation. After an uncertain
submission inspect queue/history before retrying. Restart ComfyUI only when its
queue and dependent workflow services are idle. Reopen a saved workflow from disk
before editing so a stale browser tab cannot overwrite it.

## Attribution

The encoder architecture and CPU dual-grid converter are adapted from
[Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), commit
`75fbf0183001ed9876c8dbb35de6b68552ee08bd`. The converter implementation is retained
in `comfy/ldm/trellis2/voxelize/flexible_dual_grid.cpp`; its header is reduced to the
used CPU API, with a local binding/build entrypoint. Its [MIT license](../comfy/ldm/trellis2/voxelize/LICENSE)
is preserved. Model terms remain those of the published model. Existing ComfyUI
loaders, mesh processing and texturing remain attributed to their original owners.
