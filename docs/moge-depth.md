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
