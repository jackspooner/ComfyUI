# SAM 3D Body dense checkpoint precision

The native loader accepts detection-root SAM 3D Body checkpoints through
`SAM3DBody_Loader`. The Comfy-Org DINOv3 BF16 artifact stores the backbone in BF16
and prediction layers in FP32. This fork prefers BF16 computation for that dense
checkpoint, with FP32 fallback when BF16 is unsupported. The generic diffusion
policy otherwise chooses FP16 first on CUDA, even for BF16 weights. Explicit
global UNet dtype overrides still apply. This fork constructs prediction layers
in FP32 when selecting a dense BF16 backbone, preserving values both on ordinary
loading and through AIMDO's resident-weight dtype metadata. Other checkpoint
and quantized loading policies are unchanged. ModelPatcher still
owns device placement, admission and offloading; no extra model process is used.

Use the existing loader, predict, render and export nodes; there are no new graph
controls. Install model assets through the existing download owner and select an
exact filename returned by live discovery. The change requires normal coordinated
service activation before the running loader uses it.

CPU regression checks:

```sh
COMFYUI_PYTHON=/path/to/prepared/python python3 scripts/setup_worktree.py
/path/to/prepared/python -m pytest tests-unit/comfy_test/sam3d_body_precision_test.py -q
```

The fixtures verify the real device-selection policy, precision and native
patcher handoff without GPU inference. The initial correction only preserved
FP32 heads when BF16 was selected; a deployed dancer test still failed because
the CUDA policy selected FP16. Direct CPU inference using the same checkpoint
with a BF16 backbone and FP32 heads recovered the raised limbs both with and
without hand refinement. The default-selection correction was deployed, but GPU
acceptance still failed: hand refinement encountered singular crop matrices and
body-only output contained nonfinite pose values. This is not yet a validated GPU
configuration.

SAM3D's DINOv3 backbone and decoder explicitly request
`low_precision_attention=False`. The Comfy Kitchen INT8 dispatcher now respects
that opt-out for BF16 and FP16 as well as FP32, using the existing PyTorch SDPA
path with the caller's dtype, mask, scale and layout. Both tensor and consuming
container paths honor the same contract; default calls still use INT8. Global
attention flags are unchanged. Run the CPU dispatch regression with
`tests-unit/comfy_test/attention_precision_test.py`; it reproduces eight failures
before the fix. This dispatch correction still requires activation and real GPU
visual/numeric acceptance before attributing the failed poses to attention.
Successful file creation alone is not quality acceptance. Compare source and
overlay, then verify a bounded video and animation in a compatible viewer.

Upstream [BF16 issue 16494](https://github.com/Comfy-Org/ComfyUI/issues/16494)
reports failure with the dense artifact but does not establish a cause.
[BVH issue 16963](https://github.com/Comfy-Org/ComfyUI/issues/16963) separately
reports rest-skeleton retargeting problems; this precision change does not repair
the BVH exporter. Prefer skinned GLB until that exporter is verified for the target
application. Original SAM 3D Body implementation and checkpoint credit remain with
[Meta](https://github.com/facebookresearch/sam-3d-body) and
[Comfy-Org](https://huggingface.co/Comfy-Org/sam-3d-body).
