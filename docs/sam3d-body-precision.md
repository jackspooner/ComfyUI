# SAM 3D Body dense checkpoint precision

The native loader accepts detection-root SAM 3D Body checkpoints through
`SAM3DBody_Loader`. The Comfy-Org DINOv3 BF16 artifact stores the backbone in BF16
and prediction layers in FP32. This fork constructs those prediction layers in
FP32 when selecting a dense BF16 backbone, preserving values both on ordinary
loading and through AIMDO's resident-weight dtype metadata. Existing FP16,
FP32 and quantized loading policies are otherwise unchanged. ModelPatcher still
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

The fixtures verify precision and native patcher handoff without GPU inference.
A real BF16 dancer smoke test before this change completed but produced an
incorrect pose. This fix addresses a demonstrated precision loss; its effect on
that full-model failure still needs visual and numeric acceptance after activation.
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
