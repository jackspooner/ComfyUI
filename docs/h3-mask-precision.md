# Native H3 fractional denoise masks

H3 video/audio token masks are rounded upward to a 1/4096 grid and transported
to the diffusion model in FP32. Values such as 0.9995 and 0.9997 stay below one
with BF16 or FP16 model compute. Only values at least one take the full-generation
shortcut; zero remains preserved and ordinary binary masks retain their behavior.
This fixes near-one guidance being lost through a 1/256 grid, approximate-one
shortcuts or low-precision condition casts. The sampler and latent layout remain
unchanged; fine fractional masks still control generative guidance rather than
guaranteed source identity or timing.

`BaseModel.extra_conds_dtypes` is a small class-owned condition transport policy.
Its empty default retains the model compute dtype. H3 overrides only the two mask
keys with FP32; integer tags, ordinary tensors/lists and structured payloads keep
their existing transport. Device movement still uses ComfyUI's normal cast helper.
No model class methods are replaced during node execution.

`GET /features` advertises `minimax_h3_fractional_masks: true` when the running
core implements these precision semantics. Callers must require the boolean
`true`; an absent or false value does not advertise support. This capability does
not assert model installation, custom-node availability or rendered output quality.
It is a core capability and cannot be overridden by a CLI feature flag.

Prepare the worktree using the root AGENTS setup command, then run with its
prepared interpreter:

```sh
python -m pytest tests-unit/comfy_test/test_minimax_h3_mask_precision.py tests-unit/comfy_test/test_minimax_h3_model.py tests-unit/comfy_test/test_minimax_h3_vae.py tests-unit/feature_flags_test.py -q
```

These are isolated CPU fixtures, not full-model, GPU-memory or generated-video
acceptance. The installed Motion-Context MultiRef compatibility probe can verify
that native precision is sufficient and its patch installer leaves class methods
unchanged. That optional package is not a core dependency. Updated source does not
alter a running service; coordinated idle deployment and normal-queue output
acceptance remain separate operations.
