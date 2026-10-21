# Native Fibo 2.0

This fork selectively backports upstream commit
`87ed5a066d9c4660ef42100fd01a888da6994856`
([PR #17240](https://github.com/Comfy-Org/ComfyUI/pull/17240), Jukka Seppänen)
onto v0.42.1; it does not require the later upstream master or a package upgrade.
Existing fork CLIP selectors and enum values are preserved; FIBO2 is 43 here.
The model uses native ComfyUI patching, optimized attention, prefetch and queue
ownership. It performs no model downloads or external prompt rewriting at runtime.

## Files and use

Provision [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0)
weights explicitly through Download Manager 2. Under the configured model root:

| Category | Editing setup |
| --- | --- |
| diffusion_models | fibo2_edit_turbo_bf16.safetensors |
| text_encoders | qwen3vl_4b_bf16.safetensors |
| vae | flux2-vae.safetensors |

Use `UNETLoader`, `CLIPLoader` with `type: "fibo2"`, and `VAELoader`.
Connect CLIP, VAE and an image to `TextEncodeFibo2`. Its outputs are positive,
negative and a correctly sized 128-channel latent. Connect those to `KSampler`,
then `VAEDecode` and `SaveImage`. Edit Turbo: 4 steps, CFG 1, Euler, simple,
denoise 1. Dev: upstream recommends 30 steps and CFG 5; Dev weights are separate
and were not provisioned in this task. Turbo is intended for editing.

`prompt` accepts a plain edit instruction or structured JSON with
`edit_instruction`; reference markers are `<image_1>`, `<image_2>`, etc.
`negative_prompt` defaults to the trained null caption. The optional `images`
autogrow accepts image batches and multiple references; connect the VAE to encode
them into reference latents for both positive and negative conditioning.
Without images the node has a generation path for Dev. Width/height 0 choose
roughly 1 MP from the first image's aspect ratio, or 1024 square without images;
explicit sizes round down to multiples of 16 with a minimum of 16. A single image
is scaled/cropped to the output size; multiple images use individual trained
sizes. Reference count, resolution and prompt length increase memory/time.

Discover `/object_info/TextEncodeFibo2` and the loader schemas on the selected
service before submitting. Autogrow API inputs are flattened (`images.image_1`, etc.);
editor graphs and API prompts differ. Save editable workflows in the installation's
user workflow library. Submit once through `/prompt`, follow the returned ID in
queue/history, and inspect saved output. On uncertain submission, inspect that ID
before retrying. Do not load incomplete downloads. A missing node or `fibo2` choice
means activation is missing; an inference error needs the prompt history/logs,
not another download. No separate RuntimeHost worker is needed.

## Verification and limits

Prepare with `COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py`.
Run `python -m pytest tests-unit/comfy_test/fibo2_test.py tests-unit/comfy_test/model_detection_test.py tests-unit/comfy_test/qwen_vl_image_channels_test.py -q`.
The focused checks cover the fork's CLIP identities, packed/unpacked detection,
fused feed-forward agreement, reference prompt packaging and latent conditioning.

The local Edit Turbo BF16 checkpoint SHA256 is
`028f6cf13af75840d7cc7346a5784fe55887c6b9299d173c8271161ba5ccf7d0`;
the Qwen3-VL BF16 encoder SHA256 is
`36f3ff447ef59201722e8f9ce6020c9819fdcfba6aa2608c4e09b1c0ce114e34`.
Both match the published Comfy-Org revision
`41e6a0331eff43d5a323107197056e377288c642`.
Full reference parity, Dev, quantized weights, mask editing and non-CUDA devices
have not been independently accepted here. Upstream reports reference comparisons;
those are upstream evidence, not local verification. Text glyph rendering uses
Pillow's built-in font; training's additional fallback fonts are not included.

The weights are BRIA AI's CC BY-NC 4.0 release; commercial use requires its
separate agreement. Preserve upstream attribution and the published license.

## Local acceptance — 21 October 2026

Integrated and activated the backport on the RTX 5090 with the existing
Torch 2.14.1+cu130, Kitchen 0.2.39 and ComfyUI runtime. All 1,837 previous node
registrations remained; TextEncodeFibo2 was added. No dependency/unit changes.
All 634 Edit Turbo checkpoint keys/shapes match meta-device model construction.
Focused CPU suite: 34 passed; broader loader/history/text regressions: 147 passed,
3 optional reference tests skipped. The integrated focused suite also passed.

Normal-queue BF16 editing changed a red-ball fixture to blue while retaining its
beige background and shadow; both outputs were inspected:

- 512 square, 4 steps, CFG 1, Euler/simple, seed 321:
  `478f7bd6-5228-49e4-9bcb-ba277be45b52`, 45.46 s including cold loading.
- After an idle `/free` unload request, repeated at default trained size
  (1024 square), seed 322:
  `fe85a11e-02ae-48c7-b3c8-12cb5535314b`, 16.48 s. Loader nodes reexecuted;
  the allocator reported zero rogues and the coordinated GPU hold was released.

Saved editable workflow: `user/default/workflows/Fibo 2.0 Edit Turbo BF16.json`,
with matching `api/Fibo 2.0 Edit Turbo BF16.api.json`; replace the fixture image
and edit instruction before use. Geometry/link checks passed. The first harness
submission used an unprefixed reference socket and failed before encoding;
corrected API input is `images.image_1`. This was a test-graph error.

These are bounded editing/reload smoke checks. Full upstream numerical parity,
cancellation/recovery, multi-reference quality, Dev, quantized weights and other
devices remain unverified. The existing Breeze-TTS cache_position documentation
log is unrelated to this integration. No broad edit-quality guarantee is implied.
