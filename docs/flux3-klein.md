# FLUX 3 Image Klein

Native support is backported from ComfyUI commit
`20b46065ef7f34fa21e8abebe8d0cca90a904d6d` (upstream licensing is preserved). This is a focused model addition, not a full
ComfyUI v0.41 upgrade.

Obtain the [official ComfyUI files](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
through Download Manager 2. `_sd_` identifies distilled; unmarked files are base.
Use `flux3_image_klein_sd_bf16.safetensors`, `_sd_int8_convrot.safetensors`, or
`_sd_nvfp4.safetensors` in diffusion_models, a Qwen3-VL 4B encoder in text_encoders,
and `flux2-vae.safetensors` in vae. Weights have the FLUX Kommunity license;
consult the model repository for its terms.

The distilled model consumes the next sampling time on every step. Use Euler,
the simple scheduler, 16 steps and no CFG (`BasicGuider`); do not substitute the
base model's 50-step/CFG 3.5 recipe. Its native 128-channel latent is created by
`EmptyFlux2LatentImage`. The Qwen encoder must be loaded with type `flux3`, which
selects eight hidden layers and pads tokens to a multiple of 80. The image
transformer is 5.6B parameters; encoder and optional upsampler memory is additional.

For edits, VAE-encode each reference and chain `ReferenceLatent` nodes in input
order. The provider documents up to ten images. Describe the complete finished
scene, including what remains and what changes. References anchor content but do
not lock framing or guarantee unchanged pixels. Append a layout JSON list to the
caption to guide placement: `{"id":"obj_1","bbox":[top,left,bottom,right],"desc":"..."}`
with coordinates 0–1000 and `<obj_1>` in the caption. Boxes are soft guidance,
not inpainting masks. `Flux3LayoutPrompt` can author the list from bounding boxes.
The optional 9B prompt upsampler is a separate text model; direct captions need
neither its weights nor a hosted text service.

RouterImage's `comfyui/flux3-image-klein` profile and MediaTools own the agent route
and its exact exposed controls. Discover their live cards; source support alone
does not establish installed weights or a loaded running service. On transport
failure inspect the existing prompt/run ID before retrying. Never restart a busy
ComfyUI queue to activate this model.

Developer preparation: `COMFYUI_PYTHON=/path/to/prepared/python python3 scripts/setup_worktree.py`.
Use that interpreter for `python -m pytest tests-unit/comfy_test/model_detection_test.py
 tests-unit/comfy_test/test_prefetch_recovery.py tests-unit/comfy_test/test_pinned_memory_recovery.py -q`.
These isolated checks do not establish GPU output quality or portability to other devices.
