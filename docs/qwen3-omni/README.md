# Qwen3-Omni native CLIP integration

Status: installed and verified on the local RTX 5090, 18 October 2026. The
converter, complete source validator, reconstructed reference and native adapter
are implemented and integrated. Three-pass accelerated NVFP4 has no new assertion
failures across the frozen 24-case comparison; lifecycle, boundary, resource and
live text/audio/image/combined queue checks pass. See [results and known source
limitations](status.md) and [operation/saved workflow](usage.md).

This native ComfyUI integration loads a converted version of
[`shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4`](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4)
as a CLIP object and accepts **text, audio and image inputs, separately and
together**, through the existing Generate Text node. Output is text.

The Thinker, audio encoder and vision encoder are required. Speech generation
components can be omitted. CLIP is the ComfyUI interface here; this does not
make the model a diffusion-conditioning encoder.

## Read and execute

| Document | Purpose |
| --- | --- |
| [Operation](usage.md) | Loader/node inputs, output handling, precision and recovery |
| [Requirements](requirements.md) | Required behavior, examples, exclusions and completion criteria |
| [Architecture](architecture.md) | Native owners, data flow and likely integration points |
| [Checkpoint and quantization](quantization.md) | Payload inspection, feasibility experiment and conversion contract |
| [Implementation plan](implementation.md) | Ordered work packages, dependencies and delivery gates |
| [Reconstructed reference](reference.md) | Offline environment, component and fixture commands |
| [Validation](validation.md) | Correctness, modality, memory and regression acceptance |
| [Evidence and decisions](evidence.md) | Sources, verified observations and unresolved questions |

The implementation plan records the original milestones; the status document
records completed evidence. No additional full BF16 checkpoint is required. Downloads use
Download Manager 2 and the repository's scoped Hugging Face cookie procedure.

Development requires the repository's prepared Python runtime and worktree setup.
GPU acceptance requires provisioned checkpoints, a compatible GPU/kernel runtime,
licensed test media and enough RAM/disk for source, output and working space.
The candidate is 18.78 GB; measured limits and remaining performance checks are
recorded in the status document. No latency or memory guarantee is implied.

Use the installed `qwen3_omni` loader choice and existing Generate Text node.
The saved workflow is in the ComfyUI user workflow library; its API companion is
[versioned here](workflows/demo-api.json). The original HF directory is required
for conversion/reference validation, not inference. Discover current live model
choices rather than assuming another deployment has this integration installed.
