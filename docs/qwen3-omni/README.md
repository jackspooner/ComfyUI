# Qwen3-Omni native CLIP integration

Status: implementation in progress, 18 October 2026. The offline converter,
complete source validator, reconstructed reference, native adapter and fixture
tools are implemented on the project branch. Native component tests pass; full
native loading, precision acceptance and installation remain pending. See
[current results and unresolved gates](status.md).

Build a native ComfyUI integration that loads a converted version of
[`shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4`](https://huggingface.co/shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4)
as a CLIP object and accepts **text, audio and image inputs, separately and
together**, through the existing Generate Text node. Output is text.

The Thinker, audio encoder and vision encoder are required. Speech generation
components can be omitted. CLIP is the ComfyUI interface here; this does not
make the model a diffusion-conditioning encoder.

## Read and execute

| Document | Purpose |
| --- | --- |
| [Requirements](requirements.md) | Required behavior, examples, exclusions and completion criteria |
| [Architecture](architecture.md) | Native owners, data flow and likely integration points |
| [Checkpoint and quantization](quantization.md) | Payload inspection, feasibility experiment and conversion contract |
| [Implementation plan](implementation.md) | Ordered work packages, dependencies and delivery gates |
| [Reconstructed reference](reference.md) | Offline environment, component and fixture commands |
| [Validation](validation.md) | Correctness, modality, memory and regression acceptance |
| [Evidence and decisions](evidence.md) | Sources, verified observations and unresolved questions |

Begin with milestone M0 and the small compatibility experiment in M1. Do not
download the original full BF16 model merely to begin: establish what reference
weights and execution code the experiment actually needs first. Downloads use
Download Manager 2 and the repository's scoped Hugging Face cookie procedure.

Development requires the repository's prepared Python runtime and worktree setup.
GPU acceptance requires provisioned checkpoints, a compatible GPU/kernel runtime,
licensed test media and enough RAM/disk for source, output and working space.
No measured latency, final package size or peak VRAM guarantee exists yet.

The diagnostic command in the status document is runnable; installation remains
planned. Names such
as `qwen3_omni` below are proposed interfaces until implementation lands. Runtime
activation and actual inference must be recorded separately from source delivery.
