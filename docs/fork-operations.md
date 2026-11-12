# Fork operations

This guide covers behavior specific to this fork. Start with the
[README](../README.md) and [feature index](fork-features.md); use the
[upstream README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md)
for base installation and ordinary workflow operation.

## Source, runtime and activation

The source includes upstream v0.45.0. Use its pinned frontend, templates,
Comfy Kitchen and compatible GPU runtime; the Fibo v0.42.1 backport is historical,
not the current source base. The existing five-output `TextGenerate` preserves
history, JSON and budget controls; upstream `TextGenerateV2` has a separate
two-output single-turn contract. New upstream model paths need their own weights
and acceptance. See [stable-release maintenance](stable-release-maintenance.md).

The maintained Linux installation uses an existing `.venv`, external shared
models and `comfyui.service`. These are installation-specific choices, not files
included in a public clone. Core edits require a safe restart before a running
instance can use them; refresh the editor's node definitions afterward. Preserve
accepted work, producer coordination, recovery records and configuration. Do not
infer maintenance safety from an empty runnable queue or replay uncertain work.

The fork excludes cloud-deployment uploads and their persistent headless marker.
Local web SSO and partner-token defaults remain disabled. Retained partner nodes
remain explicit external integrations. This is not a claim that every optional
third-party node is offline.

## Queue and maintenance controls

[Pending-prompt controls](queue-control.md) atomically hold and resume exact
pending IDs, preserving graphs and seeds across restarts. Running prompts reject
pause. A paused prompt does not execute automatically; a released receipt without
queue/history evidence remains uncertain. Batch dependencies and shared-stage
children must use their owner's controls.

[Maintenance fencing](maintenance.md) is opt-in through
`COMFYUI_MAINTENANCE_DIR` and `COMFYUI_MAINTENANCE_TOKEN_FILE`. It fences new
admission, drains accepted work and retains bounded non-executable receipts.
It does not pause tensors, restore unfinished prompts, deploy code or release
GPU holds. The guide covers explicit physical-work reconciliation while retaining
unknown outcomes. `COMFYUI_MAINTENANCE_MAX_RECEIPTS` configures the default
10,000-receipt capacity or `unlimited`; retained unknown-work safeguards still apply.

## External image references

**Load Image** and **Load Image (as Mask)** list images in nested input folders
as sorted relative paths, including staged `routervideo_refs/` inputs. Non-images
and `.git` directories are omitted. This dropdown behavior needs no external-path
flag and retains existing image-path validation.

Start with `--allow-external-image-paths` to allow either node to read an absolute
image filename accessible to the process. The file stays in place: modifying it
changes later executions, and deleting or moving it breaks the reference. This
does not relax the generic annotated-path resolver or other file endpoints.
`/features` reports `external_image_paths`; `/view?filename=<URL-encoded-absolute-path>`
serves verified image files with existing preview/channel options. Relative input,
output and temporary references are unchanged.

The separately installed Cutlery external-image preview extension restores these
references in frontend workflows. RouterImage can use direct local references
or import URL/inline bytes through `/upload/image`; imported inputs persist until
deleted. Old outputs whose temporary inputs were already deleted are not repaired.

## Scaled images and video extraction

**Scale Image to Total Pixels** returns `IMAGE`, `width` (`INT`) and `height`
(`INT`) in that order. The integers are the actual scaled dimensions after
`resolution_steps` rounding and can feed downstream size inputs. Existing image
connections and resize settings keep their behavior. The focused CPU check is
`tests-unit/comfy_extras_test/image_scale_total_pixels_test.py`.

For video, start with **Load Video → Trim Video → Get Video Components** and
select **Images only** when audio is unnecessary. Advanced controls select
streams, retain every Nth frame, limit the longest edge and optionally use faster
8-bit SDR conversion. HDR, float and high-bit-depth input retain the established
conversion path. Full extraction uses the native host buffer when available;
selective extraction retains bounded assembly. See [contracts and benchmarks](video-extraction.md).

## Pinned-memory and prefetch recovery

The fork's registration repair discards a failed temporary host-memory allocation
before evicting other registrations. Retry allocates from the buffer's current
offset, avoiding stale pointers and invalid truncation. Partial prefetch batches
and their VBAR faults remain reachable to existing cleanup after an exception.
Successful sampling math and model interfaces are unchanged.

A running process retains its old implementation and any stale pin state until
a safe restart. These repairs address registration retry and exception cleanup,
not all RAM/VRAM exhaustion. They do not reduce resolution or H3 refinement's
memory requirement. Third-party latent upscalers must separately use managed
model loading.

Set `COMFYUI_PYTHON` to your prepared runtime with pytest, then run:

```bash
export COMFYUI_PYTHON=/path/to/prepared/comfyui/python
OMP_NUM_THREADS=1 "$COMFYUI_PYTHON" -m pytest -q tests-unit/comfy_test/test_pinned_memory_recovery.py tests-unit/comfy_test/test_prefetch_recovery.py
```

The fixtures inject registration/eviction and prefetch failures without GPU work.
Live CUDA inference and capacity limits require separate acceptance.

The optional [archive model cache](model-cache.md) is a different capability:
verified safetensors staging from a slower archive into Linux NVMe storage,
bounded capacity, durable copy recovery and 30-minute inactivity expiry. It runs
inside ComfyUI with a local CLI, not a separate daemon. Custom loaders bypassing
the shared resolver, background loads, GGUF and multi-file repositories remain
outside verified support. Use [the example configuration](../model_cache.example.yaml)
and the guide rather than assuming that every weight path is cacheable.

## Prompt admission

`PromptExecutor.before_prompt_execution(self)` is a synchronous, no-op extension
hook after interrupt/status initialization and before cache evaluation or node
execution. Overrides compose the previous hook and raise to refuse work.
Admission failures and Stop produce prompt-level history events with null node
identity and empty outputs, run lifecycle cleanup and allow the next prompt.
Core owns no service settings or network calls.

A separately installed matching Cutlery-Nodes version can use the hook for
RuntimeHost GPU ownership on every queued CUDA-instance prompt, including cached,
CPU-only or remote-only work. CPU-configured instances bypass GPU acquisition.
That integration and its operating guidance belong to the separate package and
are not installed by this repository. Update paired versions before safe
activation. The CPU check is
`tests-unit/execution_test/test_prompt_admission.py`; it starts no GPU workload.

## Optional H3 workflow examples

The historical local **MiniMax H3 - 360 Orbit** recipe used Pablo Dawson's orbit
LoRA with the same image at both endpoints, 768×768, 73 frames and 28 steps,
saving silent MP4 video under `output/360_orbit`. Its local workflow, toy-house
input and agent skill are not tracked in this public repository. Do not expect
them in a fresh clone or treat the Workflows sidebar as a download catalogue.
Construct a graph only after separately provisioning the LoRA and compatible
nodes and checking their actual schemas. The bundled [H3 blueprints](h3-video-blueprints.md)
have their own requirements and are not a drop-in copy of that local recipe.

A house render on 7 October 2026 produced 73 frames at 24 fps, 768×768 and no
audio, in about 73 seconds on the original host. This is historical local
evidence, not a portable speed claim. The author's training examples were
human-centric; object-orbit quality was not established by the model card.

That Linux RTX 5090 installation also used the separately installed
[ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3) v0.1.9 at
`09a3320205512cd87f6110397fca1e25c19f5536`. In a compatible native H3 graph,
place **Sol-H3 Exact Runtime** or **Sol-H3 SOL Attention (Experimental)** after
other MODEL patches and before the sampler. The latter includes Exact Runtime
and defaults to one dense evaluation and two dense leading layers; sparse
attention approximates the dense result. Refer to that extension's source for
setup and recovery instead of installation-specific paths outside this repository.

GPU kernel/model-patcher checks passed on 7 October 2026; full-size render
quality and end-to-end speed were not measured in that record. Router `sol`
selection then used the separate `SolAttnPatch` Triton extension. This node pack
does not install NVIDIA's standalone Sol Engine. Extension and LoRA licenses
remain with their authors.

## Development and verification

Use the repository's documented prerequisite check with an explicitly prepared
Python runtime:

```bash
COMFYUI_PYTHON=/path/to/prepared/comfyui/python python3 scripts/setup_worktree.py
```

It checks Python >=3.10, PyTorch, einops and safetensors without copying
dependencies, installing packages, loading models or contacting a service. It is
repeatable and leaves tracked files unchanged. Use that interpreter for the
focused tests in [AGENTS.md](../AGENTS.md#worktree-development) and the feature
guides. Optional setup flags check the separately prepared model prerequisites.
The [native integration guide](native-model-integration.md) covers component
selection, offline conversion, memory ownership and reference/lifecycle acceptance.

The root `.gitnexusignore` excludes local environments, installed node packages
and disabled third-party sources from the core graph; those belong to their own
repositories. Unit fixtures and source checks do not certify real-media output.
GPU acceptance uses the main instance's supported queue and requires separately
provisioned checkpoints and a compatible runtime.
