# Integrating a model into native ComfyUI

Use this guide when asked to integrate or port an upstream model, convert its
checkpoint for ComfyUI, or replace an external inference wrapper with native
execution. It consolidates the development procedure; [AGENTS.md](../AGENTS.md)
remains the implementation policy. Read the existing `comfyui` and
`comfyui-model-runtimes` skills through `skills show <name>` for workflow operation
and native-versus-isolated placement respectively.

The intended result is a model whose weights, compute policy, loading, offloading
and cancellation participate in ComfyUI's existing owners. A safetensors file or
a node that launches upstream inference does not establish that result. Reuse
standard loaders and nodes where their interfaces fit; some model families need
a small dedicated loader returning a patcher-managed handle.

This guide adds no model support or deployment. Examples below describe existing
ports or explicitly labelled scoping decisions. Hardware acceptance is specific
to the tested checkpoint, precision, inputs and device.

## 1. Establish the inference target

Inspect upstream code, model files and configuration before deciding what to
build. A Hugging Face repository can contain application releases, training
pretrains, several architectures or supporting assets rather than one usable
inference checkpoint. Repository tags and generated loading snippets are not
proof of its format or dependencies.

Record a short integration brief in the model documentation:

| Field | Required information |
| --- | --- |
| Sources | Upstream code and model revisions, licenses, model cards and artifact hashes |
| Behavior | Task, input/output types, supported variants, references and required preprocessing |
| Components | Inference networks, tokenizer/config assets, optional indexes and training-only tensors |
| Native mapping | Existing loader/model/node interfaces to reuse; any necessary dedicated interface |
| Compute | Storage/compute/cache dtypes, quantization layout, supported devices and required kernels |
| Resources | Weight bytes, source/output disk space, conversion RAM, activation/cache/workspace estimates |
| Reference | Reproducible upstream inference, fixed fixtures, comparison boundaries and tolerances |
| Scope | Implemented target, explicit exclusions, unresolved choices and acceptance gates |

For an underspecified request, investigate first and state a reasonable inference
scope. Ask only about choices that materially change the implementation, such as
which incompatible checkpoint family or trained voice is intended. Continue
independent inspection while awaiting an answer. Do not silently add training,
realtime streaming, a desktop application, hosted services or every upstream tool.

Prefer native execution when ComfyUI's operations and patchers can support the
model. Demonstrate a concrete dependency/process conflict before selecting an
isolated runtime. Keep an incompatible upstream reference environment separate
from serving ComfyUI; a reference dependency conflict alone does not require
external production inference. Consult `comfyui-model-runtimes` for deployed
external capabilities; planned bridges or profiles are not executable fallbacks.

## 2. Prepare the repository and source assets

Check the explicitly configured prepared runtime with the repository-owned command:

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py
```

The explicit Python must be >=3.10 and already provide PyTorch, einops and
safetensors. Model-specific setup options are documented in AGENTS.md. Setup is a
read-only runtime check; it installs nothing, loads no models and starts no service.
Use that interpreter for subsequent checks.

Inventory existing files and download jobs before acquiring missing assets.
Downloads use Download Manager 2 with explicit destinations and the required scoped
Firefox Hugging Face cookie transfer. Wait for verified complete files when
conversion depends on them. Keep originals outside Git. Model loading, node
imports and inference must never download assets or contact upstream services.

## 3. Map components to the right owners

| Component | Existing integration boundary to inspect |
| --- | --- |
| Diffusion/flow network | `comfy/model_detection.py`, `comfy/supported_models.py`, `comfy/model_base.py`, model implementation and native latent format |
| Conditioning encoder or generative text/multimodal model | `comfy/sd.py` detection/CLIP construction and `comfy/text_encoders/`; reuse tokenization, encode/generate/decode contracts |
| VAE or audio codec | Existing VAE detection/construction in `comfy/sd.py` and the corresponding implementation |
| Audio/vision encoder or specialist network | Existing category loader and patcher-managed model pattern; a dedicated handle when its contract differs |
| User controls and media adaptation | Minimal nodes following the file's registration conventions, ordinary IMAGE/AUDIO/CONDITIONING outputs where applicable |
| Loading and residency | Existing ModelPatcher/CoreModelPatcher, model management and memory-management facilities |
| Optimized math | Comfy Kitchen, `comfy/ops.py`, `comfy/quant_ops.py`, shared attention and `comfy/model_prefetch.py` |

Choose by behavior. An audio content encoder is not automatically CLIP; a
waveform generator is not automatically a diffusion MODEL or VAE. Conversely,
ComfyUI's CLIP interface can support generative text without implying diffusion
conditioning, as in Qwen3-Omni. Document the actual contract.

Keep model-native math, modality preparation and latent layout in their model
owners. Put node/request mapping at the boundary that owns it. Do not teach
`execution.py` about model-specific UI identities or add a parallel loading policy.
Preserve existing public signatures, node IDs and workflows unless replacement
is explicitly requested. Validate file selections at the load/save boundary using
the existing category resolver or containment helper.

## 4. Convert only what the native implementation needs

First determine whether existing loaders already accept the source representation.
If conversion is necessary, provide a repository-owned offline command with
explicit local source and destination paths. Use the existing conversion helpers
where suitable; inspect the converters linked below before creating another writer.

The converter must:

- Validate the selected architecture, configuration, complete shard inventory,
  actual tensor shapes/dtypes and payload integrity. Reject LFS pointers and
  incomplete sources; preserve the existing safe checkpoint-loading boundary.
- Map keys and split components into appropriate category artifacts. Preserve
  configuration, tokenizers, special tokens, preprocessing settings and required
  auxiliary assets so inference does not depend on the original HF folder.
- Preserve retained values and quantization metadata. Account explicitly for
  omitted training/output-only tensors and mathematically required transforms,
  such as folded weight normalization. Do not silently cast, requantize or discard
  unsupported projections to make loading succeed.
- Bound peak conversion memory, preferably by streaming large checkpoints.
  Document originals plus temporary/output disk requirements. Avoid materializing
  a full dense model just to repackage packed weights.
- Preserve originals, refuse accidental overwrite, verify saved payloads, and
  publish a manifest recording revisions, hashes, tensor mapping/disposition and
  configuration. Retain credits and relevant redistribution notices.
- Define interruption recovery and publication order. A partial artifact or a
  manifest without its complete component set is not loadable success. Resume
  only after verifying completed artifacts; otherwise use a fresh destination.

For quantized models, validate packed layout, scales, zero points and expert-bank
transforms against an actual reference decoder before full-model integration.
Equal decoded shape does not establish numerical equivalence. Model detection
must guard every dereferenced key and use only the first dimension of linear
weights for shape signatures; packed second dimensions can differ. Put broad
signatures after established/specific ones and test near-matches for regressions.

## 5. Implement native execution and memory ownership

Use existing compatible optimized operations first, including fused, paired,
layout-specific and quantized variants. Preserve the model's exact epsilon,
scaling, tensor layout and dtype contracts. Benchmark representative shapes when
choosing between compatible optimized paths. Keep a local fallback only for a
demonstrated missing contract, such as differentiation or dynamic patches.

Use the caller-supplied ComfyUI operations, per-module `ComfyAttention`,
`AttentionTensorContainer`, selected attention via `preferred_attention`, and the
existing memory compiler/prefetch block-loop patterns. Treat backend dispatchers
as opaque. Follow current working models for setup and cleanup rather than copying
an old upstream implementation's device handling.

Weights belong to the patcher/loading owner. Do not independently move whole
models to CUDA, retain a private model singleton, use Transformers `device_map`
as a second residency owner, or unload other models from inside model math.
Use existing cast/offload helpers for raw parameters and keep compute policy
outside model constructors. Do not add inference-mode wrappers or freeze toggles.

Keep KV caches, media features and other large temporary tensors local to one
execution and pass them explicitly. Precompute conditioning through the existing
model preprocessing/`extra_conds` pattern when it avoids repeated denoising work.
Account for activations, expanded media, generation caches and workspace through
the established admission/estimation interface, not just checkpoint bytes.
Memory estimates require representative measurements; a small fixture cannot
justify a universal VRAM claim. Check shared-path implications for CPU and the
supported accelerators; report untested platforms explicitly.

Reuse sampling and output paths when the model fits them, preserving schedule,
time/velocity conventions, guidance, codec normalization and sample rate. Add
only controls the implementation reads. Optional inputs stay optional where a
valid no-input path exists. Preserve executable user values; adapt only where
the underlying operation would otherwise fail.

## 6. Establish correctness and lifecycle acceptance

Freeze fixtures and tolerances before evaluating the candidate. Retain source
revision, converted manifest/hash, interpreter/kernel versions, settings, seeds,
input hashes and output/evidence paths.

| Gate | Required evidence |
| --- | --- |
| Conversion | Complete inventory, saved-value/decoded-layout comparisons, metadata and missing/corrupt/partial-file failures |
| Component math | Relevant preprocessing, positions/masks, attention, projections, routing, encoder/decoder and cached-step comparisons |
| Full checkpoint | Real native loader construction, complete tensor coverage and representative upstream comparisons at named precision |
| Behavior | Each supported modality/reference alone and combined; boundary inputs; controlled removal/change of conditioning |
| Lifecycle | Repeated runs, cancellation during meaningful stages, recovery, explicit unload/reload and another-model transition back |
| Resources | Cold/warm timing, process RAM, Torch/driver VRAM and release observations with named inputs/device; low-VRAM checks where claimed |
| Product quality | Inspect/listen to actual outputs against the requested task; distinguish upstream limitations from port regressions |
| Regression | Relevant existing loader/model/node checks and compatible patch/quantization paths affected by shared changes |

Numerical tolerances must reflect the precision and operation. Exact token or
waveform equality is not a universal requirement, but fluent text, finite audio
or successful construction alone does not demonstrate correctness. Source and
candidate can agree on an incorrect answer; record that failed quality gate.
Skipped reference/GPU checks remain unverified, even when CPU fixtures pass.

Submit native inference through the ordinary ComfyUI queue even when busy. Retain
the prompt ID and inspect queue/history after a timeout before retrying. Isolated
upstream GPU comparisons need the existing coordinated ownership protocol; do not
run an uncoordinated second model process alongside queued work. Never release
another task's hold or infer GPU release from a lost connection. Investigate
lifecycle failures through the existing owner and retain failed-run evidence.

## 7. Deliver and distinguish activation from implementation

Update the root README, model-specific setup/conversion/use/recovery docs and
existing authoritative operating skills in the same change. Use
`shared-agent-skills` to find sources and validate/reconcile discovery; never
register a task worktree globally. Keep the main skill concise and link detailed
references. If an HTTP/MCP contract changes, follow `agent-readable-api-contracts`
and update its consumers and required checks together.

Provide an editable workflow in the configured user workflow library and, when
useful, a matching API graph. Discover actual filenames and live node schemas;
do not invent a loader selector or claim a newly merged node is already running.
Review the cumulative diff and run required checks, then commit, push, integrate
and clean up through repository policy.

Report source integration, conversion, candidate inference, quality acceptance
and serving activation separately. Integration does not itself authorize a
restart. Before an authorized activation, check active work through the service
owner; preserve completed commits if rollout is blocked. After activation, verify
live schemas and representative normal-queue inference. State remaining limits
instead of converting an unrun gate into a success claim.

## Worked scoping example: Applio

The original scoping investigation led to the [native RVC file-conversion path](applio.md).
Its [acceptance record](applio-acceptance.md) distinguishes implementation, checkpoint
comparisons and activation. Sources checked
19 October 2026: the [HF repository](https://huggingface.co/IAHispano/Applio/tree/main)
contains application/resource bundles; upstream describes separate
[voice exports and optional indexes](https://docs.applio.org/getting-started/installing-inference-models/).
Pin actual code and artifact revisions before implementation.

The documented [RVC pipeline](https://docs.applio.org/reference/architecture/)
analyzes input audio, extracts content and optional pitch, optionally retrieves
features, then synthesizes the target voice. A scoped first port would target file
voice conversion for a named compatible inference export. Identify its embedder,
pitch mode, vocoder and metadata. The [training embedder must match inference](https://docs.applio.org/getting-started/embedder/).
Training checkpoints and exported inference weights are different artifacts.

Inventory the chosen speaker generator, compatible content encoder, required F0
predictor and optional matching retrieval index. Inspect existing native audio
components for genuine mathematical reuse. Choose patcher-managed interfaces for
the networks; keep retrieval data in an explicitly owned handle with bounded
lifetime. Determine category/loader choices from those interfaces rather than
forcing the generator through diffusion loading or the embedder through CLIP.
Any conversion must preserve voice/configuration metadata and index association.

Preserve upstream analysis/synthesis rates, pitch and retrieval controls, segment
handling and output amplitude. Test reference agreement, audible conversion,
unvoiced regions, boundaries, optional-index behavior and memory lifecycle. Keep
training, realtime capture, upstream UI, unrelated effects and TTS outside the
first inference scope unless requested. Supported variants and node names are documented in the native RVC guide;
launching Applio's CLI would be an external wrapper, not this native port.

## Existing ports to inspect

| Example | Reusable guidance and owner commands |
| --- | --- |
| [AuK-Flash](auk-flash.md) | MODEL + conditioning CLIP + VAE, schedule/latent conventions; [`convert_auk.py`](../scripts/convert_auk.py) |
| [YingMusic](yingmusic.md) | Diffusion/audio-encoder/VAE categories and native flow adaptation; [`convert_yingmusic.py`](../scripts/convert_yingmusic.py) |
| [MOSS-Audio](moss-audio.md) | Streaming shards, tokenizer packaging, provenance and reference/lifecycle comparisons; [`convert_moss_audio.py`](../scripts/convert_moss_audio.py) |
| [Audio Flamingo](audio-flamingo.md) | Native encoder/projector/text generation with local processor assets; [`convert_audio_flamingo.py`](../scripts/convert_audio_flamingo.py) |
| [Qwen3-Omni](qwen3-omni/README.md) | Packed NVFP4 conversion, reference reconstruction, multimodal and precision gates; [`convert_qwen3_omni.py`](../scripts/convert_qwen3_omni.py) |
| [SoulX-Singer](soulx-singer.md) | Dedicated managed handles, component splitting and round-trip verification; [`convert_soulx_singer.py`](../scripts/convert_soulx_singer.py) |

Use each linked model document's exact command, prerequisites and recovery rules;
these converters have different arguments and supported source revisions. For
Omni, the [status record](qwen3-omni/status.md) supersedes historical proposed or
pending wording in its original design/implementation plan. No example establishes
compatibility with another checkpoint or availability on another deployment.
