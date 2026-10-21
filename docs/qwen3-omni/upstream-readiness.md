# Quantization compatibility and upstream readiness

Audit: 18 October 2026. Fork base `7aede7957b21300da9797392473bf7110ee65ce2`;
upstream `origin/master` fetched at `872eb6a3a5497721f53ebe82c0da9a3ba3414c13`.
[Accepted artifact and full-model evidence](status.md) remain valid within their
recorded scope. A useful NVFP4-only contribution does not require every format.
This audit is not a claim that the integration is broken or fully upstream-ready.

## What the precision names mean

| Part | Implemented behavior | Evidence boundary |
| --- | --- | --- |
| Weights | Original packed NVFP4 values/scales; lossless layout conversion | 599 native modules; 33 actual ModelOpt component decodes and nine expert paths |
| Activations | Eligible inference paths use three NVFP4 residual passes, FP16 products and FP32 sum/bias, then compute dtype | Approximates a higher-precision activation matmul; neither FP8 activations nor publisher W4A8 parity |
| KV cache | BF16 in the accepted runs, scoped to one generation | No FP8 storage, scale policy or FP8 attention/cache acceptance |
| Decoded BF16 reference | The same quantized weights decoded for computation | Does not recover or validate original BF16 weights |

`full_precision_mm=True` initializes a safe decoded policy. The native generation
owner's `use_quantized_matmul` context temporarily enables supported formats on
eligible modules. Descriptor overrides, disabled formats, input gradients and
patch/cast paths can select different execution. Dense Linear gradients use the
existing `QuantLinearFunc`; expert input gradients now use decoded differentiable
matmul. The residual helper is inference-only. No claim that every projection
always uses the fast path is justified by the constructor argument alone.

## Publisher reference: a concrete blocker

The pinned source declares ModelOpt 0.42.0, NVFP4/group-16, static 8-bit float
inputs and FP8 cache. Its actual tensor inventory has weight/block/global scales,
but **no `input_scale`, `k_scale`, `v_scale` or quantizer calibration state**.
There is no pinned publisher inference implementation or retained publisher output
establishing how these settings were executed. Inventing dynamic activation/cache
scales would define another numerical path.

Controlled local checks in the isolated Python 3.12/Transformers 5.20.0 reference:

- `AutoQuantizationConfig.from_dict(source_text_config['quantization_config'])`
  raises `ValueError: Unknown quantization type, got modelopt`.
- `NVFP4HfQuantizer(NVFP4Config(), pre_quantized=True).validate_environment(...)`
  rejects prequantized loading before GPU initialization.
- ModelOpt's `W4A8_NVFP4_FP8_CFG` exists, but specifies block-32 weights and an FP8
  input quantizer. Selecting it does not restore this group-16 export's omitted
  static activation calibration. The producer's existence is not an executable
  checkpoint loader.

[Probe errors, versions and executed-source/config hashes](results/publisher-runtime-audit.json)
retain the actual checks and auxiliary tensor inventory.

These checks do not load a full model or change either environment. The existing
ModelOpt decoder reference remains useful and explicitly uses BF16 compute/cache.
[Reference commands and locked versions](reference.md) identify that environment.

Current [vLLM-Omni guidance](https://docs.vllm.ai/projects/vllm-omni/en/stable/user_guide/quantization/modelopt/)
documents a different validated W4A4 checkpoint,
`YihongJin/Qwen3-Omni-30B-A3B-Instruct-NVFP4-W4A4-full-thinker-awqclip`, with
BF16 media encoders. Its
[v0.18 guide](https://docs.vllm.ai/projects/vllm-omni/en/v0.18.0/user_guide/diffusion/quantization/overview/)
explicitly labels the selected shunyang90 checkpoint experimental with unacceptable
output quality on that runtime. These are provider reports, not local validation;
the validated alternative does not establish the selected source's W4A8 execution.
No vLLM environment or alternative weights were downloaded/installed in this audit.

Priority: evaluate that documented W4A4 export's immutable inventory and reference
before extending the converter, or obtain the selected publisher's exact runtime,
calibration and reference outputs. FP8-cache work also needs a defined scale/cache
layout and a supported backend reference. A speculative cache implementation would
not close this evidence gap. BF16/FP8/INT8/GGUF are separate checkpoint/container
projects, not prerequisites for this narrowly scoped integration.

## Residual scheme: benefits and limits

The previous actual-weight CUDA evidence covers 33 projections at 1/16/128 rows.
Three FP16-output passes had worst relative RMSE 0.004482 and median 0.371 ms.
The frozen full-model comparison had zero new assertion failures, with both paths
passing 20/24. One-pass CUDA and two-pass probes had additional audio failures;
matching precision does not guarantee identical tokens. The original source's
Unicode/OCR/silence and modality-removal errors must stay reported.

The new [CPU audit](results/quantization-audit-cpu.json) exercises synthetic
attention, fused gate/up, expert down, audio FC1, padded vision FC2 and vision
merger shapes at 1/16/128 rows. Against FP32-decoded weights with BF16 final output,
three passes have relative RMSE 0.00284–0.00338 (one pass about 0.080–0.097).
The CPU matmul is an explicitly substituted FP32 oracle with FP16 partial rounding;
it establishes composition/numerical behavior, not a CPU fast backend or GPU speed.
Its one-pass residual probe uses FP16 output; the shared production one-pass
default normally uses the selected backend's compute-output dtype.

An explicit range probe with ones weights `[32,64]` and constant 2048 inputs
produces nonfinite residual output, although decoded BF16 yields 131072. FP16
partial products overflow before FP32 accumulation; extra passes cannot repair it.
This is a real generality limit, not a failure observed in the accepted fixtures.
The installed CUDA operation advertises only FP16/BF16 outputs, not FP32. Switching
to BF16 partials changes the accepted precision/quality behavior. Any remedy needs
new numerical and frozen full-model quality checks before replacing the current
policy. No silent fallback, custom kernel or unsupported dtype was introduced.

Three passes require three activation quantizations/matmuls, two activation decodes
and FP32 result temporaries per eligible projection. Packed weight residency is
preserved; temporary activations/results still cost memory. The measured end-to-end
5.01 s warm first token and 2.34 tokens/s are the useful throughput evidence;
component timing alone cannot establish a speedup over decoded execution.

The new [guarded CUDA audit](results/quantization-audit-cuda.json) completed all
18 synthetic shape/row probes on the RTX 5090. Three passes have worst relative
RMSE 0.002758; the range-overflow probe also reproduces on the real CUDA kernel.
Median warm component timings across these probes are:

| Operation | Median ms |
| --- | ---: |
| One residual pass, FP16 output | 0.124 |
| Three residual passes | 0.406 |
| Decode packed weight, then BF16 matmul | 0.0397 |
| Already resident decoded BF16 matmul | 0.0149 |

Three-pass temporary Torch allocation peaks at 10,643,968 bytes in this probe set.
These measurements use five warmups and 20 CUDA-event samples per operation, with
packed/decoded fixtures already on device. They omit model loading, bank casting,
prefetch and generation. They show a substantial component advantage for existing
decode-plus-BF16 operations over the residual composition; a first priority is a
controlled end-to-end/lifecycle comparison of that packed-storage, decoded-compute
path. Do not change the accepted default based solely on synthetic operator timings.
The earlier “faster than eager” measurement compared quantized backends and did
not establish an advantage over decoded BF16 matmul.

The shared layer already preserves full-precision/patch owners. This audit fixes
one demonstrated defect: a gradient-bearing expert input previously reached the
non-differentiable NVFP4 decoder during backward. It now selects the existing
decoded matmul. Tests cover backward and forward equality for one/three passes.
Inference and other models' one-pass default remain unchanged.

## Portability work completed here

Native loading now permits dense BF16/FP16/FP32 projections mixed with NVFP4 in the
existing normalized package. Previously it required the selected source's exact
quantization choices even though the operation layer already handles dense banks.
It checks all required weights and NVFP4 weight/blocked-scale/global-scale shapes,
dtypes and expert counts before state loading. Removing descriptors cannot turn
uint8 weights into dense values. All 599 accepted artifact headers pass; tiny
dense and mixed fixtures execute dense and indexed expert projections correctly.

The converter remains deliberately hash/revision-specific, including the narrow
stale-index correction. It still expects split NVFP4 experts, identical gate/up
global scales, native key mappings and padded vision columns. A new source requires
a separate validated inventory/conversion recipe. The detector still requires the
embedded assets and selected architecture signature. No complete original BF16,
other quantization, Talker, video, history or speech-output support is added.

## Verification and reproduction

The prepared runtime is Python 3.13.16, Torch 2.14.1+cu130, Comfy Kitchen 0.2.37 and
Transformers 5.20.0. CPU audit source hashes are retained in its JSON. It validates
the installed artifact's headers on meta without allocating/loading its weights;
the previous artifact payload checksum remains a separate integrity record.

```bash
CUDA_VISIBLE_DEVICES='' /path/to/prepared/python -m pytest \
  tests-unit/comfy_quant tests-unit/comfy_test/qwen3_omni_test.py \
  tests-unit/comfy_test/qwen3_omni_quant_test.py \
  tests-unit/comfy_test/qwen3_omni_conversion_test.py \
  tests-unit/comfy_test/text_generation_length_test.py \
  tests-unit/comfy_test/qwen_vl_image_channels_test.py -q

CUDA_VISIBLE_DEVICES='' /path/to/prepared/python \
  tests-inference/qwen3_omni_quantization_audit.py \
  --artifact /absolute/accepted.safetensors --output /absolute/new-cpu-audit.json
```

The combined suite passes 92 tests, with one expected CUDA skip. A separately held
RTX 5090 window passes all 12 residual unit tests, including real CUDA zero/bias
checks. The complete CUDA shape audit passes its queue/ownership checks before
publication, and its hold was released after process exit. Earlier contended,
unheld or interrupted attempts are excluded. No new
full-model/lifecycle/throughput result is claimed; the changes preserve the prior
inference calculation and the [retained measurements](status.md).
Ruff is unavailable in the prepared environment; syntax compilation and
`git diff --check` pass. No package was installed solely for linting.

For CUDA shape probes, first obtain an exclusive RuntimeHost hold through its
owner, with no active/pending ComfyUI work. Pass `--device cuda --runtime-host-url
URL --hold-id ID --owner-id ID` and the configured `RUNTIME_HOST_API_KEY` environment
value. The tool checks live ownership, stopped worker and the serving queue before
allocation, between probes and before publication. It never acquires/releases a
hold or unloads serving models. Stop/recover uncertain processes before releasing
the exact caller-owned hold. Use a new output path after refusal; no report means
no completed audit. ROCm, MPS, CPU accelerated execution and other GPU generations
remain unverified; CPU oracle tests do not imply those backend capabilities.

## Upstream verdict and proposed patches

GitHub issue/PR searches for `Qwen3-Omni` in titles and `qwen3_omni` in all searchable
text returned zero matches on 18 October; the broader Qwen3/Omni search returned
unrelated results. This is search evidence, not proof no private or unindexed work
exists. Current upstream already contains the base MoE, Qwen3-VL/3.5 vision,
prefetch and quantized-operation helpers; the Omni-specific adapter is absent.
[Contribution guidance](https://github.com/Comfy-Org/ComfyUI/wiki/How-to-Contribute-Code)
asks for a wanted change, small reviewable patches, compatible licensing, reference
outputs/settings and optimized attention. The audio/text equivalent of its image
reference example needs an explicit reviewer-readable fixture/answer comparison.
No issue, comment or PR was published, and maintainers were not contacted.

1. **Fix input gradients through quantized MoE experts.** Small independent patch:
   gradient guard plus a native shared-op regression test, ported onto current
   upstream without depending on the fork's residual option. Proposed body:
   “Gradient-bearing expert inputs can enter inference-only quantization kernels
   without backward formulas. Use the existing decoded differentiable path for
   these inputs; retain quantized inference. Validate forward/input gradients and
   ordinary quantized inference.” Needs an upstream-base verification run.
2. **Add native Qwen3-Omni Thinker text, image and audio generation.** Minimal
   architecture, guarded detection, embedded tokenizer/media assets, loader choice,
   pinned offline converter and tests/reference fixtures. Clearly label supported
   source, omitted speech/video, BF16 cache and the alternate activation policy.
   Proposed body: “Load a converted NVFP4 Thinker as CLIP for Generate Text with
   text, image batches and audio. Preserve packed weights and native model lifecycle.
   Frozen reference/native runs pass the same 20/24 cases; retained source failures
   and precision/resource limits are documented.” Do not claim publisher parity.
3. The residual operation is a separately reviewable prerequisite if used by the
   architecture patch. General adoption needs a bounded precision contract and
   resolution or explicit reviewer acceptance of the FP16 range limit. Keep it
   opt-in; do not change all NVFP4 models to three passes. A documented supported
   W4A4 export may remove the need for this source-specific precision workaround,
   but that remains to be measured.

Extract by behavior, not by cherry-picking the whole integration history. The fork's
`comfy/sd.py` includes unrelated native audio loaders and an admission/idle-eviction
fix; `nodes_textgen.py` includes history, extra outputs and automatic-budget changes.
Omni also calls the fork's `BaseGenerate.resolve_max_length`, absent in this upstream
snapshot. Explicitly resolve that small dependency or keep upstream's positive-token
budget contract; do not import unrelated history/automatic-budget work. Existing
upstream vision/MoE helpers can be reused directly. Preserve Qwen/Transformers
Apache-2.0 attribution and ComfyUI's GPL licensing.

Ready now: accepted local selected-artifact use and focused fork fixes. Before an
architecture submission: verify the extracted patch on current upstream, provide
portable offline fixtures/reference commands, settle the residual precision contract
or evaluate the documented alternative, and complete isolated full-model checks for
that extracted version. Publisher W4A8/FP8-cache parity and every checkpoint format
are neither established nor mandatory promises of a scoped first contribution.
