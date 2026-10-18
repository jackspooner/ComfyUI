# Published FP8 candidate and INT8 scope

Status, 19 October 2026: the published source passed every pinned hash and all
54,494 tensor shapes/dtypes. Full-model testing caught a source scale convention
incompatible with the standard HF decoder: multiplication generates token zero,
while division passes the independent arithmetic smoke. The converter now
normalizes these pinned multipliers. The completed native suite passes 18/24,
versus 15/24 for the independent reference, but adds one assertion failure in
`combined_batch`. Lifecycle checks pass with 1 GiB extra headroom. This candidate
is **experimental, not accepted for deployment**; the installed NVFP4 artifact and
serving service are unchanged. [Full evidence](results/published-fp8-validation.json)
records the quality gap and resource results.

The selected source is [marksverdhei/Qwen3-Omni-30B-A3B-FP8](https://huggingface.co/marksverdhei/Qwen3-Omni-30B-A3B-FP8),
revision `042db39a102171d5994b72ab717fade92e80061a`, based on the Instruct model.
This route uses existing quantized weights; original BF16/FP16 model execution
and the 70.52 GB original-weight download are unnecessary. BF16 and INT8 coverage
below is theoretical; no full-model BF16 or INT8 execution is claimed.

## Pinned source and assets

The [source inventory](../../scripts/qwen3_omni_fp8_source.json) pins 15 weight
shards and eight publisher metadata files. Download through Download Manager 2
with the scoped Firefox-cookie procedure. Metadata was verified against published
Git blob IDs; every source file SHA-256 and size is mandatory before conversion.
Weights occupy 37,378,054,674 bytes, with 37,370,856,674 bytes of tensor payload.
The publisher index retains the original BF16 total of 70,519,637,090 bytes. This
specific hash-pinned reconciliation never permits arbitrary stale metadata.

The repository omits its chat template. Add only the 6,772-byte
`chat_template.json` from `Qwen/Qwen3-Omni-30B-A3B-Instruct` at
`26291f793822fb6be9555850f06dfe95f2d7e695`, using the exact SHA-256 recorded in
the inventory. The local template already obtained through Download Manager 2
was verified and copied into the candidate source directory. Its separate source
provenance is retained. No remote code or automatic Hub downloads are used.

The native plan occupies 33,555,117,814 bytes including embedded assets, before
its safetensors header and provenance manifest. Reserve at least 75 GB free for
source plus native output, additional headroom for fixtures and other workloads,
and inspect current space before starting. A published download size is not a
VRAM estimate. Preserve existing models and active task data.

## Weight-preserving block FP8 packaging

```bash
/path/to/prepared/python scripts/convert_qwen3_omni.py \
  --source /absolute/published-fp8-source \
  --source-revision 042db39a102171d5994b72ab717fade92e80061a \
  --quantization fp8-block128 \
  --destination /absolute/new-qwen3-omni-fp8.safetensors
```

This mode preserves the published E4M3 weight bytes. In this exact source,
BF16 `weight_scale_inv` holds quantization multipliers: decoded weights are
`weight / weight_scale_inv`. The converter stores their FP32 reciprocals as native
128×128 block `weight_scale`, whose usual contract is multiplication. No heuristic
scale inversion or change to other FP8 formats is used. It fuses aligned gate/up
rows in expert order without requantizing weights. The native Thinker has
288 FP8 projections/banks; audio/vision encoders, norms, routers, embeddings and
output head retain BF16 weights. Dense vision input columns receive only the
native zero padding. Talker/code2wav are excluded and inventoried.

Native descriptors use `float8_e4m3fn_block128` and
`full_precision_matrix_mult: true`: **FP8 weight storage with BF16 compute and
cache**, not publisher dynamic FP8 activations or an FP8-cache claim. The shared
layout decodes 128×128 floating scales in FP32 before casting to compute dtype and
using existing matmul, cast/offload and expert-prefetch owners. Kitchen's existing
per-tensor FP8 and 32-element E8M0 MXFP8 layouts cannot represent these scales.
Other backends remain unverified until separately exercised.

The writer streams projection/expert groups, checks finite quantized data and
positive finite source multipliers, embeds local processor/tokenizer/template/config assets,
then verifies output checksums and publishes the checkpoint plus manifest without
overwrite. Originals remain unchanged. An orderly failure removes its temporary
files. After process death inspect exact owned temporary files and a possible
orphan manifest before retrying to a new destination; never overwrite an accepted
candidate. Invalid layouts, missing scales and expert-count mismatches fail before
state loading. Scalar FP8 scales retain their separate descriptor and validation.

## Independent validation

The reference keeps upstream Transformers 5.20.0 architecture and dispatch, and
uses its `Fp8Dequantize` decoder independently of the native layout, after explicit
FP32 reciprocal normalization of this publisher's source multipliers. FP8 expert
buffers remain CPU file-backed and only selected matrices move to the requested
device. Compute/cache are BF16. This avoids materializing the original BF16 model;
CPU offload can make the reference slow. ModelOpt 0.42.0 remains required only for
the existing NVFP4 reference and its separate supported Python runtime.

Run `scripts/qwen3_omni_reference.py` with the source/revision, frozen 24-case
fixture file and new output directory, then `tests-inference/check_qwen3_omni.py`
with the native artifact, the same fixtures and `--precision fp8-block128`.
For this 32 GiB RTX 5090 validation, pass `--vram-headroom 1`, matching the existing
serving CLI policy. With zero extra headroom, the corrected model answered `12`
then rejected the next call: 703,037,440 bytes were required including reserve,
but only 517,908,480 were free after weights remained resident. The explicit
headroom setting is recorded separately and does not change precision or fixtures.
Use `--capture-parity` / `--capture` for features, positions, routing and logits;
`tests-inference/compare_qwen3_omni.py` refuses mismatched source revisions.
CPU component probes use `--components --device cpu` without loading the model.
GPU runs require the established exclusive validation window/RuntimeHost hold;
inspect queues and owner state, retain failures, confirm process exit and release
only that exact hold. Never interrupt another owner or restart an active service.
Installation/activation still requires accepted quality, resource and lifecycle
checks and the established deployment workflow.

## Optional original-weight experiment

The earlier `--quantization fp8` experiment quantizes the pinned original BF16
source into per-projection/per-expert FP8 with FP32 scales. Its synthetic rounding,
zero-expert, padding and refusal tests pass, but no complete original-weight model
was downloaded, converted or executed. It is not used for this selected published
candidate. Its original metadata accounting is retained in
[the preparation report](results/eight-bit-preparation.json). The default
`--quantization nvfp4` still preserves the accepted pinned ModelOpt source.

## What can be established theoretically

[Retained preparation accounting](results/eight-bit-preparation.json) records
versions, source hashes, tensor/key counts, planned bytes and the unexecuted status.

- BF16 and FP16 store two bytes per element; original Thinker weights alone need
  about 63.44 GB before native padding/assets. They cannot all reside in this
  GPU's roughly 32 GiB. Streaming/offload can change feasibility, so this is not
  a proof that execution is impossible. Full-model runs remain excluded here.
- Signed tensor-wise INT8 would use the same unpacked eight-bit weight shapes and
  FP32 scalar-scale counts as the optional original-source scalar FP8 plan. This establishes a storage estimate,
  not a converted INT8 artifact or model-loading/quality acceptance.
- For nonzero weights under symmetric nearest-rounded INT8, an exact scale `s = max(abs(w))/127` gives
  `abs(w - s*round(w/s)) <= s/2` within the represented range. Finite scale/arithmetic
  rounding adds its own error. Zero matrices are exactly representable using zero
  integers and a positive scale. A linear output obeys the corresponding absolute
  bound `sum(abs(x))*max(abs(weight_error))`.
- That local error bound cannot prove equal generated tokens or perceptual quality:
  near-tied expert routing and output logits can change discrete choices. BF16
  dtype/layout compatibility likewise does not prove full-model numerical stability.
- The existing batch-one BF16 text KV cache needs 98,304 bytes per expanded token:
  48 layers × 4 KV heads × 128 dimensions × K/V × 2 bytes. At 65,536 tokens that is
  6 GiB before attention/media workspace, even with eight-bit stored weights.

## Verification and acceptance limits

The cumulative unit command passes **121 tests**, with one expected owned-CUDA
skip, using Python 3.13.16, Torch 2.14.1+cu130, Transformers 5.20.0 and Comfy Kitchen
0.2.37:

```bash
CUDA_VISIBLE_DEVICES='' /path/to/prepared/python -m pytest \
  tests-unit/comfy_quant tests-unit/comfy_test/qwen3_omni_test.py \
  tests-unit/comfy_test/qwen3_omni_quant_test.py \
  tests-unit/comfy_test/qwen3_omni_conversion_test.py \
  tests-unit/comfy_test/text_generation_length_test.py \
  tests-unit/comfy_test/qwen_vl_image_channels_test.py -q
```

New checks cover direct source division versus normalized native decoding,
independent FP32-to-FP8 rounding, per-expert scales, all-zero
experts, gate/up order, padding, dense exclusions, missing/invalid native layouts,
NaN refusal, insufficient-space cleanup and rejection of NVFP4 as an FP8 source.
Seeded requantization is covered for patched block FP8 weights. The earlier
accepted NVFP4 tests still pass. Six RTX 5090 component probes match
the pinned Transformers decoder and BF16 matmul outputs exactly; decode plus
matmul takes 0.022–0.054 ms in these synthetic shapes. [Retained evidence](results/published-fp8-components.json)
records shapes and source hashes. Historical real-source probes in that report
matched the upstream decoder under the wrong multiplicative interpretation; they
do not establish usable model semantics. The [scale correction record](results/published-fp8-scale-normalization.json)
retains the failed native/reference smoke, 28 comparisons against a separately
quantized NVFP4 source and the passing reciprocal reference smoke. This comparison
supports scale interpretation, not original BF16 quality parity. These timings
exclude loading/offload and are not full-model latency. Other backends remain
unverified.

Corrected packaging is complete: 33,555,350,854 bytes, SHA-256
`bb454fc267e2b46a701defb15c800a9638f6f22e351f200543d0924110f4b443`.
Both implementations completed the same 24 cases without reducing token budgets.
Native passes 18, reference 15; one previously passing assertion regresses in
`combined_batch`. Generated IDs match for 16 cases. All 24 position tensors and
all 11 compute-image inputs match exactly; captures have no shape mismatches or
non-finite values. Encoder features, routing and logits differ, and the cause has
not been isolated. Overall score improvement does not establish parity.

On the RTX 5090 with `--vram-headroom 1`, native peak driver VRAM was
28,160,557,056 bytes and median recorded decoding throughput was 3.31 tokens/s.
These are this fixture run's measurements, with existing serving residency and
uncontrolled OS file cache, not a general performance claim. Reference CPU
offload peaked at 5,293,211,648 driver bytes. Torch-only accounting misses the
native allocator's resident pages; use the driver measurements.

All twelve lifecycle steps pass: three identical repeats, cancellation during
preprocessing/prefill/decode, recovery, offload/reload, transition through the
existing MOSS model (`READY`), Omni recovery and final offload. The retained Torch
allocation delta after cancellation is zero. Process exit and release of the exact
GPU hold were confirmed; no serving restart or model replacement occurred.

The implementation is pushed on `omni-fp8-int8`; integration remains blocked by
the exact-worktree GitNexus writer's `graph-incomplete` refusal. Its previous valid
graph is preserved; no force, hidden source or manual owner-state changes were
used. The worktree, source, corrected candidate and failed-run diagnostics remain.
Further acceptance needs investigation of the numerical/combined-batch regression
and successful owner index recovery. The NVFP4 reference cannot stand in for
original BF16 quality or this different FP8 checkpoint.
