# Published FP8 candidate and INT8 scope

Status, 18 October 2026: the published FP8 source is downloading; block-scale
conversion/loading pass synthetic CPU tests, six CUDA component checks and the
full planned native meta layout. Full-model inference and lifecycle acceptance are
pending. The installed NVFP4 artifact and serving service are unchanged.

The selected source is [marksverdhei/Qwen3-Omni-30B-A3B-FP8](https://huggingface.co/marksverdhei/Qwen3-Omni-30B-A3B-FP8),
revision `042db39a102171d5994b72ab717fade92e80061a`, based on the Instruct model.
This route uses existing quantized weights; original BF16/FP16 model execution
and the 70.52 GB original-weight download are unnecessary. INT8 practical versus
theory-only scope remains unresolved; no INT8 artifact or execution is claimed.

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

The native plan occupies 33,551,468,278 bytes including embedded assets, before
its safetensors header and provenance manifest. Reserve at least 75 GB free for
source plus native output, additional headroom for fixtures and other workloads,
and inspect current space before starting. A published download size is not a
VRAM estimate. Preserve existing models and active task data.

## Lossless block FP8 packaging

```bash
/path/to/prepared/python scripts/convert_qwen3_omni.py \
  --source /absolute/published-fp8-source \
  --source-revision 042db39a102171d5994b72ab717fade92e80061a \
  --quantization fp8-block128 \
  --destination /absolute/new-qwen3-omni-fp8.safetensors
```

This mode preserves the published E4M3 weight bytes and BF16 128×128 block scales.
It fuses aligned gate/up rows in expert order and renames `weight_scale_inv` to
native `weight_scale`, without decoding or requantizing. The native Thinker has
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
positive finite scales, embeds local processor/tokenizer/template/config assets,
then verifies output checksums and publishes the checkpoint plus manifest without
overwrite. Originals remain unchanged. An orderly failure removes its temporary
files. After process death inspect exact owned temporary files and a possible
orphan manifest before retrying to a new destination; never overwrite an accepted
candidate. Invalid layouts, missing scales and expert-count mismatches fail before
state loading. Scalar FP8 scales retain their separate descriptor and validation.

## Independent validation

The reference keeps upstream Transformers 5.20.0 architecture and dispatch, and
uses its `Fp8Dequantize` decoder independently of the native layout. FP8 expert
buffers remain CPU file-backed and only selected matrices move to the requested
device. Compute/cache are BF16. This avoids materializing the original BF16 model;
CPU offload can make the reference slow. ModelOpt 0.42.0 remains required only for
the existing NVFP4 reference and its separate supported Python runtime.

Run `scripts/qwen3_omni_reference.py` with the source/revision, frozen 24-case
fixture file and new output directory, then `tests-inference/check_qwen3_omni.py`
with the native artifact, the same fixtures and `--precision fp8-block128`.
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
  FP32 scalar-scale counts as this FP8 plan. This establishes a storage estimate,
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

## Verification and remaining work

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

New checks cover independent FP32-to-FP8 rounding, per-expert scales, all-zero
experts, gate/up order, padding, dense exclusions, missing/invalid native layouts,
NaN refusal, insufficient-space cleanup and rejection of NVFP4 as an FP8 source.
Seeded requantization is covered for patched block FP8 weights. The earlier
accepted NVFP4 tests still pass. Six RTX 5090 component probes match
the pinned Transformers decoder and BF16 matmul outputs exactly; decode plus
matmul takes 0.022–0.054 ms in these synthetic shapes. [Retained evidence](results/published-fp8-components.json)
records shapes and source hashes. Eight real projections from one SHA-256-verified
source shard also decode identically on CPU/CUDA and produce identical CUDA BF16
linear outputs. These timings exclude loading/offload and are not full-model
latency. No full-model, lifecycle or quality result is claimed.
Other backends remain unverified.

Remaining: finish the pinned published FP8 download and source verification;
convert to a fresh destination; run independent quantized-reference and all 24
unchanged multimodal fixtures, plus explicit
resource/lifecycle measurements in owned GPU windows. Failed/contended runs remain
separate from completed evidence. The existing NVFP4 reference cannot be relabeled
as the original-weight FP8 reference because its underlying weights differ.
