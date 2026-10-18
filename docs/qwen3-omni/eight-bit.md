# FP8 preparation and INT8 scope

Status, 18 October 2026: **preparation only**. The experimental FP8 E4M3 converter
and native projection loader pass synthetic CPU tests. No complete FP8 or INT8
model has been converted or executed. This work is separate from the installed,
accepted NVFP4 artifact and does not change its inference policy or files.

The requested full-model FP8/INT8 tests exclude original BF16/FP16 model execution.
The request also calls INT8 theory-only in its final sentence; practical INT8 work
is pending that clarification. BF16/FP16 full-model runs are not part of this task.

## Source and current blocker

The source is [Qwen's original model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct),
revision `26291f793822fb6be9555850f06dfe95f2d7e695`, rather than decoded NVFP4 values.
The 24-file [immutable inventory](../../scripts/qwen3_omni_original_source.json)
contains published SHA-256s for 15 weight shards; downloaded metadata was checked
against its published Git object hashes before recording SHA-256s. Metadata came
through Download Manager 2. The scoped Firefox cookie read was locked; these
public files were obtained without authentication. Weight shards are not queued.

The original index's **28,010 keys** exactly match the current Transformers meta
architecture. Its expected payload is **70,519,637,090 bytes**, including excluded
speech stages; Thinker plus input encoders account for **63,438,410,976 bytes**.
These are metadata/architecture checks. They do not verify undownloaded payloads.
The planned FP8 headers also pass native layout validation on the meta device,
and three sample strings round-trip identically through the embedded tokenizer.
Neither check loads or executes model weights.

The FP8 plan has 599 quantized native projections and **32,365,008,864 bytes** of
payload before embedded assets. The mounted filesystem currently has only about
20–23 GB free. A converted artifact alone cannot fit. Source plus one candidate
needs over 103 GB; retaining both eight-bit candidates needs about 135 GB before
filesystem/headroom and embedded assets. Reserve at least 140 GB free, or use an
explicit suitable staging filesystem. No existing models were removed or drives
mounted to make room. The writer checks the minimum output space before creating
temporary files; callers must also budget for the manifest and source files.

## Experimental FP8 contract

The existing converter gains `--quantization fp8`; the default `nvfp4` still
preserves the previously pinned ModelOpt checkpoint. FP8 requires the pinned
original source and refuses converting the NVFP4 checkpoint into another format.
Every original file hash, tensor dtype/shape and index total must validate first.
There is no general revision or integrity bypass.

From a prepared task checkout containing the new option:

```bash
/path/to/prepared/python scripts/convert_qwen3_omni.py \
  --source /absolute/original-source \
  --source-revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --quantization fp8 \
  --destination /absolute/new-qwen3-omni-fp8.safetensors
```

FP8 means E4M3 **weight storage**, with BF16 compute/output/cache. The artifact's
`full_precision_matrix_mult` descriptor preserves the existing decoded-compute
owner; this is W8A16-style execution, not an FP8-activation or FP8-cache claim.
Embeddings, norms, routers, output head, biases and convolutions retain their
original BF16 representation. The same text/audio/vision linear projection set
as the verified architecture is quantized. Gate/up matrices are fused in expert
order and share a per-expert FP32 scale; other projections use one FP32 scale.

Conversion streams one projection/expert group, computes an absmax scale and uses
the existing Comfy Kitchen/ComfyUI FP8 operation. Normalization uses FP32 to avoid
an extra BF16 rounding before the FP8 cast. Zero matrices use a finite positive
scale. Vision input columns retain the native zero padding to multiples of 32.
Tokenizer, normalized architecture, original template and processor settings are
embedded using the installed Transformers processor, with local-only loading.
No full original BF16 model is materialized or executed by the converter.

The checkpoint and provenance manifest are streamed, checksummed and published
with the existing no-overwrite mechanism. Originals remain unchanged. An orderly
failure removes only its temporary files. After process death inspect owned
temporary files and a possible orphan manifest before retrying to a new destination;
do not overwrite an accepted candidate. The loader validates FP8 dtype, unpacked
weight shape, FP32 scale shape and expert counts before state loading. Unrelated
quantization descriptors and orphan NVFP4 second scales are refused.

These source-level capabilities are not an installed FP8 model or a running-service
acceptance claim. Do not install/deploy a candidate until its frozen text/audio/
image/combined, resource and lifecycle checks have passed through the established
owners. Never restart ComfyUI or evict active work merely to obtain a test window.

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

The cumulative unit command passes **104 tests**, with one expected owned-CUDA
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
The earlier accepted NVFP4 tests still pass. No new GPU, full-model, lifecycle,
latency or quality result is claimed. Other backends remain unverified.

Remaining: resolve storage and INT8 scope; download/verify the original weights
through Download Manager 2; convert to fresh destinations; extend the frozen
reference/native harnesses for the original source and eight-bit policies; run
independent quantized-reference, all 24 unchanged multimodal fixtures and explicit
resource/lifecycle measurements in owned GPU windows. Failed/contended runs remain
separate from completed evidence. The existing NVFP4 reference cannot be relabeled
as the original-weight FP8 reference because its underlying weights differ.
