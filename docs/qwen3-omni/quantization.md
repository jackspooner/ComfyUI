# Checkpoint conversion and NVFP4 feasibility

[Project index](README.md)

## Evidence boundary

The publisher describes ModelOpt NVFP4 W4A8, packed expert weights with block
scales, and BF16 audio/vision encoders. Configuration declares FP8 KV-cache
quantization. These are publisher claims/configuration, not a verified reference
execution. See [sources and open questions](evidence.md).

Local ComfyUI supports NVFP4 `weight`, `weight_scale`, `weight_scale_2` and
quantization descriptors, including expert banks. Its fast path can quantize
activations using the NVFP4 layout. This is not proof of the source's advertised
W4A8 behavior. Weight compatibility, activation precision and KV-cache precision
must be evaluated separately.

## M1 compatibility experiment

1. Resolve a full immutable Hugging Face revision. Obtain the index, configuration,
   tokenizer/processor assets and necessary shards via Download Manager 2. Import
   only Hugging Face Firefox cookies through the supported importer, without
   printing them. Public downloads can proceed if no cookies are available;
   report that condition. Record job IDs and actual destinations. Wait and verify
   completion before consuming payloads. Never infer completion from submission.
2. Inventory actual safetensors headers: keys, shapes, dtypes, byte ranges and
   checksums. Check shard/index agreement, fused versus split experts, attention
   storage, all scale tensors and calibration auxiliaries. Do not infer this from
   filenames or the model card. Preserve untouched originals.
3. Select one attention projection and a full expert gate/up/down path. Verify FP4
   nibble order, signed codebook, block axis, padding, scale ordering and global
   scale broadcasting against the source decoder. Test multiple expert indices.
4. Compare decoded weights before any matmul. Exact equality is the target for
   an exact representation mapping; document rounding when comparing decoders
   that intentionally use different arithmetic precision. Reject missing scales,
   transposed blocks or approximate repacking masquerading as a lossless map.
5. Use fixed random, zero, outlier and representative activation fixtures. Compare
   source/reference outputs, a decoded-weight compute baseline, and available
   native accelerated outputs. Record absolute/relative errors and cosine
   similarity, latency, temporary memory, GPU/runtime versions and selected path.
   Set numerical tolerances from a justified reference precision baseline before
   judging candidate results; do not choose thresholds after seeing failures.
6. Record a decision: compatible existing path; explicit alternate precision path
   needing full-model quality acceptance; or a concrete missing kernel/layout
   capability. Do not proceed with an unexplained mismatch.

BF16 activation or KV-cache experiments may help isolate errors. They do not
restore the precision lost in the original quantization. BF16 cache may increase
memory. FP4 activations in place of advertised FP8 activations are a numerical
change requiring separate acceptance, not a metadata conversion. A slow decoded
weight reference is useful evidence but is not the promised production fast path.

## Converter contract (proposed)

Inputs: an explicit local source directory, pinned revision/provenance and an
explicit destination. Outputs: a ComfyUI-readable checkpoint and manifest with
source revision, converter commit, source/output checksums, tensor inventory,
excluded speech-output modules, key/layout transformations and precision policy.
Keep all assets required for offline tokenization/preprocessing embedded or in a
documented package; a lone weights file is insufficient if it needs sidecars.

Stream or bound memory rather than loading the whole model at once. Retain packed
values and scales when an exact mapping is possible. Mark each quantized layer
with the format expected by native loading; never reinterpret uint8 weights as
ordinary dense weights. Do not silently discard activation/calibration tensors.
Either consume them correctly or document and validate the explicit alternative.

Reject missing/truncated shards, Git LFS pointer files, duplicate/unknown required
keys, invalid quantization metadata and out-of-root shard paths. Write to a
temporary destination and publish only after validation. Refuse existing outputs.
On failure, preserve originals, identify partial outputs and retry to a new or
explicitly cleaned task-owned destination; do not overwrite an accepted artifact.

Budget disk for source plus converted output and temporary files. The published
repository size is not a Thinker-only size or peak-VRAM estimate. Determine those
from the selected revision and measurements, including media and KV workspace.
