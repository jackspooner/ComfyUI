# Checkpoint conversion and NVFP4 feasibility

[Project index](README.md)

## Evidence boundary

The publisher describes ModelOpt NVFP4 W4A8, packed expert weights with block
scales, and BF16 audio/vision encoders. Configuration declares FP8 KV-cache
quantization. These are publisher claims/configuration, not a verified reference
execution. See [sources and open questions](evidence.md).

Payload inspection now contradicts the BF16-encoder claim: 195 audio and 116
vision projections are packed NVFP4. The source index also has a stale byte total.
See [the pinned findings and diagnostic command](status.md). Do not use the model
card's precision table as the converter's tensor inventory.

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

## Offline converter

The converter is implemented, with synthetic integrity/publication tests. A
converted artifact has not yet passed native GPU acceptance or been installed.
From the repository root, using the prepared ComfyUI interpreter:

```bash
python scripts/convert_qwen3_omni.py \
  --source /absolute/path/to/pinned-source \
  --destination /absolute/path/to/new-candidate.safetensors \
  --source-revision 0ed3aa2b00eeedbb0d38b6d4738f506af759213b
```

All 15 pinned input files are hash checked and every source tensor is compared
with the pinned upstream architecture on the meta device. The known index total
is reconciled only for this exact input; there is no general integrity bypass.
The script streams unchanged tensors and transforms individual projections or
experts. Gate/up banks retain expert order and require bit-identical global
scales. Original packed values/scales round-trip during transformation. The 27
vision down projections need 16 zero input columns for CUDA's K%32 alignment;
the native vision adapter must append the corresponding zero activations.

The artifact embeds tokenizer/config/processor/template data and native quantization
descriptors. Its companion `.manifest.json` accounts for every source tensor,
excluded speech tensor, source hash, output tensor hash and complete artifact hash.
Peak host RSS is recorded separately from any later inference measurement.

Publication uses validated temporary files and no-overwrite hard links. A normal
exception/interruption removes only its temporary files. After a process kill,
preserve hidden `.DESTINATION.*` files for inspection. If only the final manifest
exists, confirm no converter is active and the final weights do not exist before
removing that task-owned orphan manifest and retrying. Never remove another
process's files or overwrite an accepted artifact. Originals remain unchanged.

## Artifact contract

Inputs: an explicit local source directory, pinned revision/provenance and an
explicit destination. Outputs: a ComfyUI-readable checkpoint and manifest with
source revision, converter commit, source/output checksums, tensor inventory,
excluded speech-output modules, key/layout transformations and precision policy.
Embed all tokenizer, configuration and processor assets required for inference in
the single safetensors artifact. The provenance manifest may be a sidecar;
inference must not need the source Hugging Face directory.

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
