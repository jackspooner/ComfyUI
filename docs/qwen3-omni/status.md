# Qwen3-Omni execution status

[Project index](README.md) · updated 18 October 2026

The converter, reconstructed reference and native adapter are implemented. The
converted artifact loads its own tokenizer and all 599 quantized modules.
Three-pass accelerated NVFP4 execution completes the frozen 24-case comparison
with **zero new assertion failures**: native and reference each pass 20/24 cases.
Installation and live queue acceptance are pending; source delivery alone does
not establish installation.

## Verified source and package

- Source: `shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4`, revision
  `0ed3aa2b00eeedbb0d38b6d4738f506af759213b`.
- Download Manager 2 obtained all 15 files, including seven shards; hashes match
  pinned publisher metadata. Firefox's scoped cookie read was locked; public
  downloads completed without authentication.
- All **65,880 expected tensor shapes/dtypes** match, including excluded speech
  components. Actual payload is **25,850,816,336 bytes**; the published index says
  **25,850,817,744**. The 1,408-byte correction is allowed only for the exact
  pinned, fully hash-verified inventory. Originals are unchanged.
- Thinker/audio/vision payload: 18,769,590,222 bytes. Excluded Talker:
  6,649,192,960; code2wav: 432,033,154. All 18,935 quantized projections are
  retained, including audio and vision projections.
- Package: **18,781,862,813 bytes**, SHA-256
  `039c9d4347a4be90ddbc8994909bd9870e610228dd95b336f612bf2936f2fe9b`.
  Conversion peak host RSS: **8,454,549,504 bytes**, including mapped pages.
  The manifest records converter commit `69ccd9a51e68e24d5c6e028f548cbb92c975fa18`.

The converter swaps FP4 nibbles, permutes block scales with existing helpers,
fuses gate/up banks only with identical scalar scales, and zero-pads 27 vision
projections from 4,304 to 4,320 columns. Packed values are never requantized.
Actual ModelOpt FP32 decoding matches native transformations on 33 projections
and nine complete expert paths. Native BF16 decoding has up to 0.001953125
intermediate-rounding difference. See [source accounting](results/source-reconciliation.json),
[component evidence](results/modelopt-native-cpu.json) and
[conversion/recovery commands](quantization.md#offline-converter).

## Reference and precision

The isolated reference uses Python 3.12.15, Transformers 5.20.0, ModelOpt 0.42.0
and PyTorch 2.14.1+cu130. Its actual decoder streams one projection or selected
expert; no full BF16 checkpoint/model is materialized. ModelOpt's optional HF
integration extra is excluded because it requires Transformers <5. The decoder
works and `pip check` passes; ModelOpt warns that Transformers 5.20.0 is outside
its tested integration range. The [hash-locked environment](results/reference-environment-lock.json)
is separate from the unchanged serving environment.

The reference retains upstream preprocessing, routing, attention, positions and
generation, using BF16 compute/cache. Publisher W4A8 execution and FP8 KV cache
remain unverified. The [version 2 IMAGE-batch reference](results/reconstructed-reference-image-batch.json)
is authoritative; [earlier PIL results](results/reconstructed-reference-pil.json)
are scouting evidence only. Early CPU runs with incorrect EOS binding and an
initial package with incorrect vision-merger names were rejected and retained
privately; neither establishes acceptance.

Native decoded BF16 is diagnostic only. Its full suite passes the same 20 cases.
Single-pass CUDA NVFP4 adds speech/click failures and is rejected. Three residual
activation passes use existing CUDA NVFP4 matmuls, FP16 partial outputs, FP32
accumulation/bias and BF16 results/cache. Only activation residuals are decoded;
weights remain packed. Other models keep the one-pass default.

The [three-pass comparison](results/native-residual3-functional.json) covers all
24 frozen cases across three available GPU windows, excluding interrupted cases.
Positions and compute pixels match exactly; captured tensors are finite with
matching shapes. The [CPU input audit](results/native-preprocessing.json) checks
exact expanded IDs, FP32 pixels/mel features and positions for all cases, with
encoder forwards replaced by shape fixtures. It is not full-model parity.

The same four source errors remain: Unicode copying, hallucinated sound in
silence, standalone sign reading and combined sign reading/omitted blue object.
The source also invents a buzzer after audio removal. These are limitations,
not successful recognition. Replacing/removing modalities independently changes
relevant answers and the input audit confirms complete ingestion.

Matching-precision full-model measurements establish nonzero numerical variation;
this integration does not claim bitwise full-model logits. One-pass CUDA was
faster than eager on captured shapes (median 118.8 versus 409.2 microseconds), but
failed quality. Three-pass FP16 partial-output probes have worst relative RMSE
0.004482 across 33 projections/three shapes, with median 0.371 ms. Component
benchmarks do not predict end-to-end throughput. See
[kernel evidence](results/native-captured-kernels.json),
[rejected precision evidence](results/native-precision-investigation.json) and
[reference commands](reference.md).

## Lifecycle and measured resources

[The isolated lifecycle run](results/native-lifecycle-performance.json) passes
three repeated combined-input generations, cancellation during preprocessing,
prefill and decode, recovery, offload/reload and a MOSS Instruct transition back
to identical Omni tokens. Retained Torch allocation delta after cancellation is
zero. Cancellation propagates in under 2 ms in this fixture; this is not a bound
for every operation/input. Offload returns driver usage to about 1.22 GB.

For the combined speech/red-square fixture with a 512-token output limit, first
token latency is 25.92 seconds cold and 5.01 seconds warm; warm decoding is
2.34 tokens/second. Generation stops at EOS after 16 tokens. The process is cold,
but integrity verification warms the OS file cache. Omni resident driver memory
is about 19.73 GB; peak across the two-model transition is 31.55 GB. Torch-only
counters do not include the dynamic allocator. No other GPU work ran during
these measurements. The initial model transition exposed an admission guard
that checked physical free memory before idle dynamic weights could be evicted;
the generation owner now asks its existing memory owner to release idle weights
while preserving the requested model and its patches before refusing admission.

## Remaining release work

Complete boundary/kernel checks, merge concurrent committed main changes, run cumulative
regressions and graph change analysis, then install and verify through the normal
ComfyUI queue. Preserve current launch flags and serving dependencies. Activation
requires an idle maintenance window; queue submissions themselves need not wait
for an empty queue. Failed or interrupted runs remain outside accepted evidence.
