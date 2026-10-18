# Offline reconstructed reference

[Project index](README.md) · [Acceptance matrix](validation.md)

This tool uses the upstream Transformers architecture and actual ModelOpt decoder
with BF16 compute/cache. It preserves preprocessing, routing, positions and
generation; only packed linear weights and indexed expert access are replaced.
It does not reproduce the publisher's W4A8 execution or FP8 cache. Native inference
must separately pass acceptance using accelerated operations.

The same harness also supports the pinned published FP8 candidate with an explicit
publisher inverse-scale normalization and upstream block decoder. See
[FP8 source and acceptance](eight-bit.md); its environment and precision limits
differ from the NVFP4 reference described below.

## Environment and inputs

Use a separate Python 3.12 environment. ModelOpt 0.42.0 rejects Python 3.13;
its optional HF integration extra also requires Transformers <5 and is not used.
The tested combination is Python 3.12.15, Torch 2.14.1+cu130, Transformers 5.20.0
and ModelOpt 0.42.0. ModelOpt warns that Transformers 5.20.0 is untested. Actual
decoder execution and `pip check` pass; export/quantization integrations are not
validated. Never replace serving dependencies to prepare this reference.

[The environment lock](results/reference-environment-lock.json) records wheel
URLs, hashes, native-library RECORD hashes and the standalone Python archive.
Acquire payloads through Download Manager 2. Install hash-verified compatible
wheels offline into the isolated environment, sharing only compatible native
CUDA libraries read-only when explicitly recorded by the lock. Do not add the
serving Python's entire site-packages directory to another Python ABI. Run
`python -m pip check` and verify the exact package versions before inference.
The lock describes the tested Linux/CUDA installation, not a portable installer.

All tools below run from the checkout root, require explicit local paths, make no
network requests and leave the original checkpoint unchanged. The source is the
pinned revision `0ed3aa2b00eeedbb0d38b6d4738f506af759213b`; validation checks all
15 hashes and every expected tensor, including excluded speech components.
Each output directory/report must be new. Preserve failed outputs before retrying
with a new destination; there is no implicit overwrite or resume.

## Components and fixtures

With the reference interpreter:

```bash
/path/to/reference/python -m scripts.qwen3_omni_reference \
  --source /absolute/path/to/checkpoint \
  --source-revision 0ed3aa2b00eeedbb0d38b6d4738f506af759213b \
  --components --output /absolute/path/to/new-components
```

This decodes 33 fixed projections with actual ModelOpt, on CPU, saving FP32/BF16
tensors plus checksummed provenance. It includes attention, both encoder families,
a vision projection requiring alignment padding and gate/up/down from nine experts.
It never materializes a full decoded model.

Prepare the six-case-per-modality suite with the prepared ComfyUI interpreter:

```bash
/path/to/prepared/python -m scripts.make_qwen3_omni_fixtures \
  --audio-fixtures /absolute/path/to/licensed-audio-fixtures \
  --output /absolute/path/to/new-fixtures
```

Its audio inputs must be local licensed speech, music and bird recordings
with attribution sidecars. The tool creates synthetic images, silence, tones and
clicks, resamples clips to 16 kHz, and records all hashes and credits. Version 2
also freezes FP32 BHWC IMAGE arrays. Mixed source dimensions use the existing
ImageBatch bilinear/center operation before freezing, so both implementations
receive the same ordinary IMAGE batch. It records the batch utility's source hash.
Freeze the manifest before comparing precision candidates. Fixtures include
separate image/audio removal/replacement and multiple-image order checks. Keep
the original scouting fixtures/results labelled with their earlier convention.

Run reference generation with those frozen cases:

```bash
HF_HUB_OFFLINE=1 /path/to/reference/python -m scripts.qwen3_omni_reference \
  --source /absolute/path/to/checkpoint \
  --source-revision 0ed3aa2b00eeedbb0d38b6d4738f506af759213b \
  --cases /absolute/path/to/fixtures/cases.json \
  --capture-activations --capture-parity \
  --device cpu --output /absolute/path/to/new-reference-results
```

The source template receives images in order, audio, then text, with thinking
disabled. Images enter the upstream processor as FP32 RGB with rescaling disabled;
audio is minimally padded to a full hop above the STFT reflection minimum.
Generation binds EOS/padding to the source tokenizer, records expanded IDs,
media features, up to eight decoding logits, answers, assertions and elapsed
times. The optional capture flags record component activations, encoder features,
positions and prefill routing. Source-code hashes identify the executed harness,
upstream model/processor and actual ModelOpt decoder. A fluent answer or a
component match is not full-model parity. Source-model
fixture failures must be recorded separately from native integration failures.

## Native operation comparison

Use the prepared serving interpreter for Comfy Kitchen comparisons:

```bash
/path/to/prepared/python -m scripts.benchmark_qwen3_omni_quant \
  --source /absolute/path/to/checkpoint \
  --components /absolute/path/to/new-components \
  --device cpu --report /absolute/path/to/new-comparison.json
```

CPU mode checks exact FP32 agreement and reports BF16 rounding differences. CUDA
mode additionally tests eligible existing kernels at 1/16/128-token shapes and
records warm timing, numerical differences and temporary allocation. CUDA mode
requires an independently established GPU window and the existing RuntimeHost
ownership protocol; the script does not acquire ownership or interrupt queues.
No component report alone accepts an accelerated production path. Full multimodal
quality, resource, lifecycle and live queue acceptance remain separate gates.

## Native package acceptance

After conversion and an independently established idle GPU window/hold, use the
prepared ComfyUI interpreter. Native execution reads only the converted artifact,
its provenance manifest and the frozen test inputs:

```bash
/path/to/prepared/python tests-inference/check_qwen3_omni.py \
  --model /absolute/path/to/candidate.safetensors \
  --cases /absolute/path/to/fixtures-v2/cases.json \
  --precision decoded-reference --capture \
  --output /absolute/path/to/new-native-baseline
```

`decoded-reference` preserves packed banks and uses the existing operation to
decode individual projections/indexed experts with BF16 matmul. It is diagnostic
and cannot satisfy accelerated production acceptance. Establish matching-precision
numerical behavior first, then repeat with `--precision nvfp4` and compare the same
fixture assertions, features, routing and logits. The native run always checks its fast
operation policy; a disabled accelerated path is a failure.
`--case ID` selects named fixtures and `--tokens N` records an explicit smoke or
budget override, including zero for automatic capacity. `--profile` writes a trace
for the first case. Captures/profiling affect timings; measure performance separately.
The harness uses the same dynamic allocator and native CLIP generation owners as
serving ComfyUI, without changing the running service, configuration or weights.
It rechecks the serving queue after integrity verification and stops its own
process-local generation if queued/running work appears. `contention.json`
preserves the conflicting prompt IDs; do not count an interrupted case as an
accepted result. `--comfy-url` selects the queue endpoint. A separate caller still
owns acquiring and releasing the validation hold after confirmed process exit.

For an experimental activation policy, `--nvfp4-input-passes N` selects the
shared operation explicitly. One pass is the unchanged shared default. Additional
passes quantize the remaining activation error, run existing NVFP4 CUDA matmuls
with FP16 outputs, sum in FP32, add bias once and return the compute dtype. No
weight matrix is decoded or requantized by this operation. This is not a claim
to reproduce publisher W4A8 execution. Two-pass probes still fail speech
classification; three-pass FP16-output execution passes the full frozen fixture comparison and
is the Omni default. Lifecycle, resource and installation evidence remain separate.
The harness records inference-source hashes and refuses source changes during
evaluation. First-token/decode timing and the fast-operation policy are collected
even without `--capture`; use that mode for performance runs.

Keep failed run directories and reports. Resolve an uncertain process outcome
before releasing its exact validation hold or retrying with a new destination.
Do not compare cached logits beyond a divergent generated context as though the
inputs were still identical. Native model acceptance and live service activation
are separate requirements.

## CPU input audit and capture comparison

The input audit uses the artifact's actual tokenizer/preprocessing and compares
expanded IDs, FP32 pixels/mel features and positions against upstream captures.
Encoder forwards are replaced with shape fixtures for this CPU-only audit; it
does not claim full-model parity:

```bash
CUDA_VISIBLE_DEVICES='' /path/to/prepared/python tests-inference/check_qwen3_omni_preprocessing.py \
  --model /absolute/path/to/candidate.safetensors \
  --cases /absolute/path/to/fixtures-v2/cases.json \
  --reference /absolute/path/to/reconstructed-reference \
  --output /absolute/path/to/new-input-audit.json
```

All 24 frozen cases match exactly. For numerical/fixture comparison:

```bash
CUDA_VISIBLE_DEVICES='' /path/to/prepared/python tests-inference/compare_qwen3_omni.py \
  --reference /absolute/path/to/reconstructed-reference \
  --native /absolute/path/to/native-run-1 /absolute/path/to/native-run-2 \
  --output /absolute/path/to/new-comparison.json
```

Combining runs requires identical artifact, inference-source hashes, fixture list
and precision policy. The tool reports missing cases, retains each run's status,
compares cached logits only while generated contexts match and never marks
production accepted. The three-pass cohort spans three GPU windows (16+4+4
completed cases); interrupted cases are excluded. Its 24 completed cases have
exact positions/compute pixels, finite captured tensors and zero new assertion
failures. Four source failures remain. This is not an uninterrupted lifecycle run.
