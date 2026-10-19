# Native Applio / RVC voice conversion

This fork implements file voice conversion with RVC inference exports. It does not
launch Applio, import its application, or download during loading/inference.
[Applio](https://github.com/IAHispano/Applio), MIT, supplies the reference RVC
architecture; its [HF repository](https://huggingface.co/IAHispano/Applio) contains
application bundles, supporting encoders/predictors and training pretrains, rather
than a single ready-to-use speaker model. A trained voice export is required.

## Components and supported scope

| Component | Native interface / location | Scope |
| --- | --- | --- |
| Voice generator | `Load Applio / RVC Models`, `audio_models/rvc/*.safetensors` | RVC v1 (256 features), v2 (768), HiFi-GAN resblock 1; pitch-guided and non-pitch exports; 32/40/48 kHz |
| Content encoder | Same loader, `audio_encoders/rvc_contentvec.safetensors` | Pinned Applio HuBERT-base ContentVec, final 12th-layer output and v1 final projection; use the training embedder |
| Pitch | Same loader, `audio_encoders/soulx_singer_rmvpe.safetensors` | Shared native RMVPE, CPU FP32, 100 Hz for RVC; existing SoulX callers retain 50 Hz |
| Retrieval | `rvc_indexes/*.index`, selected on conversion | Optional CPU FAISS, eight neighbors, inverse-square distance blend; choose the index trained with this voice |
| Media | Existing `LoadAudio` and `SaveAudio` | Ordinary ComfyUI AUDIO, one recording, channels averaged to mono |

Real-checkpoint acceptance covers the **Kiki v1 and Zhanzhan v2 40 kHz
pitch-guided demo exports** from [licyk/rvc-model](https://huggingface.co/licyk/rvc-model).
Other rates and non-pitch synthesis have architecture/fixture coverage only.
These demo weights carry the original RVC research-use notice; retain the downloaded
`LICENSE-RVC.txt`. Voice rights/terms are separate from the Applio code license.
The demo is an installation test, not a promise of English or singing quality.

Unsupported: training/resume checkpoints, MRF HiFi-GAN, RefineGAN, custom/SPIN or
language-specific embedders, other pitch predictors, realtime microphone capture,
TTS, formant changes, autotune and Applio effects/separation UI. Existing ComfyUI
nodes can prepare vocal stems before RVC. No quantized or reduced-precision path
is claimed. CPU/RTX 5090 acceptance does not establish ROCm/MPS/XPU support.

## Offline setup

Use Python >=3.10 with the deployment's PyTorch, einops and safetensors, plus
preinstalled NumPy, SciPy and librosa. Optional `faiss-cpu` is needed only for
retrieval; conversion without an index works without FAISS. Linux acceptance uses
Python 3.13.16, Torch 2.14.1+cu130 and FAISS CPU 1.15.1. FAISS adds CPU RAM and
search time. No installer or model loader fetches these packages automatically.

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py --applio
```

Acquire the files in [the pinned inventory](applio-sources.json) through Download
Manager 2; transfer only the authorized Firefox Hugging Face cookies first. Keep
originals outside Git. Do not install application archives or G/D training pretrains
as voices. Upstream code for reference checks is pinned to
`5379cf261fa7d8085c3b077aa42dabca7c85310b`.

Package local source files explicitly (paths below are examples):

```sh
python scripts/convert_rvc.py contentvec /sources/contentvec.bin /models/audio_encoders/rvc_contentvec.safetensors --source-revision IAHispano/Applio@e908bdd5bf9f18cede26bbbec724b1a4bc7c2f14 --sha256 d8dd400e054ddf4e6be75dab5a2549db748cc99e756a097c496c099f65a4854e
python scripts/convert_rvc.py voice /sources/kikiV1.pth /models/audio_models/rvc/kikiV1.safetensors --source-revision licyk/rvc-model@ad931ef3dc18f2d63ef81ecafddd498a111c8ecd --sha256 fb48c0a982317073c343e440e0c88337c3c605908429d81d420ea123bc83b5b5
```

The converter uses weights-only source loading. It validates supported metadata,
complete native tensor inventory, shapes and finite payloads, folds weight norms
in FP32 (matching upstream's FP32 inference), renames layer norms, and excludes
only training posterior/mask tensors. Unchanged values retain source dtypes;
folded weights are explicitly FP32. Every saved tensor is read back bit-exact.
Each output has a `.manifest.json` with source/output SHA256, revision, exclusions
and normalization transforms. Loaders require converted metadata and strict tensor
coverage. They never open arbitrary pickles or silently change architecture.

Originals plus converted models need approximately 1 GiB for these two voices and
ContentVec; allow at least another 0.6 GiB during conversion. CPU conversion RAM
includes source and normalized output plus temporary normalization buffers. The
reused RMVPE artifact is about 0.36 GiB. Copy its existing prepared file; for a new
installation use the RMVPE section of [SoulX setup](soulx-singer.md) and its offline
converter. Retain model licenses beside installed weights. Copy a matching `.index`
into `rvc_indexes` only after checking its published hash.

Existing output/partial/manifest files cause refusal. After an interruption, inspect
and preserve them, choose a fresh output path, and rerun from verified originals.
The model is self-contained; the manifest is published after its verified payload.
A missing manifest is incomplete conversion evidence, even when weights can load.

## Workflow and behavior

`LoadAudio → Applio / RVC Voice Convert → SaveAudio`, with `Load Applio / RVC
Models` connected to the converter. The editable example is
[examples/applio-rvc.json](../examples/applio-rvc.json). Discover `/object_info`
after safe service activation; source integration does not update a running process.
Save the user's working copy in `user/default/workflows/`.

Select a voice, its training ContentVec and RMVPE (or `none` for non-pitch voices).
Start with speaker 0, semitones 0, protect 0.33, RMS mix 1 and retrieval disabled.
Kiki's supplied index can be enabled with `index_rate=0.5`. A dimension check cannot
prove that an index or embedder belongs to a particular trained voice. A missing
pitch model for a pitch-guided voice or a malformed index fails explicitly.
`none` skips retrieval; rate zero does not read the index file. Seed controls the
latent/source noise and is local to the execution.

Analysis is 16 kHz mono, high-pass filtered at 48 Hz. Pitch shifts multiply voiced
F0 by `2 ** (semitones / 12)`. Below protect 0.5, unvoiced features blend back toward
the original; smaller values protect more. RMS mix 1 keeps synthesized dynamics;
0 follows source loudness. Peaks above 0.99 are scaled down. Normal ComfyUI save
nodes own output encoding and paths.

Four-second cores crossfade over 200 ms, with one-second reflection context on each side and 20 ms extra
right context to retain the complete core. Output length follows source duration
at the voice's rate, including the final partial segment. This bounded deterministic
segmentation differs from Applio's device-dependent long-file split search and
avoids its per-segment duration loss; long files need separate boundary/listening
acceptance. Neural peak memory is segment-bounded, but input/output audio and a
selected FAISS index plus reconstructed vectors occupy CPU RAM for the execution.
Indexes are not retained in global/model caches. Very large indexes can exhaust
host RAM; choose a smaller matching index or disable retrieval.

All networks are `CoreModelPatcher` managed and admitted with activation/workspace
estimates. ContentVec and voice use the selected device, FP32 operations/casting,
and block prefetch/compiler scopes; RMVPE retains its existing CPU GRU contract.
The workspace estimate is conservative and not a universal measured VRAM bound.
ContentVec uses selected optimized attention with per-module ComfyAttention and
owned tensor containers. RVC's small relative attention includes both relative
keys and query-dependent relative values. The optimized dispatcher lacks that
value contract, so this encoder retains exact local relative math; replacing it
with ordinary SDPA changes the model. No model-owned persistent tensor caches or
manual CUDA loading/unloading are added.

Cancellation checks run between content blocks, flow/vocoder stages, pitch chunks
and audio segments. Allow the normal queue to finish or interrupt **only your own**
run. Recover through its recorded prompt ID; do not replay uncertain submissions,
clear another job's queue entry, or restart a busy service. Unload through ComfyUI's
model owner (`POST /free` with `unload_models` and `free_memory`) only when idle.

## Verification

```sh
python -m pytest tests-unit/comfy_test/rvc_test.py tests-unit/comfy_test/soulx_singer_test.py -q
python scripts/check_rvc.py --upstream /sources/Applio --sources /sources/Applio-weights --models /models --output /evidence/rvc-parity.json
```

The offline checker runs CPU only, verifies the exact upstream code revision and
compares real v1/v2 generator encoder/waveform outputs at fixed features, F0 and
seed, plus ContentVec against Transformers HuBERT. Thresholds are fixed in the
checker before execution. It starts no service or GPU workload. Upstream code and
Transformers are reference-only requirements, absent from native inference.

Initial CPU checks: both generator waveforms and text-encoder statistics bit-exact;
ContentVec max error `8.50e-6`, RMS `1.06e-6`. Unit checks cover relative-key/value
math, half-source normalization, pitch bins, duplicate-neighbor retrieval,
pitch/non-pitch inference and incompatible exports. A five-second real audio
conversion with Kiki + index produced 200,000 finite 40 kHz samples.

GPU, live queue, lifecycle and output-quality evidence is recorded separately in
[acceptance](applio-acceptance.md). Unrun gates remain limitations.
