# Native audio and grounding integration status

The shared Llama/Qwen attention module now passes single-owner tensor containers
and its `ComfyAttention` preference through the existing attention dispatcher.
No text generation, conditioning, checkpoint-weight or return contract changes.
Unconfigured fixed-cache decode retains the existing Kitchen flash operation.
An explicit per-module preference instead receives the fixed cache with a mask
using each batch item's valid length. Invalid storage is cleared in temporary
inputs so uninitialized values cannot contaminate masked attention. Cache shape
stays fixed for the existing graph execution path. This costs additional masking
and temporary storage; no speed improvement is claimed.

CPU regression coverage includes separate/merged projections, ordinary and
grouped-query attention, causal masks, dynamic/sliding caches, uneven fixed-cache
lengths with poisoned unused slots, preference dispatch and default flash-path
selection. Existing MOSS-Audio and text-generation budget tests pass. Independent
MOSS-Audio fixtures against pinned Transformers 4.57.1 reference code also pass
for audio features, tokenization, DeepStack prefill and cached logits. These are
small CPU fixtures, not GPU performance or full-model acceptance of this change.
Accelerator kernels, graph replay, cancellation and live queue acceptance remain
pending. Attention implementations overriding this module's forward, such as
T5Gemma's softcap path, retain their own attention semantics.

Run the focused checks with the prepared interpreter:

```sh
python -m pytest tests-unit/comfy_test/llama_attention_test.py tests-unit/comfy_test/moss_audio_test.py tests-unit/comfy_test/text_generation_length_test.py -q
MOSS_REFERENCE_SOURCE=/absolute/pinned/reference MOSS_REFERENCE_MODEL=/absolute/original/moss-audio HF_HUB_OFFLINE=1 /absolute/reference/python -m pytest tests-unit/comfy_test/moss_audio_reference_test.py -q
```

The reference environment and source prerequisites are documented in
[MOSS-Audio](moss-audio.md). No model downloading occurs in these tests.

## Offline VoiceGenerator conversion

`python -m scripts.convert_moss_speech` prepares the original BF16 generator
from [OpenMOSS-Team/MOSS-VoiceGenerator](https://huggingface.co/OpenMOSS-Team/MOSS-VoiceGenerator/tree/97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4),
an Apache-2.0 model with a Qwen3-1.7B backbone and 16 audio heads. The source bundle
must match the committed `scripts/moss_speech_voicegenerator.json` manifest;
this first converter does not accept other MOSS models or revisions. It uses
the prepared ComfyUI Python environment and local files only. Source Python
files are hashed for provenance and never imported or executed.

```sh
python -m scripts.convert_moss_speech --source /absolute/MOSS-VoiceGenerator --destination /absolute/moss_voicegenerator_bf16.safetensors
python -m pytest tests-unit/comfy_test/moss_speech_conversion_test.py tests-unit/comfy_test/moss_audio_test.py -q
```

The output contains all 343 BF16 tensors, with `language_model.*` renamed to
`model.*`, independent `lm_heads.*` and `emb_ext.*` weights, and the original
tokenizer JSON bytes in a U8 tensor. Metadata retains config, tokenizer settings,
added/special tokens, processor settings, chat template and source checksums.
The accompanying `.manifest.json` accounts for each tensor's source, destination,
shape, dtype and payload checksum. Every copied tensor is independently reread
before publication. Single-file and indexed safetensors inputs share the existing
MOSS-Audio stream writer; expected inventories reject extra/missing tensors and
wrong shapes/dtypes. Modified sources and interrupted writes cannot publish a
partial output. Existing destinations and manifests are never overwritten.

The pinned `added_tokens.json` has stale spellings for three audio tokens.
The tokenizer JSON and tokenizer config agree with the local Qwen tokenizer:
IDs 151654, 151656 and 151662 are `audio_user_slot`, `audio_assistant_gen_slot`
and `audio_assistant_delay_slot` with their `<|...|>` delimiters. Conversion
validates these effective spellings and retains the stale asset as provenance;
it does not overlay its aliases onto the effective tokenizer.

Allow about 4.24 GB of additional disk space plus small manifests. The generator
weights alone do not include the separate 7.10 GB FP32 MOSS audio codec.
Conversion streams tensor bytes through CPU file buffers and neither loads a
model nor uses the GPU. On hash/config failure, restore the exact pinned files
through the managed download workflow. On interrupted conversion, inspect the
named output and manifest before retrying; a published output requires a fresh
destination. This artifact is a conversion baseline, not a registered native
speech loader, an INT8 model or an accepted speech workflow.

## MOSS v1 codec baseline

The native math in `comfy/ldm/audio/moss_audio_tokenizer.py` is adapted from the
Apache-2.0 [OpenMOSS audio tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/3cd226ba2947efa357ef453bcad111b6eafba782).
The full [Apache license](../comfy/ldm/audio/moss_audio_tokenizer.LICENSE) accompanies
the adapted module. It implements the original 24 kHz, 1920-sample hop, causal transformer encoder
and decoder with residual LFQ. It uses native operations, exact GELU, shared
interleaved rotary kernels, per-module attention preferences/containers and block
prefetch. All streaming tensors belong to one decode call. Loading, eviction and
output-device policy still require the pending discrete codec owner.

Prepare its separate FP32 checkpoint with the same local runtime:

```sh
python -m scripts.convert_moss_codec --source /absolute/MOSS-Audio-Tokenizer --destination /absolute/moss_audio_tokenizer_v1_fp32.safetensors
```

The complete pinned bundle must match `scripts/moss_codec_v1.json`. Conversion
accounts for 1,600 original tensors. It folds 66 weight-normalization magnitude/
direction pairs with the reference FP32 `torch._weight_norm` operation, retaining
biases and emitting 1,534 inference tensors. Metadata records both source keys,
their shapes/checksums and the transform; the output manifest hashes every
resulting tensor. It embeds the source config and provenance, with no text
tokenizer tensor. Allow approximately 7.10 GB extra disk space, plus a small
temporary folded-weight file. It performs no model inference or download.
The same overwrite, interruption and source-change checks apply as above.

The reference's eight-second chunk decode is preserved, including its ring-cache
behavior: it inserts an entire chunk before evaluating attention. After the
first chunk, this is **not equivalent** to full causal attention with a ten-second
window. Short inputs use the reference full-attention path. Encoder padding also
preserves the reference's floor-divided valid frame count: a partially padded
last frame exists in storage but is outside the reported valid length. These
behaviors must not be silently corrected during native parity work.

Focused fixtures compare rotary math, full/ring attention, transformer blocks,
patch layouts, FP32 LFQ and the complete folded state-dict inventory. Use the
pinned source with the existing isolated Transformers 5 environment; do not
replace the serving interpreter's Transformers installation:

```sh
PYTHONPATH=/absolute/worker/transformers5 MOSS_CODEC_REFERENCE=/absolute/MOSS-Audio-Tokenizer HF_HUB_OFFLINE=1 /absolute/prepared/python -m pytest tests-unit/comfy_test/moss_codec_reference_test.py -q
```

The real checkpoint also passed stagewise CPU comparisons for a one-frame
16-book decode and a 1,937-sample encode; encode code IDs matched exactly. These
synthetic cases do not establish audible quality, accelerator support, long-form
streaming, cancellation/offload acceptance or runtime adoption. No new codec
loader/node is registered by this prerequisite.

## Remaining integrations

These are shared prerequisites for the requested native MOSS speech family,
SoundEffect v2, LocateAnything, Qwen3-ASR, Qwen3 ForcedAligner and Fish S2-Pro
integrations. Their new loaders, remaining codecs, generation/alignment nodes,
quality comparisons and live workflows are **not implemented or accepted by this
change**. Existing worker/runtime paths remain unchanged. Integer speech codes
will use an explicit discrete codec contract; SoundEffect's continuous diffusion
path and forced alignment's timestamp classifier remain separate designs.
