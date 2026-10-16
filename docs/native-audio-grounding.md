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

## Remaining integrations

These are shared prerequisites for the requested native MOSS speech family,
SoundEffect v2, LocateAnything, Qwen3-ASR, Qwen3 ForcedAligner and Fish S2-Pro
integrations. Their new loaders, codecs, generation/alignment nodes,
quality comparisons and live workflows are **not implemented or accepted by this
change**. Existing worker/runtime paths remain unchanged. Integer speech codes
will use an explicit discrete codec contract; SoundEffect's continuous diffusion
path and forced alignment's timestamp classifier remain separate designs.
