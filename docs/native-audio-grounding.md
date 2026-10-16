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

This is a shared prerequisite for the requested native MOSS speech family,
SoundEffect v2, LocateAnything, Qwen3-ASR, Qwen3 ForcedAligner and Fish S2-Pro
integrations. Their new loaders, converters, codecs, generation/alignment nodes,
quality comparisons and live workflows are **not implemented or accepted by this
change**. Existing worker/runtime paths remain unchanged. Integer speech codes
will use an explicit discrete codec contract; SoundEffect's continuous diffusion
path and forced alignment's timestamp classifier remain separate designs.
