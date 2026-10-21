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
the default variant accepts only this VoiceGenerator revision. Explicit variants
for three other pinned delay-family bundles are described below. It uses
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
destination. This artifact is the BF16 storage baseline for the native VoiceGenerator loader
below. INT8/ConvRot artifacts are not yet supported.

## Other delay-family conversion variants

The same converter accepts `--variant delay`, `--variant acting`, or
`--variant ttsd`. Each requires its own committed `scripts/moss_speech_<variant>.json`
manifest and complete local bundle. No source Python is executed and no files
are downloaded. Native generation adapters for these three variants are still
pending; the VoiceGenerator loader does not accept their converted weights.

| Variant | Immutable model revision | Inventory / extra output space |
| --- | --- | --- |
| `delay` | OpenMOSS-Team/MOSS-TTS `b6b0229853ff63c68fa6aeceb380d8c016f55daf` | 463 tensors, about 17.0 GB |
| `acting` | laion/moss-tts-v1.5-8b-voice-acting `aabb7b609c2b56e5cfa8e8641eb1d8ce9e3d1997` | 463 tensors, about 17.0 GB |
| `ttsd` | OpenMOSS-Team/MOSS-TTSD-v1.0 `c7cd852d87aff71cab5bd2b9b05509cedc0ef1ba` | 431 tensors, about 16.7 GB |

```sh
python -m scripts.convert_moss_speech --variant delay --source /absolute/MOSS-TTS --destination /absolute/moss_delay_bf16.safetensors
python -m scripts.convert_moss_speech --variant acting --source /absolute/moss-tts-v1.5-8b-voice-acting --destination /absolute/moss_acting_bf16.safetensors
python -m scripts.convert_moss_speech --variant ttsd --source /absolute/MOSS-TTSD-v1.0 --destination /absolute/moss_ttsd_bf16.safetensors
```

These preserve the 36-layer, 4096-wide Qwen backbone with 32 audio books for
Delay/acting or 16 for TTSD, plus independent text embeddings/heads. Acting's
Transformers 5 `rope_parameters` and the other bundles' original RoPE fields
are validated separately. Original generation-config data, when present, joins
the embedded tokenizer assets; it does not silently alter runtime defaults.
The converter checks the exact single-file/sharded weight set, all shapes/dtypes,
source hashes and every emitted payload using the same transaction as VoiceGenerator.
Use fresh destinations on retry; retain originals and both conversion outputs.

The selected complete base bundle's generation body now matches VoiceGenerator
apart from sampling defaults. The older research-only base source used a joint
audio-head draw; do not use that older source as this bundle's parity oracle.
VoiceGenerator, base Delay, acting and TTSD retain distinct prompt/control policies
and acceptance records even where their neural math or sampling loop is shared.

## Native VoiceGenerator

Place the converted generator in a configured `text_encoders` directory and the
converted FP32 codec in `vae`. Select **Load CLIP**, type **moss_tts**, and
`moss_voicegenerator_bf16.safetensors`. Connect its CLIP output to
**MOSS VoiceGenerator**; connect the resulting AUDIO_CODES and **Load Audio
Codec** to **Decode Audio Codes**, then **Save Audio**. The generation node reads
`text`, `instruction` (voice description), `language`, `seed`, `max_length`,
`audio_temperature`, `audio_top_p`, `audio_top_k` and `audio_repetition_penalty`.
It returns codes only; it does not load the codec, clone reference voices, return
text conditioning or use the diffusion sampler. The shared CLIP boundary owns
model loading and memory reservation, including the requested KV capacity.

A concrete API prompt fragment, after choosing the model names above:

```json
{"class_type":"MossVoiceGenerator","inputs":{"clip":["1",0],"text":"Hello world.","instruction":"A warm, natural adult voice.","language":"en","seed":7,"max_length":256,"audio_temperature":1.5,"audio_top_p":0.6,"audio_top_k":50,"audio_repetition_penalty":1.1}}
```

Text, whitespace, brackets and voice instructions are serialized as in the pinned
worker's dictionary-message path. An empty language uses its literal `None`
field. Text sampling retains reference defaults (temperature 1.5, top-p 1,
top-k 50). Audio temperature 0 selects greedy audio tokens; positive values sample
with a private seeded generator. Repetition penalties retain the reference's
flattened audio-head history. Seed reproducibility is within a fixed runtime and
backend; identical seeds across devices need not produce identical speech.

`max_length` counts outer generation steps, including delay and flush overhead,
not words or audio frames. Zero uses the remaining 40,960-position context after
the longest prompt. Large budgets reserve more KV memory even when generation
stops early. Small budgets may produce no complete frame: incomplete delayed
frames are omitted, not padded into fabricated codec IDs. The decoder returns
one AUDIO per complete segment; a request with no segments has no audio to save.
Increase the budget if speech is cut short. Codes use the first 16 ordered books
of MOSS v1; decoding produces raw mono 24 kHz PCM without loudness normalization.

The model uses the shared native Qwen3 operations, attention and prefetch paths,
learned audio-pad embeddings and independent output heads. It never executes
checkpoint Python or Transformers model code in production. KV and delay state
are call-local, with interruption checked at each generation step. A wrong model,
missing embedded tokenizer or incompatible weights fails at the loader or
CLIP boundary; use the pinned converter and correct model selection to recover.
Follow the queue prompt ID after a timeout rather than submitting duplicate work.

CPU checks cover tokenization against the pinned processor, independent greedy/
stochastic sampling, warm-up/flush/stopping/continuation, padded cached logits,
complete-frame extraction and CLIP/node ownership. The full original checkpoint
can also be checked against the reference without replacing the serving runtime:

```sh
MOSS_VOICEGENERATOR_SOURCE=/absolute/MOSS-VoiceGenerator MOSS_VOICEGENERATOR_WEIGHTS=/absolute/MOSS-VoiceGenerator/model.safetensors PYTHONPATH=/absolute/worker/transformers5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /absolute/prepared/python -m pytest tests-unit/comfy_test/moss_delay_reference_test.py -q -s
python -m pytest tests-unit/comfy_test/moss_speech_adapter_test.py -q
```

The full-checkpoint test hashes the original weights, requires about 10 GB of RAM
and compares FP32 prefill plus three cached steps across all 17 heads. Omitting
`MOSS_VOICEGENERATOR_WEIGHTS` skips that test. An isolated CPU queue run produced
four seconds of 24 kHz mono FLAC through the complete native graph in 14.81 seconds;
this is one local run, not a performance guarantee. Listening/quality comparisons,
GPU and low-memory lifecycle acceptance, long speech, quantization and serving
adoption remain pending. Existing worker workflows remain available.

## MOSS v1 codec baseline

The native math in `comfy/ldm/audio/moss_audio_tokenizer.py` is adapted from the
Apache-2.0 [OpenMOSS audio tokenizer](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/tree/3cd226ba2947efa357ef453bcad111b6eafba782).
The full [Apache license](../comfy/ldm/audio/moss_audio_tokenizer.LICENSE) accompanies
the adapted module. It implements the original 24 kHz, 1920-sample hop, causal transformer encoder
and decoder with residual LFQ. It uses native operations, exact GELU, shared
interleaved rotary kernels, per-module attention preferences/containers and block
prefetch. All streaming tensors belong to one decode call. The discrete codec
owner uses ComfyUI's model patcher, device selection, memory reservation and
offloading; output codes and PCM are copied to CPU.

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
streaming, accelerator cancellation/offload acceptance or runtime adoption.

### Discrete codec nodes

Place the converted codec in a configured `vae` model directory (or configure
an additional `vae` search path). **Load Audio Codec** validates the selected
path, pinned conversion metadata and complete FP32 tensor inventory. It returns
`AUDIO_CODEC`, separate from a continuous VAE. Wrong revisions, missing tensors,
wrong shapes and reduced-precision weights fail before model allocation. Restore
or reconvert the pinned artifact; do not rename an unrelated checkpoint.

Connect **Load Audio → Encode Audio Codes → Decode Audio Codes → Save Audio**,
with the codec connected to both encode and decode. Encoding mixes channels to
mono and resamples to 24 kHz using the shared audio implementation, without
loudness normalization. Select 1–32 ordered codebooks (32 by default). Decoding
uses the reference 100-frame chunks, returns raw FP32 PCM, and emits a ComfyUI
list of AUDIO values, one per segment/batch item. Downstream nodes run once per
item, preserving unequal lengths without padding. No volume normalization or
clipping is applied. Empty valid segments produce empty audio; saving an empty
clip may be unsupported by the selected audio writer.

`comfy.audio_codec.AudioCodes` version 1 carries an immutable codec identity
(family, revision, sample rate, hop and ordered codebook sizes) and a tuple of
`AudioCodeSegment` values. Each segment contains an integer `[B,Q,T]` tensor,
Python valid frame lengths, and optional source sample lengths at the codec
rate. Codebooks are a prefix in codec order. Decoding validates identity and
valid IDs, ignores storage beyond each valid length, and never casts floats to
IDs. MOSS protocol padding 1024 is invalid inside valid frames. Source lengths
are provenance, not a request to synthesize a partial frame: 1,937 input samples
produce two stored frames, one valid frame, and 1,920 decoded samples. Speech
protocol trimming and reference loudness processing belong to their adapters.

The owner processes one stream at a time and reserves attention/block/output
workspace plus the 169,984,000-byte streaming ring when needed. These conservative
estimates are not a measured low-VRAM guarantee. Full-sequence encoding grows
quadratically in attention workspace. An OOM is reported without silently
changing chunk policy or codebook count; shorten input or free memory and retry.
Cancellation uses normal ComfyUI interruption, with no cross-call codec caches.
The loader registers a patcher reconstruction factory for the existing lifecycle.

Run `python -m pytest tests-unit/comfy_test/audio_codec_test.py -q` for payload,
node schema, containment, lifecycle boundary and cancellation fixtures. The real
1.77-billion-parameter checkpoint also passed a CPU load/encode/decode and exact
decode after unload/reload; observed peak process RSS was about 12.75 GiB. An
isolated CPU queue also completed the five-node load/encode/decode/save graph,
producing a mono 24 kHz FLAC with the expected 1,920 samples. This is a synthetic
fixture, not a speech-quality result or live-service deployment. GPU, dynamic
offload, constrained-memory and audible quality acceptance remain open.

## Remaining integrations

These are shared prerequisites for the requested native MOSS speech family,
SoundEffect v2, LocateAnything, Qwen3-ASR, Qwen3 ForcedAligner and Fish S2-Pro
integrations. Their speech loaders, remaining codecs, generation/alignment nodes,
quality comparisons and live workflows are **not implemented or accepted by this
change**. Existing worker/runtime paths remain unchanged. Integer speech codes
use the discrete codec contract above; SoundEffect's continuous diffusion
path and forced alignment's timestamp classifier remain separate designs.
