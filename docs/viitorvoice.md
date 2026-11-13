# Native ViiTorVoice-NAR sentence editing

This fork ports the masked Qwen3 model and DualCodec/W2V-BERT components from
[ViiTorVoice-NAR](https://github.com/viitor-ai/viitor-voice-nar), implementation
`40359993534de0980eefff2d5a25bddc30f6bf58`, model snapshot
`ZzWater/ViiTorVoice-NAR@c225ea3e999c5bb0092c41f01e1ab0f2576fcf3e`.
It supports sentence insertion, replacement and deletion with the existing
Cutlery compatibility nodes. New execution uses native patchers and operations;
it never launches its ONNX/gRPC/HTTP process group or downloads models.

Source/component tests and offline preparation are implemented. Main-instance
activation, full-checkpoint reference/native comparisons, independent listening,
edit-content, cancellation, offload/reload and transition acceptance remain
pending. These additions do not establish a change to the serving instance.

## Offline preparation

Acquire the pinned model files through Download Manager 2. Keep originals and
worker environments for explicit rollback. Run from this repository with its
documented prepared Python:

```bash
python -m scripts.convert_viitorvoice --source /absolute/models/viitorvoice-nar \
  --destination /absolute/models/viitorvoice-nar.comfyui-native
```

The converter verifies published weight SHA256s, exact configuration/tokenizer
assets and complete inventories. It emits 315 language tensors plus an embedded
tokenizer, 773 W2V-BERT tensors plus normalization mean/std, and 306 folded codec
tensors plus pinned silence codes. Each component has a payload-verified manifest
recording source revisions, original hashes and weight-norm transforms. Existing
outputs/manifests cause refusal; choose a new destination instead of overwriting.
Conversion needs about 5 GB of additional disk space and bounded CPU workspaces;
it runs no inference. Preserve this layout and all sidecar manifests:

```text
viitorvoice-nar.comfyui-native/
  text_encoders/viitorvoice.safetensors
  audio_encoders/viitor_w2v_bert.safetensors
  vae/viitor_dualcodec.safetensors
```

Expose these files in their standard model categories. The codec resolves its
exact semantic companion inside the same prepared bundle, including through
category symlinks. Components from different bundles cannot mix. Reuse the
existing native Qwen3 forced aligner: ViiTor's original weights, configuration
and tokenizer match it. Its original weight SHA256 is
`47831d0e82f96b20e9034dba01a075ee06436654719f6a68289e49f1b65ce0e7`.
Do not substitute the transcription model for alignment.

## Editable graph and controls

The compatibility graph composes existing Qwen loader/alignment/result nodes and
`AudioCodecLoader` / `AudioCodecEncode` / `AudioCodecDecode` with these additions:

- `ViiTorVoiceLoader` loads the native masked model from `text_encoders`.
- `ViiTorVoicePlan` consumes twelve source codebooks and alignment JSON, retaining
  word/character selection, padding, expansion, reference context and duration
  scaling. Its input codes include the pinned 300 ms synthetic edges.
- `ViiTorVoiceGenerate` refines the draft with CFG, step count and a private seed.
  It emits `AUDIO_CODES` and the edit mapping, including structural-frame removal.
- `ViiTorVoiceSplice` joins decoded replacements to the original waveform. Its
  metadata declares source/output sample intervals and internal 10 ms blends.

The Qwen backbone is bidirectional; CFG padding retains its explicit mask.
W2V-BERT uses Shaw relative-key bias, causal convolution, hidden-state index 17,
exact SeamlessM4T features, normalization and average pooling. DualCodec uses mono
24 kHz, 25 Hz, one 16,384-entry semantic book and eleven 1,024-entry acoustic
books. Odd decoder strides have no output padding. Decode adds 25 silence-code
frames on each side and trims four samples, matching the pinned decoder.
MOSS v1/v2 codec identities and version-1 payloads are unchanged.

Source audio plus synthetic 600 ms context must fit the original 30-second encoder
limit. No internal silence is removed. Keep the original waveform separately;
generated spans return to its rate and channels. Every sample outside declared
edit/blend intervals is copied exactly, including variable-duration suffix
offsets. This replaces the old whole-recording PCM16/mono/24-kHz round-trip.
No-change requests return the original AUDIO value without loading a model.
Deleting the entire transcript remains unsupported. Alignment warnings or missing
character mappings fail explicitly.

Defaults retain 32 steps, CFG 2, position temperature 1, class temperature 0,
time shift 0.1, layer penalty 5, six minimum replacement frames and 300 reference
frames. The saved compatibility node has no seed socket; its expansion records
a fresh private seed. Direct native generation exposes an explicit seed.
The pinned word diff can retain duplicate boundary whitespace after deletion;
this does not alter copied PCM.

## Backend and validation

PyTorch attention is provisional on RTX 5090/SM120. Bounded component probes
executed Sage and Kitchen where compatible; Sage substitutes PyTorch for required
additive masks. Kitchen introduced larger numerical error. The installed Flash
wrapper substituted SDPA and is excluded as a separate backend result. No
full-model speed winner is established. Native attention registry/patcher
preferences remain the backend owner; never discard masks. Primary references:
[FlashAttention](https://github.com/Dao-AILab/flash-attention) and
[SageAttention](https://github.com/thu-ml/SageAttention).

Use normal `scripts/setup_worktree.py` preparation. CPU checks:

```bash
VIITORVOICE_SOURCE=/absolute/pinned/source \
VIITORVOICE_REFERENCE=/absolute/models/viitorvoice-nar \
DUALCODEC_REFERENCE=/absolute/extracted/dualcodec-0.4.2 \
python -m pytest tests-unit/comfy_test/viitorvoice_test.py \
  tests-unit/comfy_test/audio_codec_test.py -q
```

They compare small FP32/BF16 Qwen/W2V components, original DualCodec math,
complete meta inventories, pinned token preparation/refinement with/without CFG,
private RNG, cancellation/retry, three edit types and exact stereo PCM splicing
at two rates. Missing reference paths skip only optional reference fixtures.
Full-checkpoint inference belongs to the main queue. Retain prompt IDs, artifacts,
listening/content observations and physical RAM/VRAM/time measurements.

ViiTor source is Apache-2.0 ([retained license](../comfy/viitorvoice/LICENSE));
W2V-BERT derives from Hugging Face Transformers under Apache-2.0. DualCodec 0.4.2
is MIT ([retained license](../comfy/ldm/audio/dualcodec.LICENSE)), with DAC
components from [Descript DAC](https://github.com/descriptinc/descript-audio-codec).
