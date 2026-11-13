# Native LocateAnything

This fork implements the installed [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)
release `c32291ca5e996f5a7a485845b4f57a233936bba0`. MoonViT, its projector and
the modified Qwen2 decoder execute through native operations, attention selectors,
CoreModelPatcher, block prefetch and the normal queue. Cutlery owns task prompts,
coordinate formatting and its six compatibility outputs. The source implementation
and weights use the [NVIDIA License](../comfy/locateanything/LICENSE), including
its research/evaluation use limitation. Native Qwen components retain their
existing Qwen/Hugging Face attribution. This port does not add training or video.

## Preparation and graph

Use the normal ComfyUI runtime and documented worktree readiness command. No
additional worker, Transformers overlay, source Python import or download is
needed at inference. Acquire missing artifacts through Download Manager 2.
Preserve the original complete local bundle and run:

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py
python -m scripts.convert_locateanything --source /absolute/LocateAnything-3B --destination /absolute/LocateAnything-3B.comfyui-native
```

The converter authenticates exact source/config/tokenizer files and both pinned
shards, validates the complete 770-tensor native meta inventory, then streams and
independently verifies every output payload. It writes:

| Component | Category/file | Inventory |
| --- | --- | --- |
| MoonViT and projector | `clip_vision/locateanything_vision.safetensors` | 335 BF16 tensors, 861,021,664 weight bytes |
| Modified Qwen2 | `text_encoders/locateanything_language.safetensors` | 435 BF16 tensors, 6,800,310,272 weight bytes, embedded tokenizer |

Both files require their sibling `.manifest.json`, retaining exact source
identities, all tensor hashes and tokenizer identity. Originals remain unchanged;
existing outputs/manifests are refused, including concurrent publication. Retain
completed components after interruption; inspect them and choose a fresh output
destination for another complete conversion. Allow about 7.7 GB additional disk
space for completed weights plus a temporary component during publication.

Expose the completed category directories through normal model-root configuration.
Compose `LocateAnythingLoader` → `LocateAnythingGenerate` → `SaveText`. The loader
selects the two exact component filenames, native device or CPU, and the existing
native attention registry. Use an explicit task prompt with the generator; Cutlery
compatibility graphs preserve their existing eight task choices and prompt text.
Missing/mismatched packages fail with preparation instructions and never start a
worker or substitute another model. Native CPU execution uses FP32; CUDA uses BF16.

Image preparation preserves bicubic resizing, the pinned 25,600-patch allowance,
14-pixel patches, normalization and 2×2 merge order. It processes the first input
IMAGE item. Coordinates remain normalized integer tokens until the Cutlery result
adapter restores the original image dimensions. Image size controls visual work;
text and expanded image positions share the pinned 16,384-token generation limit.
As in the reference, multi-token emission can finish a prediction window beyond
the requested output allowance. Seeds, top-p 0.9, temperature 0.7 and repetition
penalty 1.1 preserve the existing adapter contract; request RNG and caches are private.

## Masks and backend evidence

Hybrid/fast use six-token prediction windows. Their bidirectional window excludes
the preceding duplicate token, and speculative KV positions roll back before the
next iteration. Hybrid switches to autoregressive decoding for malformed boxes
and returns on a box end. Slow remains autoregressive. This is not ordinary causal
Qwen generation. Existing native Qwen blocks and optimized rotary operations are
reused with the exact masks and positions.

On RTX 5090/SM120, bounded component tests compared PyTorch, Sage and Kitchen INT8
at vision, MTP prefill/cache and autoregressive cache shapes. Sage substituted
PyTorch for the explicit MTP masks; Kitchen executed with higher numerical error.
PyTorch is the provisional FP32/BF16 quality baseline and loader default. Attention
selection uses normal object patches and remains explicit. These component tests
do not establish a full-model speed winner or GPU quality acceptance.

The current upstream [LA Flash sparse executor](https://huggingface.co/nvidia/LocateAnything-3B/blob/c32291ca5e996f5a7a485845b4f57a233936bba0/kernel_utils/README.md)
packs visible KV ranges for `flash_attn_varlen_func`. Its stated hardware tests
cover A100, RTX 4090 and H100; the exact exported varlen package is absent from
this deployment. It has not been benchmarked or substituted with a dense kernel.
The separately inspected [Magi/FA4 source pair](https://github.com/SandAI-org/MagiAttention/tree/8472cf27fe184d6f788eafa7d273d0ff560eb565)
rejects the needed arbitrary-mask path on SM120. Dense SM120 support is not evidence
for either sparse path. Native sparse execution remains unverified.

## Verification and activation

Run the source-pinned, model-free CPU suite with both reference paths:

```sh
LOCATEANYTHING_REFERENCE=/absolute/LocateAnything-3B LOCATEANYTHING_REFERENCE_RUNTIME=/absolute/isolated-transformers-4.57.1 python -m pytest tests-unit/comfy_test/locateanything_test.py -q
```

It compares exact preprocessing/tokenization, small complete FP32/BF16 vision and
projector outputs, cached Qwen logits, samplers, masks, pattern switching, private
state, cancellation/retry and complete checkpoint headers. Full weights never run
in that suite. The unchanged worker completed 24 main-queue references covering
all eight tasks and three modes. Fast merged two nearby tomatoes in the reference
fixture; that quality limit is retained rather than treated as native acceptance.

Source integration and prepared packages do not activate serving code. Full native
main-queue comparisons, empty/multiple/resized cases, cold/warm resource measures,
pressure offload, cancellation, reload and another-model transitions remain
required. Deployment must use the existing producer/native maintenance fences;
unsettled WF5 evidence currently blocks activation. Preserve worker environments
for explicit source rollback and original accepted-job recovery. Never replay an
uncertain prompt or clear historical records to make maintenance pass.
