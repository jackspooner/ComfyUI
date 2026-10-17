# Native GLM-OCR

This fork adds the [Z.ai GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) recognizer
to the existing **Load CLIP → Generate Text → Save Text** path. Upstream ComfyUI
owns model loading, patchers, attention selection, prefetch, memory admission,
offload and cancellation. No Transformers model is loaded for production inference;
there is no extra daemon, environment, download-on-load or network access in the model.
The official weights are MIT licensed. The architecture follows Hugging Face's
Apache-2.0 [GLM-OCR implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_ocr).

## Preparation

Use Python 3.10+ and this deployment's ComfyUI requirements, including PyTorch,
Transformers, tokenizers, safetensors and Comfy Kitchen. Prepare development with
`COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py`.
The serving environment is independent of the development worktree.

Acquire the official files at revision
`2e85a62840ccac27daa451df36c736c4636b8628` through Download Manager 2:
`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`,
`preprocessor_config.json`, `README.md`. Finish downloads before conversion.

```bash
python scripts/convert_glm_ocr.py --source /absolute/model_sources/GLM-OCR/2e85a62840ccac27daa451df36c736c4636b8628 --destination /absolute/models/text_encoders/glm_ocr_bf16.safetensors
```

The offline converter validates the exact tensor inventory, BF16 shapes, payload
ranges and published weight SHA-256, and streams native keys plus the tokenizer.
It rechecks output payload hashes before atomic publication and writes a sibling
`.manifest.json` with revision, input-file hashes, mapping and omitted MTP hashes.
Layer 16's exact 16-tensor MTP inventory is intentionally omitted: ordinary greedy
or sampled autoregressive decoding uses layers 0–15. Unknown keys fail conversion.
Retain originals and manifest. Allow about 2.65 GB for originals plus the converted
checkpoint and workspace. Existing output/manifest paths are refused; inspect them
before deliberately selecting another destination. Interrupted conversion cleans
its own temporary files and does not alter originals.

## Recognition

Choose the converted filename in Load CLIP with type `stable_diffusion` (automatic
detection), connect one Load Image to Generate Text, and use `Text Recognition:`
with sampling off, thinking off, and a modest explicit maximum such as 512 new tokens.
The tokenizer supplies GLM role/image markers. A raw prompt with default template
disabled must supply `[gMASK]<sop><|user|>\n`, exactly one
`<|begin_of_image|><|image|><|end_of_image|>` and `<|assistant|>\n`.
Do not prime recognition with proposed wording. Formula/table task prompts can be
passed through; their quality is unverified. Text-only calls are executable but do
not establish general chat quality. Multiple images must be separate invocations.
Video, audio, history, tools, reasoning and MTP are outside this integration.

Generate Text preserves its first four outputs and appends **raw_text** (index 4):
decoded text with special tokens removed, before whitespace stripping or thinking
separation. Save this output as recognition evidence; generated_text remains the
existing normalized output. Raw decoded text is not raw token IDs. There are no
calibrated confidence values or word polygons. On the ordinary media path, stop
reason/truncation is unknown; do not infer completeness from fluent text or an
empty thinking output. Keep requested budget, raw output and native prompt ID.

## Pixel and memory policy

Crop from original pixels first. The model scales down only above 1,000,000 source
pixels, then adds white pixels on the right and bottom to multiples of 28. It never
enlarges smaller images. This differs deliberately from official smart-resize,
which can enlarge small crops. At grid-aligned sizes our patch packing and CLIP
normalization match the official processor with resizing disabled. Tiny text may
remain unreadable; padding supplies no additional evidence. Preserve source/frame
hash, timestamp, half-open crop xyxy, reduced size, padding, processor revision and
request/workflow IDs with every result. Full-frame reduction must follow cropping.

Generation admission includes expanded image tokens, all decoder KV layers and the
requested new-token budget, plus conservative vision/prefill/workspace allowance
(including unfused attention scores). Weight admission remains patcher-owned. After owner reclamation/loading, the
generation boundary checks actual reusable free memory plus the owner reserve
before entering generation; insufficient capacity fails before KV allocation.
Zero max_length requests all remaining context and may require substantial memory;
no free-VRAM-based shrinking occurs. Insufficient capacity fails normally. Caches
are execution-scoped; block and decode interruption follows normal ComfyUI controls.
Use prompt-specific cancellation and inspect queue/history after an uncertain
response. Never reset queues, delete GPU holds or automatically replay work.

## RouterText, MediaTools and WF5

RouterText's operator configuration selects `backend:"native_text"`,
`nativeFamily:"glm_ocr"`, `acceptsImages:true` and the exact converted filename.
It checks the deployed raw_text output before uploads, uses native GLM markers,
queues one image, defaults to 512 greedy tokens, saves normalized/raw text separately
and returns raw decoded text. Only text output is advertised; no JSON repair,
reasoning, history, tools or independent-image batching is enabled. Existing
MediaTools `text_infer` accepts an admitted image_file; WF5 uses `router.text`
following `image.crop-by-bbox` on original pixels. Discover live route capabilities
before authoring. Source integration/conversion alone do not activate a service.

## Verification

```bash
python -m pytest tests-unit/comfy_test/glm_ocr_test.py tests-unit/comfy_test/text_history_test.py tests-unit/comfy_test/text_generation_length_test.py tests-unit/comfy_test/moss_audio_test.py tests-unit/comfy_test/moss_audio_reference_test.py tests-unit/comfy_test/qwen_vl_image_channels_test.py -q
```

CPU fixtures compare vision, four-norm decoder prefill/cached decode and padded
pixel processing to the installed official implementation, and check checkpoint
ranges and budget-aware admission. They do not establish GPU quality, release,
transitions or maximum-size capacity. Record live acceptance separately, including
real cropped/blank images, cold/warm outputs, physical VRAM, targeted cancellation
and GLM → another native model → GLM / Runtime ASR transitions.

## Bounded acceptance on 17 October 2026

The pinned original checkpoint passed its published SHA-256, complete 526-tensor
inventory/range checks, and conversion to 510 native tensors plus embedded tokenizer.
The manifest records the omitted 16 MTP tensors. Reduced CPU vision/decoder tests
match the installed official implementation in FP32; padded processing and template
IDs match. Full BF16 badge generation produced identical native/reference token IDs;
full-frame generation diverged in uncertain fine print, with first-token logits
mean absolute difference about 0.071. Exact BF16 sequence parity is not promised.

The deployed RTX 5090 completed a clear-text fixture exactly on one cold and three
warm queued runs (about 1.27 s cold, 0.14–0.16 s warm). Pending, running vision and
observed-token decode cancellation passed, as did GLM → complete native Qwen → GLM
and GLM → Runtime ASR → GLM. Native holds were released; idle external ASR residency
remained explicitly accounted for. Coarse physical-memory samples during warm GLM
were about 4.2 GiB including process/context/allocator memory; the cold transition
sample included another model's cached weights and is not a GLM peak estimate.
This is bounded lifecycle evidence, not maximum-context/pressure/crash acceptance.

Literal accuracy failed. On source frame hash
`7918c129d073c892639d47eaf145ccacd1d58b8dfa0610638f1844b23a3d541f`,
the original-pixel Almay badge crop `[334,134,418,211]` produced invented
`SUSTAINABLE` text in both native and official reference. Full-frame recognition
also invented seal text. Blank input produced formatting marks rather than empty
text. Additional white padding did not repair the badge. No expected wording was
supplied and no pixels were enlarged. The production RouterText OCR route remains
unconfigured; MediaTools/WF5 OCR inference is withheld until literal acceptance
passes. The approximately 100-crop human-reviewed evaluation, isolated capacity/
crash tests, detailed allocator peaks and wider device coverage remain unverified.
