# Native PP-OCRv6

This fork adds **Load PaddleOCR Models → PaddleOCR**, using native ComfyUI
operations, attention, model patchers, prefetch and queue cancellation. It ports
PaddlePaddle/Hugging Face's Apache-2.0 LCNetV4, medium detector and LightSVTR/CTC
recognizer. [Upstream PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and
[its README](https://github.com/PaddlePaddle/PaddleOCR/blob/main/README.md) own the
original training system and broader OCR/document pipelines. This implementation
does not include those pipelines, NRTR training heads or document orientation models.

## Models and setup

Use the ordinary prepared ComfyUI Python >=3.10 environment, including PyTorch,
torchvision, NumPy and safetensors, plus OpenCV Python for region geometry.
OpenCV is already installed on the evaluated host; other installations must
provision it explicitly in their prepared runtime. It is an optional OCR
prerequisite, not an automatic installation or a new core-wide requirement.
Paddle, PaddleX, Transformers inference services, additional environments and
network access are not required during native loading or execution.

Acquire these official model repositories through Download Manager 2:

| Model | Pinned revision | model.safetensors SHA-256 |
| --- | --- | --- |
| [PP-OCRv6 medium detector](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors) | `4236c2b61741a259c091fd879dcc4edc339e916c` | `bd393266c02e1a680b1b34c301d5d0d81e6290440b7f8ab0f5d5032276b17eb1` |
| [PP-OCRv6 medium recognizer](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_safetensors) | `024cad6a831de75c2c3c26e711ba8c4a82ccd24b` | `5f43c16f2a684b1d2284662178bdb604febd3d6bfdb5ca73828d08d0f7c0c3e9` |

Keep originals. Put each model's `model.safetensors`, `config.json` and
`preprocessor_config.json` together in separate directories under a configured
`detection` model root, for example:

```
models/detection/paddleocr/PP-OCRv6_medium_det/model.safetensors
models/detection/paddleocr/PP-OCRv6_medium_det/config.json
models/detection/paddleocr/PP-OCRv6_medium_det/preprocessor_config.json
models/detection/paddleocr/PP-OCRv6_medium_rec/model.safetensors
models/detection/paddleocr/PP-OCRv6_medium_rec/config.json
models/detection/paddleocr/PP-OCRv6_medium_rec/preprocessor_config.json
```

Preserve model cards/licensing alongside those files. Both exported checkpoints
are FP32, about 165 MB combined; that is storage size, not peak execution memory.
The loader validates the pinned weight/configuration/vocabulary hashes and complete FP32
tensor shapes/inventory. The official medium recognizer uses configuration type
`pp_ocrv6_small_rec` with medium dimensions; do not rename it or substitute small
weights. No converter is needed. Wrong files fail clearly without downloads or
automatic substitution. After uncertain acquisition, recover existing download
IDs and verify payloads before loading; never queue duplicates as a retry strategy.
Validation uses an allocation-free schema so deferred ComfyUI linear parameters
remain subject to complete shape checks. OCR requests full-precision attention
through the selected dispatcher, including under `--use-ck-attention`.
The allocation compiler records one forward scope because retained backbone
features cross block lifetimes; the block queues still prefetch weights.

## Workflow and outputs

1. Select the two exact files in **Load PaddleOCR Models**.
2. Connect its `PADDLEOCR_MODEL` output and a Load Image IMAGE to **PaddleOCR**.
3. Select `detect_and_recognize` for complete images, or `recognize_lines` for
   already cropped single text lines. These are different tasks: a badge containing
   several curved lines is not a single text line.
4. Start with default `no_enlargement`. Use `official` explicitly for comparison.
5. Save or preview both STRING outputs using existing text nodes.

Detection controls default to `threshold=0.3`, `box_threshold=0.6`,
`unclip_ratio=1.5`. They control probability-map binarization, accepted region
scores and geometric expansion. Recognition has no additional confidence filter.
Do not supply expected words or repair text semantically.

The first output contains raw decoded region text, separated by newlines and
explicit `[Image N]` headers. The second is UTF-8 JSON, version 1, with operation,
preprocessing, checkpoint/config identity and an `images` list. Each image has its
index, width/height, RGB-byte `pixel_sha256`, detector transform and ordered
`regions`. Each region contains four input-coordinate corners, detection score,
text, recognition score, optional rectification and recognition transform.
Ordering is deterministic by top coordinate then left coordinate, not document
layout understanding. Scores are model scores, not correctness probabilities.
Geometry describes detected text regions, not guaranteed individual words or
characters. Recognition-only uses the entire supplied image as its region and
has no detection score. No detection returns an empty region list; blank CTC text
has null recognition score. JSON contains no NaN. Independent batch items remain
independent images, never stitched pages or shared recognition canvases.

Transforms record source/resized/input sizes, right/bottom padding, perspective
matrix, rectified size and rotation. Perspective crops use bicubic sampling and
rotate tall regions 90 degrees counterclockwise. Source pixels are never enhanced;
rectification is an explicit resampling step. Image inputs use the usual 0–1 RGB
contract and are quantized to 8-bit pixels; alpha is ignored. Pixel hashes are not
original-file hashes. Callers retain original file hashes, crop offsets, frame
timestamps, queue IDs and downstream provenance.

## Native inference batching

Optional advanced controls `detector_batch_size` and `recognition_batch_size`
default to **1**, preserving old prompts. These are actual forward-batch limits,
not simultaneous ComfyUI jobs. Start a measured GPU trial with 2 detector frames
and 8 recognition crops; this is a trial configuration, not a proven optimum.

- Detection prepares a bounded window of frames, groups identical prepared
  `(C,H,W)` shapes and concatenates only the batch dimension for each forward.
- Recognition prepares a bounded window of text crops across those frames,
  grouping identical prepared `(3,48,W)` shapes. A text-heavy single image can
  benefit from crop batching even when detector batching stays at one.
- In `recognize_lines`, the recognition control sets the source-image window;
  detector batch size is irrelevant. Partial final windows are processed.
- Different prepared widths are **not** padded together. Additional padding
  changes the recognizer's attention and squeeze-excitation calculations.
  Heterogeneous sizes can consequently yield smaller batches or singletons.
- All entries retain their original image/region position, dimensions, pixel
  hash and transforms. No detections remains an empty entry, not a dropped image.

The primary `images` socket still accepts standard BHWC IMAGE batches. Optional
Autogrow `additional_images` sockets accept further IMAGE batches, including
different dimensions. API fields are `additional_images.image0`,
`additional_images.image1`, etc. (up to the framework's 100-socket limit). The
execution order is primary frames first, then additional sockets in numeric
order, then frames within each socket. Do not resize independent source frames
merely to concatenate them before OCR. A bare HWC tensor is not the IMAGE contract.

Workspace admission scales with the actual forward batch. There is no automatic
precision reduction, silent batch fallback, retry, persistent tensor cache or
queue-policy change. Failure/interruption fails the node without publishing a
partial success. Callers that collect errors must attribute the failure to every
frame in that submitted batch and retain their slot map; never zip filtered
successes against the original source list. Source/region pixels and probabilities
are bounded to the current windows; raw JSON results accumulate for the request.

### Batching verification and activation

The 47 focused CPU tests pass, including real pinned FP32 checkpoint batch versus
concatenated singleton forwards (rtol 1e-4/atol 1e-5, exact recognizer CTC IDs),
small-architecture upstream comparisons, mixed shapes, tails, blank frames,
identity order, crop batching, old node calls, cancellation and batch workspace.
To include the two opt-in checkpoint cases, set `PADDLEOCR_BATCH_MODELS` to a
local directory containing `PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` when
running the test command below. Without it those two tests skip. No downloads
or GPU inference occur in these tests.

On 25 October, the RTX 5090 serving process was activated at `7d1038360` and
advert-frame queue tests exercised detector batches of two and recognizer batches
of eight. A warm 14-frame comparison returned 80 regions in both modes, identical
recognized strings and one changed region geometry. Queue execution decreased
from 3.348 to 1.576 seconds (about 2.1×); sampled device memory reached 3,400 MiB.
The earlier smaller comparison also exposed a text difference, so CUDA batching
is a numerical change, not an output-identical optimization. These are model-output
comparisons, not independent reading of source text, guaranteed peak-memory safety
or whole-workflow throughput measurements.

WF5 paired-frame acceptance additionally covered a singleton tail, a blank frame,
empty input and a failed pair followed by a successful tail. Both failed source
positions remain represented in errors. Detailed receipts are in the LAMBDA
project's `data/lambda/annotations-v09-streamlined/dispatch/2026-10-25-r54-batched-ocr/`.
Discover sockets and benchmark the selected device before increasing batch sizes.
Existing admitted workflows retain their pinned definitions; updating native code
does not upgrade historical or in-flight corpus outputs. GPU pressure and batched
cancellation beyond the existing CPU cancellation tests remain unverified.

## Preprocessing and limitations

`no_enlargement` white-pads detector images on the right/bottom to multiples of
32. It trims detector padding before extracting regions. Recognition reduces
lines taller than 48 pixels, preserves smaller line scale and white-pads their
height to 48; normalized width padding to at least 320 is zero. Very wide inputs
still need memory proportional to width and attention workspace.

`official` reproduces the pinned Transformers preprocessing: detector min-side
736 sizing, max-side 4000 and rounded multiples of 32; recognizer height 48,
upstream width rounding/cap at 3200 and normalized width padding. Detector
normalization/channel order and recognition RGB→BGR conversion follow the actual
exported processors. This mode can enlarge small inputs. It adds no missing detail.

Ordinary quadrilateral rectification does not unwrap circular text. Tiny, curved,
artistic and blurred text may be omitted or misread, including high-score errors.
The no-enlargement mode is a separate quality experiment, not a promise of equal
accuracy to upstream resizing. Keep failures visible; recognized words alone do
not establish advertising claims. BF16, quantization, training, semantic repair,
orientation classifiers and learned enhancement are outside this version.

## Verification and operational status

Source registration requires a separately authorized idle activation before the
nodes appear in an already running service. Do not restart active jobs. Discover
`/object_info/LoadPaddleOCRModels` and `/object_info/PaddleOCR` on the selected
instance before submitting the example workflow. Submit through the ordinary
queue even when it is busy; retain prompt IDs and inspect history after ambiguous
responses instead of replaying. Target cancellation to the returned prompt ID.
Model admission and prompt tracking own both patchers; no model-owned cache or
separate worker is introduced. Workspace estimates include detector feature maps,
recognition output probabilities and attention. Admission failures request smaller
inputs rather than silently changing a model or precision.

Prepare a worktree with the documented `scripts/setup_worktree.py` command and
the `--paddleocr` flag, which checks OpenCV/torchvision read-only, then:

```
python -m pytest tests-unit/comfy_test/paddleocr_test.py -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/paddleocr_reference.py \
  --detector /absolute/original/detector-directory \
  --recognizer /absolute/original/recognizer-directory \
  --cases /absolute/cases.json --output /absolute/new-evidence-directory
```

The optional CPU reference harness needs the prepared Transformers 5.20.0
environment; production nodes do not import its model classes. Cases are a JSON
list with `id`, `path`, optional original-pixel `crop: [x0,y0,x1,y1]`, optional
`operation: recognize_lines` and an optional manually transcribed `expected`.
`known_strings` records partial transcription and is excluded from CER. Expected
text is used only after inference. Output includes paired native/reference NPZ
probability arrays, transforms, raw text, SHA-256 identities, elapsed times and
case-sensitive CER with separate substitutions/deletions/insertions; whitespace
is collapsed and Unicode normalized to NFC for scoring. The output directory must
be new, preserving prior evidence. Failures retain evidence and exit nonzero.

CPU component and full-checkpoint parity pass at rtol 1e-4/atol 1e-5 and CTC sequence
equality across both preprocessing modes. Five manually transcribed supplied line
crops (75 normalized characters) have 0 substitutions/deletions/insertions in
official mode, and 1 substitution/14 deletions/0 insertions (20% CER) without
enlargement. This small single-advert sample is not a general accuracy benchmark.
The isolated badge yields no text without enlargement and `100` then `8` with
official preprocessing; neither correctly reads its circular lettering. A full
frame plus clear, blank, multilingual and rotated controls also exposes small-text
and orientation failures. The first mixed detection/crop suite has 30.30% CER
without enlargement and 31.06% with official preprocessing over 132 scored
characters; narrow supplied lines should use recognition-only rather than
redetection. Do not interpret these mixed-task totals as a model ranking.
Focused tests cover small randomized architectures, exported processor
parity, no enlargement, CTC blanks/repeats, detection geometry/thresholds, path
containment, independent images, strict weights, cancellation and admission order.
118 targeted/regression tests pass; one optional MOSS reference module is skipped
because its separate reference source is not configured. An isolated CPU ComfyUI
queue successfully executes cold/warm clear controls and a blank-result badge.
GPU lifecycle evidence must be reported separately. No RouterText,
MediaTools, WF5 or advert-automation production route is added by this change.


### RTX 5090 advert acceptance (17 October 2026)

Twenty normal queue submissions succeeded with the pinned FP32 models, dynamic
VRAM, allocation compilation and the deployment's Comfy Kitchen attention flag.
Two activation defects were found and repaired: lazy linear parameters had been
mistaken for missing checkpoint keys, and block allocation scopes were too short
for retained convolutional features. FP32 attention is explicitly requested.

On the original Almay frame, official preprocessing reads the main package copy
and English footer, with errors in multilingual fine print. The five supplied
single-line crops reproduce the CPU results: official has zero errors over 75
normalized characters; no-enlargement has one substitution, fourteen deletions
and no insertions (20% CER). This is one advert, not a general OCR benchmark.
The 84×77 original-pixel badge still returns no text without enlargement and
`100` then `8` with official preprocessing. It does not verify `100%` or
`BIODEGRADABLE FIBERS`; the incorrect `8` has recognition score about 0.774.

Queue execution time was 1.320 seconds for the cold clear control and 0.068
seconds warm; the full advert took 0.407 seconds without enlargement and 0.522
seconds with official preprocessing. Process GPU-memory samples at 100 ms
intervals reached 2,066 MiB during these OCR runs; this includes runtime overhead
and can miss short peaks. Timings exclude upload and queue waiting. Every
comparable GPU transcription matched its CPU counterpart. This establishes text
agreement, not GPU-native versus official probability-map parity. Other GPUs,
low-memory pressure and non-CUDA backends remain unverified. Preserve the saved
prompt histories, source/crop hashes, geometry and raw erroneous outputs when
using these observations; advert routing remains withheld.

Targeted cancellation interrupted the OCR node in a repeated-image batch, and
the following clear control succeeded. Explicit idle offload/reload and OCR →
native Qwen → OCR → RuntimeHost ASR → OCR transitions also succeeded through the
normal queue and coordinated GPU ownership. Rotated controls still misread text;
no orientation model was added. A clear/blank two-image batch preserves image
boundaries and leaves the blank result empty.
