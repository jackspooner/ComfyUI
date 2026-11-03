# Native audio-visual captioners

This fork adds offline single-file conversion and an optional native ComfyUI
extension for video-SALMONN-2-Pro-8B and PadCaptioner-3B. The dedicated model handle
is intentional: PadCaptioner returns event intervals from learned heads, plans
events, and decodes branch-isolated captions. This differs from ordinary CLIP
conditioning or one text-generation stream. Both models reuse native Qwen
text/vision, Comfy operations, ModelPatcher, attention dispatch and prefetch.
They do not launch an upstream Python inference worker or a second GPU service.

## Sources and scope

| Model | Checkpoint revision | Implementation reference |
| --- | --- | --- |
| [SALMONN Pro 8B](https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-8B) | `89319a9934169266b8d5da3d7d4850e9f714870f` | [Tsinghua/ByteDance](https://github.com/bytedance/video-SALMONN-2/tree/2d28e841c8378885d2f30982be801d0c44f7a5bb/video_SALMONN2_pro) |
| [PadCaptioner 3B](https://huggingface.co/wenzhengzeng/PadCaptioner-3B) | `a1eb578dd619be63819d8342d8d49fa2676d20e8` | [Show Lab](https://github.com/showlab/PadCaptioner/tree/ae3f3bc11a66e99d18cdeb457e3c27cf15869dd6) |

Credit Tsinghua University, ByteDance and the PadCaptioner authors. The adapted
SALMONN model code carries Apache-2.0 notices; the upstream sources and model
cards retain their own licensing and citations. The inference-only rotary and
grounding helpers retain their upstream algorithms without training losses.

SALMONN supports free-form aligned audio/video descriptions and questions.
PadCaptioner exposes dense event planning, boundary snapping, coverage masking,
attention-weighted event aggregation and isolated parallel caption branches.
Its six other upstream task APIs, training/evaluation suite, stochastic decoding,
deduplication heuristics and automatic retries are outside this interface.
Intervals are estimates, not verified timestamps or measurements.

## Preparation and conversion

Run [native worktree setup](native-model-integration.md#2-prepare-the-repository-and-source-assets)
with a prepared interpreter. The current conversion target is dense BF16; it
does not quantize weights. Download every file from the pinned model snapshot
through Download Manager 2 after the scoped Firefox cookie import. Keep source
files outside Git. Wait for verified complete downloads before conversion.

```sh
python scripts/convert_video_captioner.py \
  --source /absolute/model-sources/video-SALMONN-2-Pro-8B \
  --family salmonn2_pro_8b \
  --destination /absolute/models/text_encoders/video_captioners/video_salmonn_2_pro_8b_bf16.safetensors
python scripts/convert_video_captioner.py \
  --source /absolute/model-sources/PadCaptioner-3B \
  --family padcaptioner_3b \
  --destination /absolute/models/text_encoders/video_captioners/padcaptioner_3b_bf16.safetensors
```

The converter checks published weight SHA256 values and pinned configuration,
loads PadCaptioner's pickle container with `weights_only=True` and `mmap=True`,
and embeds tokenizer/configuration/processor assets in one safetensors package.
It verifies tied output embeddings before omitting their duplicate and omits
unused Q-Former text-position indices. Every other parameter is retained.
The manifest records mappings, omitted tensors, revisions and per-tensor hashes.
The shared streaming writer independently re-reads output payloads and publishes
without overwriting an existing package. Loading enforces exact native tensor
coverage. Neither conversion nor loading contacts the network.

Budget approximately 19.4 GB source plus 19.4 GB output for SALMONN; PadCaptioner
needs 9.4 GB source, 9.4 GB temporary safetensors and 9.4 GB output. Originals are
preserved. Resident conversion memory is bounded by mapped storage and the
serialization library, not a complete CUDA model. An interruption leaves no
published partial checkpoint; inspect only task-owned temporary files before
retrying. Use a fresh destination if a complete package already exists.

## Nodes and execution

The optional extension is `comfy_extras/video_caption_extension/`. Link that
directory into the selected instance's `custom_nodes` directory only as part of
its normal deployment, then restart after the service owner's activity and
continuity checks. This avoids modifying the shared Load CLIP detector or
registering a broad Qwen signature. A running process does not discover new
source merely because the Git branch was integrated.

Discover `LoadVideoCaptioner`, `CaptionVideo` and `CaptionVideoFile` on the live
instance. Select the exact converted filename under `text_encoders`.
`CaptionVideo` takes an IMAGE batch, actual `source_fps` and optional AUDIO.
`CaptionVideoFile` takes an uploaded input video, decodes at most a 512-pixel edge,
limits duration to 120 seconds and sampled frames to 512, and optionally includes
the original soundtrack. It never silently trims a source that exceeds the limit.
Both output text and JSON. An explicit `use_audio:false` ignores the soundtrack.
Sampling and reported intervals use source FPS. Variable-rate timestamp fidelity
has not been qualified; use a constant-rate source when timing accuracy matters.

`fps` is a sampling ceiling, default 2. Temporal patches repeat only a final odd
frame as required by the model. Audio is mono 16 kHz, padded into 30-second Whisper
chunks, emitting 60 half-second Q-Former tokens each. Padding tokens remain in the
upstream interleaved prompt; PadCaptioner's grounding ranges select actual time
windows independently. No transcript is inserted in place of learned audio.
SALMONN uses timestamp-separated visual groups and DeepStack; PadCaptioner uses
audio/video MRoPE and bidirectional media-prefix visibility.

Greedy `max_tokens` applies to the total SALMONN answer or each PadCaptioner event.
`max_events` limits PadCaptioner planning. Use the upstream dense-caption prompt:
“Could you outline the incidents that happened during different time periods in
the video?” PadCaptioner JSON marks event captions cut off at the token cap.
Context admission reserves output and estimates KV/workspace separately from
weights. Measurements are still required before claiming a universal VRAM bound.
Caches and grounding state are local to one invocation; cancellation is checked
between model blocks and decode steps. ComfyUI retains residency/offload ownership.

Submit inference through the **main instance's normal queue**, even when busy;
retain its prompt ID. After timeout, inspect that prompt's queue/history instead
of resubmitting. MediaTools adds dedicated `video_caption_models`, `video_caption`
and `video_caption_job` tools, retaining admitted source snapshots and durable
job/key identities, and publishing JSON results through MediaIO. It does not add
video files to RouterText's `text_infer` contract.

## Backend and acceptance boundaries

The inspected target is an RTX 5090 (SM 12.0, 32,607 MiB), driver 615.71.09,
Torch 2.14.1+cu130 and Transformers 5.21.1. Existing native attention dispatch
is selected so the installed Comfy Kitchen/ComfyUI path owns compatible kernels.
The query-only BERT attention uses native attention; PadCaptioner requires additive
branch masks and one attention-row readout for grounded aggregation.

Upstream SALMONN also provides a custom [vLLM plugin](https://github.com/bytedance/video-SALMONN-2/blob/2d28e841c8378885d2f30982be801d0c44f7a5bb/video_SALMONN2_pro/qwenvl/vllm_plugin.py).
It was considered alongside [vLLM's supported-model interface](https://docs.vllm.ai/en/latest/models/supported_models/)
and [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
A separate serving engine would require independent resource ownership and
PadCaptioner's custom decoding integration. Its comparative performance has not
been measured. This native selection is provisional, not an optimality claim.

Run `python -m pytest tests-unit/comfy_test/video_caption_test.py -q` with the
prepared CPU test environment. Current component fixtures cover query attention,
cached/full decoding equivalence, branch isolation, temporal frame packing,
audio features and grounding readout. Full checkpoint loading, upstream parity,
normal-queue real-video quality, repeated runs, cancellation, model transitions,
cold/warm performance and peak memory remain separate acceptance gates until
recorded. CPU fixtures and all-parameter inventory checks are not GPU acceptance.
Other accelerators, quantization and long-video capacity are unqualified.
