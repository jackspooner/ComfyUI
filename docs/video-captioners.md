# Native audio-visual captioners

This fork adds offline single-file conversion and an optional native ComfyUI
extension for video-SALMONN-2-Pro-8B, PadCaptioner-3B and InternVideo3-8B-Instruct. The dedicated model handle
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
| [InternVideo3 8B](https://huggingface.co/yanziang/InternVideo3-8B-Instruct) | `c4602918b65225650d152db2850fe34e01d21fcd` | [model-supplied implementation](https://huggingface.co/yanziang/InternVideo3-8B-Instruct/blob/c4602918b65225650d152db2850fe34e01d21fcd/modeling_internvideo3.py) |

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

InternVideo3 by Ziang Yan and collaborators is Apache-2.0. It accepts visual
questions and captions, using native M²LA with rank-896 latent KV and shared
rotary keys, a 36-layer decoder and the compatible 27-layer native Qwen vision
encoder without DeepStack. It has no audio encoder. The nodes skip its soundtrack
and report `audio_used:false`; the low-level handle rejects supplied audio.
Its provider's recursive perception/action agent, external ASR/web/segmentation
tools, training and hour-long/256K-capacity claims are outside this interface.

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
python scripts/convert_video_captioner.py \
  --source /absolute/model-sources/InternVideo3-8B-Instruct \
  --family internvideo3_8b \
  --destination /absolute/models/text_encoders/video_captioners/internvideo3_8b_instruct_bf16.safetensors
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
InternVideo3 requires about 18.7 GB each for source and output. All 732 weight
tensors are retained without casting or attention-weight transformations; its
tokenizer and processor/config assets are embedded. The converter does not execute
the downloaded remote model code. Full native loading enforces strict coverage.
The shard index may omit optional `metadata.total_size`; published file hashes,
actual payload lengths and the complete expected BF16 tensor inventory still
apply. A declared byte total must match.

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
The single-video decoder normalizes upstream `[axes,batch,tokens]` positions to
the native Qwen layout. PadCaptioner's integer audio/video layout is built on CPU
before rotary preparation transfers it to the execution device.
PadCaptioner's tied output head uses the native cast/materialization context,
including disk-backed virtual weights and registered weight transforms.
Sampling and reported intervals use source FPS. Variable-rate timestamp fidelity
has not been qualified; use a constant-rate source when timing accuracy matters.

`fps` is a sampling ceiling, default 2. Temporal patches repeat only a final odd
frame as required by the model. Audio is mono 16 kHz, padded into 30-second Whisper
chunks, emitting 60 half-second Q-Former tokens each. Padding tokens remain in the
upstream interleaved prompt; PadCaptioner's grounding ranges select actual time
windows independently. No transcript is inserted in place of learned audio.
SALMONN uses timestamp-separated visual groups and DeepStack; PadCaptioner uses
audio/video MRoPE and bidirectional media-prefix visibility.
InternVideo3 uses timestamp-separated visual groups and its pinned no-default-system
chat template. It reuses the visual preprocessing/position algorithm, with its
interleaved MRoPE and DeepSeek pair layout in the custom decoder. `max_tokens` is
its total answer budget; `max_events` is unused. Its default MediaTools prompt is
“Please describe this video in detail.” Start with short clips and modest frame
resolution; this API retains the same 120-second/512-frame bounds for all models.

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

## Constrained JSON

Both caption nodes append optional `response_format` (`text` by default) and
`json_schema` string controls. Existing API graphs may omit them. Select
`response_format="json_schema"` and supply a nonempty schema; providing a schema
in text mode, unsupported formats such as `json_object`, and invalid schemas fail.
Prepare the existing optional pins with normal package tooling and check readiness
using `COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py --text-json`.
Node execution never installs dependencies or downloads a tokenizer.

The schema policy is the existing [comfy-json-v1 subset](text-json.md#comfy-json-v1-subset):
closed root/nested objects, homogeneous arrays, scalar values and supported bounds;
no references, arbitrary keywords or network resolution. The 128 KiB schema and
grammar resource limits apply. For example:

```json
{"type":"object","properties":{"caption":{"type":"string","minLength":1}},"required":["caption"],"additionalProperties":false}
```

All three use the package's authoritative embedded tokenizer and configured stop
IDs. The shared `PreparedJSON` compiles on CPU before GPU admission, verifies the
native assistant boundary and creates request-local matchers. Schema instructions
are included in the actual input/context calculation. There is no thinking phase,
speculative decoding, schema repair or unconstrained fallback. Greedy decoding uses
the shared hard `-inf` mask and legal-logit checks on every token, including EOS.
Completion at the last allowed token succeeds; incomplete JSON or unusable legal
logits fail without publishing a successful result. No token budget is increased.

SALMONN and InternVideo3 return the exact constrained object text in `text`, its
parsed value in `structured`, and `structured_output` enforcement evidence.
PadCaptioner preserves learned planning/intervals. Each event caption starts at
its event embedding with a separate matcher; completed branches cannot mutate
another branch's grammar. The caller schema applies to **each event's caption
object**, not the interval envelope. Each event contains its JSON `text`,
`structured` value and evidence. Top-level `structured` and JSON `text` contain
`{"events":[{"intervals":[[0,1]],"caption":{"caption":"example"}}]}`;
that envelope is assembled deterministically from the learned intervals and
validated captions. Top-level evidence has `scope="event_caption"`,
`event_count` and `finish_reason="all_events_complete"`. An empty plan or any
incomplete event fails. JSON mode does not certify event detection, interval
accuracy or repair the separately observed free-text empty-caption issue.

Evidence retains `enforcement="native_schema"`, `backend="llguidance"`, backend
version, schema/tokenizer hashes, family, stops, completion and independent
validation. `request_schema_sha256` binds the exact supplied UTF-8 schema string;
`schema_sha256` identifies the native compact schema. `scope="answer"` applies
to the other two models. The existing `result_json` output contains this metadata;
the two node output positions stay unchanged. MediaTools forwards `responseFormat`
and `jsonSchema`, independently validates the collected objects, checks scope and
request-schema identity, and preserves same-job recovery. Plain text adds no
enforcement evidence and retains its existing output behavior.

Run `python -m pytest tests-unit/comfy_test/video_caption_json_test.py tests-unit/comfy_test/text_json_native_test.py -q`.
Set `VIDEO_CAPTION_JSON_FIXTURES` to the existing converted `video_captioners`
directory to additionally test all three actual embedded tokenizers, reading only
headers and tokenizer tensors. The fixtures exercise illegal EOS/control-token
preferences, final-token completion, incomplete-budget failure, Unicode and
independent PadCaptioner branches. Heavy real-media validation must use the main
ComfyUI queue and remains distinct from those CPU checks. Discovery of the new
sockets establishes source activation, not output quality or maximum capacity.

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

Local acceptance on 2026-11-04 loaded all three complete BF16 packages on the
main RTX 5090 queue through MediaTools, including submit/watch/collect and JSON
delivery. SALMONN and InternVideo3 described a ten-second interview, but both made
motion errors on a synthetic clip. Detailed identity/speech claims and broad
quality remain unqualified. PadCaptioner emitted empty event descriptions on
synthetic, upstream-demo and interview clips, including a normal 24-event budget.
Its completed queue/delivery state is not usable-caption acceptance. An adapted
pinned-upstream CPU diagnostic also emitted immediate EOS for each branch; it
used a serial generation/cache shim and native visual preparation, so this does
not establish full-checkpoint upstream parity or conclusively locate the cause.
Keep PadCaptioner experimental until nonempty captions and grounding are verified.

### InternVideo3 backend selection and checks

The same RTX 5090 SM 12.0 / Torch 2.14.1+cu130 target applies. The native vision
encoder reuses ComfyUI attention dispatch. M²LA prefill expands one layer's latent
KV, zero-pads values to the 256-wide QK head, dispatches native attention, then
discards the padded dimensions. Decode absorbs the up-projection into query/output
einsums and retains only latent/rotary tensors. ModelPatcher, native cast contexts,
prefetch and interruption checks own weight movement and cancellation.
Admission estimates compressed KV plus expanded prefill/workspace separately.

Compared the pinned upstream eager/SDPA/FlashAttention implementations with
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
and the [vLLM model table](https://docs.vllm.ai/en/latest/models/supported_models/).
That table lists InternVideo 2.5 under InternVL, not this exact M²LA architecture;
the [SGLang registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py)
discovers implementations dynamically and is not proof of model support. No
compatible external engine was established or benchmarked. The native selection
is provisional until main-queue quality, cold/warm timing, peak memory and actual
kernel execution are measured; no optimal-backend claim is made.

Run `python -m pytest tests-unit/comfy_test/internvideo3_test.py -q` in the prepared
CPU test environment. Set `INTERNVIDEO3_REFERENCE` to the pinned local
`modeling_internvideo3.py` to additionally compare the upstream attention class's
prefill/absorbed decode and the native vision encoder on a temporal non-square
grid. This compiles only selected inspected definitions;
it does not import upstream model registration or network hooks. Tiny fixtures
and strict checkpoint construction remain separate from full-checkpoint output
parity and real-video acceptance. Both require activation and the main queue.
