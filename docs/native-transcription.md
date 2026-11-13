# Native Qwen and Whisper transcription

`comfy.asr.qwen` and `comfy.asr.whisper` implement transcription inside ComfyUI.
Cutlery-Nodes supplies graph loaders, AUDIO adapters and text formatting;
RouterAudio composes those nodes. No RuntimeHost or CTranslate2 inference is used
by these native paths. Existing explicit legacy nodes remain separate.

Each encoder and decoder has a native CoreModelPatcher. Transformer blocks expose
dynamic allocation units and use the existing prefetch queues. Model management
owns admission, CPU residency, pressure reclamation and reload. Handles expose
`get_models()` to prompt resource tracking. Audio/text results contain CPU data;
KV/beam state and selected alignment heads belong to the executing call.
The reused Qwen audio encoder copies each compiled block's output into one
execution-local hidden buffer, releasing block temporaries before their allocation
scope closes. Matching core sources
are required; an allocator compilation error must not trigger a legacy fallback
or disabling dynamic memory management.

## Requirements and preparation

Use the fork with Transformers **5.22.2**, PyTorch, NumPy, tokenizers and
safetensors from the prepared runtime. `scripts/setup_worktree.py --asr` checks
these optional prerequisites without installing packages or loading models.
Whisper VAD additionally uses Faster-Whisper's CPU ONNX preprocessing package;
it never constructs a CTranslate2 inference model. Optional Qwen Japanese/Korean
alignment needs nagisa/soynlp. Missing alignment dependencies retain the transcript
with `transcription_timestamps_incomplete`; model loading failures never fall back.
For original Korean segmentation, add `--korean-dictionary /absolute/korean_dict_jieba.dict`
when converting the aligner, using the pinned Qwen ASR implementation asset. Its
scores and hash are embedded in the package; runtime performs no file lookup.

Acquire source assets through Download Manager 2, then run the offline converter:

```sh
COMFYUI_PYTHON=/absolute/prepared/python python3 scripts/setup_worktree.py --asr
/absolute/prepared/python scripts/convert_asr.py \
  --source /absolute/complete/hf-snapshot \
  --destination /absolute/models/audio/transcription/native/whisper-large-v3.safetensors \
  --revision FULL_SOURCE_COMMIT
```

The source must include complete safetensors shards, config, tokenizer and
preprocessor files. Whisper also needs generation_config.json. Original Qwen
`thinker.*` checkpoints are supported; newer `-hf` Qwen bundles have another layout
and are rejected. Converted filenames expected by Cutlery are
`qwen3-asr-1.7b.safetensors`, `qwen3-forcedaligner-0.6b.safetensors`,
`whisper-large-v3.safetensors` and `whisper-large-v3-turbo.safetensors`.
Other existing Whisper picker choices require a corresponding converted package;
picker presence does not establish provisioned weights or verified quality.

Conversion constructs only meta networks to validate the complete key/shape
inventory, streams payloads without casting, embeds offline tokenizer/config
assets, verifies output payload hashes and publishes a companion manifest.
The omitted tied Whisper output head is copied from the embedding payload.
Original files and existing destinations are preserved. Reserve original plus
output disk space (approximately 4.4 GiB Qwen ASR, 1.7 GiB aligner, 2.9 GiB Whisper
large-v3 and 1.5 GiB turbo for the inspected checkpoints). Interrupted temporary
files are not valid packages; retry into a fresh destination after inspection.
ASR packages must be local in the selected model root; this change does not expand
the archive cache's supported categories. Loading reuses native tensor storage and
fast-disk support, and adds no separate residency or disk cache.

## Behaviour

Qwen accepts one nonempty AUDIO clip, converts to mono 16 kHz and keeps contiguous
low-amplitude chunk boundaries and offsets. Default 180-second chunks support
alignment; text-only graphs may choose up to 1200 seconds subject to token context.
Transcription and alignment are independent graph components. Unsupported languages
and invalid word times retain usable text and explicit quality warnings.

Whisper retains language/auto detection, transcribe/translate, beam size, initial
prompt, previous-text conditioning, VAD and chunk/batch controls. Generation uses
Transformers' token/long-form algorithms with native tensor networks. Word timing
uses a separate bounded teacher-forced pass, selected cross-attention heads and
DTW; decode beams never accumulate all-head attention outputs. Multiple inputs
use bounded batches with private state and original slot ordering. A zero chunk
length uses native long-form decoding at batch 1 or with previous-text conditioning;
larger independent batches use model-sized windows. Explicit lengths partition the input while
retaining offsets. Translation quality depends on the selected checkpoint.

## Verification and rollout

Run the prepared interpreter with
`-m pytest tests-unit/comfy_test/asr_test.py -q` for tiny CPU reference comparisons,
cached/beam decoding, long-form controls and chunk invariants. These tests do not
establish full-checkpoint GPU offload, speed or transcript quality.

Before default cutover, use the main ComfyUI queue for each full model and aligner.
Freeze a manually checked multilingual/silence/long-audio corpus; compare WER/CER
(maximum +1 percentage point) and p95 word-boundary error (maximum +100 ms).
Repeat cold/warm transcription → image/video → transcription trials; require
correct pressure reclamation/reload, no new failures and median total duration
within 5% of baseline or better. Test cancellation and failure cleanup. Record
source/model hashes, prompt IDs, timing and physical VRAM/RAM evidence separately.
All GPU inference expected to exceed 1 GiB must use the main instance queue.

Activate only with coordinated producer/native maintenance and preserved job
continuity. Source integration and CPU checks are not GPU acceptance. RouterAudio's
`ROUTER_AUDIO_ASR_WORKFLOW` can override the bundled Qwen workflow; update that
operator setting only at accepted cutover. Keep explicit legacy recipes for
rollback and preserve historical workflow snapshots; never replay uncertain jobs.

## Sources and attribution

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), Apache-2.0; processor reference
  c17a131fe028b2e428b6e80a33d30bb4fa57b8df and Transformers 5.22.2.
- [OpenAI Whisper](https://github.com/openai/whisper), MIT; network, decoding and
  selected-head word-alignment design, with Transformers 5.22.2 (Apache-2.0)
  generation/tokenizer/DTW helpers.
- Inspected Qwen ASR source: 7278e1e70fe206f11671096ffdd38061171dd6e5;
  aligner: c7cbfc2048c462b0d63a45797104fc9db3ad62b7.
- Whisper large-v3 source: 06f233fe06e710322aca913c1bc4249a0d71fce1;
  turbo: 41f01f3fe87f28c78e2fbf8b568835947dd65ed9.

Native attention is the existing ComfyUI selected backend on each component,
including supported native optimized kernels. CUDA workers, CT2 and vLLM cannot
supply ComfyUI layer residency ownership. Hardware-specific kernel performance
and the full migration remain unaccepted until the queued trials above pass.
