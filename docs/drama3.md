# Native Drama 3

Built with Fish Audio. This fork adds native local Drama 3 inference, voice
references, multi-speaker text, chunked long-form speech and whole-take revisions.
The current NVFP4 candidate failed dialogue/long-form quality acceptance; see the
[measured acceptance record](drama3-acceptance.md). It uses ComfyUI-managed generation and codec models; no Fish application, server,
model download or internet request runs inside a node. Existing Fish S2 workflows
and the separate custom-node package are unchanged.

## Checkpoints and preparation

The offline source is `fishaudio/drama-3` revision
`b2b5ee998034b2975720906c815e7fb0e9f007bd`; architecture/reference source is
[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech/tree/541566932abbaebcb0a40e562a1f846738383f38).
Use its eight safetensors shards, index, config, tokenizer assets, LICENSE and
`codec.pth`. Acquire these explicitly through Download Manager 2. Complete source
size is about 41.5 GB; allow another approximately 27 GB for output and temporary
conversion files. Actual artifact sizes are recorded in the conversion manifests.

Run the repository worktree setup command with a prepared Python that supplies
Torch, safetensors, tokenizers and the deployment's Comfy Kitchen. Conversion is
CPU-only, bounded to one expert/bank payload at a time, with no model inference:

```sh
python -m scripts.convert_drama3 --source /absolute/source/drama3 \
  --model-output /absolute/models/text_encoders/drama3_nvfp4.safetensors \
  --codec-output /absolute/models/vae/fish_dac_drama3.safetensors
```

The converter dequantizes each original FP8 expert, then quantizes it with native
NVFP4. This is lossy, including activation quantization during accelerated GEMM.
All other generator weights and the codec remain BF16. Codec weight normalization
is folded in FP32 before BF16 storage. Nonpersistent codec masks/rotary tables are
recomputed per call. Embedded tokenizer/config/license assets remove any runtime
dependency on the source directory. Native execution requires a Blackwell CUDA
GPU with the installed Comfy Kitchen NVFP4 kernels. Other devices are unverified;
the node does not change precision or choose another runtime automatically.

The converter first checks every pinned source file against published LFS SHA256
or Git blob hashes in `scripts/drama3-source.json`, then validates shard inventories,
tensor shapes, quantization scales and output bytes. Manifests record source/output hashes, tensor disposition and expert
quantization errors. Existing model outputs are refused. A completed codec may be
reused only after its manifest and source identity verify. Interrupted temporary
files start with a dot and are not loadable packages; preserve failure evidence,
remove only task-owned incomplete files, then retry to a fresh destination. Never
load a package missing its complete conversion manifest.

## Nodes and workflow

1. `Drama3Loader`: select generation weights from `text_encoders` and the matching
   Fish codec from `vae`. Paths are resolved through the existing category owner.
2. `Drama3EncodeReference`: connect one AUDIO clip, its transcript and a speaker
   ID 0–15. It returns CPU codec tokens and metadata. Connect zero or more of
   these references to `Drama3Generate` through its autogrowing reference input.
3. `Drama3Generate`: enter text with native `<|speaker:0|>` tags and optional
   delivery cues such as `[whisper]`. Controls are seed, temperature, top-p, top-k,
   maximum audio frames per chunk and single-chunk mode. Default sampling is
   0.7/0.7/30; frame limit defaults to 1024. Zero uses 600 frames in automatic
   chunks or 2048 in single-chunk mode, subject to the runtime context capacity.
4. Connect AUDIO to existing preview/save nodes. Output is mono 44.1 kHz. The current ComfyUI autogrow framework exposes up to
   100 reference sockets; this is an interface limit, not a model speaker limit.
5. Connect the returned `DRAMA3_HISTORY` to another generation to revise a take,
   or to `Drama3SaveHistory` to persist it. `Drama3LoadHistory` discovers saved
   `.drama3.safetensors` files in the admitted input/output directories.

A revision supplies the new user turn through the same upstream prompt format;
it can change wording or request a different delivery. It generates a new whole
take. It does not retain waveform samples outside a timed region. History is
explicit immutable data: ordered user text, generated code chunks, encoded voice
references, codec/tokenizer compatibility and generation provenance. It contains
no GPU caches or model weights. Saving is an explicit output node. Copying a saved
history to the input directory is sufficient to reuse it in another installation
with the same compatible package. Treat voice-reference histories as audio data.

User-supplied history is replayed in full or fails with context overflow; it is
never silently trimmed. Within one new long-form take, automatic continuity uses
the first and most recent generated chunks, following upstream. Earlier chunks
remain in the returned history. Further explicit references are appended to those
in history, allowing multiple clips for a speaker. Model and codec operations
must not mutate supplied history/reference tensors. A failed or canceled call
returns no updated history. Branching from a previous history remains supported.

Prompt rows include text, voice-reference frames and earlier takes. The upstream
prompt builder allows 8192 rows; generated frames also fit the runtime capacity.
A larger output allowance is bounded to remaining capacity. Long text is split
near 300 UTF-8 bytes and three speakers per chunk, preserving protected tags.
Do not confuse the chunking recommendation with a global three-speaker limit.
Saved take metadata records each chunk’s EOS/length stop reason. Generation caches are scoped to the current chunk; there are no hidden sessions,
persistent CUDA graphs, device/keep-loaded node controls or separate GPU workers.

Queue API prompts normally and retain their prompt IDs. After uncertain submission,
inspect the existing queue/history before retrying. Cancel only the correlated
prompt. OOM, corrupted packages and incompatible history remain visible failures;
do not clear other jobs or retry under a different precision automatically.

## Verification and acceptance

```sh
OMP_NUM_THREADS=2 DRAMA3_UPSTREAM=/absolute/pinned/fish-speech \
  python -m pytest tests-unit/comfy_test/drama3_test.py -q
python -m scripts.check_drama3_codec --upstream /absolute/pinned/fish-speech \
  --source /absolute/source/codec.pth --output /absolute/new-evidence-directory
```

The optional upstream test source must be the pinned revision above. Tests cover
independent upstream attention, recurrent prefill/decode, MoE, tower alignment and
sampling; native NVFP4 packaging/loading; history round trips and containment.
The codec harness builds upstream on meta, supplies only bounded nonpersistent
fixture tables, folds its source weights, and compares real-checkpoint encode and
decode on CPU. It does not certify CUDA speed or long-form resource requirements.
FP32 codec math is compared after the same BF16 weight packaging to distinguish
structural errors from compute rounding. BF16 reference differences are retained
separately, not presented as exact code parity.

Source tests, successful conversion, native checkpoint loading, audible speech,
5090 lifecycle acceptance and serving activation are separate gates. See
[acceptance record](drama3-acceptance.md) for actual results and pending gates.
No realtime guarantee, automatic service restart or serving activation is implied.

## License

The [Fish Audio Community License](../comfy/fish_audio/LICENSE) applies to the
adapted Fish code and weights. Retain NOTICE and upstream attribution with any
packaged weights. The codec also retains Descript DAC's MIT notice. Commercial
and hosted-service conditions are in the actual pinned license. This integration
adds no telemetry or licensing network checks.

## Editable examples and acceptance commands

The installed examples live under `user/default/workflows/Drama 3/`: Speech,
Dialogue, Revision, and Reload. Revision chains a first take into a changed-word
second take and saves the new audio/history. The installed Reload example selects the retained plain history; fresh templates
require selecting a real history saved by an earlier execution. API templates have the `.api.json` suffix;
they are submitted to `/prompt`, not opened as editable graphs. Recreate these
files only into an empty destination with
`python -m scripts.create_drama3_workflows --library /absolute/workflow/library`.

The isolated candidate wrapper embeds `main.start_comfyui`; ordinary server,
model-root and output-path arguments can be passed to `scripts/check_drama3_server.py`.
Set GPU visibility before launch rather than relying on late allocator options; set `DRAMA3_EVIDENCE_DIRECTORY` to a private
local evidence directory. Its test-only execution wrapper records elapsed time,
process peak RSS, Torch allocated/reserved peaks and driver-used memory per prompt.
Acquire the installation's coordinated GPU admission first, release only that
hold after all candidate processes exit, and preserve serving work. This wrapper
is not a replacement service entrypoint or product telemetry.

`python -m scripts.check_drama3 --output /absolute/new-evidence --case plain`
submits one case to the isolated local port 8199 and follows its prompt ID.
Available cases include warm, clone, dialogue, long, multilingual, revision,
delivery and reload (`--history-name` must name a real admitted saved artifact).
Cold/warm case seeds differ so caching cannot skip the generation being measured.
Use a fresh case evidence directory for each invocation; uncertain submissions
are recovered through the saved submission ID, never automatically replayed.

`scripts/drama3_reference.py --source /absolute/source --upstream
/absolute/pinned/fish-speech --text "Hello" --output /absolute/new-reference`
runs the pinned original FP8 backbone/tower with one layer's weights resident on
CUDA beyond a bounded resident prefix (`--resident-layers`, default 24). It retains upstream math, prompt construction and sampling and
saves original VQ codes; no full BF16 model is materialized. Optional
`--host-cache-layers` retains that many tail layers as original CPU FP8 weights
to reduce repeated disk reads; a RAM preflight refuses a cache that leaves less than the workspace reserve in available
memory. The successful local plain baseline used
`--resident-layers 26 --host-cache-layers 14`. Validate its tiny
component fixtures and layer residency behavior before using it as an acceptance
baseline. The reference is slow and requires coordinated exclusive GPU admission.
Decode its codes with the independently checked codec for listening comparisons.

Use `python -m scripts.check_drama3_codec --upstream /absolute/pinned/fish-speech \
  --source /absolute/source/codec.pth --codes /absolute/reference/codes.safetensors \
  --output /absolute/new-waveform-evidence` to render source and native FP32 codec
waveforms from reference codes. `python -m scripts.check_drama3_quant --source
/absolute/source --model /absolute/converted/model.safetensors --upstream
/absolute/pinned/fish-speech --output /absolute/new-component-report.json` measures
real first-layer normalization/DeltaNet port error and separate NVFP4 expert
weight/activation errors. That bounded GPU check also requires exclusive admission;
it does not replace whole-speech listening or full-pipeline acceptance.
