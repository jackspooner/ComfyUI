# Local run timing observations

The server queue adapter records accepted/start/finish facts in
`user/run-timing.sqlite`, using monotonic durations and immutable prompt identities.
It covers every native prompt, including direct tools and staged execution, without
changing queue ordering, node execution, cancellation or maintenance policy.
GET `/timing/observations?after=0&limit=100` returns version-1 timing observations,
a durable cursor and an ordered active snapshot. Limit is 1–500. Follow nextCursor
until null. Pending order follows the queue owner's full tuple comparison,
including prompt-ID tie breaking for equal priorities, without changing the heap.
Expired cursors are explicitly marked. The endpoint accepts machine-local
non-browser requests only and performs no outbound requests. SSH loopback tunnels
can expose a separately configured host to the owning MediaTools collector.

The record contains hashed artifact identities, graph structure and allowlisted
numeric/settings controls, never prompts, media paths, seeds or graph documents.
Unknown residency/cache state stays unknown. Native time includes model loading,
encoding and output-node work but excludes time waiting in the native queue.
With matching Cutlery integration, GPU-owner hold waits also pause execution
accounting. They contribute to resource queue time and numeric
`observed.gpuHoldWaitSeconds`; `gpu_hold_active` marks an unresolved release time.
Telemetry failure degrades coverage and logs once per outage; inference continues.
Terminal timing data expires after 90 days, independently of job history. A restart
marks previous in-flight observations unknown and never resubmits work.

Known native loader selections include hashed registered file identity metadata
(device/inode/size/mtime/ctime); this is distinct from a content-verified owner
model-binding fingerprint. Unsupported implicit loaders retain selection-only
assurance. Repeated graph instances share a profile topology while numeric node
counts retain workload size. Node implementation file identity also separates
profiles after custom-node code changes, without retaining local source paths.
Text generation contributes numeric token counts and output limits without
storing generated text. Cutlery may describe immutable stage
expansions through an optional read-only resolver; missing bounded selections stay
unavailable until the execution owner saves them. Read-time resolution writes no
telemetry or execution state.

MediaTools owns duration learning and queue forecasts. These observations do not
prove queue completeness across other hosts, predict arbitrary future submissions,
or authorize service maintenance. Validate with the CPU run_timing_test.py suite;
live behavior requires activation of this commit on the serving instance.

## Expanded workload and stage evidence

Optional `workload.evidence` version 2 separates requested settings, resolved facts
and completed measurements. Facts carry source, observation time and availability;
full implementation provenance is separate from explicit performance compatibility.
This adapter does not declare compatibility across changed implementations.
Native implementation identity includes the core revision, Torch version and
selected node-module content hashes. File paths, timestamps and inodes only
invalidate the bounded hash cache; reinstalling identical source at a new path
does not split history. Unreviewed core/runtime changes remain separate even
when the selected node module is unchanged.

`app/run_timing_workload.py` uses explicit native node adapters. It resolves scalar
primitive links and native empty image/LTX/Hunyuan latent shapes, preserving each
sampling pass instead of adding unrelated widths/steps. Ordered native LoRA entries
retain strengths, bypass and application mode; ControlNet intervals remain settings,
not an invented count of active model evaluations. Selection identities are hashed.
Unknown custom nodes remain visible with `unsupported_workload_adapter`.

Native stage model identities include only weight selections reachable through
that stage's dependencies. Implementation identities hash the same dependency
structure, input/output edges and reachable node-module content, retaining the
core/Torch revision and any declared composite-owner implementation revision.
They exclude unrelated and downstream branches; whole-run model and implementation
provenance remains unchanged. No performance compatibility is inferred. The
`dependency-stage-1` identity namespace separates these revisions from older
whole-graph stage identities without broadening an extractor's workload coverage.
Extraction shares node-kind source hashes and propagates model selections in
dependency order; it does not rescan weight files for each stage.
Modality profiles use the stage's explicit adapter and reachable dependencies,
not previously visited unrelated branches. An execution can have an unknown
whole-run profile because it mixes image/video/text work while individual stages
retain their known profiles. Unsupported nodes and unresolved workloads still
retain their existing completeness reasons.

Optional `stages` preserve graph dependencies and the physical resource. The existing
progress registry supplies start/finish offsets and monotonic elapsed durations.
Cached nodes report reuse and zero executed duration. Overlapping stage durations
must not be added to replace whole-run elapsed time. Node spans can contain dynamic
dependencies; completion without a matching observed start is not measured work.
Progress writes are bounded to one per node per five seconds plus transitions.
Shared registry percentages can mix loading and sampling, so they remain unreliable
`node_units` and cannot justify a remaining-time estimate.

Native scalar hooks count noise evaluations, model forward calls/elements and
conditional/unconditional branch work. The generic native CFGGuider boundary,
including standard and custom samplers, observes the
actual latent shape after native correction, model dtype, dynamic/standard memory
strategy and load/offload devices. Reviewed native Euler, Euler ancestral (including
its native flow-matching dispatch), non-CFG++ res-multistep and ER-SDE paths plan
one noise evaluation per schedule interval. Their identities and effective scalar
options are execution-start facts; defaults come from the native function signature.
Euler's unbounded churn cutoff is represented as `positive_infinity`, keeping JSON
finite. `SamplerEulerAncestral` preparation retains `eta` and `s_noise`; actual
sampling options still come from the resolved native sampler. Explicit custom noise
samplers/scalers, nonempty specialized inpaint options, unsupported options,
custom sampler subclasses/wrappers and unreviewed multi-evaluation/adaptive/CFG++
paths retain unknown advance evaluations. No schedule values are read or transferred
to plan the count. Unsupported samplers,
guiders that override native prediction and CFG hooks retain coverage reasons.
Positive-only BasicGuider inherits the verified native CFG path and reports one
planned branch. Joint AV streams retain
separate axis shapes and remain incomplete for broadened workload comparison.
Repeated sampling within one node remains incomplete. Counts are CPU observations,
not synchronized GPU kernel timings. GPU worker-thread observations without a
native executing context remain unattributed. No device synchronization is added.

The native sampling callback provides separate reliable `sampling_steps`, preserving
the original preview callback and its result. Bounded progress curves retain up to
128 monotonic stage offsets at the five-second cadence for empirical residuals.
These are measured steps, not a linear extrapolation. Sampling measurements remain
separate from submission/start facts; completion-only values must not define an
in-flight target's workload population. Telemetry construction/observer errors
must leave inference and callbacks unchanged.

Native sampler plans require the model-manager loading observation before workload
coverage is complete. The requirement is committed in the sampling-plan snapshot,
so geometry alone cannot briefly qualify an unobserved load. `managedLoading.*`
describes up to 16 ordered manager calls with 64 entries each: registry-entry reuse,
dynamic/standard strategy, load device, clone detaches, force flags, VRAM policy,
the effective weight-load budget and the manager's already-computed loaded-byte
counter where available. Unknown counters stay null with an explicit known flag;
registry reuse does not prove warm physical GPU residency. Entries are ordinals
within the owning stage, not inferred weight-file identities.

Effective budgets and observed manager state remain strict comparison facts;
zero retains the native full-load sentinel and negative budgets retain native
reclaim semantics. Neither is a measured transfer-byte count. There is no broadened matching
or scaling across these memory decisions. Initial inference requirements and
weight requirements before pressure, post-load RAM/VRAM usage, success/failure
counts and load/unload/clone-detach CPU elapsed are diagnostic measurements, never
population inputs. Indexed and aggregate times can overlap; do not add them to
replace the owner's end-to-end duration. These are measurements inside the existing
node stage, with no added queue work or scheduling authority.

Plans remain incomplete until entry selection/clone/allowance evidence is present.
Malformed, repeated, over-limit or uncorrelated metadata retains explicit gaps.
Writes are coalesced at plan/readiness/batch-usage boundaries; temporary operation
bindings never enter durable records or retain models. Hooks use values the manager
already computes and add no device queries, synchronization or transfers. Dynamic
materialization details and physical residency/transfer/kernel timings remain
unknown where unsupported. Verify with the loading suites below; tiny CPU stubs and
call-trace parity do not establish actual GPU or deployed accuracy acceptance.

Expanded evidence explicitly reports missing processed references, implicit
dimensions and unsupported nodes. Standard LoRA loaders report actual file-cache
reuse, file-load time and CPU patch-preparation time separately. Native parsed
LoRAAdapter tensors provide min/max rank and supported-rank coverage without reading
tensor values. These live component facts are separate from requested strengths
and zero-strength bypass; completed durations never become submission/start facts.
Patch preparation is registration, not later GPU weight materialization or kernel
time. Other adapter formats may lack ranks. Measured GPU transfers/residency are
not yet provided by this adapter.
Standard `CLIPVisionLoader` and `CLIPVisionEncode` have explicit loading/encoding
adapters. The loader resolves `clip_name` through the native `clip_vision` category,
not the text encoder directory. The encoder observes source BHWC and actual
processed BCHW shapes, crop selection, native model type, dtype, memory strategy, load/offload
devices and the one-forward plan. CLIP/SigLIP/SigLIP2/DINO preprocessing retains its
ordinary implementation; flexible SigLIP2 resolutions are observed after resizing,
not guessed from requested size. Source facts are start evidence, processed facts
are live evidence, and encoded batch/positions/width remain completion evidence.
Preprocessing and forward-call durations use CPU monotonic boundaries without GPU
synchronization. These are not GPU kernel time, transfer or residency measurements.
Repeated calls, mismatched/malformed plans, missing successful forward/output
evidence and unknown cached input shapes remain explicit gaps. A failed forward
preserves its error and known input workload; its failed stage is not a successful
history sample. No feature-cache reuse is invented for this encoder.
An expanded envelope is not a claim of complete coverage. The collector/forecast
consumer must support this additive contract before producer activation; older
strict readers may reject the extra fields. No source change alone activates it.
Focused CPU verification: `python -m pytest tests-unit/app_test/run_timing_test.py
tests-unit/app_test/run_timing_workload_test.py tests-unit/comfy_test/execution_metrics_test.py
tests-unit/app_test/run_timing_sampler_test.py tests-unit/comfy_test/sampling_execution_metrics_test.py
tests-unit/app_test/run_timing_h3_layout_test.py tests-unit/comfy_test/h3_layout_execution_metrics_test.py
tests-unit/app_test/run_timing_model_loading_test.py tests-unit/comfy_test/model_loading_execution_metrics_test.py
tests-unit/app_test/run_timing_h3_references_test.py tests-unit/comfy_test/h3_reference_execution_metrics_test.py
tests-unit/app_test/run_timing_h3_conditioning_test.py tests-unit/comfy_test/h3_conditioning_execution_metrics_test.py
tests-unit/comfy_test/lora_execution_metrics_test.py
tests-unit/comfy_test/reference_execution_metrics_test.py
tests-unit/comfy_test/clip_vision_execution_metrics_test.py
tests-unit/comfy_test/control_execution_metrics_test.py
tests-unit/comfy_test/text_execution_metrics_test.py
tests-unit/app_test/run_timing_audio_test.py
tests-unit/comfy_test/audio_execution_metrics_test.py
tests-unit/app_test/run_timing_av_test.py
tests-unit/comfy_test/av_preparation_execution_metrics_test.py
tests-unit/app_test/run_timing_h3_allocation_test.py
tests-unit/comfy_test/h3_allocation_execution_metrics_test.py
tests-unit/comfy_test/vision_features_test.py -q`
using the prepared interpreter. Fixtures use tiny CPU tensors and mocked sampling;
they do not establish real-model or performance acceptance.
For fixture runs while the serving GPU is occupied, select CPU before importing
ComfyUI: invoke pytest through `python -c 'import comfy.cli_args;
comfy.cli_args.args.cpu=True; import pytest; raise SystemExit(pytest.main([...]))'`,
using the listed test paths in place of `...`. This avoids ComfyUI's import-time
GPU probe and does not change the serving process.

Resolved node-input observers attribute processed VAE/control image and mask
dimensions, batch sizes and encoded conditioning-token shapes without retaining
tensor contents or source paths. Only explicit adapters observe arguments; unknown
nodes do not acquire workload coverage from input names.

`MiniMaxH3ReferenceToVideo` has an explicit preparation adapter. Bounded native
`h3References.*` facts record up to 32 ordered nonempty image/video/audio inputs,
actual source resize dimensions/precision/device, resized-frame work, selected
visual frames and tokenizer frames, and visual/audio encoder selection. Video
soundtracks retain their native same-numbered socket relationship. Stills resize
only their first RGB item, so unused source batch/channel dimensions are audit
measurements. Videos resize all incoming frames before trimming to the target
length and H3's 17k+5 grid; those incoming resize frames remain strict workload
facts even when the final reference latent has the same shape. Tokenizer video
selection is observed after the native 2 FPS selection, without reading pixels.
Audio encoding records the consumed first batch's source samples, channels and
sample rate; unconsumed audio metadata remains audit-only. Source graph slots and
output indices remain explicit reference-component provenance.

The initial required marker prevents preparation coverage before native evidence.
The final prepared snapshot coalesces per-reference events, without per-frame SQL
writes. Missing, malformed, repeated, uncorrelated or over-limit metadata retains
coverage reasons instead of silently truncating. Output latent shapes and CPU
monotonic resize/visual-encode/audio-encode/tokenize/text-encode spans are
measurements, excluded from matching and queue-stage summation. Observer/probe
failures cannot fail or replay native work. No new model/device queries, tensor
value reads, GPU synchronization or retained tensor references are added.

Native `MiniMaxQwen3VL.forward` additionally records its full-sequence conditioning
plan after ordinary SDClip embedding expansion and before forwarding. Live
`h3Conditioning.*` facts retain batch/sequence/embedding width, text/image/audio/
padding positions, input dtype/device, actual configured compute dtype and layer
count, intermediate-output selection, output dtype and normalization. Token IDs,
prompt contents and embedding values are never collected. Native embedding-row
metadata must agree with the actual input shape and contain no unknown modality.
The exact native class is admitted; custom subclasses (including the distinct
p-h3-image connector path), extra options, masked calls, unknown precision,
unmatched/repeated plans and more than 64 selected layer indices retain gaps.
This is full-sequence conditioning, not autoregressive decode or a prefix-cache
claim. Input embeddings are normally float32 even with lower-precision model
compute; those two observations remain separate.

The forward plan atomically requires managed-loading evidence, so resolved token
work cannot qualify a missing loading plan. A complete reference plan proving no
visual/audio VAE calls were selected can clear
`h3_reference_encoding_work_unmeasured`. A valid native text plan can clear
`h3_conditioning_text_work_unmeasured`; nonzero image/audio positions retain
`h3_conditioning_vision_work_unmeasured` because their internal preprocessing and
feature-cache work is not established here. Selected VAE internals, tiling/
resampling, unknown custom nodes and cached stages without input proof remain
incomplete. Packed sampler tokens cannot clear these gaps or establish RefMod
extraction/file/cache coverage.

The CPU-monotonic `h3Conditioning.forward.*` measurements preserve native success
or failure without retries. They overlap `h3References.text_encode.*` and the
owner node's elapsed time; never sum them or add queue stages. First plans and
coverage transitions coalesce writes. Tiny CPU fixtures exercise native tokenizer
presentation, embedding expansion, token-weight assembly and H3 forwarding with
stub tokens/weights. Five equivalent text-only/no-VAE histories qualify exactly;
changed token work/compute precision/layers/selection and absent loading/input
evidence do not. Source delivery, configured Main/SecondPC activation and real
model/accuracy acceptance remain separate.

Native reference hooks also observe loaded-image dimensions, VAE encode shapes after native cropping and
temporal truncation, conditioning output positions, vision-forward attempts and
verified feature-cache hits. Each VAE invocation retains its own axis shapes;
positions are conditioning output lengths, not tokenizer input counts. Standard
2D tiled VAE encoding counts actual attempted and completed forwards across its
existing three passes; OOM fallback is separate. An OOM fallback changes the
observed tiled regime as a live fact, without backdating
it into the initial execution-start snapshot. Internal model tilers and 1D/3D
tilers report incomplete tile counting. Counts accumulate in memory until existing
stage snapshots, without per-tile database writes or device synchronization.
These observations do not remove broad reference-coverage reasons or establish
complete role/crop/token attribution, decode coverage or tile remaining progress.
Optional progress handlers
remain compatible with older implementations, and observer failures cannot stop
node execution.

Read-only composite resolvers may return an internal `stageGraph` and optional
`stageMetadata` shared-operation IDs. Only extracted evidence is persisted. The
optional `loaderAliases` maps explicitly audited wrappers to native loading-node
adapters; non-loader targets are ignored. This declaration belongs to the resolver,
is not a heuristic based on names, and does not replace implementation provenance.
The description uses native expansion IDs so later progress can correlate. A bounded
selection saved by the owner is retried at the first nested start event, with
execution-start availability. Read/completion-only resolution carries live/completion
availability, never backdated submission facts. `comfy.execution_metrics.stage_scope` is an optional
telemetry-only context for existing owner-controlled nested node calls; it preserves
the executing node context and scheduler, observes start/finish/failure and verified
reuse, and routes scalar sampling observations to that described substage.
Missing descriptions remain unavailable. Receipt presence alone cannot prove reuse.
Terminal storage failures clear only the attempt's in-memory timing state.
Composite owners may declare a bounded `implementationRevision`, a supported
`workloadProfile`, and `stageMetadata.kind` for an opaque native boundary. Owner
implementation identity enters the strict profile revision, without asserting
performance compatibility. A declared kind does not remove unsupported-workload
coverage; native-specific profiles are preserved. Invalid profile/implementation
declarations remain explicit coverage failures. This internal description contract
does not change the public timing wire schema or execute an owner graph.

Native ControlNet and T2I adapters report actual interval skips, processed hint
axis shapes, preparation counts, forward attempts and active calls. A temporary
noise-evaluation scope deduplicates repeated CFG/batch calls for each control;
these are measured evaluations, not percentages converted into step counts.
T2I feature encodes and actual feature reuse remain distinct from active calls.
Copied device controls preserve one application binding; a new `set_cond_hint`
starts a new binding. The timing owner maps bindings to described application
stages and selected native loader identities. Runtime handles and evaluation
tokens are never persisted. Counters write only at existing stage snapshots.
Zero-strength native application is a declared bypass. Cached applications without
observed bindings, calls without native evaluation context, varying hint shapes
and uninstrumented adapters remain incomplete. Original interval checks and
sampling results are preserved; no extra timestep reads or GPU synchronization
are introduced. Active measurements stay live/completion evidence and do not
become submission/start facts or erase broad control-coverage reasons.

`BaseGenerate.generate` reports the actual embedding input positions, batch, resolved
output allowance, execution precision and fresh per-generation KV capacity. It does
not claim cross-request prefix reuse. Prefill and decode forward counts and CPU
monotonic boundary durations are separate; asynchronous device work means these
durations are not synchronized GPU execution timings. Generated token counts reuse
the native Python token list after the existing token read, without another device
read. `generated_tokens` progress has a null total: the output allowance is a ceiling,
not a known final length. Durable progress remains bounded to five seconds plus
stage transitions. Standard `SDClipModel.process_tokens` supplies per-row expanded
text/image/audio, unknown-embedding and padding position counts using existing CPU
metadata. Counts qualify only when their rows/positions match the actual generation
input. Unrecognized embeddings, overridden preparation without these observations,
other native loops and speculative decoding remain incomplete; multiple invocations
stay incomplete. These modality counts do not establish complete reference roles
or video-frame attribution. Text progress is
observable but does not yet refine MediaTools empirical residuals. No sampler,
stopping rule, output token, constraint or failure/retry behavior is changed.
Run `text_generation_length_test.py` and `text_json_native_test.py` alongside the
focused text timing checks to validate EOS, automatic budgets and constraints.

Extractor `comfy-explicit-3` adds explicit native `LoadAudio`, `EmptyLatentAudio`,
`VAEEncodeAudio`, `VAEDecodeAudio[Tiled]` and ordinary audio save/preview adapters.
Only existing tensor shapes and integer rates are observed: batch, waveform
channels, per-item sample count and seconds stay separate. Filenames and waveform
values are excluded. `EmptyLatentAudio` uses native even rounding at the 44.1 kHz /
2048 latent ratio; that describes latent allocation, not a promised decoded length.
Linked seconds are resolved through supported primitive nodes. Audio sampling
identified from native audio-latent ancestry records ordinary 3D temporal latent
length. Exact native `LTXAV`/`MiniMaxH3AV` latent-format types declare supported 4D
layouts: LTX `[batch, features, time, frequency]` (`bctf`) and H3
`[batch, features, stereo, time]` (`bcat`). Frequency bins never become audio
channels. Unknown formats/custom subclasses retain raw axes and an unresolved
layout reason. The sampler never
fabricates spatial/video axes from an audio tensor. Unsupported shapes remain
incomplete. An audio
reference loader does not change a text-generation operation's main profile.

`LTXVAudioVAELoader`, `LTXAVTextEncoderLoader`, `LTXVAudioVAEEncode`,
`LTXVAudioVAEDecode` and `LTXVEmptyLatentAudio` have explicit adapters. The
empty-latent node observes the VAE's actual configured allocation; it does not
guess latent rate/frequency from frame requests. Requested frames/rate/batch remain
audit intent, so equal effective allocations need not split sampling history.
Audio input roles preserve graph source-stage relationships and processed work.
LTX encoding also requires internal resampled waveform, FFT/hop/mel settings,
expanded channels and actual mel-input dimensions. Observing only the generic
outer VAE wrapper cannot establish that coverage. LTX decoding requires its native
latent/target-shape plan and verifies the resulting vocoder input before a
successful span qualifies. Encoder/decoder/vocoder CPU monotonic boundaries do not
synchronize GPU kernels. Latent and vocoder output shapes stay completion evidence,
including when an intermediate snapshot is written before node completion.
Missing/mismatched plans and repeated invocations retain gaps; newly detected
audio coverage changes are saved once without per-forward journal writes.
Run `tests-unit/comfy_test/ltx_audio_execution_metrics_test.py` with the audio
collector, native audio and sampling metrics suites. CPU fixtures use real tiny
mel processing with stub model forwards; they do not establish model accuracy,
tiled internals or live deployment acceptance.

For exact native LTX/H3 formats, the sampler declares ordered `video_audio`
streams. The collector requires exactly one BCTHW video stream and one declared
BCTF/BCAT audio stream with equal positive batches. Video dimensions remain the
ordinary sampling fields; `jointAV.audio.*` retains audio batch, feature channels,
time and frequency/stereo axes separately. These are execution-start workload
facts, never decoded outcomes. Native joint evidence sets that sampling stage's
video profile; unrelated image/text stages still make whole-run modality unknown.
MediaTools policy 6 permits only exact joint matches, without broadened diffusion
bounds or stream scaling. Unknown format/order, reversed/extra streams, malformed
axes, unequal batches, custom guider behavior and repeated calls keep gaps.

Native H3 sampling additionally requires the actual `PackedLayout` plan. Until
packing is observed, `h3_packed_layout_unmeasured` keeps the stage incomplete.
Live `h3Layout.*` facts retain the padded target signature, total token count and
ordered per-segment role/token counts: text, visual/audio guides, visual/audio
references and target audio/video. This includes the native reference blocks used
by RefMods, without inspecting tensor values or filenames. Only the exact native
layout class qualifies; truncated metadata, inconsistent totals/target shapes,
custom subclasses and differing layouts within one stage remain incomplete.
Identical rebuilds count as live measurements with no per-rebuild persistence;
the first plan and later invalidation are committed at transition boundaries.
Saved pre-layout forecasts remain unknown. Packed rows do not establish upstream
reference decoding/encoding, RefMod extraction, file identity or cache reuse, and
cannot clear unrelated dependency reasons. No model execution or GPU synchronization
is added. Run the H3 layout CPU suites with the launcher above; real inference and
deployment acceptance remain separate.

`LTXVConcatAVLatent` and `LTXVSeparateAVLatent` have explicit preparation adapters.
Graph components retain video/audio roles, source stage and output slot, including
the separate node's audio output. Native plans observe raw video/audio/mask shapes,
input precision/device and an existing joint stream's target audio shape. Fitting
records its actual trim/pad axis and CPU monotonic boundary; missing masks generated
by the node remain part of its observed work. Completion-only output shapes,
precisions and devices verify the plan without becoming input matching facts.
No tensor values, GPU synchronization, model execution or new cache is added.
Malformed/extra streams, unequal batches, conflicting outputs, repeated fitting/
invocations and failed probes keep explicit gaps, saved on coverage transitions.
Cached preparation without input-shape proof remains incomplete; a zero-duration
reuse marker does not invent that proof. Unknown custom preparation dependencies
still prevent stage populations from qualifying. Policy 6 consumes these facts
through its existing exact dependency comparison. Source/preparation work remains
exact there, so different completed fitting inputs may still split equal final
sampling shapes; no broader equivalence or accuracy claim is implied.

`EmptyMiniMaxH3LatentAV` has an explicit native allocation adapter. Requested
width/height/length stay audit-only; the native temporal plan supplies its computed
frame count, video tokens, audio time length and rates. The allocation plan records
the two exact zero-allocation shapes before execution, preserving the original
device-selection order and default dtype. CPU monotonic allocation timing does
not synchronize GPU kernels. Actual dtype/device become live execution facts;
output shapes remain completion-only verification, not a substitute for the plan.
Repeated/missing/mismatched plans, failed allocations, output probe failures and
cached allocations without shape proof retain gaps. Shared allocator observations
inside other H3 nodes cannot complete their unknown conditioning/reference work.
Requests that round to the same effective grid/canvas can share exact history under
the existing policy 6; changed effective work remains separate. CPU fixtures check
native rounding, allocation/call parity, both failure positions and propagation
to independent stage forecasts. They do not establish model inference or accuracy.

Native audio VAE encode hooks observe actual post-resample B/C/sample dimensions
as live facts, preserving original-rate input metadata separately. H3's reference
encoder observes its actual first batch item, not all supplied waveform items.
Decode records the selected latent stream and existing normalization/tile options;
returned waveform dimensions/rate/duration are completion measurements and never
resolved input features. The existing sample-rate override and normalization
behavior is unchanged. Loading source dimensions are live facts available only
after that existing file decode; no header preflight or extra file read is added.
Ordinary save/preview stages observe their already resolved waveform input.

The collector clears only the corresponding measured audio-coverage reasons.
Native VAE decode memory fallback records a separate live tiling-regime change
without backdating it into the initial decode options or asserting tile counts.
Unknown/malformed waveform metadata, actual tile forward work, repeated audio
invocations, unsupported synthesis/transcription and custom suffix-decode paths
retain explicit gaps. Audio facts are bounded to 16 invocations per operation;
durable audio-boundary snapshots occur once per event type and stage. Output
durations remain per invocation rather than being summed across unrelated audio.
There is no broader audio matching, linear scaling, added model execution or GPU
synchronization. CPU fixture parity is separate from deployed real-model acceptance.

Extractor `comfy-explicit-6` keeps input-workload completeness distinct from known
measurement gaps. `audio_output_unmeasured`, `audio_output_metadata_unavailable`
and `stage_completion_unobserved` remain in `stage.reasons`, while being excluded
from `stage.workload.reasons` and whole-workload completeness. A missing end span
keeps that stage's phase/duration unknown; it does not invalidate separately known
input work or the owner's successful end-to-end duration. Stage residuals still
need actual successful spans, so a missing span may require whole-run fallback.
Unobserved execution, unsupported paths, missing inputs, tile/layout gaps and
multiple invocations are not reclassified. This change does not make future
decode/publication input work known or provide a whole-run ETA for that missing
work. Sampling and shared text-loop plans also expose their actually observed
`executionPrecision` on the stage; precision is not inferred for other operations.

Unresolved selected model/LoRA/control identities and unmeasured LoadImage reference
processing also mark their owning stage's workload incomplete. These gaps cannot
be dropped by a consumer selecting a stage dependency subgraph. Loaded-image shape
observations alone do not clear reference-processing coverage or establish crop,
tile and role attribution. Extractor 7 can resolve supported image-input work as
described below. Older extractor records retain their original evidence.

## Live transition delivery

Optional loopback `GET /timing/observations/stream` uses SSE `data` JSON conforming
to RouterContracts `TimingObservationStreamEvent`. The same local-only guard as
cursor reads rejects browser/forwarded/remote requests. The first `ready` frame
identifies source, process instance, resource and the current durable journal
cursor. Only later committed snapshots are streamed; reconnection establishes a
new baseline and never replays old events as live forecasts. Recover durable facts
through ordinary cursor reads, preserving any expired-cursor or live-delivery gaps.

Existing admission, start, stage/input/cache transitions and throttled progress
writes publish immutable snapshots after their SQLite transaction commits. Failed
commits emit nothing. Current queue/execution elapsed and GPU-hold wait exclusion
are saved from the existing monotonic owner clocks, rather than relying on a later
GET to refresh them. Interrupted records retain their last observed elapsed span
with `phase=unknown`, restart reasons and no invented finish/end duration.

At most four subscribers retain eight frames each; one JSON event is limited to
1 MiB. Pending event-loop notifications are coalesced as well as frame buffers.
Overflow/oversize emits a terminal `gap` (`timing_stream_overflow` or
`timing_stream_event_too_large`) instead of blocking inference or accumulating
unbounded work. Subscriber exhaustion returns 429 `timing_stream_limit`.
Idle connections receive 15-second heartbeat comments; a blocked network write
times out after five seconds. Observer/transport failures cannot fail or replay
execution. No model/device queries, synchronization or outbound connections are
added. No stream frame certifies a complete queue order or authorizes maintenance.

MediaTools still owns forecasts; native streaming alone does not activate direct
capture or prove accuracy. Run `tests-unit/app_test/run_timing_stream_test.py`
with the CPU launcher above, plus existing timing/queue/restart regressions. The
tests use temporary databases, fake clocks and loopback fixture servers.

## Native image input and standard 2D VAE work

Extractor `comfy-explicit-7` explicitly describes `LoadImage`, `LoadImageOutput`
and `LoadImageMask`. Native decoder version-1 summaries retain constant-size
counts of decoded and processed video frames/pixels, source geometry/format/bit
depth/codec, conversion, alignment padding, rotation/crop/resize, alpha frames
and decoded/resampled/retained audio samples. Audio decode work remains visible
even when an image loader discards the returned audio. No frame values, paths,
per-frame durable writes, device queries or GPU synchronization are added.

Complete full-decoder work, matching output/alpha/default-mask shapes and the
declared native pipeline version can clear that loader's processing gap. Animated
WebP's PIL fallback additionally requires its own consistent source modes,
geometry and decoded/selected-frame counts. Mask loading requires an observed
channel/mode and matching selection shape. Mixed geometry/formats, decode errors,
missing/legacy metadata, partial extraction and repeated calls stay incomplete.
The first repeated-work boundary invalidates previously complete live evidence;
later repeats do not cause per-call writes. Whole-stage verified reuse clears
only its source-work gap, marks `sourceReused` and retains zero executed duration
without inventing a decode measurement.

Explicit VAE/control/MoGe reference components retain their graph `sourceStage`
and processed input shapes. Standard 2D VAE encoding reports one actual native
input/plan: post-crop shape, precision/devices, selected batch size and forward
count, or the existing three-pass tiler's sizes/overlap and native step count.
Only a matching complete input/plan clears `image_encode_workload_unmeasured`;
successful completion also requires matching attempted/completed forwards.
Internal/1D/3D tilers, OOM fallback, missing plans and repeated calls retain gaps.
This does not establish complete conditioning/reference-role coverage for every
composite workflow, model identity, residency or a whole-run completion estimate.

Source decode, output preparation and channel-clone durations are CPU-monotonic
measurements, separate from workload matching. A missing preparation duration is
a measurement-only reason; it does not erase separately observed input work.
First output/mask/plan boundaries use existing correlated native snapshots.
Source delivery and CPU parity do not establish activation, GPU throughput or
forecast accuracy. Run `tests-unit/app_test/run_timing_image_test.py`,
`tests-unit/comfy_test/image_loader_execution_metrics_test.py` and
`tests-unit/comfy_test/reference_execution_metrics_test.py` with the CPU launcher
above, plus decoder/external-image regressions and schema freshness checks.

## Native MoGe workload boundaries

Extractor 6 explicitly describes `LoadMoGeModel`, `MoGeInference`,
`MoGePanoramaInference` and `SaveMoGeDepthPNG`. The loader uses hashed selected-file
identity from `geometry_estimation`; unknown lookalike/custom classes gain no
coverage. Runtime observations preserve image-reference source-stage relationships.
Requested controls are audit data. Effective image/view counts, chunk layout,
resolved token budgets, processed encoder/head shapes, model version, compute and
activation precision, native memory strategy, load/offload devices and projection
options come from the actual node/model boundaries. Oversized batch limits are
normalized to actual views; ignored v1/v2 refinement requests do not split effective
work. v3 refinement input point counts, spatial bounds and conditioning shapes are
live facts, observed before each real refiner call; unknown future passes remain
incomplete. No tensor contents or new device synchronization are used.

Model loading and forward durations are CPU-monotonic boundary measurements, not
GPU kernel timings or input matching features. The depth-PNG publisher observes
actual depth/mask dimensions before conversion. At most 16 model invocations and
16 refinement entries per model are retained; excess, malformed, duplicate or
unattributed work stays incomplete. Durable snapshots cover first input/model/
encoder/publication boundaries and readiness changes, plus existing progress and
stage transitions; forwards/refinements do not write once per call.

Panorama view/chunk and effective merge-canvas facts are observed, but merge work
retains `moge_panorama_merge_work_unmeasured`. LoadImage's reference-processing gap
and unsupported rendering/meshing/export paths remain visible. These observations
alone do not make the complete pipeline comparable or provide parent completion.
No matching/scaling policy, model selection, bridge routing or execution owner
changes. Source and CPU fixtures do not establish deployment, real-model quality
or forecast accuracy.

In addition to the timing suite above, run `tests-unit/app_test/run_timing_vision_test.py`,
`tests-unit/comfy_test/moge_execution_metrics_test.py`,
`tests-unit/comfy_extras_test/nodes_moge_test.py` and
`tests-unit/comfy_extras_test/moge_geometry_io_test.py` using the same CPU fixture launcher.
