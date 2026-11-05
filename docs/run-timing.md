# Local run timing observations

The server queue adapter records accepted/start/finish facts in
`user/run-timing.sqlite`, using monotonic durations and immutable prompt identities.
It covers every native prompt, including direct tools and staged execution, without
changing queue ordering, node execution, cancellation or maintenance policy.
GET `/timing/observations?after=0&limit=100` returns version-1 timing observations,
a durable cursor and an ordered active snapshot. Limit is 1–500. Follow nextCursor
until null. Expired cursors are explicitly marked. The endpoint accepts machine-local
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
strategy and load/offload devices. Explicit Euler plans describe their resolved
noise-evaluation count by native sampler-function identity; unsupported samplers,
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

Expanded evidence explicitly reports missing processed references, implicit
dimensions and unsupported nodes. Standard LoRA loaders report actual file-cache
reuse, file-load time and CPU patch-preparation time separately. Native parsed
LoRAAdapter tensors provide min/max rank and supported-rank coverage without reading
tensor values. These live component facts are separate from requested strengths
and zero-strength bypass; completed durations never become submission/start facts.
Patch preparation is registration, not later GPU weight materialization or kernel
time. Other adapter formats may lack ranks. Measured GPU transfers/residency are
not yet provided by this adapter.
An expanded envelope is not a claim of complete coverage. The collector/forecast
consumer must support this additive contract before producer activation; older
strict readers may reject the extra fields. No source change alone activates it.
Focused CPU verification: `python -m pytest tests-unit/app_test/run_timing_test.py
tests-unit/app_test/run_timing_workload_test.py tests-unit/comfy_test/execution_metrics_test.py
tests-unit/comfy_test/lora_execution_metrics_test.py
tests-unit/comfy_test/reference_execution_metrics_test.py
tests-unit/comfy_test/control_execution_metrics_test.py
tests-unit/comfy_test/text_execution_metrics_test.py
tests-unit/app_test/run_timing_audio_test.py
tests-unit/comfy_test/audio_execution_metrics_test.py
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
nodes do not acquire workload coverage from input names. Native reference hooks
also observe loaded-image dimensions, VAE encode shapes after native cropping and
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
length; 4D audio tensors retain raw axes and an unresolved-layout reason because
native audio VAEs can use different time/frequency/channel layouts. It never
fabricates spatial/video axes from an audio tensor. Unsupported shapes and joint
AV streams remain incomplete. An audio
reference loader does not change a text-generation operation's main profile.

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

Extractor `comfy-explicit-5` keeps input-workload completeness distinct from known
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
tile and role attribution. Older extractor records retain their original evidence.
