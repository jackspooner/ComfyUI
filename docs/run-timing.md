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
dimensions and unsupported nodes. LoRA ranks/patching durations, control active
steps and measured GPU transfers/residency are not yet provided by this adapter.
An expanded envelope is not a claim of complete coverage. The collector/forecast
consumer must support this additive contract before producer activation; older
strict readers may reject the extra fields. No source change alone activates it.
Focused CPU verification: `python -m pytest tests-unit/app_test/run_timing_test.py
tests-unit/app_test/run_timing_workload_test.py tests-unit/comfy_test/execution_metrics_test.py -q`
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
nodes do not acquire workload coverage from input names. Internal crops/tiles and
reference encoders still need their own measurements. Optional progress handlers
remain compatible with older implementations, and observer failures cannot stop
node execution.

Read-only composite resolvers may return an internal `stageGraph` and optional
`stageMetadata` shared-operation IDs. Only extracted evidence is persisted. The
description uses native expansion IDs so later progress can correlate. A bounded
selection saved by the owner is retried at the first nested start event, with
execution-start availability. Read/completion-only resolution carries live/completion
availability, never backdated submission facts. `comfy.execution_metrics.stage_scope` is an optional
telemetry-only context for existing owner-controlled nested node calls; it preserves
the executing node context and scheduler, observes start/finish/failure and verified
reuse, and routes scalar sampling observations to that described substage.
Missing descriptions remain unavailable. Receipt presence alone cannot prove reuse.
Terminal storage failures clear only the attempt's in-memory timing state.
