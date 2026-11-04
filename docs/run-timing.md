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
counts retain workload size. Text generation contributes numeric token counts and
output limits without storing generated text. Cutlery may describe immutable stage
expansions through an optional read-only resolver; missing bounded selections stay
unavailable until the execution owner saves them. Read-time resolution writes no
telemetry or execution state.

MediaTools owns duration learning and queue forecasts. These observations do not
prove queue completeness across other hosts, predict arbitrary future submissions,
or authorize service maintenance. Validate with the CPU run_timing_test.py suite;
live behavior requires activation of this commit on the serving instance.
