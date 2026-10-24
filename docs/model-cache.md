# Native archive model cache

This optional Linux feature runs inside ComfyUI. It stages single-file safetensors
weights from an archive to fast NVMe before loading and removes idle copies after
30 minutes. It adds no daemon, internet requests or dependencies beyond the existing
ComfyUI requirements and Python standard library. Implementation lives in
`comfy/model_cache`; discovery, loading, execution and execution-cache ownership
remain in their existing ComfyUI layers.

## Configure and start

1. Mount the source SSD at a stable path using the normal host storage configuration.
   Create/use a model library with `diffusion_models`, `text_encoders`, `vae`,
   `checkpoints`, `loras`, `controlnet` and/or `clip_vision` subdirectories. The cache
   never moves, deletes or modifies originals. Archive files and categories must
   be regular files/directories, not symlinks. Downloads must finish before use.
2. Copy `model_cache.example.yaml` to a private configuration path. Set absolute
   `archive_root` and `cache_root`, and the source filesystem UUID reported by
   `findmnt -n -o UUID --target /your/archive/Models`. Configure capacity, minimum
   free bytes and inactivity seconds. Unknown fields, overlapping roots, invalid
   numbers and mismatched filesystem identities are rejected.
3. Use a dedicated empty cache directory owned by the ComfyUI user with permission
   0700. ComfyUI creates it if absent and requires its existing fast-storage detector
   to positively identify fast NVMe. One ComfyUI process may own a cache directory.
   Do not put weights, unrelated files or another application's data in it.
4. Add `--model-cache-config /absolute/path/model-cache.yaml` to the existing ComfyUI
   startup command. Preserve other operator flags. Deployment requires an idle
   queue and the normal service restart procedure; installing source alone does
   not activate the feature. No second service is required. For boot startup,
   configure the existing ComfyUI service and source mount dependencies normally.

The default ceiling and free-space reserve are each 100 GiB; both are configurable.
There is no forced slow-drive fallback and no eviction of recently used entries
merely to satisfy a new request. An oversized working set returns `CACHE_FULL`.
Pinning prevents expiry but does not bypass capacity accounting.

The standard source alias mapping still applies: `unet` selects the archive's
`diffusion_models`, and `clip` selects `text_encoders`. Do not also register the
archive as an ordinary extra model path: use its reserved `archive/` namespace.
Existing local models keep their current names and loading behaviour. A local
model already named `archive/...` conflicts and must be renamed or relocated by
the operator before using this feature. Cache copies never appear as separate
selector entries. Refreshing model lists discovers new archive files without
copying weights or refreshing inactivity. Unavailable archives produce explicit
errors rather than being treated as empty source libraries.

## Normal use

Select `archive/your-model.safetensors` in an existing loader and queue normally.
At runtime, the common folder resolver stages archive weights; direct calls to
`comfy.utils.load_torch_file` also stage admitted source paths. Copying reports
standard node progress and logs the model name and byte count. First use waits
for a complete verified copy; warm use returns the existing NVMe path. The actual
NVMe path reaches ComfyUI's storage-policy detection and mmap loader.

Execution tracks file dependencies for loaded outputs, cached node objects,
downstream consumers, async continuations and cached-output reuse. Last use is
persisted at execution completion, including interruption/failure cleanup. Metadata
inspection does not count as use. The worker performs maintenance between prompts
and at least once per 60-second idle wait. A long-running prompt never gets
interrupted by expiry; cleanup waits for a safe boundary.

Expiry removes only execution cache entries associated with the expired files,
then runs existing garbage-collection/model cleanup. It checks both current-process
file descriptors and memory mappings before unlinking. Files still retained by
custom code remain present with a reason shown in status. Unrelated execution
results are preserved. External cache providers do not receive newly executed
outputs that depend on managed archive files.

## Local controls

Run from this checkout with its prepared Python environment. The configuration
must match the running instance. Commands are **queued**, not synchronously
completed; the worker processes them at its next idle boundary. If ComfyUI is
stopped they remain pending until it starts. Inspect command state and entry
state after submission. Status itself has no cache ownership or cleanup side effects.

```sh
python -m comfy.model_cache --config /absolute/path/model-cache.yaml status
python -m comfy.model_cache --config /absolute/path/model-cache.yaml prefetch diffusion_models/model.safetensors
python -m comfy.model_cache --config /absolute/path/model-cache.yaml pin ENTRY_KEY
python -m comfy.model_cache --config /absolute/path/model-cache.yaml unpin ENTRY_KEY
python -m comfy.model_cache --config /absolute/path/model-cache.yaml evict ENTRY_KEY
```

Use an entry's full `key` from status. Prefetch takes a category and relative file
path, without the `archive/` selector prefix. Prefetch uses normal capacity and
validation and starts an inactivity interval on completion. Pinning is specific
to a source generation; replacing a source creates a new generation. Unpin before
manual eviction. New workflow use cancels pending eviction on the same entry.
A command marked `done` means processed: eviction can still be deferred while
references exist. Status lists entries, logical bytes, the latest 20 commands,
copy byte counts, pins, requests and deferred reasons. No HTTP/MCP control surface
or separate dashboard is added.

If submission output is lost, inspect status before resubmitting. Failed prefetch
can be retried after addressing capacity/source errors. Interrupted running commands
become failed on restart and require inspection before retry, never blind replay.

## Persistence and failure handling

The private directory contains an ownership marker/lock, SQLite WAL database and
immutable cache files. The process-lifetime Linux lock prevents two instances from
writing the same cache. Commands are coordinated through SQLite; only the owner
copies or removes files. Copy operations are serialized, use 8 MiB buffers, account
for outstanding copies, and check the free-space reserve while writing.

Copying and checksum verification reuse bounded 8 MiB buffers rather than holding
whole models in RAM. Linux's file cache also buffers ordinary reads and writes.
Node progress still reports every chunk, but SQLite copy-byte progress is persisted
at most once per second during streaming, then exactly when the entry becomes
ready. Progress-only transactions use SQLite WAL's NORMAL synchronization to avoid
forcing a disk sync; FULL synchronization is restored before any other write.
Status can therefore lag behind node progress, and an interrupted copy's
recorded byte count can lag its partial file or be lost on power failure.
Recovery discards that partial file; the counter is not a resume offset.
Durable copying/ready state transitions,
free-space checks and the default 30-minute expiry are unchanged.

Each completed stage logs total, copy/hash, final flush and destination checksum
verification seconds. Copy/hash includes source reads, writes, streamed SHA-256,
progress and any writeback stalls; flush is the final flush/fsync wait. Verification
may read from Linux's file cache, so these timings are not raw disk benchmarks.
Use them to identify the slow phase before changing storage or buffering. Warm
reuse avoids copying; first use after a restart still verifies the saved checksum.

A generation includes source path, device/inode, size, nanosecond mtime and ctime.
Copying verifies the source remained unchanged, hashes streamed bytes, rereads the
copy for checksum verification, and validates the safetensors header/offsets.
It flushes data before atomic rename and flushes the containing directory before
publishing `ready` in SQLite. Published copies are read-only. First use after a
restart verifies the persisted digest, as does use after a cached file's stat
identity changes. Source changes invalidate affected execution results before the
next prompt and create a fresh generation rather than modifying a mapped file.

Startup reconciles recorded copying/deleting states and removes their owned partial
files. Complete ready entries survive. A crashed active execution or backward
wall-clock jump receives a fresh inactivity interval; ordinary idle timestamps
survive reboot. During a process lifetime, monotonic elapsed time controls expiry.
This deliberately favours retaining uncertain files over premature deletion.

Cleanup requires ComfyUI to be running. Expired files left during shutdown are
removed at the next maintenance boundary after startup. The worker does not rely
on filesystem atime, inotify events or a separate heartbeat to establish use.

Common errors:

| Code | Action |
| --- | --- |
| `CONFIG_INVALID`, `PLATFORM_UNSUPPORTED` | Correct configuration; use Linux. |
| `ARCHIVE_UNAVAILABLE` | Mount the configured filesystem and verify UUID/path. |
| `MODEL_INVALID`, `MODEL_MISSING`, `MODEL_CORRUPT` | Check category/path, avoid symlinks and verify complete safetensors weights. |
| `CACHE_FULL` | Free unrelated disk space, safely evict unused entries or adjust limits. |
| `CACHE_BUSY` | Use the owning ComfyUI instance or a separate cache directory. |
| `CACHE_UNSAFE`, `CACHE_IDENTITY`, `STATE_INVALID` | Stop and inspect ownership/path/state; never erase an uncertain directory. |
| `SOURCE_CHANGED`, `COPY_CORRUPT` | Stabilize/verify source and storage before retrying. |
| `CACHE_CORRUPT` | Request safe eviction, inspect completion, then reload. |
| `ENTRY_MISSING`, `ENTRY_PINNED` | Refresh status or unpin before eviction. |
| `EXECUTION_REQUIRED` | Load managed archive weights through the normal execution queue. |
| `COMMAND_INVALID` | Use the documented CLI operations. |

Unexpected filesystem/database maintenance errors are logged with a traceback and
retain files for recovery. Do not delete live database, lock or mapping files to
force recovery. To disable, remove the launch option at an idle restart; archive
workflow selections then require the feature to be enabled again. Disabling does
not remove originals or silently adopt the cache as the model library.

## Verification and limits

Run the prepared interpreter with:

```sh
python -m pytest tests-unit/comfy_test/model_cache_test.py tests-unit/execution_test tests-unit/folder_paths_test -q
```

The execution suite requires `pytest-asyncio` from `tests-unit/requirements.txt`.
Tests use isolated CPU safetensors fixtures and the real PromptExecutor in classic,
LRU, RAM-pressure and no-cache modes. They cover copying, warm reuse, discovery,
selective release of real mappings, source replacement, checksum failure,
interrupted copy, concurrent requests, owner locking, disk reserve, pins, queued
controls and restart/clock recovery.

On 2026-10-24, an isolated 1 GiB safetensors fixture on the local SATA/NTFS3 archive
drive was staged to private NVMe/ext4 cache directories in three alternating
baseline/candidate comparisons. Only the fixture's source pages received
`POSIX_FADV_DONTNEED` between runs; no live model or global page cache was cleared.
Against baseline `eaa357d32`, total seconds were 31.744/16.351/21.941 before and
25.223/18.736/10.567 after the progress/buffer changes: medians 21.941 and 18.736
seconds (about 15% lower). SQLite commits fell from 132 per run to 19/14/7; only
progress transactions use NORMAL synchronization. Host I/O varied considerably,
so these small-fixture results do not predict a 40 GB cold-copy time. Destination
checksum verification remained enabled. The 292 cache/execution/folder-path tests
passed, including progress-write failure restoration, crash recovery and checksum
rejection; real power-loss testing remains separate.

Verified scope is safetensors through the common resolver/load_torch_file within
the normal worker. GGUF, multi-file Hugging Face repositories, custom bypass loaders
and unmanaged background loading are not supported guarantees. Cache paths are
private implementation details and must not be handed to other processes. A
same-user process that ignores ownership is outside this protocol. Dynamic custom
nodes may retain mappings indefinitely; deferred eviction is explicit.

Real power-loss testing, GPU inference/aimdo acceptance, large-model throughput
benchmarks and activation on the operator's SATA archive remain separate deployment
checks. Source integration and CPU tests do not establish those outcomes.
