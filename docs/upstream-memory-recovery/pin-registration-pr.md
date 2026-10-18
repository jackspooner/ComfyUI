# Fix pinned-memory registration retry after eviction

When host registration fails, `pin_memory()` retries using a temporary view that
`free_registrations()` may already have invalidated by shortening the same host
buffer. Rollback can then also use a stale offset. Discard and roll back the
uncommitted allocation before eviction, then allocate again at the current offset.
Also roll back growth when `HostBuffer.extend()` raises after changing its size.

Related: #16907's registration-rollback traceback. The reporter's workflow and
separate finalizer failure are unverified; this does not claim to fix both.
Distinct from #16901's async pin-stealing race. #15944 rewrites the same area and
would need reconciliation if merged first.

Validation: 10 CPU regression cases, with 7 failing on master `872eb6a3` and all
passing with this patch; 55 tests pass including model-management, system-memory
and mixed-precision checks. Covers eviction, both attempts, partial allocation
failure, rollback errors and pin accounting. `py_compile` and `git diff --check`
pass. CUDA/host-buffer boundaries are mocked; no live GPU/platform validation or
full-suite run. Ruff is unavailable locally; broader model imports require an
upstream-compatible comfy-kitchen (`int8_decode` is missing from this runtime).
