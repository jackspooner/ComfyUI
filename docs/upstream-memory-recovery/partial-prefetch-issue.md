# Faulted modules are skipped by cleanup after partial dynamic-VRAM prefetch failure

Draft only; not posted.

On ComfyUI master `872eb6a3a5497721f53ebe82c0da9a3ba3414c13`, a failure in
`pin_modules()` can leave successful VBAR faults pinned after
`cleanup_prefetch_queues()`. The queue still contains the raw upcoming module,
which cleanup skips, and a nonresident module that fails preparation may not yet
have its `_prefetch` cleanup record.

Reproduction without model files or GPU work:

1. Apply only `tests-unit/comfy_test/test_prefetch_recovery.py` from the prepared
   cleanup patch to the above upstream commit.
2. Run `python -m pytest -q tests-unit/comfy_test/test_prefetch_recovery.py` in a
   prepared ComfyUI Python environment.
3. Observe 4 failed and 4 passed cases. In
   `test_failed_prefetch_keeps_all_faulted_modules_reachable`, two fake VBARs fault
   successfully and the second module's preparation raises. Cleanup makes no
   `vbar_unpin` calls; expected calls are one per faulted VBAR, with no residual
   `_prefetch` records. Other cases cover an exception in real `pin_modules()`
   after casting and failures partway through the batch.

The tests inject errors and mock device boundaries. This is a deterministic
Python cleanup reproduction; no claim is made about a particular CUDA crash,
memory-pressure threshold or live-model workflow. Registering the batch before
pinning and exposing minimal batched cleanup state before preparation makes all
8 cases pass while preserving the original exception.
