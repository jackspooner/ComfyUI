# Fix cleanup after partial dynamic-VRAM prefetch failures

If `pin_modules()` raises, the prefetch queue still contains a raw module entry
that terminal cleanup skips. A nonresident module can also fault successfully
and fail preparation before publishing its cleanup state. Register the batch
before pinning and publish the minimal per-module state early for batched
prefetch, so existing cleanup releases each completed fault once. The original
failure still propagates; direct-cast state publication is unchanged.

Validation: 8 CPU regression cases, with 4 failing on master `872eb6a3` and all
passing with this patch; 53 tests pass including model-management, system-memory
and mixed-precision checks. Keeping only the queue fix still fails 1 case; keeping
only the cast fix fails 5. Covers first/middle/last fault failure, preparation
failure, null signatures, post-cast failure, successful consumption and repeated
cleanup. `py_compile` and `git diff --check` pass. Device boundaries are mocked;
no CUDA graph, stream or live-model validation. Ruff and full-suite validation
remain outstanding in an upstream-compatible environment.

Distinct from #16615's cast-buffer allocation fallback and #17123's buffer reset
when dynamic VRAM is disabled. #15944 would need reconciliation if merged first.

Submission prerequisite: identify or open a specific bug issue. No matching
partial-prefetch-cleanup issue was found in the 18 October search; do not attach
an unrelated crash report or submit this placeholder paragraph as PR text.
